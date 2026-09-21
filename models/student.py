"""WaSR/eWaSR baseline models.

The architecture is adapted from the official Apache-2.0 eWaSR implementation:
https://github.com/tersekmatija/eWaSR

Changes in this project:
- modern torchvision weight API;
- no PyTorch Lightning or timm dependency;
- accepts either an image tensor or ``{"image": tensor, "imu_mask": tensor}``;
- always upsamples logits to the input resolution;
- exposes encoder features for future feature distillation.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Mapping, Optional, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import (
    ResNet18_Weights,
    ResNet101_Weights,
    resnet18,
    resnet101,
)


ModelInput = Union[Tensor, Mapping[str, Tensor]]


def _image_from_input(inputs: ModelInput) -> Tensor:
    return inputs["image"] if isinstance(inputs, Mapping) else inputs


class SafeBatchNorm2d(nn.BatchNorm2d):
    """Use running statistics for the degenerate N=H=W=1 training case.

    eWaSR applies BatchNorm after global average pooling in its attention
    refinement modules. A batch size of one therefore contains only one value
    per channel, for which regular training-mode BatchNorm is undefined. The
    fallback below preserves the same parameters and state-dict layout while
    allowing memory-constrained batch-size-one training.
    """

    def forward(self, input: Tensor) -> Tensor:
        values_per_channel = input.numel() // input.shape[1]
        if self.training and values_per_channel == 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps,
            )
        return super().forward(input)


class ResNetFeatureExtractor(nn.Module):
    """Return the four ResNet stages without relying on private torchvision APIs."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: Tensor) -> OrderedDict[str, Tensor]:
        x = self.stem(x)
        skip1 = self.layer1(x)
        skip2 = self.layer2(skip1)
        aux = self.layer3(skip2)
        out = self.layer4(aux)
        return OrderedDict(out=out, aux=aux, skip2=skip2, skip1=skip1)


class AttentionRefinementModule(nn.Module):
    def __init__(self, channels: int, second_pool: bool = False) -> None:
        super().__init__()
        self.second_pool = second_pool
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            SafeBatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x * self.gate(self.pool(x))
        if self.second_pool:
            out = out * self.pool(out)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, large_channels: int, small_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(large_channels + small_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, large: Tensor, small: Tensor) -> Tensor:
        if large.shape[-2:] != small.shape[-2:]:
            small = F.interpolate(small, size=large.shape[-2:], mode="nearest")
        fused = self.project(torch.cat([large, small], dim=1))
        return fused + fused * self.attention(fused)


class ASPPv2(nn.Module):
    def __init__(self, in_channels: int, rates: tuple[int, ...], out_channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate)
            for rate in rates
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([branch(x) for branch in self.branches], dim=0).sum(dim=0)


class ChannelMixer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.arm = AttentionRefinementModule(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.arm(x)


class SpatialMixer(nn.Module):
    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        avg = x.mean(dim=1, keepdim=True)
        maximum = x.amax(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, maximum], dim=1)))


class PointwiseMLP(nn.Module):
    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = channels * expansion
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class MetaFormerBlock(nn.Module):
    def __init__(self, channels: int, mixer: str) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.mixer = ChannelMixer(channels) if mixer == "C" else SpatialMixer()
        self.mlp = PointwiseMLP(channels)
        self.scale1 = nn.Parameter(torch.full((channels,), 1e-5))
        self.scale2 = nn.Parameter(torch.full((channels,), 1e-5))

    def forward(self, x: Tensor) -> Tensor:
        scale1 = self.scale1.view(1, -1, 1, 1)
        scale2 = self.scale2.view(1, -1, 1, 1)
        x = x + scale1 * self.mixer(self.norm1(x))
        return x + scale2 * self.mlp(self.norm2(x))


class PyramidPoolAggregation(nn.Module):
    def __init__(self, stride: int = 2) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, features: list[Tensor]) -> Tensor:
        height, width = features[0].shape[-2:]
        size = ((height - 1) // self.stride + 1, (width - 1) // self.stride + 1)
        return torch.cat([F.adaptive_avg_pool2d(x, size) for x in features], dim=1)


class SemanticInjectionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.global_gate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )
        self.global_value = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, local: Tensor, global_feature: Tensor) -> Tensor:
        global_feature = F.interpolate(global_feature, local.shape[-2:], mode="nearest")
        return self.local(local) * self.global_gate(global_feature) + self.global_value(global_feature)


class EWaSRDecoder(nn.Module):
    """Embedded WaSR decoder with CCCC-SS token mixing."""

    def __init__(self, num_classes: int = 3, semantic_channels: int = 256) -> None:
        super().__init__()
        channels = (512, 256, 128, 64)
        token_channels = sum(channels)
        self.channels = channels
        self.pool = PyramidPoolAggregation(stride=2)
        self.mixers = nn.Sequential(
            *[MetaFormerBlock(token_channels, letter) for letter in "CCCCSS"]
        )
        self.skip_enricher = nn.Sequential(
            MetaFormerBlock(channels[2], "S"),
            MetaFormerBlock(channels[2], "S"),
        )
        self.injections = nn.ModuleList(
            SemanticInjectionModule(ch, semantic_channels) for ch in channels
        )
        self.head = nn.Sequential(
            nn.Conv2d(semantic_channels, semantic_channels, 1, bias=False),
            nn.BatchNorm2d(semantic_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(semantic_channels, num_classes, 1),
        )

    def forward(self, out: Tensor, aux: Tensor, skip2: Tensor, skip1: Tensor) -> Tensor:
        locals_ = [out, aux, skip2, skip1]
        tokens = self.mixers(self.pool(locals_))
        skip2 = self.skip_enricher(skip2)
        locals_[2] = skip2
        chunks = torch.split(tokens, self.channels, dim=1)
        refined = [module(local, global_) for module, local, global_ in zip(self.injections, locals_, chunks)]
        target_size = refined[-1].shape[-2:]
        fused = refined[-1]
        for feature in refined[:-1]:
            fused = fused + F.interpolate(feature, target_size, mode="bilinear", align_corners=False)
        return self.head(fused)


class EWaSRStudent(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained_backbone: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        self.encoder = ResNetFeatureExtractor(resnet18(weights=weights))
        self.decoder = EWaSRDecoder(num_classes=num_classes)

    def forward(self, inputs: ModelInput) -> Dict[str, Tensor]:
        image = _image_from_input(inputs)
        features = self.encoder(image)
        logits = self.decoder(
            features["out"], features["aux"], features["skip2"], features["skip1"]
        )
        logits = F.interpolate(logits, image.shape[-2:], mode="bilinear", align_corners=False)
        return {"out": logits, "aux": features["aux"]}


class WaSRDecoder(nn.Module):
    """Original no-IMU WaSR decoder used as the distillation teacher."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.arm1 = AttentionRefinementModule(2048)
        self.arm2 = nn.Sequential(
            AttentionRefinementModule(512, second_pool=True),
            nn.Conv2d(512, 2048, 1),
        )
        self.fusion = FeatureFusionModule(256, 2048, 1024)
        self.aspp = ASPPv2(1024, (6, 12, 18, 24), num_classes)

    def forward(self, out: Tensor, skip2: Tensor, skip1: Tensor) -> Tensor:
        context = self.arm1(out) + F.interpolate(
            self.arm2(skip2), size=out.shape[-2:], mode="nearest"
        )
        context = F.interpolate(context, size=skip1.shape[-2:], mode="nearest")
        return self.aspp(self.fusion(skip1, context))


class WaSRTeacher(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained_backbone: bool = True) -> None:
        super().__init__()
        weights = ResNet101_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet101(
            weights=weights,
            replace_stride_with_dilation=(False, True, True),
        )
        self.encoder = ResNetFeatureExtractor(backbone)
        self.decoder = WaSRDecoder(num_classes=num_classes)

    def forward(self, inputs: ModelInput) -> Dict[str, Tensor]:
        image = _image_from_input(inputs)
        features = self.encoder(image)
        logits = self.decoder(features["out"], features["skip2"], features["skip1"])
        logits = F.interpolate(logits, image.shape[-2:], mode="bilinear", align_corners=False)
        return {"out": logits, "aux": features["aux"]}


def build_model(
    name: str = "ewasr_resnet18",
    num_classes: int = 3,
    pretrained_backbone: bool = True,
) -> nn.Module:
    if name == "ewasr_resnet18":
        return EWaSRStudent(num_classes=num_classes, pretrained_backbone=pretrained_backbone)
    if name == "wasr_resnet101":
        return WaSRTeacher(num_classes=num_classes, pretrained_backbone=pretrained_backbone)
    raise ValueError(f"Unknown model '{name}'. Expected ewasr_resnet18 or wasr_resnet101.")


def load_checkpoint_model(
    checkpoint_path: str,
    device: Union[str, torch.device] = "cpu",
    model_name: Optional[str] = None,
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    name = model_name or saved_args.get("model", "ewasr_resnet18")
    model = build_model(name, pretrained_backbone=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    return model.to(device)
