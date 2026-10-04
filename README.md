# KD-WaSR: WaSR/eWaSR baseline for MaSTr1325

Run on Kaggle: see [`docs/kaggle_guide_vi.md`](docs/kaggle_guide_vi.md) and
open [`kaggle_train.ipynb`](kaggle_train.ipynb) in a Kaggle Notebook.

This repository is a reproducible baseline for the research topic:

> Safety-aware knowledge distillation and quantization for maritime obstacle
> segmentation on Raspberry Pi 5.

The code provides:

- a MaSTr1325 loader with deterministic train/validation/test splits;
- an adapted eWaSR-ResNet18 student and WaSR-ResNet101 teacher;
- supervised, knowledge-distillation, boundary-band CE, and sparse-obstacle losses;
- training, evaluation, image/folder prediction, MODS prediction, ONNX export,
  ONNX static INT8 quantization, and Raspberry Pi benchmarking scripts;
- Vietnamese research proposal and experimental protocol in `docs/`.

The eWaSR/WaSR architecture implementation is adapted from the official Apache-2.0
repository by Matija Tersek, Lojze Zust, and Matej Kristan:
https://github.com/tersekmatija/eWaSR. See `NOTICE`.

## 1. Data layout

Place MaSTr1325 files under:

```text
data/MaSTr1325/
  images/
    0001.jpg
    ...
  masks/
    0001m.png
    ...
  imus/                 # optional
    0001.png
    ...
```

Mask values are `0=obstacle/environment`, `1=water`, `2=sky`, `4=ignore`.

The current workspace contains empty `images/` and `masks/` directories. Download
MaSTr1325 separately, then run the validation command below.

## 2. Environment

Python 3.10 or 3.11 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Prepare reproducible splits

The random image-level split below is intended for pipeline debugging only:

```powershell
python prepare_splits.py --data-root data/MaSTr1325 --output-dir data/splits --seed 42
```

This creates a deterministic 70/15/15 split. If sequence/session metadata becomes
available, replace these files with scene-grouped splits before reporting paper
results.

For research experiments, create the pre-registered near-duplicate grouped split:

```powershell
python prepare_grouped_splits.py --data-root data/MaSTr1325 `
  --output-dir data/grouped_splits --device cuda --seed 42
```

This writes locked split lists plus `split_report.json` with group statistics,
cross-split nearest pairs, class distributions, and SHA-256 checksums.

## 4. Train baselines

All research runs below use the locked grouped split. Train the eWaSR student:

```powershell
python train_student.py --data-root data/MaSTr1325 --split-dir data/grouped_splits `
  --experiment-id S0 --model ewasr_resnet18 --epochs 50 --batch-size 4 `
  --output-dir outputs/ewasr_fp32
```

Train the WaSR teacher:

```powershell
python train_student.py --data-root data/MaSTr1325 --split-dir data/grouped_splits `
  --experiment-id T0 --model wasr_resnet101 --epochs 50 --batch-size 2 `
  --output-dir outputs/wasr_teacher
```

Train standard logit distillation with the approved boundary-band CE
(`--boundary-weight` is `alpha_band`, not a second additive loss):

```powershell
python train_student.py --data-root data/MaSTr1325 --split-dir data/grouped_splits `
  --experiment-id K1 --model ewasr_resnet18 `
  --teacher-checkpoint outputs/wasr_teacher/best.pt `
  --kd-weight 1.0 --boundary-weight 2.0 `
  --epochs 50 --batch-size 4 --output-dir outputs/ewasr_kd
```

## 5. Evaluate and predict

```powershell
python evaluate.py --data-root data/MaSTr1325 --split-dir data/grouped_splits `
  --checkpoint outputs/ewasr_fp32/best.pt --split test

python predict.py --input path/to/image_or_folder --checkpoint outputs/ewasr_fp32/best.pt `
  --output-dir outputs/predictions
```

For square or otherwise non-4:3 inputs, do not silently stretch the image.
Run the locked geometry diagnostic first:

```powershell
python predict.py --input path/to/image_or_folder --checkpoint outputs/ewasr_fp32/best.pt `
  --geometry letterbox --output-dir outputs/predictions_letterbox

python predict.py --input path/to/image_or_folder --checkpoint outputs/ewasr_fp32/best.pt `
  --geometry center_crop --output-dir outputs/predictions_crop
```

`letterbox` pads the normalized tensor with zeros (ImageNet-mean RGB before
normalization) and removes the padding from logits before upsampling.  The
`center_crop` mode is diagnostic only: source pixels outside the crop are saved
as label `255` (unknown), never as water/sky/obstacle.

### A/B test with the authors' official pretrained eWaSR

Before changing the training recipe, compare the local checkpoint against the
official non-IMU eWaSR ResNet-18 model trained on MaSTr1325:

```powershell
python predict_official_ewasr.py --input path/to/image_or_folder --download `
  --output-dir outputs/official_ewasr_predictions
```

The script downloads release `0.1.0`, verifies its SHA-256 checksum, reproduces
the official ImageNet preprocessing and upsamples logits before argmax. Compare
the generated `_overlay.jpg` with the local model on exactly the same images.

Create the fixed-order qualitative comparison after both prediction folders
exist:

```powershell
python compare_geometry.py --input path/to/image_or_folder `
  --checkpoint outputs/ewasr_fp32/best.pt `
  --official-dir outputs/official_ewasr_predictions `
  --output-dir outputs/geometry_comparison --device cuda
```

This produces `Original | stretch | letterbox | center-crop | official eWaSR`
contact sheets for every input image. Without target-domain masks these sheets
are failure analysis only, not quantitative evidence that one model is better.

## 6. Pre-training gates and controlled R1/R2 screening

Generate the grouped-split balance audit and inspect the locked domain
augmentation before starting a new training run:

```powershell
python audit_split_balance.py --data-root data/MaSTr1325 `
  --split-dir data/grouped_splits

python preview_augmentations.py --data-root data/MaSTr1325 `
  --split-file data/grouped_splits/train.txt `
  --profile domain --output outputs/augmentation_domain_montage.jpg
```

The 15-epoch commands below are exploratory screening runs. They may only use
train/validation; do not evaluate test while choosing the winner.

```powershell
# R1: domain augmentation only
python train_student.py --data-root data/MaSTr1325 --split-dir data/grouped_splits `
  --experiment-id R1 --augmentation-profile domain --epochs 15 `
  --early-stopping-patience 10 `
  --output-dir outputs/R1_screen

# R1-B: isolate boundary-band CE
python train_student.py --data-root data/MaSTr1325 --split-dir data/grouped_splits `
  --experiment-id R1-B --augmentation-profile domain --boundary-weight 1.0 --epochs 15 `
  --output-dir outputs/R1_B_screen

# R1-O: isolate sparse-obstacle weighting
python train_student.py --data-root data/MaSTr1325 --split-dir data/grouped_splits `
  --experiment-id R1-O --augmentation-profile domain `
  --sparse-obstacle-weight 0.25 --epochs 15 `
  --output-dir outputs/R1_O_screen

# R2: interaction of both objectives
python train_student.py --data-root data/MaSTr1325 --split-dir data/grouped_splits `
  --experiment-id R2 --augmentation-profile domain --boundary-weight 1.0 `
  --sparse-obstacle-weight 0.25 --epochs 15 --output-dir outputs/R2_screen
```

Only after selecting a configuration by validation `safety_score`, rerun S0,
R1 and the final loss configuration for up to 50 epochs with seeds
`42, 1337, 2026`. `config.json` records the exact augmentation ranges, metric
definitions, checkpoint rule and early-stopping rule.

## 7. Export, quantize, and benchmark

```powershell
python export_onnx.py --checkpoint outputs/ewasr_fp32/best.pt `
  --output outputs/ewasr_fp32/model.onnx

python quantize_onnx.py --model outputs/ewasr_fp32/model.onnx `
  --data-root data/MaSTr1325 --split-dir data/grouped_splits `
  --output outputs/ewasr_fp32/model_int8.onnx

python benchmark_onnx.py --model outputs/ewasr_fp32/model.onnx --runs 300 --warmup 50
```

Run the benchmark command directly on Raspberry Pi 5 and record the JSON output.
Always use batch size 1, fixed input resolution, active cooling, and the same thread
count for fair comparisons.

## 7. Smoke test without MaSTr1325

```powershell
python -m unittest discover -s tests -v
```

The smoke test builds a tiny synthetic dataset, runs forward/backward, computes
metrics, and checks ONNX-compatible model output shapes.
