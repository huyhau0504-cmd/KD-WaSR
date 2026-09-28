# KD-WaSR: WaSR/eWaSR baseline for MaSTr1325

Run on Kaggle: see [`docs/kaggle_guide_vi.md`](docs/kaggle_guide_vi.md) and
open [`kaggle_train.ipynb`](kaggle_train.ipynb) in a Kaggle Notebook.

This repository is a reproducible baseline for the research topic:

> Safety-aware knowledge distillation and quantization for maritime obstacle
> segmentation on Raspberry Pi 5.

The code provides:

- a MaSTr1325 loader with deterministic train/validation/test splits;
- an adapted eWaSR-ResNet18 student and WaSR-ResNet101 teacher;
- supervised, knowledge-distillation, boundary, and sparse-obstacle losses;
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
  --model ewasr_resnet18 --epochs 50 --batch-size 4 --output-dir outputs/ewasr_fp32
```

Train the WaSR teacher:

```powershell
python train_student.py --data-root data/MaSTr1325 --split-dir data/grouped_splits `
  --model wasr_resnet101 --epochs 50 --batch-size 2 --output-dir outputs/wasr_teacher
```

Train with logit distillation and safety-aware losses:

```powershell
python train_student.py --data-root data/MaSTr1325 --split-dir data/grouped_splits `
  --model ewasr_resnet18 --teacher-checkpoint outputs/wasr_teacher/best.pt `
  --kd-weight 1.0 --boundary-weight 0.2 --sparse-obstacle-weight 0.5 `
  --epochs 50 --batch-size 4 --output-dir outputs/ewasr_kd
```

## 5. Evaluate and predict

```powershell
python evaluate.py --data-root data/MaSTr1325 --split-dir data/grouped_splits `
  --checkpoint outputs/ewasr_fp32/best.pt --split test

python predict.py --input path/to/image_or_folder --checkpoint outputs/ewasr_fp32/best.pt `
  --output-dir outputs/predictions
```

## 6. Export, quantize, and benchmark

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
