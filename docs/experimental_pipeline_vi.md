# Pipeline thi nghiem

## 1. Nguyen tac

1. Khoa test split truoc khi tuning.
2. Calibration INT8 chi dung train split, khong dung validation/test.
3. Moi so lieu accuracy cua model ONNX/INT8 phai duoc tinh lai tu artifact da export.
4. Moi so lieu latency dung batch size 1 va cung input resolution.
5. mIoU khong du de ket luan an toan; phai co obstacle va boundary metrics.

## 2. Du lieu

### Giai doan A - phat trien

- MaSTr1325: 70% train, 15% validation, 15% test.
- Neu co session/sequence metadata, group split theo session.
- Neu khong co metadata, dung split co seed co dinh; sau do kiem tra near-duplicate
  bang perceptual hash truoc khi cong bo ket qua.
- Ignore label = 4.

### Giai doan B - bai bao

- Train van chi dung MaSTr1325.
- Test them MODS bang official evaluator.
- Bo du lieu dia phuong: uu tien 3-5 video o dieu kien khac nhau; trich keyframe
  cach nhau du xa de tranh trung lap va gan nhan mot tap nho chat luong cao.

## 3. Tien xu ly va augmentation

- Resize baseline: 512 x 384 (width x height).
- ImageNet normalization.
- Train: horizontal flip, color jitter nhe.
- Khong augmentation tren validation/test.
- Cac augmentation fog, glare, reflection chi them nhu mot ablation rieng de tranh
  tron dong gop voi KD.

## 4. Ma tran thi nghiem

| ID | Model | KD | Boundary | Sparse obstacle | Precision |
|---|---|---:|---:|---:|---|
| T0 | WaSR-R101 teacher | 0 | 0 | 0 | FP32 |
| S0 | eWaSR-R18 | 0 | 0 | 0 | FP32 |
| S1 | eWaSR-R18 | 1 | 0 | 0 | FP32 |
| S2 | eWaSR-R18 | 1 | 1 | 0 | FP32 |
| S3 | eWaSR-R18 | 1 | 1 | 1 | FP32 |
| Q0 | S3 | - | - | - | PTQ INT8 |
| Q1 | S3 | - | - | - | QAT INT8 |
| Q2 | S3 | - | - | - | Mixed INT8/FP32 |

Toi thieu chay S0-S3 voi ba seed `42, 1337, 2026`. Teacher co the chi train mot
lan neu chi dung lam nguon soft target co dinh.

## 5. Hyperparameter ban dau

- Optimizer: AdamW.
- Learning rate: 3e-4 student, 1e-4 teacher.
- Weight decay: 1e-4.
- Epoch: 50, early stopping patience 10.
- Batch size: 4 student, 2 teacher, dieu chinh theo GPU.
- KD temperature: 4.
- `lambda_kd`: 1.0.
- `lambda_boundary`: 0.2.
- `lambda_sparse`: 0.5.

Chi tuning tren validation. Sau khi chot, test dung mot lan cho bang chinh.

## 6. Metrics

### Pixel-level

- mIoU.
- IoU tung lop.
- pixel accuracy.
- obstacle precision, recall, F1.

### Safety-oriented

- boundary F1.
- water-edge MAE/RMSE noi bo.
- connected-component recall theo bin dien tich: tiny, very small, small, medium+
  (dung official MODS evaluator khi test MODS).
- false-positive blobs/100 images.
- MODS overall F1 va danger-zone F1.

### Deployment

- p50, p95, mean latency va FPS.
- model size va peak RSS.
- CPU utilization, temperature truoc/sau.
- cong suat trung binh va joule/frame neu co USB power meter.

## 7. Benchmark Raspberry Pi 5

- Raspberry Pi OS 64-bit.
- Active cooling, performance governor neu duoc phep.
- Ghi ro ONNX Runtime version, thread count va CPU frequency.
- Warm-up 50 frame; do it nhat 300 frame.
- Chay 3 lan rieng biet; bao cao mean va standard deviation.
- Do ca model-only latency va end-to-end camera latency.
- Khong so sanh hai runtime voi preprocessing khac nhau.

## 8. Bang ket qua can co

### Bang A - accuracy/ablation

`Model | Params | mIoU | Obstacle IoU | Pr | Re | F1 | Boundary F1 | Edge MAE`

### Bang B - quantization

`Variant | Size MB | F1 | Delta F1 | Boundary F1 | ONNX parity error`

### Bang C - Raspberry Pi 5

`Variant | Threads | p50 ms | p95 ms | FPS | Peak RAM | Temp | J/frame`

### Bang D - external generalization

`Variant | MaSTr F1 | MODS F1 | MODS F1_D | Local F1 | Domain gap`

## 9. Kiem dinh thong ke

- Bao cao mean +/- standard deviation cua ba seed.
- So sanh S0 va S3 bang bootstrap confidence interval tren tung anh/test sequence.
- Khong chon model chi dua tren mot seed tot nhat.

## 10. Thu tu thuc thi

1. `prepare_splits.py` va luu danh sach file.
2. Train S0 va T0.
3. Train S1-S3.
4. `evaluate.py` tren validation, chot hyperparameter.
5. Chay test cho bang accuracy.
6. Export ONNX va kiem tra parity.
7. PTQ/QAT, danh gia lai accuracy artifact.
8. Copy artifact sang Pi 5 va benchmark.
9. Chay MODS evaluator va phan tich qualitative failure cases.

