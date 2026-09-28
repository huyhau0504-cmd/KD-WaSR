# Pipeline thí nghiệm — bản sửa sau phản biện

## 1. Nguyên tắc khóa trước

1. Random image-level split hiện tại chỉ dùng debug, không dùng cho bài báo.
2. Khóa split, metric, safety score và hyperparameter search space trước test.
3. Test và external held-out không được dùng để chọn checkpoint hoặc tuning.
4. INT8 calibration chỉ dùng train split; mọi accuracy phải đo lại từ artifact
   ONNX thực tế.
5. QAT/mixed precision là tùy chọn; pipeline cốt lõi kết thúc ở PTQ INT8.

## 2. Audit và chia dữ liệu

Ưu tiên metadata session/sequence. Nếu metadata không tồn tại:

1. tính pHash 64-bit và embedding ResNet-18 ImageNet penultimate cho toàn bộ ảnh;
2. tạo graph, nối hai ảnh nếu Hamming pHash `<= 6` **hoặc** cosine similarity
   embedding `>= 0.985`; các threshold này được khóa trước khi xem metric model;
3. gom connected components thành group;
4. chia theo group thành train/validation/test, có cân bằng thô tỷ lệ obstacle;
5. xuất `split_report.json` gồm số ảnh/group, phân bố lớp, khoảng cách gần nhất
   xuyên split và danh sách các cặp đáng ngờ;
6. khóa ba file split trong Git trước khi train chính thức.

Ảnh `old_*` và ảnh tên số đều phải đi qua cùng phép kiểm tra; không suy luận
sequence chỉ từ tên file nếu chưa xác minh metadata.

## 3. Tiền xử lý và augmentation

- Kích thước chuẩn: 384×512, ImageNet normalization.
- Baseline: horizontal flip và color jitter nhẹ.
- Domain-robustness ablation: blur, gamma/exposure, haze/color cast và glare.
- Không augmentation validation/test.
- Không dùng ảnh external test để thiết kế augmentation sau khi xem kết quả; nếu
  cần vòng phát triển domain adaptation, phải tạo development set riêng.

## 4. Loss và KD

### Boundary objective

Thay BCE giữa obstacle probability và binary boundary bằng boundary-band
weighted CE:

```text
L_total   = L_band_CE + lambda_kd * L_KD
L_band_CE = sum_i CE_i * (1 + alpha_band * band_i)
            / sum_i (1 + alpha_band * band_i)
```

`band_i` là dải bán kính 3 px quanh biên obstacle ground truth. Pixel ignore
không xuất hiện trong tử hoặc mẫu. Pixel trong ruột obstacle vẫn giữ đúng target
obstacle. Unit test bắt buộc:

- dự đoán đúng toàn bộ obstacle cho loss thấp hơn dự đoán chỉ đúng biên;
- tăng lỗi trong dải biên làm `L_band` tăng;
- ignore pixels không đóng góp gradient.

### Region-weighted KD

So sánh tuần tự:

- S0: supervised-only;
- S1a: standard KL KD;
- S1b: class-weighted KD;
- S2: region-weighted KD cho obstacle nhỏ, boundary và disagreement;
- S3: S2 + boundary-band weighted CE.

Với pixel hợp lệ `i`, định nghĩa:

```text
o_i = 1 nếu ground-truth là obstacle, ngược lại 0
s_i = 1 nếu pixel thuộc obstacle component có area <= 1024 px
b_i = 1 nếu pixel nằm trong boundary band radius 3 px
d_i = 1 nếu argmax teacher khác argmax student (student detach khi tạo weight)
c_i = clip((max softmax(z_teacher)_i - 0.5) / 0.5, 0, 1)
r_i = 1 + 1*o_i + 2*s_i + 1*b_i + 1*d_i
w_i = clip(r_i * (0.25 + 0.75*c_i), 0.25, 6.0)
w_i = w_i / mean_valid(w)
L_region_KD = sum_i w_i * KL_i / sum_i w_i
```

Các vùng chồng lấp được cộng theo `r_i`; clamp diễn ra sau phép cộng. Softmax
confidence dùng temperature 1, còn `KL_i` dùng temperature `T`. Với S1b,
class weights là nghịch đảo căn bậc hai tần suất pixel trên train split, rồi
chuẩn hóa weighted mean bằng 1. Các công thức này không được đổi sau khi test.

## 5. Teacher gate

1. Train S0 trên split khóa.
2. Train hoặc load WaSR teacher cố định.
3. Đánh giá cùng input, preprocessing và validation split.
4. Chỉ tiếp tục KD nếu `S_teacher >= S_S0 + 0.005` và
   `FP_blobs_teacher <= FP_blobs_S0 + 5` trên mỗi 100 ảnh validation.
5. Lưu checksum/config teacher; dùng đúng một teacher cho mọi seed KD.

Nếu gate thất bại, thử checkpoint chính thức tương thích. Nếu vẫn thất bại, dừng
KD và chuyển kết quả thành phân tích âm thay vì ép teacher yếu vào pipeline.

## 6. Metric

### Pixel-level

- mIoU và IoU từng lớp;
- obstacle precision, recall, F1 và pixel FPR.

### Safety-oriented

- boundary F1 với tolerance cố định 3 px ở 384×512; đồng thời báo cáo sensitivity
  ở 1 px và 5 px;
- unit test mask dịch 1, 2, 3 và >3 px để xác nhận metric;
- connected-component recall theo bin diện tích;
- false-positive components trên 100 ảnh, sau khi loại component dưới ngưỡng
  nhiễu được khóa trước;
- water-edge MAE/RMSE nội bộ, ghi rõ không thay thế MODS evaluator;
- MODS overall/danger-zone F1 nếu dùng MODS.

Component dùng 8-connectivity sau resize 384×512. Các bin ground truth là:
`tiny=1–64`, `very_small=65–256`, `small=257–1024`, `medium_plus=>1024` px.
`small_component_recall` trong safety score gộp ba bin `<=1024`.

Ghép prediction–ground truth one-to-one bằng Hungarian, tối đa hóa IoU. Một cặp
được tính true positive khi prediction che phủ ít nhất 50% diện tích component
ground truth. Một predicted component không được ghép nhiều ground-truth, nên
merge nhiều vật cản bị phạt. Predicted component không ghép và có area `>=16 px`
là false-positive blob. Recall báo cáo micro theo component trên MaSTr1325 và
macro theo sequence cho external video.

### Checkpoint score

```text
S_val = 0.40 F1_obstacle
      + 0.30 Recall_small
      + 0.20 BF1_tol3
      + 0.10 (1 - obstacle_pixel_FPR)
```

Không thay đổi công thức sau khi mở test.

## 7. Ma trận thí nghiệm cốt lõi

| ID | Student objective | Mục đích |
|---|---|---|
| T0 | WaSR teacher | teacher gate |
| S0 | CE | supervised baseline |
| S1a | CE + standard KL | KD baseline |
| S1b | CE + class-weighted KL | baseline mạnh hơn |
| S2 | CE + region-weighted KD | đóng góp chính |
| S3 | S2 + boundary-band CE | đóng góp biên |
| Q0 | ONNX FP32 của model được chọn | parity/deployment baseline |
| Q1 | PTQ INT8 của cùng model | quantization |

S0, S1a, S1b, S2 và S3 chạy seed `42, 1337, 2026`. Teacher được cố định.
QAT/mixed precision chỉ là Q2/Q3 tùy chọn.

## 8. Training protocol

- Pilot chung cho mọi model: learning rate `{1e-4, 3e-4}`, weight decay cố định
  `1e-4`; chọn một cặp dùng cho toàn bộ ablation.
- Mỗi S1a, S1b, S2 và S3 được tối đa bốn cấu hình tuning với seed 42. S1a/S1b/S2
  dùng grid `T in {2,4}` × `lambda_kd in {0.5,1.0}`. S3 dùng bốn tuple
  `(T, lambda_kd, alpha_band)` là `(2,.5,1)`, `(2,1,2)`, `(4,.5,2)`, `(4,1,4)`.
  Region coefficients, confidence transform và disagreement rule được giữ cố
  định như mục 4, không tuning thêm.
- Tối đa 50 epoch ở baseline ban đầu; early stopping patience 10 theo `S_val`.
- Log train loss, validation metrics, learning rate và epoch tốt nhất.
- Không dùng thời gian train ngắn/dài làm bằng chứng hội tụ; quyết định dựa trên
  learning curve và validation plateau.
- Lưu config, seed, Git commit, checksum split và checkpoint.

## 9. External evaluation

Tối thiểu một trong hai:

- MODS với official evaluator; hoặc
- local held-out sequences có ground truth, tách theo video/session và không dùng
  để tuning.

Ảnh/video không có ground truth chỉ dùng qualitative failure analysis, không tính
mIoU/F1. Phải phân biệt overfit (train–validation gap) với domain shift
(in-domain tốt, external-domain giảm).

## 10. ONNX, PTQ và parity

1. Export model được chọn sang ONNX FP32.
2. So PyTorch/ONNX trên 32 ảnh validation được chọn bằng seed 42.
3. Chỉ tiếp tục PTQ nếu max absolute logit error `<=1e-4` và pixel agreement
   `>=99.99%`.
4. Calibration từ train split đại diện, không dùng validation/test.
5. Đánh giá lại toàn bộ metric từ ONNX FP32 và INT8 artifact.
6. Báo cáo model size, delta safety score và các operator không quantize được.

PTQ đạt non-inferiority nếu suy giảm `S_val` không quá `0.02` tuyệt đối (hai
điểm phần trăm), không phải 2% tương đối. Mức FP tăng mất kiểm soát được định
nghĩa là lớn hơn `max(5 blobs/100 ảnh, 10% so với S0)`.

## 11. Benchmark Raspberry Pi 5

- Raspberry Pi OS 64-bit, active cooling; ghi OS/kernel, ONNX Runtime, governor,
  CPU frequency và thread count.
- Batch 1, cùng resolution và preprocessing.
- Warm-up 50, đo ít nhất 300 frame thật, ba lần chạy độc lập.
- Báo cáo model-only và end-to-end preprocessing + inference + postprocessing.
- p50/p95/mean latency, FPS và peak RSS.
- Ghi nhiệt độ đầu/cuối và sự kiện throttling.
- Chỉ báo cáo joule/frame nếu có power meter; nếu không, ghi rõ không đo.

## 12. Thống kê và bảng kết quả

- S0–S3: mean ± std ba seed.
- Bootstrap confidence interval theo ảnh hoặc sequence phù hợp; external video
  phải bootstrap theo sequence/clip, không giả định mọi frame độc lập.
- Bảng accuracy/ablation, external-domain gap, ONNX/PTQ và Pi benchmark.
- Báo cáo cả failure cases: glare/reflection, vật cản nhỏ, đường chân trời, camera
  thấp và nước phẳng.

## 13. Thứ tự triển khai sau Gate A

1. Audit/split report.
2. Boundary loss/metric và unit tests.
3. S0 và teacher gate.
4. S1a/S1b/S2/S3 ba seed.
5. External evaluation và thống kê.
6. ONNX parity, PTQ.
7. Raspberry Pi 5 benchmark.
8. QAT/mixed precision nếu còn thời gian.
