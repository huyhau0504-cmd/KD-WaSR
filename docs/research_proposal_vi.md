# Đề cương nghiên cứu — bản sửa sau phản biện

## 1. Tên đề tài

**Tiếng Việt:** Chưng cất tri thức có trọng số theo vùng an toàn và triển khai
INT8 cho phân đoạn vật cản hàng hải trên Raspberry Pi 5.

**Tiếng Anh:** Safety-Aware Region-Weighted Knowledge Distillation and INT8
Deployment for Maritime Obstacle Segmentation on Raspberry Pi 5.

## 2. Bài toán và khoảng trống nghiên cứu

Tàu mặt nước tự hành cần phân đoạn `obstacle/environment`, `water` và `sky` để
xác định vùng có thể di chuyển. WaSR có độ chính xác cao nhưng nặng; eWaSR giảm
chi phí tính toán để phù hợp phần cứng nhúng. Đề tài không lặp lại mục tiêu làm
nhẹ eWaSR, mà tập trung vào hai vấn đề hẹp hơn:

1. standard logit KD bị chi phối bởi các vùng water/sky có diện tích lớn, trong
   khi vật cản nhỏ, vùng biên và vùng teacher–student bất đồng quan trọng hơn về
   an toàn;
2. chưa rõ lợi ích của KD còn được giữ lại sau PTQ INT8 và khi chạy end-to-end
   trên CPU Raspberry Pi 5 hay không.

Kết quả từ random image-level split hiện tại chỉ là kết quả debug. Chúng không
được dùng làm bằng chứng khoa học do nguy cơ ảnh gần trùng nằm khác split.

## 3. Mục tiêu

1. Xây dựng protocol chia dữ liệu theo group/near-duplicate và khóa split trước
   khi tuning.
2. So sánh eWaSR supervised, standard KD, class-weighted KD và region-weighted
   safety-aware KD.
3. Đánh giá vật cản nhỏ, biên có tolerance, false-positive blobs và khả năng
   tổng quát ngoài miền, không chỉ mIoU.
4. Export ONNX, PTQ INT8, kiểm tra parity và benchmark FP32/INT8 trên Raspberry
   Pi 5 bằng ảnh/video thật.

QAT và mixed precision là phần mở rộng nếu còn thời gian, không phải điều kiện
hoàn thành cốt lõi.

## 4. Câu hỏi nghiên cứu

- **RQ1:** Standard KD và class-weighted KD có cải thiện student so với
  supervised-only trên split độc lập hay không?
- **RQ2:** Region-weighted KD ưu tiên vật cản nhỏ, vùng biên và vùng bất đồng có
  tăng small-object recall tại mức false positive chấp nhận được hay không?
- **RQ3:** Boundary-band objective có cải thiện boundary F1 có tolerance và
  water-edge error mà không làm giảm phần ruột vật cản hay không?
- **RQ4:** PTQ INT8 ảnh hưởng thế nào đến các metric an toàn và latency/RAM thực
  tế trên Raspberry Pi 5?
- **RQ5:** Cải thiện trên MaSTr1325 có chuyển sang MODS hoặc local held-out set
  hay chỉ phù hợp miền huấn luyện?

## 5. Giả thuyết

- **H1:** Region-weighted KD tăng component recall của vật cản nhỏ so với
  standard KD và class-weighted KD.
- **H2:** Boundary-band weighted CE cải thiện BF1 có tolerance mà không xung đột
  với segmentation CE.
- **H3:** PTQ INT8 giảm kích thước và latency; mức suy giảm safety score không
  quá 2 điểm phần trăm so với ONNX FP32.
- **H4:** Mô hình tốt nhất theo mIoU không nhất thiết tốt nhất theo safety score
  và external-domain evaluation.

## 6. Phạm vi

### Bắt buộc

- RGB monocular, ba lớp segmentation.
- MaSTr1325 để train/validation theo split chống near-duplicate.
- WaSR-ResNet101 teacher và eWaSR-ResNet18 student.
- S0/R1/R2 robustness ablation và K1–K4 KD ablation, ba random seed cho các
  full run đã khóa.
- Một external held-out set: MODS official evaluator hoặc local set được khóa và
  không dùng để tuning.
- ONNX FP32, PTQ INT8 và benchmark Raspberry Pi 5.

### Tùy chọn

- QAT, selective/mixed precision, IMU input và energy/frame bằng power meter.

## 7. Phương pháp đề xuất

Teacher phải vượt S0 trên validation đã khóa trước khi được dùng. Nếu teacher tự
train không đạt gate này, dùng checkpoint WaSR chính thức tương thích; nếu vẫn
không đạt, dừng nhánh KD và báo cáo kết quả âm.

Hàm mục tiêu được khóa như sau:

```text
L_total   = L_band_CE + lambda_kd * L_region_KD
L_band_CE = sum_i CE_i * (1 + alpha_band * band_i)
            / sum_i (1 + alpha_band * band_i)
```

- `CE_i`: cross entropy ba lớp tại pixel hợp lệ, ignore index 4.
- `band_i`: dải bán kính 3 px quanh biên obstacle ground truth.
- `alpha_band`: chọn trong search budget đã khóa; CE không được cộng lần thứ hai.
- `L_region_KD`: KL theo pixel với weight map được đặc tả đầy đủ trong pipeline:
  obstacle, component có diện tích không quá 1024 px, boundary band 3 px,
  teacher confidence và argmax disagreement. Weight được clamp `[0.25, 6]` và
  chuẩn hóa mean bằng 1 trên pixel hợp lệ.

Standard KD và class-weighted KD là baseline bắt buộc để chứng minh đóng góp của
region weighting, thay vì chỉ so với supervised-only.

## 8. Đóng góp dự kiến

1. Region-weighted KD gắn với vùng rủi ro trong maritime segmentation.
2. Protocol chống near-duplicate và bộ metric gồm component recall, tolerated
   boundary F1, false-positive blobs và external-domain gap.
3. Đánh giá từ PyTorch đến ONNX FP32/PTQ INT8 và đo end-to-end trên Raspberry Pi
   5.

Các đóng góp trên là giả thuyết cần kiểm chứng, không được trình bày như kết quả
đã đạt.

## 9. Tiêu chí chọn checkpoint và thành công

Tiêu chí chọn model được khóa trước khi mở test:

```text
S_val = 0.40 * obstacle_F1
      + 0.30 * small_component_recall
      + 0.20 * boundary_F1_tolerance
      + 0.10 * (1 - obstacle_pixel_FPR)
```

Mọi thành phần nằm trong `[0, 1]`. Test chỉ chạy sau khi chốt hyperparameter và
checkpoint bằng validation.

Đề tài thành công ở mức tối thiểu khi:

- protocol split vượt kiểm tra near-duplicate;
- teacher vượt S0 ít nhất `0.005` theo `S_val` và không tăng quá 5
  false-positive blobs/100 ảnh trước khi KD;
- K3 hoặc K4 cải thiện small-component recall/BF1 so với S0 và K1 mà
  không tăng quá `max(5 blobs/100 ảnh, 10% so với S0)`;
- ONNX FP32 đạt max absolute logit error `<= 1e-4` và pixel agreement
  `>= 99.99%`; PTQ được đánh giá lại từ artifact;
- có benchmark FP32/INT8 thực trên Raspberry Pi 5 qua ba lần chạy;
- báo cáo mean ± std ba seed và confidence interval trên test unit phù hợp.

Kết quả âm vẫn hợp lệ nếu protocol chặt và được phân tích trung thực.

## 10. Kế hoạch tám tuần

| Tuần | Công việc bắt buộc | Đầu ra |
|---|---|---|
| 1 | Audit dữ liệu, group/pHash split, khóa protocol | split report, split files |
| 2 | Sửa boundary objective/metric, unit test; train S0 | baseline hợp lệ |
| 3 | Train/kiểm tra teacher; standard và class-weighted KD | teacher gate, S1 |
| 4 | Region-weighted KD và boundary-band objective | K3, K4 |
| 5 | Ba seed, component metrics, error analysis | bảng ablation |
| 6 | External held-out/MODS; ONNX parity và PTQ | bảng generalization/quantization |
| 7 | Benchmark Raspberry Pi 5 model-only và end-to-end | latency/RAM/thermal table |
| 8 | Thống kê, Pareto, viết báo cáo/bài báo | bản thảo và demo |

QAT/mixed precision chỉ được bắt đầu nếu các đầu ra bắt buộc đến tuần 6 đã hoàn
thành.

## 11. Hai cổng phê duyệt

### Gate A — phê duyệt kế hoạch trước triển khai

- split protocol, loss, metric, teacher gate, ma trận thí nghiệm và phạm vi tám
  tuần được xác định rõ;
- không tuyên bố implementation/kết quả chưa có;
- phần bắt buộc và tùy chọn được tách riêng.

### Gate B — nghiệm thu khoa học sau triển khai

- có bằng chứng split sạch, unit test loss/metric, các full run đã khóa ba seed,
  external test,
  ONNX parity/PTQ và benchmark Pi 5;
- mọi kết luận chỉ dựa trên artifact và protocol đã khóa.
