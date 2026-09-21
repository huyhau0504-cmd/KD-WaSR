# De cuong nghien cuu

## 1. Ten de tai

**Tieng Viet:** Phan doan chuong ngai vat hang hai nhe su dung chung cat tri
thuc nhan biet vat can nho va luong tu hoa hon hop tren Raspberry Pi 5.

**Tieng Anh:** Safety-Aware Knowledge Distillation and Mixed-Precision
Quantization for Maritime Obstacle Segmentation on Raspberry Pi 5.

## 2. Bai toan va dong luc

Tau mat nuoc tu hanh can phan tach ba thanh phan obstacle/environment, water va
sky de xac dinh vung co the di chuyen. WaSR co do chinh xac cao nhung dung
ResNet-101 va kho trien khai tren may tinh nhung. eWaSR giam chi phi tinh toan
bang ResNet-18 va decoder nhe, nhung van con khoang trong nghien cuu ve:

1. kha nang bao toan vat can nho sau khi nen va luong tu hoa;
2. sai so tai bien water-obstacle, noi anh huong truc tiep den vung di chuyen;
3. trade-off do chinh xac, latency, bo nho va nang luong tren CPU Raspberry Pi 5;
4. do ben vung ngoai mien khi train tren MaSTr1325 va test tren canh moi.

## 3. Muc tieu

1. Xay dung baseline WaSR teacher va eWaSR student tren MaSTr1325.
2. De xuat safety-aware distillation loss uu tien vat can thua pixel va bien
   water-obstacle.
3. Khao sat FP32, PTQ INT8, QAT INT8 va mixed precision tren Raspberry Pi 5.
4. Xac dinh cau hinh Pareto tot nhat giua obstacle F1, small-obstacle recall,
   water-edge error, latency, RAM va energy/frame.
5. Danh gia ban dau tren test split MaSTr1325; sau do danh gia ngoai mien bang
   MODS va bo du lieu dia phuong.

## 4. Cau hoi nghien cuu

- **RQ1:** Logit distillation tu WaSR co giup eWaSR tang obstacle recall ma khong
  tang dang ke false positive hay khong?
- **RQ2:** Boundary-aware va sparse-obstacle-aware loss co bao toan vat can nho
  tot hon standard KD hay khong?
- **RQ3:** PTQ, QAT va mixed precision anh huong the nao den obstacle F1,
  water-edge error va small-obstacle recall?
- **RQ4:** Cau hinh nao dat Pareto frontier tren Raspberry Pi 5 ve latency,
  accuracy, memory va energy?
- **RQ5:** Cai tien tren MaSTr1325 co chuyen sang MODS/du lieu dia phuong hay
  chi overfit vao mien huan luyen?

## 5. Gia thuyet

- H1: Standard logit KD cai thien obstacle F1 cua student so voi supervised-only.
- H2: Safety-aware KD tang small-obstacle recall va boundary F1 so voi standard KD.
- H3: Static INT8 PTQ giam latency/model size nhung gay mat accuracy o mot so lop;
  QAT hoac mixed precision khoi phuc phan lon suy giam nay.
- H4: Cau hinh toi uu theo mIoU khong nhat thiet la cau hinh tot nhat theo
  danger-zone/object-level F1.

## 6. Pham vi

- Dau vao: anh RGB don, tuy chon IMU mask trong giai doan sau.
- Dau ra: semantic mask ba lop obstacle, water, sky.
- Train chinh: MaSTr1325.
- Test giai doan dau: test split MaSTr1325 duoc khoa truoc khi tuning.
- Test bai bao: them MODS va bo du lieu dia phuong neu co.
- Phan dieu khien tau trong do an dung free-space mask va bo lap ke hoach don
  gian; dong gop bai bao tap trung vao perception.

## 7. Phuong phap de xuat

Teacher la WaSR-ResNet101. Student la eWaSR-ResNet18. Ham muc tieu:

```text
L_total = L_seg
        + lambda_kd       * L_KD
        + lambda_boundary * L_boundary
        + lambda_sparse   * L_sparse_obstacle
```

- `L_seg`: cross entropy co ignore index 4.
- `L_KD`: KL divergence giua logits teacher/student voi temperature T.
- `L_boundary`: BCE tren bien hinh thai cua obstacle probability.
- `L_sparse_obstacle`: tang trong so obstacle pixel theo ti le nghich voi dien
  tich obstacle trong anh, co clamp de tranh gradient qua lon.

Mixed precision duoc xac dinh bang quantization sensitivity: cac lop lam giam
obstacle F1/boundary F1 manh khi INT8 se duoc giu FP32.

## 8. Dong gop du kien

1. Safety-aware distillation objective cho maritime obstacle segmentation.
2. Quantization sensitivity study va mixed-precision policy cho eWaSR tren ARM CPU.
3. Protocol danh gia gan voi an toan, gom small-object recall va water-edge quality,
   ben can mIoU.
4. Benchmark tren Raspberry Pi 5 voi p50/p95 latency, RAM, temperature va
   energy/frame.

## 9. Tieu chi thanh cong

Muc tieu, khong phai ket qua cam ket truoc:

- student safety-aware tot hon supervised eWaSR ve obstacle F1 va boundary F1;
- cau hinh nen nhanh hon it nhat 1.5 lan so voi eWaSR FP32 cung phan cung;
- suy giam obstacle F1 sau nen khong qua 1-2 diem phan tram;
- ket qua duoc lap lai voi ba random seed;
- co ablation tach rieng dong gop KD, boundary va sparse-obstacle loss.

## 10. Ke hoach 8 tuan

| Tuan | Cong viec | Dau ra |
|---|---|---|
| 1 | Kiem tra du lieu, khoa split, chay eWaSR baseline | Dataset report, baseline config |
| 2 | Train eWaSR va WaSR teacher | Checkpoint, metric, qualitative masks |
| 3 | Standard logit KD | KD baseline |
| 4 | Boundary va sparse-obstacle loss | Safety-aware model, ablation ban dau |
| 5 | Lap lai 3 seed, phan tich loi | Bang accuracy day du |
| 6 | Export ONNX, PTQ, QAT/mixed precision | FP32/INT8 artifacts |
| 7 | Benchmark Raspberry Pi 5, MODS/du lieu moi | Latency-memory-energy table |
| 8 | Tong hop Pareto, viet bao cao/bai bao | Ban thao va demo |

## 11. Rui ro va phuong an du phong

- Neu WaSR teacher qua nang: train o do phan giai thap hon hoac dung pretrained
  checkpoint; van giu eWaSR supervised lam baseline.
- Neu INT8 khong nhanh hon: profile operator, giu mixed precision va bao cao ket
  qua am tinh co kiem soat; khong suy dien tu FLOPs.
- Neu MaSTr1325 qua nho: tang augmentation, khoa split va dung MODS chi de test.
- Neu khong kip QAT: hoan thanh PTQ + sensitivity + selective FP32 truoc.

