# Đưa dự án lên GitHub và chạy trên Kaggle

## 1. Nội dung đưa lên GitHub

Chỉ đưa mã nguồn, tài liệu và các file chia tập dữ liệu lên GitHub. Không đưa
MaSTr1325, `.venv`, checkpoint hay thư mục `outputs` lên Git. Các nội dung này
đã được loại trừ trong `.gitignore`.

Tại thư mục dự án, tạo commit đầu tiên:

```powershell
git add .
git commit -m "Add maritime segmentation baseline and Kaggle training"
```

Tạo một repository trống trên GitHub, không chọn tạo sẵn README hoặc
`.gitignore`. Sau đó nối repository và đẩy nhánh `main`:

```powershell
git remote add origin https://github.com/USERNAME/maritime_kd.git
git push -u origin main
```

Thay `USERNAME` bằng tài khoản GitHub. Nếu repository đã có remote `origin`,
dùng `git remote set-url origin URL` thay cho `git remote add origin URL`.

## 2. Đưa MaSTr1325 lên Kaggle

Tạo một Kaggle Dataset riêng tư và tải lên thư mục MaSTr1325. Sau khi gắn
dataset vào notebook, cấu trúc tối thiểu cần có ở một vị trí dưới
`/kaggle/input` là:

```text
MaSTr1325/
├── images/
│   ├── 0001.jpg
│   └── ...
└── masks/
    ├── 0001m.png
    └── ...
```

Notebook tự dò thư mục cha có đồng thời `images/` và `masks/`, vì vậy tên slug
của Kaggle Dataset có thể khác.

## 3. Chạy notebook

1. Trên Kaggle, chọn **Create > New Notebook** rồi import
   `kaggle_train.ipynb` từ GitHub.
2. Gắn Kaggle Dataset MaSTr1325 vào mục **Input**.
3. Trong **Notebook options**, chọn **Accelerator: GPU**.
4. Bật Internet để notebook có thể clone repository GitHub. Repository riêng tư
   cần Kaggle secret/token; repository công khai không cần token.
5. Sửa `REPO_URL` trong ô cấu hình và chạy lần lượt tất cả các ô.

Kaggle chỉ cho phép ghi vào `/kaggle/working`. Notebook vì vậy tạo split tại
`/kaggle/working/splits` và checkpoint tại
`/kaggle/working/outputs/ewasr_fp32`.

## 4. Lưu kết quả

Sau khi train, chọn **Save Version** để giữ lại output của phiên chạy. Có thể tải
`best.pt`, `last.pt`, `history.json` và `config.json` từ thư mục output, hoặc tạo
một Kaggle Dataset mới từ output để dùng làm teacher/checkpoint cho lần chạy
sau.

Không ghi GitHub token, Kaggle token hay mật khẩu trực tiếp vào notebook hoặc
commit Git.

## 5. So sánh với eWaSR pretrained chính thức

Trước khi kết luận kiến trúc eWaSR không phù hợp với ảnh ngoài miền, chạy cùng
ảnh qua model `ewasr_resnet18.onnx` do tác giả công bố:

```python
from pathlib import Path
import subprocess
import sys

test_images = sorted(Path("/kaggle/input").rglob("left_*.jpg"))
assert test_images, "Không tìm thấy ảnh left_*.jpg"

subprocess.run([
    sys.executable,
    "predict_official_ewasr.py",
    "--input", str(test_images[0].parent),
    "--download",
    "--output-dir", "/kaggle/working/official_ewasr_predictions",
], check=True)
```

Script tải release chính thức `0.1.0`, kiểm tra SHA-256 và sinh ba file cho mỗi
ảnh: mask class-id, mask màu và overlay. So sánh `_overlay.jpg` của model chính
thức với output model tự train trên chính xác cùng tập ảnh. Nếu model chính thức
tốt hơn rõ rệt, lỗi chính nằm ở training recipe/checkpoint chứ không phải bản
thân kiến trúc eWaSR.
