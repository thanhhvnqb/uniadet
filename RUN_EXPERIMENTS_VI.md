# Chạy từng thí nghiệm với train\_test.py

Các lệnh dưới đây đã có preset trong `train_test.py`; bạn không cần sửa
`TRAIN_CONFIG` giữa các lần chạy. Chạy lần lượt từ thư mục repo. Mỗi preset
training lưu trong một thư mục riêng.

Các preset mặc định train trên toàn bộ VisA test split có nhãn và đánh giá
zero-shot trên toàn bộ MVTec ở epoch 15. Không loại pcb1/pcb2 khỏi training.
Bạn sẽ chọn cấu hình bằng kết quả MVTec theo yêu cầu; đây là lựa chọn cấu hình
dựa trên target-test, và cần ghi rõ cách chọn này khi báo cáo kết quả.
Các lựa chọn lấy từ AdaptCLIP là giả thuyết cần kiểm tra cho UniADet.

## 1. Kích hoạt môi trường, kiểm tra CUDA và chuẩn bị metadata

```bash
cd /home/islab/DATA/anomaly/uniadet
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest -q
python scripts/prepare_datasets.py --datasets visa_mvtec
```

CUDA cần in `True`. Nếu chưa có môi trường, chạy `uv sync --locked` trước khi
activate. Metadata preparation cần ảnh đã giải nén tại `datasets/VisA` và
`datasets/MVTecAD`; nó không tải ảnh.

## 2. Xem trước cấu hình baseline

```bash
python train_test.py --experiment baseline --device cuda --dry-run
```

Lệnh này chỉ in cấu hình, không train và không tạo kết quả. Kiểm tra:

- `exclude_categories`: `null` trong training: dùng toàn bộ VisA.
- Dataset đánh giá là MVTecAD, `categories`: `null`: dùng toàn bộ category.
- `epochs`: 15 trong training; `[15]` trong testing.
- `preserve_anomaly_mask`: `true`; Gaussian sigma: 0.

## 3. Train và đánh giá baseline full VisA → zero-shot MVTec

```bash
python -u train_test.py --experiment baseline --device cuda
```

Chờ hoàn tất trước khi chạy hai preset Gaussian. Kết quả:

```text
runs/uniadet-final-baseline/train_VisA/epoch_15.pt
runs/uniadet-final-baseline/MVTecAD_0shot/epoch_15/metrics.csv
runs/uniadet-final-baseline/test_summary.csv
```

## 4. Thử softmax trước resize - Tăng từ 90.2 -&gt; 90.7

```bash
python -u train_test.py --experiment softmax-first --device cuda
```

Preset này train head mới với luồng:
`cosine logits -> softmax -> resize probability map`.
Những thiết lập khoa học khác giữ nguyên baseline. So sánh:

```text
runs/uniadet-final-softmax-first/test_summary.csv
runs/uniadet-final-baseline/test_summary.csv
```

## 5. Tạo đối chứng Gaussian OFF với map pooling

```bash
python -u train_test.py --experiment gaussian-off --device cuda
```

Lệnh này **chỉ đánh giá**, dùng lại checkpoint baseline từ bước 3. Nó đặt
`zero_score_pooling="map"`, `gaussian_sigma=0`.
Kết quả: `runs/uniadet-final-gaussian-off/test_summary.csv`.

## 6. Bật Gaussian sigma 4 trên cùng checkpoint

```bash
python -u train_test.py --experiment gaussian-on --device cuda
```

Lệnh này cũng **chỉ đánh giá**, với cùng checkpoint và map pooling của bước 5;
chỉ `gaussian_sigma` thay đổi thành 4. Map được làm mượt trước spatial max.
So sánh bước 6 với bước 5 để tách riêng ảnh hưởng của Gaussian. So sánh bước 5
với baseline cho biết ảnh hưởng của thứ tự pooling.

Kết quả: `runs/uniadet-final-gaussian-on/test_summary.csv`.

## 7. Tắt layer normalization của global token

```bash
python -u train_test.py --experiment no-global-norm --device cuda
```

Train mới với `clip_global_norm=False`; patch-token normalization vẫn bật.
Kết quả: `runs/uniadet-final-no-global-norm/test_summary.csv`.

## 8. Đổi positional interpolation sang bilinear

```bash
python -u train_test.py --experiment bilinear-pos --device cuda
```

Train mới với `clip_pos_interpolation="bilinear"`; global normalization trở
về baseline `True`, không kế thừa thay đổi ở bước 7.
Kết quả: `runs/uniadet-final-bilinear-pos/test_summary.csv`.

## 9. Bật focal smoothing giống AdaptCLIP

```bash
python -u train_test.py --experiment smoothed-focal --device cuda
```

Train mới, chỉ đổi `focal_smoothing=1e-5`.
Kết quả: `runs/uniadet-final-smoothed-focal/test_summary.csv`.

## 10. Tùy chọn: thử bilinear cho mask

```bash
python -u train_test.py --experiment bilinear-mask --device cuda
```

Train mới với resize mask bilinear và threshold 0.5. Guard bảo toàn mask vẫn
bật: nếu augmentation làm mất defect, dùng lại ảnh gốc; nếu bilinear làm mất
cả defect nhỏ trong mask gốc, fallback training dùng nearest. Điều này cần thiết
cho 5 mask VisA đã phát hiện khi kiểm tra ở kích thước 518. Evaluation dùng
bilinear/threshold đúng cấu hình, không dùng fallback training.

Kết quả: `runs/uniadet-final-bilinear-mask/test_summary.csv`.

## 11. In bảng so sánh các kết quả đã hoàn tất

```bash
python - <<'PY'
import csv
from pathlib import Path
for path in sorted(Path("runs").glob("uniadet-final-*/test_summary.csv")):
    with path.open() as stream:
        for row in csv.DictReader(stream):
            print(f"{path.parent.name:36} epoch={row['Epoch']:>2} "
                  f"I-AUROC={float(row['I-AUROC']):6.2f} "
                  f"P-AUPR={float(row['P-AUPR']):6.2f}")
PY
```

Xác định trước metric chính; kiểm tra cả image AUROC và pixel AUPR. Các preset
training thay đổi từng yếu tố riêng biệt từ baseline, không tích lũy các thay
đổi trước đó. Đừng tự động gộp những lựa chọn chưa được thử chung.

## 12. Chọn cấu hình theo kết quả zero-shot MVTec

So sánh các file `runs/uniadet-final-*/test_summary.csv`. Tất cả preset ở trên
đã dùng full VisA và zero-shot MVTec; không cần train lại bằng `--full-data`.
Cờ `--full-data` vẫn được giữ để tương thích và tương đương mặc định.

Nếu sau này muốn kiểm tra bằng holdout VisA, thêm `--aux-holdout`:

```bash
python -u train_test.py --experiment baseline --aux-holdout --device cuda
```

Chế độ tùy chọn này lưu riêng dưới `runs/uniadet-aux-baseline/` và không thay
đổi các kết quả MVTec ở trên. Các preset Gaussian có `--aux-holdout` sẽ dùng
checkpoint baseline trong chính nhánh auxiliary đó.

## Chạy lại, điều chỉnh GPU và dùng cấu hình thủ công

Nếu training đã xong và chỉ cần chạy lại evaluation:

```bash
python -u train_test.py --experiment softmax-first --mode test --device cuda
```

Nếu cần giảm batch size, thêm `--batch-size 4` vào **tất cả** các thí nghiệm
đang so sánh. Có thể thêm `--workers 0` để tắt data-loader workers. Ví dụ:

```bash
python -u train_test.py --experiment baseline --device cuda --batch-size 4 --workers 0
```

Preset dùng batch size 8 và seed 10 nếu không override. Chạy lại cùng preset
sẽ dùng lại đường dẫn đó; training không hỗ trợ resume và có thể ghi đè
checkpoint cũ. Dùng `--mode test` nếu chỉ cần đánh giá lại.

Không truyền `--experiment` thì runner vẫn dùng `MODEL`, `TRAIN_CONFIG` và
`TEST_CONFIG` bạn chỉnh trong file như trước. Đây là cách tạo thí nghiệm phối
hợp hoặc lặp nhiều seed với tên `MODEL` riêng. Xem
[REPRODUCTION\_GUIDE.md](REPRODUCTION_GUIDE.md) để biết đầy đủ các khóa cấu hình.

Việc tham khảo AdaptCLIP chưa xác nhận các lựa chọn này là implementation
chính thức của UniADet; cải thiện điểm số cần được đo qua các lần chạy trên.