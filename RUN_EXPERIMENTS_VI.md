# Chạy và phát triển UniADet

Mọi cấu hình chỉnh trực tiếp trong `config.py`. Không còn lựa chọn thí nghiệm
bằng tên hoặc tự đổi dữ liệu/checkpoint theo tên thí nghiệm.

```bash
uv run python train_test.py --dry-run
uv run python train_test.py --mode train
uv run python train_test.py --mode test
uv run python train_test.py
```

- `MODEL`: tên thư mục kết quả; đổi tên cho mỗi cấu hình mới.
- `TRAIN_CONFIG`: backbone, dữ liệu, augmentation, loss, optimizer và prediction mặc định.
- `TEST_CONFIG`: dữ liệu đích, shots, epochs, checkpoint và prediction override.
- `MODE`: chế độ mặc định khi không truyền `--mode`.

Mặc định train toàn bộ VisA, test MVTec tại epoch 15; softmax trước resize,
Gaussian sigma 4 và layer pooling. `None` trong các prediction override của
`TEST_CONFIG` lấy giá trị đã lưu trong checkpoint. Đổi representation cần train
head mới. Checkpoint cũ vẫn dùng các giá trị mặc định lịch sử khi thiếu trường.

Ví dụ tắt smoothing khi test: đặt `TEST_CONFIG["gaussian_sigma"] = 0.0`,
`TEST_CONFIG["checkpoint"]` trỏ đến checkpoint đã train, rồi đổi `MODEL` để lưu
kết quả riêng và chạy `--mode test`. So sánh sigma 0 và 4 phải giữ cùng checkpoint,
pooling, dữ liệu và seed.

Để hold out category, đặt `TRAIN_CONFIG["exclude_categories"] = ["pcb1", "pcb2"]`,
`TEST_CONFIG["data"] = ["datasets/VisA"]` và
`TEST_CONFIG["categories"] = ["pcb1", "pcb2"]`. Đổi `MODEL` trước khi train.

`--device`, `--batch-size`, `--workers`, `--seed` chỉ override trong lần chạy hiện tại.
`--start-epoch` chỉ lọc checkpoint khi test, không resume training.
`TEST_CONFIG["epochs"] = None` quét mọi epoch đã lưu; mặc định là `[15]`.

## Thêm thay đổi mới ở đâu?

| Phần cần thay đổi | File |
|---|---|
| Cấu hình một lần chạy | `config.py` |
| CLI và vòng train/evaluate | `uniadet/cli.py` |
| Resolve, validate và restore checkpoint settings | `uniadet/settings.py` |
| Đường dẫn, quét checkpoint, tổng hợp nhiều lần test | `uniadet/workflow.py` |
| Image/pixel metrics và xuất báo cáo | `uniadet/evaluation_metrics.py` |
| Backbone và feature extraction | `uniadet/backbones.py` |
| Head, memory bank và prediction | `uniadet/model.py` |
| Các thành phần loss | `uniadet/loss.py` |
| Dataset và augmentation | `uniadet/data.py` |

`train_test.py` chỉ đọc cấu hình và điều phối. `RunConfig` tạo bản sao sâu cho từng
lần chạy; các hàm workflow nhận cấu hình qua tham số, không sửa biến toàn cục.
Defaults cho flag nằm trong parser; các setting cần restore theo checkpoint nằm
trong `settings.py`. Khi thêm setting, xác định rõ nó thay đổi representation
(cần train lại) hay chỉ prediction (được override khi test), rồi thêm validation
và kiểm tra checkpoint phù hợp. Không thêm nhánh theo tên thí nghiệm.

Kiểm tra trước khi chạy dữ liệu thật:

```bash
uv run python -m pytest -q
uv run python train_test.py --dry-run
```

Output: `runs/<MODEL>/train_<dataset>/epoch_<N>.pt`,
`runs/<MODEL>/<target>_<shots>shot/epoch_<N>/`, và `test_summary.csv`.
Xem `REPRODUCTION_GUIDE.md` để biết chi tiết chuẩn bị dữ liệu và các tham số.
