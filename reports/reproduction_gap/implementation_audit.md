# Đối chiếu implement và trạng thái xác minh

[Chi tiết công thức implementation](../feature_ablation/implementation.md). Review này dùng code hiện tại, config thực chạy và artifact code review; không nói mọi lựa chọn local đã được tác giả paper xác nhận.

| Thành phần | Local implementation/evidence | Kết luận có thể rút ra |
|---|---|---|
| Encoder | Frozen eval/no_grad, DINOv3 normalized CLS/patch bỏ storage tokens; upstream parity trên input được test | Không thấy wrapper discrepancy lớn trên mẫu parity; chưa xác minh exhaustive |
| Layers/resolution | 12/15/18/21/24, 512, ViT-L/16 | Match cấu trúc được báo cáo |
| Head | Two-class image/pixel cosine heads độc lập theo layer; 20,480 params | Giữ cấu trúc baseline; không đổi capacity theo claim ngắn trong abstract |
| Train data | VisA labeled test cho MVTecAD; MVTecAD labeled test cho VisA | Match source routing; không dùng normal-only source train |
| Loss | Sum per-layer CE + focal + normal/anomaly Dice | Match family công thức; exact undocumented hyperparameters vẫn chưa chắc |
| Training | Epoch 15, lr 0.001, batch 8, fixed checkpoint | Không best-target-epoch selection trong family này |
| Augmentation | CAA p=.5, grids 2/3, mosaic=.5; mask-preservation, geometry tắt | Mask supervision hợp lệ; exact crop distribution paper còn uncertainty |
| Zero-shot | Không target references, descriptor per-query, source normalization fixed eval | Không phát hiện target-label/cross-image descriptor access trong DINOv3 feature family |
| Image fusion | Mean g_l và mean layer peak, lambda=.5; sigma=4 | Lựa chọn explicit, chưa chứng nhận exact final reduction paper |
| Metric | PR area, region-balanced AUPRO FPR .3; macro/category alias | Không thay AP hoặc tính average alias thành category |
| Checkpoint | Feature config/state/buffers restore; eval override ghi riêng; old defaults giữ lịch sử | Không tự dùng train default mới cho old checkpoint |
| Pretrained provenance | HF commit/explicit weight SHA lưu và verify ở code mới | Checkpoint cũ không có identity nội tại thì vẫn có giới hạn |
| Artifact | Snapshot code/dirty diff, job statuses, recovery riêng | Không xóa failure ban đầu hoặc trộn rerun |

## Repairs đã có và giới hạn của chúng

| Repair | Bằng chứng kiểm tra đã lưu | Không được suy diễn |
|---|---|---|
| Reject nonempty train output | Regression bảo toàn checkpoint bytes | Không chứng minh accuracy tăng |
| CPU-backed exact-NN bank | CUDA comparison max difference 0.0 | Không phải benchmark latency; không liên quan zero-shot bank-free |
| Image-score sigma độc lập | Old implicit vs explicit sigma=4 bitwise equal trên bốn real image; smoke pixel metrics equal | Không chứng minh raw sigma=0 tốt hơn |
| Source-validation planner | Holdout loại khỏi train/CAA, config explicit, seeds/paired aggregation checked | Hai-category holdout không phải full-dataset benchmark |
| Pipe-safe progress/tee/wait | Regression reader đóng cho success/failure; real residual recovery hoàn tất | Reader gốc đóng do đâu chưa biết; không giải thích baseline gap |
| Weight identity/check | Real smoke HF revision verified | Không retroactively xác minh mọi historical checkpoint |

[Review gốc](../../runs/baseline/reproduction_review/20261106_code_review/discussion.md): **210 tests pass** và single-seed CUDA smoke train/save/load/eval. Đây là số liệu validation lịch sử đã lưu, không phải full test suite chạy mới trong phiên viết docs. Smoke 4 source images/8 heldout images không là benchmark. [Audit cũ](../../runs/baseline/dinov3/0_shot/audit_20261106/validation.json) có 189 tests ở thời điểm trước; không cộng 189+210 thành một test run.

## Đọc đúng timeline

1. Original feature launch bị sai default target so với ý định; corrected plan VisA↔MVTecAD và baseline reuse sau verify cấu hình.
2. Audit đầu: 33 complete, 3 residual evaluation failed, 6 spatial queued.
3. Recovery: ba residual VisA complete, giữ train/checkpoint; total family 36/42 evaluation complete.
4. Code review: repairs, 210 tests, single-seed smoke; văn bản lúc đó ghi “no new full benchmark”.
5. Sau đó source-validation `source-score-20261106T043532Z` đã chạy đủ ba training seed/sáu evaluation, chọn retain_baseline. Câu “source validation chưa chạy” trong review cũ là trạng thái tại thời điểm review, không là trạng thái hiện tại.
6. Phiên tổng hợp hiện tại: verify raw/CSV và viết reports, không chạy training/evaluation.

Không trộn original/recovery counts, smoke với benchmark, hoặc biến những repaired operational bug thành nguyên nhân accuracy mà chưa có paired experiment.
