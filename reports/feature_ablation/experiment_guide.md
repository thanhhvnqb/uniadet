# Artifact, tái lập bảng và đọc cấu hình

## Rebuild offline

Chạy từ root repository:

```bash
python scripts/build_experiment_report.py --help
python scripts/build_experiment_report.py
python scripts/build_experiment_archive.py
```

Lệnh đầu kiểm tra interface, lệnh hai build entry point comparison hiện có; lệnh ba đối chiếu raw JSON, backbone/shot/seed/epoch và source holdout trước khi xuất CSV/provenance trong thư mục này và `reproduction_gap/paper_gaps.csv`. Không tải paper/weights, không train/evaluate, không sửa checkpoint. Markdown giải thích là tài liệu biên tập; cập nhật cùng bảng nếu có run mới. Script archive dùng đúng các nguồn dưới đây, chưa tự chọn run mới theo thời gian.

## Nguồn được chọn

| Nhóm | Nguồn chuẩn trong phiên tổng hợp |
|---|---|
| Paper | [cache v1](../paper/uniadet_2601.05552v1/README.md), detailed Tables 12–34, đúng backbone/shots |
| DINOv3 family | [family_results.csv](../../runs/baseline/dinov3/0_shot/audit_20261106/family_results.csv), [family_summary.csv](../../runs/baseline/dinov3/0_shot/audit_20261106/family_summary.csv) |
| Original plan/snapshots | [corrected launch](../../runs/feature_ablation/launches/dinov3-features-visa-mvtec-fix/plan.json) |
| Residual recovery | [manifest](../../runs/residual/dinov3/0_shot/recovery_20261106/manifest.json); giữ original errors và dùng metrics recovery |
| Score diagnostic | [audit summary](../../runs/baseline/dinov3/0_shot/audit_20261106/summary.csv), [discussion](../../runs/baseline/dinov3/0_shot/audit_20261106/discussion.md) |
| Source validation | [comparison](../../runs/baseline/reproduction_review/source-score-20261106T043532Z/source-validation/comparison.md), [selection](../../runs/baseline/reproduction_review/source-score-20261106T043532Z/source-validation/selection.json) |
| Code review/smoke | [review](../../runs/baseline/reproduction_review/20261106_code_review/discussion.md), không thay cho benchmark |
| Context archive | [CSV](../context/experimental_results.csv), [guide](../context/context_experiment_guide.md); raw không còn |

Đường dẫn raw `source` trong CSV được giữ nguyên để truy dấu, có thể là absolute path của workspace gốc. Khi chuyển workspace, phải map lại root một cách tường minh; script không tự tìm checkpoint khác để thay thế.

`results.csv` chứa variant × regime × seed × category và macro; `summary.csv` chứa mean/sample SD, valid categories/seeds, paired delta và số seed tăng. Spatial thiếu có mặt trong paper-first comparison với status incomplete; không tạo giả hàng complete trong results. Holdout CSV riêng vì dataset-name VisA không đồng nghĩa full-VisA benchmark. Context CSV có `shots=0` ở normal mosaic nhưng phải dùng thêm actual target access để phân nhóm.

## Cách chạy lại nếu sau này có yêu cầu

[run_feature_ablation.py](../../scripts/run_feature_ablation.py), [launcher](../../scripts/run_feature_ablation_parallel.sh), [source-validation launcher](../../scripts/run_reproduction_review.sh). Đọc `--help`/`--dry-run` và snapshot trước khi thực thi; các ví dụ context cũ không còn script trong checkout này. Mọi run mới phải output riêng, immutable config, giữ encoder frozen, seeds 10/20/30 và epoch 15; không ghi đè dữ liệu cũ. `--start-epoch` không resume train. Báo cáo này không cấp kết quả benchmark cho một command mới chưa chạy.

GPU vật lý 1 bị lỗi: không probe/schedule. Khi có chạy mới, kiểm tra GPUs 0/2/3 và môi trường Python thực tế, queue nếu bận. Không chạy benchmark chỉ để dựng lại báo cáo.
