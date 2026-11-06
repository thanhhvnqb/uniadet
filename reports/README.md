# Reports

1. [Paper results reference](paper/uniadet_2601.05552v1/README.md): versioned dataset and category results for all three backbones and reported shot counts. Reuse the local CSV/JSON cache; ordinary comparisons require no paper parsing.
2. [Current experiment comparison](experiment_comparison.md): paper results first, reproduced baseline second, other experiments third. Includes recovered residual evaluations and flags spatial as unrun.
3. [Category comparisons](experiment_categories.md) and [category CSV](experiment_category_comparison.csv): matching paper rows above the baseline and variants for every category.
4. [Training/testing reproduction review](reproduction_code_review.md): verified behavior, repairs, validation evidence, and commands for source-only validation and independent benchmarks.
5. [Reusable experiment lessons](lessons/experiment_lessons.md).

Experiment narratives, checkpoints, logs and raw metrics remain in runs/<direction>/. Reports here index and compare those artifacts. Rebuild current comparisons offline with `python scripts/build_experiment_report.py`.

## Tổng hợp thực nghiệm và khoảng cách reproduction

- [Toàn bộ kết quả và implementation](feature_ablation/README.md): context CLIP lịch sử, DINOv3 feature ablations, score diagnostics, source holdout mới; đủ bảy metric, seed/category deltas và provenance.
- [Các vấn đề còn mở so với paper](reproduction_gap/README.md): quantitative gap, issue tracker, implementation audit và kế hoạch kiểm chứng; phân biệt defect đã sửa với giả thuyết chưa chứng minh.

Cập nhật 2026-11-06: source-validation ba seed đã hoàn tất và chọn `retain_baseline`. Context raw không còn trong workspace; báo cáo context được giữ như archive. DINOv3 family có 36/42 job hoàn tất sau residual recovery, sáu spatial chưa chạy. Rebuild bộ CSV mới bằng `python scripts/build_experiment_archive.py` sau khi build comparison hiện hành.
