# Toàn bộ thực nghiệm và implementation

Cập nhật 2026-11-06. Kết luận hiện tại: giữ baseline DINOv3; chưa có descriptor cải thiện nhất quán trên hai hướng chuyển miền. Thử nghiệm image score trên source holdout đã hoàn tất và chọn `retain_baseline`.

| Tài liệu | Nội dung |
|---|---|
| [discussion.md](discussion.md) | Toàn bộ nhóm kết quả, đánh giá, seed stability, category và chi phí |
| [implementation.md](implementation.md) | Công thức baseline, sáu descriptor, chuẩn hóa, head, loss, checkpoint và runner |
| [experiment_guide.md](experiment_guide.md) | Nguồn artifact, cách rebuild, giới hạn và cách kiểm tra cấu hình |
| [summary.csv](summary.csv), [results.csv](results.csv) | DINOv3 zero-shot: bảy metric, summary và variant × seed × category; residual gồm recovery |
| [experiment_comparison.csv](experiment_comparison.csv), [experiment_category_comparison.csv](experiment_category_comparison.csv) | Paper → baseline → variants; không trộn backbone hoặc shots |
| [source_validation_summary.csv](source_validation_summary.csv), [source_validation_results.csv](source_validation_results.csv) | Holdout pcb1/pcb2 riêng; không phải cross-dataset benchmark |
| [score_diagnostics.csv](score_diagnostics.csv) | Các image-score diagnostic trên target labels, không dùng chọn baseline |
| [context_results_archived.csv](context_results_archived.csv) | 28 hàng summary context CLIP từ báo cáo lịch sử, tách shot/data access |
| [artifact_inventory.csv](artifact_inventory.csv) | 42 evaluation: protocol, source/target, checkpoint/config/raw paths và SHA-256 |
| [provenance.json](provenance.json) | SHA-256 các đầu vào, số evaluation đã đối chiếu, phần chưa kiểm chứng được |
| [Khoảng cách với paper](../reproduction_gap/README.md) | Vấn đề đã xác nhận, giả thuyết, bằng chứng loại trừ và kế hoạch kiểm chứng |

Phạm vi bao gồm context CLIP lịch sử (báo cáo 84 evaluation), DINOv3 feature family (36/42 job có evaluation), image-score audit (6 bộ prediction được phân tích bằng nhiều luật), source holdout (3 training job, 6 evaluation) và smoke/code review. Không cộng diagnostic, alias `mosaic_a`, bản recovery và bản báo cáo cũ thành các training job độc lập.

JSON gốc của 36 evaluation chuyển miền và 6 holdout còn hiện diện, đã kiểm tra lại bảy metric. Context chỉ còn summary/report ở workspace này: 84 đường dẫn raw JSON không tồn tại; trạng thái hoàn tất của context là thông tin lịch sử, không phải xác minh lại trong phiên này. Không có completed benchmark Real-IAD hay spatial mới để báo cáo. Dữ liệu, checkpoint, log và lịch sử vẫn ở vị trí gốc; không di chuyển hoặc thay thế kết quả cũ.
