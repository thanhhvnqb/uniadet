# Vì sao chưa tái lập được kết quả paper?

**Chưa xác định được một nguyên nhân duy nhất bằng thực nghiệm có kiểm soát.** Đã xác nhận gap chủ yếu ở image detection, một số chi tiết training/inference chưa có đối chiếu chính xác với paper, và protocol lịch sử có target-based development. Không được biến những điểm chưa rõ này thành kết luận “đây là lỗi gây giảm X điểm”.

| Tài liệu | Nội dung |
|---|---|
| [gap_analysis.md](gap_analysis.md) | Bằng chứng gap, vấn đề còn mở, giả thuyết và điều đã loại trừ một phần |
| [implementation_audit.md](implementation_audit.md) | Đối chiếu công thức/protocol, repair, test và giới hạn xác minh |
| [validation_plan.md](validation_plan.md) | Thứ tự kiểm chứng tiếp theo, control và tiêu chí dừng |
| [issues.csv](issues.csv) | Danh sách vấn đề có trạng thái, evidence, impact và action |
| [paper_gaps.csv](paper_gaps.csv) | Gap đủ bảy metric ở dataset/category; external unpaired, DINOv3 zero-shot |
| [Toàn bộ kết quả và implement](../feature_ablation/README.md) | Kết quả feature/context/holdout và chi tiết implementation |

Không có benchmark Real-IAD ba seed hoàn chỉnh để xác nhận protocol mặc định. Spatial chưa chạy. Cả hai là **thiếu phạm vi đánh giá**, không tự giải thích gap của baseline MVTecAD/VisA. Source validation mới chọn `retain_baseline`; raw_image_peak không được chấp nhận làm fix accuracy.

Paper được đọc từ [cache v1](../paper/uniadet_2601.05552v1/README.md), không reparse hoặc lấy kết quả khác backbone/shots thay thế. Bảng paper trước, local baseline sau. Tài liệu cập nhật 2026-11-06, không khởi chạy thí nghiệm mới.
