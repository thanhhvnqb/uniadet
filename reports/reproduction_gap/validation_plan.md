# Kế hoạch kiểm chứng tiếp theo

Đây là kế hoạch, không phải công việc đã thực thi. Không tự launch sweep trong phiên báo cáo. Tất cả thay đổi giữ backbone frozen, output mới, snapshots/config rõ ràng, seeds 10/20/30, epoch_15.pt và metric definitions hiện hành.

## 1. Khóa bằng chứng hiện có

Giữ baseline và `retain_baseline` từ source validation. Không chọn raw_image_peak bằng target diagnostic VisA. Bảo toàn checkpoint/config/code diff ban đầu; dùng [provenance](../feature_ablation/provenance.json) và manifests để truy dấu. Nếu khôi phục context raw từ storage khác, kiểm tra full paths/config/shot/reference/seed rồi mới nâng trạng thái từ archived-only sang raw-verified.

## 2. Cô lập training details trên source holdout

Ưu tiên một giả thuyết mỗi lượt: temperature, hoặc initialization, hoặc một chi tiết optimizer/loss/augmentation đã chỉ rõ. Dùng cùng source training/heldout categories, augmentation, backbone revision, batch, seed và checkpoint khi không phải yếu tố đang thử. Holdout categories phải bị loại cả train và CAA. Nếu thay normalization/representation phải retrain head; không reuse checkpoint không tương thích.

Tạo matched baseline cùng protocol/code version. Khai báo metric chọn, giá trị ứng viên và điều kiện chọn trước; primary P-AUPR và I-AUROC, báo cả bảy metric. Không tạo sweep sau khi xem target để tìm số khớp paper. Với score-only change, dùng cùng head/ảnh, fixed returned map để tách ảnh hưởng scoring; source holdout mới đã cho negative result cho raw nên chỉ nghiên cứu tiếp khi có giả thuyết mới rõ ràng.

Hai category pcb1/pcb2 hẹp: có thể thiết kế thêm source holdout groups trước lần chạy mới để kiểm tra tính ổn định. Không coi hai-category aggregate 19.39 P-AUPR là tụt từ full-VisA 33.42: population và source protocol khác nhau.

## 3. Đánh giá độc lập sau khi khóa cấu hình

Chọn trên source, khóa inference/training rule, rồi benchmark target độc lập. VisA→Real-IAD là default cần ba seed và aggregate đầy đủ; reciprocal VisA/MVTecAD là family riêng. Nếu thay bug ảnh hưởng prediction/protocol, tạo baseline version mới, không rewrite baseline lịch sử. Một kết quả Real-IAD thiếu seed hoặc chỉ đến epoch cuối train không chứng minh evaluation hoàn tất.

Chạy tối đa ba job trên physical GPUs 0/2/3 theo resources thật; không probe GPU1, không terminate workload khác. Budget RAM/workers/threads; cache weights trước, log PID/GPU/command/status, wait từng child. Smoke train→save→load→evaluate trước long benchmark và full suite theo project instructions.

## 4. Nếu tiếp tục descriptors

Components có tín hiệu MVTecAD segmentation: dùng control có cùng MLP/capacity với descriptor không mang thông tin đã định nghĩa trước, cùng normalization/training schedule, để phân biệt capacity và content. Chưa đủ bằng chứng để combine nhiều descriptor. Residual cấu hình hiện tại thất bại rõ nên không ưu tiên scale thêm capacity. Spatial là experiment chưa chạy, không phải fix baseline đã biết.

## 5. Tiêu chí hoàn tất báo cáo mới

Đủ ba seed train và evaluation hợp lệ; verify epoch/config/backbone/shot/reference; category macro trước seed aggregate; sample SD, paired delta và số seed tăng. Báo cả chi phí parameter, full-run peak VRAM, loading/inference tách metrics. Mean tăng đơn độc không phải significance. Kết luận nguyên nhân chỉ khi controlled change ủng hộ; vẫn phân biệt source validation, independent benchmark, target diagnostic và paper external gap.

Lệnh/runner hiện hành ở [experiment guide](../feature_ablation/experiment_guide.md); đọc help/dry-run trước khi chạy. Không dùng lệnh context cũ khi script đã không còn.
