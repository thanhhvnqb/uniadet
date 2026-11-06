# Phân tích khoảng cách với paper

## 1. Khoảng cách đã đo

DINOv3 ViT-L/16, zero-shot, cùng nguồn phụ trợ theo target; paper detailed Tables 14/17. Local là category macro rồi mean ± sample SD ba seeds 10/20/30, epoch 15. Paper không có paired seed local, giữ precision công bố.

| Nguồn | MVTecAD P-AUPR | MVTecAD I-AUROC | VisA P-AUPR | VisA I-AUROC |
|---|---:|---:|---:|---:|
| Paper external | 52.7 | 94.0 | 32.5 | 91.9 |
| Local baseline | 51.61 ± 0.14 | 92.47 ± 0.27 | 33.42 ± 0.42 | 88.91 ± 0.53 |
| Local − paper, pp | −1.09 | −1.53 | +0.92 | −2.99 |

Đây không phải tình huống mọi metric cùng thấp: VisA pixel AUPR đã cao hơn published value, trong khi image detection thấp. Không nên thay toàn pipeline chỉ vì một số tổng thấp hơn. [paper_gaps.csv](paper_gaps.csv) lưu bảy metric, từng category, source table, seed/checkpoint và raw source.

I-AUROC gap lớn ở MVTecAD: toothbrush khoảng −5.97, pill −5.21, transistor −5.19, cable −3.47 pp; ở VisA: pcb1 −18.51, cashew −8.09, macaroni2 −4.34, pcb3 −3.39 pp. Đây là nơi gap tập trung so với category paper đã làm tròn, không cho phép chọn inference riêng từng category. Dùng ranking này để chẩn đoán, không để tune target.

## 2. Image scoring là điểm nhạy, nhưng raw peak không phải fix đã xác nhận

Baseline kết hợp global classification và peak của segmentation, mean theo layer. Smoothing trước peak làm mất đỉnh hẹp; `max(mean(map))` khác `mean(max(map))`. Đây là cơ chế toán học giải thích vì sao image score có thể đổi mà returned map không đổi, chưa chứng minh paper dùng quy tắc nào khác.

Diagnostic trên target:

| Rule | MVTecAD I-AUROC | VisA I-AUROC | Diễn giải |
|---|---:|---:|---|
| Baseline layer_fusion | 92.47 ± 0.27 | 88.91 ± 0.53 | Baseline giữ nguyên |
| Unsmoothed layer fusion | 92.34 ± 0.29 | 90.94 ± 0.28 | VisA +2.03 pp nhưng MVTecAD −0.13 pp |
| Global only | 91.70 ± 0.17 | 90.61 ± 0.47 | Bỏ map peak cũng không thắng cả hai dataset |

Các diagnostic dùng target labels để tính so sánh; không phải cấu hình được chọn độc lập. Giả thuyết “bỏ smoothing image peak sẽ tạo baseline tốt hơn” đã qua thử nghiệm source-only:

| Source holdout pcb1/pcb2 | I-AUROC | P-AUPR |
|---|---:|---:|
| Matched baseline | 88.41 ± 0.34 | 19.39 ± 0.37 |
| Raw image peak | 87.87 ± 0.52 | 19.39 ± 0.37 |

Raw giảm 0.535 pp, 0/3 seed tăng; quyết định `retain_baseline`. Hai-category holdout không chứng minh raw luôn kém trên mọi distribution, nhưng không đạt điều kiện chọn đã khai báo. Không chọn raw vì nó gần paper VisA hơn. [Evidence/selection](../../runs/baseline/reproduction_review/source-score-20261106T043532Z/source-validation/comparison.md).

## 3. Những lựa chọn implement còn chưa đối chiếu đủ

| Lựa chọn local | Vì sao có thể ảnh hưởng | Trạng thái bằng chứng |
|---|---|---|
| Temperature 0.07 | Thay logit scale và gradient khi train | Paper chưa đủ chi tiết để chứng nhận exact match; chưa có controlled sweep |
| Gaussian W init std D^−0.5 | Cosine normalize giữ hướng nhưng initialization ảnh hưởng optimization | Chưa có ba-seed control chỉ đổi init |
| Adam betas (0.5,0.999), loss details | Ảnh hưởng head optimization; focal smoothing/normal Dice có thể khác cách code | Core loss structure đã review; exact undocumented values chưa được xác nhận |
| Softmax trước resize, sigma=4 | Thay calibration/map và image peak; thứ tự softmax/resize ảnh hưởng loss | Explicit local defaults; historical selection dùng MVTec target |
| Multi-layer peak reduction | Mean của maxima khác maximum của mean | Paper chưa xác định đủ final reduction; diagnostic không giải quyết chọn độc lập |
| Crop boundaries/geometric transforms | Có thể thay phân bố supervision/feature | Mask-preservation đã kiểm tra; exact distribution paper chưa xác nhận |
| Pretrained identity/checkpoint metadata cũ | Khác pretrained tensors có thể thay toàn representation | Cache revision và parity đã kiểm tra có giới hạn; checkpoint cũ thiếu provenance nội tại đầy đủ |

Đây là **hypothesis/unresolved match**, không phải danh sách bug đã chứng minh. Không cộng giả định contribution của từng mục thành gap 1.53 hoặc 2.99 điểm. Phải thay một yếu tố một lần, source holdout selection, matched three-seed control trước khi đưa ra giải thích nhân quả.

## 4. Những vấn đề đã xác nhận nhưng không giải thích trực tiếp accuracy gap

- **Target development lịch sử:** softmax/smoothing defaults từng chọn từ MVTec target measurements theo [reproduction guide](../../REPRODUCTION_GUIDE.md). Đây là giới hạn tính độc lập của benchmark; không có bằng chứng nó gây chính xác X pp giảm. Kết quả phải công bố là local reproduction với hạn chế này.
- **Broken pipe và launcher:** ba residual VisA evaluation từng dừng; worker/tee có đường SIGPIPE/EPIPE khi reader đóng. Đã sửa forwarding, disk logging và wait-status; ba evaluation phục hồi bằng exact checkpoint. Việc này giải thích thiếu kết quả/scheduler dừng, không giải thích baseline đã hoàn thành kém paper. Nguyên nhân reader gốc đóng chưa có record.
- **Output overwrite protection:** `cli.train` trước đây có thể ghi đè/mix output; đã reject thư mục không rỗng. Chưa có bằng chứng sáu baseline checkpoint bị ghi đè hoặc corruption bởi lỗi này.
- **Few-shot GPU bank:** CPU backing sửa khả năng chứa reference bank, kiểm tra exact NN sai khác 0.0 trên test CUDA. Zero-shot không dùng bank nên vấn đề này không giải thích gap zero-shot.
- **Artifact scope:** spatial chưa chạy, Real-IAD chưa có baseline hoàn chỉnh. Context raw không còn tại workspace. Các điểm này giới hạn kết luận/phạm vi tái lập, không phải nguyên nhân định lượng đã chứng minh.

## 5. Những nghi vấn đã có bằng chứng loại trừ một phần

Audit encoder: cùng cached HF tensors được chuyển vào Meta DINOv3 upstream với strict shape/key mapping. Một real query tại 224/512, năm intermediate CLS/patch layers: minimum mean token cosine 0.99999976, max absolute error 0.0000849. Điều này làm yếu nghi vấn wrapper sai lớn trên các input được kiểm tra; **không** chứng minh pretrained source độc lập hoàn toàn hay mọi ảnh đều parity. [encoder_parity.json](../../runs/baseline/dinov3/0_shot/audit_20261106/encoder_parity.json).

Dataset test counts VisA 962 normal + 1200 anomalous, MVTecAD 467 normal + 1258 anomalous; layers 12/15/18/21/24, source labeled split, image size 512, encoder frozen, 20,480 head params đã kiểm tra. AUPR/AUPRO definitions và raw-to-CSV aggregation đã kiểm tra; không có căn cứ đổi AUPR sang AP để “đạt paper”. Những kiểm tra này thu hẹp nghi vấn, không thay cho exact official training reproduction.

## 6. Feature mới không phải lời giải cho baseline gap

Cluster/components/directional/kNN/residual là architecture hypotheses, không phải repair baseline. Components có segmentation gain nhỏ ở MVTecAD nhưng giảm VisA; residual giảm mạnh. Mọi adapter tăng capacity, chưa có matched capacity control. Không dùng một variant thắng một category để tuyên bố baseline đã tái lập paper.

Contextual memory CLIP trong archive là one-shot; normal mosaic cũng dùng target normal dù flag 0, random mosaic là transductive. Các số đó không giải thích gap zero-shot DINOv3. Paper summary/detail inconsistencies đã cache thuộc những row khác (đặc biệt DINOv2 Real-IAD 2/4-shot và DINOv3 Real-IAD 2-shot); không được dùng để phủ nhận gap DINOv3 0-shot MVTecAD/VisA.

Kết luận bằng chứng hiện tại: core structure phù hợp và chưa thấy defect toán học được chứng minh giải thích cả gap; final scoring/training details còn uncertainty, trong khi kiểm chứng source mới bác bỏ raw-peak promotion. [Kế hoạch kiểm chứng](validation_plan.md) là bước tiếp theo, không phải benchmark đã chạy.
