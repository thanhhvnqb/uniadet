# Tổng hợp kết quả và đánh giá

Ngày 2026-11-06. Giữ baseline DINOv3. Components chỉ có tín hiệu segmentation MVTecAD; chưa có biến thể descriptor thắng nhất quán cả hai target. Source holdout bác bỏ điều kiện chọn raw_image_peak. Context CLIP có tín hiệu tích cực của contextual one-shot memory trong archive, thuộc backbone/shot khác nên không gộp vào xếp hạng DINOv3.

## Phạm vi và mức xác minh

| Family | Hoàn thành được báo cáo | Kiểm chứng trong workspace hiện tại |
|---|---|---|
| Context CLIP | 84 evaluation; 14 configuration × 2 target × 3 seed; 36 training logs theo báo cáo lịch sử | 28 summary CSV rows còn; 84 raw JSON không còn; không tái xác nhận completion |
| DINOv3 feature family | 36/42 job có evaluation, gồm 3 residual VisA recovery | Đối chiếu JSON/category/macro bảy metric; sáu spatial chưa chạy |
| Fixed image-score audit | Sáu baseline evaluation, nhiều scoring rule mỗi prediction | Kết quả diagnostic riêng, không thêm training và không thay baseline |
| Source category holdout | 3 training job × 2 score rule = 6 evaluation | Bảy metric khớp raw; pcb1/pcb2 loại khỏi training/CAA |
| Reproduction code review | 210 tests và 1 seed smoke theo artifact review | Smoke không phải benchmark; không cộng vào bảng mean ba seed |

Mean/SD tính category macro trong từng seed, sau đó mean và sample SD (`ddof=1`) qua seeds 10/20/30; epoch 15 cố định. JSON [0,1], CSV/table %, delta pp. Không điền metric thiếu bằng zero. AUPR là PR-curve area, không phải AP; AUPRO chuẩn hóa FPR [0,0.3]. Không trộn dataset, backbone, shot hoặc actual data access. Các paper rows là external reported, không paired với seed local.

## 1. DINOv3 zero-shot cross-dataset

VisA auxiliary test có nhãn → MVTecAD test 15 category; MVTecAD auxiliary test có nhãn → VisA test 12 category. Không phải VisA → Real-IAD. Cấu hình/implementation tại [implementation.md](implementation.md).

### MVTecAD: đủ bảy metric

| Phương pháp | Seeds | I-AUROC | I-AUPR | I-F1max | P-AUROC | P-AUPR | P-F1max | P-AUPRO |
|---|---|---|---|---|---|---|---|---|
| paper | external | 94.0 | 97.1 | 94.0 | 93.2 | 52.7 | 51.1 | 90.7 |
| baseline | 3/3 | 92.47 ± 0.27 | 96.43 ± 0.22 | 92.69 ± 0.21 | 91.95 ± 0.11 | 51.61 ± 0.14 | 50.80 ± 0.12 | 89.99 ± 0.13 |
| cluster | 3/3 | 92.07 ± 0.55 | 96.32 ± 0.30 | 93.03 ± 0.37 | 91.81 ± 0.08 | 49.70 ± 0.78 | 48.91 ± 0.53 | 89.85 ± 0.10 |
| components | 3/3 | 92.20 ± 0.17 | 96.47 ± 0.21 | 93.19 ± 0.26 | 92.15 ± 0.11 | 51.88 ± 0.51 | 50.87 ± 0.36 | 90.02 ± 0.04 |
| directional | 3/3 | 92.51 ± 0.15 | 96.58 ± 0.17 | 93.24 ± 0.15 | 92.00 ± 0.16 | 51.38 ± 0.24 | 50.51 ± 0.16 | 89.97 ± 0.10 |
| knn | 3/3 | 92.17 ± 0.21 | 96.50 ± 0.16 | 92.67 ± 0.25 | 91.89 ± 0.18 | 51.22 ± 0.75 | 50.36 ± 0.52 | 90.01 ± 0.06 |
| residual | 3/3 | 89.73 ± 0.86 | 95.25 ± 0.62 | 91.27 ± 0.58 | 87.72 ± 0.25 | 36.47 ± 1.28 | 39.14 ± 0.89 | 83.21 ± 0.58 |
| spatial | 0/3 | — | — | — | — | — | — | — |

| Variant | Δ P-AUPR | Seed tăng | Δ I-AUROC | Seed tăng |
|---|---|---|---|---|
| cluster | -1.911 | 0/3 | -0.391 | 1/3 |
| components | +0.267 | 2/3 | -0.268 | 0/3 |
| directional | -0.229 | 1/3 | +0.047 | 2/3 |
| knn | -0.387 | 2/3 | -0.292 | 0/3 |
| residual | -15.141 | 0/3 | -2.738 | 0/3 |

Kết quả từng seed (P-AUPR / I-AUROC):

| Variant | 10 | 20 | 30 |
|---|---|---|---|
| baseline | 51.691 / 92.744 | 51.684 / 92.201 | 51.451 / 92.452 |
| cluster | 48.794 / 91.492 | 50.124 / 92.137 | 50.176 / 92.593 |
| components | 52.258 / 92.227 | 51.293 / 92.012 | 52.075 / 92.353 |
| directional | 51.563 / 92.339 | 51.107 / 92.636 | 51.469 / 92.563 |
| knn | 51.699 / 92.205 | 50.357 / 91.945 | 51.610 / 92.369 |
| residual | 35.478 / 88.746 | 36.006 / 90.098 | 37.920 / 90.340 |

Category delta P-AUPR so với matched baseline (mean theo seed):

| Variant | Category tăng | Ba delta cao nhất (có thể vẫn âm) | Ba delta thấp nhất |
|---|---|---|---|
| cluster | 0/15 | transistor -0.00; capsule -0.21; leather -0.56 | screw -5.62; cable -4.83; zipper -4.80 |
| components | 11/15 | capsule +1.39; carpet +1.19; bottle +1.01 | screw -0.93; zipper -0.86; metal_nut -0.80 |
| directional | 8/15 | bottle +1.46; carpet +0.92; capsule +0.63 | screw -2.56; leather -1.70; grid -1.37 |
| knn | 4/15 | cable +2.06; hazelnut +1.07; pill +0.22 | metal_nut -1.92; wood -1.81; tile -1.29 |
| residual | 0/15 | transistor -0.61; capsule -3.33; metal_nut -6.84 | tile -40.02; bottle -23.56; toothbrush -23.14 |

### VisA: đủ bảy metric

| Phương pháp | Seeds | I-AUROC | I-AUPR | I-F1max | P-AUROC | P-AUPR | P-F1max | P-AUPRO |
|---|---|---|---|---|---|---|---|---|
| paper | external | 91.9 | 92.8 | 88.1 | 96.3 | 32.5 | 36.0 | 91.7 |
| baseline | 3/3 | 88.91 ± 0.53 | 90.83 ± 0.50 | 85.62 ± 0.09 | 96.40 ± 0.07 | 33.42 ± 0.42 | 37.08 ± 0.30 | 90.50 ± 0.07 |
| cluster | 3/3 | 89.22 ± 0.02 | 91.06 ± 0.19 | 85.83 ± 0.27 | 96.46 ± 0.06 | 33.44 ± 0.35 | 37.20 ± 0.32 | 90.82 ± 0.28 |
| components | 3/3 | 89.03 ± 0.08 | 90.95 ± 0.06 | 85.84 ± 0.17 | 96.38 ± 0.02 | 32.95 ± 0.30 | 36.48 ± 0.35 | 90.63 ± 0.35 |
| directional | 3/3 | 89.20 ± 0.15 | 91.04 ± 0.18 | 86.08 ± 0.04 | 96.44 ± 0.05 | 33.27 ± 0.34 | 36.97 ± 0.36 | 90.52 ± 0.22 |
| knn | 3/3 | 89.24 ± 0.10 | 91.09 ± 0.08 | 85.90 ± 0.12 | 96.37 ± 0.02 | 32.92 ± 0.25 | 36.60 ± 0.30 | 90.44 ± 0.32 |
| residual | 3/3 | 87.98 ± 0.77 | 89.94 ± 0.90 | 84.61 ± 0.31 | 94.80 ± 0.01 | 25.60 ± 0.45 | 30.16 ± 0.53 | 81.40 ± 0.53 |
| spatial | 0/3 | — | — | — | — | — | — | — |

| Variant | Δ P-AUPR | Seed tăng | Δ I-AUROC | Seed tăng |
|---|---|---|---|---|
| cluster | +0.027 | 2/3 | +0.314 | 2/3 |
| components | -0.469 | 1/3 | +0.120 | 1/3 |
| directional | -0.150 | 1/3 | +0.294 | 1/3 |
| knn | -0.494 | 1/3 | +0.330 | 2/3 |
| residual | -7.817 | 0/3 | -0.932 | 1/3 |

Kết quả từng seed (P-AUPR / I-AUROC):

| Variant | 10 | 20 | 30 |
|---|---|---|---|
| baseline | 33.541 / 89.128 | 32.949 / 88.303 | 33.757 / 89.300 |
| cluster | 33.146 / 89.217 | 33.357 / 89.207 | 33.824 / 89.250 |
| components | 32.881 / 89.127 | 33.272 / 88.971 | 32.688 / 88.994 |
| directional | 32.877 / 89.046 | 33.452 / 89.340 | 33.466 / 89.227 |
| knn | 32.904 / 89.361 | 33.184 / 89.171 | 32.677 / 89.191 |
| residual | 25.078 / 87.097 | 25.867 / 88.305 | 25.849 / 88.535 |

Category delta P-AUPR so với matched baseline (mean theo seed):

| Variant | Category tăng | Ba delta cao nhất (có thể vẫn âm) | Ba delta thấp nhất |
|---|---|---|---|
| cluster | 7/12 | pcb3 +1.99; pcb4 +1.26; pcb1 +1.08 | cashew -2.42; macaroni1 -1.80; candle -1.42 |
| components | 3/12 | pcb4 +0.29; fryum +0.23; pcb3 +0.04 | macaroni2 -1.60; candle -1.03; macaroni1 -0.89 |
| directional | 5/12 | pcb4 +1.94; pcb2 +0.55; pcb1 +0.47 | candle -2.35; macaroni2 -1.11; macaroni1 -0.82 |
| knn | 3/12 | pcb2 +0.19; fryum +0.11; pcb4 +0.11 | cashew -1.43; macaroni2 -1.39; candle -1.02 |
| residual | 0/12 | pcb4 -0.65; pcb1 -0.86; pcb3 -1.08 | capsules -31.40; candle -18.46; macaroni1 -14.27 |

### Đánh giá

Components: MVTecAD P-AUPR +0.267 pp, 2/3 seed, 11/15 category tăng; nhưng I-AUROC giảm cả ba seed, VisA P-AUPR −0.469 pp. Directional gần baseline, I-AUROC VisA mean tăng nhưng chỉ 1/3 seed tăng; mean gain không đồng nghĩa ổn định. kNN tăng I-AUROC VisA +0.331 pp nhưng P-AUPR giảm hai dataset. Cluster mất P-AUPR MVTecAD ở cả 15 category. Residual mất P-AUPR ở cả ba seed trên hai dataset, đồng thời head lớn hơn nhiều; cấu hình này không được ủng hộ. Cơ chế nguyên nhân chưa xác định.

Original family audit là 33 complete/3 failed/6 queued; recovery hoàn tất ba residual VisA mà không train lại, thành 36 complete/6 unrun. Failure cũ giữ nguyên để audit. Không dùng summary residual cũ thiếu seed thay cho file đã merge recovery. Spatial 0/3 seed mỗi target: chưa có kết quả để đánh giá.

## 2. DINOv3 image-score diagnostic trên target

Paper và matched baseline đã trình bày ở phần 1; dưới đây là diagnostic dùng chính sáu baseline head. Không có pixel gain từ việc đổi image-score peak, không chọn cấu hình bằng target labels. Đây là các luật phân tích, không 11 training variant độc lập.

### MVTecAD

| Rule | I-AUROC ± SD | Paired Δ pp | Seed tăng |
|---|---|---|---|
| global_only | 91.70 ± 0.17 | -0.762 | 0/3 |
| map_only | 89.41 ± 0.19 | -3.051 | 0/3 |
| legacy_map_fusion | 91.39 ± 0.41 | -1.076 | 0/3 |
| layer_fusion | 92.47 ± 0.27 | +0.000 | 0/3 |
| unsmoothed_layer_fusion | 92.34 ± 0.29 | -0.125 | 0/3 |
| unsmoothed_map_only | 87.99 ± 0.22 | -4.478 | 0/3 |
| global_layer_12 | 79.19 ± 0.15 | -13.273 | 0/3 |
| global_layer_15 | 86.55 ± 0.15 | -5.919 | 0/3 |
| global_layer_18 | 89.95 ± 0.03 | -2.518 | 0/3 |
| global_layer_21 | 90.82 ± 0.29 | -1.641 | 0/3 |
| global_layer_24 | 91.20 ± 0.15 | -1.265 | 0/3 |

### VisA

| Rule | I-AUROC ± SD | Paired Δ pp | Seed tăng |
|---|---|---|---|
| global_only | 90.61 ± 0.47 | +1.700 | 3/3 |
| map_only | 82.24 ± 0.51 | -6.665 | 0/3 |
| legacy_map_fusion | 87.40 ± 0.54 | -1.512 | 0/3 |
| layer_fusion | 88.91 ± 0.53 | +0.000 | 0/3 |
| unsmoothed_layer_fusion | 90.94 ± 0.28 | +2.027 | 3/3 |
| unsmoothed_map_only | 84.67 ± 0.47 | -4.245 | 0/3 |
| global_layer_12 | 77.64 ± 0.47 | -11.270 | 0/3 |
| global_layer_15 | 82.86 ± 0.34 | -6.048 | 0/3 |
| global_layer_18 | 86.16 ± 0.15 | -2.753 | 0/3 |
| global_layer_21 | 86.66 ± 0.32 | -2.252 | 0/3 |
| global_layer_24 | 85.72 ± 0.33 | -3.193 | 0/3 |

## 3. Source-only validation của image score

Paper full-VisA không có matched protocol cho hai category holdout; không tính gap với paper ở đây. Train trên VisA loại pcb1/pcb2 khỏi train và CAA, validate đúng hai category đó. Mỗi seed reuse một head cho baseline và raw_image_peak; returned map sigma=4 cố định, chỉ sigma của image-score peak đổi từ inherit=4 sang 0.

| Variant | I-AUROC | I-AUPR | I-F1max | P-AUROC | P-AUPR | P-F1max | P-AUPRO |
|---|---|---|---|---|---|---|---|
| baseline | 88.41 ± 0.34 | 90.75 ± 0.23 | 81.87 ± 0.74 | 93.34 ± 0.20 | 19.39 ± 0.37 | 25.32 ± 0.40 | 83.59 ± 0.10 |
| raw_image_peak | 87.87 ± 0.52 | 90.16 ± 0.38 | 81.28 ± 1.19 | 93.34 ± 0.20 | 19.39 ± 0.37 | 25.32 ± 0.40 | 83.59 ± 0.10 |

Raw_image_peak I-AUROC giảm **0.535 pp**, 0/3 seed tăng; I-AUPR giảm 0.595 pp; pixel metrics giữ nguyên. Tiêu chí khai báo trước đòi đủ ba seed, paired mean I-AUROC dương, ≥2 seed tăng, P-AUPR không đổi. Quyết định là **retain_baseline**. Không tiếp tục chọn raw bằng diagnostic VisA +2.03 pp vì đã không vượt source validation. Hai holdout category là giới hạn; chưa chứng minh kết quả cho mọi nhóm hoặc statistical significance.

## 4. Context CLIP lịch sử, tách data access

[Discussion/implementation gốc](../context/discussion.md). Các số sau lấy archive CSV (bảy metric và source paths ở context_results_archived.csv), chưa đối chiếu raw được trong workspace hiện tại. Paper CLIP zero-shot: MVTecAD P-AUPR 42.8, I-AUROC 92.4; VisA 28.0/88.0. Paper CLIP one-shot: MVTecAD 55.7±1.3/96.3±0.7; VisA 32.8±0.4/89.3±1.9. Các published ± giữ nguyên, không giả định paired seed hoặc ddof. Normal-mosaic và transductive không có matched paper protocol; paper memory row không được dùng làm matched mosaic baseline.

### Zero-shot, query-only

mvtec (3 seed, epoch 15):

| Experiment | P-AUPR ± SD | I-AUROC ± SD | Control/giới hạn |
|---|---|---|---|
| baseline | 43.10 ± 0.55 | 91.62 ± 0.33 | anchor |
| self | 42.21 ± 0.89 | 90.40 ± 0.20 | baseline |
| residual_control | 42.74 ± 0.42 | 92.19 ± 0.19 | baseline |
| b_near | 40.57 ± 0.70 | 90.62 ± 0.25 | residual_control |
| c_mid | 37.87 ± 0.21 | 89.90 ± 0.28 | residual_control |
| d_global | 39.88 ± 0.26 | 90.39 ± 0.17 | residual_control |
| e_shared | 40.58 ± 0.22 | 90.70 ± 0.36 | d_global |
| mosaic_a | 43.10 ± 0.55 | 91.62 ± 0.33 | baseline |
| mosaic_query | 37.89 ± 0.42 | 88.61 ± 0.26 | baseline |

visa (3 seed, epoch 15):

| Experiment | P-AUPR ± SD | I-AUROC ± SD | Control/giới hạn |
|---|---|---|---|
| baseline | 28.58 ± 0.26 | 88.08 ± 0.35 | anchor |
| self | 29.57 ± 0.23 | 88.48 ± 0.37 | baseline |
| residual_control | 28.42 ± 0.12 | 87.64 ± 0.22 | baseline |
| b_near | 27.54 ± 0.09 | 87.72 ± 0.17 | residual_control |
| c_mid | 25.46 ± 0.24 | 87.96 ± 0.10 | residual_control |
| d_global | 26.27 ± 0.12 | 87.51 ± 0.47 | residual_control |
| e_shared | 26.45 ± 0.16 | 87.70 ± 0.12 | d_global |
| mosaic_a | 28.58 ± 0.26 | 88.08 ± 0.35 | baseline |
| mosaic_query | 25.16 ± 0.30 | 83.91 ± 0.11 | baseline |

### One-shot normal-reference memory

mvtec (3 seed, epoch 15):

| Experiment | P-AUPR ± SD | I-AUROC ± SD | Control/giới hạn |
|---|---|---|---|
| memory_appearance | 51.80 ± 0.45 | 94.17 ± 0.11 | one-shot anchor; không có paired zero-shot delta |
| memory_context | 53.81 ± 0.52 | 95.00 ± 0.14 | memory_appearance |
| memory_position | 52.65 ± 0.43 | 94.42 ± 0.37 | memory_context |

visa (3 seed, epoch 15):

| Experiment | P-AUPR ± SD | I-AUROC ± SD | Control/giới hạn |
|---|---|---|---|
| memory_appearance | 32.59 ± 0.36 | 91.16 ± 0.23 | one-shot anchor; không có paired zero-shot delta |
| memory_context | 33.70 ± 0.37 | 91.58 ± 0.24 | memory_appearance |
| memory_position | 33.13 ± 0.30 | 91.56 ± 0.27 | memory_context |

### Normal-reference mosaic (actual one-shot access; saved shots=0)

mvtec (3 seed, epoch 15):

| Experiment | P-AUPR ± SD | I-AUROC ± SD | Control/giới hạn |
|---|---|---|---|
| mosaic_normal | 47.75 ± 0.41 | 93.28 ± 0.54 | không có matched-access ordinary baseline |

visa (3 seed, epoch 15):

| Experiment | P-AUPR ± SD | I-AUROC ± SD | Control/giới hạn |
|---|---|---|---|
| mosaic_normal | 30.83 ± 1.06 | 90.39 ± 0.95 | không có matched-access ordinary baseline |

### Transductive target-test companion

mvtec (3 seed, epoch 15):

| Experiment | P-AUPR ± SD | I-AUROC ± SD | Control/giới hạn |
|---|---|---|---|
| mosaic_random | 46.84 ± 0.84 | 91.35 ± 0.48 | không có matched-access ordinary baseline |

visa (3 seed, epoch 15):

| Experiment | P-AUPR ± SD | I-AUROC ± SD | Control/giới hạn |
|---|---|---|---|
| mosaic_random | 31.33 ± 1.12 | 85.87 ± 2.05 | không có matched-access ordinary baseline |

Contextual memory tăng P-AUPR so với appearance-only one-shot khoảng +2.00 pp MVTecAD và +1.11 pp VisA; đây là so sánh cùng shot, checkpoint và reference theo báo cáo lịch sử. Residual context không thắng control tắt mosaic. Self-similarity giúp VisA nhưng hại MVTecAD. Normal mosaic và random mosaic thay đổi cả canvas/resolution/TTA và nguồn ảnh; không quy gain riêng cho “normal context”, không gọi saved shots=0 là zero-shot. Không lấy 53.81 one-shot CLIP trừ baseline DINOv3 zero-shot để mô tả improvement.

## 5. Chi phí và validation kỹ thuật

Head parameters baseline/cluster/components/directional/kNN/residual lần lượt 20,480/27,045/26,405/27,525/26,405/190,085. Spatial 27,845 là architecture count, chưa có benchmark. Chưa có capacity control. KNN exact vẫn O(N²); components có CPU sync; residual projection D→32 lớn. Không có full-run peak VRAM, không kết luận memory/speed win.

[Timing audit cũ](../../runs/feature_ablation/reports/20261106_zero_shot_audit/evaluation_timing.csv) tách loading/inference và metrics; chưa phủ đầy đủ recovery, có log append/reuse nên không coi wall time toàn job là latency độc lập. Smoke DINOv3 batch 4 trên subset mất 14.22s, peak allocated 3562.9 MiB theo review; không ngoại suy thành chi phí benchmark. 210 tests pass là bằng chứng code review, không bằng chứng đạt accuracy paper.

## 6. Hướng tiếp theo

Giữ baseline, không promote raw score. Nếu tiếp tục components, giới hạn giả thuyết segmentation và thêm capacity-matched control trên source holdout trước khi test target. Spatial còn thiếu nếu muốn hoàn tất family. Cần independent Real-IAD benchmark cho protocol mặc định; không suy diễn từ VisA/MVTecAD. Phân tích nguyên nhân gap, lựa chọn chưa khớp paper và ưu tiên kiểm chứng nằm riêng tại [reproduction_gap](../reproduction_gap/README.md).

Nguồn: [provenance](provenance.json), [hướng dẫn artifact](experiment_guide.md), [consolidated evidence](../../runs/feature_ablation/reports/20261106_consolidated/discussion.md). Phiên này chỉ build báo cáo offline, không train/evaluate mới.
