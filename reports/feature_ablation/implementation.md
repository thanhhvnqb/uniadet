# Chi tiết implementation

Tài liệu phân biệt code hiện tại với cấu hình đã thực sự chạy. Khi tái lập, snapshot/config/checkpoint của job là nguồn xác định hành vi; default trong code hôm nay không thay thế giá trị lịch sử.

## 1. Baseline và luồng dữ liệu

Nguồn code: [backbones.py](../../uniadet/backbones.py), [model.py](../../uniadet/model.py), [loss.py](../../uniadet/loss.py), [data.py](../../uniadet/data.py), [settings.py](../../uniadet/settings.py), [workflow.py](../../uniadet/workflow.py), [cli.py](../../uniadet/cli.py).

```text
Ảnh query → encoder frozen → (CLS_l, patch_l, grid_l), l ∈ {12,15,18,21,24}
                            ├─ CLS → image cosine head → probability image mỗi layer
                            └─ patch → [descriptor adapter tùy chọn] → pixel cosine head
                                                                ↓ resize, probabilities
                            mean layer maps → Gaussian → returned anomaly map
                            mean layer peaks + mean image probabilities → image score
```

Encoder giữ `requires_grad_(False)`, `eval()`, trích xuất trong `no_grad`; optimizer chỉ cập nhật head. DINOv3 ViT-L/16 dùng intermediate tokens đã normalize, bỏ storage tokens, kích thước 512, patch grid 32×32 và feature D=1024. Hai ma trận 2×D độc lập cho classification/segmentation ở mỗi layer: tổng baseline `5×2×2×1024 = 20,480` tham số. Context CLIP là một family khác, không dùng baseline DINOv3 làm control.

Với unit feature u và ma trận W có từng hàng normalize:

`logits = u @ normalize(W).T / temperature`, hai class normal/anomalous.

Global CLS và patch có W riêng; W mỗi layer riêng. Khởi tạo Gaussian std `D**(-0.5)` khi `head_init_std=None`; temperature của family đã chạy là 0.07. Hai image/pixel W gốc được lấy mẫu trước khi tạo adapter để giữ khởi tạo gốc giống nhau theo seed. Không fine-tune encoder.

Family đã chạy dùng `softmax_before_resize=True`: tính two-class softmax trên patch grid rồi bilinear resize (`align_corners=False`) về kích thước ảnh. Code trả `log(probability)` để giữ interface loss; softmax ở consumer khôi phục xác suất đã resize. Đây khác với resize logits rồi softmax; đổi lựa chọn này làm thay đổi training, cần retrain.

## 2. Sáu descriptor độc lập

Nguồn: [patch_features.py](../../uniadet/patch_features.py). `patch_descriptors` detach feature, chuyển float, normalize từng patch thành u. Descriptor chỉ dùng **một query**, từng layer; không labels/masks, target references hay ảnh test khác. Các mode chạy riêng, chưa ghép descriptor groups.

| Mode | Công thức/đặc trưng tại patch i | Input dim | Xử lý biên/trường hợp rỗng |
|---|---|---:|---|
| `none` | Dùng patch gốc, không adapter | D | Formula, parameter count và state dict baseline giữ nguyên |
| `spatial` | Với ba vành Chebyshev `(0,1]`, `(1,3]`, `(3,7]`: mean/std/min của `1-u_i·u_j`; cộng mean vành đầu trừ mean vành cuối | 10 | Chỉ đếm neighbor thật; vành rỗng trả 0; std là population std |
| `knn` | Mean 5 cosine distance nhỏ nhất tới patch khác trong cùng ảnh | 1 | Loại chính nó; dùng min(k,N−1); N=1 trả 0; không loại neighborhood gần |
| `cluster` | Own-centroid distance; own/RMS; cluster rarity; centroid separation; competing margin | 5 | K=min(8,N), 10 iteration; nhóm rỗng không làm competitor |
| `components` | `log(N / area(component_i))` từ nhãn clustering, liên thông 4 hướng | 1 | Nhãn ra CPU, SciPy connected components; chỉ area vào adapter |
| `directional` | `1-u_i·u_neighbor` theo phải/trái/xuống/lên và bốn đường chéo | 8 | Hướng ngoài ảnh điền 0; không mask riêng trong classifier |
| `residual` | `u_i − mean(u_j)` với tám neighbor hợp lệ trong cửa sổ 3×3, loại tâm | D=1024 | Biên chỉ lấy neighbor có thật; không neighbor trả vector 0 |

Spatial dùng pooling tổng/count cho mean và duyệt offset cho squares/min, tránh tensor window×D lớn. Exact kNN chunk query 256 patch, working distance tensor khoảng `B×chunk×N`; tính toán vẫn bậc hai N. Với N=1024, chunk giảm bộ nhớ trung gian, không phải approximate retrieval.

Clustering là spherical Lloyd, farthest-first khởi đầu `u[0]`, cập nhật centroid bằng normalize vector tổng. Cluster rỗng hoặc tổng gần zero giữ centroid cũ. Duplicate centers được phép; label dùng argmin. Không đưa ID cluster như một con số classifier.

Năm descriptor cluster:

- `own = clamp(1 − u_i·center[label_i], 0, 2)`.
- `normalized = own / max(RMS_own_cluster, 0.05)`; RMS không phải std.
- `rarity = −log(count_cluster/N)`.
- `separation = min cosine_distance(center_own, occupied_other_centers)`.
- `margin = min cosine_distance(u_i, occupied_other_centers) − own`.

Nếu chỉ một occupied group: separation/margin bằng 0. Components dùng cùng clustering để lấy label nhưng **không nối năm descriptor cluster**. Residual DINOv3 ở đây khác residual nhiều nhánh near/mid/global của context CLIP.

## 3. Adapter, normalization và capacity

Mỗi layer có adapter riêng:

`descriptor → RunningStandardize → Linear(d,32) → GELU → Linear(32,32) → alpha×extra`.

Nối `[normalize(original_patch), alpha×extra]`, rồi normalize toàn vector để chấm cosine với W_seg mở rộng `(2,D+32)`. Image head không đổi cấu trúc, nhưng segmentation loss và image loss được tối ưu cùng quá trình train. Alpha khởi tạo 0.1 và là scalar học được, không bị clamp dương sau optimizer step. Bias Linear khởi tạo 0; weight Linear dùng initializer PyTorch hiện hành, không ép cùng initializer W gốc.

RunningStandardize tích lũy count/mean/population variance của patch **source training**, mỗi patch có trọng số như nhau. Trong train, merge moment của batch rồi normalize batch bằng moment mới; eval không update. Chia bằng `sqrt(max(variance,1e-4))`. Buffers được lưu cùng checkpoint. Đây là running normalization, không phải fit một lần trên target, và không phải GroupNorm/BatchNorm mặc định.

| Mode | Head parameters | Tăng so với baseline |
|---|---:|---:|
| none | 20,480 | 0 |
| spatial (chưa chạy) | 27,845 | 7,365 |
| knn/components | 26,405 | 5,925 |
| cluster | 27,045 | 6,565 |
| directional | 27,525 | 7,045 |
| residual | 190,085 | 169,605 |

Adapter không phải backbone adapter. Chưa có capacity-matched random/constant-descriptor control nên không thể tách lợi ích descriptor khỏi thêm capacity. Mỗi representation mới retrain head; không dùng adapter mới với baseline checkpoint.

## 4. Loss và augmentation đã chạy

Tổng loss là tổng theo layer của CE classification, focal pixel và Dice của cả normal/anomaly. Focal `−(1−p_t)^gamma log(p_t)`, gamma=2, smoothing=0; Dice từng ảnh/class có smooth=1. Các weight classification/segmentation/focal/normal-Dice/anomaly-Dice đều 1. Feature family giữ nguyên số loss branch; không thêm supervision riêng trên descriptor. Tổng loss theo layer, không chia trung bình layer.

VisA→MVTecAD và MVTecAD→VisA dùng auxiliary **test split có nhãn**; không dùng normal-only train split để train head. Train batch 8, Adam lr=0.001, betas=(0.5,0.999), weight_decay=0, 15 epoch; test batch 16. CAA p=0.5, grid 2/3, mosaic p=0.5; rotation/flips tắt. `preserve_anomaly_mask=True`: reject/fallback khi crop/resize/mosaic làm mất anomaly mask của mẫu positive. Source normalization gồm feature của dữ liệu augmented trong quá trình train. Target masks chỉ dùng tính metric.

## 5. Inference và diagnostic image score

Mỗi layer cho map anomaly A_l và global probability g_l. Baseline zero-shot `zero_score_pooling=layer`, map sigma=4:

`returned_map = Gaussian_4(mean_l A_l)`

`image_score = 0.5 mean_l(g_l) + 0.5 mean_l(max_xy Gaussian_4(A_l))`.

`mean(max(A_l))` khác `max(mean(A_l))`. `image_score_gaussian_sigma=None` kế thừa map sigma. `raw_image_peak` chỉ đặt image-score sigma=0, giữ returned map sigma=4; head và pixel metric không đổi. `global_only`, `map_only`, `legacy_map_fusion`, `global_layer_*` là luật audit của [audit_zero_shot.py](../../scripts/audit_zero_shot.py), không phải các head mới hoặc baseline mới.

Few-shot NormalMemory hiện tại lưu normalized backbone patch trên CPU theo category/view, exact cosine NN bằng query/bank chunk, đưa từng chunk lên GPU; map distance average layer, fuse với baseline bằng lambda_f=0.5. Feature-family zero-shot không dùng branch này. Context CLIP memory dùng descriptor riêng như tài liệu lịch sử, không được suy ra từ NormalMemory hiện tại.

## 6. Serialization, orchestration và provenance

Feature settings được validate (`mode` hợp lệ, integer dương, alpha khởi tạo hữu hạn/dương), serialize vào config/checkpoint và restore qua settings. Representation mismatch khi eval bị từ chối. Checkpoint cũ thiếu feature fields dùng `none` cùng historical defaults; setting eval `None` kế thừa checkpoint. Explicit inference overrides ghi riêng trong `evaluation_config.json`. Model state lưu projection, alpha và normalization buffers.

Runner tạo `workflow.RunConfig` riêng, resolve bằng `workflow.training_args`, seed qua `cli.seed_all`, train qua `cli.train`, evaluate qua `workflow.test_checkpoints`. `--start-epoch` chỉ lọc checkpoint test, không resume optimizer. Training từ chối thư mục output không rỗng. Bash launcher dùng GPUs vật lý 0/2/3 cho seeds 10/20/30 khi đủ tài nguyên, mỗi process `CUDA_VISIBLE_DEVICES` riêng và logical `cuda:0`, chờ exit từng PID; worker chỉ ghi seed artifacts, coordinator aggregate.

Snapshot của đợt thực chạy: [plan](../../runs/feature_ablation/launches/dinov3-features-visa-mvtec-fix/plan.json). Weights HF revision `ea8dc2863c51be0a264bab82070e3e8836b02d51`; code/dirty diff/environment theo launch/audit manifests. New checkpoints lưu pretrained identity và verify khi eval; metadata mới không tự chứng minh provenance nội tại của checkpoint cũ.

## 7. Context CLIP lịch sử: implementation khác biệt

[Giải thích đầy đủ](../context/context_experiment_guide.md), [kết quả](../context/discussion.md). Dưới đây là mô tả archive, không xác nhận code cũ còn chạy được trong checkout này.

| Nhóm | Implement và matched control |
|---|---|
| residual_control | Baseline tắt mosaic stitching, giữ crop CAA; control cho residual |
| b_near | Residual `(x−mean_near(x))/||x||`, vùng 5×5 bỏ tâm, layers 12/15 |
| c_mid | Near và vành (2,6] trên mọi layer |
| d_global | Thêm mean toàn grid, gồm cả patch hiện tại; W riêng từng range/layer |
| e_shared | Như d_global nhưng share residual head giữa ranges trong layer |
| self | Exact top-5 similarity trong query, loại neighborhood radius 3; chuẩn hóa map từng ảnh; fuse 20% self + 80% baseline; đổi image pooling theo fused map |
| memory_appearance | One-shot target normal per category/view; nearest appearance key |
| memory_context | Joint key normalize `[unit appearance, sqrt(0.5) unit near, sqrt(0.5) unit mid]`; một NN chung, fuse memory 50% |
| memory_position | Cùng contextual key, chỉ match cửa sổ ±8 patch theo vị trí |
| mosaic_normal | Canvas 518, 2×2, query một tile 259; một target normal lặp ba tile, four-corner TTA |
| mosaic_random | Companion từ target test chưa xác minh normal: transductive, không gọi zero-shot inductive |
| mosaic_query | Bốn tile cùng query, không reference ngoài; vẫn có resolution/TTA confound |

Residual context huấn luyện baseline và residual cùng từ đầu, CE classification và focal/Dice nhiều branch/fusion; không tương đương descriptor MLP DINOv3. Normal mosaic có target normal access dù saved `shots=0`; xếp reference-assisted one-shot theo actual access. `mosaic_a` là ordinary baseline inference, không một training mới. Raw artifacts và scripts `run_context.py` không còn trong workspace; dùng tài liệu archive để đọc kết quả, không đưa lệnh cũ như lệnh tái lập hiện hành đã kiểm chứng.
