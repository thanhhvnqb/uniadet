# Code and paper review — 2026-10-31

Implementation follow-up: both confirmed issues below are now fixed. Optional
feature/loss/initialization flags, shared evaluation restoration, and auxiliary
category holdouts are implemented. Follow [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md)
for current commands. The remainder records the pre-fix review and measurements.

The implementation is close to the published method. The existing CLIP results
have a remaining image classification gap; they do not show a general collapse
in segmentation. No single demonstrated defect explains the whole gap.

Reviewed `uniadet/`, the training/evaluation entrypoints, dataset preparation,
saved configurations/checkpoints/results, and the paper and supplement.
This is a review: model code, experiment settings, and checkpoints were not changed.

## Compare the same experiment

The local CLIP checkpoint is
`runs/uniadet-clip/train_VisA/epoch_15.pt`; its embedded configuration confirms
VisA test supervision, 518-pixel input, five layers, CAA, seed 10, batch size 8,
and learning rate 0.001. Rotation and both flips are disabled. The run trained
for 20 epochs, but evaluating its saved epoch 15 preserves the reported epoch.

| MVTec, CLIP, zero-shot (%) | Local epoch 15 | Paper | Local minus paper (pp) |
|---|---:|---:|---:|
| Image AUROC | 90.2328 | 92.4 | -2.1672 |
| Image AUPR | 94.6692 | 96.1 | -1.4308 |
| Pixel AUROC | 90.8324 | 91.8 | -0.9676 |
| Pixel AUPR | 42.9513 | 42.8 | +0.1513 |

Local source: `runs/uniadet-clip/MVTecAD_0shot/epoch_15/metrics.json`.
Paper source: [Tables 2 and 12](https://arxiv.org/pdf/2601.05552).
The EVA02-CLIP experiment has image AUROC 90.2693% and pixel AUPR 41.2128%
at epoch 15. EVA02-CLIP is an additional backbone experiment, not a paper column.

The largest image AUROC deficits are toothbrush (70.8333 versus 90.6), cable
(69.3028 versus 76.7), and bottle (88.5714 versus 93.7). Toothbrush alone
contributes about 1.32 percentage points to the macro gap. Its pixel AUPR is
28.9144%, above the reported 26.9%; this points toward auditing image scoring
and global heads before redesigning segmentation. These are diagnostic
comparisons, not a basis for selecting category-specific settings.

## Confirmed code findings

1. **[P2] `uniadet/data.py:74`: mosaic resizing can erase the only anomaly.**
   `__getitem__` still returns the original positive image label at line 113.
   With candidate `candle/Data/Images/Anomaly/058.JPG`, grid 3, the candidate
   in the first tile, and eight same-class normal tiles, the actual dataset
   returns `label=1`, `has_mask=True`, and zero positive mask pixels. Its
   original mask contains 37 positive pixels. The existing label-preservation
   test uses a larger synthetic defect and misses this case.

   Reject an anomalous augmentation if its final resized mask is empty; retry
   or fall back to the unaugmented candidate. Keep the original image label.
   A scan of all 1,200 anomalous VisA masks found one empty mask at 172x172
   and four at 173x173, versus none at 518x518 or 259x259. These are counts
   for individual tile sizes, not an observed frequency of invalid full
   mosaics; other anomalous tiles can keep a mosaic positive. This is a real
   supervision defect, but its measured scope is small.

2. **[P2] `uniadet/cli.py:260`: standalone evaluation does not restore saved
   fusion settings.**
   Model and temperature settings restore from the checkpoint, but
   `lambda_p`, `lambda_f`, and `zero_score_pooling` retain parser defaults.
   A checkpoint configured for nondefault fusion can therefore produce a
   different prediction rule when evaluated through `test.py` without flags.
   Restore saved values unless the user explicitly overrides them; use an
   unset parser default to distinguish an override. The reviewed CLIP run uses
   the current defaults, so this does not explain its 2.17-point gap.

## Reproduction uncertainties to test, not declare bugs

- **Image-score aggregation — `uniadet/model.py:87-99`.** The implementation
  averages per-layer probabilities and, by default, averages their spatial
  maxima. Taking the maximum after averaging maps gives another result.
  Section 3.5 uses layer-indexed predictions without explicitly giving the
  final layer reduction. Keep both definitions recorded and avoid describing
  one as verified official behavior.
- **Frozen feature extraction — `uniadet/backbones.py:31,38`.** The wrapper
  uses bicubic positional interpolation and applies final layer normalization
  and projection at every selected layer. The inherited AdaptCLIP path uses
  bilinear positional interpolation (`lib/adaptclip.py:420`) and omits final
  layer normalization from its separately returned global token (line 445).
  Neither difference proves what UniADet used. These choices can affect the
  classification branch and deserve controlled auxiliary-data ablations.
  Retrain heads for any changed feature representation.
- **Training defaults — `train_test.py:25-26` versus `uniadet/cli.py:50-55`.**
  The dev runner disables rotation/flips; standalone training enables them.
  Thus the two entrypoints produce different experiments even with the same
  seed. The paper's CAA description does not establish a required setting for
  these additional transforms. Make the policy explicit when comparing runs.
- **Temperature and optimization — `uniadet/model.py:8,13-14`,
  `uniadet/cli.py:88`.** Temperature 0.07, random initialization, Adam betas,
  and the mosaic/crop mixture are reproduction choices. Published details do
  not settle every value. Test them on held-out auxiliary categories before
  fixing a configuration for target evaluation.

Changing the overall CE-versus-segmentation weight is not a strong first
experiment: the frozen encoder and disjoint heads prevent classification and
segmentation gradients from conflicting. Uniform loss scaling is also largely
cancelled by Adam, apart from epsilon/numerical effects. Changing relative
focal-versus-Dice weights can change segmentation; the current configuration
does not expose those weights separately. The focal implementation differs
slightly from AnomalyCLIP's smoothed form, while both Dice terms follow its
structure. See the [AnomalyCLIP training code](https://github.com/zqhang/AnomalyCLIP/blob/main/train.py)
and [loss implementation](https://github.com/zqhang/AnomalyCLIP/blob/main/loss.py).

## Checks that already match

The frozen encoder, independent task/layer matrices, cosine logits, CE plus
focal and two Dice terms, and exact cosine reference retrieval implement the
stated structure. Reference selection uses normal target-train images.
The auxiliary and target metadata counts match Table 9: VisA test has 962
normal/1,200 anomalous images; MVTec test has 467/1,258. The auxiliary test
split is intentional; switching to its all-normal train split would remove
the supervision needed here. Layers 12/15/18/21/24, resolution, CAA probability
and grids, learning rate, epoch 15, and default fusion weights match the
[implementation details](https://arxiv.org/pdf/2601.05552#page=11).

There is no evidence that adding DPAM/adapters, sharing the task heads, or
shrinking the parameter count is required. The current author README describes
0.015M/0.02M trainable parameters, consistent with the explicit matrices here;
the smaller abstract claim is not a reason to change the architecture.
The [official repository](https://github.com/gaobb/UniADet) still contains
README/assets rather than the training implementation. Its linked Hugging Face
Space also exposes only metadata/README, so an exact implementation comparison
was unavailable.

## Validation and next experiment

- Fresh inference completed on all 15 MVTec categories using the real CLIP
  backbone and epoch-15 head. The current fusion reproduced every saved
  per-category image AUROC, AUPR, and F1max exactly.
- Existing regression suite: **66 passed** on PyTorch 2.7.1+cu126; CUDA available.
- The augmentation finding was reproduced through `AnomalyDataset.__getitem__`,
  not just a hand-written resizing example.
- A mocked evaluation checkpoint with saved `lambda_p=0.2`, `lambda_f=0.8`,
  and pooling `map` reached model construction with `0.5`, `0.5`, and `layer`,
  respectively, confirming the standalone configuration-restoration finding.
- Image AP computed from the saved predictions is 94.7499%, versus
  trapezoidal AUPR 94.6692%. This 0.0807-point definition difference cannot
  explain the AUROC gap. Keep the metric definitions explicit; no evidence
  establishes that changing AUPR to AP reproduces the authors' evaluator.

| Fresh diagnostic, same checkpoint (%) | Image AUROC | Image AUPR |
|---|---:|---:|
| Mean global-head probabilities | 89.6345 | 94.8815 |
| Maximum of mean segmentation map | 83.7620 | 90.3465 |
| Legacy fusion: maximum after layer averaging | 88.1756 | 93.2779 |
| Current fusion: maximum before layer averaging | 90.2328 | 94.6692 |

The existing pooling interpretation recovers 2.0572 AUROC points over the
legacy rule, but is already included in the saved result. Global heads also
generalize very differently: layer 12 alone has 75.0286% macro image AUROC,
versus 91.9108% for layer 21. This motivates investigating their representation
and optimization on auxiliary validation data. It does not authorize picking
layer 21 from target-test performance. On toothbrush, global-only image AUROC
is 78.3333%, while current fused AUROC is 70.8333%; high pixel AUPR therefore
does not establish that map maxima are reliable image-level evidence.

First repair augmentation label preservation. Then use held-out auxiliary
categories to compare the image-score reductions and global-token extraction,
followed by temperature/initialization. Freeze choices before target evaluation;
report epoch 15 and repeat seeds to measure variability. Preserve this checkpoint
as the baseline. The current 20-epoch summaries do not demonstrate an improvement
from extending training, and choosing epochs using target labels would bias the
comparison.

Fresh image-score audit results are recorded separately under
`runs/review_20261031/clip_epoch15/score_audit.json`; they do not change
checkpoint parameters. Reproduce with:

```bash
.venv/bin/python scripts/audit_zero_shot.py \
  --data datasets/MVTecAD \
  --checkpoint runs/uniadet-clip/train_VisA/epoch_15.pt \
  --shots 0 --device cuda --workers 4 --batch-size 8 \
  --output runs/review_20261031/clip_epoch15
```
