# Paper/code review

Reviewed arXiv:2601.05552v1 (2026-01-09), especially Sections 3.3–3.5,
Table 2, and Appendix Sections 6–7. Source: https://arxiv.org/pdf/2601.05552.
The authors' public repository at commit
`f49d2edd5340276b9ece39fbe8da4df26f10db4c` contained README/assets only when
checked; an exact comparison with their training/inference implementation was
therefore unavailable. Source: https://github.com/gaobb/UniADet.

## Correct comparison

The supplied `runs/visa/epoch_15.pt` is CLIP, not DINO. Its configuration uses
VisA test supervision, blocks 12/15/18/21/24, resolution 518, CAA, 15 epochs,
learning rate 0.001, batch size 8, and seed 10. These documented paper settings
match the code. VisA metadata contains 962 normal and 1,200 anomalous test
images, matching Table 9. Training on this auxiliary test split is intentional.

Table 2's MVTec **CLIP zero-shot** reference is image AUROC 92.4%, image AUPR
96.1%, pixel AUROC 91.8%, pixel AUPR 42.8%. DINOv2/DINOv3 results require their
own pretrained encoders and separately trained heads.

The existing saved evaluation reports image AUROC 88.1756%, image AUPR
93.2779%, pixel AUROC 90.8324%, pixel AUPR 42.9513%. The pixel AUPR already
matches the paper closely; the main discrepancy is image scoring.

## Changes

1. Zero-shot pooling: the old implementation fused the mean global score with
   `max(mean(layer_maps))`. Section 3.5 describes fusing the layer-indexed global
   score and the maximum of the layer-indexed map. The revised interpretation
   uses `mean(max(layer_maps))` before image-score fusion. These operations differ
   whenever different levels predict peaks at different positions. Pixel maps,
   trained parameters, preprocessing, and few-shot fusion are unchanged.
   `--zero-score-pooling map` reproduces the old behavior. The paper does not
   explicitly state the reduction order; official code is needed to settle it.
2. Evaluation previously discarded checkpoint `weights`/`dino_repo` settings,
   potentially loading a different frozen encoder. These now restore unless
   explicitly overridden. This does not affect the supplied default-CLIP run.
3. Effective evaluation configuration and checkpoint epoch are now saved to
   `evaluation_config.json`.
4. `scripts/audit_zero_shot.py` compares fixed score formulas and individual
   global heads, writing `score_audit.json`. It does not select a configuration.

## Validation and limits

The image-score audit completed on all 15 MVTec categories with the supplied
checkpoint:

| Image metric (%) | Saved/legacy fusion | Revised layer fusion | Paper CLIP |
|---|---:|---:|---:|
| AUROC | 88.1756 | 90.2328 | 92.4 |
| AUPR | 93.2779 | 94.6692 | 96.1 |
| F1max | 90.4865 | 91.4104 | — |

The change recovers **2.0572 percentage points** of image AUROC without
retraining. The remaining AUROC gap is 2.1672 points. Legacy audit scores match
the original saved metrics across all categories. Individual heads vary strongly
in quality; that is diagnostic evidence, not permission to choose the best head
using target-test labels. Complete diagnostic results are in
`runs/mvtec_zero_review/score_audit.json`.

The full zero-shot evaluation also completed successfully. All 15 categories'
pixel metrics are exactly equal to the saved baseline, and all revised image
metrics exactly match the independent image-score audit. Full outputs are in
`runs/mvtec_zero_review/metrics.json`, `metrics.csv`, and `predictions.json`.
Pixel macro AUROC remains 90.8324%, and pixel macro AUPR remains 42.9513%.
The regression suite passes all **19 tests**.

Regression tests exercise spatially disagreeing layer peaks, legacy scoring,
few-shot fusion, and pretrained source restoration/overrides, alongside the
existing pipeline tests. No target-test labels were used to tune fusion weights,
choose layers, or select checkpoints. The epoch-15 checkpoint remains unchanged.

The paper leaves temperature, initialization, optimizer details, feature
postprocessing, and some multi-layer reductions underspecified. This independent
implementation cannot promise exact reproduction without official code/weights.
Changing the encoder representation or training objective would require retraining
the head; it would be invalid to silently apply that change to this checkpoint.

Re-run the revised evaluation into a separate directory:

```bash
.venv/bin/python test.py --data datasets/MVTecAD --checkpoint runs/visa/epoch_15.pt --shots 0 --output runs/mvtec_zero_review
.venv/bin/python scripts/audit_zero_shot.py --data datasets/MVTecAD --checkpoint runs/visa/epoch_15.pt --shots 0 --output runs/mvtec_zero_review
```

## Independent reproduction blog

Source: https://zhuanlan.zhihu.com/p/1999979404848284065
(retrieved through the public Jina Reader endpoint because direct Zhihu requests
returned HTTP 403). Title: UniADet零样本异常检测复现（魔改）之路.

The article's MVTec experiment uses **DINOv3 ViT-L/16**, rather than our CLIP
checkpoint. It reports the following first/second training runs:

| Metric (%) | First | Second |
|---|---:|---:|
| Image AUROC | 92.1 | 94.2 |
| Image AP | 95.9 | 97.5 |
| Pixel AUROC | 92.8 | 92.3 |
| Pixel AP | 53.4 | 54.4 |

The writer attributes the second run to changing loss proportions and
postprocessing, without publishing their values, processing algorithm, or code.
The article also identifies temperature as an unoptimized hyperparameter in the
first run. It provides no evidence for a particular temperature, Gaussian sigma,
pooling order, or focal/Dice weighting. Its AP implementation is unspecified;
our AUPR uses trapezoidal PR-curve area, so exact equivalence is unverified.

Useful implications for this implementation (our analysis, not blog claims):

- Audit focal-versus-Dice weighting and inference postprocessing explicitly.
  `training_loss` currently sums CE, focal, normal Dice, and anomaly Dice at each
  level; there are no independent loss-weight controls. Pixel maps are not
  smoothed. These are experimental choices, not confirmed bugs.
- Distinguish relative segmentation terms from overall CE-versus-segmentation
  scaling. With a frozen encoder and disjoint task parameters, scaling the whole
  segmentation loss does not change classification-head gradients. Adam also
  largely cancels uniform gradient scaling, apart from epsilon/numerics. Changing
  focal-versus-Dice weighting can change the segmentation decision boundary and
  hence the map contribution to final image scores.
- The reported second run improves image detection and pixel AP while reducing
  pixel AUROC: assess all metrics rather than assuming every change helps both
  tasks. Keep the existing checkpoint/configuration as the baseline.
- Evaluate any proposed settings on held-out auxiliary categories first, then
  freeze them before target evaluation. Temperature/loss changes during training
  require a new head checkpoint; do not silently alter this checkpoint's
  temperature at evaluation.
- A DINOv3 reproduction needs separately trained DINOv3 heads and the appropriate
  encoder weights. The blog's numbers do not establish a CLIP reproduction fix.

No numerical defaults were changed based on this article because the necessary
settings are not disclosed. Unrelated repositories found by search were not
attributed to this blogger; the article supplied no repository link.
