# UniADet — independent PyTorch reimplementation

Implementation of [One Language-Free Foundation Model Is Enough for Universal Vision Anomaly Detection](https://arxiv.org/abs/2601.05552) (arXiv v1, January 2026), based on [AdaptCLIP](https://github.com/gaobb/AdaptCLIP). This is a paper reimplementation, not the authors’ official UniADet release. Benchmark numbers have **not** been reproduced.

## Install

Python 3.10+ (DINOv3 may require newer Python/PyTorch; follow its upstream environment requirements).

```bash
python -m venv .venv
source .venv/bin/activate
# Install the torch/torchvision build appropriate for your CPU or CUDA first.
pip install -r requirements.txt
pip install pytest
python -m pytest -q
```

CLIP and DINOv2 download pretrained weights on first use. DINOv3 requires separately obtained official weights and accepts a local official repository checkout via `--dino-repo`. Backbone code/weights retain their respective upstream licenses and access conditions.

## Data and training protocol

Use AdaptCLIP’s `meta.json` schema, with paths relative to each dataset root:

```json
{
  "train": {
    "bottle": [{"img_path": "bottle/train/good/000.png", "mask_path": "", "anomaly": 0}]
  },
  "test": {
    "bottle": [{"img_path": "bottle/test/broken/001.png", "mask_path": "bottle/ground_truth/broken/001_mask.png", "anomaly": 1}]
  }
}
```

Category names come from dictionary keys. `view_id` is optional, defaults to `"0"`, and separates multi-view reference banks. Dataset generators from AdaptCLIP are retained in `dataset/`; edit their root paths before running them. Images use PIL-supported formats. Nonzero mask values indicate anomaly. Missing anomalous masks are allowed for classification-only evaluation and excluded from pixel metrics; training requires masks.

The paper trains on the **auxiliary dataset’s test split**, which contains both normal and annotated anomaly images. Train on VisA, evaluate on MVTec and other domains; train a separate model on MVTec when evaluating VisA. The default `--split test` is intentional. Zero-shot means no target-domain training, not no auxiliary supervision.

```bash
python train.py --data datasets/visa --backbone clip --output runs/visa
python test.py --data datasets/mvtec --checkpoint runs/visa/epoch_15.pt --shots 0 --output runs/mvtec_zero
python test.py --data datasets/mvtec --checkpoint runs/visa/epoch_15.pt --shots 4 --output runs/mvtec_four

python train.py --data datasets/visa --backbone dinov2 --output runs/visa_dinov2
python train.py --data datasets/visa --backbone dinov3 --weights /weights/dinov3_vitl16.pth --dino-repo /repos/dinov3 --output runs/visa_dinov3
```

Evaluation restores `--weights` and `--dino-repo` from the checkpoint unless explicitly overridden; those paths must still be accessible. Optional local CLIP or DINOv2 weights can also be passed via `--weights`. CLIP weights must use AdaptCLIP’s CLIP checkpoint layout, DINOv2 weights a plain official model state dictionary.

Few-shot references come exclusively from normal images in the **target train split**, sampled without replacement per category/view. No target-test fallback is used. `references.json` records the selected samples; `--seed` controls selection. Use distinct output directories for different shot counts and seeds. Checkpoints store the learned head, optimizer, epoch, and configuration; pretrained encoder weights are loaded separately. Evaluation restores the training backbone, layers, image size, and temperature from checkpoint configuration.

## Method mapping

| Paper | Implementation |
|---|---|
| Eq. 8: independent cosine classifiers per task and layer | `uniadet/model.py`: `DecoupledHead` |
| Eqs. 9–10: normal patch memory, exact cosine nearest neighbor, layer mean | `NormalMemory`, with chunked query/reference products |
| Eqs. 11–12: map fusion then image score fusion | `UniADet.predict` |
| Sec. 3.5: per-layer CE, focal, normal/anomaly Dice | `uniadet/loss.py` |
| Appendix 6: class-aware grid mosaic and grid cropping | `uniadet/data.py` |
| CLIP, DINOv2 with registers, DINOv3 frozen features | `uniadet/backbones.py` |

Default blocks are **12, 15, 18, 21, 24** (one-based). Images resize to 518 for CLIP/DINOv2 and 512 for DINOv3. Only the head is optimized, for 15 epochs at learning rate 0.001. Both fusion coefficients default to 0.5. CAA runs with probability 0.5; grids 2×2 and 3×3 are equally likely. Normal mosaics contain only normal images of the same class; anomalous mosaics retain the candidate and sample any same-class images. Anomalous crops must intersect the mask.

Implementation choices where the paper is underspecified:

- Temperature 0.07, Gaussian weight initialization with standard deviation `d**-0.5`, Adam betas `(0.5, 0.999)` inherited from AdaptCLIP, no weight decay or schedule. All are explicit in code; temperature and learning rate have CLI options.
- Mosaic versus cropping is sampled equally. Cropping uses a uniform grid; the appendix does not specify random boundary distributions. Mosaic candidate position is randomized.
- Per-layer losses are summed. Inference averages per-layer probabilities and upsamples logits bilinearly before softmax. Zero-shot image scoring averages the per-layer spatial maxima before fusion with global scores, following the layer-indexed description in Sec. 3.5. `--zero-score-pooling map` restores the earlier implementation (maximum after averaging maps). The paper does not explicitly specify the layer reduction order; this is an interpretation, not confirmed by official code. Few-shot image scoring uses the maximum of the **fused** map, following Eqs. 11–12. No Gaussian postprocessing is applied.
- CLIP uses unmodified frozen attention blocks, with final layer normalization and visual projection applied to each selected layer. AdaptCLIP’s DPAM replacement and all adapters are disabled. Positional embeddings interpolate without mutating pretrained parameters. DINO uses the official normalized intermediate class/patch tokens, excluding register/storage tokens.
- Explicit two-class matrices for both tasks at five layers contain `4 * sum(layer_dims)` parameters: 15,360 for projected CLIP-L (768 dimensions) and 20,480 for DINO-L (1024). This does not match the paper abstract’s 0.002M parameter claim; no undocumented bottleneck or weight sharing is introduced to force that number.

## Evaluation and validation

Evaluation batches images using `--batch-size` (default 8); reduce it if GPU memory is limited. Few-shot batches share a category/view reference bank. CUDA evaluation uses pinned host memory and nonblocking image transfers.

`evaluation_config.json` records the effective evaluation settings, training settings,
and checkpoint epoch. To diagnose image-score discrepancies without running expensive
pixel metrics, use `python scripts/audit_zero_shot.py` with the same arguments as
`test.py`. It saves `score_audit.json`, comparing global-only scores, map-only scores,
both fixed fusion definitions, and individual global heads. These are diagnostic
comparisons; selecting weights or layers using target-test labels would invalidate
the zero-shot protocol.

Evaluation writes `metrics.json` with per-category and macro image/pixel AUROC, PR-curve area (AUPR), and F1max, plus pixel AUPRO (8-connected regions, normalized over FPR 0–0.3). `metrics.json` includes an explicit `average` entry (also available as `macro_mean`), and `metrics.csv` ends with an `Average` row. These are unweighted category averages for each metric, excluding undefined values. `metrics.csv` stores the seven paper columns as percentages; JSON values remain in [0, 1]. Evaluation also saves `predictions.json` with image scores. Undefined metrics (only one ground-truth class) are `null` and excluded from macro means. Medical segmentation-only datasets have undefined image metrics. Visualization exports are not implemented in the new evaluator. It stores dense predictions for one category at a time; very large categories can still need substantial host RAM. Memory banks retain all reference patches; only nearest-neighbor working memory is chunked.

Current review validation: Python 3.12.3, PyTorch 2.7.1+cu126 and torchvision 0.22.1+cu126: **19 tests passed**. See [REPRODUCTION_REVIEW.md](REPRODUCTION_REVIEW.md) for the paper/code comparison and benchmark commands.

Tests cover cosine scoring, task/layer gradient separation, frozen encoder behavior, exact chunked retrieval, class/view bank isolation, fusion arithmetic, checkpoint roundtrip, CAA label preservation, reference split isolation, encoder token contracts, and synthetic training/evaluation. They require no pretrained downloads. These checks establish implementation behavior, not benchmark accuracy. Real backbone integrations and full dataset results require separate validation with pretrained weights.

## Provenance and license

AdaptCLIP files in `lib/`, `dataset/`, `metrics/`, and `tools/` originate from commit `354d9e3332ec5348b3d8e4439111d34f8e94c0a9`. Two deprecated `pkg_resources` imports were replaced with `packaging.version`; the local state-dict loader now rewinds the file after a failed JIT probe and uses `weights_only=True`. The retained `metrics/` and `tools/` are legacy utilities and are not imported by the new pipeline; their optional dependencies are not included in the minimal requirements. The upstream GPL-2.0 license is preserved in `LICENSE`; this derivative is distributed under it.

```bibtex
@article{gao2026uniadet,
  title={One Language-Free Foundation Model Is Enough for Universal Vision Anomaly Detection},
  author={Gao, Bin-Bin and Wang, Chengjie},
  journal={arXiv preprint arXiv:2601.05552},
  year={2026}
}
```

## Local prepared benchmark run

`datasets` links to `/home/islab/datasets/anomaly`. Generate and validate metadata with
`python scripts/prepare_datasets.py`, then run `.venv/bin/python -u scripts/run_paper.py`.
The runner executes CLIP auxiliary training and zero/1/2/4-shot industrial evaluation
(seed 10), plus zero-shot medical evaluation. It uses the epoch-15 checkpoints and
skips completed jobs on restart. Logs, command records, checkpoints, and metrics
are under `runs/paper_clip`; dataset counts and comparison caveats are in
`runs/preparation/datasets.json` and `runs/preparation/notes.md`.
Real-IAD is absent; the supplied KSDD split differs from the paper. This run covers
CLIP with one seed; DINO backbones and multi-seed error bars are not included.
