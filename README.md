# UniADet — independent PyTorch reimplementation

Implementation of [One Language-Free Foundation Model Is Enough for Universal Vision Anomaly Detection](https://arxiv.org/abs/2601.05552) (arXiv v1, January 2026), based on [AdaptCLIP](https://github.com/gaobb/AdaptCLIP). This is a paper reimplementation, not the authors’ official UniADet release. Benchmark numbers have **not** been reproduced.

Start with [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md) for the `train_test.py`
workflow, on/off configuration switches, checkpoint compatibility, and auxiliary-validation
experiments. The mask-preservation fix is enabled by default; feature-extraction
alternatives remain explicit ablations because the paper does not settle them.
For ordered, copy-and-run experiment commands in Vietnamese, use
[RUN_EXPERIMENTS_VI.md](RUN_EXPERIMENTS_VI.md). `--experiment baseline` uses a
full VisA training and zero-shot MVTec evaluation. `--aux-holdout` opts into
a separate VisA category holdout; `--full-data` remains a compatibility alias
for the default preset protocol. `--dry-run` prints the plan without running a model.

Run the selected configuration with `python train_test.py`: softmax before
resize, Gaussian sigma 4, layer pooling, full VisA training and zero-shot MVTec
at epoch 15. Outputs use `runs/uniadet-best-current/`. Saved checkpoint settings
remain authoritative during evaluation; old missing fields retain legacy defaults.

## Install

Python 3.10+ (DINOv3 may require newer Python/PyTorch; follow its upstream environment requirements).

With uv, PyTorch and torchvision use the CUDA 12.6 wheel index on Linux and Windows, so `uv sync` preserves compatibility with the training machine's driver. Other dependencies use PyPI. The default development group includes pytest.

Keep `pyproject.toml`, `requirements.txt`, `uv.lock`, and `.python-version` together in version control. The lockfile records exact dependency versions; `.python-version` selects Python 3.11.15 to match the validated training environment. Use `uv sync --locked` to reject an outdated lockfile instead of changing it. Packages installed manually outside the project dependencies may be removed by sync.

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
uv run python -m pytest -q
uv run python train.py --data datasets/VisA --backbone clip --device cuda --output runs/visa
```

Alternatively, install with pip:

```bash
python -m venv .venv
source .venv/bin/activate
# Install the torch/torchvision build appropriate for your CPU or CUDA first.
pip install -r requirements.txt
pip install pytest
python -m pytest -q
```

CLIP, EVA02-CLIP and DINOv2 download pretrained weights on first use. DINOv3 requires separately obtained official weights and accepts a local official repository checkout via `--dino-repo`. Backbone code/weights retain their respective upstream licenses and access conditions.

If training reports that the NVIDIA driver is too old, check `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`. An unavailable CUDA device makes the default training device CPU. On a driver supporting CUDA 12.6, install compatible wheels with `python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126`, then restart training with `--device cuda` to require GPU execution.

## Prepare datasets

Download and extract the datasets before generating metadata. Run these commands
from the repository root. By default, `datasets/` is the dataset parent directory;
on this machine it is a symlink to `/home/islab/datasets/anomaly`. For another
location, pass `--base /path/to/datasets` to the preparation script.

The preparation script expects these paths relative to that parent:

| Dataset | Extracted directory / required layout |
|---|---|
| VisA | `VisA/`, including `split_csv/1cls.csv` and the original image/mask paths referenced by that CSV |
| MVTec AD | `MVTecAD/<category>/{train,test,ground_truth}/` |
| BTAD | `btad/BTech_Dataset_transformed/<category>/{train,test,ground_truth}/` (normal folder: `ok`) |
| KSDD | `KSDD/SDD/<category>/{train,test,ground_truth}/` (local converted split) |
| DTD-Synthetic | `DTD/Zero-shot texture AD/DTD-Synthetic.zip`; extracted automatically if needed |
| HeadCT | `HeadCT_anomaly_detection/<category>/{train,test}/` (classification only) |
| ISIC | `ISBI-ISIC/<category>/test/<label>/`, with masks in `<category>/ISBI2016_ISIC_Part1_Test_GroundTruth/` |
| BrainMRI / Br35H | `BrainMRI/brain_tumor_dataset/{no,yes}/` and `Br35H/{no,yes}/` |
| ClinicDB / ColonDB / Kvasir | `CVC-Kvasir/{CVC-ClinicDB,CVC-ColonDB,Kvasir}/{images,masks}/` |
| EndoTect | `EndoTect_2020_Segmentation_Test_Dataset/EndoTect_2020_Segmentation_Test_Dataset/{images,masks}/` |
| Real-IAD | `Real-IAD/<category>/{OK,NG}/` and `Real-IAD/realiad_jsons/<category>.json` |

For Real-IAD, obtain the 30 category image archives and `realiad_jsons.zip` from
the [official dataset](https://huggingface.co/datasets/Real-IAD/Real-IAD).
Accept its access conditions and authenticate with Hugging Face before downloading.
Extract category archives directly under `datasets/Real-IAD/`; for example,
`datasets/Real-IAD/audiojack/OK/...`. Place `realiad_jsons.zip` in that same root;
the script extracts it automatically. It uses the official train/test splits,
converts them to `meta.json`, and preserves camera IDs `C1`–`C5` in `view_id`.
See the [official preparation instructions](https://github.com/Tencent/AnomalyDetection_Real-IAD)
for the source archive layout; this project keeps category folders directly under
`Real-IAD/`, without the extra `realiad_1024/` level.

```bash
# Prepare just VisA and MVTec for the baseline (no other datasets required).
uv run python scripts/prepare_datasets.py --datasets visa_mvtec

# Prepare only Real-IAD (does not require the other datasets).
uv run python scripts/prepare_datasets.py --datasets realiad

# Prepare all datasets listed above (default).
uv run python scripts/prepare_datasets.py

# Prepare the other datasets when Real-IAD is unavailable.
uv run python scripts/prepare_datasets.py --datasets existing

# Override input and preparation-report locations.
uv run python scripts/prepare_datasets.py --datasets realiad \
  --base /path/to/datasets --output-dir runs/preparation
```

Each successful conversion validates image/mask file paths and writes `meta.json`
inside the dataset root used for training/evaluation. It overwrites existing metadata.
`runs/preparation/datasets.json` records absolute roots and normal/anomaly counts
per split; preparing only Real-IAD preserves existing report entries for other datasets.
No images are downloaded by this script. Segmentation-only medical datasets have
only a test split; few-shot evaluation requires normal target-train images.

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

Category names come from dictionary keys. `view_id` is optional, defaults to `"0"`, and separates multi-view reference banks. Use `scripts/prepare_datasets.py` as described above. Legacy AdaptCLIP generators remain in `dataset/` and require editing their root paths. Images use PIL-supported formats. Nonzero mask values indicate anomaly. Missing anomalous masks are allowed for classification-only evaluation and excluded from pixel metrics; training requires masks.

The paper trains on the **auxiliary dataset’s test split**, which contains both normal and annotated anomaly images. Train on VisA, evaluate on MVTec and other domains; train a separate model on MVTec when evaluating VisA. The default `--split test` is intentional. Zero-shot means no target-domain training, not no auxiliary supervision.

Training applies a uniformly sampled rotation of 0°, 90°, 180°, or 270° and independent horizontal and vertical flips (50% probability each). Images and ground-truth masks receive identical transforms before resizing, preserving binary masks and image-level labels. These augmentations remain enabled with `--no-caa`; evaluation and reference images are not augmented.

Rotation is disabled by default. Use `python train.py --no-rotation` to disable it or `--rotation` to enable it explicitly. In `train_test.py`, set `TRAIN_CONFIG["rotation"]` to `False` or `True`. Horizontal and vertical flips remain enabled independently of this setting.

Flips can also be controlled independently with `--horizontal-flip` / `--no-horizontal-flip` and `--vertical-flip` / `--no-vertical-flip`. Both default to disabled; when enabled, each has a 50% chance per image. In `train_test.py`, set `TRAIN_CONFIG["horizontal_flip"]` and `TRAIN_CONFIG["vertical_flip"]` to `False` or `True`. To disable all geometric augmentation, pass `--no-rotation --no-horizontal-flip --no-vertical-flip`.

```bash
python train.py --data datasets/visa --backbone clip --output runs/visa
python test.py --data datasets/mvtec --checkpoint runs/visa/epoch_15.pt --shots 0 --output runs/mvtec_zero
python test.py --data datasets/mvtec --checkpoint runs/visa/epoch_15.pt --shots 4 --output runs/mvtec_four

python train.py --data datasets/visa --backbone dinov2 --output runs/visa_dinov2
python train.py --data datasets/visa --backbone dinov3 --weights /weights/dinov3_vitl16.pth --dino-repo /repos/dinov3 --output runs/visa_dinov3
```

For comparison with CLIP, select `--backbone eva02_clip`. This uses
[`EVA02_CLIP_L_336_psz14_s6B`](https://github.com/baaivision/EVA/tree/master/EVA-CLIP)
via the timm vision-only checkpoint
[`eva02_large_patch14_clip_336.merged2b`](https://huggingface.co/timm/eva02_large_patch14_clip_336.merged2b).
It uses the same five blocks (12, 15, 18, 21, 24), CLIP normalization, and
768-dimensional projected class/patch tokens as the CLIP interface. EVA's
rotary position embeddings are handled by timm. The default input remains 518
for comparison at the same resolution; use `--image-size 336` for the pretrained
resolution. Keep image size, data split, augmentation, seed and shots identical
when comparing backbones, and use separate output directories.

```bash
python train.py --data datasets/VisA --backbone clip --output runs/visa_clip
python train.py --data datasets/VisA --backbone eva02_clip --output runs/visa_eva02_clip
python test.py --data datasets/MVTecAD --checkpoint runs/visa_eva02_clip/epoch_15.pt --shots 0 --output runs/mvtec_eva02_clip
```

EVA's `--weights` accepts a local **timm vision tower** state dictionary,
including the pretrained 1024-to-768 projection (`head`); the original full
EVA-CLIP checkpoint must first be converted to timm layout. Evaluation restores
the backbone from the training checkpoint. In `train_test.py`, set
`TRAIN_CONFIG["backbone"] = "eva02_clip"` and give `MODEL` a separate experiment
name to keep its checkpoints and comparison CSV separate from CLIP runs.

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
| CLIP, EVA02-CLIP, DINOv2 with registers, DINOv3 frozen features | `uniadet/backbones.py` |

Default blocks are **12, 15, 18, 21, 24** (one-based). Images resize to 518 for CLIP/DINOv2 and 512 for DINOv3. Only the head is optimized, for 15 epochs at learning rate 0.001. Both fusion coefficients default to 0.5. CAA runs with probability 0.5; grids 2×2 and 3×3 are equally likely. Normal mosaics contain only normal images of the same class; anomalous mosaics retain the candidate and sample any same-class images. Anomalous crops must intersect the mask.

Training checks that positive masks survive final resizing. If augmentation erases
the entire mask, `--preserve-anomaly-mask` (default) falls back to the original
candidate; an unusable original mask raises an error. `--no-preserve-anomaly-mask`
reproduces legacy behavior for ablation. Evaluation masks are unchanged. Use
`--categories` / `--exclude-categories` to split auxiliary categories; excluded
images are also excluded from CAA sampling.

Evaluation and `scripts/audit_zero_shot.py` restore `lambda_f`, `lambda_p`, and
`zero_score_pooling` from the checkpoint unless explicit CLI values override
them. Missing fields in old checkpoints use 0.5, 0.5, and `layer`. The dev
runner's `TEST_CONFIG` accepts the same overrides; `None` restores saved values.
The checkpoint's `--clip-global-norm` / `--no-clip-global-norm` and
`--clip-pos-interpolation {bicubic,bilinear}` settings also restore. Changing
these features requires a newly trained head; conflicting evaluation flags
raise an error. Defaults preserve the old checkpoint representation.

Implementation choices where the paper is underspecified:

- Temperature 0.07, Gaussian weight initialization with standard deviation `d**-0.5`, Adam betas `(0.5, 0.999)` inherited from AdaptCLIP, no weight decay or schedule. Training flags include `--temperature`, `--head-init-std`, `--betas`, `--weight-decay`, and `--lr`. Independent focal/normal-Dice/anomaly-Dice weights default to 1; setting a weight to 0 disables that term. `--focal-smoothing 1e-5` enables AnomalyCLIP's smoothed focal form; default 0 preserves the existing loss.
- Mosaic versus cropping is sampled equally. Cropping uses a uniform grid; the appendix does not specify random boundary distributions. Mosaic candidate position is randomized.
- Per-layer losses are summed. Inference averages per-layer probabilities and, by default, resizes probabilities after softmax. `--no-softmax-before-resize` restores legacy logit resizing and requires a separately trained head. Zero-shot image scoring averages the per-layer spatial maxima before fusion with global scores. `--zero-score-pooling map` takes the maximum after averaging maps. The paper does not explicitly specify the layer reduction order. Few-shot image scoring uses the maximum of the **fused** map, following Eqs. 11–12. Gaussian smoothing now defaults to sigma 4 for new runs; `--gaussian-sigma 0` disables it. This is the best measured local configuration, not a confirmed official UniADet setting.
- CLIP uses unmodified frozen attention blocks, with final layer normalization and visual projection applied to each selected layer by default. `--no-clip-global-norm` disables only global-token layer normalization; patch tokens remain normalized. CLIP positional interpolation defaults to bicubic and can be switched to bilinear for a separately trained experiment. AdaptCLIP’s DPAM replacement and all adapters are disabled. Positional embeddings interpolate without mutating pretrained parameters. DINO uses the official normalized intermediate class/patch tokens, excluding register/storage tokens.
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

Current validation: Python 3.12.3, PyTorch 2.7.1+cu126 and torchvision 0.22.1+cu126: **120 tests passed**. The real VisA candle case now retains a nonempty mask, and a real CLIP toothbrush audit exactly matches the old checkpoint scores. Full benchmark retraining after the fixes has not been run. See [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md) for current commands and [CODE_PAPER_REVIEW.md](CODE_PAPER_REVIEW.md) for the pre-fix comparison.

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
Real-IAD preparation is supported, including its official splits and five camera views;
the supplied KSDD split differs from the paper. This run covers
CLIP with one seed; DINO backbones and multi-seed error bars are not included.

## Dev training and checkpoint testing

`train.py` and `test.py` retain their original CLI parameters and output paths.
Training additionally saves `training_config.json`, `training.csv`, and
`training.jsonl` with step/epoch loss components, learning rate, sample counts,
and elapsed epoch seconds. Epoch losses are sample-weighted averages. Logs
append when reusing an output directory; matching checkpoints are overwritten.

For the configuration-driven workflow, edit `MODEL`, `TRAIN_CONFIG`, and
`TEST_CONFIG` at the top of `train_test.py`, then run:

```bash
python train_test.py                         # train, then evaluate epoch 15 by default
python train_test.py --mode train            # train only
python train_test.py --mode test             # evaluate configured saved checkpoints
```

Training always starts and saves checkpoints from epoch 1. `start_epoch` only
sets an inclusive lower bound for testing. `TEST_CONFIG["epochs"]` defaults to
`[15]`; change it to another list to evaluate selected epochs, or `None` to sweep
all saved epochs. For a sweep starting at epoch 5, set it to `None`, then run
`python train_test.py --mode test --start-epoch 5`. The combined mode completes
training before testing. `MODEL` defaults to `uniadet-best-current`; use a new
name per experiment. The runner saves checkpoints under
`runs/<MODEL>/train_<dataset>/epoch_<N>.pt` and results under
`runs/<MODEL>/<target>_<shots>shot/epoch_<N>/`. The default root is `runs` for both.
`TEST_CONFIG["checkpoint"]` can select another checkpoint directory or a single
checkpoint file; a single file is evaluated directly without epoch filtering.
Test fusion/source settings default to `None` to restore the checkpoint.
Optional test batch size, workers, device, and seed override only evaluation;
`None` inherits the current training execution settings. All switches and
step-by-step examples are in [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md).

After the dev runner finishes testing all selected epochs, it writes
`runs/<MODEL>/test_summary.csv`. Each row contains dataset, shot count, epoch,
checkpoint path, and the seven `average` metrics (percentages, matching
`metrics.csv`; undefined metrics are blank). The file compares all datasets,
shot counts, and epochs tested in that invocation and is replaced on the next
successful test run.
