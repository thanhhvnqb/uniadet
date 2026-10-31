# UniADet: step-by-step reproduction and controlled ablations

For copy-and-run commands with experiment presets, see the Vietnamese
[RUN_EXPERIMENTS_VI.md](RUN_EXPERIMENTS_VI.md). These presets include softmax
before resize and paired Gaussian OFF/ON evaluations on full zero-shot MVTec.
Preset defaults train on all VisA; `--aux-holdout` explicitly opts into a VisA
holdout. Configuration selection uses MVTec test results as requested.
Use `python train_test.py --experiment baseline --dry-run` to inspect a plan.
Without `--experiment`, the editable configuration workflow below is unchanged.

Run commands from the repository root in Bash. The baseline below follows the
paper's documented CLIP settings and keeps the selected softmax-before-resize, Gaussian sigma 4, and layer-pooling
configuration. It enables the mask-preservation fix. Additional geometry
is explicitly disabled to match the saved local baseline and the dev runner.

The paper does not specify every implementation detail. The configuration switches make
those choices testable; they do not establish that a particular alternative is
the authors' implementation or guarantee the reported scores.

## 1. Set up the environment and check the changes

```bash
uv sync --locked
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
uv run python -m pytest -q
```

CUDA availability should print `True` for GPU training. In `train_test.py`,
`TRAIN_CONFIG["device"] = None` automatically selects CUDA or CPU. Set it to
`"cuda"` to require GPU execution, or `"cpu"` for a small functional check.
Pretrained CLIP weights download on first use if they are not already cached.

If using the existing environment without uv, replace `uv run python` with
`.venv/bin/python` throughout this guide.

## 2. Prepare only VisA and MVTec

The extracted layouts should be:

```text
datasets/VisA/split_csv/1cls.csv
datasets/VisA/<original paths referenced by 1cls.csv>
datasets/MVTecAD/<category>/{train,test,ground_truth}/...
```

On this machine `datasets/` already links to the dataset parent directory.
Generate metadata without requiring the other twelve benchmarks:

```bash
uv run python scripts/prepare_datasets.py --datasets visa_mvtec
```

For another dataset location:

```bash
uv run python scripts/prepare_datasets.py --datasets visa_mvtec --base /path/to/datasets
```

Use that same dataset parent in subsequent `--data` arguments. Preparation
validates paths and overwrites the two datasets' `meta.json` files. It writes
counts to `runs/preparation/datasets.json` and preserves other report entries.
It does not download images.

Expected test counts: VisA has 962 normal and 1,200 anomaly images; MVTec has
467 normal and 1,258 anomaly images. These are the labeled auxiliary splits
used for training as well as the target splits used for evaluation.

## 3. Configure `train_test.py`

The selected defaults run with `python train_test.py`: full VisA training,
zero-shot MVTec evaluation at epoch 15, softmax before resize, Gaussian sigma 4,
and layer pooling. Experiment presets still retain their original ablation
settings for reproducible comparisons.

The default file is ready for a fixed CLIP baseline. Edit configuration entries
at the top of `train_test.py`; no long training or evaluation command is needed.
Use a new `MODEL` name for each experiment because logs append and checkpoints
can be overwritten when you reuse a name.

| Location | Default | Meaning |
|---|---|---|
| `MODE` | `"train_test"` | Train, then evaluate selected checkpoints |
| `MODEL` | `"uniadet-best-current"` | Experiment directory under `runs/` |
| `TRAIN_CONFIG["data"]` | `"datasets/VisA"` | Auxiliary dataset |
| `TRAIN_CONFIG["backbone"]` | `"clip"` | CLIP reference experiment |
| `TRAIN_CONFIG["split"]` | `"test"` | Labeled auxiliary split used by the paper |
| `TRAIN_CONFIG["epochs"]` | `15` | Number of training epochs |
| `TRAIN_CONFIG["layers"]` | `[12, 15, 18, 21, 24]` | One-based feature blocks |
| `TRAIN_CONFIG["image_size"]` | `None` | Automatically 518 for CLIP/DINOv2, 512 for DINOv3 |
| `TRAIN_CONFIG["lr"]` | `0.001` | Learning rate |
| `TRAIN_CONFIG["batch_size"]` | `8` | Training batch size |
| `TRAIN_CONFIG["seed"]` | `10` | Training seed |
| `TEST_CONFIG["data"]` | `["datasets/MVTecAD"]` | Target datasets |
| `TEST_CONFIG["shots"]` | `[0]` | Zero-shot evaluation |
| `TEST_CONFIG["epochs"]` | `[15]` | Evaluate epoch 15 only |
| `TEST_CONFIG["checkpoint"]` | `None` | Use this experiment's training directory |

CAA and the mask-preservation fix are enabled. Extra rotations and flips are
disabled to match the saved local baseline. An augmented positive with an empty
final mask falls back to its original candidate. An unusable original mask
raises an error with its image path.

`TRAIN_CONFIG["categories"]` and `["exclude_categories"]` default to `None`,
so baseline training uses all auxiliary categories. The frozen CLIP model has
15,360 trainable head parameters. The numerical defaults for temperature,
initialization, and some feature processing remain reproduction choices where
the paper is underspecified.

## 4. Train and evaluate with one command

```bash
uv run python -u train_test.py
```

This trains for 15 epochs on VisA, saves every checkpoint, then evaluates only
`epoch_15.pt` on MVTec. Training always starts at epoch 1; resume is not
implemented. Use these modes when running stages separately:

```bash
uv run python -u train_test.py --mode train
uv run python -u train_test.py --mode test
```

Outputs with the default configuration:

```text
runs/uniadet-best-current/
  train_VisA/
    training_config.json
    training.csv
    training.jsonl
    epoch_1.pt ... epoch_15.pt
  MVTecAD_0shot/epoch_15/
    evaluation_config.json
    metrics.csv
    metrics.json
    predictions.json
  test_summary.csv
```

Inspect `test_summary.csv`, or the final `Average` row of `metrics.csv`. Both
use percentages; JSON metrics use [0, 1]. The summary covers the current test
invocation and replaces its previous version after a successful run.

For the CLIP zero-shot comparison, the paper reports MVTec image AUROC 92.4%,
image AUPR 96.1%, pixel AUROC 91.8%, and pixel AUPR 42.8%.
Source: [Tables 2 and 12](https://arxiv.org/pdf/2601.05552).
No full benchmark retraining after these fixes is claimed here.

## 5. Select shots, checkpoints, and evaluation settings

Edit these existing entries in `TEST_CONFIG` as needed:

| Setting | Example value | Effect |
|---|---|---|
| `shots` | `[0, 1, 2, 4]` | Evaluate zero-, one-, two-, and four-shot |
| `epochs` | `[15]` | Paper comparison at epoch 15 |
| `epochs` | `[5, 10, 15]` | Evaluate a specific set of saved epochs |
| `epochs` | `None` | Sweep all saved epochs at/after `start_epoch` |
| `start_epoch` | `5` | Additional inclusive lower bound on selected epochs |
| `checkpoint` | `"runs/uniadet-clip/train_VisA"` | Evaluate another run's checkpoint directory |
| `checkpoint` | `"runs/uniadet-clip/train_VisA/epoch_15.pt"` | Evaluate this file directly, ignoring epoch filters |
| `batch_size` | `4` | Reduce only evaluation batch size |
| `workers` | `0` | Disable only evaluation data-loader workers |
| `device` | `"cuda"` | Require CUDA for evaluation |
| `seed` | `20` | Change reference-selection seed |

Run `uv run python -u train_test.py --mode test` after editing test settings.
Leave `checkpoint=None` to test your current `MODEL`; set an external directory
or file to evaluate an existing run without retraining. A directory uses the
selected epoch list and creates `epoch_<N>/` result folders. A single file
writes directly to `<target>_<shots>shot/` and records its actual epoch in the
summary and `evaluation_config.json`.

The optional execution fields default to `None`, which inherits `TRAIN_CONFIG`.
Backbone, feature layers, resolution, temperature, and CLIP representation
settings always restore from the selected checkpoint. Pretrained source paths
also restore; `TEST_CONFIG["weights"]` and `["dino_repo"]` may explicitly
relocate them if needed. The current training source paths do not override a
different checkpoint's sources.

For an epoch sweep, set `TEST_CONFIG["epochs"] = None`, then optionally run:

```bash
uv run python -u train_test.py --mode test --start-epoch 5
```

`--start-epoch` affects only testing. If `epochs=[15]`, this example still tests
only epoch 15. A missing requested epoch raises an error. Use epoch 15 for the
paper comparison; choosing an epoch using target-test scores biases the result.

Few-shot references come only from normal images in the target train split;
selected records are saved to `references.json`. Change the experiment name
when preserving outputs for several reference seeds.

## 6. Turn fixes and ablations on/off in `TRAIN_CONFIG`

Edit the existing dictionary entries in `train_test.py` using Python `True`,
`False`, and `None`. These settings apply to the next training run, so give it
a fresh `MODEL` name.

| Key | Baseline | Alternative / off setting |
|---|---|---|
| `preserve_anomaly_mask` | `True` | `False` reproduces the old empty-mask behavior |
| `no_caa` | `False` (CAA ON) | `True` turns CAA OFF; the legacy key is inverted |
| `caa_probability` | `0.5` | Probability of applying CAA |
| `caa_grids` | `[2, 3]` | Available mosaic/crop grid sizes |
| `mosaic_probability` | `0.5` | `1.0` = mosaic only; `0.0` = crop only within CAA |
| `rotation` | `False` | `True` enables right-angle rotations |
| `horizontal_flip` / `vertical_flip` | `False` | `True` enables each flip independently |
| `clip_global_norm` | `True` | `False` bypasses only global-token layer normalization |
| `clip_pos_interpolation` | `"bicubic"` | `"bilinear"` tests the inherited interpolation choice |
| `softmax_before_resize` | `True` | `False` restores logit resizing; requires retraining |
| `mask_interpolation` | `"nearest"` | `"bilinear"` resizes masks then thresholds at 0.5; requires retraining |
| `temperature` | `0.07` | Another positive training temperature |
| `head_init_std` | `None` (`feature_dim**-0.5`) | E.g. `0.02` fixes Gaussian initialization scale |
| `focal_smoothing` | `0.0` | `1e-5` enables AnomalyCLIP's smoothed focal form |
| `focal_weight` | `1.0` | `0.0` disables focal loss |
| `normal_dice_weight` | `1.0` | `0.0` disables normal Dice |
| `anomaly_dice_weight` | `1.0` | `0.0` disables anomaly Dice |
| `classification_weight` / `segmentation_weight` | `1.0` | `0.0` disables the respective task loss |

CLIP feature options apply only to `backbone="clip"`. Patch-token final
normalization and projection remain enabled when global normalization is off.
The pretrained positional parameters are never mutated. Feature, temperature,
initialization, augmentation, and loss changes require retraining; changing a
training setting and running only `--mode test` cannot apply it retroactively.

The bilinear-mask training variant keeps the mask-preservation guard enabled.
If even the original bilinear-resized mask is empty, training falls back to
nearest resizing for that candidate; evaluation retains bilinear/threshold.

## 7. Override image scoring without retraining

The default configuration now uses `softmax_before_resize=True`,
`gaussian_sigma=4.0`, and `zero_score_pooling="layer"`. These reproduce the
best measured configuration in the current seed-10 runs, not a guarantee of
identical scores across all environments. Existing checkpoints retain their
saved settings; missing historical fields still mean softmax-first OFF and
sigma 0.

Prediction defaults are saved from `TRAIN_CONFIG["lambda_f"]`, `["lambda_p"]`,
and `["zero_score_pooling"]`. Their `TEST_CONFIG` counterparts default to
`None`, which restores the selected checkpoint. Explicit test values override
them, including a value equal to the old default.

| `TEST_CONFIG` key | Value | Meaning |
|---|---|---|
| `lambda_p` | `None` | Restore saved image fusion weight |
| `lambda_p` | `0.0` | Use only global-head image scores |
| `lambda_p` | `1.0` | Use only map-derived image scores |
| `lambda_p` | `0.5` | Explicit equal-weight image fusion |
| `lambda_f` | `None` | Restore saved reference-map fusion weight |
| `lambda_f` | `0.0` to `1.0` | Override reference fusion for few-shot evaluation |
| `zero_score_pooling` | `None` | Restore the checkpoint's pooling choice |
| `zero_score_pooling` | `"layer"` | Average per-layer spatial maxima |
| `zero_score_pooling` | `"map"` | Take the maximum after averaging layer maps |
| `gaussian_sigma` | `None` | Restore the checkpoint's smoothing setting; old checkpoints use 0 |
| `gaussian_sigma` | `0.0` / `4.0` | Disable smoothing / test AdaptCLIP's sigma 4 before spatial pooling |

Run in test-only mode after editing these entries. `MODEL` also names the
result directory, so use a new name plus an explicit checkpoint source to keep
separate diagnostics. Old checkpoints missing the new fields retain the old
CLIP representation, weights 0.5, and layer pooling. Few-shot image scoring
always uses the maximum of the fused map, following Eqs. 11–12.

The paper does not explicitly settle the final zero-shot layer reduction.
The default presets select settings using zero-shot MVTec results, as requested.
Record that selection protocol alongside the reported scores.

For map pooling, Gaussian smoothing applies to the final mean/fused anomaly
map before its spatial maximum. For zero-shot layer pooling, each layer is
smoothed before its maximum; the returned map is still the smoothed mean map.
Few-shot smoothing applies after reference-map fusion. `gaussian-off` and
`gaussian-on` presets both use map pooling and the same baseline head, isolating
sigma 0 versus 4 without retraining.

## 8. Optional: use the runner for held-out auxiliary categories

To investigate the unspecified choices, edit these fields for an auxiliary
baseline before choosing final target settings:

| Field | Value |
|---|---|
| `MODEL` | `"uniadet-aux-baseline"` |
| `TRAIN_CONFIG["data"]` | `"datasets/VisA"` |
| `TRAIN_CONFIG["exclude_categories"]` | `["pcb1", "pcb2"]` |
| `TRAIN_CONFIG["categories"]` | `None` |
| `TEST_CONFIG["data"]` | `["datasets/VisA"]` |
| `TEST_CONFIG["categories"]` | `["pcb1", "pcb2"]` |
| `TEST_CONFIG["exclude_categories"]` | `None` |
| `TEST_CONFIG["epochs"]` | `[15]` |
| `TEST_CONFIG["checkpoint"]` | `None` |
| `TEST_CONFIG["shots"]` | `[0]` |

Run `uv run python -u train_test.py`. Excluded categories cannot enter CAA
sampling, and test filters are independent of training filters. Results go to
`runs/uniadet-aux-baseline/VisA_0shot/epoch_15/metrics.csv`.

Repeat with a fresh `MODEL` name and one changed training entry per experiment:

| Example `MODEL` | Single change from the auxiliary baseline |
|---|---|
| `"uniadet-aux-no-global-norm"` | `clip_global_norm=False` |
| `"uniadet-aux-bilinear"` | `clip_pos_interpolation="bilinear"` |
| `"uniadet-aux-temperature005"` | `temperature=0.05` |
| `"uniadet-aux-init002"` | `head_init_std=0.02` |
| `"uniadet-aux-smoothed-focal"` | `focal_smoothing=1e-5` |
| `"uniadet-aux-focal2"` | `focal_weight=2.0` |

Restore each other entry to the baseline before starting the next variant;
do not accumulate changes. The example values are validation candidates, not
settings disclosed by UniADet. Decide the validation objective in advance and
inspect both image and pixel metrics. Keep the same epoch, split, seed, and
batch size. Changing relative focal/Dice weights changes the objective;
uniformly scaling the entire segmentation loss has less effect with Adam and
independent task heads.

Once settings are selected, use a fresh final `MODEL` name, set
`TRAIN_CONFIG["exclude_categories"]=None`, switch the test dataset back to
`["datasets/MVTecAD"]`, and reset both test category filters to `None`.
Keep `checkpoint=None` to test this newly trained model. Run the same command
to retrain on **all VisA categories** and evaluate epoch 15 on MVTec.

For a VisA target result, use a separate experiment: train on MVTec's labeled
test split and test on VisA. Repeat final runs with predefined seeds and fresh
names to measure variability. EVA02-CLIP, DINOv2, and DINOv3 require their own
trained heads and comparisons; see README for backbone setup.

## 9. Optional image-only audit

Training and normal evaluation use `train_test.py` throughout this guide. This
additional utility compares score components without expensive pixel metrics:

```bash
uv run python scripts/audit_zero_shot.py \
  --data datasets/MVTecAD \
  --checkpoint runs/uniadet-best-current/train_VisA/epoch_15.pt \
  --shots 0 --batch-size 8 --workers 4 --device cuda \
  --output runs/uniadet-best-current/score_audit
```

It writes `score_audit.json` with both fixed pooling definitions, global-only,
map-only, and individual global-head results. It shares checkpoint restoration
with the runner. For auxiliary holdouts, point it at the corresponding head and
VisA with `--categories pcb1 pcb2`. Individual target-head scores are diagnostic;
do not choose a head from target labels.

The standalone `train.py` and `test.py` entrypoints remain available; their CLI
options are documented by `--help` and in [README.md](README.md). See
[CODE_PAPER_REVIEW.md](CODE_PAPER_REVIEW.md) for the measured pre-fix baseline.

Validation: **120 tests passed**, including selected epochs, checkpoint settings,
evaluation overrides, category isolation, feature options, loss terms, and
small-defect preservation. The real VisA candle example retains 6 positive
pixels with the fix versus 0 in legacy mode, and the real CLIP toothbrush audit
matches the existing checkpoint's previous scores exactly. These checks do not
establish improved full-benchmark accuracy after retraining.
