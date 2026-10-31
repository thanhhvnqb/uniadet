"""Configure and run UniADet with python train_test.py.

Edit TRAIN_CONFIG and TEST_CONFIG below. By default, train on VisA for 15
epochs and evaluate only epoch 15 on MVTec. See REPRODUCTION_GUIDE.md.
"""
import argparse
import copy
import csv
import json
from contextlib import contextmanager
from pathlib import Path

from uniadet import cli

MODE = "train_test"  # train, test, or train_test (test saved checkpoints after training)
MODEL = "uniadet-best-current"  # Softmax-first + Gaussian 4 + layer pooling
RUNS_ROOT = "runs"
TRAIN_DATASET = None  # None uses the training data directory name

TRAIN_CONFIG = {
    # Data and frozen backbone. Excluded categories cannot enter CAA mosaics.
    "data": "datasets/VisA", "backbone": "clip",  # clip, eva02_clip, dinov2, dinov3
    "weights": None, "dino_repo": None, "layers": [12, 15, 18, 21, 24],
    "image_size": None,  # Auto: 518 for CLIP/DINOv2; 512 for DINOv3
    "split": "test",  # Labeled auxiliary test split, as used in the paper
    "categories": None, "exclude_categories": None,  # E.g. exclude ["pcb1", "pcb2"]

    # Representation ablations: changing these requires a new trained head.
    "clip_global_norm": True,  # False bypasses only CLIP global-token layer norm
    "clip_pos_interpolation": "bicubic",  # "bicubic" or "bilinear"; CLIP only
    "softmax_before_resize": True,  # Selected configuration: resize probabilities
    "mask_interpolation": "nearest",  # "bilinear" uses threshold 0.5 after resizing
    "temperature": 0.07,
    "head_init_std": None,  # None uses feature_dim**-0.5; e.g. 0.02 for ablation

    # Optimization and execution. None device automatically selects CUDA/CPU.
    "epochs": 15, "lr": 0.001, "betas": [0.5, 0.999], "weight_decay": 0.0,
    "batch_size": 8, "workers": 4, "seed": 10, "device": None,
    "log_interval": 50,

    # Augmentation: True/False switches; mask-preservation fix enabled by default.
    "no_caa": False,  # False = CAA ON; True = CAA OFF (legacy key)
    "preserve_anomaly_mask": True,  # False reproduces the old empty-mask behavior
    "caa_probability": 0.5, "caa_grids": [2, 3], "mosaic_probability": 0.5,
    "rotation": False, "horizontal_flip": False, "vertical_flip": False,

    # Independent loss terms. Weight 0 disables a term; baseline weights are 1.
    "focal_gamma": 2.0, "dice_smooth": 1.0,
    "classification_weight": 1.0, "segmentation_weight": 1.0,
    "focal_weight": 1.0, "normal_dice_weight": 1.0, "anomaly_dice_weight": 1.0,
    "focal_smoothing": 0.0,  # 1e-5 matches AnomalyCLIP's smoothed focal form

    # Prediction defaults saved in the checkpoint. TEST_CONFIG can override them.
    "lambda_f": 0.5, "lambda_p": 0.5,
    "zero_score_pooling": "layer",  # "layer" or "map"; final layer reduction is underspecified
    "gaussian_sigma": 4.0,  # Selected configuration; 0 disables inference smoothing
}

TEST_CONFIG = {
    "data": ["datasets/MVTecAD"],  # Target dataset roots
    "shots": [0],  # Set [0, 1, 2, 4] for zero- and few-shot evaluation
    "epochs": [15],  # Paper comparison: epoch 15 only. None = sweep all saved epochs
    "start_epoch": 1,  # Additional inclusive lower bound; never resumes training
    "checkpoint": None,  # None uses runs/<MODEL>/train_<dataset>

    # None restores the selected checkpoint's settings; values are explicit overrides.
    "lambda_f": None, "lambda_p": None, "zero_score_pooling": None,  # None restores checkpoint
    "gaussian_sigma": None,  # None restores checkpoint; 0 disables smoothing
    "weights": None, "dino_repo": None,  # Only override to relocate pretrained sources
    "categories": None, "exclude_categories": None,  # Target/validation categories

    # Optional evaluation-only execution settings. None inherits TRAIN_CONFIG.
    "batch_size": None, "workers": None, "device": None, "seed": None,
}


EXPERIMENTS = {
    "baseline": {},
    "softmax-first": {"train": {"softmax_before_resize": True}},
    "no-global-norm": {"train": {"clip_global_norm": False}},
    "bilinear-pos": {"train": {"clip_pos_interpolation": "bilinear"}},
    "smoothed-focal": {"train": {"focal_smoothing": 1e-5}},
    "bilinear-mask": {"train": {"mask_interpolation": "bilinear"}},
    # Both reuse the baseline head and map pooling: their only difference is sigma.
    "gaussian-off": {"test_only": True, "test": {"zero_score_pooling": "map", "gaussian_sigma": 0.}},
    "gaussian-on": {"test_only": True, "test": {"zero_score_pooling": "map", "gaussian_sigma": 4.}},
}


@contextmanager
def experiment_config(name=None, full_data=True, overrides=None):
    """Isolate each preset/CLI invocation, retaining the editable-config workflow."""
    global MODEL, TRAIN_CONFIG, TEST_CONFIG, TRAIN_DATASET
    original = MODEL, TRAIN_CONFIG, TEST_CONFIG, TRAIN_DATASET
    TRAIN_CONFIG, TEST_CONFIG = copy.deepcopy(TRAIN_CONFIG), copy.deepcopy(TEST_CONFIG)
    try:
        if name is not None:
            scope = "final" if full_data else "aux"
            MODEL, TRAIN_DATASET = f"uniadet-{scope}-{name}", None
            # Explicit scientific settings keep variants controlled even if the
            # editable dictionaries previously contained another experiment.
            TRAIN_CONFIG.update(
                data="datasets/VisA", backbone="clip", weights=None, dino_repo=None,
                layers=[12, 15, 18, 21, 24], image_size=518, split="test",
                categories=None, exclude_categories=None if full_data else ["pcb1", "pcb2"],
                clip_global_norm=True, clip_pos_interpolation="bicubic", softmax_before_resize=False,
                mask_interpolation="nearest", temperature=0.07, head_init_std=None,
                epochs=15, lr=0.001, betas=[0.5, 0.999], weight_decay=0., batch_size=8, seed=10,
                no_caa=False, preserve_anomaly_mask=True, caa_probability=0.5,
                caa_grids=[2, 3], mosaic_probability=0.5, rotation=False,
                horizontal_flip=False, vertical_flip=False, focal_gamma=2., dice_smooth=1.,
                classification_weight=1., segmentation_weight=1., focal_weight=1.,
                normal_dice_weight=1., anomaly_dice_weight=1., focal_smoothing=0.,
                lambda_f=0.5, lambda_p=0.5, zero_score_pooling="layer", gaussian_sigma=0.)
            TEST_CONFIG.update(
                data=["datasets/MVTecAD" if full_data else "datasets/VisA"],
                categories=None if full_data else ["pcb1", "pcb2"], exclude_categories=None,
                shots=[0], epochs=[15], start_epoch=1, checkpoint=None,
                weights=None, dino_repo=None, lambda_f=None, lambda_p=None,
                zero_score_pooling=None, gaussian_sigma=None,
                batch_size=None, workers=None, device=None, seed=None)
            preset = EXPERIMENTS[name]
            TRAIN_CONFIG.update(preset.get("train", {}))
            TEST_CONFIG.update(preset.get("test", {}))
            if preset.get("test_only"):
                TEST_CONFIG["checkpoint"] = str(Path(RUNS_ROOT) / f"uniadet-{scope}-baseline" / "train_VisA")
        for key, value in (overrides or {}).items():
            if value is not None:
                TRAIN_CONFIG[key] = value
                TEST_CONFIG[key] = value
        yield
    finally:
        MODEL, TRAIN_CONFIG, TEST_CONFIG, TRAIN_DATASET = original


def output_path(data, *, training=False, shots=0):
    dataset = (TRAIN_DATASET or Path(data).name) if training else Path(data).name
    for name in (MODEL, dataset):
        if name in ("", ".", "..") or "/" in name or "\\" in name:
            raise ValueError("Model and dataset names must be simple directory names")
    return Path(RUNS_ROOT) / MODEL / (f"train_{dataset}" if training else f"{dataset}_{shots}shot")


def training_args():
    args = cli.parser().parse_args(["train"])
    vars(args).update(copy.deepcopy(TRAIN_CONFIG))
    args.device = args.device or ("cuda" if cli.torch.cuda.is_available() else "cpu")
    args.image_size = args.image_size or (512 if args.backbone == "dinov3" else 518)
    cli.resolve_training_settings(args)
    args.output = str(output_path(args.data, training=True))
    args.model = MODEL
    if args.epochs < 1 or args.batch_size < 1 or args.log_interval < 1:
        raise ValueError("epochs, batch_size and log_interval must be positive")
    if not 0 <= args.caa_probability <= 1 or not 0 <= args.mosaic_probability <= 1:
        raise ValueError("Augmentation probabilities must be in [0, 1]")
    if any(grid < 1 for grid in args.caa_grids):
        raise ValueError("CAA grids must be positive")
    if args.focal_gamma < 0 or args.dice_smooth <= 0:
        raise ValueError("focal_gamma must be nonnegative and dice_smooth positive")
    return args


SUMMARY_METRICS = [
    ("image", "auroc", "I-AUROC"), ("image", "aupr", "I-AUPR"),
    ("image", "f1max", "I-F1max"), ("pixel", "auroc", "P-AUROC"),
    ("pixel", "aupr", "P-AUPR"), ("pixel", "f1max", "P-F1max"),
    ("pixel", "aupro", "P-AUPRO"),
]


def test_checkpoints(args, start_epoch):
    if start_epoch < 1 or any(shots < 0 for shots in TEST_CONFIG["shots"]):
        raise ValueError("start_epoch must be positive and shots nonnegative")
    source = Path(TEST_CONFIG["checkpoint"] or args.output)
    if source.is_file():
        # An explicit file is used directly, regardless of epoch filters.
        checkpoints = [(None, source)]
    else:
        requested = TEST_CONFIG.get("epochs")
        if requested is not None and (not isinstance(requested, (list, tuple)) or not requested or any(
            not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1 for epoch in requested
        )):
            raise ValueError("TEST_CONFIG['epochs'] must be None or a nonempty list of positive integers")
        checkpoints = sorted((int(p.stem[6:]), p) for p in source.glob("epoch_*.pt")
                             if p.stem[6:].isdigit() and int(p.stem[6:]) >= start_epoch
                             and (requested is None or int(p.stem[6:]) in requested))
        if requested is not None:
            missing = {epoch for epoch in requested if epoch >= start_epoch} - {epoch for epoch, _ in checkpoints}
            if missing:
                raise ValueError(f"Missing requested checkpoint epochs {sorted(missing)} in {source}")
    if not checkpoints:
        raise ValueError(f"No checkpoints at or after epoch {start_epoch} in {source}")
    summary = []
    for epoch, checkpoint in checkpoints:
        for target in TEST_CONFIG["data"]:
            for shots in TEST_CONFIG["shots"]:
                current = copy.deepcopy(args)
                current.mode, current.data, current.shots = "evaluate", target, shots,
                current.checkpoint = str(checkpoint)
                # Testing another run must use that checkpoint's representation
                # and fusion, rather than this file's current training settings.
                for name in (*cli.PREDICTION_DEFAULTS, "weights", "dino_repo", "categories", "exclude_categories"):
                    setattr(current, name, copy.deepcopy(TEST_CONFIG.get(name)))
                for name in cli.TRAINED_SETTING_DEFAULTS:
                    setattr(current, name, None)
                for name in ("batch_size", "workers", "device", "seed"):
                    value = TEST_CONFIG.get(name)
                    if value is not None:
                        setattr(current, name, value)
                if current.batch_size < 1 or current.workers < 0:
                    raise ValueError("Evaluation batch_size must be positive and workers nonnegative")
                output = output_path(target, shots=shots)
                current.output = str(output / f"epoch_{epoch}" if epoch is not None else output)
                cli.seed_all(current.seed)
                print(f"Testing {checkpoint}: {target}, shots={shots}", flush=True)
                cli.evaluate(current)
                report = json.loads((Path(current.output) / "metrics.json").read_text())
                average = report.get("average", report.get("macro_mean"))
                if average is None:
                    raise ValueError(f"Missing average metrics in {current.output}/metrics.json")
                saved_epoch = epoch
                if saved_epoch is None:
                    config_path = Path(current.output) / "evaluation_config.json"
                    saved_epoch = json.loads(config_path.read_text()).get("checkpoint_epoch")
                row = {"Dataset": Path(target).name, "Shots": shots, "Epoch": saved_epoch,
                       "Checkpoint": str(checkpoint)}
                for task, metric, column in SUMMARY_METRICS:
                    value = (average.get(task) or {}).get(metric)
                    row[column] = 100 * value if value is not None else ""
                summary.append(row)
    summary_path = Path(RUNS_ROOT) / MODEL / "test_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Dataset", "Shots", "Epoch", "Checkpoint"] +
                                [column for _, _, column in SUMMARY_METRICS])
        writer.writeheader()
        writer.writerows(summary)
    temporary.replace(summary_path)
    print(f"Saved epoch comparison: {summary_path}", flush=True)


# Prevent pytest from collecting the dev helper as a test.
test_checkpoints.__test__ = False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "test", "train_test"),
                        help="Default: configured MODE; Gaussian presets default to test only")
    parser.add_argument("--start-epoch", type=int, help="First checkpoint epoch to test only")
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS),
                        help="Train on full VisA and evaluate zero-shot MVTec, with a separate output directory")
    protocol = parser.add_mutually_exclusive_group()
    protocol.add_argument("--full-data", dest="full_data", action="store_true",
                          help="Train on full VisA and evaluate MVTec (preset default)")
    protocol.add_argument("--aux-holdout", dest="full_data", action="store_false",
                          help="Optional: hold out VisA pcb1/pcb2 and evaluate those categories")
    parser.set_defaults(full_data=True)
    parser.add_argument("--dry-run", action="store_true", help="Print planned settings without training or evaluation")
    parser.add_argument("--device", help="Override execution device for this invocation")
    parser.add_argument("--batch-size", type=int, help="Override batch size; keep identical across compared runs")
    parser.add_argument("--workers", type=int, help="Override data-loader workers")
    parser.add_argument("--seed", type=int, help="Override seed for this invocation")
    options = parser.parse_args(argv)
    test_only = EXPERIMENTS.get(options.experiment, {}).get("test_only", False)
    mode = options.mode or ("test" if test_only else MODE)
    if test_only and mode != "test":
        parser.error("Gaussian presets reuse the baseline checkpoint; use --mode test")
    if mode == "train" and options.start_epoch is not None:
        parser.error("--start-epoch only applies to testing")
    overrides = {key: getattr(options, key) for key in ("device", "batch_size", "workers", "seed")}
    with experiment_config(options.experiment, options.full_data, overrides):
        args = training_args()
        if args.workers < 0:
            parser.error("workers must be nonnegative")
        start_epoch = options.start_epoch if options.start_epoch is not None else TEST_CONFIG["start_epoch"]
        if options.dry_run:
            print(json.dumps({"model": MODEL, "mode": mode, "training": vars(args),
                              "test_overrides": TEST_CONFIG, "test_start_epoch": start_epoch,
                              "summary": str(Path(RUNS_ROOT) / MODEL / "test_summary.csv")}, indent=2))
            return
        if mode in ("train", "train_test"):
            cli.seed_all(args.seed)
            cli.train(args)  # Always saves epoch_1.pt through epoch_<epochs>.pt.
        if mode in ("test", "train_test"):
            test_checkpoints(args, start_epoch)


if __name__ == "__main__":
    main()
