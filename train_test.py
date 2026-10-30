"""Dev runner. Edit configuration below, then run python train_test.py.

Legacy train.py/test.py CLI arguments and output paths remain unchanged.
"""
import argparse
import copy
import csv
import json
from pathlib import Path

from uniadet import cli

MODE = "train_test"  # train, test, or train_test (test saved checkpoints after training)
MODEL = "uniadet-norotate"  # Experiment name; e.g. uniadetv1 (same architecture)
RUNS_ROOT = "runs"
TRAIN_DATASET = None  # None uses the training data directory name

TRAIN_CONFIG = {
    "data": "datasets/VisA", "backbone": "clip",
    "weights": None, "dino_repo": None, "layers": [12, 15, 18, 21, 24],
    "image_size": None, "temperature": 0.07, "lambda_f": 0.5, "lambda_p": 0.5,
    "epochs": 20, "lr": 0.001, "betas": [0.5, 0.999], "weight_decay": 0.0,
    "batch_size": 8, "workers": 4, "seed": 10, "device": None,
    "no_caa": False, "split": "test", "log_interval": 50,
    "rotation": False,
    "horizontal_flip": False, "vertical_flip": False,  # Set False to disable each flip
    "caa_probability": 0.5, "caa_grids": [2, 3], "mosaic_probability": 0.5,
    "focal_gamma": 2.0, "dice_smooth": 1.0,
    "classification_weight": 1.0, "segmentation_weight": 1.0,
}

TEST_CONFIG = {
    "data": ["datasets/MVTecAD"],  # Target dataset roots
    "shots": [0],   # [0, 1, 2, 4],
    "start_epoch": 1,  # Test checkpoints from this epoch, inclusive
    "checkpoint": None,  # None uses runs/<MODEL>/train_<dataset>
}


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
        checkpoints = [(None, source)]
    else:
        checkpoints = sorted((int(p.stem[6:]), p) for p in source.glob("epoch_*.pt")
                             if p.stem[6:].isdigit() and int(p.stem[6:]) >= start_epoch)
    if not checkpoints:
        raise ValueError(f"No checkpoints at or after epoch {start_epoch} in {source}")
    summary = []
    for epoch, checkpoint in checkpoints:
        for target in TEST_CONFIG["data"]:
            for shots in TEST_CONFIG["shots"]:
                current = copy.deepcopy(args)
                current.mode, current.data, current.shots = "evaluate", target, shots,
                current.checkpoint = str(checkpoint)
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
    parser.add_argument("--mode", choices=("train", "test", "train_test"), default=MODE)
    parser.add_argument("--start-epoch", type=int, help="First checkpoint epoch to test only")
    options = parser.parse_args(argv)
    if options.mode == "train" and options.start_epoch is not None:
        parser.error("--start-epoch only applies to testing")
    args = training_args()
    if options.mode in ("train", "train_test"):
        cli.seed_all(args.seed)
        cli.train(args)  # Always saves epoch_1.pt through epoch_<epochs>.pt.
    if options.mode in ("test", "train_test"):
        test_checkpoints(args, options.start_epoch if options.start_epoch is not None
                         else TEST_CONFIG["start_epoch"])


if __name__ == "__main__":
    main()
