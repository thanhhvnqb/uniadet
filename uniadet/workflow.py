"""Train/evaluate orchestration with explicit, invocation-local configuration."""
import copy
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import cli
from .evaluation_metrics import METRIC_COLUMNS


@dataclass
class RunConfig:
    model: str
    train: dict = field(default_factory=dict)
    test: dict = field(default_factory=dict)
    runs_root: str = "runs"
    train_dataset: str | None = None

    def __post_init__(self):
        self.train = copy.deepcopy(self.train)
        self.test = copy.deepcopy(self.test)


def output_path(config, data, *, training=False, shots=0):
    dataset = (config.train_dataset or Path(data).name) if training else Path(data).name
    for name in (config.model, dataset):
        if name in ("", ".", "..") or "/" in name or "\\" in name:
            raise ValueError("Model and dataset names must be simple directory names")
    return Path(config.runs_root) / config.model / (f"train_{dataset}" if training else f"{dataset}_{shots}shot")


def training_args(config):
    args = cli.parser().parse_args(["train"])
    unknown = config.train.keys() - vars(args).keys()
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    vars(args).update(copy.deepcopy(config.train))
    args.data = args.data or "datasets/VisA"
    args.device = args.device or ("cuda" if cli.torch.cuda.is_available() else "cpu")
    args.image_size = args.image_size or (512 if args.backbone == "dinov3" else 518)
    cli.resolve_training_settings(args)
    args.output = str(output_path(config, args.data, training=True))
    args.model = config.model
    if args.epochs < 1 or args.batch_size < 1 or args.log_interval < 1:
        raise ValueError("epochs, batch_size and log_interval must be positive")
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    return args


def test_checkpoints(config, args, start_epoch):
    if start_epoch < 1 or any(shots < 0 for shots in config.test["shots"]):
        raise ValueError("start_epoch must be positive and shots nonnegative")
    source = Path(config.test["checkpoint"] or args.output)
    if source.is_file():
        # An explicit file is used directly, regardless of epoch filters.
        checkpoints = [(None, source)]
    else:
        requested = config.test.get("epochs")
        if requested is not None and (not isinstance(requested, (list, tuple)) or not requested or any(
            not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1 for epoch in requested
        )):
            raise ValueError("test configuration 'epochs' must be None or a nonempty list of positive integers")
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
        for target in config.test["data"]:
            for shots in config.test["shots"]:
                current = copy.deepcopy(args)
                current.mode, current.data, current.shots = "evaluate", target, shots,
                current.checkpoint = str(checkpoint)
                # Testing another run must use that checkpoint's representation
                # and fusion, rather than this file's current training settings.
                for name in (*cli.PREDICTION_DEFAULTS, "weights", "dino_repo", "categories", "exclude_categories"):
                    setattr(current, name, copy.deepcopy(config.test.get(name)))
                for name in cli.TRAINED_SETTING_DEFAULTS:
                    setattr(current, name, None)
                for name in ("batch_size", "workers", "device", "seed"):
                    value = config.test.get(name)
                    if value is not None:
                        setattr(current, name, value)
                if current.batch_size < 1 or current.workers < 0:
                    raise ValueError("Evaluation batch_size must be positive and workers nonnegative")
                output = output_path(config, target, shots=shots)
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
                for task, metric, column in METRIC_COLUMNS:
                    value = (average.get(task) or {}).get(metric)
                    row[column] = 100 * value if value is not None else ""
                summary.append(row)
    summary_path = Path(config.runs_root) / config.model / "test_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Dataset", "Shots", "Epoch", "Checkpoint"] +
                                [column for _, _, column in METRIC_COLUMNS])
        writer.writeheader()
        writer.writerows(summary)
    temporary.replace(summary_path)
    print(f"Saved epoch comparison: {summary_path}", flush=True)


# Prevent pytest from collecting the dev helper as a test.
test_checkpoints.__test__ = False

