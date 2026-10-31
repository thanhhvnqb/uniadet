"""Run the editable configuration in config.py: python train_test.py --dry-run."""
import argparse
import json
from pathlib import Path

from config import MODE, MODEL, RUNS_ROOT, TRAIN_DATASET, TRAIN_CONFIG, TEST_CONFIG
from uniadet import cli, workflow


def run_config():
    return workflow.RunConfig(MODEL, TRAIN_CONFIG, TEST_CONFIG, RUNS_ROOT, TRAIN_DATASET)


def training_args():
    return workflow.training_args(run_config())


def test_checkpoints(args, start_epoch, *, config=None):
    return workflow.test_checkpoints(config or run_config(), args, start_epoch)


test_checkpoints.__test__ = False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "test", "train_test"),
                        help="Default: configured MODE")
    parser.add_argument("--start-epoch", type=int, help="First checkpoint epoch to test only")
    parser.add_argument("--dry-run", action="store_true", help="Print planned settings without training or evaluation")
    parser.add_argument("--device", help="Override execution device for this invocation")
    parser.add_argument("--batch-size", type=int, help="Override batch size; keep identical across compared runs")
    parser.add_argument("--workers", type=int, help="Override data-loader workers")
    parser.add_argument("--seed", type=int, help="Override seed for this invocation")
    options = parser.parse_args(argv)
    mode = options.mode or MODE
    if mode == "train" and options.start_epoch is not None:
        parser.error("--start-epoch only applies to testing")
    overrides = {key: getattr(options, key) for key in ("device", "batch_size", "workers", "seed")}
    config = run_config()
    for key, value in overrides.items():
        if value is not None:
            config.train[key] = value
            config.test[key] = value
    args = workflow.training_args(config)
    if args.workers < 0:
        parser.error("workers must be nonnegative")
    start_epoch = options.start_epoch if options.start_epoch is not None else config.test["start_epoch"]
    if options.dry_run:
        print(json.dumps({"model": MODEL, "mode": mode, "training": vars(args),
                          "test_overrides": config.test, "test_start_epoch": start_epoch,
                          "summary": str(Path(RUNS_ROOT) / MODEL / "test_summary.csv")}, indent=2))
        return
    if mode in ("train", "train_test"):
        cli.seed_all(args.seed)
        cli.train(args)  # Always saves epoch_1.pt through epoch_<epochs>.pt.
    if mode in ("test", "train_test"):
        test_checkpoints(args, start_epoch, config=config)


if __name__ == "__main__":
    main()
