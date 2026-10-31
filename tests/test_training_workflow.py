import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import train_test as dev
from uniadet.cli import parser


def test_legacy_defaults_are_unchanged():
    args = parser().parse_args(["train"])
    assert args.output == "runs/uniadet"
    assert args.epochs == 15
    assert not hasattr(args, "start_epoch")
    assert not hasattr(args, "model")


def test_rotation_can_be_toggled(monkeypatch):
    assert parser().parse_args(["train"]).rotation is False
    assert parser().parse_args(["train", "--rotation"]).rotation is True
    assert parser().parse_args(["train", "--no-rotation"]).rotation is False
    monkeypatch.setattr(dev, "TRAIN_CONFIG", dict(dev.TRAIN_CONFIG, rotation=False))
    assert dev.training_args().rotation is False


@pytest.mark.parametrize("name", ["horizontal_flip", "vertical_flip"])
def test_flips_can_be_toggled(monkeypatch, name):
    flag = name.replace("_", "-")
    assert getattr(parser().parse_args(["train"]), name) is False
    assert getattr(parser().parse_args(["train", f"--{flag}"]), name) is True
    assert getattr(parser().parse_args(["train", f"--no-{flag}"]), name) is False
    monkeypatch.setattr(dev, "TRAIN_CONFIG", dict(dev.TRAIN_CONFIG, **{name: False}))
    assert getattr(dev.training_args(), name) is False


def test_dev_paths_and_sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(dev, "MODEL", "uniadetv1")
    monkeypatch.setattr(dev, "RUNS_ROOT", str(tmp_path))
    monkeypatch.setattr(dev, "TRAIN_CONFIG", dict(dev.TRAIN_CONFIG, data="datasets/visa"))
    args = dev.training_args()
    assert args.output == str(tmp_path / "uniadetv1/train_visa")
    source = Path(args.output)
    source.mkdir(parents=True)
    for epoch in (1, 2, 10):
        (source / f"epoch_{epoch}.pt").touch()
    monkeypatch.setattr(dev, "TEST_CONFIG", dict(data=["datasets/mvtec"], shots=[0, 1], checkpoint=None))
    def fake_evaluate(current):
        output = Path(current.output)
        output.mkdir(parents=True, exist_ok=True)
        epoch = int(Path(current.checkpoint).stem[6:])
        (output / "metrics.json").write_text(json.dumps({"average": {
            "image": {"auroc": epoch / 10, "aupr": 0.7, "f1max": 0.6},
            "pixel": None if current.shots else {"auroc": 0.9, "aupro": 0.5}}}))

    with patch("uniadet.cli.evaluate", side_effect=fake_evaluate) as evaluate:
        dev.test_checkpoints(args, 2)
    calls = [call.args[0] for call in evaluate.call_args_list]
    assert [Path(call.checkpoint).name for call in calls] == [
        "epoch_2.pt", "epoch_2.pt", "epoch_10.pt", "epoch_10.pt"]
    assert calls[0].output == str(tmp_path / "uniadetv1/mvtec_0shot/epoch_2")
    assert calls[1].output == str(tmp_path / "uniadetv1/mvtec_1shot/epoch_2")
    with (tmp_path / "uniadetv1/test_summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["Epoch"] for row in rows] == ["2", "2", "10", "10"]
    assert [row["Shots"] for row in rows] == ["0", "1", "0", "1"]
    assert float(rows[0]["I-AUROC"]) == 20
    assert float(rows[-1]["I-AUROC"]) == 100
    assert float(rows[0]["P-AUPRO"]) == 50
    assert rows[1]["P-AUPRO"] == ""
    with pytest.raises(ValueError, match="No checkpoints"):
        dev.test_checkpoints(args, 11)


def test_start_epoch_only_affects_test():
    with patch.object(dev.cli, "train") as train, patch.object(dev, "test_checkpoints") as test:
        dev.main(["--mode", "train_test", "--start-epoch", "5"])
    args = train.call_args.args[0]
    assert not hasattr(args, "start_epoch")
    test.assert_called_once_with(args, 5)
    with pytest.raises(SystemExit):
        dev.main(["--mode", "train", "--start-epoch", "5"])


def test_train_mode_without_start_epoch():
    with patch.object(dev.cli, "train") as train, patch.object(dev, "test_checkpoints") as test:
        dev.main(["--mode", "train"])
    train.assert_called_once()
    test.assert_not_called()


@pytest.mark.parametrize("epochs,expected", [([15], [15]), ([2, 15], [2, 15]), (None, [1, 2, 15, 20])])
def test_config_selects_checkpoint_epochs(tmp_path, monkeypatch, epochs, expected):
    monkeypatch.setattr(dev, "RUNS_ROOT", str(tmp_path))
    source = tmp_path / "checkpoints"
    source.mkdir()
    for epoch in (1, 2, 15, 20):
        (source / f"epoch_{epoch}.pt").touch()
    monkeypatch.setattr(dev, "TEST_CONFIG", dict(
        data=["datasets/mvtec"], shots=[0], checkpoint=str(source), epochs=epochs))
    seen = []
    def evaluate(current):
        seen.append(int(Path(current.checkpoint).stem[6:]))
        output = Path(current.output)
        output.mkdir(parents=True)
        (output / "metrics.json").write_text(json.dumps({"average": {"image": {}, "pixel": {}}}))
    with patch.object(dev.cli, "evaluate", side_effect=evaluate):
        dev.test_checkpoints(dev.training_args(), 1)
    assert seen == expected


@pytest.mark.parametrize("epochs", [[], [0], [True], [1.5], "15", 15])
def test_invalid_checkpoint_epoch_list(tmp_path, monkeypatch, epochs):
    monkeypatch.setattr(dev, "TEST_CONFIG", dict(data=["target"], shots=[0], checkpoint=str(tmp_path), epochs=epochs))
    with pytest.raises(ValueError, match="TEST_CONFIG.*epochs"):
        dev.test_checkpoints(dev.training_args(), 1)


def test_missing_selected_checkpoint_is_reported(tmp_path, monkeypatch):
    (tmp_path / "epoch_10.pt").touch()
    monkeypatch.setattr(dev, "TEST_CONFIG", dict(data=["target"], shots=[0], checkpoint=str(tmp_path), epochs=[15]))
    with pytest.raises(ValueError, match="Missing requested checkpoint epochs.*15"):
        dev.test_checkpoints(dev.training_args(), 1)


def test_single_checkpoint_ignores_epoch_filters_and_uses_test_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(dev, "RUNS_ROOT", str(tmp_path))
    source = tmp_path / "epoch_3.pt"
    source.touch()
    monkeypatch.setattr(dev, "TEST_CONFIG", dict(
        data=["target"], shots=[0], checkpoint=str(source), epochs=[15],
        batch_size=4, workers=0, seed=123, device="cpu",
        lambda_p=0.5, weights=None, dino_repo=None))
    args = dev.training_args()
    args.weights, args.dino_repo = "current_training.pt", "current_training_repo"
    def evaluate(current):
        assert current.checkpoint == str(source)
        assert (current.batch_size, current.workers, current.seed, current.device) == (4, 0, 123, "cpu")
        assert current.weights is None and current.dino_repo is None
        assert current.lambda_p == 0.5
        output = Path(current.output)
        output.mkdir(parents=True)
        (output / "metrics.json").write_text(json.dumps({"average": {"image": {}, "pixel": {}}}))
        (output / "evaluation_config.json").write_text(json.dumps({"checkpoint_epoch": 3}))
    with patch.object(dev.cli, "evaluate", side_effect=evaluate), patch.object(dev.cli, "seed_all") as seed:
        dev.test_checkpoints(args, 15)
    seed.assert_called_once_with(123)
    with (tmp_path / dev.MODEL / "test_summary.csv").open() as stream:
        assert list(csv.DictReader(stream))[0]["Epoch"] == "3"


def test_selected_defaults_and_legacy_checkpoint_fallback():
    from uniadet import cli
    args = cli.parser().parse_args(["train"])
    cli.resolve_training_settings(args)
    assert args.softmax_before_resize is True
    assert args.gaussian_sigma == 4. and args.zero_score_pooling == "layer"
    assert not args.rotation and not args.horizontal_flip and not args.vertical_flip
    runner = dev.training_args()
    assert runner.softmax_before_resize is True and runner.gaussian_sigma == 4.
    assert dev.TEST_CONFIG["epochs"] == [15]
    old = cli.parser().parse_args(["evaluate"])
    cli.restore_evaluation_settings(old, {"config": {"backbone": "clip"}})
    assert old.softmax_before_resize is False and old.gaussian_sigma == 0.


def test_cli_entrypoints_use_selected_defaults_and_dataset_paths():
    from uniadet import cli
    with patch.object(cli, "train") as train:
        cli.main(["train"])
    args = train.call_args.args[0]
    cli.resolve_training_settings(args)
    assert args.data == "datasets/VisA"
    assert args.softmax_before_resize and args.gaussian_sigma == 4.
    with patch.object(cli, "evaluate") as evaluate:
        cli.main(["evaluate", "--checkpoint", "saved.pt"])
    assert evaluate.call_args.args[0].data == "datasets/MVTecAD"
    assert evaluate.call_args.args[0].gaussian_sigma is None
