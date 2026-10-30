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
    assert parser().parse_args(["train"]).rotation is True
    assert parser().parse_args(["train", "--rotation"]).rotation is True
    assert parser().parse_args(["train", "--no-rotation"]).rotation is False
    monkeypatch.setattr(dev, "TRAIN_CONFIG", dict(dev.TRAIN_CONFIG, rotation=False))
    assert dev.training_args().rotation is False


@pytest.mark.parametrize("name", ["horizontal_flip", "vertical_flip"])
def test_flips_can_be_toggled(monkeypatch, name):
    flag = name.replace("_", "-")
    assert getattr(parser().parse_args(["train"]), name) is True
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
