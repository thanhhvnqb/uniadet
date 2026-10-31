"""Regression checks for supervision preservation and reproducible ablations."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from uniadet import cli
from uniadet.backbones import CLIPEncoder
from uniadet.data import AnomalyDataset, normal_references
from uniadet.loss import training_loss


@pytest.fixture
def tiny_dataset(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    rows = []
    rng = np.random.default_rng(43)
    for index in range(4):
        Image.fromarray(rng.integers(0, 256, (18, 18, 3), dtype=np.uint8)).save(root / f"{index}.png")
        mask = np.zeros((18, 18), dtype=np.uint8)
        if index >= 2:
            mask[0, 0] = 255
        Image.fromarray(mask).save(root / f"mask{index}.png")
        rows.append(dict(img_path=f"{index}.png", mask_path=f"mask{index}.png", anomaly=int(index >= 2)))
    (root / "meta.json").write_text(json.dumps({
        "train": {"a": rows[:2], "b": rows[:1]},
        "test": {"a": rows, "b": rows[:1]},
    }))
    return root


@pytest.mark.parametrize("preserve", [True, False])
def test_tiny_mosaic_defect_falls_back_to_candidate(tiny_dataset, preserve):
    dataset = AnomalyDataset(
        tiny_dataset, image_size=18, training=True, caa=True, caa_probability=1.,
        caa_grids=(3,), mosaic_probability=1., rotation=False,
        horizontal_flip=False, vertical_flip=False, preserve_anomaly_mask=preserve)
    # The one-pixel defect survives full-size resizing but vanishes at 6x6.
    # All other tiles are normal, so this used to produce label=1, mask=0.
    with patch("uniadet.data.random.choices", return_value=[0]*8), patch(
        "uniadet.data.random.shuffle", side_effect=lambda items: None
    ):
        item = dataset[2]
    assert item["label"] == 1 and item["has_mask"]
    assert bool(item["mask"].any()) == preserve
    if preserve:
        original = AnomalyDataset(tiny_dataset, image_size=18)[2]
        torch.testing.assert_close(item["image"], original["image"])
        torch.testing.assert_close(item["mask"], original["mask"])


def test_mask_guard_rejects_unusable_source_but_leaves_evaluation_unchanged(tiny_dataset):
    training = AnomalyDataset(tiny_dataset, image_size=1, training=True,
                              rotation=False, horizontal_flip=False, vertical_flip=False)
    with pytest.raises(ValueError, match="Anomaly mask is empty"):
        training[2]
    evaluation = AnomalyDataset(tiny_dataset, image_size=1)
    assert evaluation[2]["label"] == 1
    assert evaluation[2]["mask"].sum() == 0


def test_category_holdout_excludes_caa_sources_and_reference_requirements(tiny_dataset):
    training = AnomalyDataset(tiny_dataset, exclude_categories=["b"], caa=True)
    assert set(training.groups) == {"a"}
    assert all(row["cls_name"] == "a" for row in training.records)
    heldout = AnomalyDataset(tiny_dataset, categories=["b"])
    assert set(heldout.groups) == {"b"}
    # b has only one reference; its count must not prevent a's 2-shot test.
    references = normal_references(tiny_dataset, 2, 10, categories=["a"])
    assert len(references) == 2
    assert all(row["cls_name"] == "a" for row in references)
    with pytest.raises(ValueError, match="Unknown categories"):
        AnomalyDataset(tiny_dataset, exclude_categories=["typo"])
    with pytest.raises(ValueError, match="No records"):
        AnomalyDataset(tiny_dataset, exclude_categories=["a", "b"])


def test_clip_global_norm_flag_changes_only_global_tokens():
    from lib.clip import VisionTransformer
    torch.manual_seed(19)
    visual = VisionTransformer(8, 2, 8, 2, 2, 4).eval()
    images = torch.randn(2, 3, 8, 12)
    captured = []
    hook = visual.transformer.resblocks[-1].register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach().transpose(0, 1)))
    try:
        normalized = CLIPEncoder(visual, [2], global_norm=True)(images)[0]
        raw = CLIPEncoder(visual, [2], global_norm=False)(images)[0]
    finally:
        hook.remove()
    expected = visual.ln_post(captured[0]) @ visual.proj
    torch.testing.assert_close(normalized[0], expected[:, 0])
    torch.testing.assert_close(raw[0], captured[1][:, 0] @ visual.proj)
    torch.testing.assert_close(normalized[1], raw[1])
    assert not torch.allclose(normalized[0], raw[0])


def test_clip_positional_interpolation_preserves_pretrained_parameters():
    from lib.clip import VisionTransformer
    torch.manual_seed(27)
    visual = VisionTransformer(8, 2, 8, 2, 2, 4).eval()
    pos = visual.positional_embedding.detach().clone()
    images = torch.randn(1, 3, 8, 12)
    bicubic = CLIPEncoder(visual, [2], pos_interpolation="bicubic")(images)[0]
    bilinear = CLIPEncoder(visual, [2], pos_interpolation="bilinear")(images)[0]
    assert not torch.allclose(bicubic[1], bilinear[1])
    torch.testing.assert_close(pos, visual.positional_embedding)


def checkpoint_config(**overrides):
    return {"config": dict(backbone="clip", layers=[12, 15, 18, 21, 24],
                           image_size=518, temperature=0.07,
                           lambda_p=0.2, lambda_f=0.8, zero_score_pooling="map",
                           clip_global_norm=False, clip_pos_interpolation="bilinear",
                           head_init_std=0.02, **overrides)}


def test_restore_saved_fusion_and_representation_or_explicit_default_override():
    checkpoint = checkpoint_config()
    args = cli.parser().parse_args(["evaluate"])
    cli.restore_evaluation_settings(args, checkpoint)
    assert (args.lambda_p, args.lambda_f, args.zero_score_pooling) == (0.2, 0.8, "map")
    assert args.clip_global_norm is False
    assert args.clip_pos_interpolation == "bilinear"
    assert args.head_init_std == 0.02
    args = cli.parser().parse_args(["evaluate", "--lambda-p", "0.5", "--lambda-f", "0",
                                   "--zero-score-pooling", "layer"])
    cli.restore_evaluation_settings(args, checkpoint)
    assert (args.lambda_p, args.lambda_f, args.zero_score_pooling) == (0.5, 0., "layer")


@pytest.mark.parametrize("flag", [["--clip-global-norm"], ["--clip-pos-interpolation", "bicubic"]])
def test_evaluation_cannot_silently_change_trained_features(flag):
    args = cli.parser().parse_args(["evaluate", *flag])
    with pytest.raises(ValueError, match="retrain the head"):
        cli.restore_evaluation_settings(args, checkpoint_config())


def test_legacy_checkpoint_uses_original_defaults():
    args = cli.parser().parse_args(["evaluate"])
    legacy = {"config": dict(backbone="clip", layers=[24], image_size=518, temperature=0.07)}
    cli.restore_evaluation_settings(args, legacy)
    assert args.layers == [24]
    assert args.clip_global_norm is True
    assert args.clip_pos_interpolation == "bicubic"
    assert (args.lambda_p, args.lambda_f, args.zero_score_pooling) == (0.5, 0.5, "layer")
    # A pre-resolution Namespace serialized by a library caller can contain None.
    legacy["config"].update(lambda_p=None, clip_global_norm=None)
    args = cli.parser().parse_args(["evaluate"])
    cli.restore_evaluation_settings(args, legacy)
    assert args.lambda_p == 0.5 and args.clip_global_norm is True


@pytest.mark.parametrize("flags", [
    ["--lambda-p", "nan"], ["--caa-probability", "2"], ["--head-init-std", "0"],
    ["--focal-weight", "-1"], ["--focal-smoothing", "0.5"], ["--betas", "1", "0.9"],
    ["--backbone", "dinov2", "--no-clip-global-norm"],
])
def test_invalid_ablation_options_fail_before_model_loading(flags):
    args = cli.parser().parse_args(["train", *flags])
    with pytest.raises(ValueError):
        cli.resolve_training_settings(args)


def test_independent_segmentation_weights_and_anomalyclip_smoothing():
    from lib.loss import FocalLoss
    torch.manual_seed(3)
    logits = torch.randn(2, 2, 3, 4, requires_grad=True)
    outputs = ([torch.zeros(2, 2)], [logits])
    labels = torch.tensor([0, 1])
    masks = torch.randint(0, 2, (2, 3, 4)).float()
    common = dict(classification_weight=0., focal_smoothing=1e-5)
    focal = training_loss(outputs, labels, masks, normal_dice_weight=0., anomaly_dice_weight=0., **common)
    normal = training_loss(outputs, labels, masks, focal_weight=0., anomaly_dice_weight=0., **common)
    anomaly = training_loss(outputs, labels, masks, focal_weight=0., normal_dice_weight=0., **common)
    torch.testing.assert_close(focal, FocalLoss()(logits.softmax(1), masks))
    combined = training_loss(outputs, labels, masks, focal_weight=2., normal_dice_weight=3.,
                             anomaly_dice_weight=4., **common)
    torch.testing.assert_close(combined, 2*focal + 3*normal + 4*anomaly)
    combined.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


class TinyEncoder(nn.Module):
    dims = [4]

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 2, stride=2)

    def forward(self, images):
        patches = self.conv(images)
        tokens = patches.flatten(2).transpose(1, 2)
        return [(tokens.mean(1), tokens, patches.shape[-2:])]


@pytest.mark.parametrize("softmax_first,sigma,mask_mode", [(False, 0., "nearest"), (True, 1., "bilinear")])
def test_train_evaluate_and_audit_share_checkpoint_options(tiny_dataset, tmp_path, softmax_first, sigma, mask_mode):
    from scripts.audit_zero_shot import main as audit
    encoder = TinyEncoder()
    output = tmp_path / "train"
    args = cli.parser().parse_args([
        "train", "--data", str(tiny_dataset), "--output", str(output), "--image-size", "18",
        "--epochs", "1", "--workers", "0", "--batch-size", "2", "--device", "cpu",
        "--no-caa", "--no-rotation", "--no-horizontal-flip", "--no-vertical-flip",
        "--categories", "a", "--no-clip-global-norm", "--clip-pos-interpolation", "bilinear",
        "--head-init-std", "0.02", "--focal-smoothing", "1e-5", "--focal-weight", "2",
        "--lambda-p", "0.2", "--lambda-f", "0.8", "--zero-score-pooling", "map",
        "--softmax-before-resize" if softmax_first else "--no-softmax-before-resize",
        "--gaussian-sigma", str(sigma), "--mask-interpolation", mask_mode,
    ])
    with patch("uniadet.cli.create_encoder", side_effect=lambda *a, **kw: copy.deepcopy(encoder)) as factory:
        cli.train(args)
        config = json.loads((output / "training_config.json").read_text())
        assert config["preserve_anomaly_mask"] is True and config["focal_weight"] == 2.
        checkpoint = output / "epoch_1.pt"
        command = ["--data", str(tiny_dataset), "--checkpoint", str(checkpoint),
                   "--categories", "a", "--workers", "0", "--device", "cpu"]
        eval_args = cli.parser().parse_args(["evaluate", *command, "--output", str(tmp_path / "eval")])
        cli.evaluate(eval_args)
        audit([*command, "--output", str(tmp_path / "audit")])
        assert factory.call_args.kwargs == dict(clip_global_norm=False, clip_pos_interpolation="bilinear")
    effective = json.loads((tmp_path / "eval/evaluation_config.json").read_text())
    diagnostic = json.loads((tmp_path / "audit/score_audit.json").read_text())
    for key in ("lambda_p", "lambda_f", "zero_score_pooling", "clip_global_norm", "clip_pos_interpolation",
                "softmax_before_resize", "gaussian_sigma", "mask_interpolation"):
        assert effective["evaluation"][key] == config[key] == diagnostic["config"][key]
    report = json.loads((tmp_path / "eval/metrics.json").read_text())
    assert set(report) == {"a", "average", "macro_mean"}
    assert report["a"]["image"] == pytest.approx(diagnostic["categories"]["a"]["legacy_map_fusion"])


def test_dev_evaluation_does_not_inherit_current_training_ablation(tmp_path, monkeypatch):
    import train_test as dev
    monkeypatch.setattr(dev, "RUNS_ROOT", str(tmp_path))
    monkeypatch.setattr(dev, "MODEL", "test")
    source = tmp_path / "source"
    source.mkdir()
    (source / "epoch_15.pt").touch()
    monkeypatch.setattr(dev, "TEST_CONFIG", dict(data=["target"], shots=[0], checkpoint=str(source)))
    args = dev.training_args()
    args.exclude_categories = ["heldout_training_category"]
    def evaluate(current):
        assert current.clip_global_norm is None and current.clip_pos_interpolation is None
        assert current.lambda_p is None and current.zero_score_pooling is None
        assert current.categories is None and current.exclude_categories is None
        path = Path(current.output)
        path.mkdir(parents=True)
        (path / "metrics.json").write_text(json.dumps({"average": {"image": {}, "pixel": {}}}))
    with patch.object(dev.cli, "evaluate", side_effect=evaluate):
        dev.test_checkpoints(args, 15)


def test_prepare_only_visa_mvtec_requires_no_other_datasets(tmp_path, monkeypatch):
    from scripts import prepare_datasets as prepare
    visa = tmp_path / "datasets/VisA"
    (visa / "split_csv").mkdir(parents=True)
    for name in ("normal.png", "anomaly.png", "mask.png"):
        Image.new("RGB", (4, 4)).save(visa / name)
    (visa / "split_csv/1cls.csv").write_text(
        "object,split,label,image,mask\n"
        "candle,train,normal,normal.png,\n"
        "candle,test,normal,normal.png,\n"
        "candle,test,anomaly,anomaly.png,mask.png\n")
    mvtec = tmp_path / "datasets/MVTecAD"
    for name in ("bottle/train/good/000.png", "bottle/test/good/000.png",
                 "bottle/test/broken/001.png", "bottle/ground_truth/broken/001_mask.png"):
        path = mvtec / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(path)
    monkeypatch.setattr(prepare, "summary", {})
    output = tmp_path / "preparation"
    monkeypatch.setattr("sys.argv", ["prepare_datasets.py", "--datasets", "visa_mvtec",
                                    "--base", str(tmp_path / "datasets"), "--output-dir", str(output)])
    prepare.main()
    report = json.loads((output / "datasets.json").read_text())
    assert set(report) == {"visa", "mvtec"}
    assert report["visa"]["test"] == dict(normal=1, anomaly=1)
    assert report["mvtec"]["test"] == dict(normal=1, anomaly=1)
    assert (visa / "meta.json").is_file() and (mvtec / "meta.json").is_file()
