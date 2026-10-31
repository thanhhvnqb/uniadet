"""AdaptCLIP-derived ablations retain old checkpoints and isolate experiments."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch import nn
from torchvision.transforms import functional as TF

import train_test as dev
from uniadet import cli
from uniadet.data import AnomalyDataset
from uniadet.model import DecoupledHead, UniADet, smooth_maps


def test_softmax_before_resize_matches_probability_and_gradient_reference():
    head = DecoupledHead([2], temperature=0.25, softmax_before_resize=True)
    with torch.no_grad():
        head.seg[0].copy_(torch.eye(2))
    patches = torch.eye(2)[None]
    features = [(torch.ones(1, 2), patches, (1, 2))]
    _, local = head(features, (1, 7))
    actual = local[0].softmax(1)
    raw = F.linear(F.normalize(patches, dim=-1), F.normalize(head.seg[0], dim=-1)) / 0.25
    raw = raw.transpose(1, 2).reshape(1, 2, 1, 2)
    reference = F.interpolate(raw.softmax(1), (1, 7), mode="bilinear", align_corners=False)
    legacy = F.interpolate(raw, (1, 7), mode="bilinear", align_corners=False).softmax(1)
    torch.testing.assert_close(actual, reference)
    assert (actual-legacy).abs().max() > 0.1
    weights = torch.arange(14).reshape(1, 2, 1, 7).float()
    expected_grad = torch.autograd.grad((reference*weights).sum(), head.seg[0], retain_graph=True)[0]
    actual_grad = torch.autograd.grad((actual*weights).sum(), head.seg[0])[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)


def test_probability_resizing_remains_finite_for_saturated_logits():
    head = DecoupledHead([2], temperature=1e-5, softmax_before_resize=True)
    with torch.no_grad():
        head.seg[0].copy_(torch.eye(2))
    _, local = head([(torch.ones(1, 2), torch.eye(2)[None], (1, 2))], (1, 7))
    assert torch.isfinite(local[0]).all()
    local[0].softmax(1)[:, 1].sum().backward()
    assert torch.isfinite(head.seg[0].grad).all()


def test_spatial_smoothing_matches_scipy_without_mixing_images_or_layers():
    maps = torch.zeros(2, 3, 9, 11)
    maps[0, 0, 4, 5] = 1.
    maps[1, 2, 0, 0] = 0.8
    expected = np.stack([gaussian_filter(item.numpy(), sigma=1.5, mode="reflect")
                         for item in maps.flatten(0, 1)]).reshape(maps.shape)
    torch.testing.assert_close(smooth_maps(maps, 1.5), torch.from_numpy(expected), rtol=0., atol=0.)
    assert smooth_maps(maps, 0.) is maps
    assert smooth_maps(maps, 1.5)[0, 1].sum() == 0
    with pytest.raises(ValueError, match="gaussian_sigma"):
        smooth_maps(maps, float("nan"))


class FixedEncoder(nn.Module):
    dims = [2, 2]

    def forward(self, images):
        return [(torch.ones(len(images), 2), torch.ones(len(images), 9, 2), (3, 3))]*2


@pytest.mark.parametrize("pooling", ["layer", "map"])
@pytest.mark.parametrize("few_shot", [False, True])
def test_gaussian_is_applied_before_spatial_pooling_and_after_reference_fusion(pooling, few_shot):
    model = UniADet(FixedEncoder(), lambda_p=1., lambda_f=0.4)
    local = [torch.zeros(1, 2, 3, 3), torch.zeros(1, 2, 3, 3)]
    local[0][0, 1, 1, 1] = 5.
    local[1][0, 1, 0, 0] = 3.
    layers = torch.stack([item.softmax(1)[:, 1] for item in local])
    reference = torch.arange(9).float().reshape(1, 3, 3) / 9.
    class Memory:
        def predict(self, key, features, size):
            return reference
    raw_map = layers.mean(0)
    if few_shot:
        raw_map = 0.6*raw_map + 0.4*reference
    expected_map = torch.from_numpy(gaussian_filter(raw_map[0].numpy(), sigma=1.))[None]
    if not few_shot and pooling == "layer":
        expected_peak = torch.stack([torch.from_numpy(gaussian_filter(item[0].numpy(), sigma=1.)).max()
                                     for item in layers]).mean()[None]
    else:
        expected_peak = expected_map.flatten(1).amax(1)
    with patch.object(model.head, "forward", return_value=([torch.zeros(1, 2)]*2, local)):
        score, amap = model.predict(torch.ones(1, 3, 3, 3), memory=Memory() if few_shot else None,
                                    zero_score_pooling=pooling, gaussian_sigma=1.)
    torch.testing.assert_close(amap, expected_map)
    torch.testing.assert_close(score, expected_peak)


def test_bilinear_masks_match_torchvision_resize_and_half_threshold(tmp_path):
    pixels = np.zeros((8, 8), dtype=np.uint8)
    pixels[1, 1] = 255
    Image.new("RGB", (8, 8)).save(tmp_path / "image.png")
    Image.fromarray(pixels).save(tmp_path / "mask.png")
    (tmp_path / "meta.json").write_text(json.dumps({"test": {"a": [dict(
        img_path="image.png", mask_path="mask.png", anomaly=1)]}}))
    bilinear = AnomalyDataset(tmp_path, image_size=3, mask_interpolation="bilinear")[0]
    expected = (TF.to_tensor(TF.resize(Image.fromarray(pixels), (3, 3))) > 0.5).float()[0]
    nearest = AnomalyDataset(tmp_path, image_size=3)[0]
    torch.testing.assert_close(bilinear["mask"], expected)
    assert not torch.equal(bilinear["mask"], nearest["mask"])
    assert bilinear["label"] == nearest["label"] == 1
    training = AnomalyDataset(tmp_path, image_size=3, training=True, mask_interpolation="bilinear",
                              rotation=False, horizontal_flip=False, vertical_flip=False)[0]
    torch.testing.assert_close(training["mask"], nearest["mask"])


def test_legacy_defaults_and_new_settings_restore_with_override_protection():
    config = dict(backbone="clip", layers=[24], image_size=518, temperature=0.07)
    args = cli.parser().parse_args(["evaluate"])
    cli.restore_evaluation_settings(args, {"config": config})
    assert args.softmax_before_resize is False and args.gaussian_sigma == 0.
    assert args.mask_interpolation == "nearest"
    config.update(softmax_before_resize=True, mask_interpolation="bilinear", gaussian_sigma=4.)
    args = cli.parser().parse_args(["evaluate", "--gaussian-sigma", "0"])
    cli.restore_evaluation_settings(args, {"config": config})
    assert args.softmax_before_resize is True and args.mask_interpolation == "bilinear"
    assert args.gaussian_sigma == 0.
    args = cli.parser().parse_args(["evaluate", "--no-softmax-before-resize"])
    with pytest.raises(ValueError, match="retrain"):
        cli.restore_evaluation_settings(args, {"config": config})


@pytest.mark.parametrize("name,changed", [
    ("softmax-first", {"softmax_before_resize": True}),
    ("no-global-norm", {"clip_global_norm": False}),
    ("bilinear-pos", {"clip_pos_interpolation": "bilinear"}),
    ("smoothed-focal", {"focal_smoothing": 1e-5}),
    ("bilinear-mask", {"mask_interpolation": "bilinear"}),
])
def test_training_presets_change_only_one_factor_and_isolate_configs(name, changed):
    original = dev.MODEL, dev.TRAIN_CONFIG, dev.TEST_CONFIG, dev.TRAIN_DATASET
    with dev.experiment_config("baseline"):
        baseline = copy.deepcopy(dev.TRAIN_CONFIG)
        baseline_test = copy.deepcopy(dev.TEST_CONFIG)
    with dev.experiment_config(name):
        difference = {key: value for key, value in dev.TRAIN_CONFIG.items() if baseline[key] != value}
        assert difference == changed
        assert dev.TEST_CONFIG == baseline_test
        assert dev.TRAIN_CONFIG["exclude_categories"] is None
        assert dev.TEST_CONFIG["categories"] is None
        assert dev.MODEL == f"uniadet-final-{name}"
    assert dev.MODEL == original[0] and dev.TRAIN_CONFIG is original[1] and dev.TEST_CONFIG is original[2]


def test_gaussian_presets_reuse_one_baseline_and_match_other_settings():
    with dev.experiment_config("gaussian-off"):
        train = copy.deepcopy(dev.TRAIN_CONFIG)
        off = copy.deepcopy(dev.TEST_CONFIG)
    with dev.experiment_config("gaussian-on"):
        assert dev.TRAIN_CONFIG == train
        difference = {key: value for key, value in dev.TEST_CONFIG.items() if off[key] != value}
        assert difference == {"gaussian_sigma": 4.}
        assert dev.TEST_CONFIG["zero_score_pooling"] == "map"
        assert Path(dev.TEST_CONFIG["checkpoint"]).parts[-2:] == ("uniadet-final-baseline", "train_VisA")


def test_full_data_preset_clears_holdout_and_points_to_target():
    with dev.experiment_config("softmax-first", full_data=True):
        assert dev.MODEL == "uniadet-final-softmax-first"
        assert dev.TRAIN_CONFIG["exclude_categories"] is None
        assert dev.TRAIN_CONFIG["categories"] is None
        assert dev.TEST_CONFIG["data"] == ["datasets/MVTecAD"]
        assert dev.TEST_CONFIG["categories"] is None and dev.TEST_CONFIG["checkpoint"] is None
    with dev.experiment_config("gaussian-on", full_data=True):
        assert "uniadet-final-baseline" in dev.TEST_CONFIG["checkpoint"]


def test_dry_run_has_no_training_evaluation_or_filesystem_side_effects(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dev, "RUNS_ROOT", str(tmp_path / "runs"))
    with patch.object(dev.cli, "train") as train, patch.object(dev, "test_checkpoints") as evaluate:
        dev.main(["--experiment", "softmax-first", "--device", "cpu", "--workers", "0", "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["model"] == "uniadet-final-softmax-first"
    assert plan["training"]["softmax_before_resize"] is True
    assert plan["training"]["device"] == "cpu" and plan["training"]["workers"] == 0
    train.assert_not_called()
    evaluate.assert_not_called()
    assert not (tmp_path / "runs").exists()


def test_gaussian_commands_evaluate_without_retraining():
    with patch.object(dev.cli, "train") as train, patch.object(dev, "test_checkpoints") as evaluate:
        dev.main(["--experiment", "gaussian-on", "--device", "cpu"])
    train.assert_not_called()
    evaluate.assert_called_once()
    assert evaluate.call_args.args[0].model == "uniadet-final-gaussian-on"
    with pytest.raises(SystemExit):
        dev.main(["--experiment", "gaussian-on", "--mode", "train_test"])


def test_aux_holdout_is_explicit_opt_in(capsys):
    dev.main(["--experiment", "baseline", "--aux-holdout", "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["model"] == "uniadet-aux-baseline"
    assert plan["training"]["exclude_categories"] == ["pcb1", "pcb2"]
    assert plan["test_overrides"]["data"] == ["datasets/VisA"]
    assert plan["test_overrides"]["categories"] == ["pcb1", "pcb2"]
