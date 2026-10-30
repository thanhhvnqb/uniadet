import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F

from uniadet.model import DecoupledHead, NormalMemory, UniADet
from uniadet.loss import training_loss
from uniadet.data import AnomalyDataset, normal_references
from uniadet.backbones import CLIPEncoder, DINOEncoder


class TinyEncoder(nn.Module):
    dims = [4, 4]

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = x.flatten(2).transpose(1, 2)
        return [(p.mean(1), p, x.shape[-2:]), (p.mean(1)*2, p*2, x.shape[-2:])]


def test_gradient_decoupling_and_frozen_encoder():
    model = UniADet(TinyEncoder()).train()
    assert not model.encoder.training
    images = torch.randn(2, 3, 8, 8)
    output = model(images)
    F.cross_entropy(output[0][0], torch.tensor([0, 1])).backward()
    assert model.head.cls[0].grad is not None
    assert model.head.cls[1].grad is None
    assert all(p.grad is None for p in model.head.seg)
    model.zero_grad()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=0.01)
    old = model.head.cls[0].detach().clone()
    loss = training_loss(model(images), torch.tensor([0, 1]), torch.randint(0, 2, (2, 8, 8)))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters())
    assert all(p.grad is None for p in model.encoder.parameters())
    optimizer.step()
    assert not torch.equal(old, model.head.cls[0])


def test_cosine_head_equation():
    head = DecoupledHead([2], temperature=0.5)
    with torch.no_grad():
        head.cls[0].copy_(torch.eye(2))
        head.seg[0].copy_(torch.eye(2))
    features = [(torch.tensor([[1., 0.]]), torch.eye(2)[None], (1, 2))]
    global_logits, local_logits = head(features, (1, 2))
    torch.testing.assert_close(global_logits[0], torch.tensor([[2., 0.]]))
    torch.testing.assert_close(local_logits[0], torch.tensor([[[[2., 0.]], [[0., 2.]]]]))


def test_memory_matches_exact_nn_and_is_class_specific():
    encoder = TinyEncoder()
    reference = encoder(torch.randn(2, 3, 8, 8))
    query = encoder(torch.randn(1, 3, 8, 8))
    memory = NormalMemory(chunk_size=3)
    memory.add(("a", "0"), reference)
    result = memory.predict(("a", "0"), query, (4, 4))
    q = F.normalize(query[0][1], dim=-1)
    m = F.normalize(reference[0][1].flatten(0, 1), dim=-1)
    expected = (1-(q @ m.T).amax(-1)).reshape(1, 4, 4)
    torch.testing.assert_close(result, expected)
    self_map = memory.predict(("a", "0"), reference, (4, 4))
    assert self_map.abs().max() < 1e-6
    with pytest.raises(KeyError):
        memory.predict(("b", "0"), query, (4, 4))


def test_inference_fusion_and_checkpoint(tmp_path):
    model = UniADet(TinyEncoder(), lambda_f=0.3, lambda_p=0.7).eval()
    x = torch.randn(1, 3, 8, 8)
    memory = NormalMemory()
    features = model.encoder(x)
    memory.add("a", features)
    g, p = model.head(features, (8, 8))
    raw_score = torch.stack([v.softmax(-1)[:, 1] for v in g]).mean(0)
    raw_map = torch.stack([v.softmax(1)[:, 1] for v in p]).mean(0)
    ref_map = memory.predict("a", features, (8, 8))
    zero_score, zero_map = model.predict(x)
    torch.testing.assert_close(zero_map, raw_map)
    layer_peak = torch.stack([v.softmax(1)[:, 1].flatten(1).amax(1) for v in p]).mean(0)
    torch.testing.assert_close(zero_score, 0.3*raw_score+0.7*layer_peak)
    score, amap = model.predict(x, memory, "a")
    torch.testing.assert_close(amap, 0.7*raw_map+0.3*ref_map)
    torch.testing.assert_close(score, 0.3*raw_score+0.7*amap.flatten(1).amax(1))
    torch.save(model.head.state_dict(), tmp_path/"head.pt")
    new = DecoupledHead([4, 4])
    new.load_state_dict(torch.load(tmp_path/"head.pt", weights_only=True))
    for a, b in zip(model.head.parameters(), new.parameters()):
        torch.testing.assert_close(a, b)


def test_zero_shot_pools_each_layer_before_averaging():
    model = UniADet(TinyEncoder(), lambda_p=1).eval()
    # The two levels put their peaks at opposite locations. Pooling the mean
    # map loses both peaks, whereas Sec. 3.5 retains each level's evidence.
    local = [torch.tensor([[[[0., 0.]], [[4., -4.]]]]),
             torch.tensor([[[[0., 0.]], [[-4., 4.]]]])]
    global_logits = [torch.zeros(1, 2), torch.zeros(1, 2)]
    with patch.object(model.head, "forward", return_value=(global_logits, local)):
        score, amap = model.predict(torch.zeros(1, 3, 2, 2))
        legacy, legacy_map = model.predict(torch.zeros(1, 3, 2, 2), zero_score_pooling="map")
    torch.testing.assert_close(amap, legacy_map)
    assert score.item() == pytest.approx(torch.sigmoid(torch.tensor(4.)).item())
    assert legacy.item() == pytest.approx(0.5)


@pytest.fixture
def dataset_root(tmp_path):
    rows = []
    for i in range(4):
        Image.new("RGB", (17, 11), (i*50, 0, 0)).save(tmp_path/f"{i}.png")
        mask = np.zeros((11, 17), dtype=np.uint8)
        if i >= 2:
            mask[2:6, 3:8] = 255
        Image.fromarray(mask).save(tmp_path/f"m{i}.png")
        rows.append({"img_path": f"{i}.png", "mask_path": f"m{i}.png", "anomaly": int(i>=2), "cls_name": "a"})
    (tmp_path/"meta.json").write_text(json.dumps({"train": {"a": rows[:2]}, "test": {"a": rows}}))
    return tmp_path


@pytest.mark.parametrize("mosaic", [True, False])
def test_class_aware_augmentation_preserves_labels(dataset_root, mosaic):
    dataset = AnomalyDataset(dataset_root, image_size=18, caa=True)
    for idx in range(4):
        image, mask, _ = dataset.raw(idx)
        with patch("uniadet.data.random.random", return_value=0.1 if mosaic else 0.9):
            image, mask = dataset.augment(idx, image, mask)
        assert image.size == mask.size
        assert bool(mask.getbbox()) == bool(idx >= 2)
    item = dataset[0]
    assert item["image"].shape == (3, 18, 18)
    assert item["mask"].shape == (18, 18)


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
@pytest.mark.parametrize("horizontal,vertical", [(False, False), (True, False), (False, True), (True, True)])
def test_geometry_keeps_image_and_groundtruth_aligned(angle, horizontal, vertical):
    pixels = np.zeros((5, 9), dtype=np.uint8)
    pixels[1:3, 2:5] = 255
    image = Image.fromarray(np.repeat(pixels[..., None], 3, axis=2))
    mask = Image.fromarray(pixels)
    expected = np.rot90(pixels, angle // 90)
    if horizontal:
        expected = np.fliplr(expected)
    if vertical:
        expected = np.flipud(expected)
    with patch("uniadet.data.random.choice", return_value=angle), patch(
        "uniadet.data.random.random", side_effect=[0.1 if horizontal else 0.9, 0.1 if vertical else 0.9]
    ):
        image, mask = AnomalyDataset.augment_geometry(image, mask)
    np.testing.assert_array_equal(np.array(mask), expected)
    np.testing.assert_array_equal(np.array(image)[..., 0], expected)
    assert np.count_nonzero(expected) == np.count_nonzero(pixels)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("rotation", [False, True])
def test_geometry_runs_only_during_training_without_caa(dataset_root, training, rotation):
    dataset = AnomalyDataset(dataset_root, image_size=17, training=training, rotation=rotation)
    with patch.object(dataset, "augment_geometry", wraps=dataset.augment_geometry) as geometry:
        item = dataset[2]
    assert geometry.call_count == int(training)
    if training:
        assert geometry.call_args.kwargs == {"rotation": rotation, "horizontal_flip": True, "vertical_flip": True}
    assert item["label"] == 1
    assert item["mask"].any()
    assert set(item["mask"].unique().tolist()) == {0., 1.}


@pytest.mark.parametrize("horizontal,vertical", [(False, False), (True, False), (False, True), (True, True)])
def test_flip_toggles(dataset_root, horizontal, vertical):
    dataset = AnomalyDataset(dataset_root, image_size=17, training=True, rotation=False,
                             horizontal_flip=horizontal, vertical_flip=vertical)
    with patch.object(dataset, "augment_geometry", wraps=dataset.augment_geometry) as geometry:
        dataset[2]
    assert geometry.call_args.kwargs == {
        "rotation": False, "horizontal_flip": horizontal, "vertical_flip": vertical}
    pixels = np.arange(45, dtype=np.uint8).reshape(5, 9)
    image = Image.fromarray(np.repeat(pixels[..., None], 3, axis=2))
    mask = Image.fromarray(pixels)
    expected = pixels
    if horizontal:
        expected = np.fliplr(expected)
    if vertical:
        expected = np.flipud(expected)
    with patch("uniadet.data.random.random", return_value=0.1) as draw:
        image, mask = dataset.augment_geometry(image, mask, rotation=False,
                                               horizontal_flip=horizontal, vertical_flip=vertical)
    assert draw.call_count == int(horizontal) + int(vertical)
    np.testing.assert_array_equal(np.array(mask), expected)
    np.testing.assert_array_equal(np.array(image)[..., 0], expected)


@pytest.mark.parametrize("horizontal,vertical", [(False, False), (True, False), (False, True), (True, True)])
def test_rotation_disabled_preserves_flips(horizontal, vertical):
    pixels = np.arange(45, dtype=np.uint8).reshape(5, 9)
    image = Image.fromarray(np.repeat(pixels[..., None], 3, axis=2))
    mask = Image.fromarray(pixels)
    expected = pixels
    if horizontal:
        expected = np.fliplr(expected)
    if vertical:
        expected = np.flipud(expected)
    with patch("uniadet.data.random.choice") as choice, patch(
        "uniadet.data.random.random", side_effect=[0.1 if horizontal else 0.9, 0.1 if vertical else 0.9]
    ):
        image, mask = AnomalyDataset.augment_geometry(image, mask, rotation=False)
    choice.assert_not_called()
    np.testing.assert_array_equal(np.array(mask), expected)
    np.testing.assert_array_equal(np.array(image)[..., 0], expected)


def test_references_train_only_and_without_replacement(dataset_root):
    a = normal_references(dataset_root, 2, 10)
    assert a == normal_references(dataset_root, 2, 10)
    assert len({x["img_path"] for x in a}) == 2
    assert all(x["anomaly"] == 0 for x in a)
    with pytest.raises(ValueError):
        normal_references(dataset_root, 3, 10)


def test_clip_multilevel_rectangular_features():
    from lib.clip import VisionTransformer
    visual = VisionTransformer(8, 2, 8, 2, 2, 4)
    # Original CLIP blocks return a tensor; same interface as unmodified AdaptCLIP blocks.
    encoder = CLIPEncoder(visual, [1, 2])
    pos = visual.positional_embedding.detach().clone()
    features = encoder(torch.randn(1, 3, 8, 12))
    assert len(features) == 2
    assert features[0][0].shape == (1, 4)
    assert features[0][1].shape == (1, 24, 4)
    assert features[0][2] == (4, 6)
    torch.testing.assert_close(pos, visual.positional_embedding)
    with pytest.raises(ValueError):
        CLIPEncoder(visual, [2, 1])


def test_dino_intermediate_contract():
    class FakeDINO(nn.Module):
        embed_dim = 4
        n_blocks = 24
        blocks = [None]
        def get_intermediate_layers(self, x, n, return_class_token, norm):
            assert n == [11, 23] and return_class_token and norm
            return [(torch.ones(1, 6, 4), torch.ones(1, 4)) for _ in n]
    features = DINOEncoder(FakeDINO(), [12, 24], 2)(torch.randn(1, 3, 4, 6))
    assert features[0][2] == (2, 3)


def test_training_and_evaluation_cli(dataset_root, tmp_path):
    from uniadet.cli import parser, seed_all, train, evaluate
    seed_all(10)
    output = tmp_path/"run"
    output.mkdir()
    args = parser().parse_args(["train", "--data", str(dataset_root), "--output", str(output), "--image-size", "8", "--epochs", "1", "--workers", "0", "--device", "cpu"])
    with patch("uniadet.cli.build", side_effect=lambda _: UniADet(TinyEncoder())):
        train(args)
        history = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
        assert history[-1]["kind"] == "epoch"
        assert history[-1]["samples"] == 4
        assert history[-1]["loss"] == pytest.approx(
            history[-1]["classification"] + history[-1]["segmentation"], rel=1e-6)
        assert (output / "training.csv").is_file()
        assert json.loads((output / "training_config.json").read_text())["epochs"] == 1
        args.checkpoint = str(output/"epoch_1.pt")
        args.shots = 1
        evaluate(args)
    report = json.loads((output/"metrics.json").read_text())
    assert 0 <= report["macro_mean"]["image"]["auroc"] <= 1
    assert len(json.loads((output/"predictions.json").read_text())) == 4
    assert len(json.loads((output/"references.json").read_text())) == 1


def test_adaptclip_checkpoint_loading(tmp_path):
    from lib.adaptclip import AdaptCLIP
    from uniadet.backbones import create_encoder
    # Exercise the real upstream loader with a small local checkpoint.
    upstream = AdaptCLIP(32, 8, 2, 64, 2, 8, 100, 64, 1, 2)
    checkpoint = tmp_path/"tiny_clip.pt"
    torch.save(upstream.state_dict(), checkpoint)
    encoder = create_encoder("clip", [1, 2], str(checkpoint))
    result = encoder(torch.randn(1, 3, 8, 12))
    assert result[1][1].shape == (1, 24, 32)
    assert not any(p.requires_grad for p in encoder.parameters())
    assert not hasattr(encoder, "token_embedding")


def test_classification_only_and_undefined_metrics(dataset_root):
    from uniadet.cli import metrics
    (dataset_root/"m2.png").unlink()
    data = AnomalyDataset(dataset_root, image_size=8)
    assert not data[2]["has_mask"]
    assert data[2]["label"] == 1
    train = AnomalyDataset(dataset_root, image_size=8, caa=True)
    with pytest.raises(ValueError):
        train[2]
    assert metrics([1, 1], [0.1, 0.8])["auroc"] is None


def test_paper_aupr_is_curve_area():
    from sklearn.metrics import auc, precision_recall_curve
    from uniadet.cli import metrics
    labels, scores = [0, 1, 0, 1], [0.9, 0.8, 0.7, 0.6]
    precision, recall, _ = precision_recall_curve(labels, scores)
    assert metrics(labels, scores)["aupr"] == pytest.approx(auc(recall, precision))


def test_aupro_region_balance_and_edge_cases():
    from uniadet.cli import aupro
    # A one-pixel and a three-pixel region must each receive half the weight.
    mask = np.array([[1, 0, 1, 1, 1]], dtype=bool)
    scores = np.array([[1., 0.5, 0., 0., 0.]])
    assert aupro([mask], [scores]) == pytest.approx(0.5)
    assert aupro([mask], [mask.astype(float)]) == pytest.approx(1.)
    assert aupro([mask], [np.ones_like(scores)]) == pytest.approx(0.15)
    assert aupro([np.zeros_like(mask)], [scores]) is None
    assert aupro([np.ones_like(mask)], [scores]) is None


@pytest.mark.parametrize("score_kind", ["continuous", "tied", "constant", "perfect", "reversed"])
def test_shared_pixel_sort_matches_reference_metrics(score_kind):
    from uniadet.cli import metrics, aupro, pixel_metrics
    rng = np.random.default_rng(42)
    masks = np.zeros((4, 13, 17), dtype=bool)
    masks[1, 2:5, 3:7] = True
    masks[1, 9, 12] = True
    masks[2, 1:11, 2:15] = True
    masks[3] = True
    scores = rng.random(masks.shape).astype(np.float32)
    if score_kind == "tied":
        scores = np.round(scores, 1)
    elif score_kind == "constant":
        scores[:] = 0.5
    elif score_kind == "perfect":
        scores = masks.astype(np.float32)
    elif score_kind == "reversed":
        scores = (~masks).astype(np.float32)
    expected = metrics(masks.ravel(), scores.ravel())
    expected["aupro"] = aupro(masks, scores)
    assert pixel_metrics(masks, scores) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("value", [False, True])
def test_shared_pixel_sort_undefined(value):
    from uniadet.cli import pixel_metrics
    masks = np.full((2, 3, 4), value, dtype=bool)
    assert pixel_metrics(masks, masks.astype(float)) == dict(
        auroc=None, aupr=None, f1max=None, aupro=None)


def test_paper_metric_exports(tmp_path):
    import csv
    from uniadet.cli import save_metrics
    image = {"auroc": 0.8, "aupr": 0.7, "f1max": 0.6}
    pixel = dict(image, aupro=0.5)
    report = {"a": {"image": image, "pixel": pixel},
              "macro_mean": {"image": image, "pixel": pixel},
              "unannotated": {"image": dict(image, auroc=0.4), "pixel": None}}
    save_metrics(report, tmp_path)
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["a"] == report["a"]
    assert saved["average"] == saved["macro_mean"]
    assert saved["average"]["image"]["auroc"] == pytest.approx(0.6)
    assert saved["average"]["pixel"]["aupro"] == pytest.approx(0.5)
    with (tmp_path / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == ["Category", "I-AUROC", "I-AUPR", "I-F1max", "P-AUROC", "P-AUPR", "P-F1max", "P-AUPRO"]
    assert float(rows[0]["P-AUPRO"]) == 50.
    assert rows[-2]["P-AUPRO"] == ""
    assert rows[-1]["Category"] == "Average"
    assert float(rows[-1]["P-AUPRO"]) == pytest.approx(50.)


@pytest.mark.parametrize("shots", [0, 1])
def test_batched_evaluation_matches_single_image(dataset_root, tmp_path, shots):
    from uniadet.cli import parser, evaluate
    # Constant images create tied pixel scores whose ranking can change under
    # harmless float32 batch-kernel rounding. Use spatially varied inputs to
    # check ranking metrics, and compare the dense predictions directly below.
    rng = np.random.default_rng(123)
    for index in range(4):
        pixels = rng.integers(0, 256, size=(11, 17, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(dataset_root / f"{index}.png")
    metadata = json.loads((dataset_root / "meta.json").read_text())
    for split in metadata.values():
        for index, row in enumerate(split["a"]):
            row["view_id"] = str(index % 2)
    (dataset_root / "meta.json").write_text(json.dumps(metadata))
    args = parser().parse_args(["evaluate", "--data", str(dataset_root),
                               "--image-size", "8", "--workers", "0", "--device", "cpu"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = UniADet(TinyEncoder()).eval()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"config": vars(args), "head": model.head.state_dict()}, checkpoint)
    args.checkpoint, args.shots = str(checkpoint), shots
    reports, predictions, dense_predictions = [], [], []
    original_predict = model.predict

    def capture_predictions(*args, **kwargs):
        scores, maps = original_predict(*args, **kwargs)
        dense_predictions[-1].extend(maps.cpu().clone().unbind())
        return scores, maps

    for size in (1, 3):
        output = tmp_path / f"batch_{size}"
        output.mkdir()
        args.batch_size, args.output = size, str(output)
        dense_predictions.append([])
        with patch("uniadet.cli.build", return_value=model), patch.object(
            model, "predict", side_effect=capture_predictions
        ):
            evaluate(args)
        reports.append(json.loads((output / "metrics.json").read_text()))
        predictions.append(json.loads((output / "predictions.json").read_text()))
    torch.testing.assert_close(torch.stack(dense_predictions[0]),
                               torch.stack(dense_predictions[1]), rtol=1e-5, atol=1e-6)
    for task in ("image", "pixel"):
        assert reports[0]["a"][task] == pytest.approx(reports[1]["a"][task], abs=1e-6)
    assert [row["path"] for row in predictions[0]] == [row["path"] for row in predictions[1]]
    assert [row["score"] for row in predictions[0]] == pytest.approx(
        [row["score"] for row in predictions[1]], abs=1e-6)


def test_evaluation_restores_encoder_source(dataset_root, tmp_path):
    from uniadet.cli import parser, evaluate
    args = parser().parse_args(["evaluate", "--data", str(dataset_root),
                               "--output", str(tmp_path/"result"), "--workers", "0",
                               "--device", "cpu"])
    model = UniADet(TinyEncoder()).eval()
    config = dict(vars(args), weights="training_backbone.pt", dino_repo="training_repo", image_size=8)
    checkpoint = tmp_path/"checkpoint.pt"
    torch.save({"config": config, "head": model.head.state_dict(), "epoch": 15}, checkpoint)
    args.checkpoint = str(checkpoint)
    with patch("uniadet.cli.build", return_value=model):
        evaluate(args)
    assert args.weights == "training_backbone.pt"
    assert args.dino_repo == "training_repo"
    saved = json.loads((Path(args.output)/"evaluation_config.json").read_text())
    assert saved["checkpoint_epoch"] == 15
    assert saved["evaluation"]["weights"] == "training_backbone.pt"
    args.weights = "explicit_override.pt"
    with patch("uniadet.cli.build", return_value=model):
        evaluate(args)
    assert args.weights == "explicit_override.pt"



def test_dev_runner_saves_from_epoch_one(dataset_root, tmp_path, monkeypatch):
    import train_test as dev
    monkeypatch.setattr(dev, "RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setattr(dev, "MODEL", "uniadetv1")
    monkeypatch.setattr(dev, "TRAIN_CONFIG", dict(
        dev.TRAIN_CONFIG, data=str(dataset_root), image_size=8, epochs=2, workers=0, device="cpu"))
    with patch("uniadet.cli.build", side_effect=lambda _: UniADet(TinyEncoder())), patch.object(
        dev, "test_checkpoints") as test:
        dev.main(["--mode", "train_test", "--start-epoch", "2"])
    args, start_epoch = test.call_args.args
    assert start_epoch == 2
    assert (Path(args.output) / "epoch_1.pt").is_file()
    assert (Path(args.output) / "epoch_2.pt").is_file()
