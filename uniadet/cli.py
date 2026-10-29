import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .backbones import create_encoder
from .data import AnomalyDataset, normal_references
from .loss import training_loss
from .model import UniADet, NormalMemory


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parser():
    p = argparse.ArgumentParser(description="UniADet training / zero- and few-shot evaluation")
    p.add_argument("mode", choices=("train", "evaluate"))
    p.add_argument("--data", type=str, default="dataset", help="Dataset root containing AdaptCLIP meta.json")
    p.add_argument("--backbone", choices=("clip", "dinov2", "dinov3"), default="clip")
    p.add_argument("--weights", help="Pretrained backbone weights (required for DINOv3)")
    p.add_argument("--dino-repo", help="Optional local official DINOv3 checkout")
    p.add_argument("--layers", type=int, nargs="+", default=[12, 15, 18, 21, 24])
    p.add_argument("--image-size", type=int)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--lambda-f", type=float, default=0.5)
    p.add_argument("--lambda-p", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="runs/uniadet")
    p.add_argument("--checkpoint")
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--zero-score-pooling", choices=("layer", "map"), default="layer",
                   help="Zero-shot spatial maximum per layer (Sec. 3.5), or legacy averaged map")
    p.add_argument("--no-caa", action="store_true")
    p.add_argument("--split", default="test", help="Auxiliary training split; paper uses test")
    return p


def build(args):
    encoder = create_encoder(args.backbone, args.layers, args.weights, args.dino_repo)
    return UniADet(encoder, args.temperature, args.lambda_f, args.lambda_p).to(args.device)


def train(args):
    model = build(args).train()
    dataset = AnomalyDataset(args.data, args.split, args.image_size, args.backbone, caa=not args.no_caa)
    # Explicitly reject incomplete supervision even when augmentation is disabled.
    for i, record in enumerate(dataset.records):
        if record["anomaly"] and not (dataset.root / record.get("mask_path", "")).is_file():
            raise ValueError(f"Missing training mask: {record['img_path']}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr, betas=(0.5, 0.999))
    print(f"Trainable parameters: {sum(p.numel() for p in model.head.parameters()):,}")
    for epoch in range(1, args.epochs+1):
        total = 0.
        started = time.monotonic()
        for step, batch in enumerate(loader, 1):
            outputs = model(batch["image"].to(args.device))
            loss = training_loss(outputs, batch["label"].to(args.device), batch["mask"].to(args.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch["label"])
            if step % 50 == 0 or step == len(loader):
                print(f"epoch {epoch}/{args.epochs} step {step}/{len(loader)} loss={loss.item():.6f} elapsed={time.monotonic()-started:.1f}s", flush=True)
        print(f"epoch {epoch}/{args.epochs} loss={total/len(dataset):.6f}", flush=True)
        torch.save({"head": model.head.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "config": vars(args)}, Path(args.output)/f"epoch_{epoch}.pt")


def metrics(labels, scores):
    from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
    if len(np.unique(labels)) < 2:
        return {"auroc": None, "aupr": None, "f1max": None}
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {"auroc": float(roc_auc_score(labels, scores)), "aupr": float(auc(recall, precision)), "f1max": float(np.max(2*precision*recall/np.maximum(precision+recall, 1e-12)))}


def aupro(masks, maps, fpr_limit=0.3):
    """Exact region-balanced PRO area, normalized over FPR [0, 0.3].

    Components use 8-connectivity within each image. Each region contributes
    equally regardless of its area; normal pixels determine the global FPR.
    """
    from scipy.ndimage import label
    from sklearn.metrics import auc, roc_curve

    if not 0 < fpr_limit <= 1:
        raise ValueError("fpr_limit must be in (0, 1]")
    weights = []
    for mask in masks:
        components, count = label(mask, structure=np.ones((3, 3)))
        areas = np.bincount(components.ravel())
        region_weights = np.zeros(count + 1, dtype=np.float64)
        region_weights[1:] = 1.0 / areas[1:]
        weights.append(region_weights[components].ravel())
    labels = np.concatenate([mask.ravel() for mask in masks])
    if not labels.any() or labels.all():
        return None
    weights = np.concatenate(weights)
    weights[~labels] = 1.0
    fpr, pro, _ = roc_curve(labels, np.concatenate([amap.ravel() for amap in maps]),
                            sample_weight=weights, drop_intermediate=False)
    # Retain vertical segments, and interpolate the boundary at the FPR limit.
    stop = np.searchsorted(fpr, fpr_limit, side="right")
    x, y = fpr[:stop], pro[:stop]
    if x[-1] < fpr_limit:
        y_limit = np.interp(fpr_limit, fpr[stop-1:stop+1], pro[stop-1:stop+1])
        x, y = np.append(x, fpr_limit), np.append(y, y_limit)
    return float(auc(x, y) / fpr_limit)


def save_metrics(report, output):
    """Store raw [0, 1] JSON values and paper-style percentage CSV rows."""
    categories = {name: result for name, result in report.items()
                  if name not in ("macro_mean", "average")}
    average = {}
    for task, names in (("image", ("auroc", "aupr", "f1max")),
                        ("pixel", ("auroc", "aupr", "f1max", "aupro"))):
        average[task] = {}
        for name in names:
            values = [(result.get(task) or {}).get(name) for result in categories.values()]
            values = [value for value in values if value is not None]
            average[task][name] = float(np.mean(values)) if values else None
    # Keep the original key for consumers of earlier reports.
    report = dict(categories, macro_mean=average, average=average)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    columns = [("image", "auroc", "I-AUROC"), ("image", "aupr", "I-AUPR"),
               ("image", "f1max", "I-F1max"), ("pixel", "auroc", "P-AUROC"),
               ("pixel", "aupr", "P-AUPR"), ("pixel", "f1max", "P-F1max"),
               ("pixel", "aupro", "P-AUPRO")]
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Category"] + [column for _, _, column in columns])
        for category, results in [*categories.items(), ("Average", average)]:
            values = [(results.get(task) or {}).get(metric) for task, metric, _ in columns]
            writer.writerow([category] + [100 * value if value is not None else "" for value in values])


@torch.inference_mode()
def evaluate(args):
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for evaluation")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    # Model/preprocessing must match training. Fusion weights may be overridden.
    for name in ("backbone", "layers", "image_size", "temperature"):
        setattr(args, name, config[name])
    for name in ("weights", "dino_repo"):
        if getattr(args, name) is None:
            setattr(args, name, config.get(name))
    # Preserve the effective settings, including the encoder source and pooling,
    # so comparisons cannot silently mix different evaluation configurations.
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output)/"evaluation_config.json").write_text(json.dumps(
        {"evaluation": vars(args), "training": config,
         "checkpoint_epoch": checkpoint.get("epoch")}, indent=2))
    model = build(args).eval()
    model.head.load_state_dict(checkpoint["head"])
    dataset = AnomalyDataset(args.data, "test", args.image_size, args.backbone)
    memory = None
    if args.shots:
        records = normal_references(args.data, args.shots, args.seed)
        (Path(args.output)/"references.json").write_text(json.dumps(records, indent=2))
        references = AnomalyDataset(args.data, image_size=args.image_size, backbone=args.backbone, records=records)
        memory = NormalMemory()
        for item in references:
            key = (item["category"], item["view"])
            memory.add(key, model.encoder(item["image"][None].to(args.device)))
    report, predictions = {}, []
    # Evaluate one class at a time to bound dense metric storage on large datasets.
    for category in sorted(dataset.groups):
        rows = [dataset.records[i] for i in dataset.groups[category]]
        subset = AnomalyDataset(args.data, image_size=args.image_size, backbone=args.backbone, records=rows)
        labels, scores, masks, maps = [], [], [], []
        # A few-shot batch must use one view's normal memory. Zero-shot
        # batches can mix views because no reference bank is consulted.
        groups = {}
        for index, row in enumerate(rows):
            view = str(row.get("view_id", "0")) if memory is not None else "all"
            groups.setdefault(view, []).append(index)
        batches = [indices[start:start + args.batch_size]
                   for indices in groups.values()
                   for start in range(0, len(indices), args.batch_size)]
        use_cuda = torch.device(args.device).type == "cuda"
        loader = DataLoader(subset, batch_sampler=batches, num_workers=args.workers,
                            pin_memory=use_cuda)
        # Restore dataset order when batches are grouped by reference view.
        category_predictions = {}
        for indices, batch in zip(batches, loader):
            key = (category, batch["view"][0])
            score, amap = model.predict(
                batch["image"].to(args.device, non_blocking=use_cuda), memory, key,
                zero_score_pooling=args.zero_score_pooling)
            batch_scores = score.cpu().tolist()
            batch_labels = batch["label"].tolist()
            valid = batch["has_mask"].bool()
            if valid.any():
                masks.extend(batch["mask"][valid].numpy().astype(bool))
                maps.extend(amap[valid.to(amap.device)].cpu().numpy())
            labels.extend(batch_labels)
            scores.extend(batch_scores)
            for index, path, value in zip(indices, batch["path"], batch_scores):
                category_predictions[index] = {"path": path, "category": category, "score": value}
        predictions.extend(category_predictions[index] for index in range(len(rows)))
        pixel = None
        if masks:
            pixel = metrics(np.concatenate([mask.ravel() for mask in masks]),
                            np.concatenate([amap.ravel() for amap in maps]))
            pixel["aupro"] = aupro(masks, maps)
        report[category] = {"image": metrics(labels, scores), "pixel": pixel}
        print(category, json.dumps(report[category]), flush=True)
    save_metrics(report, args.output)
    (Path(args.output)/"predictions.json").write_text(json.dumps(predictions, indent=2))


def main():
    args = parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.shots < 0 or args.epochs < 1:
        raise ValueError("shots must be nonnegative and epochs positive")
    args.image_size = args.image_size or (512 if args.backbone == "dinov3" else 518)
    seed_all(args.seed)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (train if args.mode == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
