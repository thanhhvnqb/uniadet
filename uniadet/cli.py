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


from .settings import (
    PREDICTION_DEFAULTS, CLIP_FEATURE_DEFAULTS, TRAINED_SETTING_DEFAULTS,
    LEGACY_PREDICTION_DEFAULTS, LEGACY_TRAINED_SETTING_DEFAULTS,
    resolve_training_settings, restore_evaluation_settings, validate_settings,
)
from .evaluation_metrics import metrics, aupro, save_metrics, pixel_metrics


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parser():
    p = argparse.ArgumentParser(description="UniADet training / zero- and few-shot evaluation")
    p.add_argument("mode", choices=("train", "evaluate"))
    p.add_argument("--data", type=str, help="Dataset root; defaults to datasets/VisA for training, datasets/MVTecAD for evaluation")
    p.add_argument("--backbone", choices=("clip", "eva02_clip", "dinov2", "dinov3"), default="clip")
    p.add_argument("--weights", help="Pretrained backbone weights (required for DINOv3)")
    p.add_argument("--dino-repo", help="Optional local official DINOv3 checkout")
    p.add_argument("--layers", type=int, nargs="+", default=[12, 15, 18, 21, 24])
    p.add_argument("--image-size", type=int)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--lambda-f", type=float, default=None,
                   help="Map fusion weight; train default 0.5, evaluate defaults to checkpoint")
    p.add_argument("--lambda-p", type=float, default=None,
                   help="Image fusion weight; train default 0.5, evaluate defaults to checkpoint")
    p.add_argument("--clip-global-norm", action=argparse.BooleanOptionalAction, default=None,
                   help="Normalize CLIP global tokens before projection (train default: enabled)")
    p.add_argument("--clip-pos-interpolation", choices=("bilinear", "bicubic"), default=None,
                   help="CLIP positional interpolation (train default: bicubic); requires retraining")
    p.add_argument("--head-init-std", type=float,
                   help="Gaussian head initialization std; default is feature_dim**-0.5")
    p.add_argument("--softmax-before-resize", action=argparse.BooleanOptionalAction, default=None,
                   help="Resize patch probabilities before interpolation (new training default: enabled)")
    p.add_argument("--mask-interpolation", choices=("nearest", "bilinear"), default=None,
                   help="Mask resizing: nearest (default) or bilinear followed by threshold 0.5")
    p.add_argument("--gaussian-sigma", type=float, default=None,
                   help="Spatial smoothing: new training default 4; evaluation restores checkpoint (legacy fallback 0)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--betas", type=float, nargs=2, default=[0.5, 0.999])
    p.add_argument("--weight-decay", type=float, default=0.)
    p.add_argument("--focal-gamma", type=float, default=2.)
    p.add_argument("--dice-smooth", type=float, default=1.)
    p.add_argument("--focal-weight", type=float, default=1.)
    p.add_argument("--normal-dice-weight", type=float, default=1.)
    p.add_argument("--anomaly-dice-weight", type=float, default=1.)
    p.add_argument("--focal-smoothing", type=float, default=0.,
                   help="0 disables smoothing; 1e-5 uses AnomalyCLIP's two-class focal smoothing")
    p.add_argument("--classification-weight", type=float, default=1.)
    p.add_argument("--segmentation-weight", type=float, default=1.)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-interval", type=int, default=50, help="Training steps between progress log entries")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="runs/uniadet")
    p.add_argument("--checkpoint")
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--zero-score-pooling", choices=("layer", "map"), default=None,
                   help="Zero-shot maximum per layer or after map averaging; defaults to checkpoint, else layer")
    p.add_argument("--caa", dest="no_caa", action="store_false", help="Enable class-aware augmentation (default)")
    p.add_argument("--no-caa", action="store_true")
    p.set_defaults(no_caa=False)
    p.add_argument("--caa-probability", type=float, default=0.5)
    p.add_argument("--caa-grids", type=int, nargs="+", default=[2, 3])
    p.add_argument("--mosaic-probability", type=float, default=0.5)
    p.add_argument("--preserve-anomaly-mask", action=argparse.BooleanOptionalAction, default=True,
                   help="Fall back to the candidate when augmentation erases its mask (default: enabled)")
    p.add_argument("--rotation", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable random right-angle rotations during training (default: disabled)")
    p.add_argument("--horizontal-flip", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable random horizontal flips during training (default: disabled)")
    p.add_argument("--vertical-flip", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable random vertical flips during training (default: disabled)")
    p.add_argument("--split", default="test", help="Auxiliary training split; paper uses test")
    p.add_argument("--categories", nargs="+", help="Include only these categories in this training/evaluation dataset")
    p.add_argument("--exclude-categories", nargs="+", help="Exclude categories, including from CAA sampling")
    return p


def build(args):
    resolve_training_settings(args)
    encoder = create_encoder(args.backbone, args.layers, args.weights, args.dino_repo,
                             clip_global_norm=args.clip_global_norm,
                             clip_pos_interpolation=args.clip_pos_interpolation)
    return UniADet(encoder, args.temperature, args.lambda_f, args.lambda_p,
                   getattr(args, "head_init_std", None), args.softmax_before_resize).to(args.device)


def train(args):
    resolve_training_settings(args)
    print(f"Training device: {args.device} (PyTorch {torch.__version__}, CUDA build {torch.version.cuda})", flush=True)
    if torch.device(args.device).type == "cpu" and torch.version.cuda and not torch.cuda.is_available():
        print("CUDA is unavailable; training will run on CPU. Check PyTorch CUDA and NVIDIA driver compatibility.", flush=True)
    print(f"Loading {args.backbone} backbone...", flush=True)
    args.image_size = args.image_size or (512 if args.backbone == "dinov3" else 518)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_config.json").write_text(json.dumps(vars(args), indent=2))
    model = build(args).train()
    print(f"Loading training data from {args.data} (split={args.split})...", flush=True)
    dataset = AnomalyDataset(args.data, args.split, args.image_size, args.backbone, caa=not args.no_caa, training=True,
                             caa_probability=getattr(args, "caa_probability", 0.5),
                             caa_grids=getattr(args, "caa_grids", (2, 3)),
                             mosaic_probability=getattr(args, "mosaic_probability", 0.5),
                             rotation=getattr(args, "rotation", True),
                             horizontal_flip=getattr(args, "horizontal_flip", True),
                             vertical_flip=getattr(args, "vertical_flip", True),
                             preserve_anomaly_mask=getattr(args, "preserve_anomaly_mask", True),
                             categories=getattr(args, "categories", None),
                             exclude_categories=getattr(args, "exclude_categories", None),
                             mask_interpolation=args.mask_interpolation)
    # Explicitly reject incomplete supervision even when augmentation is disabled.
    for i, record in enumerate(dataset.records):
        if record["anomaly"] and not (dataset.root / record.get("mask_path", "")).is_file():
            raise ValueError(f"Missing training mask: {record['img_path']}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr, betas=tuple(getattr(args, "betas", (0.5, 0.999))),
                                 weight_decay=getattr(args, "weight_decay", 0.0))
    print(f"Trainable parameters: {sum(p.numel() for p in model.head.parameters()):,}", flush=True)
    print(f"Training samples: {len(dataset):,}; batches per epoch: {len(loader):,}; workers: {args.workers}", flush=True)
    def log_row(row):
        with (output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        csv_path = output / "training.csv"
        new = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)

    for epoch in range(1, args.epochs+1):
        totals = dict(loss=0., classification=0., segmentation=0.)
        seen = 0
        started = time.monotonic()
        print(f"epoch {epoch}/{args.epochs}: loading batches...", flush=True)
        for step, batch in enumerate(loader, 1):
            if step == 1:
                print(f"epoch {epoch}/{args.epochs}: first batch loaded in {time.monotonic()-started:.1f}s; running model...", flush=True)
            outputs = model(batch["image"].to(args.device))
            components = training_loss(outputs, batch["label"].to(args.device), batch["mask"].to(args.device),
                                       focal_gamma=getattr(args, "focal_gamma", 2.), dice_smooth=getattr(args, "dice_smooth", 1.),
                                       classification_weight=getattr(args, "classification_weight", 1.),
                                       segmentation_weight=getattr(args, "segmentation_weight", 1.),
                                       focal_weight=getattr(args, "focal_weight", 1.),
                                       normal_dice_weight=getattr(args, "normal_dice_weight", 1.),
                                       anomaly_dice_weight=getattr(args, "anomaly_dice_weight", 1.),
                                       focal_smoothing=getattr(args, "focal_smoothing", 0.), return_components=True)
            loss = components["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(batch["label"])
            seen += count
            values = {name: value.item() for name, value in components.items()}
            for name, value in values.items():
                totals[name] += value * count
            if step == 1 or step % getattr(args, "log_interval", 50) == 0 or step == len(loader):
                log_row(dict(kind="step", epoch=epoch, step=step, samples=seen, **values,
                             lr=optimizer.param_groups[0]["lr"], seconds=time.monotonic()-started))
                print(f"epoch {epoch}/{args.epochs} step {step}/{len(loader)} loss={loss.item():.6f} elapsed={time.monotonic()-started:.1f}s", flush=True)
        averages = {name: value / seen for name, value in totals.items()}
        log_row(dict(kind="epoch", epoch=epoch, step=len(loader), samples=seen, **averages,
                     lr=optimizer.param_groups[0]["lr"], seconds=time.monotonic()-started))
        print(f"epoch {epoch}/{args.epochs} " + " ".join(f"{k}={v:.6f}" for k, v in averages.items()), flush=True)
        torch.save({"head": model.head.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "config": vars(args)}, Path(args.output)/f"epoch_{epoch}.pt")



@torch.inference_mode()
def evaluate(args):
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for evaluation")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    restore_evaluation_settings(args, checkpoint)
    # Preserve the effective settings, including the encoder source and pooling,
    # so comparisons cannot silently mix different evaluation configurations.
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output)/"evaluation_config.json").write_text(json.dumps(
        {"evaluation": vars(args), "training": config,
         "checkpoint_epoch": checkpoint.get("epoch")}, indent=2))
    model = build(args).eval()
    model.head.load_state_dict(checkpoint["head"])
    dataset = AnomalyDataset(args.data, "test", args.image_size, args.backbone,
                             categories=getattr(args, "categories", None),
                             exclude_categories=getattr(args, "exclude_categories", None),
                             mask_interpolation=args.mask_interpolation)
    memory = None
    if args.shots:
        records = normal_references(args.data, args.shots, args.seed, categories=dataset.groups)
        (Path(args.output)/"references.json").write_text(json.dumps(records, indent=2))
        references = AnomalyDataset(args.data, image_size=args.image_size, backbone=args.backbone, records=records,
                                    mask_interpolation=args.mask_interpolation)
        memory = NormalMemory()
        for item in references:
            key = (item["category"], item["view"])
            memory.add(key, model.encoder(item["image"][None].to(args.device)))
    report, predictions = {}, []
    # Evaluate one class at a time to bound dense metric storage on large datasets.
    for category in sorted(dataset.groups):
        started = time.monotonic()
        rows = [dataset.records[i] for i in dataset.groups[category]]
        print(f"Evaluating {category}: {len(rows)} images...", flush=True)
        subset = AnomalyDataset(args.data, image_size=args.image_size, backbone=args.backbone, records=rows,
                                mask_interpolation=args.mask_interpolation)
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
                zero_score_pooling=args.zero_score_pooling, gaussian_sigma=args.gaussian_sigma)
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
        inference_seconds = time.monotonic() - started
        metric_started = time.monotonic()
        pixel = None
        if masks:
            pixel = pixel_metrics(masks, maps)
        report[category] = {"image": metrics(labels, scores), "pixel": pixel}
        print(category, json.dumps(report[category]), flush=True)
        print(f"  loading/inference={inference_seconds:.1f}s; "
              f"metrics={time.monotonic()-metric_started:.1f}s", flush=True)
    save_metrics(report, args.output)
    (Path(args.output)/"predictions.json").write_text(json.dumps(predictions, indent=2))


def main(argv=None):
    args = parser().parse_args(argv)
    args.data = args.data or ("datasets/VisA" if args.mode == "train" else "datasets/MVTecAD")
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
