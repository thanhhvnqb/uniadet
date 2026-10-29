"""Compare fixed zero-shot score definitions without selecting on target labels.

Run with the same arguments as test.py; --output receives score_audit.json.
This intentionally computes image metrics only, avoiding dense pixel sorting.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from uniadet.cli import build, metrics, parser, seed_all
from uniadet.data import AnomalyDataset


@torch.inference_mode()
def main():
    args = parser().parse_args(["evaluate", *sys.argv[1:]])
    if not args.checkpoint or args.shots:
        raise ValueError("Provide --checkpoint and use --shots 0")
    seed_all(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    for name in ("backbone", "layers", "image_size", "temperature"):
        setattr(args, name, checkpoint["config"][name])
    for name in ("weights", "dino_repo"):
        if getattr(args, name) is None:
            setattr(args, name, checkpoint["config"].get(name))
    model = build(args).eval()
    model.head.load_state_dict(checkpoint["head"])
    dataset = AnomalyDataset(args.data, image_size=args.image_size, backbone=args.backbone)
    report = {}
    for category, indices in sorted(dataset.groups.items()):
        subset = torch.utils.data.Subset(dataset, indices)
        labels, values = [], {}
        for batch in DataLoader(subset, batch_size=args.batch_size, num_workers=args.workers):
            images = batch["image"].to(args.device)
            global_logits, local_logits = model(images)
            cls = torch.stack([v.softmax(-1)[:, 1] for v in global_logits])
            maps = torch.stack([v.softmax(1)[:, 1] for v in local_logits])
            global_score = cls.mean(0)
            mean_map_peak = maps.mean(0).flatten(1).amax(1)
            layer_peak = maps.flatten(2).amax(2).mean(0)
            scores = {"global_only": global_score, "map_only": mean_map_peak,
                      "legacy_map_fusion": (1-args.lambda_p)*global_score + args.lambda_p*mean_map_peak,
                      "layer_fusion": (1-args.lambda_p)*global_score + args.lambda_p*layer_peak}
            scores.update({f"global_layer_{layer}": score for layer, score in zip(args.layers, cls)})
            for name, score in scores.items():
                values.setdefault(name, []).extend(score.cpu().tolist())
            labels.extend(batch["label"].tolist())
        report[category] = {name: metrics(labels, scores) for name, scores in values.items()}
        print(category, json.dumps(report[category]), flush=True)
    average = {name: {metric: float(np.mean([row[name][metric] for row in report.values()
                                           if row[name][metric] is not None]))
                      if any(row[name][metric] is not None for row in report.values()) else None
                      for metric in ("auroc", "aupr", "f1max")}
               for name in next(iter(report.values()))}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"score_audit.json").write_text(json.dumps(
        {"config": vars(args), "categories": report, "average": average}, indent=2))
    print("Average", json.dumps(average), flush=True)


if __name__ == "__main__":
    main()
