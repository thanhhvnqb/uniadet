"""Image/pixel metrics and report serialization, independent of model execution."""
import csv
import json
from pathlib import Path

import numpy as np


METRIC_COLUMNS = [
    ("image", "auroc", "I-AUROC"), ("image", "aupr", "I-AUPR"),
    ("image", "f1max", "I-F1max"), ("pixel", "auroc", "P-AUROC"),
    ("pixel", "aupr", "P-AUPR"), ("pixel", "f1max", "P-F1max"),
    ("pixel", "aupro", "P-AUPRO"),
]


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
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Category"] + [column for _, _, column in METRIC_COLUMNS])
        for category, results in [*categories.items(), ("Average", average)]:
            values = [(results.get(task) or {}).get(metric) for task, metric, _ in METRIC_COLUMNS]
            writer.writerow([category] + [100 * value if value is not None else "" for value in values])


def pixel_metrics(masks, maps):
    """Exact dense metrics sharing one score sort, including region-balanced PRO."""
    from scipy.ndimage import label
    from sklearn.metrics import auc

    labels = np.concatenate([mask.ravel() for mask in masks]).astype(bool, copy=False)
    positives = int(labels.sum())
    if positives == 0 or positives == labels.size:
        return dict(auroc=None, aupr=None, f1max=None, aupro=None)
    scores = np.concatenate([amap.ravel() for amap in maps])
    # Match sklearn's stable descending order and aggregate tied thresholds.
    order = np.argsort(scores, kind="stable")[::-1]
    scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), labels.size - 1]
    del scores
    ranked_labels = labels[order]
    tp = np.cumsum(ranked_labels, dtype=np.float64)[ends]
    fp = (ends + 1) - tp
    recall = tp / positives
    precision = tp / (ends + 1)
    fpr = np.r_[0., fp / (labels.size - positives)]
    result = {
        "auroc": float(auc(fpr, np.r_[0., recall])),
        "aupr": float(auc(np.r_[0., recall], np.r_[1., precision])),
        "f1max": float(np.max(2 * tp / (ends + 1 + positives))),
    }
    del ranked_labels, tp, fp, recall, precision
    weights = []
    for mask in masks:
        components, count = label(mask, structure=np.ones((3, 3)))
        areas = np.bincount(components.ravel())
        region_weights = np.zeros(count + 1, dtype=np.float64)
        region_weights[1:] = 1. / areas[1:]
        weights.append(region_weights[components].ravel())
    weighted_tp = np.cumsum(np.concatenate(weights)[order], dtype=np.float64)[ends]
    pro = np.r_[0., weighted_tp / weighted_tp[-1]]
    stop = np.searchsorted(fpr, 0.3, side="right")
    x, y = fpr[:stop], pro[:stop]
    if x[-1] < 0.3:
        boundary = np.interp(0.3, fpr[stop-1:stop+1], pro[stop-1:stop+1])
        x, y = np.append(x, 0.3), np.append(y, boundary)
    result["aupro"] = float(auc(x, y) / 0.3)
    return result


