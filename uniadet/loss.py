import torch
import torch.nn.functional as F


def training_loss(outputs, labels, masks):
    """Sum independent per-layer CE + focal + normal/anomaly Dice losses."""
    global_logits, local_logits = outputs
    masks = masks.float()
    if masks.ndim == 4:
        masks = masks[:, 0]
    target = masks.long()
    classification = sum(F.cross_entropy(x, labels.long()) for x in global_logits)
    segmentation = masks.new_zeros(())
    for logits in local_logits:
        probs = logits.softmax(1)
        pt = probs.gather(1, target[:, None]).squeeze(1).clamp_min(1e-7)
        segmentation = segmentation + (-(1-pt).pow(2) * pt.log()).mean()
        for c in (0, 1):
            pred = probs[:, c].flatten(1)
            gt = (target == c).float().flatten(1)
            segmentation = segmentation + (1 - (2*(pred*gt).sum(1)+1)/(pred.sum(1)+gt.sum(1)+1)).mean()
    return classification + segmentation
