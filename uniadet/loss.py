import torch
import torch.nn.functional as F


def training_loss(outputs, labels, masks, *, focal_gamma=2., dice_smooth=1.,
                  classification_weight=1., segmentation_weight=1., return_components=False):
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
        segmentation = segmentation + (-(1-pt).pow(focal_gamma) * pt.log()).mean()
        for c in (0, 1):
            pred = probs[:, c].flatten(1)
            gt = (target == c).float().flatten(1)
            segmentation = segmentation + (1 - (2*(pred*gt).sum(1)+dice_smooth)/(pred.sum(1)+gt.sum(1)+dice_smooth)).mean()
    loss = classification_weight * classification + segmentation_weight * segmentation
    return {"loss": loss, "classification": classification, "segmentation": segmentation} if return_components else loss
