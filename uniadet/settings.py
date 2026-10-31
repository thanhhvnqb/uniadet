"""Resolve and validate settings, including historical checkpoint defaults."""
import math

PREDICTION_DEFAULTS = {"lambda_f": 0.5, "lambda_p": 0.5, "zero_score_pooling": "layer", "gaussian_sigma": 4.}
CLIP_FEATURE_DEFAULTS = {"clip_global_norm": True, "clip_pos_interpolation": "bicubic"}
TRAINED_SETTING_DEFAULTS = {**CLIP_FEATURE_DEFAULTS, "softmax_before_resize": True, "mask_interpolation": "nearest"}
# Missing checkpoint fields describe historical behavior, not new-run defaults.
LEGACY_PREDICTION_DEFAULTS = {**PREDICTION_DEFAULTS, "gaussian_sigma": 0.}
LEGACY_TRAINED_SETTING_DEFAULTS = {**TRAINED_SETTING_DEFAULTS, "softmax_before_resize": False}


def resolve_training_settings(args):
    """Resolve unset flags before building a model or serializing a training run."""
    for name, default in {**PREDICTION_DEFAULTS, **TRAINED_SETTING_DEFAULTS}.items():
        if getattr(args, name, None) is None:
            setattr(args, name, default)
    validate_settings(args)


def restore_evaluation_settings(args, checkpoint):
    """Restore the trained representation; allow explicit prediction overrides.

    Older checkpoints predate the CLIP flags and use the original defaults.
    None marks an absent CLI override, including for BooleanOptionalAction.
    """
    config = checkpoint["config"]
    args.backbone = config.get("backbone") or "clip"
    defaults = {"layers": [12, 15, 18, 21, 24], "temperature": 0.07,
                "image_size": 512 if args.backbone == "dinov3" else 518}
    for name, default in defaults.items():
        setattr(args, name, config.get(name) if config.get(name) is not None else default)
    for name, default in LEGACY_TRAINED_SETTING_DEFAULTS.items():
        saved = config.get(name) if config.get(name) is not None else default
        requested = getattr(args, name, None)
        if requested is not None and requested != saved:
            raise ValueError(f"{name}={requested!r} differs from checkpoint value {saved!r}; "
                             "retrain the head for a changed feature representation")
        setattr(args, name, saved)
    for name in ("weights", "dino_repo"):
        if getattr(args, name, None) is None:
            setattr(args, name, config.get(name))
    for name, default in LEGACY_PREDICTION_DEFAULTS.items():
        if getattr(args, name, None) is None:
            setattr(args, name, config.get(name) if config.get(name) is not None else default)
    args.head_init_std = config.get("head_init_std")
    validate_settings(args)


def validate_settings(args):
    for name in ("lambda_f", "lambda_p", "caa_probability", "mosaic_probability"):
        value = getattr(args, name, 0.5)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    for name in ("temperature", "lr", "dice_smooth"):
        value = getattr(args, name, 1.)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name in ("focal_gamma", "focal_weight", "normal_dice_weight", "anomaly_dice_weight",
                 "classification_weight", "segmentation_weight", "weight_decay", "gaussian_sigma"):
        value = getattr(args, name, 0.)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
    smoothing = getattr(args, "focal_smoothing", 0.)
    if not math.isfinite(smoothing) or not 0 <= smoothing < 0.5:
        raise ValueError("focal_smoothing must be finite and in [0, 0.5)")
    std = getattr(args, "head_init_std", None)
    if std is not None and (not math.isfinite(std) or std <= 0):
        raise ValueError("head_init_std must be positive and finite")
    if any(grid < 1 for grid in getattr(args, "caa_grids", (2, 3))):
        raise ValueError("CAA grids must be positive")
    if any(not math.isfinite(beta) or not 0 <= beta < 1 for beta in getattr(args, "betas", (0.5, 0.999))):
        raise ValueError("Adam betas must be finite and in [0, 1)")
    if args.backbone != "clip" and (not args.clip_global_norm or args.clip_pos_interpolation != "bicubic"):
        raise ValueError("CLIP feature ablations require --backbone clip")

