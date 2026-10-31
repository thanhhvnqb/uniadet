"""Editable run configuration. No named experiments or implicit overrides."""

MODE = "train_test"  # train, test, or train_test (test saved checkpoints after training)
MODEL = "uniadet-best-current"  # Output directory name; use a new name for each run
RUNS_ROOT = "runs"
TRAIN_DATASET = None  # None uses the training data directory name

TRAIN_CONFIG = {
    # Data and frozen backbone. Excluded categories cannot enter CAA mosaics.
    "data": "datasets/VisA", "backbone": "clip",  # clip, eva02_clip, dinov2, dinov3
    "weights": None, "dino_repo": None, "layers": [12, 15, 18, 21, 24],
    "image_size": None,  # Auto: 518 for CLIP/DINOv2; 512 for DINOv3
    "split": "test",  # Labeled auxiliary test split, as used in the paper
    "categories": None, "exclude_categories": None,  # E.g. exclude ["pcb1", "pcb2"]

    # Representation settings: changing these requires a new trained head.
    "clip_global_norm": True,  # False bypasses only CLIP global-token layer norm
    "clip_pos_interpolation": "bicubic",  # "bicubic" or "bilinear"; CLIP only
    "softmax_before_resize": True,  # Selected configuration: resize probabilities
    "mask_interpolation": "nearest",  # "bilinear" uses threshold 0.5 after resizing
    "temperature": 0.07,
    "head_init_std": None,  # None uses feature_dim**-0.5; e.g. 0.02 for ablation

    # Optimization and execution. None device automatically selects CUDA/CPU.
    "epochs": 15, "lr": 0.001, "betas": [0.5, 0.999], "weight_decay": 0.0,
    "batch_size": 8, "workers": 4, "seed": 10, "device": None,
    "log_interval": 50,

    # Augmentation: True/False switches; mask-preservation fix enabled by default.
    "no_caa": False,  # False = CAA ON; True = CAA OFF (legacy key)
    "preserve_anomaly_mask": True,  # False reproduces the old empty-mask behavior
    "caa_probability": 0.5, "caa_grids": [2, 3], "mosaic_probability": 0.5,
    "rotation": False, "horizontal_flip": False, "vertical_flip": False,

    # Independent loss terms. Weight 0 disables a term; baseline weights are 1.
    "focal_gamma": 2.0, "dice_smooth": 1.0,
    "classification_weight": 1.0, "segmentation_weight": 1.0,
    "focal_weight": 1.0, "normal_dice_weight": 1.0, "anomaly_dice_weight": 1.0,
    "focal_smoothing": 0.0,  # 1e-5 matches AnomalyCLIP's smoothed focal form

    # Prediction defaults saved in the checkpoint. TEST_CONFIG can override them.
    "lambda_f": 0.5, "lambda_p": 0.5,
    "zero_score_pooling": "layer",  # "layer" or "map"; final layer reduction is underspecified
    "gaussian_sigma": 4.0,  # Selected configuration; 0 disables inference smoothing
}

TEST_CONFIG = {
    "data": ["datasets/MVTecAD"],  # Target dataset roots
    "shots": [0],  # Set [0, 1, 2, 4] for zero- and few-shot evaluation
    "epochs": [15],  # Paper comparison: epoch 15 only. None = sweep all saved epochs
    "start_epoch": 1,  # Additional inclusive lower bound; never resumes training
    "checkpoint": None,  # None uses runs/<MODEL>/train_<dataset>

    # None restores the selected checkpoint's settings; values are explicit overrides.
    "lambda_f": None, "lambda_p": None, "zero_score_pooling": None,  # None restores checkpoint
    "gaussian_sigma": None,  # None restores checkpoint; 0 disables smoothing
    "weights": None, "dino_repo": None,  # Only override to relocate pretrained sources
    "categories": None, "exclude_categories": None,  # Target/validation categories

    # Optional evaluation-only execution settings. None inherits TRAIN_CONFIG.
    "batch_size": None, "workers": None, "device": None, "seed": None,
}

