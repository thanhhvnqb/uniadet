"""Use only after source-only validation accepts raw image peaks; keep matched baseline."""
from copy import deepcopy
from configs.reproduction_benchmark import TRAIN_CONFIG as _TRAIN, TEST_CONFIG as _TEST, REVIEW_CONFIG as _REVIEW
TRAIN_CONFIG=deepcopy(_TRAIN)
TEST_CONFIG=deepcopy(_TEST)
REVIEW_CONFIG=dict(_REVIEW,score='raw_image_peak')
