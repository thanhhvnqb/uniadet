"""Fixed baseline benchmark; keep score='baseline' until source validation is complete."""
from copy import deepcopy
from config import TRAIN_CONFIG as _TRAIN, TEST_CONFIG as _TEST
TRAIN_CONFIG=deepcopy(_TRAIN)
TEST_CONFIG=deepcopy(_TEST)
REVIEW_CONFIG=dict(phase='benchmark',score='baseline',holdout=[],root='runs/baseline/reproduction_review')
