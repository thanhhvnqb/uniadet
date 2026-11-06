"""Source-only selection of image-score smoothing; no target benchmark tuning."""
from copy import deepcopy
from config import TRAIN_CONFIG as _TRAIN, TEST_CONFIG as _TEST
TRAIN_CONFIG=deepcopy(_TRAIN)
TEST_CONFIG=deepcopy(_TEST)
REVIEW_CONFIG=dict(phase='source-validation',score='baseline',holdout=['pcb1','pcb2'],root='runs/baseline/reproduction_review')
