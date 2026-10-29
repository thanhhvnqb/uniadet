#!/usr/bin/env bash
set -euo pipefail
# Train on auxiliary VisA test data; evaluate on other domains.
python train.py --data "${VISA_ROOT:?Set VISA_ROOT}" --backbone clip --output runs/visa "$@"
