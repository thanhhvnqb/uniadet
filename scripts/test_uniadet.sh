#!/usr/bin/env bash
set -euo pipefail
python test.py --data "${TARGET_ROOT:?Set TARGET_ROOT}" --checkpoint "${CHECKPOINT:?Set CHECKPOINT}" --shots "${SHOTS:-0}" --output "runs/eval_${SHOTS:-0}shot" "$@"
