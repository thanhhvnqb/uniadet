#!/usr/bin/env bash
# Reuse the checked three-GPU scheduler; no benchmark is launched by default.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE=source-validation
FORWARDED=()
while (($#)); do
    case "$1" in
        --phase) PHASE="${2:?Missing phase}"; shift 2 ;;
        --help|-h)
            echo 'Usage: bash scripts/run_reproduction_review.sh --phase source-validation|benchmark [scheduler options]'
            echo 'Defaults to --dry-run. Use --execute to train and evaluate seeds 10/20/30.'
            echo 'source-validation: VisA pcb1/pcb2 holdout; baseline and raw image peaks share a head.'
            echo 'benchmark: fixed baseline, VisA -> Real-IAD. Override --target-data explicitly for paper transfer checks.'
            echo 'Scheduler options: --run-id ID --gpus 0,2,3 --workers 2 --threads 2 --ram-per-job-gib N --dry-run|--execute'
            exit 0 ;;
        *) FORWARDED+=("$1"); shift ;;
    esac
done
case "$PHASE" in
    source-validation) CONFIG="$ROOT/configs/reproduction_source_validation.py"; TARGET=datasets/VisA; RAM=24 ;;
    benchmark) CONFIG="$ROOT/configs/reproduction_benchmark.py"; TARGET=datasets/Real-IAD; RAM=48 ;;
    *) echo "Unknown phase: $PHASE" >&2; exit 2 ;;
esac
exec bash "$ROOT/scripts/run_reproduction_parallel.sh" \
    --coordinator "$ROOT/scripts/baseline_review_jobs.py" --config "$CONFIG" \
    --variants baseline --target-data "$TARGET" --test-batch-size 16 \
    --ram-per-job-gib "$RAM" "${FORWARDED[@]}"
