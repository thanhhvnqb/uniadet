#!/usr/bin/env bash
# Three seeds per configuration, one train -> validate job per physical GPU.
set -euo pipefail
# Let best-effort progress writes return EPIPE rather than terminating Bash.
trap '' PIPE
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CONFIG="$ROOT/configs/reproduction_benchmark.py"
COORDINATOR="$ROOT/scripts/baseline_review_jobs.py"
VARIANTS="baseline"
GPUS="0,2,3"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
SHOTS=0
WORKERS=2
THREADS=2
MIN_FREE_MIB=9500
RAM_PER_JOB_GIB=24
EXECUTE=0
TARGET_DATA="datasets/Real-IAD"
TEST_BATCH_SIZE=16
PREPARED_PLAN=""
usage() {
    cat <<'HELP'
Usage: bash scripts/run_reproduction_parallel.sh [options]
  --variants CSV       Configurations, executed alphabetically (default: baseline)
  --gpus CSV           Physical GPUs, up to three (default: 0,2,3)
  --config FILE        Base configuration (default: configs/reproduction_benchmark.py)
  --coordinator FILE   Planning/worker/aggregation backend (default: baseline_review_jobs.py)
  --run-id NAME        Unique artifact ID (default: UTC timestamp + PID)
  --target-data CSV    Evaluation paths (default: datasets/Real-IAD)
  --test-batch-size N  Evaluation-only batch size (default: 16)
  --prepared-plan DIR Execute an existing validated plan, without rewriting snapshots
  --shots 0|1          Separate zero-shot / one-shot run (default: 0)
  --workers N          DataLoader workers per job (default: 2)
  --threads N          CPU compute threads per job (default: 2)
  --min-free-mib N     Minimum free VRAM per GPU (default: 9500)
  --ram-per-job-gib N   RAM scheduling budget per job (default: 24)
  --python FILE        Python environment (default: .venv/bin/python or $PYTHON)
  --execute            Cache weights, create snapshots, train then validate
  --dry-run            Only check GPU/CUDA/data and show plan (default)
Each configuration runs seeds 10/20/30 on GPUs 0/2/3 respectively.
GPU 1 is disabled. Fewer usable GPUs still run all three seeds.
No benchmark starts without --execute. CPU fallback and overwriting are disabled.
HELP
}
while (($#)); do
    case "$1" in
        --execute) EXECUTE=1; shift ;;
        --dry-run) EXECUTE=0; shift ;;
        --help|-h) usage; exit 0 ;;
        --variants|--gpus|--config|--coordinator|--run-id|--shots|--workers|--threads|--min-free-mib|--ram-per-job-gib|--python|--target-data|--test-batch-size|--prepared-plan)
            if (($# < 2)) || [[ "$2" == --* ]]; then
                echo "Missing value for $1" >&2; exit 2
            fi
            case "$1" in
                --prepared-plan) PREPARED_PLAN="$2" ;;
                --target-data) TARGET_DATA="$2" ;;
                --test-batch-size) TEST_BATCH_SIZE="$2" ;;
                --variants) VARIANTS="$2" ;;
                --gpus) GPUS="$2" ;;
                --config) CONFIG="$2" ;;
                --coordinator) COORDINATOR="$2" ;;
                --run-id) RUN_ID="$2" ;;
                --shots) SHOTS="$2" ;;
                --workers) WORKERS="$2" ;;
                --threads) THREADS="$2" ;;
                --min-free-mib) MIN_FREE_MIB="$2" ;;
                --ram-per-job-gib) RAM_PER_JOB_GIB="$2" ;;
                --python) PYTHON="$2" ;;
            esac
            shift 2 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ -x "$PYTHON" ]] || { echo "Python is not executable: $PYTHON" >&2; exit 2; }
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export PYTHONUNBUFFERED=1
PREPARE=("$PYTHON" "$COORDINATOR" prepare --config "$CONFIG"
    --variants "$VARIANTS" --gpus "$GPUS" --run-id "$RUN_ID" --shots "$SHOTS"
    --workers "$WORKERS" --threads "$THREADS" --min-free-mib "$MIN_FREE_MIB" --ram-per-job-gib "$RAM_PER_JOB_GIB")
if [[ -n "$TARGET_DATA" ]]; then PREPARE+=(--target-data "$TARGET_DATA"); fi
if [[ -n "$TEST_BATCH_SIZE" ]]; then PREPARE+=(--test-batch-size "$TEST_BATCH_SIZE"); fi
if ((EXECUTE == 0)); then
    if [[ -n "$PREPARED_PLAN" ]]; then
        cat "$PREPARED_PLAN/queue.tsv"
        exit 0
    fi
    CUDA_VISIBLE_DEVICES="" "${PREPARE[@]}" --dry-run
    exit 0
fi
command -v setsid >/dev/null || { echo "setsid is required for isolated worker groups" >&2; exit 2; }
command -v flock >/dev/null || { echo "flock is required" >&2; exit 2; }
if [[ -n "$PREPARED_PLAN" ]]; then
    PLAN_DIR="$(cd -- "$PREPARED_PLAN" && pwd)"
    [[ -f "$PLAN_DIR/plan.json" && -f "$PLAN_DIR/queue.tsv" ]] || { echo "Prepared plan is incomplete" >&2; exit 2; }
    [[ ! -e "$PLAN_DIR/pids.tsv" ]] || { echo "Prepared plan was already launched; refusing to overwrite outcomes" >&2; exit 2; }
else
    CUDA_VISIBLE_DEVICES="" "${PREPARE[@]}" --execute
    PLAN_DIR="$ROOT/runs/baseline/launches/$RUN_ID"
fi
# All HF weights must now resolve from the version cached by preflight.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Prevent concurrent launchers from sharing a physical GPU. External workloads are
# checked by preflight; locks also cover other instances of this launcher.
declare -a LOCK_FDS=()
while IFS= read -r gpu; do
    exec {lock_fd}>"${TMPDIR:-/tmp}/uniadet-gpu-${UID}-${gpu}.lock"
    flock -n "$lock_fd" || { echo "GPU $gpu already reserved by another launcher" >&2; exit 1; }
    LOCK_FDS+=("$lock_fd")
done < <(cut -f1 "$PLAN_DIR/queue.tsv" | sort -nu)
declare -a PIDS=() RECORDS=() ACTIVE_GPUS=()
MAX_JOBS=3
if [[ -f "$PLAN_DIR/max_parallel" ]]; then read -r MAX_JOBS < "$PLAN_DIR/max_parallel"; fi
[[ "$MAX_JOBS" =~ ^[1-3]$ ]] || { echo 'Invalid plan concurrency limit' >&2; exit 2; }
FAILED=0
TOTAL=$(wc -l < "$PLAN_DIR/queue.tsv")
STARTED=0
COMPLETED=0
# Progress is optional; terminal disconnection must not stop queue collection.
progress() { printf "$@" 2>/dev/null || true; }
printf 'pid\tgpu\trecord\tlog\n' > "$PLAN_DIR/pids.tsv"
: > "$PLAN_DIR/exits.tsv"
collect_batch() {
    local index status
    for index in "${!PIDS[@]}"; do
        status=0
        wait "${PIDS[$index]}" || status=$?
        printf '%s\t%s\n' "${RECORDS[$index]}" "$status" >> "$PLAN_DIR/exits.tsv"
        COMPLETED=$((COMPLETED + 1))
        progress '[%s] FINISHED %s/%s GPU=%s exit=%s %s\n' "$(date -Is)" "$COMPLETED" "$TOTAL" "${ACTIVE_GPUS[$index]}" "$status" "${RECORDS[$index]}"
        if ((status != 0)); then FAILED=1; fi
    done
    PIDS=(); RECORDS=(); ACTIVE_GPUS=()
}
interrupted() {
    trap '' INT TERM HUP
    local pid
    for pid in "${PIDS[@]}"; do
        # Only terminate worker process groups created by this launcher.
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
    collect_batch
    "$PYTHON" "$COORDINATOR" aggregate "$PLAN_DIR/plan.json" || true
    exit 130
}
trap interrupted INT TERM HUP
while IFS=$'\t' read -r gpu record log; do
    # Queue when all slots are occupied or the next physical GPU is still busy.
    for active in "${ACTIVE_GPUS[@]}"; do
        if [[ "$active" == "$gpu" ]]; then collect_batch; break; fi
    done
    if ((${#PIDS[@]} >= MAX_JOBS)); then collect_batch; fi
    STARTED=$((STARTED + 1))
    progress '[%s] START %s/%s GPU=%s %s log=%s\n' "$(date -Is)" "$STARTED" "$TOTAL" "$gpu" "$record" "$log"
    CUDA_VISIBLE_DEVICES="$gpu" setsid "$PYTHON" "$COORDINATOR" job "$record" > >(tee --output-error=warn-nopipe "$log") 2>&1 &
    pid=$!
    PIDS+=("$pid"); RECORDS+=("$record"); ACTIVE_GPUS+=("$gpu")
    printf '%s\t%s\t%s\t%s\n' "$pid" "$gpu" "$record" "$log" >> "$PLAN_DIR/pids.tsv"
done < "$PLAN_DIR/queue.tsv"
collect_batch
progress '[%s] AGGREGATE: collecting metrics and writing reports\n' "$(date -Is)"
"$PYTHON" "$COORDINATOR" aggregate "$PLAN_DIR/plan.json" || FAILED=1
progress 'Finished. Exit records: %s/exits.tsv\n' "$PLAN_DIR"
exit "$FAILED"
