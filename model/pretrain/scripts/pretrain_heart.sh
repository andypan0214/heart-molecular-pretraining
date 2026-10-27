#!/bin/bash
#SBATCH --job-name=heart-pretrain-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=55G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --output=/home/ypan/heart_liver/model/pretrain/logs/%x-%j.out
#SBATCH --error=/home/ypan/heart_liver/model/pretrain/logs/%x-%j.err
# =============================================================================
# HEART general molecular pretraining -- SLURM launcher for CMRI HPC.
#
# Adapted from CMRI_pretrian.txt (the user's CMRI edit of LUMI-lab
# model/pretrain/pretrain.sh). Same Uni-Core workflow:
#   unicore-train <data> --user-dir ./unimol --task unimol_contrastive
#       --loss unimol_contrastive --arch unimol_contrastive_base ...
# Changes vs. CMRI_pretrian.txt are listed in pretrain/IMPLEMENTATION_MAP.md.
#
# Usage (the logs/ directory must exist BEFORE sbatch, or SLURM drops output):
#   cd /home/ypan/heart_liver/model/pretrain && mkdir -p logs
#   sbatch scripts/pretrain_heart.sh configs/pilot.yaml                # 10-update pilot
#   RESUME_DIR=<run dir> MAX_UPDATE=12 \
#       sbatch scripts/pretrain_heart.sh configs/pilot.yaml            # resume test
#   HEART_CONFIRM_FULL=1 sbatch <resource overrides> \
#       scripts/pretrain_heart.sh configs/full_pretrain.yaml           # full run (later)
#
# Environment overrides (all optional):
#   RESUME_DIR=<dir>      reuse an existing run directory (auto-resumes from
#                         <dir>/checkpoint_last.pt)
#   MAX_UPDATE=<n>        override unicore max-update (e.g. resume test)
#   HEART_SET="k=v ..."   extra config overrides, e.g. "batch-size=8 max-valid-steps=2"
#   HEART_CONFIRM_FULL=1  required for configs with requires_confirmation: true
#   HEART_DRY_RUN=1       run all pre-flight checks, print the command, exit
#   HEART_MODEL_ROOT, HEART_CONDA_MODULE, HEART_CONDA_SH, HEART_CONDA_ENV
#   HEART_SKIP_ENV=1, HEART_PYTHON, HEART_TRAIN_ENTRY
#                         local-testing hooks: skip module/conda activation and
#                         choose the interpreter / training entry point
# =============================================================================
set -eo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[heart $(date '+%F %T')] $*"; }

# >>> Heart Slurm information >>>
echo "SLURM_JOBID=${SLURM_JOBID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}"
echo "SLURM_NNODES=${SLURM_NNODES:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# >>> Paths >>>
MODEL_ROOT="${HEART_MODEL_ROOT:-/home/ypan/heart_liver/model}"
PRETRAIN_DIR="$MODEL_ROOT/pretrain"
CONFIG="${1:-$PRETRAIN_DIR/configs/pilot.yaml}"
if [ ! -f "$CONFIG" ] && [ -f "$PRETRAIN_DIR/$CONFIG" ]; then
    CONFIG="$PRETRAIN_DIR/$CONFIG"
fi
[ -f "$CONFIG" ] || die "config not found: $CONFIG"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
[ -d "$MODEL_ROOT/unimol" ] || die "$MODEL_ROOT/unimol not found (unimol/ must be a sibling of pretrain/)"
log "model root : $MODEL_ROOT"
log "config     : $CONFIG"

# >>> Runtime environment >>>
export PYTHONNOUSERSITE=1          # ignore ~/.local site-packages
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# One BLAS/OpenMP thread per process: the DataLoader workers already use the
# allocated cores (the original OMP_NUM_THREADS=6 per worker oversubscribes).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# >>> conda initialize >>>
CONDA_MODULE="${HEART_CONDA_MODULE:-miniconda3/py311_24.1.2}"
CONDA_SH="${HEART_CONDA_SH:-/mnt/software/miniconda3/py311_24.1.2/etc/profile.d/conda.sh}"
CONDA_ENV="${HEART_CONDA_ENV:-/home/ypan/.conda/envs/heart_gpu}"
if [ "${HEART_SKIP_ENV:-0}" != "1" ]; then
    if ! type module >/dev/null 2>&1; then
        for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do
            # shellcheck disable=SC1090
            [ -f "$f" ] && . "$f"
        done
    fi
    if type module >/dev/null 2>&1; then
        module load "$CONDA_MODULE" || log "WARNING: 'module load $CONDA_MODULE' failed; using $CONDA_SH directly"
    fi
    [ -f "$CONDA_SH" ] || die "conda bootstrap not found: $CONDA_SH"
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$CONDA_ENV" || die "cannot activate $CONDA_ENV"
fi
PY="${HEART_PYTHON:-python}"

echo "=== Python ==="
command -v "$PY" || die "python not found"
"$PY" --version
"$PY" -c 'import sys; print(sys.executable)'

# >>> HEART config >>>
SET_ARGS=()
for kv in ${HEART_SET:-}; do SET_ARGS+=(--set "$kv"); done
[ -n "${MAX_UPDATE:-}" ] && SET_ARGS+=(--set "max-update=$MAX_UPDATE")
CFG_SH="$("$PY" "$PRETRAIN_DIR/tools/yaml_to_env.py" "$CONFIG" "${SET_ARGS[@]}")" || die "invalid config $CONFIG"
eval "$CFG_SH"

if [ "${HEART_REQUIRES_CONFIRMATION:-0}" = "1" ] && [ "${HEART_CONFIRM_FULL:-0}" != "1" ]; then
    die "$CONFIG is a full-scale config; resubmit with HEART_CONFIRM_FULL=1 once the pilot has passed"
fi

data_path="$HEART_DATA_PATH"
n_gpu="${HEART_N_GPU:-1}"
USE_CPU=0
for a in "${HEART_UNICORE_ARGS[@]}"; do [ "$a" = "--cpu" ] && USE_CPU=1; done

# Cap DataLoader workers at (allocated CPUs - 1); the main process needs one.
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    max_workers=$(( SLURM_CPUS_PER_TASK > 1 ? SLURM_CPUS_PER_TASK - 1 : 1 ))
    for i in "${!HEART_UNICORE_ARGS[@]}"; do
        if [ "${HEART_UNICORE_ARGS[$i]}" = "--num-workers" ]; then
            j=$((i + 1))
            if [ "${HEART_UNICORE_ARGS[$j]}" -gt "$max_workers" ]; then
                log "num-workers ${HEART_UNICORE_ARGS[$j]} -> $max_workers (SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK)"
                HEART_UNICORE_ARGS[$j]=$max_workers
            fi
        fi
    done
fi

# >>> Pre-flight: dataset (read-only) >>>
for f in train.lmdb valid.lmdb dict.txt; do
    [ -r "$data_path/$f" ] || die "missing or unreadable: $data_path/$f"
done
ls -lh "$data_path/train.lmdb" "$data_path/valid.lmdb" "$data_path/dict.txt"

# >>> Run directory >>>
if [ -n "${RESUME_DIR:-}" ]; then
    save_dir="$RESUME_DIR"
    [ -f "$save_dir/checkpoint_last.pt" ] || die "RESUME_DIR has no checkpoint_last.pt: $save_dir"
elif [ "$HEART_RUN_DIR_MODE" = "fixed" ]; then
    save_dir="$HEART_OUTPUT_ROOT/$HEART_RUN_NAME"
else
    save_dir="$HEART_OUTPUT_ROOT/${HEART_RUN_NAME}-$(date '+%Y%m%d-%H%M')-j${SLURM_JOB_ID:-local}"
fi
DRY_RUN="${HEART_DRY_RUN:-0}"
mkdir -p "$HEART_OUTPUT_ROOT" || die "cannot create $HEART_OUTPUT_ROOT"
probe="$HEART_OUTPUT_ROOT/.heart_write_test_$$"
{ touch "$probe" && rm -f "$probe"; } || die "$HEART_OUTPUT_ROOT is not writable"
save_dir="$(realpath -m "$save_dir")"
data_abs="$(realpath -m "$data_path")"
case "$save_dir/" in "$data_abs"/*) die "save dir must not be inside the dataset directory" ;; esac
case "$data_abs/" in "$save_dir"/*) die "save dir must not contain the dataset" ;; esac
if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$save_dir" || die "cannot create $save_dir"
fi
if [ -f "$save_dir/checkpoint_last.pt" ]; then
    log "run dir    : $save_dir (RESUMING from checkpoint_last.pt)"
else
    log "run dir    : $save_dir (new run, random initialisation)"
fi
echo "=== Storage ==="
df -h "$HEART_OUTPUT_ROOT" || true
command -v quota >/dev/null 2>&1 && { quota -s 2>/dev/null || true; }

# >>> Pre-flight: GPU / PyTorch >>>
echo "=== GPU ==="
if [ "$USE_CPU" = "1" ]; then
    log "--cpu requested: skipping GPU checks (local testing only)"
else
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || log "WARNING: nvidia-smi not available"
fi
echo "=== PyTorch ==="
# A dry run outside a SLURM job (e.g. on the login node) only warns about CUDA.
GPU_OPTIONAL=0
[ "$DRY_RUN" = "1" ] && [ -z "${SLURM_JOB_ID:-}" ] && GPU_OPTIONAL=1
"$PY" - "$USE_CPU" "${HEART_MIN_GPU_MEM_GB:-0}" "$GPU_OPTIONAL" <<'PYEOF' || die "PyTorch/CUDA pre-flight failed"
import sys, torch
use_cpu, min_gb, gpu_optional = sys.argv[1] == "1", float(sys.argv[2]), sys.argv[3] == "1"
print("torch", torch.__version__, "| CUDA build", torch.version.cuda,
      "| cuda available", torch.cuda.is_available())
if not use_cpu:
    if not torch.cuda.is_available():
        if gpu_optional:
            print("WARNING: no GPU visible (dry run outside SLURM); GPU checks skipped")
            sys.exit(0)
        sys.exit("CUDA is not available in this job")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print("GPU{}: {} | {:.1f} GiB | sm_{}{}".format(i, p.name, p.total_memory / 2**30, p.major, p.minor))
    x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    print("fp16 matmul OK:", float((x @ x).float().abs().mean()) > 0)
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    if gb < min_gb:
        sys.exit("GPU0 has {:.1f} GiB < min_gpu_mem_gb {} required by this config".format(gb, min_gb))
PYEOF

# >>> Pre-flight: Uni-Core + ./unimol registrations >>>
cd "$MODEL_ROOT"
"$PY" "$PRETRAIN_DIR/tools/check_registry.py" --user-dir ./unimol || die "Uni-Core cannot discover the unimol_contrastive registrations"
if [ -n "${HEART_TRAIN_ENTRY:-}" ]; then
    read -r -a TRAIN_ENTRY <<< "$HEART_TRAIN_ENTRY"
else
    entry="$(command -v unicore-train || true)"
    [ -n "$entry" ] || die "unicore-train not on PATH (install Uni-Core into $CONDA_ENV, see README_HEART.md)"
    TRAIN_ENTRY=("$entry")
fi

# >>> SLURM time-limit guard >>>
STOP_ARGS=()
if [ -n "${SLURM_JOB_ID:-}" ] && [ "${HEART_STOP_MARGIN_MINUTES:-0}" != "0" ] && command -v squeue >/dev/null 2>&1; then
    time_left="$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null || true)"
    stop_h="$("$PY" "$PRETRAIN_DIR/tools/stop_time_hours.py" --time-left "$time_left" \
        --margin-minutes "$HEART_STOP_MARGIN_MINUTES" --save-dir "$save_dir")" || stop_h=0
    if [ "$stop_h" != "0" ]; then
        STOP_ARGS=(--stop-time-hours "$stop_h")
        log "time left $time_left -> --stop-time-hours $stop_h (cumulative)"
    fi
fi

# >>> Training command >>>
CMD=("${TRAIN_ENTRY[@]}" "$data_path" --user-dir ./unimol
     "${HEART_UNICORE_ARGS[@]}"
     --save-dir "$save_dir" --tmp-save-dir "$save_dir"
     --tensorboard-logdir "$save_dir/tsb"
     "${STOP_ARGS[@]}")
if [ "$n_gpu" -gt 1 ]; then
    # torchrun exports RANK/WORLD_SIZE/MASTER_*/LOCAL_RANK, which Uni-Core reads
    # (it does not accept the --local-rank flag torch.distributed.launch adds).
    CMD=(torchrun --standalone --nproc_per_node="$n_gpu" "${CMD[@]}")
fi

echo "=== Command ==="
printf '%q ' "${CMD[@]}"; echo
if [ "$DRY_RUN" = "1" ]; then
    log "HEART_DRY_RUN=1: pre-flight passed; would write to $save_dir; not training"
    exit 0
fi

# >>> Provenance (small text files only) >>>
prov="$save_dir/provenance/job-${SLURM_JOB_ID:-local}-$(date '+%Y%m%d-%H%M%S')"
mkdir -p "$prov"
cp "$CONFIG" "$prov/"
cp "$0" "$prov/pretrain_heart.sh.used" 2>/dev/null || true
printf '%q ' "${CMD[@]}" > "$prov/command.txt"; echo >> "$prov/command.txt"
(cd "$MODEL_ROOT" && find unimol pretrain/tools -name '*.py' -type f | sort | xargs sha256sum) > "$prov/source_sha256.txt" 2>/dev/null || true
"$PY" -m pip freeze > "$prov/pip_freeze.txt" 2>/dev/null || true
{ env | grep -E '^(SLURM_|CUDA_|PYTORCH_|OMP_|HEART_)' | sort; } > "$prov/env.txt" || true
log "provenance : $prov"

train_log="$save_dir/train-${SLURM_JOB_ID:-local}.log"
log "training log: $train_log"
set +e
"${CMD[@]}" 2>&1 | tee -a "$train_log"
rc=${PIPESTATUS[0]}
set -e

echo "=== Summary ==="
"$PY" "$PRETRAIN_DIR/tools/summarize_log.py" "$train_log" || log "summary reported FAIL items (see above)"
ls -lh "$save_dir"
log "training exit code: $rc"
exit "$rc"
