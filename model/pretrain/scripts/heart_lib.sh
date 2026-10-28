# shellcheck shell=bash
# =============================================================================
# HEART shared shell helpers. Sourced (never executed) by
#   scripts/pretrain_smoke.sh, scripts/pretrain_heart.sh, scripts/check_heart_env.sh
# after they have set REPO. Every function only reads, prints, or creates
# directories under the git-ignored runs/ tree; nothing here installs packages.
#
# CMRI layout (fixed; every path can be overridden through the HEART_* variable):
#   BASE=/home/ypan/heart_liver/model_train
#   ├── heart-molecular-pretraining/       REPO   (this git checkout)
#   │   ├── model/{pretrain,unimol}/       MODEL_ROOT (cwd for unicore-train)
#   │   └── runs/                          OUTPUT_ROOT (git-ignored)
#   ├── third_party/Uni-Core/              UNICORE_SRC (pinned ace6fae)
#   └── UniMolReference/ligand_data/ligands/   DATA_PATH (read-only LMDBs)
# =============================================================================

HEART_DEFAULT_BASE=/home/ypan/heart_liver/model_train
HEART_UNICORE_COMMIT=ace6fae1c8479a9751f2bb1e1d6e4047427bc134

die()  { echo "ERROR: $*" >&2; exit 1; }
log()  { echo "[heart $(date '+%F %T')] $*"; }
warn() { echo "[heart $(date '+%F %T')] WARNING: $*" >&2; }

# Locate the repository root from the path of the calling script. Under sbatch
# the script body runs from a copy in the SLURM spool directory, so the path
# that was submitted is recovered from `scontrol show job`.
heart_find_repo() {
    local script_path="$1" submitted
    if [ -n "${HEART_REPO:-}" ]; then
        REPO="$HEART_REPO"
        return
    fi
    if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
        submitted="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Command=//p' | head -1)"
        submitted="${submitted%% *}"
        [ -n "$submitted" ] && [ -f "$submitted" ] && script_path="$submitted"
    fi
    REPO="$(cd "$(dirname "$script_path")/../../.." 2>/dev/null && pwd)"
    if [ ! -d "$REPO/model/unimol" ]; then
        REPO="$HEART_DEFAULT_BASE/heart-molecular-pretraining"
    fi
}

heart_paths() {
    [ -d "$REPO/model/unimol" ] || die "HEART repository not found at $REPO (set HEART_REPO)"
    REPO="$(cd "$REPO" && pwd)"
    BASE="${HEART_BASE:-$(dirname "$REPO")}"
    MODEL_ROOT="$REPO/model"
    PRETRAIN_DIR="$MODEL_ROOT/pretrain"
    UNICORE_SRC="${HEART_UNICORE_SRC:-$BASE/third_party/Uni-Core}"
    DATA_PATH="${HEART_DATA_PATH:-$BASE/UniMolReference/ligand_data/ligands}"
    OUTPUT_ROOT="${HEART_OUTPUT_ROOT:-$REPO/runs}"
    CONDA_ENV="${HEART_CONDA_ENV:-/home/ypan/.conda/envs/heart_gpu}"
    CONDA_MODULE="${HEART_CONDA_MODULE:-miniconda3/py311_24.1.2}"
    CONDA_SH="${HEART_CONDA_SH:-/mnt/software/miniconda3/py311_24.1.2/etc/profile.d/conda.sh}"
}

heart_print_paths() {
    echo "BASE        = $BASE"
    echo "REPO        = $REPO"
    echo "MODEL_ROOT  = $MODEL_ROOT"
    echo "UNICORE_SRC = $UNICORE_SRC"
    echo "DATA_PATH   = $DATA_PATH"
    echo "OUTPUT_ROOT = $OUTPUT_ROOT"
    echo "CONDA_ENV   = $CONDA_ENV"
}

heart_slurm_info() {
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
    echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
    echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
    echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}"
    echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
    echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
    echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-}"
    echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
}

# Activate heart_gpu exactly as CMRI_pretrian.txt did and set PY. Never installs.
# HEART_SKIP_ENV=1 + HEART_PYTHON=<python> is the local-testing hook.
heart_activate_env() {
    export PYTHONNOUSERSITE=1          # ~/.local packages must never shadow heart_gpu
    if [ "${HEART_SKIP_ENV:-0}" != "1" ]; then
        if ! type module >/dev/null 2>&1; then
            for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do
                # shellcheck disable=SC1090
                [ -f "$f" ] && . "$f"
            done
        fi
        if type module >/dev/null 2>&1; then
            module load "$CONDA_MODULE" >/dev/null 2>&1 || warn "'module load $CONDA_MODULE' failed; using $CONDA_SH directly"
        fi
        [ -f "$CONDA_SH" ] || die "conda bootstrap not found: $CONDA_SH"
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        conda activate "$CONDA_ENV" || die "cannot activate $CONDA_ENV"
        PY="$CONDA_ENV/bin/python"
        case "$(command -v python)" in
            "$CONDA_ENV"/*) ;;
            *) die "python on PATH is $(command -v python), not $CONDA_ENV/bin/python" ;;
        esac
    else
        PY="${HEART_PYTHON:-python}"
    fi
    "$PY" -c 'import sys; print("python", sys.version.split()[0], sys.executable)' || die "python not usable: $PY"
}

heart_runtime_env() {
    export NCCL_IB_DISABLE=1
    export NCCL_ASYNC_ERROR_HANDLING=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    # One BLAS/OpenMP thread per process: the DataLoader workers already use the
    # allocated cores (OMP_NUM_THREADS=6 per worker oversubscribes 6 cores).
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

# The Uni-Mol ligand LMDBs are only ever opened read-only.
heart_check_data() {
    local f
    for f in train.lmdb valid.lmdb dict.txt; do
        [ -r "$DATA_PATH/$f" ] || die "missing or unreadable: $DATA_PATH/$f"
    done
    ls -lh "$DATA_PATH/train.lmdb" "$DATA_PATH/valid.lmdb" "$DATA_PATH/dict.txt"
}

# $1 = run directory. Refuses run directories inside (or containing) the data.
heart_check_save_dir() {
    local save_abs data_abs probe
    mkdir -p "$OUTPUT_ROOT" || die "cannot create $OUTPUT_ROOT"
    probe="$OUTPUT_ROOT/.heart_write_test_$$"
    { touch "$probe" && rm -f "$probe"; } || die "$OUTPUT_ROOT is not writable"
    save_abs="$(realpath -m "$1")"
    data_abs="$(realpath -m "$DATA_PATH")"
    case "$save_abs/" in "$data_abs"/*) die "save dir must not be inside the dataset directory" ;; esac
    case "$data_abs/" in "$save_abs"/*) die "save dir must not contain the dataset" ;; esac
}

# $1 = use_cpu (0/1), $2 = minimum GPU memory in GiB, $3 = gpu optional (0/1)
heart_gpu_preflight() {
    if [ "$1" != "1" ]; then
        if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi || true; else warn "nvidia-smi not available"; fi
    fi
    "$PY" - "$1" "$2" "$3" <<'PYEOF' || die "PyTorch/CUDA pre-flight failed"
import sys, torch
use_cpu, min_gb, gpu_optional = sys.argv[1] == "1", float(sys.argv[2]), sys.argv[3] == "1"
print("torch", torch.__version__, "| CUDA build", torch.version.cuda,
      "| cuda available", torch.cuda.is_available())
if use_cpu:
    print("--cpu requested: GPU checks skipped (local testing only)")
    sys.exit(0)
if not torch.cuda.is_available():
    if gpu_optional:
        print("WARNING: no GPU visible (dry run outside SLURM); GPU checks skipped")
        sys.exit(0)
    sys.exit("CUDA is not available in this job")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print("GPU{}: {} | {:.1f} GiB | sm_{}{} | bf16 native: {}".format(
        i, p.name, p.total_memory / 2**30, p.major, p.minor, p.major >= 8))
x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
print("fp16 matmul OK:", float((x @ x).float().abs().mean()) > 0)
gb = torch.cuda.get_device_properties(0).total_memory / 2**30
if gb < min_gb:
    sys.exit("GPU0 has {:.1f} GiB < required {} GiB".format(gb, min_gb))
PYEOF
}

# Uni-Core must discover task/model/arch/loss through --user-dir ./unimol
# with cwd = MODEL_ROOT, exactly as unicore-train will.
heart_registry_check() {
    (cd "$MODEL_ROOT" && "$PY" pretrain/tools/check_registry.py --user-dir ./unimol) \
        || die "Uni-Core cannot discover the unimol_contrastive registrations"
}

# Sets TRAIN_ENTRY (array). HEART_TRAIN_ENTRY is the local-testing hook.
heart_train_entry() {
    local entry
    if [ -n "${HEART_TRAIN_ENTRY:-}" ]; then
        read -r -a TRAIN_ENTRY <<< "$HEART_TRAIN_ENTRY"
        return
    fi
    entry="$(command -v unicore-train || true)"
    [ -n "$entry" ] || die "unicore-train not on PATH; run: bash scripts/check_heart_env.sh --fix"
    if [ "${HEART_SKIP_ENV:-0}" != "1" ]; then
        case "$entry" in
            "$CONDA_ENV"/*) ;;
            *) die "unicore-train resolves to $entry, not inside $CONDA_ENV; run: bash scripts/check_heart_env.sh --fix" ;;
        esac
    fi
    TRAIN_ENTRY=("$entry")
}

# Options that the scripts own, or that would load pretrained weights, may not
# be passed through HEART_EXTRA_ARGS.
heart_reject_forbidden_args() {
    local a
    for a in "$@"; do
        case "$a" in
            --save-dir*|--tmp-save-dir*|--user-dir*|--tensorboard-logdir*|--stop-time-hours*|\
            --restore-file*|--finetune-from-model*|--reset-optimizer|--reset-dataloader|\
            --reset-lr-scheduler|--reset-meters)
                die "HEART_EXTRA_ARGS must not contain $a (owned by the script / no pretrained weights)" ;;
        esac
    done
}

# Prints the number of optimizer updates stored in a checkpoint.
heart_ckpt_updates() {
    "$PY" - "$1" <<'PYEOF'
import sys, torch
try:
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except TypeError:
    ck = torch.load(sys.argv[1], map_location="cpu")
print(ck["optimizer_history"][-1]["num_updates"])
PYEOF
}

# $1 = run dir, $2 = margin minutes. Sets STOP_ARGS (cumulative --stop-time-hours).
heart_stop_time_args() {
    local time_left stop_h
    STOP_ARGS=()
    if [ -n "${SLURM_JOB_ID:-}" ] && [ "$2" != "0" ] && command -v squeue >/dev/null 2>&1; then
        time_left="$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null || true)"
        stop_h="$("$PY" "$PRETRAIN_DIR/tools/stop_time_hours.py" --time-left "$time_left" \
            --margin-minutes "$2" --save-dir "$1")" || stop_h=0
        if [ "$stop_h" != "0" ]; then
            STOP_ARGS=(--stop-time-hours "$stop_h")
            log "SLURM time left $time_left -> --stop-time-hours $stop_h (cumulative, margin $2 min)"
        fi
    fi
}

# --- CPU memory diagnostics (Linux only; sacct MaxRSS is not always recorded) ---
# Prints the memory file of this job's cgroup: $1 = current | peak | max
heart_cgroup_mem_file() {
    local cg d
    [ -r /proc/self/cgroup ] || return 0
    cg="$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)"
    d="/sys/fs/cgroup$cg"
    if [ -n "$cg" ] && [ -r "$d/memory.current" ]; then         # cgroup v2
        case "$1" in current) echo "$d/memory.current" ;; peak) echo "$d/memory.peak" ;; max) echo "$d/memory.max" ;; esac
        return 0
    fi
    cg="$(awk -F: '$2 ~ /(^|,)memory(,|$)/ {print $3}' /proc/self/cgroup)"
    d="/sys/fs/cgroup/memory$cg"
    if [ -n "$cg" ] && [ -r "$d/memory.usage_in_bytes" ]; then  # cgroup v1
        case "$1" in current) echo "$d/memory.usage_in_bytes" ;; peak) echo "$d/memory.max_usage_in_bytes" ;; max) echo "$d/memory.limit_in_bytes" ;; esac
    fi
}

# Limits that can turn an allocation into a Python MemoryError.
heart_memory_limits() {
    local f
    echo "=== CPU memory limits ==="
    echo "ulimit -v (virtual, KiB): $(ulimit -v)  | ulimit -m: $(ulimit -m)  | ulimit -n (fds): $(ulimit -n)"
    [ -r /proc/meminfo ] && grep -E '^(MemTotal|MemAvailable)' /proc/meminfo | tr -s ' ' | tr '\n' ' ' && echo
    f="$(heart_cgroup_mem_file max)"
    [ -n "$f" ] && [ -r "$f" ] && echo "cgroup memory limit: $(cat "$f") ($f)"
    df -h /dev/shm 2>/dev/null | tail -1 | awk '{print "/dev/shm: size " $2 ", used " $3 ", free " $4}'
    echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-} SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU:-}"
}

# Background sampler: $1 = TSV file, $2 = interval seconds. Sets MEM_MON_PID.
# Columns: total/max RSS and max VSZ over this job's processes (session),
# cgroup usage, /dev/shm used (DataLoader workers pass tensors through it).
heart_mem_monitor_start() {
    local out="$1" interval="${2:-10}" sid cgf
    MEM_MON_PID=""
    if [ "$(uname -s)" != "Linux" ] || ! ps -o sid= -p $$ >/dev/null 2>&1; then
        log "memory monitor: needs Linux procps; skipped"
        return 0
    fi
    sid="$(ps -o sid= -p $$ | tr -d ' ')"
    cgf="$(heart_cgroup_mem_file current)"
    printf 'time\tprocs\trss_total_mb\trss_max_proc_mb\tvsz_max_proc_mb\tcgroup_mb\tshm_used_mb\tmem_available_mb\n' > "$out"
    (
        while :; do
            stats="$(ps -s "$sid" -o rss=,vsz= 2>/dev/null | awk '{n++; s+=$1; if ($1>m) m=$1; if ($2>v) v=$2}
                END {printf "%d\t%.0f\t%.0f\t%.0f", n, s/1024, m/1024, v/1024}')"
            cgm=NA; [ -n "$cgf" ] && [ -r "$cgf" ] && cgm=$(( $(cat "$cgf") / 1048576 ))
            shm="$(df -Pm /dev/shm 2>/dev/null | awk 'NR==2 {print $3}')"
            avail="$(awk '/^MemAvailable/ {printf "%.0f", $2/1024}' /proc/meminfo 2>/dev/null)"
            printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$stats" "$cgm" "${shm:-NA}" "${avail:-NA}" >> "$out"
            sleep "$interval"
        done
    ) &
    MEM_MON_PID=$!
    log "memory monitor: every ${interval}s -> $out (pid $MEM_MON_PID)"
}

# Stops the sampler and prints peaks; sets MEM_PEAK_RSS_MB and MEM_CGROUP_PEAK_MB.
heart_mem_monitor_stop() {
    local out="$1" f
    MEM_PEAK_RSS_MB=NA
    MEM_CGROUP_PEAK_MB=NA
    [ -n "${MEM_MON_PID:-}" ] && kill "$MEM_MON_PID" 2>/dev/null && wait "$MEM_MON_PID" 2>/dev/null
    if [ -s "$out" ]; then
        MEM_PEAK_RSS_MB="$(awk -F'\t' 'NR>1 && $3>m {m=$3} END {print m+0}' "$out")"
        echo "=== CPU memory (sampled, $out) ==="
        awk -F'\t' 'NR>1 {if ($3>r) r=$3; if ($4>p) p=$4; if ($5>v) v=$5; if ($7>s) s=$7; if ($2>n) n=$2}
            END {printf "peak total RSS %d MB | peak single-process RSS %d MB | peak VSZ %d MB | peak /dev/shm used %d MB | max processes %d\n", r, p, v, s, n}' "$out"
    fi
    f="$(heart_cgroup_mem_file peak)"
    if [ -n "$f" ] && [ -r "$f" ]; then
        MEM_CGROUP_PEAK_MB=$(( $(cat "$f") / 1048576 ))
        echo "cgroup peak memory: ${MEM_CGROUP_PEAK_MB} MB ($f)"
    fi
    return 0
}

# $1 = run dir, $2 = script path, remaining = command. Small text files only.
heart_provenance() {
    local save_dir="$1" script="$2" prov
    shift 2
    prov="$save_dir/provenance/job-${SLURM_JOB_ID:-local}-$(date '+%Y%m%d-%H%M%S')"
    mkdir -p "$prov"
    cp "$script" "$prov/$(basename "$script").used" 2>/dev/null || true
    printf '%q ' "$@" > "$prov/command.txt"; echo >> "$prov/command.txt"
    git -C "$REPO" rev-parse HEAD > "$prov/git_commit.txt" 2>/dev/null || true
    git -C "$REPO" status --short > "$prov/git_status.txt" 2>/dev/null || true
    (cd "$MODEL_ROOT" && find unimol pretrain/tools pretrain/scripts -type f \( -name '*.py' -o -name '*.sh' \) \
        | sort | xargs sha256sum) > "$prov/source_sha256.txt" 2>/dev/null || true
    "$PY" -m pip freeze > "$prov/pip_freeze.txt" 2>/dev/null || true
    { env | grep -E '^(SLURM_|CUDA_|PYTORCH_|OMP_|HEART_|CONDA_PREFIX=)' | sort; } > "$prov/env.txt" || true
    log "provenance : $prov"
}
