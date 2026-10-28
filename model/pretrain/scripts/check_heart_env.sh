#!/bin/bash
# =============================================================================
# HEART environment check (default) and explicit repair (--fix) for the
# persistent heart_gpu conda env used by Terminal, Jupyter and SLURM alike.
#
#   cd /home/ypan/heart_liver/model_train/heart-molecular-pretraining/model/pretrain
#   bash scripts/check_heart_env.sh          # CHECK ONLY: no modifications; PASS/FAIL
#   bash scripts/check_heart_env.sh --fix    # repair once, then re-check everything
#   srun --partition=gpu --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=00:10:00 \
#        bash scripts/check_heart_env.sh     # same check + CUDA/fp16 on an allocated GPU
#
# --fix (idempotent; does nothing when every check already passes):
#   * installs only missing/incompatible packages from requirements-heart.txt
#     with "$CONDA_ENV/bin/python -m pip", PIP_USER=0, and a constraints file that
#     pins the installed torch (2.2.2+cu121) and numpy<2 -- never --user,
#     never ~/.local, never base conda, never system python;
#   * (re)installs Uni-Core from the EXISTING checkout
#     /home/ypan/heart_liver/model_train/third_party/Uni-Core, only if it is at
#     commit ace6fae; nothing is cloned or checked out;
#   * creates the git-ignored runs/slurm/ log directory needed before sbatch;
#   * registers the Jupyter kernel "heart_gpu" (PYTHONNOUSERSITE=1);
#   * logs versions and locations to runs/env/.
# It never touches torch/CUDA. Problems it must not fix (wrong interpreter,
# torch/CUDA build, Uni-Core checkout at another commit) are reported as
# BLOCK and the script exits non-zero.
#
# Training scripts never call --fix; repairs are always a deliberate step.
# =============================================================================
set -o pipefail

usage() { sed -n '2,29p' "${BASH_SOURCE[0]}"; }
MODE=check
WITH_OPTIONAL=0
for arg in "$@"; do
    case "$arg" in
        --fix) MODE=fix ;;
        --with-optional) WITH_OPTIONAL=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $arg (use --fix, --with-optional or --help)" >&2; exit 2 ;;
    esac
done

# shellcheck source=heart_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/heart_lib.sh"
heart_find_repo "${BASH_SOURCE[0]}"
heart_paths

NFAIL=0
pass()  { echo "PASS  $*"; }
fail()  { echo "FAIL  $*"; NFAIL=$((NFAIL + 1)); }
info()  { echo "INFO  $*"; }
section() { echo; echo "=== $* ==="; }

if [ "$MODE" = fix ]; then
    mkdir -p "$OUTPUT_ROOT/env" || die "cannot create $OUTPUT_ROOT/env"
    FIX_LOG="$OUTPUT_ROOT/env/check_heart_env-fix-$(date '+%Y%m%d-%H%M%S').log"
    exec > >(tee -a "$FIX_LOG") 2>&1
    echo "fix log: $FIX_LOG"
fi

section "Mode: $MODE   host=$(hostname)  user=$(whoami)  $(date '+%F %T')"
heart_print_paths
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<not in a job>}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

section "model_train filesystem"
for d in "$REPO" "$UNICORE_SRC" "$DATA_PATH"; do
    [ -d "$d" ] && pass "dir $d" || fail "missing primary directory $d"
done
for d in "$MODEL_ROOT/pretrain/scripts" "$MODEL_ROOT/unimol/data" "$MODEL_ROOT/unimol/tasks" \
         "$MODEL_ROOT/unimol/models" "$MODEL_ROOT/unimol/losses"; do
    [ -d "$d" ] && pass "dir $d" || fail "missing $d"
done
[ -d "$MODEL_ROOT/pretrain/unimol" ] && fail "$MODEL_ROOT/pretrain/unimol exists: unimol/ must be a sibling of pretrain/"
for f in train.lmdb valid.lmdb dict.txt; do
    if [ -r "$DATA_PATH/$f" ]; then pass "data $(ls -lhL "$DATA_PATH/$f" | awk '{print $5}')  $DATA_PATH/$f"
    else fail "unreadable $DATA_PATH/$f"; fi
done
[ -w "$DATA_PATH" ] && info "dataset directory is writable by you; HEART only opens it read-only"
if [ "$MODE" = fix ]; then
    mkdir -p "$OUTPUT_ROOT/slurm" && info "ensured $OUTPUT_ROOT/slurm"
fi
if [ -d "$OUTPUT_ROOT/slurm" ] && [ -w "$OUTPUT_ROOT/slurm" ]; then
    pass "SLURM log dir $OUTPUT_ROOT/slurm exists and is writable (required before sbatch)"
else
    fail "missing/unwritable $OUTPUT_ROOT/slurm -> mkdir -p $OUTPUT_ROOT/slurm (or run --fix)"
fi
if [ -d "$OUTPUT_ROOT" ] && [ -w "$OUTPUT_ROOT" ]; then
    df -h "$OUTPUT_ROOT" 2>/dev/null | tail -1
    echo "runs/ usage: $(du -sh "$OUTPUT_ROOT" 2>/dev/null | cut -f1)"
fi
command -v quota >/dev/null 2>&1 && quota -s 2>/dev/null | tail -3

section "Conda environment"
if [ "${HEART_SKIP_ENV:-0}" = "1" ]; then
    PY="${HEART_PYTHON:-python}"
    info "HEART_SKIP_ENV=1: using $PY without conda activation (local testing only)"
else
    if ! type module >/dev/null 2>&1; then
        for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do [ -f "$f" ] && . "$f"; done
    fi
    if type module >/dev/null 2>&1; then
        module load "$CONDA_MODULE" >/dev/null 2>&1 && pass "module load $CONDA_MODULE" || info "module load $CONDA_MODULE failed"
    fi
    if [ -f "$CONDA_SH" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH" && pass "sourced $CONDA_SH"
        conda activate "$CONDA_ENV" && pass "activated $CONDA_ENV" || fail "conda activate $CONDA_ENV"
    else
        fail "conda bootstrap missing: $CONDA_SH"
    fi
    PY="$CONDA_ENV/bin/python"
    [ -x "$PY" ] || { fail "$PY not found"; echo; echo "$NFAIL CHECK(S) FAILED"; exit 1; }
    case "$(command -v python)" in
        "$CONDA_ENV"/*) pass "python on PATH: $(command -v python)" ;;
        *) fail "python on PATH is $(command -v python), not from $CONDA_ENV" ;;
    esac
fi
export PYTHONNOUSERSITE=1
export PIP_USER=0
ENV_PREFIX="${HEART_ENV_PREFIX:-$CONDA_ENV}"

run_python_checks() {
    "$PY" "$PRETRAIN_DIR/tools/heart_env_check.py" --env-prefix "$ENV_PREFIX" \
        --requirements "$PRETRAIN_DIR/requirements-heart.txt" \
        --unicore-src "$UNICORE_SRC" --unicore-commit "$HEART_UNICORE_COMMIT" \
        --emit-fix "$FIX_LIST" $([ "$WITH_OPTIONAL" = 1 ] && echo --with-optional)
}

FIX_LIST="$(mktemp)"
trap 'rm -f "$FIX_LIST" "${CONSTRAINTS:-}"' EXIT
section "Python packages and Uni-Core (import paths, versions)"
run_python_checks
PYRC=$?

if [ "$MODE" = fix ] && [ -s "$FIX_LIST" ]; then
    section "Repair"
    if [ "$PYRC" = 2 ]; then
        echo "BLOCK items above must be resolved by hand; refusing to install into this environment."
        exit 2
    fi
    SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    [ -w "$SITE" ] || die "$SITE is not writable: pip would fall back to --user; not installing"
    TORCH_VER="$("$PY" -c 'import torch; print(torch.__version__)')"
    CONSTRAINTS="$(mktemp)"
    printf 'torch==%s\nnumpy<2\n' "$TORCH_VER" > "$CONSTRAINTS"
    echo "constraints: $(tr '\n' ' ' < "$CONSTRAINTS")"
    mapfile -t PIP_SPECS < <(grep -v '^UNICORE$' "$FIX_LIST")
    if [ "${#PIP_SPECS[@]}" -gt 0 ]; then
        echo "+ $PY -m pip install --constraint <constraints> ${PIP_SPECS[*]}"
        "$PY" -m pip install --upgrade-strategy only-if-needed --constraint "$CONSTRAINTS" "${PIP_SPECS[@]}" \
            || fail "pip install ${PIP_SPECS[*]}"
    fi
    if grep -qx UNICORE "$FIX_LIST"; then
        # Pure-Python build (CUDA extensions are off by default at ace6fae).
        # setup.py writes untracked build files (unicore/version.py, build/,
        # *.egg-info) into the checkout; the checked-out commit is unchanged.
        echo "+ $PY -m pip install --no-build-isolation --no-deps --force-reinstall $UNICORE_SRC"
        "$PY" -m pip install --no-build-isolation --no-deps --force-reinstall "$UNICORE_SRC" \
            || fail "Uni-Core install from $UNICORE_SRC"
    fi
    "$PY" -m pip check || info "pip check reported conflicts (see above)"
    hash -r
    section "Re-check after repair"
    run_python_checks
    PYRC=$?
fi
case "$PYRC" in
    0) pass "Python packages, Uni-Core and unicore-train" ;;
    2) fail "BLOCK: problems --fix will not change (see BLOCK lines)" ;;
    *) fail "Python packages / Uni-Core (run: bash scripts/check_heart_env.sh --fix)" ;;
esac

section "Uni-Core registry (cwd = $MODEL_ROOT, --user-dir ./unimol)"
(cd "$MODEL_ROOT" && "$PY" pretrain/tools/check_registry.py --user-dir ./unimol) && pass "registry" || fail "registry"

section "Jupyter kernel"
KERNEL_DIR="${JUPYTER_DATA_DIR:-$HOME/.local/share/jupyter}/kernels/heart_gpu"
if [ "$MODE" = fix ] && "$PY" -c 'import ipykernel' 2>/dev/null; then
    mkdir -p "$KERNEL_DIR"
    "$PY" - "$KERNEL_DIR/kernel.json" "$PY" "$ENV_PREFIX" <<'PYEOF'
import json, sys
path, py, prefix = sys.argv[1:4]
spec = {"argv": [py, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python 3.11 (heart_gpu)", "language": "python",
        "env": {"PYTHONNOUSERSITE": "1", "CONDA_PREFIX": prefix}}
new = json.dumps(spec, indent=1)
try:
    old = open(path).read()
except OSError:
    old = None
if old != new:
    open(path, "w").write(new)
print("kernelspec", "unchanged" if old == new else "written", path)
PYEOF
fi
if [ -f "$KERNEL_DIR/kernel.json" ] && grep -q "\"$PY\"" "$KERNEL_DIR/kernel.json"; then
    pass "Jupyter kernel 'heart_gpu' -> $PY"
else
    info "Jupyter kernel 'heart_gpu' not registered for $PY (run --fix; then pick 'Python 3.11 (heart_gpu)')"
fi

section "CUDA / GPU"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "NoDevFound" ]; then
    nvidia-smi && pass "nvidia-smi" || fail "nvidia-smi"
    "$PY" - <<'PYEOF' && pass "torch CUDA + fp16 forward/backward on allocated GPU" || fail "torch CUDA"
import sys, torch
print("torch", torch.__version__, "CUDA build", torch.version.cuda, "available", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print("GPU0", p.name, "{:.1f} GiB".format(p.total_memory / 2**30), "sm_{}{}".format(p.major, p.minor))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16, requires_grad=True)
(x @ x).float().sum().backward()
torch.cuda.synchronize()
print("fp16 OK; bf16 native:", p.major >= 8, "(T4 = sm_75: use fp16)")
PYEOF
else
    info "no GPU allocated: CUDA checks skipped (use the srun command in the header)"
fi

section "Result ($MODE)"
if [ "$NFAIL" -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "$NFAIL CHECK(S) FAILED"; fi
[ -n "${FIX_LOG:-}" ] && echo "fix log: $FIX_LOG"
exit $(( NFAIL > 0 ? 1 : 0 ))
