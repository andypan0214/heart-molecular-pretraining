#!/bin/bash
#SBATCH --job-name=heart-env-check
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --output=/home/ypan/heart_liver/model/pretrain/logs/%x-%j.out
#SBATCH --error=/home/ypan/heart_liver/model/pretrain/logs/%x-%j.err
# =============================================================================
# HEART pre-flight check (non-destructive: reads, imports, prints; writes only
# a tiny temp file under heart_training_runs/ to test writability).
#
#   Login node (paths, env, imports, registry; no GPU):
#       bash /home/ypan/heart_liver/model/pretrain/scripts/check_heart_env.sh
#   Inside a GPU allocation (adds CUDA / fp16 checks on the allocated GPU):
#       cd /home/ypan/heart_liver/model/pretrain && mkdir -p logs
#       sbatch scripts/check_heart_env.sh
#
# Prints PASS / FAIL / WARN per item and exits non-zero if any FAIL.
# =============================================================================
set -o pipefail

MODEL_ROOT="${HEART_MODEL_ROOT:-/home/ypan/heart_liver/model}"
DATA="${HEART_DATA_PATH:-$MODEL_ROOT/UniMolReference/ligand_data/ligands}"
OUT_ROOT="${HEART_OUTPUT_ROOT:-$MODEL_ROOT/heart_training_runs}"
CONDA_MODULE="${HEART_CONDA_MODULE:-miniconda3/py311_24.1.2}"
CONDA_SH="${HEART_CONDA_SH:-/mnt/software/miniconda3/py311_24.1.2/etc/profile.d/conda.sh}"
CONDA_ENV="${HEART_CONDA_ENV:-/home/ypan/.conda/envs/heart_gpu}"

NFAIL=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; NFAIL=$((NFAIL + 1)); }
warn() { echo "WARN  $*"; }
section() { echo; echo "=== $* ==="; }

section "Host / SLURM"
echo "host=$(hostname)  user=$(whoami)  pwd=$(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<not in a job>}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
command -v sbatch >/dev/null 2>&1 && pass "sbatch: $(command -v sbatch)" || warn "sbatch not on PATH (fine inside a job)"
if command -v sinfo >/dev/null 2>&1; then
    echo "GPU partitions / GRES (use the type name in --gres=gpu:<type>:1):"
    sinfo -o "%P %G %N" 2>/dev/null | grep -i -E "gpu|PARTITION" | head -20
fi

section "Paths"
for d in "$MODEL_ROOT" "$MODEL_ROOT/pretrain" "$MODEL_ROOT/pretrain/scripts" "$MODEL_ROOT/unimol" "$CONDA_ENV"; do
    [ -d "$d" ] && pass "dir  $d -> $(readlink -f "$d")" || fail "missing dir $d"
done
[ -d "$MODEL_ROOT/pretrain/logs" ] && pass "logs/ exists (needed before sbatch)" \
    || fail "missing $MODEL_ROOT/pretrain/logs  ->  mkdir -p $MODEL_ROOT/pretrain/logs"
[ -d "$MODEL_ROOT/pretrain/unimol" ] && fail "$MODEL_ROOT/pretrain/unimol exists: unimol/ must be a sibling of pretrain/, not inside it"
for f in train.lmdb valid.lmdb dict.txt; do
    if [ -r "$DATA/$f" ]; then pass "data $(ls -lh "$DATA/$f" | awk '{print $5}')  $DATA/$f"; else fail "unreadable $DATA/$f"; fi
done
for f in clean_smi.csv.gz Readme.txt; do
    [ -e "$DATA/$f" ] && pass "data $DATA/$f" || warn "not found (informational): $DATA/$f"
done
[ -w "$DATA" ] && warn "dataset directory is writable by you; HEART only opens it read-only"
mkdir -p "$OUT_ROOT" 2>/dev/null
tmpf="$OUT_ROOT/.heart_write_test_$$"
if touch "$tmpf" 2>/dev/null; then rm -f "$tmpf"; pass "writable output root $OUT_ROOT"; else fail "cannot write $OUT_ROOT"; fi
df -h "$OUT_ROOT" 2>/dev/null | tail -1
command -v quota >/dev/null 2>&1 && quota -s 2>/dev/null | tail -3
[ -d "$OUT_ROOT" ] && echo "current heart_training_runs usage: $(du -sh "$OUT_ROOT" 2>/dev/null | cut -f1)"

section "Conda environment"
if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do [ -f "$f" ] && . "$f"; done
fi
if type module >/dev/null 2>&1; then
    module load "$CONDA_MODULE" 2>/dev/null && pass "module load $CONDA_MODULE" || warn "module load $CONDA_MODULE failed"
    module avail git 2>&1 | grep -q -i "git/" && echo "git modules available: $(module avail git 2>&1 | grep -o 'git/[0-9.]*' | tr '\n' ' ')"
fi
if [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH" && pass "sourced $CONDA_SH"
    conda activate "$CONDA_ENV" && pass "activated $CONDA_ENV" || fail "conda activate $CONDA_ENV"
else
    fail "conda bootstrap missing: $CONDA_SH"
fi
export PYTHONNOUSERSITE=1
echo "python: $(command -v python)"
case "$(command -v python)" in "$CONDA_ENV"/*) pass "python comes from heart_gpu" ;; *) fail "python is not from $CONDA_ENV" ;; esac

section "Python packages"
python - <<'PYEOF'
import importlib, sys
print("python", sys.version.split()[0], sys.executable)
# (module, required?, note)
mods = [("torch", True, "2.2.2+cu121 expected"), ("numpy", True, "<2 for torch 2.2"),
        ("lmdb", True, ""), ("rdkit", True, "2D conformer view"), ("scipy", True, "distance matrices"),
        ("tokenizers", True, "imported by unicore.data"), ("unicore", True, "Uni-Core"),
        ("tensorboardX", True, "--tensorboard-logdir"), ("yaml", True, "launcher configs"),
        ("ml_collections", False, ""), ("iopath", False, ""), ("wandb", False, "not used")]
bad = 0
for name, req, note in mods:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "?")
        print("PASS  import {:15s} {:12s} {}".format(name, str(v), note))
        if name == "numpy" and int(v.split(".")[0]) >= 2:
            print("FAIL  numpy {} >= 2 is not ABI-compatible with torch 2.2.x wheels".format(v)); bad += 1
    except Exception as e:
        print("{}  import {:15s} {}: {}".format("FAIL" if req else "WARN", name, type(e).__name__, e))
        bad += req
sys.exit(1 if bad else 0)
PYEOF
[ $? -eq 0 ] && pass "required Python packages" || fail "required Python packages (see README_HEART.md install section)"
command -v unicore-train >/dev/null 2>&1 && pass "unicore-train: $(command -v unicore-train)" || fail "unicore-train not on PATH"

section "Uni-Core registry (cwd = model root, --user-dir ./unimol)"
( cd "$MODEL_ROOT" && python pretrain/tools/check_registry.py --user-dir ./unimol ) && pass "registry" || fail "registry"

section "CUDA / GPU"
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    nvidia-smi && pass "nvidia-smi" || fail "nvidia-smi"
    python - <<'PYEOF' && pass "torch CUDA + fp16 on allocated GPU" || fail "torch CUDA"
import sys, torch
print("torch", torch.__version__, "CUDA build", torch.version.cuda, "available", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print("GPU0", p.name, "{:.1f} GiB".format(p.total_memory / 2**30), "sm_{}{}".format(p.major, p.minor))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16, requires_grad=True)
(x @ x).float().sum().backward()
torch.cuda.synchronize()
print("fp16 matmul + backward OK; bf16 supported:", torch.cuda.is_bf16_supported())
PYEOF
else
    warn "not inside a GPU allocation: CUDA checks skipped (run: sbatch scripts/check_heart_env.sh)"
fi

section "Result"
if [ "$NFAIL" -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "$NFAIL CHECK(S) FAILED"; fi
exit "$NFAIL"
