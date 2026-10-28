#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=heart-smoke-test
#SBATCH --mem=55G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --output=/home/ypan/heart_liver/model_train/heart-molecular-pretraining/runs/slurm/%x-%j.out
#SBATCH --error=/home/ypan/heart_liver/model_train/heart-molecular-pretraining/runs/slurm/%x-%j.err
# =============================================================================
# HEART 30-minute GPU SMOKE TEST: 10 genuine optimizer updates on the real
# Uni-Mol ligand train/valid LMDB (1 Tesla T4, fp16), validation at updates 5
# and 10, checkpoint save + inspection. It proves the pipeline works; it does
# NOT pretrain the model, and it never starts the full run (pretrain_heart.sh).
#
# Same Uni-Core workflow as LUMI-lab model/pretrain/pretrain.sh:
#   cd <model root> && unicore-train <data> --user-dir ./unimol --task unimol_contrastive ...
#
# Submit (runs/slurm/ must exist before sbatch; it is created by git pull/--fix):
#   cd /home/ypan/heart_liver/model_train/heart-molecular-pretraining/model/pretrain
#   sbatch scripts/pretrain_smoke.sh
# Resume test (continue the same run directory from update 10 to 12):
#   RESUME_DIR=<run dir> MAX_UPDATE=12 sbatch --job-name=heart-smoke-resume scripts/pretrain_smoke.sh
# T4 out of memory: resubmit with a documented smaller micro-batch (same model/losses):
#   BATCH_SIZE=2 sbatch scripts/pretrain_smoke.sh        # then BATCH_SIZE=1
# Pre-flight only (login node; prints the command, creates nothing):
#   HEART_DRY_RUN=1 bash scripts/pretrain_smoke.sh
#
# Optional environment overrides:
#   RESUME_DIR, MAX_UPDATE (10..20), BATCH_SIZE, HEART_DRY_RUN=1,
#   HEART_REPO / HEART_BASE / HEART_DATA_PATH / HEART_OUTPUT_ROOT / HEART_CONDA_ENV.
# Local CPU testing hooks only: HEART_SKIP_ENV=1, HEART_PYTHON, HEART_TRAIN_ENTRY,
#   HEART_LOCAL_CPU_TEST=1 (replaces fp16 by --cpu), HEART_EXTRA_ARGS="--flag v ..."
# =============================================================================
set -eo pipefail

# >>> Locate the repository (sbatch runs a spooled copy of this file) >>>
script_path="${BASH_SOURCE[0]}"
if [ -z "${HEART_REPO:-}" ] && [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    submitted="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Command=//p' | head -1)"
    submitted="${submitted%% *}"
    [ -n "$submitted" ] && [ -f "$submitted" ] && script_path="$submitted"
fi
lib="$(dirname "$script_path")/heart_lib.sh"
[ -f "$lib" ] || lib="${HEART_REPO:-/home/ypan/heart_liver/model_train/heart-molecular-pretraining}/model/pretrain/scripts/heart_lib.sh"
# shellcheck source=heart_lib.sh
. "$lib"
heart_find_repo "$script_path"
heart_paths

# >>> Heart Slurm information >>>
heart_slurm_info
heart_print_paths

# >>> conda initialize (heart_gpu; this script never installs packages) >>>
heart_activate_env
heart_runtime_env

# >>> HEART smoke-test parameters (CMRI pilot values; LUMI objectives) >>>
run_name=heart-smoke-test
n_gpu=1
num_workers=5                    # 6 CPUs: 5 DataLoader workers + main process
lr=1e-5
wd=1e-4
batch_size="${BATCH_SIZE:-4}"    # molecules per GPU (x2 contrastive views); T4 OOM -> 2, then 1
batch_size_valid="$batch_size"
update_freq=1
heads=64
masked_token_loss=1
masked_coord_loss=5
masked_dist_loss=10
x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01
contrastive_loss=10.0
contrastive_view_mode=independent    # HEART P3; LUMI stock data path == identical
mask_prob=0.15
only_polar=0
noise_type="uniform"
noise=1.0
max_atoms=256
seed=1
warmup_steps=3
total_num_update=20              # > max_update so a resume test still has lr > 0
max_update="${MAX_UPDATE:-10}"   # 10 genuine optimizer updates (fp16 overflow steps do not count)
log_interval=1
validate_interval_updates=5
save_interval_updates=5
keep_interval_updates=1
max_valid_steps=5
stop_margin_minutes=5            # stop cleanly (validate + checkpoint) 5 min before the limit

case "$batch_size" in ''|*[!0-9]*|0) die "BATCH_SIZE must be a positive integer (got '$batch_size')" ;; esac
case "$max_update" in ''|*[!0-9]*) die "MAX_UPDATE must be an integer (got '$max_update')" ;; esac
[ "$max_update" -ge 1 ] && [ "$max_update" -le "$total_num_update" ] \
    || die "MAX_UPDATE must be in 1..$total_num_update for the smoke test (full training: pretrain_heart.sh)"
[ "$batch_size" = 4 ] || warn "batch-size $batch_size (pilot default 4): record this deviation with the results"

precision_args=(--fp16 --fp16-init-scale 4 --fp16-scale-window 256)
USE_CPU=0
if [ "${HEART_LOCAL_CPU_TEST:-0}" = "1" ]; then
    precision_args=(--cpu)       # local CPU test suite only; never on CMRI
    USE_CPU=1
fi
read -r -a extra_args <<< "${HEART_EXTRA_ARGS:-}"
heart_reject_forbidden_args "${extra_args[@]}"
if [ -n "${SLURM_CPUS_PER_TASK:-}" ] && [ "$num_workers" -ge "$SLURM_CPUS_PER_TASK" ]; then
    num_workers=$(( SLURM_CPUS_PER_TASK > 1 ? SLURM_CPUS_PER_TASK - 1 : 0 ))
fi

# >>> Pre-flight: real dataset, opened read-only >>>
heart_check_data

# >>> Run directory (one per job; RESUME_DIR continues an existing run) >>>
DRY_RUN="${HEART_DRY_RUN:-0}"
if [ -n "${RESUME_DIR:-}" ]; then
    save_dir="$(realpath -m "$RESUME_DIR")"
    [ -f "$save_dir/checkpoint_last.pt" ] || die "RESUME_DIR has no checkpoint_last.pt: $save_dir"
else
    save_dir="$OUTPUT_ROOT/smoke/${run_name}-$(date '+%Y%m%d-%H%M%S')-j${SLURM_JOB_ID:-local}"
fi
heart_check_save_dir "$save_dir"
resume_from=""
if [ -f "$save_dir/checkpoint_last.pt" ]; then
    resume_from="$(heart_ckpt_updates "$save_dir/checkpoint_last.pt")" || die "cannot read $save_dir/checkpoint_last.pt"
    [ "$max_update" -gt "$resume_from" ] || die "checkpoint is already at $resume_from updates; set MAX_UPDATE > $resume_from"
    log "run dir    : $save_dir (RESUMING from checkpoint_last.pt @ $resume_from updates)"
else
    log "run dir    : $save_dir (new run, random initialisation)"
fi

# >>> Pre-flight: GPU / PyTorch / Uni-Core registry >>>
gpu_optional=0
[ "$DRY_RUN" = "1" ] && [ -z "${SLURM_JOB_ID:-}" ] && gpu_optional=1
heart_gpu_preflight "$USE_CPU" 0 "$gpu_optional"
heart_registry_check
heart_train_entry
[ "$DRY_RUN" = "1" ] || mkdir -p "$save_dir"
heart_stop_time_args "$save_dir" "$stop_margin_minutes"

# >>> Training command (single GPU: unicore-train directly; see IMPLEMENTATION_MAP finding 3) >>>
cd "$MODEL_ROOT"
CMD=("${TRAIN_ENTRY[@]}" "$DATA_PATH" --user-dir ./unimol --train-subset train --valid-subset valid
    --num-workers "$num_workers" --data-buffer-size 10 --ddp-backend=c10d
    --task unimol_contrastive --loss unimol_contrastive --arch unimol_contrastive_base
    --encoder-attention-heads "$heads"
    --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay "$wd"
    --lr-scheduler polynomial_decay --lr "$lr" --warmup-updates "$warmup_steps" --total-num-update "$total_num_update"
    --update-freq "$update_freq" --seed "$seed"
    "${precision_args[@]}" --tensorboard-logdir "$save_dir/tsb"
    --max-update "$max_update" --log-interval "$log_interval" --log-format simple
    --save-interval-updates "$save_interval_updates" --validate-interval-updates "$validate_interval_updates"
    --keep-interval-updates "$keep_interval_updates" --no-epoch-checkpoints --max-valid-steps "$max_valid_steps"
    --masked-token-loss "$masked_token_loss" --masked-coord-loss "$masked_coord_loss" --masked-dist-loss "$masked_dist_loss"
    --x-norm-loss "$x_norm_loss" --delta-pair-repr-norm-loss "$delta_pair_repr_norm_loss"
    --mask-prob "$mask_prob" --noise-type "$noise_type" --noise "$noise" --max-atoms "$max_atoms"
    --batch-size "$batch_size" --batch-size-valid "$batch_size_valid"
    --save-dir "$save_dir" --tmp-save-dir "$save_dir" --only-polar "$only_polar" --mode train
    --dict-name dict.txt --contrastive-loss "$contrastive_loss" --contrastive-view-mode "$contrastive_view_mode"
    "${STOP_ARGS[@]}" "${extra_args[@]}")

echo "=== Command ==="
printf '%q ' "${CMD[@]}"; echo
if [ "$DRY_RUN" = "1" ]; then
    log "HEART_DRY_RUN=1: pre-flight passed; would write to $save_dir; not training"
    exit 0
fi

heart_provenance "$save_dir" "$script_path" "${CMD[@]}"
train_log="$save_dir/train-${SLURM_JOB_ID:-local}.log"
log "training log: $train_log"
set +e
"${CMD[@]}" 2>&1 | tee -a "$train_log"
rc=${PIPESTATUS[0]}

# >>> Smoke-test verdict: losses, update count, checkpoint reload >>>
echo "=== Summary ==="
summary_args=(--expect-updates "$max_update")
[ -n "$resume_from" ] && summary_args+=(--expect-resume-from "$resume_from")
"$PY" "$PRETRAIN_DIR/tools/summarize_log.py" "$train_log" "${summary_args[@]}"
src=$?
echo "=== Checkpoint ==="
ls -lh "$save_dir"
"$PY" "$PRETRAIN_DIR/tools/verify_checkpoint.py" "$save_dir/checkpoint_last.pt" \
    --expect-updates "$max_update" --data "$DATA_PATH"
vrc=$?
set -e

verdict=PASS
[ "$rc" -eq 0 ] && [ "$src" -eq 0 ] && [ "$vrc" -eq 0 ] || verdict=FAIL
{
    echo "verdict=$verdict"
    echo "job_id=${SLURM_JOB_ID:-local}"
    echo "date=$(date '+%F %T')"
    echo "max_update=$max_update"
    echo "resume_from=${resume_from:-none}"
    echo "batch_size=$batch_size"
    echo "train_exit=$rc summary_exit=$src checkpoint_exit=$vrc"
    echo "train_log=$train_log"
} > "$save_dir/smoke_result-${SLURM_JOB_ID:-local}.txt"
log "smoke verdict: $verdict (training $rc, log summary $src, checkpoint $vrc) -> $save_dir/smoke_result-${SLURM_JOB_ID:-local}.txt"
log "run dir: $save_dir"
[ "$verdict" = PASS ]
