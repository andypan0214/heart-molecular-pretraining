#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=heart-pretrain-full
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=/home/ypan/heart_liver/model_train/heart-molecular-pretraining/runs/slurm/%x-%j.out
#SBATCH --error=/home/ypan/heart_liver/model_train/heart-molecular-pretraining/runs/slurm/%x-%j.err
# =============================================================================
# HEART FULL general molecular pretraining (random initialisation, Uni-Mol
# ligand LMDB, LUMI-lab objectives). This is the production launcher; the
# 10-update engineering test is scripts/pretrain_smoke.sh.
#
# Same Uni-Core workflow as LUMI-lab model/pretrain/pretrain.sh, with the CMRI
# environment and these guards:
#   * refuses to start unless HEART_CONFIRM_FULL=1, BATCH_SIZE is given
#     explicitly (micro-batch chosen from a VRAM probe) and HEART_SMOKE_RUN
#     points to a smoke run whose smoke_result-*.txt says verdict=PASS;
#   * checks GPU memory, free disk, CPU count and SLURM time left first;
#   * fixed run directory runs/full/<RUN_NAME>: every resubmission resumes from
#     checkpoint_last.pt (model, optimizer, LR schedule, fp16 scaler, data
#     position) and refuses changed scientific settings;
#   * stops cleanly before the SLURM limit: --stop-time-hours = cumulative
#     training time + (time left - 30 min); Uni-Core then validates, writes
#     checkpoint_last.pt and exits 0. SIGUSR1 (sent 30 min before the limit)
#     and SIGTERM are logged and forwarded only if training is still running.
#
# Never submitted automatically. Typical submission (H200 GRES name: sinfo -o "%P %G"):
#   cd /home/ypan/heart_liver/model_train/heart-molecular-pretraining/model/pretrain
#   HEART_CONFIRM_FULL=1 BATCH_SIZE=<probed> UPDATE_FREQ=1 \
#   HEART_SMOKE_RUN=<runs/smoke/heart-smoke-test-...> \
#       sbatch --gres=gpu:<h200-type>:1 scripts/pretrain_heart.sh
# Continue after a time limit (same command; or chain with --dependency=afterany:<jobid>).
# Pre-flight only, on the login node (no training, nothing created):
#   HEART_DRY_RUN=1 HEART_CONFIRM_FULL=1 BATCH_SIZE=... HEART_SMOKE_RUN=... bash scripts/pretrain_heart.sh
#
# Optional overrides: RUN_NAME, UPDATE_FREQ, NUM_WORKERS, MIN_GPU_MEM_GB (40),
#   MIN_FREE_GB (50), STOP_MARGIN_MINUTES (30), N_GPU (1; >1 uses torchrun, untested),
#   HEART_ALLOW_ARG_CHANGE=1 (resume despite changed settings), plus the
#   HEART_* path variables of heart_lib.sh. Local CPU testing hooks as in
#   pretrain_smoke.sh.
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

# >>> Gate 1: explicit human confirmation (checked before anything else) >>>
[ "${HEART_CONFIRM_FULL:-0}" = "1" ] || die "full pretraining requires HEART_CONFIRM_FULL=1 (after the smoke test has passed; see README_HEART.md)"
[ -n "${BATCH_SIZE:-}" ] || die "set BATCH_SIZE explicitly: per-GPU micro-batch chosen from the VRAM probe (README_HEART.md, Stage D)"
[ -n "${HEART_SMOKE_RUN:-}" ] || die "set HEART_SMOKE_RUN to the smoke run directory whose smoke_result-*.txt says verdict=PASS"

# >>> Heart Slurm information >>>
heart_slurm_info
heart_print_paths

# >>> conda initialize (heart_gpu; this script never installs packages) >>>
heart_activate_env
heart_runtime_env

# >>> HEART full-pretraining parameters (LUMI-lab pretrain.sh reference values) >>>
run_name="${RUN_NAME:-heart-pretrain-full-v1}"
n_gpu="${N_GPU:-1}"
num_workers="${NUM_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-6} > 1 ? ${SLURM_CPUS_PER_TASK:-6} - 1 : 0 ))}"
lr=1e-5
wd=1e-4
batch_size="$BATCH_SIZE"               # micro-batch: molecules per GPU per forward (x2 views); LUMI used 448
batch_size_valid="$batch_size"
update_freq="${UPDATE_FREQ:-1}"        # gradient accumulation steps
heads=64
masked_token_loss=1
masked_coord_loss=5
masked_dist_loss=10
x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01
contrastive_loss=10.0
contrastive_view_mode=independent      # HEART P3 (same as the smoke test); LUMI stock data path == identical
mask_prob=0.15
only_polar=0
noise_type="uniform"
noise=1.0
max_atoms=256
seed=1
warmup_steps=10000
max_steps=1000000
log_interval=100
validate_interval_updates=10000
save_interval_updates=10000
keep_interval_updates=10               # + checkpoint_last.pt + checkpoint_best.pt
stop_margin_minutes="${STOP_MARGIN_MINUTES:-30}"
min_gpu_mem_gb="${MIN_GPU_MEM_GB:-40}"
min_free_gb="${MIN_FREE_GB:-50}"

for v in batch_size update_freq n_gpu num_workers stop_margin_minutes min_free_gb; do
    case "${!v}" in ''|*[!0-9]*) die "$v must be a non-negative integer (got '${!v}')" ;; esac
done
[ "$batch_size" -ge 1 ] && [ "$update_freq" -ge 1 ] && [ "$n_gpu" -ge 1 ] || die "BATCH_SIZE, UPDATE_FREQ and N_GPU must be >= 1"
effective_batch=$(( batch_size * update_freq * n_gpu ))
log "micro-batch $batch_size x update-freq $update_freq x GPUs $n_gpu = effective $effective_batch molecules/update"
log "NT-Xent negatives come from one micro-batch per GPU (gathered across GPUs): $(( 2 * batch_size * n_gpu - 2 )) per positive pair"
[ "$update_freq" -gt 1 ] && warn "update-freq $update_freq: accumulation reproduces the effective batch for the reconstruction losses, not for the contrastive negatives"

precision_args=(--fp16 --fp16-init-scale 4 --fp16-scale-window 256)
USE_CPU=0
if [ "${HEART_LOCAL_CPU_TEST:-0}" = "1" ]; then
    precision_args=(--cpu)             # local CPU test suite only; never on CMRI
    USE_CPU=1
fi
read -r -a extra_args <<< "${HEART_EXTRA_ARGS:-}"
heart_reject_forbidden_args "${extra_args[@]}"

DRY_RUN="${HEART_DRY_RUN:-0}"
if [ -z "${SLURM_JOB_ID:-}" ] && [ "$DRY_RUN" != "1" ] && [ "$USE_CPU" != "1" ]; then
    die "full pretraining runs only under sbatch (use HEART_DRY_RUN=1 for a login-node pre-flight)"
fi

# >>> Gate 2: evidence that the smoke test passed >>>
smoke_dir="$(realpath -m "$HEART_SMOKE_RUN")"
smoke_pass=""
smoke_resume=""
for f in "$smoke_dir"/smoke_result-*.txt; do
    [ -f "$f" ] && grep -qx 'verdict=PASS' "$f" || continue
    n="$(sed -n 's/^max_update=//p' "$f")"
    [ "${n:-0}" -ge 10 ] 2>/dev/null || continue
    smoke_pass="$f"
    grep -q '^resume_from=[0-9]' "$f" && smoke_resume="$f"
done
[ -n "$smoke_pass" ] || die "no smoke_result-*.txt with verdict=PASS and max_update>=10 in $smoke_dir"
log "smoke evidence: $smoke_pass"
[ -n "$smoke_resume" ] && log "smoke resume evidence: $smoke_resume" \
    || warn "no passed smoke RESUME test in $smoke_dir (recommended before long runs)"

# >>> Pre-flight: real dataset, opened read-only >>>
heart_check_data

# >>> Fixed run directory: resubmission resumes >>>
save_dir="$OUTPUT_ROOT/full/$run_name"
heart_check_save_dir "$save_dir"
mkdir -p "$OUTPUT_ROOT/full"
free_gb="$(df -Pk "$OUTPUT_ROOT/full" | awk 'NR==2 {print int($4/1048576)}')"
log "free disk under $OUTPUT_ROOT/full: ${free_gb} GiB (need >= $min_free_gb)"
[ "$free_gb" -ge "$min_free_gb" ] || die "less than $min_free_gb GiB free for checkpoints (~0.6-0.8 GB each, up to 12 kept)"
if [ -n "${SLURM_CPUS_PER_TASK:-}" ] && [ "$num_workers" -ge "$SLURM_CPUS_PER_TASK" ]; then
    die "NUM_WORKERS=$num_workers must be < SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
fi

resume_from=""
if [ -f "$save_dir/checkpoint_last.pt" ]; then
    resume_from="$(heart_ckpt_updates "$save_dir/checkpoint_last.pt")" || die "cannot read $save_dir/checkpoint_last.pt"
    [ "$resume_from" -lt "$max_steps" ] || die "run already reached $resume_from updates (max $max_steps)"
    log "run dir    : $save_dir (RESUMING from checkpoint_last.pt @ $resume_from updates)"
else
    log "run dir    : $save_dir (new run, random initialisation)"
fi

# >>> Pre-flight: GPU / PyTorch / Uni-Core registry / time >>>
gpu_optional=0
[ "$DRY_RUN" = "1" ] && [ -z "${SLURM_JOB_ID:-}" ] && gpu_optional=1
heart_gpu_preflight "$USE_CPU" "$min_gpu_mem_gb" "$gpu_optional"
heart_registry_check
heart_train_entry
[ "$DRY_RUN" = "1" ] || mkdir -p "$save_dir"
heart_stop_time_args "$save_dir" "$stop_margin_minutes"
if [ -n "${SLURM_JOB_ID:-}" ] && [ "${#STOP_ARGS[@]}" -eq 0 ] && [ "$USE_CPU" != "1" ]; then
    die "could not derive --stop-time-hours from the SLURM time limit; refusing an unguarded full run"
fi

# >>> Training command >>>
cd "$MODEL_ROOT"
CMD=("${TRAIN_ENTRY[@]}" "$DATA_PATH" --user-dir ./unimol --train-subset train --valid-subset valid
    --num-workers "$num_workers" --data-buffer-size 20 --ddp-backend=c10d
    --task unimol_contrastive --loss unimol_contrastive --arch unimol_contrastive_base
    --encoder-attention-heads "$heads"
    --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay "$wd"
    --lr-scheduler polynomial_decay --lr "$lr" --warmup-updates "$warmup_steps" --total-num-update "$max_steps"
    --update-freq "$update_freq" --seed "$seed"
    "${precision_args[@]}" --tensorboard-logdir "$save_dir/tsb"
    --max-update "$max_steps" --log-interval "$log_interval" --log-format simple
    --save-interval-updates "$save_interval_updates" --validate-interval-updates "$validate_interval_updates"
    --keep-interval-updates "$keep_interval_updates" --no-epoch-checkpoints
    --masked-token-loss "$masked_token_loss" --masked-coord-loss "$masked_coord_loss" --masked-dist-loss "$masked_dist_loss"
    --x-norm-loss "$x_norm_loss" --delta-pair-repr-norm-loss "$delta_pair_repr_norm_loss"
    --mask-prob "$mask_prob" --noise-type "$noise_type" --noise "$noise" --max-atoms "$max_atoms"
    --batch-size "$batch_size" --batch-size-valid "$batch_size_valid"
    --save-dir "$save_dir" --tmp-save-dir "$save_dir" --only-polar "$only_polar" --mode train
    --dict-name dict.txt --contrastive-loss "$contrastive_loss" --contrastive-view-mode "$contrastive_view_mode"
    "${STOP_ARGS[@]}" "${extra_args[@]}")
if [ "$n_gpu" -gt 1 ]; then
    # torchrun exports RANK/WORLD_SIZE/LOCAL_RANK, which Uni-Core reads
    # (torch>=2 torch.distributed.launch passes --local-rank, which Uni-Core rejects).
    CMD=(torchrun --standalone --nproc_per_node="$n_gpu" "${CMD[@]}")
fi

echo "=== Command ==="
printf '%q ' "${CMD[@]}"; echo
if [ -n "$resume_from" ]; then
    allow=(); [ "${HEART_ALLOW_ARG_CHANGE:-0}" = "1" ] && allow=(--allow-change)
    "$PY" "$PRETRAIN_DIR/tools/check_resume_args.py" "$save_dir/checkpoint_last.pt" "${allow[@]}" -- "${CMD[@]}" \
        || die "resume settings differ from the checkpoint (HEART_ALLOW_ARG_CHANGE=1 to override, and document why)"
fi
if [ "$DRY_RUN" = "1" ]; then
    log "HEART_DRY_RUN=1: pre-flight passed; would write to $save_dir; not training"
    exit 0
fi

heart_provenance "$save_dir" "$script_path" "${CMD[@]}"
train_log="$save_dir/train-${SLURM_JOB_ID:-local}.log"
log "training log: $train_log"

# >>> Run in the background so SLURM signals reach this shell >>>
train_pid=""
on_signal() {
    log "received SIG$1: Uni-Core stops itself at --stop-time-hours (validate + checkpoint_last.pt)"
    if [ "$1" = TERM ] && [ -n "$train_pid" ] && kill -0 "$train_pid" 2>/dev/null; then
        log "SIGTERM: forwarding to training (pid $train_pid); resume from the last saved checkpoint"
        kill -TERM "$train_pid" 2>/dev/null || true
    fi
}
trap 'on_signal USR1' USR1
trap 'on_signal TERM' TERM
set +e
"${CMD[@]}" > >(tee -a "$train_log") 2>&1 &
train_pid=$!
rc=0
while :; do
    wait "$train_pid"
    rc=$?
    # wait returns >128 when interrupted by a trapped signal; keep waiting
    kill -0 "$train_pid" 2>/dev/null || break
done
trap - USR1 TERM

echo "=== Summary ==="
"$PY" "$PRETRAIN_DIR/tools/summarize_log.py" "$train_log" || log "summary reported FAIL items (see above)"
ls -lh "$save_dir"
if [ -f "$save_dir/checkpoint_last.pt" ]; then
    now="$(heart_ckpt_updates "$save_dir/checkpoint_last.pt" || echo '?')"
    log "checkpoint_last.pt @ $now / $max_steps updates"
    [ "$now" != "$max_steps" ] && log "not finished: resubmit the same command to resume from $now updates"
fi
log "training exit code: $rc"
exit "$rc"
