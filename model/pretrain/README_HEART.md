# HEART general molecular pretraining on CMRI (LUMI / Uni-Mol / Uni-Core)

HEART trains a **randomly initialised**, 3D-aware molecular encoder on the Uni-Mol ligand conformer data. It uses LUMI-lab's original pretraining objectives: masked atom prediction, coordinate denoising, distance prediction and NT-Xent contrastive learning. Training runs on the Uni-Core trainer (`unicore-train`) with the `unimol/` user module, exactly like LUMI-lab `model/pretrain/pretrain.sh`:

```bash
cd "$MODEL_ROOT"
unicore-train "$DATA_PATH" --user-dir ./unimol \
    --task unimol_contrastive --loss unimol_contrastive --arch unimol_contrastive_base ...
```

There are two separate SLURM scripts. Each one writes out every training flag plainly in a LUMI-style parameter block:

| Script | Purpose | Resources | Guard |
|---|---|---|---|
| `scripts/pretrain_smoke.sh` | **30-minute engineering smoke test**: 10 real fp16 optimizer updates on the real train/valid LMDB, validation at 5 and 10, checkpoint save, inspection and verdict | 1 GPU (T4), 6 CPUs, 55G, 00:30:00 | never starts full training |
| `scripts/pretrain_heart.sh` | **Full pretraining**: LUMI schedule (10k warmup, 1M updates), checkpoints, clean stop before the time limit, resume on resubmission | 1 GPU (H200 recommended), 16 CPUs, 128G, 2 days per segment | `HEART_CONFIRM_FULL=1` + explicit `BATCH_SIZE` + a passed smoke run |

> The smoke test is an **engineering check**. It does not pretrain the model.

## Fixed CMRI layout

The old root `/home/ypan/heart_liver/model/` is obsolete. Everything now lives under `model_train/`, which has exactly three primary directories:

```text
/home/ypan/heart_liver/model_train/                     BASE
├── heart-molecular-pretraining/                        REPO (this git checkout)
│   ├── model/                                          MODEL_ROOT (cwd of unicore-train)
│   │   ├── pretrain/{scripts,configs,tools,tests,notebooks}/   PRETRAIN_DIR
│   │   └── unimol/{data,tasks,models,losses}/          Uni-Core --user-dir
│   └── runs/                                           OUTPUT_ROOT (git-ignored)
│       ├── slurm/        %x-%j.out / %x-%j.err   (tracked empty dir: exists after git pull)
│       ├── smoke/        heart-smoke-test-<date>-j<jobid>/   (one per smoke job)
│       ├── full/         heart-pretrain-full-v1/              (fixed; resubmission resumes)
│       └── env/          check_heart_env --fix logs
├── third_party/Uni-Core/                               UNICORE_SRC (existing checkout @ ace6fae)
└── UniMolReference/ligand_data/ligands/                DATA_PATH (read-only)
    ├── train.lmdb   (18,837,320 molecules)
    ├── valid.lmdb   (98,844 molecules)
    └── dict.txt
```

The scripts find `REPO` from their own location. Under `sbatch` they ask `scontrol` for the submitted path, because SLURM runs a spooled copy. Every path can be overridden with an environment variable: `HEART_REPO`, `HEART_BASE`, `HEART_DATA_PATH`, `HEART_OUTPUT_ROOT`, `HEART_UNICORE_SRC`, `HEART_CONDA_ENV`. Only the `#SBATCH --output/--error` lines are absolute, because SLURM cannot expand variables in them. Pass `sbatch --output=... --error=...` to change them.

Environment: `/home/ypan/.conda/envs/heart_gpu` (Python 3.11, torch 2.2.2+cu121 = CUDA 12.1 build). The smoke test runs on a Tesla T4 (sm_75, about 14.6 GiB), using fp16 because bf16 is not native on the T4. The CUDA toolkit of the torch build is `torch.version.cuda`, not the driver version that `nvidia-smi` shows.

---

## 1. Update the code on CMRI

```bash
cd /home/ypan/heart_liver/model_train/heart-molecular-pretraining
git status --short          # must be empty (or only your own deliberate edits)
git pull --ff-only          # public repository: no token needed
git log -1 --format='%H %s'
ls runs/slurm               # exists after the pull (tracked .gitkeep)
```

If `git` is missing on the login node, run `module load git/2.42.0`.

## 2. Environment: check, and repair once

```bash
cd /home/ypan/heart_liver/model_train/heart-molecular-pretraining/model/pretrain
bash scripts/check_heart_env.sh --fix     # repairs only what is missing; logs to runs/env/
bash scripts/check_heart_env.sh           # CHECK ONLY: must end with "ALL CHECKS PASSED"
```

* **Default mode** reads and prints only. It checks the three primary directories, the LMDB files, `runs/slurm` and the interpreter (`heart_gpu`, Python 3.11, `PYTHONNOUSERSITE=1`). It verifies the version **and the import path** of torch, numpy<2, lmdb, rdkit, scipy, tokenizers, tensorboardX, PyYAML, tqdm and ipykernel. It also checks that the Uni-Core source is at `ace6fae`, that `unicore` is imported from `heart_gpu`, that `unicore-train` resolves inside `heart_gpu/bin`, the task/model/arch/loss registry with `cwd = model/`, the Jupyter kernel and, if a GPU is allocated, CUDA and fp16.
* **`--fix`** runs `$CONDA_ENV/bin/python -m pip` with `PIP_USER=0` and a constraints file that pins the installed torch and `numpy<2`. It installs only the missing or incompatible packages, using the pins in `requirements-heart.txt`. It reinstalls Uni-Core from the **existing** `third_party/Uni-Core` checkout (`pip install --no-build-isolation --no-deps --force-reinstall`, pure-Python build), creates `runs/slurm`, registers the Jupyter kernel **Python 3.11 (heart_gpu)** with `PYTHONNOUSERSITE=1`, and then re-checks everything. Running it a second time changes nothing.
* It **never** installs or upgrades torch/CUDA. It never uses `--user` or `~/.local`, and never touches base conda or system Python. It never clones or checks out Uni-Core. Any of these problems is reported as `BLOCK` with a non-zero exit, and nothing is installed.
* Uni-Core's `setup.py` writes untracked build files (`unicore/version.py`, `build/`, `*.egg-info`) into the checkout. The checked-out commit does not change.
* GPU check on an allocated T4:
  ```bash
  srun --partition=gpu --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=00:10:00 bash scripts/check_heart_env.sh
  ```
* The training scripts never run pip. If `heart_gpu` breaks, they stop and tell you to run `--fix`.

Why this fixes the recurring "missing package" errors: the training scripts set `PYTHONNOUSERSITE=1`, which hides packages that were installed earlier with `pip install --user` into `~/.local`. `--fix` installs them into `heart_gpu` itself. That makes them visible in the terminal, in Jupyter (through the registered kernel) and in SLURM alike.

## 3. Notebook validation (Parts A–E, then F–G)

Open `model/pretrain/notebooks/HEART_HPC_Validation.ipynb` in CMRI Jupyter and select the kernel **Python 3.11 (heart_gpu)**. Run the cells in order.

| Part | Content | Needs |
|---|---|---|
| A | Three primary directories, LMDB files, `runs/` and `runs/slurm/` writable | CPU |
| B | Kernel is `heart_gpu`; versions and import paths; repair command (printed only) | CPU |
| C | Uni-Core, `unicore-train`, registry, real model (48,144,227 parameters) | CPU |
| D | A few real records read-only; dict; 3D + RDKit 2D conformer; paired independent views; padding | CPU |
| E | One real-batch forward, loss and backward (no optimizer step); NaN/Inf; GPU memory | GPU optional |
| F | `sbatch scripts/pretrain_smoke.sh` (guarded by `SUBMIT_SMOKE = True`); job ID, status, logs | SLURM |
| G | Losses, 10 updates, `checkpoint_last.pt` reload, resume to 12, fatal-error scan, final verdict | CPU / SLURM |

Do not submit until cell E5 prints `READY_FOR_SMOKE = True`.

Without Jupyter, the read-only real-LMDB inspector covers Part D:

```bash
cd /home/ypan/heart_liver/model_train/heart-molecular-pretraining/model
python pretrain/tools/inspect_real_lmdb.py /home/ypan/heart_liver/model_train/UniMolReference/ligand_data/ligands \
    --samples 5 --pipeline --json ../runs/inspect_real_lmdb.json
```

## 4. Smoke test: 10 updates, 30 minutes, 1 T4

```bash
cd /home/ypan/heart_liver/model_train/heart-molecular-pretraining/model/pretrain
HEART_DRY_RUN=1 bash scripts/pretrain_smoke.sh      # login node: pre-flight + exact command, creates nothing
JOBID=$(sbatch --parsable scripts/pretrain_smoke.sh); echo "smoke job $JOBID"
squeue -j "$JOBID"                                   # PENDING / RUNNING
LOG=/home/ypan/heart_liver/model_train/heart-molecular-pretraining/runs/slurm
tail -f "$LOG/heart-smoke-test-$JOBID.out"           # Ctrl-C stops tail, not the job
sacct -j "$JOBID" -X --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
```

At the end, the job prints the loss table and checkpoint check and writes `smoke_result-<jobid>.txt` (`verdict=PASS|FAIL`). Review the evidence:

```bash
RUN=$(grep -o 'run dir: .*' "$LOG/heart-smoke-test-$JOBID.out" | tail -1 | cut -d' ' -f3); echo "$RUN"
ls -lh "$RUN"                       # checkpoint_last.pt, checkpoint_best.pt, checkpoint_1_10.pt, provenance/, tsb/
cat "$RUN/smoke_result-$JOBID.txt"
cd /home/ypan/heart_liver/model_train/heart-molecular-pretraining/model
python pretrain/tools/summarize_log.py "$RUN/train-$JOBID.log" --expect-updates 10
python pretrain/tools/verify_checkpoint.py "$RUN/checkpoint_last.pt" --expect-updates 10
grep -E "CUDA out of memory|Traceback|DataLoader worker" "$LOG/heart-smoke-test-$JOBID".{out,err}   # expect no output
```

Expected: 10 `train_inner` rows with loss, masked_token_loss, masked_coord_loss, masked_dist_loss, contrastive_loss, x_norm_loss and delta_pair_repr_norm_loss. There should be validation rows at updates 5 and 10, `num_updates == 10` in `checkpoint_last.pt`, a strict model rebuild, and every line PASS. fp16 overflow steps are skipped by Uni-Core and do not count as updates. `gb_free` is the remaining GPU memory.

**Resume test** (same run directory, from update 10 to 12):

```bash
cd /home/ypan/heart_liver/model_train/heart-molecular-pretraining/model/pretrain
RJOB=$(RESUME_DIR="$RUN" MAX_UPDATE=12 sbatch --parsable --job-name=heart-smoke-resume scripts/pretrain_smoke.sh)
# after it finishes:
cat "$RUN/smoke_result-$RJOB.txt"                    # verdict=PASS, resume_from=10, max_update=12
grep "Loaded checkpoint" "$RUN/train-$RJOB.log"      # ... (epoch 1 @ 10 updates)
```

**If the T4 runs out of memory**, resubmit with a smaller micro-batch and record the change: `BATCH_SIZE=2 sbatch scripts/pretrain_smoke.sh`, then `BATCH_SIZE=1`. The model, objectives and loss weights stay the same.

## 5. Full pretraining: a separate decision, never automatic

Gate, in order:

1. The smoke test passed, including the resume test.
2. **VRAM probe** on the target GPU. Find the H200 GRES name and the partition time limit with `sinfo -o "%P %G %l"`. Then run short probe jobs with growing micro-batches:
   ```bash
   cd /home/ypan/heart_liver/model_train/heart-molecular-pretraining/model/pretrain
   for bs in 64 128 256 448; do
     BATCH_SIZE=$bs MAX_UPDATE=20 sbatch --gres=gpu:<h200-type>:1 --job-name=heart-probe-bs$bs scripts/pretrain_smoke.sh
   done
   # per job: summarize_log.py -> gb_free column, and no "CUDA out of memory"
   ```
3. Choose and write down:
   * **micro-batch** `BATCH_SIZE`: molecules per GPU per forward pass. Each molecule gives 2 views, so the model sees 2×BATCH_SIZE sequences.
   * **accumulation** `UPDATE_FREQ`.
   * **effective batch** = `BATCH_SIZE × UPDATE_FREQ × GPUs` molecules per optimizer update. The script prints it.

   Recommendation, still to be confirmed by the probe: use the H200 with `BATCH_SIZE=448 UPDATE_FREQ=1` if 448 fits with several GiB free. That equals LUMI's per-GPU batch, with no accumulation, so the NT-Xent negatives match. If it does not fit, use the largest probed micro-batch `b` with `UPDATE_FREQ = 448 / b`. This gives the same effective batch for the reconstruction losses, but **contrastive negatives come only from each micro-batch** (2b−2 per positive pair instead of 894). Document that as a deviation. A T4 is not a realistic device for 1M updates. Even if a micro-batch fits, the wall time would be many months.
4. Submit by hand (from `model/pretrain`):
   ```bash
   HEART_DRY_RUN=1 HEART_CONFIRM_FULL=1 BATCH_SIZE=<b> UPDATE_FREQ=<k> HEART_SMOKE_RUN="$RUN" \
       bash scripts/pretrain_heart.sh                    # login-node pre-flight
   HEART_CONFIRM_FULL=1 BATCH_SIZE=<b> UPDATE_FREQ=<k> HEART_SMOKE_RUN="$RUN" \
       sbatch --gres=gpu:<h200-type>:1 scripts/pretrain_heart.sh
   ```
5. Continue after each time limit. Run the same command again, or chain the next segment:
   `HEART_CONFIRM_FULL=1 BATCH_SIZE=<b> UPDATE_FREQ=<k> HEART_SMOKE_RUN="$RUN" sbatch --dependency=afterany:<jobid> --gres=gpu:<h200-type>:1 scripts/pretrain_heart.sh`

What `pretrain_heart.sh` does:

* **Before any GPU work**, it refuses to start without `HEART_CONFIRM_FULL=1`, an explicit `BATCH_SIZE`, and a `HEART_SMOKE_RUN` containing `verdict=PASS` with ≥ 10 updates. It also refuses outside `sbatch` (except `HEART_DRY_RUN=1`), a GPU with less than `MIN_GPU_MEM_GB` (40), less than `MIN_FREE_GB` (50) free disk, `NUM_WORKERS ≥ CPUs`, and an unknown SLURM time limit.
* **Fixed run directory** `runs/full/heart-pretrain-full-v1` (set with `RUN_NAME`). On resubmission, Uni-Core restores the model, optimizer, LR schedule, fp16 scaler and data-iterator position from `checkpoint_last.pt`. `tools/check_resume_args.py` refuses a resume whose batch size, LR, schedule, architecture or loss weights differ from the checkpoint, unless `HEART_ALLOW_ARG_CHANGE=1` is set.
* **Clean stop.** `--stop-time-hours` = cumulative training time + (time left − 30 min). Uni-Core then validates, writes `checkpoint_last.pt` and exits 0. `#SBATCH --signal=B:USR1@1800` is logged. SIGTERM is forwarded to training, which then resumes from the last saved checkpoint.
* **Checkpoints** are written every 10,000 updates; the newest 10 are kept, plus `checkpoint_last.pt` and `checkpoint_best.pt`. Each holds fp16 weights, fp32 master weights and Adam moments, about 0.6–0.8 GB. Validation runs every 10,000 updates on the whole `valid.lmdb`, as in LUMI.
* **Provenance** goes to `runs/full/<run>/provenance/job-<id>-<time>/`: the script used, the exact command, the git commit and status, sha256 of the sources, `pip freeze`, and the SLURM/CUDA environment.

## 6. Local tests (development PC, CPU only)

```bash
# Windows Git Bash, from the repository root; Uni-Core ace6fae checkout on PYTHONPATH
cd model && PYTHONPATH=<Uni-Core checkout> ../../.venv_heart_cpu/Scripts/python.exe -m pytest pretrain/tests -v
```

The tests use a synthetic 8-SMILES fixture and a tiny 2-layer CPU model. They run the real `unicore-train` code path through both SLURM scripts using the local-testing hooks `HEART_SKIP_ENV`, `HEART_LOCAL_CPU_TEST`, `HEART_TRAIN_ENTRY` and `HEART_EXTRA_ARGS`. Never use these hooks on CMRI. The tests prove nothing about the GPU, fp16, the real LMDB or SLURM.

## 7. Notes

* **Contrastive views.** `--contrastive-view-mode independent` is the HEART default, used in both scripts. The two views of a molecule get independently sampled conformers, masks and coordinate noise. With the stock Uni-Mol data modules, LUMI's code produces **identical** views that differ only by dropout. `identical` reproduces that exactly. See `IMPLEMENTATION_MAP.md` §6.
* **Expected warnings.** `fused_* is not installed corrected` is expected: Uni-Core's CUDA extensions are not compiled, and PyTorch fallbacks are used.
* **Configs.** `configs/pilot.yaml` and `configs/full_pretrain.yaml` are machine-readable records of the two scripts' parameter blocks. The scripts do not read them. `tests/test_scripts.py` fails if a script and its record disagree. The notebook builds its Part C/E arguments from `pilot.yaml`.
