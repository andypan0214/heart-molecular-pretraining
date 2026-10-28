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
grep -E "CUDA out of memory|Traceback|DataLoader worker [(]pid" "$LOG/heart-smoke-test-$JOBID".{out,err}   # expect no output
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

**If host (CPU) memory or the DataLoader fails**, for example with `MemoryError` in `multiprocessing` / `ForkingPickler`, use the low-memory mode `NUM_WORKERS=0 sbatch scripts/pretrain_smoke.sh`. The default is 5 workers, and values must be between 0 and CPUs−1. See section 8.

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

## 8. Troubleshooting: `MemoryError` in `multiprocessing` / `ForkingPickler.dumps` (smoke job 1011484)

**Symptom.** The job reaches `num_updates=1` with a finite loss, then logs repeated tracebacks like the following and makes no further progress until the time limit:

```text
MemoryError
  ... multiprocessing/queues.py ... _feed
  ... multiprocessing/reduction.py ... ForkingPickler.dumps(obj)
```

**What it means.** A DataLoader process failed to allocate memory while serializing a batch, or an index list, for the queue between the workers and the main process. That item is then lost. Uni-Core's DataLoader has no timeout, so the main process waits for it indefinitely, and the job looks stalled. The batch itself is small: about 1 KB of handles, with the tensors passed through `/dev/shm` (measured locally for batch size 4). The failure is therefore an allocation limit or exhaustion in that process or node, not one oversized batch.

**Possible causes. None is proven for job 1011484, because `sacct` recorded no RAM figures:**

1. **Host RAM growth across 6 processes.** The 5 workers are forked from a main process that already holds the CUDA context, the model and Uni-Core's per-epoch batch list (about 4.7M index arrays for 18.8M molecules at batch 4, roughly 1 GB). Copy-on-write and per-worker caches add to that.
2. **`/dev/shm` full or small.** Worker tensors travel through shared memory.
3. **A per-process virtual-memory limit** (`ulimit -v`, SLURM `VSizeFactor`). Each worker maps the ~150 GB `train.lmdb` read-only, which counts as virtual size, on top of the CUDA mappings inherited from the main process.
4. **An LMDB environment inherited across `fork()`.** This was a confirmed code defect and is now fixed as patch P5. Uni-Core reads the first batch in the main process, which opened the LMDB environment there, and all workers inherited it. LMDB forbids that. Each process now opens its own environment. It is not certain that this caused the `MemoryError`.

**What to do.**

```bash
cd /home/ypan/heart_liver/model_train/heart-molecular-pretraining
git pull --ff-only                                   # includes P5 + memory diagnostics
cd model/pretrain
NUM_WORKERS=0 HEART_DRY_RUN=1 bash scripts/pretrain_smoke.sh | grep -E "DataLoader workers|num-workers"
JOBID=$(NUM_WORKERS=0 sbatch --parsable scripts/pretrain_smoke.sh); echo $JOBID
```

`NUM_WORKERS=0` loads and collates data in the training process itself, using a background buffer thread, with no worker processes or inter-process queue. The batch size (4), model, losses and the 10-update, validation, checkpoint and resume workflow are unchanged. This is a **diagnostic, low-memory mode**. It removes the multiprocessing path; it does not by itself explain the underlying problem. Data loading becomes slower, which is fine for 10 updates but would starve the GPU in full training.

**Evidence each job now records:**

* At start, in the SLURM `.out`: `ulimit -v/-m/-n`, MemTotal/MemAvailable, the cgroup memory limit, `/dev/shm` size, and the DataLoader worker count.
* `$RUN/memory-<jobid>.tsv`: every 10 s, the number of processes, total and maximum per-process RSS, maximum VSZ, cgroup usage, `/dev/shm` used and MemAvailable. Change the interval with `HEART_MEM_INTERVAL`; `0` turns sampling off.
* At the end: peak RSS, VSZ and `/dev/shm`, the cgroup peak, and `num_workers` and `peak_rss_total_mb` in `smoke_result-<jobid>.txt`.

```bash
LOG=/home/ypan/heart_liver/model_train/heart-molecular-pretraining/runs/slurm
grep -A6 "=== CPU memory limits ===" $LOG/heart-smoke-test-$JOBID.out
sstat -j $JOBID.batch --format=JobID,MaxRSS,MaxVMSize,AveRSS       # while running
sacct -j $JOBID --format=JobID,State,Elapsed,MaxRSS,MaxVMSize,ReqMem,ExitCode
seff $JOBID 2>/dev/null                                           # if installed
RUN=$(grep -o 'run dir: .*' $LOG/heart-smoke-test-$JOBID.out | tail -1 | cut -d' ' -f3)
column -t $RUN/memory-$JOBID.tsv | tail -20
grep -A2 "=== CPU memory (sampled" $LOG/heart-smoke-test-$JOBID.out
grep -E "MemoryError|ForkingPickler|DataLoader worker [(]pid|Killed" $LOG/heart-smoke-test-$JOBID.{out,err}   # expect nothing
```

How to read the result:

| Outcome | Interpretation | Next step |
|---|---|---|
| `NUM_WORKERS=0` passes and RSS stays far below 55G | the multiprocessing path was involved (memory, shm, VSZ limit or the fixed LMDB inheritance) | resubmit with `NUM_WORKERS=2`, then 5, comparing `memory-<job>.tsv`. Use the largest count that passes, which matters for full training |
| `NUM_WORKERS=0` passes, but VSZ is near `ulimit -v` | virtual-memory limit | ask CMRI support about `VSizeFactor` / `ulimit -v`, and quote the TSV |
| a run with workers fails while `/dev/shm` used approaches its size | shared memory | ask for a larger `/dev/shm`, or use fewer workers |
| `NUM_WORKERS=0` also shows a `MemoryError` or a cgroup peak near 55G | main-process memory | send the `.out`, `.err` and TSV for analysis. Do not raise `--mem` blindly |

Nothing here changes the scientific settings. If you do change `--mem`, `NUM_WORKERS` or `BATCH_SIZE` for a run, record it next to the results; `smoke_result-<jobid>.txt` already stores the latter two.
