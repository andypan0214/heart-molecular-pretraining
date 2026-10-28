# HEART pretraining: implementation map

This page records what HEART reuses, what it modifies, and why. HEART follows LUMI-lab's **original pretraining stage** (`model/pretrain/pretrain.sh`). It does not implement continual pretraining, fine-tuning, or lipid-specific prediction.

## 1. Upstream sources audited

| Repository | Commit | License | Used for |
|---|---|---|---|
| [bowenli-lab/LUMI-lab](https://github.com/bowenli-lab/LUMI-lab) | `611e02b3f68144ddf45c8055b1198cd8ec28a7f2` | GPL-3.0 | `pretrain.sh`, contrastive task/model/loss, pair encoder |
| [deepmodeling/Uni-Mol](https://github.com/deepmodeling/Uni-Mol) | `90f52c41299a1a582da0f9765e9f87aa21faa16a` | MIT | `unimol/unimol/data/*`, `dict.txt` format |
| [dptech-corp/Uni-Core](https://github.com/dptech-corp/Uni-Core) | `ace6fae1c8479a9751f2bb1e1d6e4047427bc134` | MIT | trainer, CLI (`unicore-train`), checkpointing, fp16, logging. Installed as a dependency, not copied |

### Audit findings that shaped the implementation

1. **LUMI-lab never published `model/unimol/data/`.** `model/.gitignore` contains `data/`, but `unimol/__init__.py` and `tasks/unimol_contrastive.py` import `unimol.data`. This is why notebook cell 21 reported *"LUMI data module … FAIL"*. The data transforms are taken from Uni-Mol, which LUMI's own `unimol/readme.md` names as its source.
2. **The LUMI contrastive "views" are identical with the stock Uni-Mol data modules.** `ConformerSampleDataset` and `MaskPointsDataset` seed every random choice with `(seed, epoch, index)` and LRU-cache the result. `PairSampledNestedDictionaryDataset` calls `ds[index]` twice, so both views get the same conformer, mask and noise. The NT-Xent positives then differ only by dropout. `test_identical_view_mode_reproduces_upstream` demonstrates this. LUMI's private data code may have behaved differently, which cannot be checked. See P3.
3. **Uni-Core and `torch.distributed.launch`.** PyTorch ≥ 2.0's `torch.distributed.launch` passes `--local-rank=N`, but Uni-Core only declares `--local_rank`. A single GPU therefore calls `unicore-train` directly (the non-distributed `main()` path). Multi-GPU uses `torchrun`, which passes rank information through environment variables that Uni-Core reads (`LOCAL_RANK`, `RANK`, `WORLD_SIZE`, `MASTER_*`).
4. **`--tmp-save-dir` defaults to `./`.** Uni-Core writes each checkpoint there first and then copies it, which would put temporary files in the model root. Both scripts set it to the run directory.
5. **`--stop-time-hours` is cumulative** across resumes: it is stored in the checkpoint as `previous_training_time`. `tools/stop_time_hours.py` adds the per-job budget to that stored value.
6. **Uni-Core builds without CUDA extensions by default** at this commit. The fused kernels (layer norm, softmax-dropout, Adam, multi-tensor) fall back to PyTorch ops and print `fused_* is not installed corrected`. This is expected and harmless. `tokenizers` is imported unconditionally by `unicore.data`, so it is a hard dependency.
7. On Windows, Uni-Core's `os.rename(tmp, ckpt)` cannot overwrite an existing checkpoint. This does not affect Linux/HPC. The test-only entry point `tests/unicore_train_entry.py` aliases `os.rename` to `os.replace` on win32 only.

## 2. `model/unimol/`: reused vs. modified

| File | Status |
|---|---|
| `data/*.py` (16 files) | Copied verbatim from Uni-Mol, except P1 and P2 |
| `data/lmdb_dataset.py` | **P1**: length comes from `txn.stat()["entries"]` (O(1)) instead of enumerating all 18.8M keys at start-up. Upstream enumeration takes minutes of I/O and over 1 GB of RAM per process on a 150 GB LMDB, and `__getitem__` never uses the key list. Keys `b'0'` and `b'N-1'` are verified. A missing key raises `KeyError`. `HEART_LMDB_FULL_KEY_SCAN=1` restores the full scan and also checks that keys are contiguous. **P5** (2026-10-28, smoke job 1011484 investigation): Uni-Core reads the first training batch in the main process (`Trainer.get_train_iterator` → `first_batch`) before the DataLoader forks, so the lazily opened environment was inherited by every worker. LMDB forbids use across `fork()`, and the earlier claim that it was "closed before workers fork" was wrong. The environment is now opened per process (tracked by PID), dropped on pickling, and each read uses a short-lived transaction. Records and order are unchanged (`tests/test_lmdb_workers.py`). |
| `data/add_2d_conformer_dataset.py` | **P2**: builds a new list instead of `coordinates.append(...)`. Upstream mutated the record cached by `LMDBDataset`, so a molecule could accumulate several copies of its 2D conformer. |
| `tasks/unimol_contrastive.py` | LUMI file plus **P3**. Adds `--contrastive-view-mode {independent,identical}` (CLI default `independent`) and `--view-seed-offset`. `independent` builds view 1 with seed `seed + 1000003`, so the two views are independently sampled conformer/mask/noise augmentations. `identical` reproduces upstream exactly. If `contrastive_view_mode` is absent from the args, as when the task is constructed from a bare namespace, behaviour falls back to `identical`. The LMDB reader and `Add2DConformerDataset` are built once and shared by both views. |
| `models/unimol_contrastive.py` | LUMI file plus **P4**. The coordinate head used `padding_mask.float()` although `forward()` sets `padding_mask=None` when a batch has no padding. That case raised `AttributeError`, and an all-False mask is now substituted. |
| `models/transformer_encoder_with_pair.py` | Verbatim (LUMI's copy is byte-identical to Uni-Mol's) |
| `losses/unimol_contrastive.py` | Verbatim LUMI loss (NT-Xent temperature 0.1, masked token, masked coordinate, masked distance, x-norm, delta-pair-norm). Only an attribution header was added |
| `__init__.py`, `models/__init__.py` | Trimmed from LUMI to the modules used for pretraining. LUMI's `import unimol.utils` (docking and conformer-generation helpers) is dropped |
| `tasks/__init__.py`, `losses/__init__.py` | Verbatim LUMI (auto-import every module in the directory) |

Registered names are unchanged: task `unimol_contrastive`, model `unimol_contrastive`, arch `unimol_contrastive_base`, loss `unimol_contrastive`. The architecture defaults are LUMI's: 15 layers, 512-dim embeddings, 2048 FFN, 64 attention heads (set by both scripts), **48,144,227 parameters**. Weights are randomly initialised (`init_bert_params`). Both scripts refuse `--finetune-from-model`, `--restore-file`, and the `--reset-*` flags.

## 3. `CMRI_pretrian.txt` / LUMI `pretrain.sh` → `scripts/pretrain_smoke.sh` and `scripts/pretrain_heart.sh`

Both scripts keep the LUMI structure: SLURM header, SLURM info, conda activation, a plain parameter block (`lr=1e-5`, `heads=64`, …), `cd "$MODEL_ROOT"` and one explicit `unicore-train` command with every flag visible. Shared, non-scientific helpers live in `scripts/heart_lib.sh`: path resolution, environment activation, pre-flight checks and provenance.

| `CMRI_pretrian.txt` (LUMI-derived) | `pretrain_smoke.sh` | `pretrain_heart.sh` | Reason |
|---|---|---|---|
| `#!/bin/sh` | `#!/bin/bash`, `set -eo pipefail` | same | Bash arrays; fail early |
| `--job-name=unimol-pretrain` | `heart-smoke-test` | `heart-pretrain-full` | Two distinct jobs |
| `--cpus-per-task=6 --mem=55GB --gres=gpu:1 --partition=gpu` | `6 / 55G / gpu:1 / gpu` (CMRI pilot values) | `16 / 128G / gpu:1 / gpu`. GRES type on the `sbatch` line | Smoke on a T4, full on an H200 |
| `--time=3-00:00:00` | `00:30:00` | `2-00:00:00` per segment + `--signal=B:USR1@1800` | 30-minute limit for the smoke test only. Full runs resume across segments |
| `--output=logs/%x-%j.out` (relative) | absolute `…/model_train/heart-molecular-pretraining/runs/slurm/%x-%j.{out,err}` | same | Works from any submit directory. `runs/slurm/` is a tracked empty directory, so it exists before `sbatch` |
| LUMI: `--partition=a100 --qos=a100_bowang`, `module load cuda-11.7`, `CUDA_HOME=/pkgs/cuda-11.7`, `/pkgs/anaconda3` | removed | removed | Author-specific. torch 2.2.2+cu121 ships its own CUDA 12.1 runtime |
| `module load miniconda3/py311_24.1.2`, `source …/conda.sh`, `conda activate …/heart_gpu` | kept, and checks that `python`/`unicore-train` come from `heart_gpu` | same | Already correct for CMRI |
| `OMP_NUM_THREADS=6` | `1` (+ `MKL_NUM_THREADS=1`) | same | DataLoader workers × 6 threads oversubscribed the cores |
| none | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `PYTHONNOUSERSITE=1` | same | Memory fragmentation; torch-2 NCCL name; `~/.local` must not shadow `heart_gpu` |
| `batch_size=448`, `update_freq=1` | `4` (`BATCH_SIZE=2/1` documented for T4 OOM), `1` | `BATCH_SIZE` **required** (from the VRAM probe), `UPDATE_FREQ` (default 1); effective batch printed | 448 is LUMI's A100 value, not a tested T4 micro-batch |
| `warmup_steps=10000`, `max_steps=1000000` | `3`, `--total-num-update 20`, `--max-update 10` (`MAX_UPDATE` ≤ 20 for the resume test) | `10000`, `1000000` | Smoke vs LUMI schedule |
| `--log-interval 1000`, save/validate every `10000`, keep `10` | `1`, `5`/`5`, keep `1`, `--max-valid-steps 5` | `100`, `10000`/`10000`, keep `10`, full validation | Bounded smoke validation |
| `save_dir=…/pretrain-${run_name}-${date_time}` (unquoted) | `runs/smoke/heart-smoke-test-<date>-j<jobid>`; `RESUME_DIR` continues | fixed `runs/full/<RUN_NAME>`; resubmission resumes | Per-run outputs |
| `cp $0 $save_dir` | `provenance/job-<id>-<time>/`: script, command, git commit/status, sha256, `pip freeze`, environment | same | Reproducibility (text only) |
| `cd /fs01/home/haotian/SDL-LNP/model` | `cd "$MODEL_ROOT"` (derived from the script location; `HEART_REPO` override) | same | Former author's path removed |
| `python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=… $(which unicore-train)` | `unicore-train …` directly | 1 GPU directly; `N_GPU>1` → `torchrun --standalone` (untested) | Finding 3 |
| `--num-workers 6` | `5` | `SLURM_CPUS_PER_TASK − 1` | One core for the main process |
| none | `--tmp-save-dir`, `--batch-size-valid`, `--max-atoms 256`, `--dict-name dict.txt`, `--data-buffer-size` | same | Finding 4; explicit defaults |
| none | `--contrastive-view-mode independent` | same | P3, see §6 |
| none | `--stop-time-hours` from `squeue %L` − 5 min | − 30 min; refuses to run if the time limit is unknown | Clean stop with `checkpoint_last.pt` (finding 5) |
| none | pre-flight: data readable, output not inside the data, GPU/CUDA/fp16, registry, `unicore-train` in `heart_gpu`, forbidden extra args | + `HEART_CONFIRM_FULL=1`, smoke `verdict=PASS` evidence, GPU memory ≥ 40 GiB, free disk ≥ 50 GiB, sbatch-only, resume-settings check | Fail before using GPU time; no accidental full run |
| none | post-run `summarize_log.py` + `verify_checkpoint.py` → `smoke_result-<job>.txt` verdict; non-zero exit on FAIL | log summary + checkpoint position | Evidence-based PASS/FAIL |
| unchanged | all LUMI objectives and weights (token 1, coord 5, dist 10, x-norm 0.01, delta-pair 0.01, contrastive 10), `mask_prob 0.15`, uniform noise 1.0, `only_polar 0`, 64 heads, `lr 1e-5`, `wd 1e-4`, Adam β (0.9, 0.99), eps 1e-6, clip 1.0, polynomial decay, fp16 init scale 4 / window 256, `--ddp-backend=c10d`, seed 1 | same | Method fidelity |

`configs/pilot.yaml` and `configs/full_pretrain.yaml` record the same values in a machine-readable form. The scripts do not read them, so there is no YAML-to-shell indirection. `tests/test_scripts.py` compares each script's printed `unicore-train` command flag by flag with its YAML record.

## 4. Local ↔ CMRI directory mapping

The old executable root `/home/ypan/heart_liver/model/` is obsolete. The CMRI base is `/home/ypan/heart_liver/model_train/`, with three primary directories:

| Local | CMRI HPC |
|---|---|
| `pretrain/heart-molecular-pretraining/` (the only git working tree) | `model_train/heart-molecular-pretraining/` (`git pull --ff-only`, no rsync) |
| `…/model/pretrain/`, `…/model/unimol/` | `model_train/heart-molecular-pretraining/model/{pretrain,unimol}/` |
| `…/runs/` (git-ignored, `runs/slurm/.gitkeep` tracked) | SLURM logs, `smoke/`, `full/`, `env/` |
| (not in repo) | `model_train/third_party/Uni-Core/`: existing checkout @ `ace6fae`, installed into `heart_gpu` |
| (not in repo) | `model_train/UniMolReference/ligand_data/ligands/`: real data, read-only |
| `pretrain/model/`, `CMRI_pretrian.txt`, handoff documents, CPU notebook | historical references only; not in this repository |

## 5. Files under `model/pretrain/`

| File | Purpose |
|---|---|
| `scripts/pretrain_smoke.sh` | 30-minute, 10-update real-LMDB GPU smoke test, with resume test and verdict file |
| `scripts/pretrain_heart.sh` | Guarded, resumable full pretraining |
| `scripts/check_heart_env.sh` | Check-only by default; `--fix` repairs `heart_gpu` once (packages, Uni-Core from `third_party`, `runs/slurm`, Jupyter kernel) |
| `scripts/heart_lib.sh` | Shared helpers: paths, env activation, pre-flight, provenance |
| `configs/pilot.yaml`, `configs/full_pretrain.yaml` | Parameter records of the two scripts (checked by tests; used by the notebook) |
| `notebooks/HEART_HPC_Validation.ipynb` | Step-by-step HPC validation, Parts A–G |
| `tools/heart_env_check.py` | Interpreter, package version and import-path checks, Uni-Core origin/commit, repair list |
| `tools/check_resume_args.py` | Refuses a resume with different scientific settings |
| `tools/yaml_to_env.py` | YAML record → `unicore-train` flags (tests, notebook). Rejects launcher-owned and pretrained-weight options |
| `tools/check_registry.py` | Confirms Uni-Core discovers task/model/arch/loss through `--user-dir ./unimol` |
| `tools/inspect_real_lmdb.py` | Read-only check of the real LMDBs and `dict.txt`, plus the HEART pipeline on real records |
| `tools/make_fixture_lmdb.py` | Synthetic 8-SMILES fixture (6 train / 2 valid) |
| `tools/summarize_log.py` | Train/valid loss tables, checkpoint list, resume detection, PASS/FAIL |
| `tools/verify_checkpoint.py` | Checkpoint keys, update count, finite weights, random-init provenance, strict rebuild |
| `tools/stop_time_hours.py` | SLURM time left → cumulative `--stop-time-hours` |
| `tests/` | Local CPU tests (synthetic fixture). `run_notebook_cells.py` executes notebook parts as a kernel stand-in |

## 6. Scientific comparison: HEART vs upstream LUMI `unimol_contrastive`

Compared file by file against LUMI-lab `611e02b` (`diff -w` against the public upstream repository).

| Component | LUMI-lab | HEART | Scientific effect |
|---|---|---|---|
| Model `unimol_contrastive` / arch `unimol_contrastive_base` | 15 layers, 512 dim, FFN 2048, pair encoder, LM/coord/dist heads, projection head | identical; 64 heads set by both scripts (LUMI `--encoder-attention-heads 64`) | none. **48,144,227 parameters** |
| Model forward, batch without padding | `padding_mask.float()` on `None`, so `AttributeError` | all-False mask substituted (P4) | none; prevents a crash only |
| Loss `unimol_contrastive` | NT-Xent (temperature 0.1, cosine similarity, `GatherLayer` across GPUs) + masked token / coord / dist + x-norm + delta-pair-norm; normalisation by sample size | **verbatim** | none |
| Task data pipeline | Uni-Mol transforms (not published by LUMI; their `.gitignore` drops `data/`) | Uni-Mol `90f52c4` transforms + P1 (O(1) LMDB length) + P2 (no mutation of cached records); LMDB reader and 2D-conformer step built once and shared by both views | P1 none. P2 removes a bug where the 2D conformer could be appended to a cached record more than once, which would bias conformer sampling towards 2D |
| Contrastive views | `PairSampledNestedDictionaryDataset` calls the **same** pipeline twice per index | `--contrastive-view-mode independent` (default): second pipeline seeded `seed + 1000003`; `identical` reproduces LUMI | **yes**, see below |

**Independent vs identical views.** In Uni-Mol's data modules, `ConformerSampleDataset` (conformer choice), `MaskPointsDataset` (masked atoms, random tokens, coordinate noise) and `CroppingDataset` (only for molecules with more than 256 heavy atoms) seed every random choice with `(seed, epoch, index)` and LRU-cache the result. LUMI's pair dataset calls `ds[index]` twice through one pipeline, so **view 0 and view 1 are byte-identical**: same conformer, same mask, same noise. The NT-Xent positive pair then differs only through dropout inside the network, which makes it a SimCSE-style objective. `tests/test_data_pipeline.py::test_identical_view_mode_reproduces_upstream` demonstrates this. HEART's `independent` mode builds the second view with a different conformer and mask/noise seed. Cropping keeps the task seed in both views, as in LUMI. The positives become two independently sampled augmentations of the same molecule: possibly different 3D/2D conformers, different masked atoms and different coordinate noise, as in SimCLR. Both modes use the same reconstruction targets, loss weights and normalisation. In both, the masked-token/coord/dist losses are computed on both views. LUMI's unpublished data code may have done something else, which cannot be checked. HEART keeps the previously tested `independent` default in **both** scripts and does not switch silently. `--contrastive-view-mode identical` is a one-flag change for an exact LUMI reproduction and should be reported if used.

**Batch semantics.** LUMI used `batch_size=448` per GPU with `n_gpu=$SLURM_GPUS_ON_NODE`, and NT-Xent gathered negatives across GPUs (`GatherLayer`). HEART's full script prints micro-batch × `UPDATE_FREQ` × GPUs. With `UPDATE_FREQ > 1`, the reconstruction losses see the same effective batch, but contrastive negatives come only from each micro-batch. That is a documented deviation, not an equivalence.
