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
4. **`--tmp-save-dir` defaults to `./`.** Uni-Core writes each checkpoint there first and then copies it, which would put temporary files in the model root. The launcher sets it to the run directory.
5. **`--stop-time-hours` is cumulative** across resumes: it is stored in the checkpoint as `previous_training_time`. `tools/stop_time_hours.py` adds the per-job budget to that stored value.
6. **Uni-Core builds without CUDA extensions by default** at this commit. The fused kernels (layer norm, softmax-dropout, Adam, multi-tensor) fall back to PyTorch ops and print `fused_* is not installed corrected`. This is expected and harmless. `tokenizers` is imported unconditionally by `unicore.data`, so it is a hard dependency.
7. On Windows, Uni-Core's `os.rename(tmp, ckpt)` cannot overwrite an existing checkpoint. This does not affect Linux/HPC. The test-only entry point `tests/unicore_train_entry.py` aliases `os.rename` to `os.replace` on win32 only.

## 2. `model/unimol/`: reused vs. modified

| File | Status |
|---|---|
| `data/*.py` (16 files) | Copied verbatim from Uni-Mol, except P1 and P2 |
| `data/lmdb_dataset.py` | **P1**: length comes from `txn.stat()["entries"]` (O(1)) instead of enumerating all 18.8M keys at start-up. Upstream enumeration takes minutes of I/O and over 1 GB of RAM per process on a 150 GB LMDB, and `__getitem__` never uses the key list. Keys `b'0'` and `b'N-1'` are verified. A missing key raises `KeyError`. The environment is closed before DataLoader workers fork. `HEART_LMDB_FULL_KEY_SCAN=1` restores the full scan and also checks that keys are contiguous. |
| `data/add_2d_conformer_dataset.py` | **P2**: builds a new list instead of `coordinates.append(...)`. Upstream mutated the record cached by `LMDBDataset`, so a molecule could accumulate several copies of its 2D conformer. |
| `tasks/unimol_contrastive.py` | LUMI file plus **P3**. Adds `--contrastive-view-mode {independent,identical}` (CLI default `independent`) and `--view-seed-offset`. `independent` builds view 1 with seed `seed + 1000003`, so the two views are independently sampled conformer/mask/noise augmentations. `identical` reproduces upstream exactly. If `contrastive_view_mode` is absent from the args, as when the task is constructed from a bare namespace, behaviour falls back to `identical`. The LMDB reader and `Add2DConformerDataset` are built once and shared by both views. |
| `models/unimol_contrastive.py` | LUMI file plus **P4**. The coordinate head used `padding_mask.float()` although `forward()` sets `padding_mask=None` when a batch has no padding. That case raised `AttributeError`, and an all-False mask is now substituted. |
| `models/transformer_encoder_with_pair.py` | Verbatim (LUMI's copy is byte-identical to Uni-Mol's) |
| `losses/unimol_contrastive.py` | Verbatim LUMI loss (NT-Xent temperature 0.1, masked token, masked coordinate, masked distance, x-norm, delta-pair-norm). Only an attribution header was added |
| `__init__.py`, `models/__init__.py` | Trimmed from LUMI to the modules used for pretraining. LUMI's `import unimol.utils` (docking and conformer-generation helpers) is dropped |
| `tasks/__init__.py`, `losses/__init__.py` | Verbatim LUMI (auto-import every module in the directory) |

Registered names are unchanged: task `unimol_contrastive`, model `unimol_contrastive`, arch `unimol_contrastive_base`, loss `unimol_contrastive`. The architecture defaults are LUMI's: 15 layers, 512-dim embeddings, 2048 FFN, 64 attention heads (set by the launcher), **48,144,227 parameters**. Weights are randomly initialised (`init_bert_params`). The launcher refuses `--finetune-from-model`, `--restore-file`, and the `--reset-*` flags.

## 3. `CMRI_pretrian.txt` → `scripts/pretrain_heart.sh`

| `CMRI_pretrian.txt` | `pretrain_heart.sh` | Reason |
|---|---|---|
| `#!/bin/sh` | `#!/bin/bash`, `set -eo pipefail` | Script uses bash arrays; fail early |
| `--job-name=unimol-pretrain` | `heart-pretrain-test` | HEART naming |
| `--mem=55GB` | `--mem=55G` | SLURM unit syntax |
| `--time=3-00:00:00` | `00:30:00` | Pilot. The full run overrides this on the `sbatch` command line |
| `--output=logs/%x-%j.out` (relative) | absolute `/home/ypan/heart_liver/model/pretrain/logs/…` | Works from any submit directory. `logs/` must exist before `sbatch` |
| commented `module unload cuda-12.1 / load cuda-11.7`, `CUDA_HOME=/pkgs/cuda-11.7` | removed | Author-specific; torch 2.2.2+cu121 ships its own CUDA runtime |
| `module load miniconda3/py311_24.1.2`, `source …/conda.sh`, `conda activate …/heart_gpu` | kept (overridable via `HEART_CONDA_*`) | Already correct for CMRI |
| `OMP_NUM_THREADS=6` | `1` (plus `MKL_NUM_THREADS=1`) | Six DataLoader workers × 6 threads oversubscribed 6 cores |
| none | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` | GPU memory fragmentation; torch-2 NCCL variable name |
| hard-coded hyper-parameters | `configs/pilot.yaml` / `configs/full_pretrain.yaml` via `tools/yaml_to_env.py` | One launcher, two configs, CLI overrides |
| `batch_size=448`, `max_steps=1000000`, `warmup=10000`, intervals `10000`, `--log-interval 1000` | pilot: 4, 10, 3, 5, 1 | Engineering pilot |
| `save_dir=…/pretrain-${run_name}-${date_time}` (unquoted) | quoted, `…/<run_name>-<date>-j<jobid>` or fixed `<run_name>`. `RESUME_DIR` reuses a directory | Unique per job; resumable |
| `cp $0 $save_dir` | `provenance/job-<id>-<time>/`: config, script, command, sha256 of the source, `pip freeze`, environment | Reproducibility (text files only) |
| `cd /fs01/home/haotian/SDL-LNP/model` | `cd /home/ypan/heart_liver/model` (`HEART_MODEL_ROOT`) | Former author's path removed; `--user-dir ./unimol` resolves to the sibling of `pretrain/` |
| `python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=… $(which unicore-train)` | 1 GPU: `unicore-train …` directly. More than 1: `torchrun --standalone` | Finding 3 |
| `--num-workers 6` | 5 (capped at `SLURM_CPUS_PER_TASK - 1`) | Leaves one core for the main process |
| none | `--tmp-save-dir $save_dir` | Finding 4 |
| none | `--max-valid-steps`, `--batch-size-valid` | Bounded validation (`valid.lmdb` has 98,844 records) |
| none | `--stop-time-hours` from `squeue %L` minus a margin | Clean stop with `checkpoint_last.pt` before the SLURM kill (finding 5) |
| `--keep-interval-updates 10` | pilot 1, full 10 | Storage |
| none | `--contrastive-view-mode independent` | P3 |
| none | pre-flight: data files, output writability, GPU/CUDA/fp16, `min_gpu_mem_gb`, Uni-Core registry, `unicore-train` on PATH, full-run confirmation | Fail before using GPU time |
| none | post-run `tools/summarize_log.py` | Loss table and PASS/FAIL |
| unchanged | all LUMI objectives and weights (token 1, coord 5, dist 10, x-norm 0.01, delta-pair 0.01, contrastive 10), `mask_prob 0.15`, `noise uniform 1.0`, `only_polar 0`, `heads 64`, `lr 1e-5`, `wd 1e-4`, Adam β (0.9, 0.99), eps 1e-6, clip 1.0, polynomial decay, fp16 with init scale 4 / window 256, `--ddp-backend c10d`, `seed 1` | Method fidelity |

## 4. Local ↔ CMRI directory mapping

The repository root (`heart-molecular-pretraining/`) holds `model/`, which maps one-to-one to the HPC model root. Deploy only `model/pretrain/` and `model/unimol/` (see README_HEART.md §0):

| Local (repository) | CMRI HPC |
|---|---|
| `model/pretrain/` | `/home/ypan/heart_liver/model/pretrain/` |
| `model/unimol/` | `/home/ypan/heart_liver/model/unimol/` (sibling of `pretrain/`) |
| (not in repo) | `/home/ypan/heart_liver/model/UniMolReference/ligand_data/ligands/`: real data, read-only |
| (not in repo) | `/home/ypan/heart_liver/model/heart_training_runs/`: runs and checkpoints |
| (not in repo) | `/home/ypan/heart_liver/model/third_party/Uni-Core/`: Uni-Core source used for `pip install` |
| `CMRI_pretrian.txt` (original user script), CPU notebook (`andypan0214/molecular_pretraining`) | provenance only; not in this repository |

## 5. Tools added under `model/pretrain/`

| File | Purpose |
|---|---|
| `scripts/pretrain_heart.sh` | SLURM launcher (pilot, resume, full run) |
| `scripts/check_heart_env.sh` | Non-destructive pre-flight, on the login node or as an `sbatch` GPU job |
| `configs/pilot.yaml`, `configs/full_pretrain.yaml` | Run configurations |
| `tools/yaml_to_env.py` | Config → shell/CLI. Rejects launcher-owned and pretrained-weight options |
| `tools/check_registry.py` | Confirms Uni-Core discovers task/model/arch/loss through `--user-dir ./unimol` |
| `tools/inspect_real_lmdb.py` | Read-only check of the real LMDBs and `dict.txt`, plus the optional HEART pipeline on real records |
| `tools/make_fixture_lmdb.py` | Notebook 8-SMILES fixture (6 train / 2 valid) |
| `tools/summarize_log.py` | Train/valid loss tables, checkpoint list, resume detection, PASS/FAIL |
| `tools/verify_checkpoint.py` | Checkpoint keys, update count, finite weights, random-init provenance, strict rebuild |
| `tools/stop_time_hours.py` | SLURM time limit → cumulative `--stop-time-hours` |
| `tests/` | 48 pytest tests (CPU, synthetic fixture) |
