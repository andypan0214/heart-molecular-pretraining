# HEART general molecular pretraining (LUMI / Uni-Mol / Uni-Core)

HEART trains a **randomly initialised**, 3D-aware molecular encoder on the Uni-Mol ligand conformer data. It uses LUMI-lab's original pretraining objectives: masked atom prediction, coordinate denoising, distance prediction and NT-Xent contrastive learning. The workflow is the LUMI one, with the Uni-Core trainer and the `unimol/` user module:

```
cd /home/ypan/heart_liver/model
unicore-train <ligands dir> --user-dir ./unimol \
    --task unimol_contrastive --loss unimol_contrastive --arch unimol_contrastive_base ...
```

`scripts/pretrain_heart.sh` assembles that command from a YAML config. `IMPLEMENTATION_MAP.md` lists every change relative to LUMI-lab, Uni-Mol and `CMRI_pretrian.txt`, and `IMPLEMENTATION_REPORT.md` records what has been verified.

> A 10-update pilot is an **engineering check**. It does not pretrain the model.

```
/home/ypan/heart_liver/model/            <- MODEL_ROOT (cwd for unicore-train)
├── pretrain/
│   ├── scripts/pretrain_heart.sh        SLURM launcher (pilot / resume / full)
│   ├── scripts/check_heart_env.sh       pre-flight (login node or sbatch GPU job)
│   ├── configs/pilot.yaml               1 GPU, batch 4, 10 updates
│   ├── configs/full_pretrain.yaml       prepared, guarded, NOT launched
│   ├── tools/                           inspector, log/ckpt verifiers, helpers
│   ├── tests/                           CPU tests on the synthetic 8-SMILES fixture
│   ├── requirements-heart.txt
│   └── logs/                            SLURM stdout/stderr (create before sbatch)
├── unimol/                              Uni-Core user module (sibling of pretrain/)
├── UniMolReference/ligand_data/ligands/ train.lmdb valid.lmdb dict.txt (read-only)
├── heart_training_runs/                 one directory per run (checkpoints, logs, tsb)
└── third_party/Uni-Core/                Uni-Core source used for pip install
```

---

## 0. Deploy the source to CMRI

Source: https://github.com/andypan0214/heart-molecular-pretraining (private). Its `model/` folder maps 1:1 onto `/home/ypan/heart_liver/model/`. Only `model/pretrain/` and `model/unimol/` are deployed. The dataset under `UniMolReference/` and the runs under `heart_training_runs/` are never touched, because rsync copies only these two directories and never deletes anything.

```bash
# 1. Clone into a checkout directory OUTSIDE the model root
module load git/2.42.0
cd /home/ypan/heart_liver
git clone https://github.com/andypan0214/heart-molecular-pretraining.git
#    private repo: git asks for a username (andypan0214) and a password. Paste a
#    GitHub fine-grained personal access token (read-only "Contents" access
#    to this repository only). Never put the token in the URL or in a script.
#    Later updates:  git -C /home/ypan/heart_liver/heart-molecular-pretraining pull

# 2. Preview, then deploy (files that differ are kept as *.bak-<date>; nothing is deleted)
SRC=/home/ypan/heart_liver/heart-molecular-pretraining/model
DST=/home/ypan/heart_liver/model
rsync -avn "$SRC/pretrain/" "$DST/pretrain/"
rsync -avn "$SRC/unimol/"   "$DST/unimol/"
rsync -av --backup --suffix=".bak-$(date +%Y%m%d)" "$SRC/pretrain/" "$DST/pretrain/"
rsync -av --backup --suffix=".bak-$(date +%Y%m%d)" "$SRC/unimol/"   "$DST/unimol/"
mkdir -p "$DST/pretrain/logs" "$DST/heart_training_runs"
chmod +x "$DST"/pretrain/scripts/*.sh
find "$DST/pretrain" "$DST/unimol" -maxdepth 1 | sort
```

Without git, download the source on a machine that has access and copy it over: run `git archive --format=tar.gz -o heart_src.tar.gz HEAD model` in a checkout, then `scp heart_src.tar.gz ypan@<CMRI-login-host>:/home/ypan/heart_liver/`. On CMRI, extract it with `mkdir -p heart-molecular-pretraining && tar -xzf heart_src.tar.gz -C heart-molecular-pretraining` and run the same `rsync` commands. LUMI-derived files are GPL-3.0; see `../unimol/THIRD_PARTY_NOTICES.md`.

## 1. Stage A: environment (inside `heart_gpu`; never the base or `py3_11` env)

```bash
module load miniconda3/py311_24.1.2
source /mnt/software/miniconda3/py311_24.1.2/etc/profile.d/conda.sh
conda activate /home/ypan/.conda/envs/heart_gpu
export PYTHONNOUSERSITE=1
python -c "import sys, torch; print(sys.executable, torch.__version__, torch.version.cuda)"
#   expected: .../envs/heart_gpu/bin/python 2.2.2+cu121 12.1  -- do NOT reinstall torch

# Python dependencies (torch is deliberately not listed)
python -m pip install -r /home/ypan/heart_liver/model/pretrain/requirements-heart.txt

# Uni-Core at the audited commit. Pure-Python build: no nvcc needed, fused
# kernels fall back to PyTorch ops.
mkdir -p /home/ypan/heart_liver/model/third_party && cd /home/ypan/heart_liver/model/third_party
UC=ace6fae1c8479a9751f2bb1e1d6e4047427bc134
# (a) with git:
module load git/2.42.0 && git clone https://github.com/dptech-corp/Uni-Core.git && git -C Uni-Core checkout $UC
# (b) or without git:
python -c "import urllib.request as u; u.urlretrieve('https://github.com/dptech-corp/Uni-Core/archive/$UC.zip','Uni-Core.zip')" \
  && unzip -q Uni-Core.zip && mv Uni-Core-$UC Uni-Core
python -m pip install --no-build-isolation --no-deps ./Uni-Core
command -v unicore-train && python -c "import unicore; print(unicore.__file__)"
```

Pre-flight checks:

```bash
bash /home/ypan/heart_liver/model/pretrain/scripts/check_heart_env.sh          # login node: paths, imports, registry
cd /home/ypan/heart_liver/model/pretrain && sbatch scripts/check_heart_env.sh  # GPU job: + nvidia-smi, CUDA, fp16
cat logs/heart-env-check-<jobid>.out                                            # must end with ALL CHECKS PASSED
```

Optional: run the CPU test suite on the HPC as well (about 1 minute, synthetic data only):

```bash
cd /home/ypan/heart_liver/model && python -m pytest pretrain/tests -q
```

## 2. Stage B: local CPU validation (already run on the development PC)

```bash
# from the repository root (Linux, macOS or Windows Git Bash; use .venv_heart_cpu/bin/python on Linux)
python3.11 -m venv .venv_heart_cpu
./.venv_heart_cpu/Scripts/python.exe -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
./.venv_heart_cpu/Scripts/python.exe -m pip install -r model/pretrain/requirements-heart.txt
# Uni-Core: pip install --no-build-isolation --no-deps <checkout>, or put the checkout on PYTHONPATH
cd model && PYTHONPATH=<Uni-Core checkout> ../.venv_heart_cpu/Scripts/python.exe -m pytest pretrain/tests -v
```

## 3. Stage C: GPU pilot on CMRI

Every command below runs on CMRI with `heart_gpu` active (see Stage A). Report a step as passed only after you have seen its output.

**C1. Paths and dependencies.** Run `check_heart_env.sh` as in Stage A. Then do a dry run of the launcher on the login node. It checks the config, data, Uni-Core registry and `unicore-train`, prints the exact command, and creates nothing:

```bash
cd /home/ypan/heart_liver/model/pretrain
HEART_DRY_RUN=1 bash scripts/pretrain_heart.sh configs/pilot.yaml
```

**C2. Real-LMDB compatibility (read-only; decodes about 10 records).** Run this on the login node, which takes under a minute, or inside `srun --partition=gpu --gres=gpu:1 --time=00:15:00 --pty bash`:

```bash
cd /home/ypan/heart_liver/model
python pretrain/tools/inspect_real_lmdb.py \
    /home/ypan/heart_liver/model/UniMolReference/ligand_data/ligands \
    --samples 5 --pipeline --json heart_training_runs/inspect_real_lmdb.json
```

The `==== SUMMARY ====` section must show PASS for: entry counts (18,837,320 / 98,844), contiguous keys, the `atoms`/`coordinates`/`smi` fields, `(n_atoms,3)` finite conformers, no elements missing from `dict.txt`, RDKit atom order (needed for the 2D view), and `pipeline[train|valid]`. Paste the summary back if any line FAILs.

**C3. Submit the 10-update pilot.** This uses 1 GPU, 6 CPUs, 55 GB and 30 minutes, with batch size 4:

```bash
cd /home/ypan/heart_liver/model/pretrain && mkdir -p logs
sbatch scripts/pretrain_heart.sh configs/pilot.yaml
```

**C4. Monitor.**

```bash
squeue -u $USER
tail -f /home/ypan/heart_liver/model/pretrain/logs/heart-pretrain-test-<jobid>.out
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
```

**C5. Training and validation losses.**

```bash
RUN=$(ls -dt /home/ypan/heart_liver/model/heart_training_runs/heart-gpu-pilot-* | head -1); echo $RUN
python /home/ypan/heart_liver/model/pretrain/tools/summarize_log.py $RUN/train-<jobid>.log --expect-updates 10
```

Expected: 10 `train_inner` rows, each showing loss, masked_token_loss, masked_coord_loss, masked_dist_loss, contrastive_loss, x_norm_loss and delta_pair_repr_norm_loss. Validation rows should appear at updates 5 and 10, with every check PASS. `gb_free` gives the remaining GPU memory. fp16 overflow steps are skipped by Uni-Core and do not count as updates.

**C6. Checkpoints.**

```bash
ls -lh $RUN                     # checkpoint_last.pt, checkpoint_best.pt, checkpoint_1_10.pt, provenance/, tsb/
cd /home/ypan/heart_liver/model
python pretrain/tools/verify_checkpoint.py $RUN/checkpoint_last.pt --expect-updates 10
```

**C7. Resume from the checkpoint.** This continues the same run to 12 updates:

```bash
cd /home/ypan/heart_liver/model/pretrain
RESUME_DIR=$RUN MAX_UPDATE=12 sbatch --job-name=heart-pretrain-resume scripts/pretrain_heart.sh configs/pilot.yaml
# after it finishes:
python /home/ypan/heart_liver/model/pretrain/tools/summarize_log.py $RUN/train-<resume jobid>.log \
    --expect-updates 12 --expect-resume-from 10
cd /home/ypan/heart_liver/model && python pretrain/tools/verify_checkpoint.py $RUN/checkpoint_last.pt --expect-updates 12
```

The log must contain `Loaded checkpoint …checkpoint_last.pt (epoch 1 @ 10 updates)`.

## 4. Stage D: full pretraining (prepared, not launched)

1. **GPU type and limits.** Run `sinfo -o "%P %G %l"` to find the H200 GRES name and the partition time limit.
2. **Memory probe** on the target GPU. This uses the pilot config with larger batches and a fresh run directory each time:
   ```bash
   cd /home/ypan/heart_liver/model/pretrain
   for bs in 64 128 256 448; do
     HEART_SET="batch-size=$bs batch-size-valid=$bs max-update=20 log-interval=5" \
       sbatch --job-name=heart-probe-bs$bs --gres=gpu:<h200-type>:1 scripts/pretrain_heart.sh configs/pilot.yaml
   done
   # per run: summarize_log.py -> gb_free column, and no "CUDA out of memory"
   ```
   Pick the largest batch size that leaves several GB free, then set `batch-size` and `batch-size-valid` in `configs/full_pretrain.yaml`. LUMI used 448 on an A100.
3. **Launch** (deliberately, by hand). `run_dir_mode: fixed` means every resubmission resumes the same run:
   ```bash
   cd /home/ypan/heart_liver/model/pretrain
   HEART_CONFIRM_FULL=1 sbatch --job-name=heart-pretrain-full --time=2-00:00:00 \
       --cpus-per-task=16 --mem=128G --gres=gpu:<h200-type>:1 \
       scripts/pretrain_heart.sh configs/full_pretrain.yaml
   # chain the next segment; it resumes automatically from checkpoint_last.pt:
   HEART_CONFIRM_FULL=1 sbatch --dependency=afterany:<previous jobid> --job-name=heart-pretrain-full \
       --time=2-00:00:00 --cpus-per-task=16 --mem=128G --gres=gpu:<h200-type>:1 \
       scripts/pretrain_heart.sh configs/full_pretrain.yaml
   ```

What the full configuration provides:

- **Lazy, read-only data access.** Reads use the 18.8M-record LMDB in place with O(1) start-up (patch P1). Nothing is copied. Each epoch is an `EpochShuffleDataset` permutation.
- **Clean stop before the time limit.** `--stop-time-hours` is derived from the SLURM time left minus 20 minutes. Uni-Core then validates, writes `checkpoint_last.pt` and exits normally.
- **Checkpointing and validation.** A checkpoint is saved every 10k updates and the newest 10 are kept, plus `last` and `best`. Validation runs every 10k updates over 250 batches, which covers about all of `valid.lmdb`. The run refuses to start on a GPU with less than 80 GB.
- **Storage.** The model has 48.1M parameters. A checkpoint holds fp16 weights, fp32 master weights and Adam moments, roughly 0.6–0.8 GB. Confirm the real size with `verify_checkpoint.py`. Expect about 12 checkpoints, or ≤ 10 GB per full run, plus small text and tensorboard logs. The pilot keeps 3 checkpoints. Delete old pilot directories when they are no longer needed: `rm -r heart_training_runs/heart-gpu-pilot-<…>`.

## 5. Notes

- **Contrastive views.** `contrastive-view-mode: independent` (HEART default) samples the conformer, mask and noise independently for the two views. Use `identical` to reproduce LUMI with stock Uni-Mol data modules exactly, where the views differ only by dropout. See IMPLEMENTATION_MAP finding 2.
- **Expected warnings.** Lines like `fused_softmax is not installed corrected` are expected: the CUDA extensions of Uni-Core are not compiled and PyTorch fallbacks are used.
- **Precision.** fp16 is used on both the T4 (sm_75) and the H200. bf16 is native on the H200 only.
- **Overrides.** These work for every submission: `HEART_SET="key=value ..."` (Uni-Core flags or `launcher.<key>`), `MAX_UPDATE`, `RESUME_DIR`, and `HEART_DRY_RUN=1`.
