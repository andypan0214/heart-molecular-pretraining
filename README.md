# HEART general molecular pretraining

HEART pretrains a 3D-aware molecular encoder from **random initialisation** on the Uni-Mol ligand conformer dataset. It follows LUMI-lab's original pretraining stage: masked atom prediction, coordinate denoising, distance prediction, and NT-Xent contrastive learning. Training runs on the Uni-Core trainer with a LUMI-compatible `unimol/` user module.

> **Status.** Local CPU tests pass on a synthetic 8-molecule fixture (see `model/pretrain/IMPLEMENTATION_REPORT.md`). The CMRI GPU smoke test (10 updates on the real LMDB) and every other HPC step have **not been run yet**. No pretrained model exists.

## Layout (CMRI: `/home/ypan/heart_liver/model_train/`)

```text
/home/ypan/heart_liver/model_train/
├── heart-molecular-pretraining/          <- this repository
│   ├── model/
│   │   ├── pretrain/
│   │   │   ├── scripts/   pretrain_smoke.sh   30-min, 10-update GPU smoke test (T4)
│   │   │   │              pretrain_heart.sh   guarded full pretraining (resumable)
│   │   │   │              check_heart_env.sh  check (default) / --fix for the heart_gpu env
│   │   │   │              heart_lib.sh        shared shell helpers
│   │   │   ├── configs/   pilot.yaml, full_pretrain.yaml  (parameter records of the scripts)
│   │   │   ├── notebooks/ HEART_HPC_Validation.ipynb      (step-by-step HPC checks A-G)
│   │   │   ├── tools/     LMDB inspector, log/checkpoint verifiers, env/resume checks
│   │   │   ├── tests/     local CPU tests (synthetic fixture only)
│   │   │   ├── requirements-heart.txt, README_HEART.md, IMPLEMENTATION_MAP.md, IMPLEMENTATION_REPORT.md
│   │   └── unimol/        Uni-Core --user-dir: data/ tasks/ models/ losses/ + licences/notices
│   └── runs/              git-ignored outputs (runs/slurm/ is tracked empty so it exists before sbatch)
├── third_party/Uni-Core/  existing checkout @ ace6fae (installed into heart_gpu by check_heart_env.sh --fix)
└── UniMolReference/ligand_data/ligands/  train.lmdb, valid.lmdb, dict.txt (read-only, never copied)
```

The old project root `/home/ypan/heart_liver/model/` is obsolete and is not used by any script.

This repository contains no data, checkpoints, logs or environments.

## Quick start on CMRI

```bash
cd /home/ypan/heart_liver/model_train/heart-molecular-pretraining
git status --short
git pull --ff-only
cd model/pretrain
bash scripts/check_heart_env.sh --fix      # repair heart_gpu once (never touches torch/CUDA)
bash scripts/check_heart_env.sh            # must end with ALL CHECKS PASSED
# Jupyter: notebooks/HEART_HPC_Validation.ipynb, kernel "Python 3.11 (heart_gpu)", Parts A-E
sbatch scripts/pretrain_smoke.sh           # 10 updates, 30 min, 1 GPU; logs in ../../runs/slurm/
# full pretraining: separate, explicit decision (README_HEART.md section 5); never automatic
```

Full commands for monitoring, evidence review, the resume test, the VRAM probe and the full run are in [`model/pretrain/README_HEART.md`](model/pretrain/README_HEART.md).

## Sources and licences

| Component | Source | Licence |
|---|---|---|
| `model/unimol/tasks`, `models`, `losses` | [LUMI-lab](https://github.com/bowenli-lab/LUMI-lab) @ `611e02b` (modified; derived from Uni-Mol) | GPL-3.0 |
| `model/unimol/data` | [Uni-Mol](https://github.com/deepmodeling/Uni-Mol) @ `90f52c4` (2 small patches) | MIT |
| Trainer (dependency, not included) | [Uni-Core](https://github.com/dptech-corp/Uni-Core) @ `ace6fae` | MIT |
| Scripts, tools, tests, notebook under `model/pretrain/` | HEART | see below |

The LUMI-lab-derived files are GPL-3.0. This repository is public, so its distribution must comply with GPL-3.0: the licence texts and notices are kept in `model/unimol/` (`LICENSE.LUMI-lab`, `LICENSE.Uni-Mol`, `THIRD_PARTY_NOTICES.md`). Every HEART change to reused code is documented in `model/pretrain/IMPLEMENTATION_MAP.md`. No project-level licence has been chosen yet for the HEART-original files, pending licensing and institutional review. Please cite Uni-Mol (Zhou et al., ICLR 2023) and LUMI-lab when using this work.
