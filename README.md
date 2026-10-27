# HEART general molecular pretraining

HEART pretrains a 3D-aware molecular encoder from **random initialisation** on the Uni-Mol ligand conformer dataset. It follows LUMI-lab's original pretraining stage: masked atom prediction, coordinate denoising, distance prediction, and NT-Xent contrastive learning. Training runs on the Uni-Core trainer with a LUMI-compatible `unimol/` user module.

> **Status.** The pipeline has passed 48 local CPU tests on a synthetic 8-molecule fixture. HPC validation (environment, real LMDB, 10-update GPU pilot, resume) has **not been run yet**. No pretrained model exists. See `model/pretrain/IMPLEMENTATION_REPORT.md`.

## Layout

```text
model/
├── pretrain/                 -> /home/ypan/heart_liver/model/pretrain/
│   ├── scripts/              pretrain_heart.sh (SLURM launcher), check_heart_env.sh (pre-flight)
│   ├── configs/              pilot.yaml (1 GPU, batch 4, 10 updates), full_pretrain.yaml (guarded)
│   ├── tools/                real-LMDB inspector, log/checkpoint verifiers, helpers
│   ├── tests/                48 CPU tests (synthetic fixture only)
│   ├── README_HEART.md       installation and execution commands (Stages A-D)
│   ├── IMPLEMENTATION_MAP.md what was reused vs modified, and why
│   ├── IMPLEMENTATION_REPORT.md
│   └── requirements-heart.txt
└── unimol/                   -> /home/ypan/heart_liver/model/unimol/   (Uni-Core --user-dir)
    ├── data/  tasks/  models/  losses/
    └── THIRD_PARTY_NOTICES.md, LICENSE.Uni-Mol, LICENSE.LUMI-lab
```

This repository contains no data, checkpoints, logs or environments. The Uni-Mol ligand LMDBs stay in `/home/ypan/heart_liver/model/UniMolReference/ligand_data/ligands/` on CMRI and are only ever opened read-only.

## Deploy on CMRI HPC

```bash
module load git/2.42.0
cd /home/ypan/heart_liver
git clone https://github.com/andypan0214/heart-molecular-pretraining.git   # private: use a read-only fine-grained PAT when prompted

SRC=/home/ypan/heart_liver/heart-molecular-pretraining/model
DST=/home/ypan/heart_liver/model
rsync -avn "$SRC/pretrain/" "$DST/pretrain/" ; rsync -avn "$SRC/unimol/" "$DST/unimol/"     # preview
rsync -av --backup --suffix=".bak-$(date +%Y%m%d)" "$SRC/pretrain/" "$DST/pretrain/"
rsync -av --backup --suffix=".bak-$(date +%Y%m%d)" "$SRC/unimol/"   "$DST/unimol/"
mkdir -p "$DST/pretrain/logs" "$DST/heart_training_runs" && chmod +x "$DST"/pretrain/scripts/*.sh
```

Only `pretrain/` and `unimol/` are copied, and nothing is deleted, so the dataset and existing runs are never overwritten. Next, install Uni-Core into `heart_gpu`, run `scripts/check_heart_env.sh`, and submit the pilot, all as described in [`model/pretrain/README_HEART.md`](model/pretrain/README_HEART.md).

## Sources and licences

| Component | Source | Licence |
|---|---|---|
| `model/unimol/tasks`, `models`, `losses` | [LUMI-lab](https://github.com/bowenli-lab/LUMI-lab) @ `611e02b` (modified; derived from Uni-Mol) | GPL-3.0 |
| `model/unimol/data` | [Uni-Mol](https://github.com/deepmodeling/Uni-Mol) @ `90f52c4` (2 small patches) | MIT |
| Trainer (dependency, not included) | [Uni-Core](https://github.com/dptech-corp/Uni-Core) @ `ace6fae` | MIT |

The LUMI-lab-derived files are GPL-3.0. Any distribution of this repository, including making it public, must comply with GPL-3.0, and keeping the repository private does not remove those obligations for copies shared with others. Every HEART change is documented in `model/pretrain/IMPLEMENTATION_MAP.md` and `model/unimol/THIRD_PARTY_NOTICES.md`. No project-level licence has been chosen yet, pending licensing and institutional review. Please cite Uni-Mol (Zhou et al., ICLR 2023) and LUMI-lab when using this work.
