# Third-party code in `model/unimol/`

| Path | Origin | Upstream commit | License | HEART changes |
|---|---|---|---|---|
| `data/*.py` | [Uni-Mol](https://github.com/deepmodeling/Uni-Mol) `unimol/unimol/data/` | `90f52c4` | MIT (`LICENSE.Uni-Mol`) | `lmdb_dataset.py` (P1, P5), `add_2d_conformer_dataset.py` (P2); the rest is verbatim |
| `tasks/unimol_contrastive.py` | [LUMI-lab](https://github.com/bowenli-lab/LUMI-lab) `model/unimol/tasks/` | `611e02b` | GPL-3.0 (`LICENSE.LUMI-lab`); the file header keeps the original DP Technology MIT notice | P3 (view mode, shared 2D-conformer reader) |
| `tasks/__init__.py`, `losses/__init__.py` | LUMI-lab | `611e02b` | GPL-3.0 | verbatim |
| `models/unimol_contrastive.py` | LUMI-lab | `611e02b` | GPL-3.0 | P4 (padding-mask guard) |
| `models/transformer_encoder_with_pair.py` | LUMI-lab (byte-identical to Uni-Mol) | `611e02b` / `90f52c4` | GPL-3.0 / MIT | verbatim |
| `losses/unimol_contrastive.py` | LUMI-lab | `611e02b` | GPL-3.0 | verbatim (attribution header only) |
| `__init__.py`, `models/__init__.py` | adapted from LUMI-lab | `611e02b` | GPL-3.0 | trimmed to the modules used for pretraining |

Runtime dependency, not vendored: [Uni-Core](https://github.com/dptech-corp/Uni-Core), commit `ace6fae`, MIT.

**License consequence.** LUMI-lab is distributed under GPL-3.0. If you redistribute this tree (for example in a public GitHub repo), that distribution must comply with GPL-3.0: ship the license, keep the notices, and provide the source. The Uni-Mol-derived files keep their MIT notices.

**Data and weights.** No datasets, checkpoints, or pretrained weights are included. The Uni-Mol ligand pretraining data (`ligands.tar.gz`) is documented at https://github.com/deepmodeling/Uni-Mol/blob/main/unimol/README.md and is subject to its own terms of use. Cite Uni-Mol (Zhou et al., ICLR 2023) and LUMI-lab when publishing.
