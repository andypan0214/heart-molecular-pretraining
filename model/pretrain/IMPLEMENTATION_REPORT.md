# HEART pretraining: implementation report

## Update 2026-10-28: model_train paths, two LUMI-style scripts, heart_gpu repair, HPC notebook

Scope is unchanged: LUMI-lab's original general pretraining stage, random initialisation, Uni-Mol ligand data, Uni-Core trainer. **No model has been pretrained, and no HPC/GPU step has been run.**

### Changes

| Item | Where |
|---|---|
| Fixed CMRI layout `/home/ypan/heart_liver/model_train/{heart-molecular-pretraining,third_party/Uni-Core,UniMolReference}`; paths derived from the script location (`scontrol` under sbatch) with `HEART_*` overrides; the old `/home/ypan/heart_liver/model/` root is gone from every script, config, tool and notebook | `scripts/heart_lib.sh`, all docs |
| `scripts/pretrain_smoke.sh` (new): 30 min / 1 GPU / 6 CPU / 55G; real train/valid LMDB; 10 fp16 updates; validation and checkpoints at 5/10; post-run log + checkpoint verification → `smoke_result-<job>.txt`; `RESUME_DIR` + `MAX_UPDATE` resume test; `BATCH_SIZE` override for T4 OOM | `scripts/` |
| `scripts/pretrain_heart.sh` (rewritten): full LUMI schedule; gates (`HEART_CONFIRM_FULL=1`, explicit `BATCH_SIZE`, passed smoke evidence, sbatch-only, GPU memory, disk, time limit); fixed resumable run directory; refusal of changed settings on resume; clean stop before the limit; USR1/TERM handling; effective-batch reporting | `scripts/`, `tools/check_resume_args.py` |
| Both scripts: LUMI-style parameter block + one explicit `unicore-train` command (no YAML indirection). The YAML files are now parameter records, checked flag by flag against the scripts | `configs/`, `tests/test_scripts.py` |
| `scripts/check_heart_env.sh`: check-only default; `--fix` installs only missing/incompatible packages into `heart_gpu` (env python, `PIP_USER=0`, torch/numpy constraints) and reinstalls Uni-Core from the existing `third_party/Uni-Core` @ `ace6fae`; Jupyter kernel; `runs/slurm`; logs; blockers never "fixed" | `scripts/`, `tools/heart_env_check.py`, `requirements-heart.txt` |
| `notebooks/HEART_HPC_Validation.ipynb`: 49 cells (38 code), Parts A–G with PASS/FAIL bookkeeping, guarded `sbatch`, evidence-only verdict, separate full-training gate | `notebooks/` |
| `runs/` git-ignored, except the tracked empty `runs/slurm/` so SLURM log paths exist before the first `sbatch` | `.gitignore` |
| HEART vs LUMI scientific comparison (task/model/loss), independent vs identical views, batch semantics | `IMPLEMENTATION_MAP.md` §6 |

`model/unimol/` is **unchanged** in this update (only a path comment in `__init__.py`). P1–P4 and the registration names are as before.

### Locally tested (Windows 11, Git Bash, Python 3.11.7, torch 2.2.2+cpu, Uni-Core `ace6fae` on PYTHONPATH)

`python -m pytest pretrain/tests -v`: **95 passed, 0 failed, 0 skipped** (118 s).

| File | Tests | Covers |
|---|---|---|
| `test_data_pipeline.py`, `test_model_loss.py`, `test_registry.py`, `test_fixture_data.py`, `test_inspect_lmdb.py`, `test_unicore_train_e2e.py` | 34 | unchanged from 2026-10-27 (pipeline, patches, views, model/loss, registry, real `unicore-train` CLI and resume) |
| `test_launcher.py` | 7 | YAML records: CMRI pilot values, LUMI weights, model_train paths, forbidden keys, overrides |
| `test_scripts.py` | 37 | `bash -n` + LF for all 4 scripts; exact smoke SBATCH directives; full directives; no LUMI-author settings or pip in training scripts; LUMI-style parameter block; smoke never launches full; no obsolete root in scripts/configs/tools/notebook; `.gitignore`; **smoke dry-run command == `pilot.yaml`** and **full dry-run command == `full_pretrain.yaml`** flag by flag; bad overrides rejected; missing data; output inside data; **smoke run (2 updates) → verify → verdict → resume to 4 → refusal at 4**; full refusals (confirmation, batch, smoke evidence, non-PASS / < 10-update evidence, outside sbatch) with nothing created; accumulation reporting; **full run 2 updates → resume to 3 → refuses changed batch size**; `check_heart_env.sh` help, bad argument, check mode creates nothing, `--fix` refuses to install over blockers |
| `test_env_tools.py` | 11 | requirements parsing and pins, version ranges, prefix containment, git HEAD (detached/branch/packed), foreign interpreter → BLOCK with no torch repair, resume-argument parsing |
| `test_notebook.py` | 6 | nbformat 4, `heart_gpu` kernel, unique ids, every code cell compiles; ≥ 30 short labelled cells, Parts A–G; no magics/pip, submit guards off, never submits full training, read-only LMDB; **Parts 0–E executed on the fixture** (real 48,144,227-parameter model, 2D conformer, independent views, padding, forward/backward finite); **Part G executed on a real local 10-update smoke run** (G1–G4 PASS); Part G without a job records nothing |

The previous 48 tests: the 34 unchanged tests still pass. The 14 tests of the old single launcher (`test_launcher.py`) were ported to the new scripts with the same assertions or stricter ones (`test_launcher.py` + `test_scripts.py`). None were deleted to obtain a PASS.

Locally, `check_heart_env.sh` and notebook Part B correctly report the dev venv as **not** `heart_gpu`-compatible: `torch 2.2.2+cpu` is a BLOCK, there is no `ipykernel`, and Uni-Core is on PYTHONPATH rather than installed. These are expected local FAILs, not HPC results.

### Not run (requires CMRI; only you can run these)

`check_heart_env.sh --fix` / check on CMRI and on a GPU node; notebook Parts A–E on the real LMDB and kernel; `sbatch scripts/pretrain_smoke.sh` (fp16, T4, real data, 10 updates); the resume job to 12; the H200 VRAM probe; any full run. `shellcheck` was not available locally. The SLURM-only code paths (`scontrol` script path recovery, `squeue` time-left → `--stop-time-hours`, `--signal` trap, `sacct` polling in the notebook) are therefore untested.

---

## 2026-10-27 (previous delivery)

Scope: LUMI-lab's original **general molecular pretraining** stage. Random initialisation, Uni-Mol ligand data, Uni-Core trainer. No continual pretraining, fine-tuning or lipid-specific prediction. **No model has been pretrained.** Every training run in this report was a few updates on 8 synthetic molecules on a CPU.

### 1. Implemented

| Item | Where |
|---|---|
| LUMI contrastive task/model/loss with registrations unchanged (`unimol_contrastive`, `unimol_contrastive_base`) | `model/unimol/{tasks,models,losses}` |
| Missing `unimol/data/` package taken from Uni-Mol (LUMI never published it) | `model/unimol/data` |
| Four minimal patches: P1 O(1) LMDB length; P2 2D-conformer cache mutation; P3 independent contrastive views (option; `identical` = upstream); P4 no-padding batch crash | see `IMPLEMENTATION_MAP.md` §2 |
| Corrected SLURM launcher from `CMRI_pretrian.txt` (paths, shebang, mem, time, logs, env, single-GPU launch, `tmp-save-dir`, bounded validation, time-limit guard, provenance, resume, guards against pretrained weights and accidental full runs) | `scripts/pretrain_heart.sh` |
| Pre-flight (login node or GPU job) | `scripts/check_heart_env.sh` |
| Pilot config (1 GPU, batch 4, 10 updates, validation and checkpoints at 5 and 10) and a full config (guarded, resumable, fixed run directory) | `configs/` |
| Read-only real-LMDB and `dict.txt` inspector, including the HEART pipeline on real records | `tools/inspect_real_lmdb.py` |
| Log summariser, checkpoint verifier and rebuilder, stop-time helper, registry check, fixture builder | `tools/` |
| 48 CPU tests | `tests/` |
| Licences and attributions (LUMI GPL-3.0, Uni-Mol/Uni-Core MIT) | `model/unimol/THIRD_PARTY_NOTICES.md`, `LICENSE.*` |

### 2. Locally tested (Windows 11, Python 3.11.7, torch 2.2.2+cpu, RDKit 2024.09.6, NumPy 1.26.4, Uni-Core `ace6fae` from source)

`python -m pytest pretrain/tests`: **48 passed, 0 failed.**

| Check | Result |
|---|---|
| Notebook fixture reproduced: 8 valid SMILES, `C1CC` rejected, ETKDGv3 ×2 conformers, 6 train / 2 valid; atom counts and split identical to the notebook's saved tables | PASS |
| LMDB round trip, numeric ASCII keys, `(N,3)` finite float32 conformers | PASS |
| Uni-Core discovers task, model, arch and loss through `--user-dir ./unimol`, and `unimol` resolves to `model/unimol` | PASS |
| P1 length equals a full key scan; non-contiguous keys rejected; P2 no record mutation | PASS |
| Batch shapes `[2B,L]`, `[2B,L,3]`, `[2B,L,L]`, with L a multiple of 8; BOS/EOS/padding positions; zero geometry on padding; symmetric distances; edge types | PASS |
| Hydrogens removed (`only_polar 0`); per-row length equals heavy-atom count; mask targets consistent | PASS |
| Views paired `[v0 of B \| v1 of B]`; `identical` mode gives byte-identical views (upstream behaviour confirmed); `independent` mode gives differing views | PASS |
| Random-init forward: all 6 LUMI objectives finite, total equals the weighted sum | PASS |
| Backward: finite, non-zero gradients in the embeddings, both encoder layers, Gaussian/edge layers, LM head, coordinate head, distance head, projection head | PASS |
| Contrastive-only loss trains `proj_head`; optimizer steps change weights; loss falls on a fixed batch | PASS |
| P4: batch without padding runs | PASS |
| `state_dict` and optimizer checkpoint round trip gives identical outputs | PASS |
| LUMI base architecture (15×512, FFN 2048, 64 heads) builds: **48,144,227 parameters**, finite forward on CPU | PASS |
| Real `unicore-train` CLI driven by `pilot.yaml` (tiny-arch CPU overrides): 3 updates, validation, `checkpoint_last`/`best`/interval files, `keep-interval-updates` retention | PASS |
| Resume through Uni-Core: loads @ 3 updates, continues to 5; cumulative training time grows | PASS |
| `summarize_log.py` and `verify_checkpoint.py` (strict rebuild from stored args) | PASS |
| **Launcher run end to end in bash**: pre-flight, fresh run, provenance, summary; then `RESUME_DIR` + `MAX_UPDATE` resume | PASS |
| Launcher guards: full config refused without `HEART_CONFIRM_FULL=1`; missing dataset refused; output inside dataset refused; dry run has no side effects; forbidden config keys (`finetune-from-model`, `restore-file`, `save-dir`, `user-dir`) rejected | PASS |
| Inspector on the fixture (read-only; data-dir mtimes unchanged; refuses to write a report inside the data directory) | PASS |
| `bash -n` and LF line endings for both shell scripts | PASS |
| Notebook cells 25 (loader), 27 (forward/backward) and 29 (3-step mini-training plus checkpoint) run verbatim against the new package. In the saved notebook they never ran: cells 19 and 21 had FAILed and cell 23 had no output | PASS |

Not run locally: GPU, fp16, CUDA, multi-GPU, real LMDB, SLURM. The fixture is not the training data.

#### Notebook saved-output audit

| Cells | Saved state |
|---|---|
| 2–12 | PASS. Notebook runs under Jupyter's own Python on `/mnt/home/...`, with no GPU and no `git` |
| 14–16 | ZIP download succeeded, but cell 17 shows `Uni-Mol data: MISSING` because it looked for the wrong path (`Uni-Mol/unimol/data` instead of `unimol/unimol/data`) |
| 19 (`[13]`) | FAIL: `git` not found. This stale clone cell also only accepts `.git` directories, so it does not recognise the ZIP sources |
| 21 (`[8]`) | FAIL: "Uni-Core source unavailable". Its execution count shows it ran **before** the ZIP download (`[10]`–`[16]`), so the failure was execution order, not missing code |
| 23–31 | Never executed; the dashboard has no output |

The earlier "13/13 PASS" therefore cannot be reproduced from this notebook copy. Its useful parts are reproduced and extended by the tests above.

### 3. HPC verification required (NOT RUN; only you can run these)

1. `check_heart_env.sh` on the login node and as an `sbatch` GPU job: `heart_gpu` imports, `unicore-train`, registry, CUDA and fp16 on the allocated T4/H200.
2. Uni-Core installed into `heart_gpu` at `ace6fae` (pure-Python build).
3. `inspect_real_lmdb.py --pipeline` on the real `train.lmdb`/`valid.lmdb`/`dict.txt`: counts, contiguous keys, record layout, conformers per record, hydrogens present, dictionary coverage, RDKit atom-order agreement for the 2D view.
4. 10-update GPU pilot (now `scripts/pretrain_smoke.sh`): real-data forward and backward at LUMI base size with fp16, per-objective losses, bounded validation, checkpoints, wall time and `gb_free`.
5. `verify_checkpoint.py` on the pilot checkpoint, then the resume job to 12 updates.
6. Batch-size memory probe on the H200, and the real checkpoint size.
7. Multi-GPU (`torchrun`) path, if it is ever used.

### 4. Remaining blockers and risks

- **Uni-Core in `heart_gpu` is unverified.** It must be installed (README Stage A). If pip has no network access on CMRI, use the ZIP or scp route.
- **Real-data assumptions are unconfirmed.** Hydrogens present, conformers stored as a list of `(N,3)` arrays, keys `0..N-1`, and an RDKit `AddHs` order that matches `atoms` all need step 3 above. If the RDKit atom-order check fails, the 2D view will pair coordinates with the wrong atoms. In that case, disable the 2D view or fix it before full training.
- **The contrastive view definition is a scientific choice.** `independent` (HEART default) or `identical` (upstream behaviour) has to be decided. LUMI's unpublished data code cannot be checked.
- **The full-run batch size is unvalidated.** 448 is LUMI's A100 value and needs the H200 memory probe. The H200 GRES name and the partition's maximum wall time are unknown.
- **Licensing.** If the repository is public, GPL-3.0 obligations apply to the LUMI-derived files.
- **No model has been pretrained.** Convergence requires the full run, which is roughly 1M updates and many days, followed by downstream evaluation.
