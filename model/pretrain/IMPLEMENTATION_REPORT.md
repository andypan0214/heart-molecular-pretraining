# HEART pretraining: implementation report (2026-10-27)

Scope: LUMI-lab's original **general molecular pretraining** stage. Random initialisation, Uni-Mol ligand data, Uni-Core trainer. No continual pretraining, fine-tuning or lipid-specific prediction. **No model has been pretrained.** Every training run in this report was a few updates on 8 synthetic molecules on a CPU.

## 1. Implemented

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

## 2. Locally tested (Windows 11, Python 3.11.7, torch 2.2.2+cpu, RDKit 2024.09.6, NumPy 1.26.4, Uni-Core `ace6fae` from source)

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

### Notebook saved-output audit

| Cells | Saved state |
|---|---|
| 2–12 | PASS. Notebook runs under Jupyter's own Python on `/mnt/home/...`, with no GPU and no `git` |
| 14–16 | ZIP download succeeded, but cell 17 shows `Uni-Mol data: MISSING` because it looked for the wrong path (`Uni-Mol/unimol/data` instead of `unimol/unimol/data`) |
| 19 (`[13]`) | FAIL: `git` not found. This stale clone cell also only accepts `.git` directories, so it does not recognise the ZIP sources |
| 21 (`[8]`) | FAIL: "Uni-Core source unavailable". Its execution count shows it ran **before** the ZIP download (`[10]`–`[16]`), so the failure was execution order, not missing code |
| 23–31 | Never executed; the dashboard has no output |

The earlier "13/13 PASS" therefore cannot be reproduced from this notebook copy. Its useful parts are reproduced and extended by the tests above.

## 3. HPC verification required (NOT RUN; only you can run these)

1. `check_heart_env.sh` on the login node and as an `sbatch` GPU job: `heart_gpu` imports, `unicore-train`, registry, CUDA and fp16 on the allocated T4/H200.
2. Uni-Core installed into `heart_gpu` at `ace6fae` (pure-Python build).
3. `inspect_real_lmdb.py --pipeline` on the real `train.lmdb`/`valid.lmdb`/`dict.txt`: counts, contiguous keys, record layout, conformers per record, hydrogens present, dictionary coverage, RDKit atom-order agreement for the 2D view.
4. 10-update GPU pilot (`configs/pilot.yaml`): real-data forward and backward at LUMI base size with fp16, per-objective losses, bounded validation, checkpoints, wall time and `gb_free`.
5. `verify_checkpoint.py` on the pilot checkpoint, then the resume job to 12 updates.
6. Batch-size memory probe on the H200, and the real checkpoint size.
7. Multi-GPU (`torchrun`) path, if it is ever used.

## 4. Remaining blockers and risks

- **Uni-Core in `heart_gpu` is unverified.** It must be installed (README Stage A). If pip has no network access on CMRI, use the ZIP or scp route.
- **Real-data assumptions are unconfirmed.** Hydrogens present, conformers stored as a list of `(N,3)` arrays, keys `0..N-1`, and an RDKit `AddHs` order that matches `atoms` all need step 3 above. If the RDKit atom-order check fails, the 2D view will pair coordinates with the wrong atoms. In that case, disable the 2D view or fix it before full training.
- **The contrastive view definition is a scientific choice.** `independent` (HEART default) or `identical` (upstream behaviour) has to be decided. LUMI's unpublished data code cannot be checked.
- **The full-run batch size is unvalidated.** 448 is LUMI's A100 value and needs the H200 memory probe. The H200 GRES name and the partition's maximum wall time are unknown.
- **Licensing.** If the repository is public, GPL-3.0 obligations apply to the LUMI-derived files.
- **No model has been pretrained.** Convergence requires the full run, which is roughly 1M updates and many days, followed by downstream evaluation.
