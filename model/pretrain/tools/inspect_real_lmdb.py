#!/usr/bin/env python
"""Read-only compatibility check of the real Uni-Mol ligand LMDBs (run on CMRI HPC).

Never writes to the data directory: environments are opened with
readonly=True, lock=False and only a handful of records are decoded.

Checks
  1. train.lmdb / valid.lmdb / dict.txt exist; entry counts vs. expected.
  2. Key convention: contiguous ASCII integers (b'0' and b'N-1' present).
  3. 3-5 records per split: field types, atom count, conformer count, each
     conformer (N,3) & finite, hydrogen content, elements not in dict.txt,
     and whether RDKit's AddHs(MolFromSmiles(smi)) atom order matches the
     record (this is what Add2DConformerDataset relies on for the 2D view).
  4. dict.txt: special tokens present, compared with the Uni-Mol molecule dict.
  5. --pipeline: run the HEART task (unimol_contrastive) on the real data and
     collate one batch, printing tensor shapes (needs Uni-Core; run from the
     model root so ./unimol is importable).

Usage (from /home/ypan/heart_liver/model):
    python pretrain/tools/inspect_real_lmdb.py \
        /home/ypan/heart_liver/model/UniMolReference/ligand_data/ligands \
        --samples 5 --pipeline --json heart_training_runs/inspect_real_lmdb.json
"""
import argparse
import json
import os
import pickle
import random
import sys
from collections import Counter

import numpy as np

EXPECTED_ENTRIES = {"train": 18837320, "valid": 98844}
UNIMOL_MOLECULE_DICT = [
    "[PAD]", "[CLS]", "[SEP]", "[UNK]", "C", "N", "O", "S", "H", "Cl", "F", "Br",
    "I", "Si", "P", "B", "Na", "K", "Al", "Ca", "Sn", "As", "Hg", "Fe", "Zn", "Cr",
    "Se", "Gd", "Au", "Li",
]
SPECIALS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]


def read_dict(path):
    tokens = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.strip():
                tokens.append(line.rsplit(" ", 1)[0] if " " in line else line)
    return tokens


def open_ro(path):
    import lmdb

    return lmdb.open(
        path, subdir=False, readonly=True, lock=False, readahead=False,
        meminit=False, max_readers=256,
    )


def describe_record(rec, dict_tokens):
    out = {"fields": {k: type(v).__name__ for k, v in rec.items()}}
    atoms = list(rec["atoms"])
    coords = rec["coordinates"]
    out["n_atoms"] = len(atoms)
    out["n_heavy_atoms"] = sum(a != "H" for a in atoms)
    out["n_hydrogens"] = len(atoms) - out["n_heavy_atoms"]
    out["n_conformers"] = len(coords)
    shapes, finite = [], True
    for c in coords:
        arr = np.asarray(c)
        shapes.append([list(arr.shape), str(arr.dtype)])
        finite &= bool(np.isfinite(arr).all())
    out["conformer_shapes"] = sorted({json.dumps(s) for s in shapes})
    out["all_conformers_match_atoms"] = all(
        np.asarray(c).shape == (len(atoms), 3) for c in coords
    )
    out["finite"] = finite
    out["smi"] = rec.get("smi")
    out["elements"] = dict(Counter(atoms))
    out["oov_elements"] = sorted(set(atoms) - set(dict_tokens))
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(rec["smi"])
        if mol is None:
            out["rdkit_2d_atom_order_match"] = "SMILES_PARSE_FAILED"
        else:
            sym = [a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms()]
            out["rdkit_2d_atom_order_match"] = sym == atoms
            if sym != atoms:
                out["rdkit_2d_atom_count"] = len(sym)
    except ImportError:
        out["rdkit_2d_atom_order_match"] = "RDKIT_NOT_INSTALLED"
    return out


def inspect_split(data_dir, split, n_samples, dict_tokens, rng):
    path = os.path.join(data_dir, split + ".lmdb")
    res = {"path": path, "exists": os.path.isfile(path)}
    if not res["exists"]:
        return res
    res["size_gb"] = round(os.path.getsize(path) / 1024 ** 3, 2)
    env = open_ro(path)
    try:
        with env.begin() as txn:
            n = txn.stat()["entries"]
            res["entries"] = n
            res["expected_entries"] = EXPECTED_ENTRIES.get(split)
            res["entries_match_expected"] = n == EXPECTED_ENTRIES.get(split)
            first_keys = []
            cur = txn.cursor()
            for i, key in enumerate(cur.iternext(values=False)):
                first_keys.append(key.decode("ascii", "replace"))
                if i >= 4:
                    break
            res["first_keys_lexicographic"] = first_keys
            res["key_0_present"] = txn.get(b"0") is not None
            res["key_last_present"] = txn.get(f"{n - 1}".encode("ascii")) is not None
            res["key_n_absent"] = txn.get(f"{n}".encode("ascii")) is None
            # first, middle and last record, topped up with random ones
            idx = {0, n // 2, n - 1} if n > 0 else set()
            while len(idx) < min(n_samples, n):
                idx.add(rng.randrange(n))
            res["records"] = {}
            for i in sorted(idx):
                raw = txn.get(f"{i}".encode("ascii"))
                if raw is None:
                    res["records"][str(i)] = "MISSING"
                    continue
                res["records"][str(i)] = describe_record(pickle.loads(raw), dict_tokens)
    finally:
        env.close()
    return res


def run_pipeline(data_dir, split, batch, seed):
    """Run the real HEART task on `split` and collate one batch."""
    from types import SimpleNamespace

    import torch
    from unicore import utils

    utils.import_user_module(SimpleNamespace(user_dir="./unimol"))
    from unimol.tasks.unimol_contrastive import UniMolTaskContrastive

    args = SimpleNamespace(
        data=data_dir, dict_name="dict.txt", seed=seed, only_polar=0,
        remove_hydrogen=False, remove_polar_hydrogen=False, mode="train",
        max_atoms=256, max_seq_len=512, mask_prob=0.15, leave_unmasked_prob=0.05,
        random_token_prob=0.05, noise_type="uniform", noise=1.0,
        contrastive_loss=10.0, contrastive_view_mode="independent",
        view_seed_offset=1000003,
    )
    task = UniMolTaskContrastive.setup_task(args)
    task.load_dataset(split)
    ds = task.dataset(split)
    ds.set_epoch(1)
    samples = [ds[i] for i in range(batch)]
    b = ds.collater(samples)
    pad = task.dictionary.pad()
    shapes = {k: list(v.shape) for k, v in b["net_input"].items()}
    shapes.update({k: list(v.shape) for k, v in b["target"].items() if torch.is_tensor(v)})
    names = list(b["target"]["smi_name"])
    toks = b["net_input"]["src_tokens"]
    return {
        "split": split,
        "dataset_len": len(ds),
        "dictionary_size_with_mask": len(task.dictionary),
        "shapes": shapes,
        "views_paired": names[:batch] == names[batch:],
        "masked_atoms": int(b["target"]["tokens_target"].ne(pad).sum()),
        "unk_tokens": int(toks.eq(task.dictionary.unk()).sum()),
        "max_len": int(toks.size(1)),
        "pad_multiple_of_8": toks.size(1) % 8 == 0,
        "coords_finite": bool(torch.isfinite(b["net_input"]["src_coord"]).all()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir")
    parser.add_argument("--samples", type=int, default=5, help="records per split (3-5 recommended)")
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument("--pipeline", action="store_true", help="also run the HEART task on real data")
    parser.add_argument("--pipeline-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", help="write the full report to this path (outside the data dir)")
    parser.add_argument("--expected-train", type=int, default=EXPECTED_ENTRIES["train"])
    parser.add_argument("--expected-valid", type=int, default=EXPECTED_ENTRIES["valid"])
    ns = parser.parse_args(argv)
    EXPECTED_ENTRIES.update(train=ns.expected_train, valid=ns.expected_valid)

    data_dir = os.path.abspath(ns.data_dir)
    report = {"data_dir": data_dir, "python": sys.executable}
    dict_path = os.path.join(data_dir, "dict.txt")
    if os.path.isfile(dict_path):
        tokens = read_dict(dict_path)
        report["dict"] = {
            "path": dict_path,
            "n_tokens": len(tokens),
            "specials_present": all(s in tokens for s in SPECIALS),
            "pad_index": tokens.index("[PAD]") if "[PAD]" in tokens else None,
            "identical_to_unimol_molecule_dict": tokens == UNIMOL_MOLECULE_DICT,
            "missing_vs_unimol": sorted(set(UNIMOL_MOLECULE_DICT) - set(tokens)),
            "extra_vs_unimol": sorted(set(tokens) - set(UNIMOL_MOLECULE_DICT)),
        }
    else:
        tokens = []
        report["dict"] = {"path": dict_path, "exists": False}

    rng = random.Random(ns.seed)
    report["splits"] = {
        s: inspect_split(data_dir, s, ns.samples, tokens, rng) for s in ns.splits.split(",")
    }
    if ns.pipeline:
        report["pipeline"] = {}
        for s in ns.splits.split(","):
            try:
                report["pipeline"][s] = run_pipeline(data_dir, s, ns.pipeline_batch, ns.seed)
            except Exception as exc:  # report, don't hide
                import traceback

                report["pipeline"][s] = {"error": repr(exc), "trace": traceback.format_exc()}

    # ---- verdicts ---------------------------------------------------------
    verdicts = []

    def add(name, ok):
        verdicts.append((name, bool(ok)))

    d = report["dict"]
    add("dict.txt present with [PAD]/[CLS]/[SEP]/[UNK]", d.get("specials_present"))
    for s, r in report["splits"].items():
        add(f"{s}.lmdb exists", r.get("exists"))
        if not r.get("exists"):
            continue
        add(f"{s}: entry count == {r['expected_entries']}", r.get("entries_match_expected"))
        add(f"{s}: keys 0..N-1 contiguous at both ends",
            r["key_0_present"] and r["key_last_present"] and r["key_n_absent"])
        recs = [v for v in r["records"].values() if isinstance(v, dict)]
        add(f"{s}: sampled records decoded ({len(recs)}/{len(r['records'])})",
            len(recs) == len(r["records"]))
        add(f"{s}: fields atoms/coordinates/smi present",
            all({"atoms", "coordinates", "smi"} <= set(x["fields"]) for x in recs))
        add(f"{s}: every conformer is (n_atoms,3) and finite",
            all(x["all_conformers_match_atoms"] and x["finite"] for x in recs))
        add(f"{s}: no elements outside dict.txt", all(not x["oov_elements"] for x in recs))
        add(f"{s}: RDKit AddHs atom order matches record (2D view)",
            all(x["rdkit_2d_atom_order_match"] is True for x in recs))
    for s, p in report.get("pipeline", {}).items():
        ok = "error" not in p and p.get("views_paired") and p.get("coords_finite") and p.get("masked_atoms", 0) > 0
        add(f"pipeline[{s}]: HEART task loads, collates, views paired", ok)
    report["verdicts"] = [{"check": n, "status": "PASS" if ok else "FAIL"} for n, ok in verdicts]

    print(json.dumps(report, indent=2, default=str))
    print("\n==== SUMMARY ====")
    for n, ok in verdicts:
        print("{}  {}".format("PASS" if ok else "FAIL", n))
    if ns.json:
        out = os.path.abspath(ns.json)
        if out.startswith(data_dir + os.sep):
            print("refusing to write the report inside the data directory", file=sys.stderr)
            return 2
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print("report written to", out)
    return 0 if all(ok for _, ok in verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
