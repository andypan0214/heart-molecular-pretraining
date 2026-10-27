#!/usr/bin/env python
"""Build the tiny synthetic 8-SMILES LMDB fixture (6 train / 2 valid).

This reproduces cells 8, 10 and 12 of LUMI_Jupyter_Standalone_CPU_Validation.ipynb:
  * the notebook's eight valid SMILES plus one intentionally broken SMILES
    ('C1CC'), which must be rejected and never written;
  * RDKit ETKDGv3 (randomSeed=42, numThreads=1), two 3D conformers per
    molecule, MMFF optimisation where parameters exist;
  * records {'atoms': [...with H...], 'coordinates': [float32 (N,3)] * 2,
    'smi': canonical heavy-atom SMILES}, pickled (protocol 4) under ASCII
    integer keys b'0', b'1', ...; sorted by SMILES, first 6 -> train.lmdb,
    last 2 -> valid.lmdb.

The fixture is an engineering smoke test only. It is NOT the training data:
real pretraining reads the Uni-Mol ligands train.lmdb/valid.lmdb on CMRI HPC.

Usage:
    python tools/make_fixture_lmdb.py OUT_DIR [--dict tests/fixtures/dict.txt]
"""
import argparse
import os
import pickle
import shutil
import sys

import numpy as np

TEST_SMILES = [
    ("Ethanol", "CCO"),
    ("Benzene", "c1ccccc1"),
    ("Acetic acid", "CC(=O)O"),
    ("Pyridine", "c1ccncc1"),
    ("Cyclohexane", "C1CCCCC1"),
    ("L-Alanine", "C[C@@H](N)C(=O)O"),
    ("Triethylamine", "CCN(CC)CC"),
    ("Lipid-like amine", "CCCCCCCCCCCCN(C)C"),
    ("Broken SMILES", "C1CC"),  # intentionally invalid; never train on it
]
SEED = 42
N_CONFS = 2
N_TRAIN = 6

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DICT = os.path.join(HERE, "..", "tests", "fixtures", "dict.txt")


def build_records(smiles_list=TEST_SMILES, seed=SEED, n_confs=N_CONFS):
    """Return (records, rejected_names) following the notebook workflow."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")  # the broken SMILES is expected to fail
    records, rejected = [], []
    for name, smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rejected.append(name)
            continue
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.numThreads = 1
        ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))
        if len(ids) < n_confs:
            rejected.append(name)
            continue
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=100)
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        xyzs = [mol.GetConformer(cid).GetPositions().astype(np.float32) for cid in ids]
        assert all(x.shape == (len(atoms), 3) and np.isfinite(x).all() for x in xyzs)
        records.append(
            {
                "atoms": atoms,
                "coordinates": xyzs,
                "smi": Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True),
            }
        )
    return records, rejected


def write_lmdb(path, items):
    import lmdb

    if os.path.exists(path):
        os.remove(path)
    lock = path + "-lock"
    if os.path.exists(lock):
        os.remove(lock)
    env = lmdb.open(path, map_size=64 * 1024 * 1024, subdir=False)
    with env.begin(write=True) as txn:
        for i, item in enumerate(items):
            txn.put(str(i).encode("ascii"), pickle.dumps(item, protocol=4))
    env.close()


def build_fixture(out_dir, dict_path=DEFAULT_DICT):
    os.makedirs(out_dir, exist_ok=True)
    records, rejected = build_records()
    ordered = sorted(records, key=lambda r: r["smi"])
    splits = {"train": ordered[:N_TRAIN], "valid": ordered[N_TRAIN:]}
    for split, items in splits.items():
        write_lmdb(os.path.join(out_dir, split + ".lmdb"), items)
    shutil.copyfile(dict_path, os.path.join(out_dir, "dict.txt"))
    return {"records": ordered, "rejected": rejected, "splits": splits}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir")
    parser.add_argument("--dict", default=DEFAULT_DICT)
    ns = parser.parse_args(argv)
    info = build_fixture(ns.out_dir, ns.dict)
    print("valid molecules :", len(info["records"]))
    print("rejected        :", ", ".join(info["rejected"]) or "-")
    for split, items in info["splits"].items():
        print("{:5s} {:d} : {}".format(split, len(items), ", ".join(r["smi"] for r in items)))
    if len(info["records"]) != 8 or info["rejected"] != ["Broken SMILES"]:
        print("FAIL: expected 8 valid molecules and only 'Broken SMILES' rejected")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
