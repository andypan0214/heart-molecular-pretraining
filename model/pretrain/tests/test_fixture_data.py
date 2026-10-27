"""Stage B.1 -- the notebook's 8-SMILES fixture and LMDB round trip."""
import pickle

import lmdb
import numpy as np

# Saved outputs of notebook cells 8/10/12 (LUMI_Jupyter_Standalone_CPU_Validation.ipynb)
NOTEBOOK_ATOMS_WITH_H = {
    "CCO": 9, "c1ccccc1": 12, "CC(=O)O": 8, "c1ccncc1": 11, "C1CCCCC1": 18,
    "C[C@@H](N)C(=O)O": 13, "CCN(CC)CC": 22, "CCCCCCCCCCCCN(C)C": 46,
}
NOTEBOOK_TRAIN = ["C1CCCCC1", "CC(=O)O", "CCCCCCCCCCCCN(C)C", "CCN(CC)CC", "CCO", "C[C@@H](N)C(=O)O"]
NOTEBOOK_VALID = ["c1ccccc1", "c1ccncc1"]


def read_split(path):
    env = lmdb.open(path, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        n = txn.stat()["entries"]
        keys = [k for k, _ in txn.cursor()]
        recs = [pickle.loads(txn.get(str(i).encode("ascii"))) for i in range(n)]
    env.close()
    return keys, recs


def test_eight_valid_smiles_and_broken_rejected(fixture_info):
    assert len(fixture_info["records"]) == 8
    assert fixture_info["rejected"] == ["Broken SMILES"]
    assert all(r["smi"] != "C1CC" for r in fixture_info["records"])


def test_split_matches_notebook(fixture_info):
    train = [r["smi"] for r in fixture_info["splits"]["train"]]
    valid = [r["smi"] for r in fixture_info["splits"]["valid"]]
    assert train == NOTEBOOK_TRAIN
    assert valid == NOTEBOOK_VALID
    assert set(train).isdisjoint(valid)


def test_lmdb_roundtrip_keys_and_geometry(fixture_dir):
    import os

    for split, expected in (("train", NOTEBOOK_TRAIN), ("valid", NOTEBOOK_VALID)):
        keys, recs = read_split(os.path.join(fixture_dir, split + ".lmdb"))
        assert sorted(keys) == sorted(str(i).encode("ascii") for i in range(len(expected)))
        assert [r["smi"] for r in recs] == expected
        for r in recs:
            assert set(r) == {"atoms", "coordinates", "smi"}
            assert len(r["atoms"]) == NOTEBOOK_ATOMS_WITH_H[r["smi"]]
            assert len(r["coordinates"]) == 2  # two ETKDG conformers
            for xyz in r["coordinates"]:
                assert xyz.shape == (len(r["atoms"]), 3) and xyz.dtype == np.float32
                assert np.isfinite(xyz).all()
            # the two 3D views are genuinely different conformers
            assert np.abs(r["coordinates"][0] - r["coordinates"][1]).max() > 1e-3
