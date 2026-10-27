"""Stage B.1/B.2 -- LUMI contrastive data pipeline: loading, shapes, padding, views."""
import os
import pickle

import lmdb
import numpy as np
import pytest
import torch

B = 3  # molecules per batch -> 2B sequences (two views each)


def collate(ds, n=B, start=0):
    return ds.collater([ds[i] for i in range(start, start + n)])


def test_lmdb_length_without_key_scan(unimol_pkg, fixture_dir, monkeypatch):
    from unimol.data import LMDBDataset

    path = os.path.join(fixture_dir, "train.lmdb")
    fast = LMDBDataset(path)
    monkeypatch.setenv("HEART_LMDB_FULL_KEY_SCAN", "1")
    full = LMDBDataset(path)
    assert len(fast) == len(full) == 6
    assert fast[5]["smi"] == full[5]["smi"]


def test_lmdb_rejects_non_contiguous_keys(unimol_pkg, tmp_path):
    from unimol.data import LMDBDataset

    path = str(tmp_path / "bad.lmdb")
    env = lmdb.open(path, subdir=False, map_size=1 << 20)
    with env.begin(write=True) as txn:
        for k in (b"0", b"2"):  # 2 entries but key b"1" missing
            txn.put(k, pickle.dumps({"atoms": ["C"], "coordinates": [np.zeros((1, 3))], "smi": "C"}))
    env.close()
    with pytest.raises(ValueError, match="contiguous"):
        LMDBDataset(path)


def test_add2d_does_not_mutate_cached_record(unimol_pkg):
    from unimol.data import Add2DConformerDataset

    rec = {"smi": "CCO", "atoms": ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
           "coordinates": [np.zeros((9, 3), np.float32)]}

    class Raw(list):  # always returns the *same* dict object, like an LRU cache
        def set_epoch(self, e):
            pass

    ds = Add2DConformerDataset(Raw([rec]), "smi", "atoms", "coordinates")
    for epoch in (1, 2, 3):  # distinct cache keys -> recomputed each time
        ds.set_epoch(epoch)
        out = ds[0]
        assert len(out["coordinates"]) == 2
    assert len(rec["coordinates"]) == 1  # upstream would have grown to 4


def test_batch_shapes_and_padding(make_task):
    task, ds = make_task()
    d = task.dictionary
    b = collate(ds)
    ni, tg = b["net_input"], b["target"]
    n, L = ni["src_tokens"].shape
    assert n == 2 * B and L % 8 == 0
    assert ni["src_coord"].shape == (n, L, 3)
    assert ni["src_distance"].shape == (n, L, L)
    assert ni["src_edge_type"].shape == (n, L, L)
    assert tg["tokens_target"].shape == (n, L)
    assert tg["distance_target"].shape == (n, L, L)
    assert tg["coord_target"].shape == (n, L, 3)
    assert len(tg["smi_name"]) == n

    tok = ni["src_tokens"]
    pad = tok.eq(d.pad())
    assert (tok[:, 0] == d.bos()).all()
    lengths = (~pad).sum(1)
    for r in range(n):
        assert tok[r, lengths[r] - 1] == d.eos()
        assert not pad[r, : lengths[r]].any() and pad[r, lengths[r]:].all()
    # padded / BOS / EOS geometry is zero; distances symmetric, zero diagonal
    assert ni["src_coord"][pad].abs().sum() == 0
    assert ni["src_distance"][pad].abs().sum() == 0
    assert torch.allclose(ni["src_distance"], ni["src_distance"].transpose(1, 2), atol=1e-5)
    assert ni["src_distance"].diagonal(dim1=1, dim2=2).abs().max() < 1e-5
    assert torch.isfinite(ni["src_coord"]).all() and torch.isfinite(ni["src_distance"]).all()
    # edge type = tok_i * |V| + tok_j for real pairs (|V| includes [MASK]); 0 on padding
    V = len(d)
    real = (~pad).unsqueeze(-1) & (~pad).unsqueeze(1)
    expected = tok.unsqueeze(-1) * V + tok.unsqueeze(1)
    assert torch.equal(ni["src_edge_type"][real], expected[real])
    assert ni["src_edge_type"][~real].abs().sum() == 0
    assert ni["src_edge_type"].max() < V * V  # fits the model's GaussianLayer edge table


def test_hydrogens_removed_and_atom_counts(make_task):
    from rdkit import Chem

    task, ds = make_task()
    d = task.dictionary
    b = collate(ds)
    tok, tgt = b["net_input"]["src_tokens"], b["target"]["tokens_target"]
    h = d.index("H")
    assert not tok.eq(h).any() and not tgt.eq(h).any()  # only_polar=0 removes all H
    lengths = tok.ne(d.pad()).sum(1) - 2  # minus BOS/EOS
    for r, smi in enumerate(b["target"]["smi_name"]):
        assert lengths[r] == Chem.MolFromSmiles(smi).GetNumHeavyAtoms()


def test_masking_targets(make_task):
    task, ds = make_task()
    d = task.dictionary
    b = collate(ds)
    tok, tgt = b["net_input"]["src_tokens"], b["target"]["tokens_target"]
    masked = tgt.ne(d.pad())
    assert masked.sum() > 0
    # leave_unmasked_prob = random_token_prob = 0 -> every target position shows [MASK]
    assert torch.equal(masked, tok.eq(task.mask_idx))
    assert not masked[:, 0].any()  # never mask BOS


def test_views_are_paired_in_order(make_task):
    _, ds = make_task()
    names = list(collate(ds)["target"]["smi_name"])
    assert names[:B] == names[B:]
    assert len(set(names[:B])) == B


def test_identical_view_mode_reproduces_upstream(make_task):
    """Stock LUMI + Uni-Mol data: both views are the same tensors (dropout-only positives)."""
    _, ds = make_task(contrastive_view_mode="identical")
    ni = collate(ds)["net_input"]
    for k in ("src_tokens", "src_coord", "src_distance"):
        assert torch.equal(ni[k][:B], ni[k][B:]), k


def test_independent_view_mode_differs(make_task):
    """HEART default: second view re-samples conformer / mask / noise."""
    _, ds = make_task(contrastive_view_mode="independent")
    ni = collate(ds, n=6)["net_input"]
    diff_coord = (ni["src_coord"][:6] - ni["src_coord"][6:]).abs().amax(dim=(1, 2))
    diff_tok = ni["src_tokens"][:6].ne(ni["src_tokens"][6:]).any(1)
    assert (diff_coord > 1e-4).sum() + diff_tok.sum() >= 4  # most molecules differ


def test_deterministic_per_epoch_and_changes_across_epochs(make_task):
    _, ds = make_task()
    a = collate(ds)["net_input"]["src_coord"]
    b = collate(ds)["net_input"]["src_coord"]
    assert torch.equal(a, b)  # seeded by (seed, epoch, index)
    ds.set_epoch(2)
    c = collate(ds)["net_input"]["src_coord"]
    assert not torch.equal(a, c)


def test_validation_split_loads(make_task):
    _, ds = make_task(split="valid")
    assert len(ds) == 2
    b = ds.collater([ds[0], ds[1]])
    assert b["net_input"]["src_tokens"].shape[0] == 4
