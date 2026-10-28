"""Patch P5: LMDB environments are per process; multi-worker loading matches num_workers=0.

Regression for the CMRI smoke job 1011484 investigation: Uni-Core reads the
first training batch in the main process before the DataLoader forks its
workers, so the lazily opened LMDB environment used to be inherited by every
worker (LMDB forbids using an environment across fork()).
"""
import multiprocessing as mp
import os
import pickle

import pytest
import torch


@pytest.fixture
def lmdb_ds(unimol_pkg, fixture_dir):
    from unimol.data import LMDBDataset

    ds = LMDBDataset(os.path.join(fixture_dir, "train.lmdb"))
    type(ds).__getitem__.cache_clear()
    yield ds
    type(ds).__getitem__.cache_clear()


def test_environment_not_opened_at_construction(lmdb_ds):
    assert not hasattr(lmdb_ds, "env")


def test_environment_reopened_in_a_child_process(lmdb_ds, monkeypatch):
    parent_rec = lmdb_ds[0]
    parent_env = lmdb_ds.env
    assert lmdb_ds._env_pid == os.getpid()
    # simulate a forked DataLoader worker: same object, different PID
    child_pid = os.getpid() + 100000
    monkeypatch.setattr(os, "getpid", lambda: child_pid)
    type(lmdb_ds).__getitem__.cache_clear()
    child_rec = lmdb_ds[0]
    assert lmdb_ds.env is not parent_env, "inherited LMDB environment reused after fork"
    assert lmdb_ds._env_pid == child_pid
    assert child_rec["smi"] == parent_rec["smi"]
    assert len(child_rec["coordinates"]) == len(parent_rec["coordinates"])


def test_environment_not_pickled(lmdb_ds):
    lmdb_ds[0]  # opens the environment in this process
    clone = pickle.loads(pickle.dumps(lmdb_ds))
    assert not hasattr(clone, "env") and not hasattr(clone, "_env_pid")
    type(clone).__getitem__.cache_clear()
    assert clone[1]["smi"] == lmdb_ds[1]["smi"] and len(clone) == len(lmdb_ds)


def _batches(ds, num_workers, context=None):
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=[[0, 1], [2, 3], [4, 5]], collate_fn=ds.collater,
        num_workers=num_workers, multiprocessing_context=context, timeout=120 if num_workers else 0)
    return list(loader)


def _same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if torch.is_tensor(a):
        return torch.equal(a, b)
    return a == b


HAS_FORK = "fork" in mp.get_all_start_methods()


def _records(batch):
    return [(r["smi"], len(r["coordinates"])) for r in batch]


def test_lmdb_reads_in_spawned_workers_after_main_read(lmdb_ds):
    """Real separate worker processes (spawn works on every OS) after the parent opened the env."""
    ref = [lmdb_ds[i] for i in range(6)]
    loader = torch.utils.data.DataLoader(lmdb_ds, batch_sampler=[[0, 1, 2], [3, 4, 5]], collate_fn=_records,
                                         num_workers=2, multiprocessing_context="spawn", timeout=120)
    got = [r for batch in loader for r in batch]
    assert got == _records(ref)


@pytest.mark.skipif(not HAS_FORK, reason="fork start method unavailable (Windows); CMRI/Linux uses fork")
def test_worker_batches_equal_main_process_batches(make_task):
    """Full HEART pipeline with forked workers, after the main process already read data (as Uni-Core does).

    The full pipeline is not picklable for spawn (upstream Uni-Mol MaskPointsDataset holds a lambda);
    Linux DataLoaders fork, so that is not changed here.
    """
    task, ds = make_task("train", mask_prob=0.15)
    reference = _batches(ds, 0)             # main process opens the LMDB env first
    workers = _batches(ds, 2, "fork")
    assert len(workers) == len(reference) == 3
    for ref, got in zip(reference, workers):
        assert _same(ref, got)
