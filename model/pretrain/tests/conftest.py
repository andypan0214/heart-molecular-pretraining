"""Shared fixtures for the HEART local CPU tests.

Run from the model root (the directory containing unimol/ and pretrain/):
    python -m pytest pretrain/tests -v

Uni-Core must be importable (installed, or a source checkout on PYTHONPATH).
All data is the synthetic 8-SMILES fixture built in a temp directory; the real
Uni-Mol LMDB is never touched by these tests.
"""
import os
import sys
from types import SimpleNamespace

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_DIR = os.path.dirname(TESTS_DIR)
MODEL_ROOT = os.path.dirname(PRETRAIN_DIR)
TOOLS_DIR = os.path.join(PRETRAIN_DIR, "tools")
CONFIG_DIR = os.path.join(PRETRAIN_DIR, "configs")
SCRIPTS_DIR = os.path.join(PRETRAIN_DIR, "scripts")
sys.path.insert(0, TOOLS_DIR)


@pytest.fixture(scope="session")
def unimol_pkg():
    """Import ./unimol exactly the way unicore-train does (--user-dir)."""
    from unicore import utils

    utils.import_user_module(SimpleNamespace(user_dir=os.path.join(MODEL_ROOT, "unimol")))
    import unimol

    assert os.path.dirname(os.path.abspath(unimol.__file__)) == os.path.join(MODEL_ROOT, "unimol")
    return unimol


@pytest.fixture(scope="session")
def fixture_info(tmp_path_factory):
    import make_fixture_lmdb

    out = tmp_path_factory.mktemp("heart_fixture")
    info = make_fixture_lmdb.build_fixture(str(out))
    info["dir"] = str(out)
    return info


@pytest.fixture(scope="session")
def fixture_dir(fixture_info):
    return fixture_info["dir"]


def task_args(data_dir, **overrides):
    """Task/model args as unicore-train would produce for the pilot config."""
    args = dict(
        data=data_dir, dict_name="dict.txt", seed=1, only_polar=0,
        remove_hydrogen=False, remove_polar_hydrogen=False, mode="train",
        max_atoms=256, max_seq_len=512,
        mask_prob=0.4, leave_unmasked_prob=0.0, random_token_prob=0.0,
        noise_type="uniform", noise=1.0,
        contrastive_loss=10.0, contrastive_view_mode="independent", view_seed_offset=1000003,
        # LUMI loss weights (pretrain.sh)
        masked_token_loss=1.0, masked_coord_loss=5.0, masked_dist_loss=10.0,
        x_norm_loss=0.01, delta_pair_repr_norm_loss=0.01, masked_coord_dist_loss=-1.0,
        # tiny CPU architecture
        encoder_layers=2, encoder_embed_dim=64, encoder_ffn_embed_dim=128,
        encoder_attention_heads=4,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.fixture
def make_task(unimol_pkg, fixture_dir):
    from unimol.tasks.unimol_contrastive import UniMolTaskContrastive

    def _make(split="train", **overrides):
        task = UniMolTaskContrastive.setup_task(task_args(fixture_dir, **overrides))
        task.load_dataset(split)
        ds = task.dataset(split)
        ds.set_epoch(1)
        return task, ds

    return _make


def find_bash():
    """Git Bash on Windows, /bin/bash elsewhere; never WSL's System32 bash."""
    import shutil

    candidates = [shutil.which("bash")]
    if sys.platform == "win32":
        candidates = [r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"] + candidates
    for c in candidates:
        if c and os.path.isfile(c) and "system32" not in c.lower():
            return c
    return None
