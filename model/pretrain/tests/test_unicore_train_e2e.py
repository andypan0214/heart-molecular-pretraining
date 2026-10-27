"""Stage B end-to-end -- the real Uni-Core trainer (unicore-train) on the CPU fixture.

The pilot.yaml config is translated by tools/yaml_to_env.py exactly as the
SLURM launcher does, with CPU/tiny-architecture overrides, then:
  run 1: 3 optimizer updates from random init -> validation + checkpoints
  run 2: resume from checkpoint_last.pt to 5 updates
and the log / checkpoint tools must report PASS.
"""
import os
import subprocess
import sys

import pytest
import torch

from conftest import CONFIG_DIR, MODEL_ROOT, TESTS_DIR

TINY = [
    "cpu=true", "fp16=false", "num-workers=0", "encoder-layers=2", "encoder-embed-dim=64",
    "encoder-ffn-embed-dim=128", "encoder-attention-heads=4", "batch-size=2",
    "batch-size-valid=2", "save-interval-updates=2", "validate-interval-updates=2",
    "max-valid-steps=1", "mask-prob=0.4", "lr=1e-4", "warmup-updates=1",
]


def unicore_cmd(data_dir, save_dir, max_update):
    import yaml_to_env

    _, uc = yaml_to_env.load(os.path.join(CONFIG_DIR, "pilot.yaml"), TINY + ["max-update=%d" % max_update])
    return ([sys.executable, os.path.join(TESTS_DIR, "unicore_train_entry.py"), data_dir,
             "--user-dir", "./unimol"] + yaml_to_env.to_cli(uc)
            + ["--save-dir", save_dir, "--tmp-save-dir", save_dir])


def run(cmd, log_path):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run(cmd, cwd=MODEL_ROOT, env=env, capture_output=True, text=True, timeout=900)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(p.stdout + p.stderr)
    assert p.returncode == 0, (p.stdout + p.stderr)[-4000:]
    return p.stdout + p.stderr


@pytest.fixture(scope="module")
def trained_run(fixture_dir, tmp_path_factory):
    save = str(tmp_path_factory.mktemp("run"))
    log = os.path.join(save, "train.log")
    out1 = run(unicore_cmd(fixture_dir, save, 3), log)
    snap = torch.load(os.path.join(save, "checkpoint_last.pt"), map_location="cpu")
    out2 = run(unicore_cmd(fixture_dir, save, 5), log)
    return {"save": save, "log": log, "out1": out1, "out2": out2, "ckpt_after_run1": snap}


def test_fresh_run_trains_validates_and_saves(trained_run):
    out = trained_run["out1"]
    assert "No existing checkpoint found" in out  # random initialisation, nothing loaded
    assert "num_updates=3" in out
    assert "valid on 'valid' subset" in out
    assert trained_run["ckpt_after_run1"]["optimizer_history"][-1]["num_updates"] == 3


def test_checkpoint_files(trained_run):
    files = set(os.listdir(trained_run["save"]))
    assert {"checkpoint_last.pt", "checkpoint_best.pt"} <= files
    # keep-interval-updates=1 keeps only the newest interval checkpoint
    assert len([f for f in files if f.startswith("checkpoint_1_") or f.startswith("checkpoint_2_")]) <= 1


def test_resume_continues_from_checkpoint(trained_run):
    out = trained_run["out2"]
    assert "Loaded checkpoint" in out and "@ 3 updates" in out
    assert "num_updates=4" in out and "num_updates=5" in out
    ck = torch.load(os.path.join(trained_run["save"], "checkpoint_last.pt"), map_location="cpu")
    assert ck["optimizer_history"][-1]["num_updates"] == 5
    prev = trained_run["ckpt_after_run1"]["extra_state"]["previous_training_time"]
    assert ck["extra_state"]["previous_training_time"] > prev  # cumulative across resumes


def test_summarize_log_tool(trained_run):
    import summarize_log

    assert summarize_log.main([trained_run["log"], "--expect-updates", "5", "--expect-resume-from", "3"]) == 0


def test_verify_checkpoint_tool(trained_run, monkeypatch, unimol_pkg):
    import verify_checkpoint

    monkeypatch.chdir(MODEL_ROOT)
    ck = os.path.join(trained_run["save"], "checkpoint_last.pt")
    assert verify_checkpoint.main([ck, "--expect-updates", "5"]) == 0


def test_stop_time_hours_uses_cumulative_time(trained_run):
    import stop_time_hours

    prev = stop_time_hours.previous_training_seconds(trained_run["save"])
    assert prev > 0
    assert stop_time_hours.parse_slurm_time("29:13") == 29 * 60 + 13
    assert stop_time_hours.parse_slurm_time("1-02:03:04") == ((24 + 2) * 60 + 3) * 60 + 4
    assert stop_time_hours.parse_slurm_time("2-00") == 48 * 3600
    assert stop_time_hours.parse_slurm_time("UNLIMITED") is None
