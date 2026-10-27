"""Stage A/C/D -- configs, launcher guards and a local launcher run (bash required)."""
import os
import subprocess
import sys

import pytest
import yaml

from conftest import CONFIG_DIR, MODEL_ROOT, PRETRAIN_DIR, SCRIPTS_DIR, TESTS_DIR, find_bash

BASH = find_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="bash not available")


def flag_value(args, flag):
    return args[args.index(flag) + 1]


def test_pilot_config_matches_requested_pilot():
    import yaml_to_env

    launcher, uc = yaml_to_env.load(os.path.join(CONFIG_DIR, "pilot.yaml"), [])
    args = yaml_to_env.to_cli(uc)
    assert launcher["data_path"] == "/home/ypan/heart_liver/model/UniMolReference/ligand_data/ligands"
    assert launcher["output_root"] == "/home/ypan/heart_liver/model/heart_training_runs"
    assert launcher["n_gpu"] == 1 and not launcher["requires_confirmation"]
    assert flag_value(args, "--batch-size") == "4"
    assert flag_value(args, "--max-update") == "10"
    assert int(flag_value(args, "--save-interval-updates")) <= 10
    assert int(flag_value(args, "--validate-interval-updates")) <= 10
    assert flag_value(args, "--log-interval") == "1"
    assert flag_value(args, "--max-valid-steps") == "5"
    for f in ("--task", "--loss"):
        assert flag_value(args, f) == "unimol_contrastive"
    assert flag_value(args, "--arch") == "unimol_contrastive_base"
    # LUMI objective weights are preserved
    expected = {"--masked-token-loss": 1, "--masked-coord-loss": 5, "--masked-dist-loss": 10,
                "--x-norm-loss": 0.01, "--delta-pair-repr-norm-loss": 0.01, "--contrastive-loss": 10.0}
    for f, v in expected.items():
        assert float(flag_value(args, f)) == v, f
    assert "--fp16" in args and "--cpu" not in args
    assert flag_value(args, "--adam-betas") == "(0.9, 0.99)"


def test_full_config_is_guarded_and_resumable():
    import yaml_to_env

    launcher, uc = yaml_to_env.load(os.path.join(CONFIG_DIR, "full_pretrain.yaml"), [])
    assert launcher["requires_confirmation"] is True
    assert launcher["run_dir_mode"] == "fixed"
    assert launcher["stop_margin_minutes"] > 0
    assert uc["keep-interval-updates"] > 0 and uc["max-valid-steps"] > 0


@pytest.mark.parametrize("key,value", [
    ("finetune-from-model", "/some/pretrained.pt"),
    ("restore-file", "/some/pretrained.pt"),
    ("save-dir", "/tmp/x"),
    ("user-dir", "./elsewhere"),
])
def test_forbidden_keys_rejected(tmp_path, key, value):
    import yaml_to_env

    cfg = yaml.safe_load(open(os.path.join(CONFIG_DIR, "pilot.yaml"), encoding="utf-8"))
    cfg["unicore"][key] = value
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(SystemExit):
        yaml_to_env.load(str(p), [])


def test_overrides():
    import yaml_to_env

    launcher, uc = yaml_to_env.load(os.path.join(CONFIG_DIR, "pilot.yaml"),
                                    ["max-update=12", "launcher.run_name=resume-test"])
    assert uc["max-update"] == 12 and launcher["run_name"] == "resume-test"


@needs_bash
@pytest.mark.parametrize("script", ["pretrain_heart.sh", "check_heart_env.sh"])
def test_scripts_parse_and_use_lf(script):
    path = os.path.join(SCRIPTS_DIR, script)
    assert b"\r\n" not in open(path, "rb").read()
    subprocess.run([BASH, "-n", path], check=True)


def _posix(p):
    p = str(p).replace("\\", "/")
    if sys.platform == "win32" and len(p) > 1 and p[1] == ":":
        p = "/" + p[0].lower() + p[2:]
    return p


def launcher_env(fixture_dir, out_root, extra_set=""):
    py = _posix(sys.executable)
    env = dict(os.environ)
    env.update(
        HEART_MODEL_ROOT=_posix(MODEL_ROOT), HEART_SKIP_ENV="1", HEART_PYTHON=py,
        HEART_TRAIN_ENTRY="{} {}".format(py, _posix(os.path.join(TESTS_DIR, "unicore_train_entry.py"))),
        HEART_SET=" ".join(["launcher.data_path=" + _posix(fixture_dir),
                            "launcher.output_root=" + _posix(out_root),
                            "cpu=true", "fp16=false", "num-workers=0", "encoder-layers=2",
                            "encoder-embed-dim=64", "encoder-ffn-embed-dim=128",
                            "encoder-attention-heads=4", "batch-size=2", "batch-size-valid=2",
                            "max-update=2", "max-valid-steps=1", "mask-prob=0.4"] + extra_set.split()),
        PYTHONIOENCODING="utf-8",
    )
    for k in ("SLURM_JOB_ID", "RESUME_DIR", "MAX_UPDATE", "HEART_CONFIRM_FULL", "HEART_DRY_RUN"):
        env.pop(k, None)
    return env


def run_launcher(env, config):
    return subprocess.run([BASH, os.path.join(SCRIPTS_DIR, "pretrain_heart.sh"), config],
                          env=env, capture_output=True, text=True, timeout=900)


@needs_bash
def test_launcher_refuses_full_config_without_confirmation(fixture_dir, tmp_path):
    env = launcher_env(fixture_dir, tmp_path)
    p = run_launcher(env, os.path.join(CONFIG_DIR, "full_pretrain.yaml"))
    assert p.returncode != 0
    assert "HEART_CONFIRM_FULL=1" in p.stderr
    assert not any(tmp_path.iterdir())  # nothing created


@needs_bash
def test_launcher_rejects_missing_dataset(tmp_path):
    env = launcher_env(str(tmp_path / "no_such_data"), tmp_path)
    p = run_launcher(env, os.path.join(CONFIG_DIR, "pilot.yaml"))
    assert p.returncode != 0 and "missing or unreadable" in p.stderr


@needs_bash
def test_launcher_dry_run(fixture_dir, tmp_path):
    env = launcher_env(fixture_dir, tmp_path)
    env["HEART_DRY_RUN"] = "1"
    p = run_launcher(env, os.path.join(CONFIG_DIR, "pilot.yaml"))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "--user-dir ./unimol" in p.stdout and "--tmp-save-dir" in p.stdout
    assert "PASS  task  unimol_contrastive" in p.stdout
    assert "would write to" in p.stdout
    assert not any(d.name.startswith("heart-gpu-pilot-") for d in tmp_path.iterdir())  # no side effects


@needs_bash
def test_launcher_trains_and_resumes(fixture_dir, tmp_path):
    env = launcher_env(fixture_dir, tmp_path)
    p = run_launcher(env, os.path.join(CONFIG_DIR, "pilot.yaml"))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "new run, random initialisation" in p.stdout
    assert "PASS  checkpoint saved" in p.stdout
    (run_dir,) = [d for d in tmp_path.iterdir() if d.name.startswith("heart-gpu-pilot-")]
    assert (run_dir / "checkpoint_last.pt").exists()
    prov = list((run_dir / "provenance").iterdir())[0]
    assert {"pilot.yaml", "command.txt", "source_sha256.txt", "pip_freeze.txt"} <= {f.name for f in prov.iterdir()}

    env.update(RESUME_DIR=_posix(str(run_dir)), MAX_UPDATE="4")
    p = run_launcher(env, os.path.join(CONFIG_DIR, "pilot.yaml"))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "RESUMING" in p.stdout and "@ 2 updates" in p.stdout and "num_updates=4" in p.stdout


@needs_bash
def test_launcher_refuses_output_inside_dataset(fixture_dir):
    env = launcher_env(fixture_dir, os.path.join(fixture_dir, "runs"))
    env["HEART_DRY_RUN"] = "1"
    p = run_launcher(env, os.path.join(CONFIG_DIR, "pilot.yaml"))
    assert p.returncode != 0 and "inside the dataset" in p.stderr
