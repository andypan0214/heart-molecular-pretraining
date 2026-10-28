"""Stage A/C/D -- parameter records (configs/*.yaml) and the YAML -> CLI helper.

The SLURM scripts themselves are tested in test_scripts.py, which also checks
that each script's command matches its YAML record.
"""
import os

import pytest
import yaml

from conftest import CONFIG_DIR

MODEL_TRAIN = "/home/ypan/heart_liver/model_train"


def flag_value(args, flag):
    return args[args.index(flag) + 1]


def test_pilot_config_matches_requested_pilot():
    import yaml_to_env

    launcher, uc = yaml_to_env.load(os.path.join(CONFIG_DIR, "pilot.yaml"), [])
    args = yaml_to_env.to_cli(uc)
    assert launcher["data_path"] == MODEL_TRAIN + "/UniMolReference/ligand_data/ligands"
    assert launcher["output_root"] == MODEL_TRAIN + "/heart-molecular-pretraining/runs"
    assert launcher["script"] == "scripts/pretrain_smoke.sh"
    assert launcher["n_gpu"] == 1 and not launcher["requires_confirmation"]
    assert launcher["slurm"] == {"partition": "gpu", "gres": "gpu:1", "cpus_per_task": 6,
                                 "mem": "55G", "time": "00:30:00"}
    assert flag_value(args, "--batch-size") == "4"
    assert flag_value(args, "--batch-size-valid") == "4"
    assert flag_value(args, "--max-update") == "10"
    assert flag_value(args, "--total-num-update") == "20"
    assert flag_value(args, "--warmup-updates") == "3"
    assert flag_value(args, "--save-interval-updates") == "5"
    assert flag_value(args, "--validate-interval-updates") == "5"
    assert flag_value(args, "--log-interval") == "1"
    assert flag_value(args, "--max-valid-steps") == "5"
    assert flag_value(args, "--seed") == "1"
    for f in ("--task", "--loss"):
        assert flag_value(args, f) == "unimol_contrastive"
    assert flag_value(args, "--arch") == "unimol_contrastive_base"
    # LUMI objective weights are preserved
    expected = {"--masked-token-loss": 1, "--masked-coord-loss": 5, "--masked-dist-loss": 10,
                "--x-norm-loss": 0.01, "--delta-pair-repr-norm-loss": 0.01, "--contrastive-loss": 10.0,
                "--mask-prob": 0.15, "--noise": 1.0, "--lr": 1e-5, "--weight-decay": 1e-4,
                "--encoder-attention-heads": 64, "--max-atoms": 256, "--only-polar": 0,
                "--fp16-init-scale": 4, "--fp16-scale-window": 256, "--clip-norm": 1.0, "--adam-eps": 1e-6}
    for f, v in expected.items():
        assert float(flag_value(args, f)) == v, f
    assert "--fp16" in args and "--cpu" not in args
    assert flag_value(args, "--adam-betas") == "(0.9, 0.99)"
    assert flag_value(args, "--noise-type") == "uniform"
    assert flag_value(args, "--contrastive-view-mode") == "independent"


def test_full_config_is_guarded_and_resumable():
    import yaml_to_env

    launcher, uc = yaml_to_env.load(os.path.join(CONFIG_DIR, "full_pretrain.yaml"), [])
    assert launcher["requires_confirmation"] is True
    assert launcher["run_dir_mode"] == "fixed"
    assert launcher["stop_margin_minutes"] > 0
    assert launcher["script"] == "scripts/pretrain_heart.sh"
    assert uc["keep-interval-updates"] > 0
    assert uc["max-valid-steps"] is None  # LUMI validates on the whole valid split
    # LUMI full-run reference values
    assert uc["warmup-updates"] == 10000 and uc["max-update"] == uc["total-num-update"] == 1000000
    assert uc["lr"] == 1e-5 and uc["weight-decay"] == 1e-4 and uc["encoder-attention-heads"] == 64


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
