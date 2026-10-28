"""SLURM scripts: syntax, CMRI directives, model_train paths, guards, and local CPU runs.

pretrain_smoke.sh and pretrain_heart.sh are run with bash on the synthetic
fixture through the local-testing hooks (HEART_SKIP_ENV, HEART_LOCAL_CPU_TEST,
HEART_TRAIN_ENTRY, HEART_EXTRA_ARGS). The real unicore-train CLI code path is
used (tests/unicore_train_entry.py); only the tiny CPU architecture differs.
"""
import os
import re
import shlex
import subprocess
import sys

import pytest

from conftest import CONFIG_DIR, MODEL_ROOT, PRETRAIN_DIR, SCRIPTS_DIR, TESTS_DIR, find_bash

BASH = find_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="bash not available")
REPO_ROOT = os.path.dirname(MODEL_ROOT)
SCRIPTS = ["pretrain_smoke.sh", "pretrain_heart.sh", "check_heart_env.sh", "heart_lib.sh"]
TINY = ("--num-workers 0 --encoder-layers 2 --encoder-embed-dim 64 --encoder-ffn-embed-dim 128 "
        "--encoder-attention-heads 4 --batch-size 2 --batch-size-valid 2 --max-valid-steps 1 --mask-prob 0.4")
OWNED = {"save-dir", "tmp-save-dir", "tensorboard-logdir", "user-dir", "stop-time-hours"}


def read(name):
    with open(os.path.join(SCRIPTS_DIR, name), "r", encoding="utf-8") as fh:
        return fh.read()


def sbatch_directives(text):
    return dict(re.findall(r"^#SBATCH --([a-z-]+)=(\S+)", text, flags=re.M))


def _posix(p):
    p = str(p).replace("\\", "/")
    if sys.platform == "win32" and len(p) > 1 and p[1] == ":":
        p = "/" + p[0].lower() + p[2:]
    return p


# --------------------------------------------------------------------------- static checks
@needs_bash
@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_parse_and_use_lf(script):
    path = os.path.join(SCRIPTS_DIR, script)
    assert b"\r\n" not in open(path, "rb").read()
    subprocess.run([BASH, "-n", path], check=True)


def test_smoke_sbatch_directives_are_cmri_pilot():
    d = sbatch_directives(read("pretrain_smoke.sh"))
    assert d == {
        "nodes": "1", "ntasks-per-node": "1", "cpus-per-task": "6", "job-name": "heart-smoke-test",
        "mem": "55G", "gres": "gpu:1", "partition": "gpu", "time": "00:30:00",
        "output": "/home/ypan/heart_liver/model_train/heart-molecular-pretraining/runs/slurm/%x-%j.out",
        "error": "/home/ypan/heart_liver/model_train/heart-molecular-pretraining/runs/slurm/%x-%j.err",
    }


def test_full_sbatch_directives():
    d = sbatch_directives(read("pretrain_heart.sh"))
    assert d["partition"] == "gpu" and d["gres"] == "gpu:1" and d["nodes"] == "1"
    assert d["job-name"] == "heart-pretrain-full"
    assert d["time"] != "00:30:00"           # the 30-minute limit is for the smoke test only
    assert d["signal"].startswith("B:USR1@")
    assert d["output"].startswith("/home/ypan/heart_liver/model_train/heart-molecular-pretraining/runs/slurm/")


@pytest.mark.parametrize("script", ["pretrain_smoke.sh", "pretrain_heart.sh"])
def test_no_lumi_author_hardware_settings(script):
    text = read(script)
    for bad in ("a100", "qos", "/fs01/home/haotian", "cuda-11.7", "/pkgs/", "torch.distributed.launch --",
                "pip install"):
        assert bad not in text, bad


@pytest.mark.parametrize("script", ["pretrain_smoke.sh", "pretrain_heart.sh"])
def test_scripts_are_direct_unicore_train_lumi_style(script):
    text = read(script)
    assert 'cd "$MODEL_ROOT"' in text
    assert "--user-dir ./unimol" in text
    assert "--task unimol_contrastive --loss unimol_contrastive --arch unimol_contrastive_base" in text
    assert "yaml_to_env" not in text          # parameters are written out plainly
    for var in ("lr=1e-5", "wd=1e-4", "heads=64", "masked_token_loss=1", "masked_coord_loss=5",
                "masked_dist_loss=10", "x_norm_loss=0.01", "delta_pair_repr_norm_loss=0.01",
                "contrastive_loss=10.0", "mask_prob=0.15", 'noise_type="uniform"', "noise=1.0",
                "only_polar=0", "seed=1", "contrastive_view_mode=independent"):
        assert re.search(r"^" + re.escape(var) + r"(\s|$)", text, flags=re.M), var


def test_smoke_and_full_are_distinct():
    smoke, full = read("pretrain_smoke.sh"), read("pretrain_heart.sh")
    code = re.sub(r"^\s*#.*$", "", smoke, flags=re.M)
    assert not re.search(r"\b(sbatch|srun|bash|source|exec)\b[^\n]*pretrain_heart\.sh", code)  # never launches full
    assert not re.search(r"^\s*sbatch\b", code, flags=re.M)
    assert "HEART_CONFIRM_FULL" in full and "HEART_SMOKE_RUN" in full
    assert re.search(r"^warmup_steps=10000$", full, flags=re.M)
    assert re.search(r"^max_steps=1000000$", full, flags=re.M)
    assert re.search(r'^max_update="\$\{MAX_UPDATE:-10\}"', smoke, flags=re.M)
    assert re.search(r"^warmup_steps=3$", smoke, flags=re.M)


OBSOLETE = re.compile(r"/home/ypan/heart_liver/model(?!_train)")


def _repo_files():
    out = subprocess.run(["git", "-C", REPO_ROOT, "ls-files", "--cached", "--others", "--exclude-standard"],
                         capture_output=True, text=True, check=True).stdout.split()
    return [f for f in out if os.path.isfile(os.path.join(REPO_ROOT, f))]


def test_no_obsolete_model_root_in_executables_configs_notebooks():
    offenders = []
    for f in _repo_files():
        if not f.endswith((".sh", ".py", ".yaml", ".ipynb", ".txt")):
            continue
        text = open(os.path.join(REPO_ROOT, f), encoding="utf-8", errors="replace").read()
        for line in text.splitlines():
            if OBSOLETE.search(line) and "test_scripts.py" not in f:
                offenders.append((f, line.strip()))
    assert not offenders, offenders


def test_docs_mention_obsolete_root_only_as_obsolete():
    for f in _repo_files():
        if f.endswith(".md"):
            for line in open(os.path.join(REPO_ROOT, f), encoding="utf-8").read().splitlines():
                if OBSOLETE.search(line):
                    assert "obsolete" in line.lower() or "old" in line.lower(), (f, line)


def test_gitignore_protects_outputs_and_data():
    text = open(os.path.join(REPO_ROOT, ".gitignore"), encoding="utf-8").read()
    for pat in ("runs/*", "*.lmdb", "*.pt", ".venv_heart_cpu/", "third_party/"):
        assert pat in text, pat
    assert os.path.isfile(os.path.join(REPO_ROOT, "runs", "slurm", ".gitkeep"))
    tracked = subprocess.run(["git", "-C", REPO_ROOT, "ls-files"], capture_output=True, text=True).stdout.split()
    assert not [f for f in tracked if f.endswith((".lmdb", ".pt", ".pth")) or f.startswith("runs/smoke")]


# --------------------------------------------------------------------------- run helpers
def base_env(fixture_dir, out_root, **extra):
    py = _posix(sys.executable)
    env = dict(os.environ)
    for k in list(env):
        if k.startswith(("SLURM_", "HEART_")) or k in ("RESUME_DIR", "MAX_UPDATE", "BATCH_SIZE", "UPDATE_FREQ"):
            env.pop(k)
    env.update(
        HEART_REPO=_posix(REPO_ROOT), HEART_SKIP_ENV="1", HEART_PYTHON=py,
        HEART_TRAIN_ENTRY="{} {}".format(py, _posix(os.path.join(TESTS_DIR, "unicore_train_entry.py"))),
        HEART_DATA_PATH=_posix(fixture_dir), HEART_OUTPUT_ROOT=_posix(out_root), PYTHONIOENCODING="utf-8",
    )
    env.update({k: str(v) for k, v in extra.items()})
    return env


def cpu_env(fixture_dir, out_root, **extra):
    return base_env(fixture_dir, out_root, HEART_LOCAL_CPU_TEST="1", HEART_EXTRA_ARGS=TINY, **extra)


def run_script(script, env, args=()):
    return subprocess.run([BASH, os.path.join(SCRIPTS_DIR, script), *args], env=env,
                          capture_output=True, text=True, timeout=900)


def printed_command(stdout):
    lines = stdout.splitlines()
    return shlex.split(lines[lines.index("=== Command ===") + 1])


def yaml_flags(name, overrides=()):
    import yaml_to_env

    _, uc = yaml_to_env.load(os.path.join(CONFIG_DIR, name), list(overrides))
    from check_resume_args import command_values

    return command_values(yaml_to_env.to_cli(uc))


def assert_command_matches_yaml(cmd, yaml_name, overrides=()):
    from check_resume_args import command_values, same

    got = command_values(cmd)
    want = yaml_flags(yaml_name, overrides)
    for k, v in want.items():
        assert k in got, "--{} from {} missing in script command".format(k, yaml_name)
        assert same(got[k], v), "--{}: script {} vs {} {}".format(k, got[k], yaml_name, v)
    extra = set(got) - set(want) - OWNED
    assert not extra, "script flags not recorded in {}: {}".format(yaml_name, sorted(extra))


def fake_smoke_evidence(path, max_update=10, verdict="PASS"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "smoke_result-123.txt").write_text(
        "verdict={}\nmax_update={}\nresume_from=none\n".format(verdict, max_update), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- smoke script
@needs_bash
def test_smoke_dry_run_matches_pilot_record_and_has_no_side_effects(fixture_dir, tmp_path):
    p = run_script("pretrain_smoke.sh", base_env(fixture_dir, tmp_path / "runs", HEART_DRY_RUN=1))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "PASS  task  unimol_contrastive" in p.stdout and "would write to" in p.stdout
    cmd = printed_command(p.stdout)
    assert_command_matches_yaml(cmd, "pilot.yaml")
    assert "--fp16" in cmd and "--cpu" not in cmd
    assert not (tmp_path / "runs" / "smoke").exists() or not any((tmp_path / "runs" / "smoke").iterdir())


@needs_bash
def test_smoke_batch_override_is_explicit(fixture_dir, tmp_path):
    p = run_script("pretrain_smoke.sh", base_env(fixture_dir, tmp_path, HEART_DRY_RUN=1, BATCH_SIZE=2))
    assert p.returncode == 0, p.stderr[-2000:]
    cmd = printed_command(p.stdout)
    assert cmd[cmd.index("--batch-size") + 1] == "2" and "deviation" in p.stderr


@needs_bash
@pytest.mark.parametrize("extra,msg", [
    ({"MAX_UPDATE": "25"}, "MAX_UPDATE must be in 1..20"),
    ({"MAX_UPDATE": "abc"}, "MAX_UPDATE must be an integer"),
    ({"BATCH_SIZE": "0"}, "BATCH_SIZE must be a positive integer"),
    ({"HEART_EXTRA_ARGS": "--restore-file /x.pt"}, "must not contain --restore-file"),
    ({"HEART_EXTRA_ARGS": "--finetune-from-model /x.pt"}, "must not contain --finetune-from-model"),
])
def test_smoke_rejects_bad_overrides(fixture_dir, tmp_path, extra, msg):
    p = run_script("pretrain_smoke.sh", base_env(fixture_dir, tmp_path, HEART_DRY_RUN=1, **extra))
    assert p.returncode != 0 and msg in p.stderr, p.stderr[-2000:]


@needs_bash
def test_smoke_rejects_missing_dataset(tmp_path):
    p = run_script("pretrain_smoke.sh", base_env(tmp_path / "no_such_data", tmp_path))
    assert p.returncode != 0 and "missing or unreadable" in p.stderr


@needs_bash
def test_smoke_refuses_output_inside_dataset(fixture_dir):
    p = run_script("pretrain_smoke.sh", base_env(fixture_dir, os.path.join(fixture_dir, "runs"), HEART_DRY_RUN=1))
    assert p.returncode != 0 and "inside the dataset" in p.stderr


@needs_bash
def test_smoke_trains_verifies_and_resumes(fixture_dir, tmp_path):
    env = cpu_env(fixture_dir, tmp_path, MAX_UPDATE=2)
    p = run_script("pretrain_smoke.sh", env)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "new run, random initialisation" in p.stdout
    assert "PASS  reached 2 optimizer updates" in p.stdout
    assert "PASS  num_updates == 2" in p.stdout
    assert "PASS  model rebuilt from stored args and weights loaded (strict)" in p.stdout
    (run_dir,) = list((tmp_path / "smoke").iterdir())
    assert (run_dir / "checkpoint_last.pt").exists()
    result = (run_dir / "smoke_result-local.txt").read_text()
    assert "verdict=PASS" in result and "max_update=2" in result and "resume_from=none" in result
    (prov,) = list((run_dir / "provenance").iterdir())
    assert {"pretrain_smoke.sh.used", "command.txt", "source_sha256.txt", "pip_freeze.txt",
            "git_commit.txt"} <= {f.name for f in prov.iterdir()}

    env.update(RESUME_DIR=_posix(run_dir), MAX_UPDATE="4")
    p = run_script("pretrain_smoke.sh", env)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "RESUMING from checkpoint_last.pt @ 2 updates" in p.stdout
    assert "Loaded checkpoint" in p.stdout and "@ 2 updates" in p.stdout
    assert "PASS  resumed from checkpoint @ 2 updates" in p.stdout and "PASS  num_updates == 4" in p.stdout
    assert "resume_from=2" in (run_dir / "smoke_result-local.txt").read_text()

    env["MAX_UPDATE"] = "4"  # already reached -> refuse
    p = run_script("pretrain_smoke.sh", env)
    assert p.returncode != 0 and "already at 4 updates" in p.stderr


# --------------------------------------------------------------------------- full script
@needs_bash
@pytest.mark.parametrize("extra,msg", [
    ({}, "requires HEART_CONFIRM_FULL=1"),
    ({"HEART_CONFIRM_FULL": "yes"}, "requires HEART_CONFIRM_FULL=1"),
    ({"HEART_CONFIRM_FULL": "1"}, "set BATCH_SIZE explicitly"),
    ({"HEART_CONFIRM_FULL": "1", "BATCH_SIZE": "64"}, "set HEART_SMOKE_RUN"),
])
def test_full_refuses_without_confirmation_batch_and_smoke(fixture_dir, tmp_path, extra, msg):
    out = tmp_path / "runs"
    p = run_script("pretrain_heart.sh", base_env(fixture_dir, out, **extra))
    assert p.returncode != 0 and msg in p.stderr, p.stderr[-2000:]
    assert not out.exists()  # nothing created


@needs_bash
@pytest.mark.parametrize("max_update,verdict", [(2, "PASS"), (10, "FAIL")])
def test_full_requires_passed_10_update_smoke(fixture_dir, tmp_path, max_update, verdict):
    smoke = fake_smoke_evidence(tmp_path / "smoke", max_update, verdict)
    p = run_script("pretrain_heart.sh", base_env(fixture_dir, tmp_path / "runs", HEART_CONFIRM_FULL=1,
                                                 BATCH_SIZE=64, HEART_SMOKE_RUN=_posix(smoke), HEART_DRY_RUN=1))
    assert p.returncode != 0 and "no smoke_result-*.txt with verdict=PASS" in p.stderr


@needs_bash
def test_full_refuses_outside_slurm_without_dry_run(fixture_dir, tmp_path):
    smoke = fake_smoke_evidence(tmp_path / "smoke")
    p = run_script("pretrain_heart.sh", base_env(fixture_dir, tmp_path / "runs", HEART_CONFIRM_FULL=1,
                                                 BATCH_SIZE=64, HEART_SMOKE_RUN=_posix(smoke)))
    assert p.returncode != 0 and "runs only under sbatch" in p.stderr


@needs_bash
def test_full_dry_run_matches_full_record(fixture_dir, tmp_path):
    smoke = fake_smoke_evidence(tmp_path / "smoke")
    p = run_script("pretrain_heart.sh", base_env(fixture_dir, tmp_path / "runs", HEART_CONFIRM_FULL=1,
                                                 BATCH_SIZE=448, NUM_WORKERS=15, MIN_FREE_GB=0,
                                                 HEART_SMOKE_RUN=_posix(smoke), HEART_DRY_RUN=1))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    cmd = printed_command(p.stdout)
    assert_command_matches_yaml(cmd, "full_pretrain.yaml")
    assert "effective 448 molecules/update" in p.stdout
    assert "--max-valid-steps" not in cmd
    assert not (tmp_path / "runs" / "full" / "heart-pretrain-full-v1").exists()


@needs_bash
def test_full_accumulation_is_reported(fixture_dir, tmp_path):
    smoke = fake_smoke_evidence(tmp_path / "smoke")
    p = run_script("pretrain_heart.sh", base_env(fixture_dir, tmp_path / "runs", HEART_CONFIRM_FULL=1,
                                                 BATCH_SIZE=112, UPDATE_FREQ=4, MIN_FREE_GB=0,
                                                 HEART_SMOKE_RUN=_posix(smoke), HEART_DRY_RUN=1))
    assert p.returncode == 0, p.stderr[-2000:]
    assert "micro-batch 112 x update-freq 4 x GPUs 1 = effective 448" in p.stdout
    assert "not for the contrastive negatives" in p.stderr


@needs_bash
def test_full_trains_resumes_and_refuses_changed_settings(fixture_dir, tmp_path):
    smoke = fake_smoke_evidence(tmp_path / "smoke")
    env = cpu_env(fixture_dir, tmp_path / "runs", HEART_CONFIRM_FULL=1, BATCH_SIZE=2, MIN_FREE_GB=0,
                  HEART_SMOKE_RUN=_posix(smoke))
    env["HEART_EXTRA_ARGS"] = TINY + " --max-update 2 --save-interval-updates 1 --validate-interval-updates 1"
    p = run_script("pretrain_heart.sh", env)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    run_dir = tmp_path / "runs" / "full" / "heart-pretrain-full-v1"
    assert (run_dir / "checkpoint_last.pt").exists()
    assert "checkpoint_last.pt @ 2 / 1000000 updates" in p.stdout

    env["HEART_EXTRA_ARGS"] = TINY + " --max-update 3"
    p = run_script("pretrain_heart.sh", env)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "RESUMING from checkpoint_last.pt @ 2 updates" in p.stdout
    assert "PASS  resume settings match the checkpoint" in p.stdout
    assert "checkpoint_last.pt @ 3 / 1000000 updates" in p.stdout

    env["BATCH_SIZE"] = "3"
    env["HEART_EXTRA_ARGS"] = TINY.replace("--batch-size 2 ", "") + " --max-update 4"
    p = run_script("pretrain_heart.sh", env)
    assert p.returncode != 0 and "MISMATCH batch-size" in p.stdout and "resume settings differ" in p.stderr


# --------------------------------------------------------------------------- check_heart_env.sh
@needs_bash
def test_check_env_help_and_bad_argument():
    p = subprocess.run([BASH, os.path.join(SCRIPTS_DIR, "check_heart_env.sh"), "--help"], capture_output=True, text=True)
    assert p.returncode == 0 and "--fix" in p.stdout
    p = subprocess.run([BASH, os.path.join(SCRIPTS_DIR, "check_heart_env.sh"), "--frobnicate"], capture_output=True, text=True)
    assert p.returncode == 2


def env_check_env(fixture_dir, out_root):
    import unicore

    env = base_env(fixture_dir, out_root)
    env.update(HEART_ENV_PREFIX=_posix(sys.prefix),
               HEART_UNICORE_SRC=_posix(os.path.dirname(os.path.dirname(os.path.abspath(unicore.__file__)))))
    return env


@needs_bash
def test_check_env_default_mode_modifies_nothing(fixture_dir, tmp_path):
    out = tmp_path / "runs"
    p = run_script("check_heart_env.sh", env_check_env(fixture_dir, out))
    assert "=== Mode: check" in p.stdout
    assert "PASS  registry" in p.stdout
    assert "+ " not in p.stdout                     # no install command was run
    assert not out.exists()                         # check mode creates nothing
    if "CUDA build None" in p.stdout:               # local CPU torch: reported, never repaired
        assert p.returncode != 0 and "BLOCK torch" in p.stdout


@needs_bash
def test_check_env_fix_refuses_to_install_over_blockers(fixture_dir, tmp_path):
    import torch

    if torch.version.cuda == "12.1":
        pytest.skip("this interpreter has the CUDA 12.1 torch build; no blocker to test")
    p = run_script("check_heart_env.sh", env_check_env(fixture_dir, tmp_path / "runs"), ["--fix"])
    assert p.returncode != 0
    if "repairs available to --fix: none" not in p.stdout:
        assert "refusing to install into this environment" in p.stdout
    assert not [l for l in p.stdout.splitlines() if l.startswith("+ ")]  # no pip / Uni-Core install ran
    assert list((tmp_path / "runs" / "env").glob("check_heart_env-fix-*.log"))


# --------------------------------------------------------------------------- NUM_WORKERS (job 1011484)
def _flag(cmd, flag):
    from check_resume_args import command_values

    return command_values(cmd)[flag.lstrip("-")]


@needs_bash
@pytest.mark.parametrize("value,expected", [(None, "5"), ("0", "0"), ("2", "2"), ("5", "5")])
def test_smoke_num_workers_reaches_unicore_command(fixture_dir, tmp_path, value, expected):
    extra = {"HEART_DRY_RUN": 1, "SLURM_CPUS_PER_TASK": 6}
    if value is not None:
        extra["NUM_WORKERS"] = value
    p = run_script("pretrain_smoke.sh", base_env(fixture_dir, tmp_path, **extra))
    assert p.returncode == 0, p.stderr[-2000:]
    cmd = printed_command(p.stdout)
    assert _flag(cmd, "--num-workers") == expected
    assert cmd.count("--num-workers") == 1
    assert ("low-memory mode" in p.stdout) == (expected == "0")
    # scientific settings untouched by the worker count
    assert _flag(cmd, "--batch-size") == "4" and _flag(cmd, "--max-update") == "10"
    assert _flag(cmd, "--encoder-attention-heads") == "64" and _flag(cmd, "--contrastive-loss") == "10.0"


@needs_bash
@pytest.mark.parametrize("value,msg", [
    ("-1", "NUM_WORKERS must be a non-negative integer"),
    ("abc", "NUM_WORKERS must be a non-negative integer"),
    ("1.5", "NUM_WORKERS must be a non-negative integer"),
    ("0 --restore-file x", "NUM_WORKERS must be a non-negative integer"),
    ("6", "must be < SLURM_CPUS_PER_TASK=6"),
    ("65", "unreasonably large"),
])
def test_smoke_rejects_invalid_num_workers(fixture_dir, tmp_path, value, msg):
    extra = {"HEART_DRY_RUN": 1, "NUM_WORKERS": value}
    if value == "6":
        extra["SLURM_CPUS_PER_TASK"] = 6
    p = run_script("pretrain_smoke.sh", base_env(fixture_dir, tmp_path, **extra))
    assert p.returncode != 0 and msg in p.stderr, p.stderr[-2000:]


@needs_bash
def test_smoke_trains_with_num_workers_0(fixture_dir, tmp_path):
    """NUM_WORKERS=0 through the real unicore-train CLI (no --num-workers in HEART_EXTRA_ARGS)."""
    tiny_no_workers = TINY.replace("--num-workers 0 ", "")
    assert "--num-workers" not in tiny_no_workers
    env = cpu_env(fixture_dir, tmp_path, MAX_UPDATE=2, NUM_WORKERS=0)
    env["HEART_EXTRA_ARGS"] = tiny_no_workers
    p = run_script("pretrain_smoke.sh", env)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "num_workers=0" in p.stdout                      # Uni-Core Namespace
    assert "PASS  reached 2 optimizer updates" in p.stdout and "PASS  num_updates == 2" in p.stdout
    (run_dir,) = list((tmp_path / "smoke").iterdir())
    result = (run_dir / "smoke_result-local.txt").read_text()
    assert "verdict=PASS" in result and "num_workers=0" in result and "peak_rss_total_mb=" in result
    assert "=== CPU memory limits ===" in p.stdout


@needs_bash
@pytest.mark.parametrize("value,ok", [("0", True), ("-1", False), ("x", False)])
def test_full_num_workers(fixture_dir, tmp_path, value, ok):
    smoke = fake_smoke_evidence(tmp_path / "smoke")
    p = run_script("pretrain_heart.sh", base_env(fixture_dir, tmp_path / "runs", HEART_CONFIRM_FULL=1, BATCH_SIZE=448,
                                                 NUM_WORKERS=value, MIN_FREE_GB=0, HEART_SMOKE_RUN=_posix(smoke),
                                                 HEART_DRY_RUN=1))
    if ok:
        assert p.returncode == 0, p.stderr[-2000:]
        assert _flag(printed_command(p.stdout), "--num-workers") == "0"
    else:
        assert p.returncode != 0 and "num_workers must be a non-negative integer" in p.stderr


@needs_bash
def test_memory_summary_parses_sampler_tsv(tmp_path):
    tsv = tmp_path / "memory-1.tsv"
    tsv.write_text(
        "time\tprocs\trss_total_mb\trss_max_proc_mb\tvsz_max_proc_mb\tcgroup_mb\tshm_used_mb\tmem_available_mb\n"
        "2026-10-28 10:00:00\t7\t9000\t4000\t180000\t9500\t120\t200000\n"
        "2026-10-28 10:00:10\t8\t52000\t9000\t190000\t54000\t31000\t1000\n"
        "2026-10-28 10:00:20\t2\t3000\t3000\t30000\t3100\t0\t250000\n", encoding="utf-8")
    script = ('. "{}"; MEM_MON_PID=""; heart_mem_monitor_stop "{}"; echo "PEAK=$MEM_PEAK_RSS_MB"'
              .format(_posix(os.path.join(SCRIPTS_DIR, "heart_lib.sh")), _posix(tsv)))
    p = subprocess.run([BASH, "-c", script], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "peak total RSS 52000 MB | peak single-process RSS 9000 MB | peak VSZ 190000 MB" in p.stdout
    assert "peak /dev/shm used 31000 MB | max processes 8" in p.stdout and "PEAK=52000" in p.stdout
