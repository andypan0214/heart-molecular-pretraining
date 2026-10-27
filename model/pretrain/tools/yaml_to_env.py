#!/usr/bin/env python
"""Translate a HEART YAML config into shell assignments for pretrain_heart.sh.

Usage (inside the launcher):
    eval "$(python tools/yaml_to_env.py configs/pilot.yaml --set max-update=12)"

Emits:
    HEART_<KEY>=<value>        one per entry of the `launcher:` section
    HEART_UNICORE_ARGS=(...)   bash array of `unicore-train` flags built from
                               the `unicore:` section (true -> bare flag,
                               false/null -> omitted)

Paths and options that the launcher owns (data path, save dir, user dir ...)
are rejected in `unicore:` so a config cannot silently redirect output or load
pretrained weights. HEART pretraining always starts from random weights.
"""
import argparse
import shlex
import sys

try:
    import yaml
except ImportError:  # pragma: no cover - reported to the user by the launcher
    sys.stderr.write(
        "ERROR: PyYAML is not importable in this Python ({}). "
        "Install it into heart_gpu: python -m pip install pyyaml\n".format(sys.executable)
    )
    sys.exit(2)

REQUIRED_LAUNCHER = [
    "run_name",
    "data_path",
    "output_root",
    "n_gpu",
    "run_dir_mode",
]
REQUIRED_UNICORE = [
    "task",
    "loss",
    "arch",
    "batch-size",
    "max-update",
    "save-interval-updates",
    "validate-interval-updates",
    "max-valid-steps",
]
# Owned by the launcher (computed per job) or forbidden for from-scratch runs.
FORBIDDEN_UNICORE = {
    "user-dir": "set by the launcher (./unimol relative to the model root)",
    "save-dir": "set by the launcher from launcher.output_root/run_name",
    "tmp-save-dir": "set by the launcher (= save dir; Uni-Core defaults to ./)",
    "tensorboard-logdir": "set by the launcher (<save dir>/tsb)",
    "stop-time-hours": "computed by the launcher from the SLURM time limit",
    "finetune-from-model": "HEART pretraining is from random initialisation",
    "restore-file": "resume is automatic from <save dir>/checkpoint_last.pt; "
    "pretrained weights must not be loaded",
    "reset-optimizer": "not allowed for from-scratch pretraining",
    "reset-dataloader": "not allowed for from-scratch pretraining",
    "reset-lr-scheduler": "not allowed for from-scratch pretraining",
    "reset-meters": "not allowed for from-scratch pretraining",
}


def parse_scalar(text):
    """Parse a --set value with YAML rules (so 1e-5, true, 12 behave as in the file)."""
    return yaml.safe_load(text)


def to_cli(unicore_cfg):
    args = []
    for key, value in unicore_cfg.items():
        if value is None or value is False:
            continue
        flag = "--" + key
        if value is True:
            args.append(flag)
        elif isinstance(value, (list, tuple)):
            args.append(flag)
            args.extend(str(v) for v in value)
        else:
            args.extend([flag, str(value)])
    return args


def load(path, overrides):
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    launcher = dict(cfg.get("launcher") or {})
    unicore = dict(cfg.get("unicore") or {})

    for item in overrides:
        if "=" not in item:
            raise SystemExit("ERROR: --set expects key=value, got {!r}".format(item))
        key, value = item.split("=", 1)
        key = key.strip()
        if key.startswith("launcher."):
            launcher[key[len("launcher."):]] = parse_scalar(value)
        else:
            unicore[key.lstrip("-")] = parse_scalar(value)

    missing = [k for k in REQUIRED_LAUNCHER if k not in launcher]
    missing += ["unicore." + k for k in REQUIRED_UNICORE if k not in unicore]
    if missing:
        raise SystemExit("ERROR: {} is missing keys: {}".format(path, ", ".join(missing)))
    for key, why in FORBIDDEN_UNICORE.items():
        if unicore.get(key) not in (None, False):
            raise SystemExit("ERROR: unicore.{} must not be set in the config: {}".format(key, why))
    if launcher["run_dir_mode"] not in ("timestamped", "fixed"):
        raise SystemExit("ERROR: launcher.run_dir_mode must be 'timestamped' or 'fixed'")
    if int(unicore["max-update"]) <= 0:
        raise SystemExit("ERROR: unicore.max-update must be > 0")
    return launcher, unicore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a unicore flag (e.g. max-update=12) or a launcher key "
        "(e.g. launcher.run_name=foo); may be repeated",
    )
    ns = parser.parse_args(argv)
    launcher, unicore = load(ns.config, ns.set)

    lines = []
    for key, value in launcher.items():
        if isinstance(value, bool):
            value = int(value)
        lines.append("HEART_{}={}".format(key.upper(), shlex.quote(str(value))))
    lines.append(
        "HEART_UNICORE_ARGS=({})".format(" ".join(shlex.quote(a) for a in to_cli(unicore)))
    )
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
