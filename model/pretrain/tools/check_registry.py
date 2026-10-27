#!/usr/bin/env python
"""Verify that Uni-Core discovers the HEART/LUMI registrations via --user-dir.

Run from the model root (the directory that contains `unimol/` and `pretrain/`):
    cd /home/ypan/heart_liver/model
    python pretrain/tools/check_registry.py --user-dir ./unimol

Exit code 0 only if the task, model, architecture and loss are registered and
`unimol` resolves to the directory passed with --user-dir (not to some other
installed package).
"""
import argparse
import os
import sys
from types import SimpleNamespace

EXPECTED = {
    "task": "unimol_contrastive",
    "model": "unimol_contrastive",
    "arch": "unimol_contrastive_base",
    "loss": "unimol_contrastive",
}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-dir", default="./unimol")
    ns = parser.parse_args(argv)

    import unicore
    from unicore import utils
    from unicore.losses import CRITERION_REGISTRY as LOSS_REGISTRY
    from unicore.models import ARCH_MODEL_REGISTRY, MODEL_REGISTRY
    from unicore.tasks import TASK_REGISTRY

    user_dir = os.path.abspath(ns.user_dir)
    print("cwd        :", os.getcwd())
    print("user-dir   :", user_dir)
    print("unicore    :", unicore.__file__)
    utils.import_user_module(SimpleNamespace(user_dir=ns.user_dir))
    unimol = sys.modules.get("unimol")
    unimol_dir = os.path.dirname(os.path.abspath(unimol.__file__)) if unimol else None
    print("unimol     :", unimol_dir)

    checks = [
        ("unimol package resolves to --user-dir", unimol_dir == user_dir),
        ("task  " + EXPECTED["task"], EXPECTED["task"] in TASK_REGISTRY),
        ("model " + EXPECTED["model"], EXPECTED["model"] in MODEL_REGISTRY),
        ("arch  " + EXPECTED["arch"], EXPECTED["arch"] in ARCH_MODEL_REGISTRY),
        ("loss  " + EXPECTED["loss"], EXPECTED["loss"] in LOSS_REGISTRY),
    ]
    ok = True
    for name, passed in checks:
        print("{}  {}".format("PASS" if passed else "FAIL", name))
        ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
