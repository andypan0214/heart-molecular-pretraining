#!/usr/bin/env python
"""Verify a Uni-Core checkpoint written by HEART pretraining and reload it.

Run from the model root (so ./unimol is importable):
    cd /home/ypan/heart_liver/model_train/heart-molecular-pretraining/model
    python pretrain/tools/verify_checkpoint.py <save_dir>/checkpoint_last.pt [--expect-updates 10]

Checks: required Uni-Core keys, update counter, finite weights, random-init
provenance (no --finetune-from-model), and -- unless --no-rebuild -- that the
registered unimol_contrastive model can be rebuilt from the stored args and
load the weights with strict=True. Runs on CPU; no GPU needed.
"""
import argparse
import os
import sys
from types import SimpleNamespace

import torch


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13
        return torch.load(path, map_location="cpu")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--expect-updates", type=int)
    ap.add_argument("--user-dir", default="./unimol")
    ap.add_argument("--no-rebuild", action="store_true")
    ap.add_argument("--data", help="directory holding dict.txt (default: the run's data path)")
    ns = ap.parse_args(argv)

    ck = load(ns.checkpoint)
    checks = []
    required = {"args", "model", "optimizer_history", "extra_state", "last_optimizer_state"}
    checks.append(("required keys {}".format(sorted(required)), required <= set(ck)))
    args = ck["args"]
    hist = ck["optimizer_history"][-1]
    n_upd = hist["num_updates"]
    extra = ck["extra_state"]
    n_params = sum(t.numel() for t in ck["model"].values())
    finite = all(torch.isfinite(t.float()).all().item() for t in ck["model"].values() if t.is_floating_point())
    print("file            :", ns.checkpoint, "({:.1f} MB)".format(os.path.getsize(ns.checkpoint) / 2 ** 20))
    print("task/arch/loss  :", args.task, args.arch, args.loss)
    print("num_updates     :", n_upd)
    print("train_iterator  :", extra.get("train_iterator"))
    print("val_loss / best :", extra.get("val_loss"), "/", extra.get("best"))
    print("training time   : {:.1f} s".format(extra.get("previous_training_time", float("nan"))))
    print("parameters      : {:,}".format(n_params))
    print("data            :", getattr(args, "data", None))
    print("fp16 / bf16     :", getattr(args, "fp16", None), "/", getattr(args, "bf16", None))
    checks.append(("all model weights finite", finite))
    checks.append(("random initialisation (no --finetune-from-model)",
                   getattr(args, "finetune_from_model", None) is None))
    checks.append(("optimizer state stored", bool(ck.get("last_optimizer_state"))))
    if ns.expect_updates is not None:
        checks.append(("num_updates == {}".format(ns.expect_updates), n_upd == ns.expect_updates))

    if not ns.no_rebuild:
        try:
            from unicore import tasks, utils

            utils.import_user_module(SimpleNamespace(user_dir=ns.user_dir))
            if ns.data:
                args.data = ns.data
            task = tasks.setup_task(args)  # reads <data>/dict.txt only
            model = task.build_model(args)
            model.load_state_dict(ck["model"], strict=True)
            checks.append(("model rebuilt from stored args and weights loaded (strict)", True))
        except Exception as exc:
            print("rebuild error   :", repr(exc))
            checks.append(("model rebuilt from stored args and weights loaded (strict)", False))

    print()
    for name, ok in checks:
        print("{}  {}".format("PASS" if ok else "FAIL", name))
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
