#!/usr/bin/env python
"""Refuse to resume a run with different scientific settings than its checkpoint.

    python tools/check_resume_args.py <run>/checkpoint_last.pt [--allow-change] -- <unicore-train command ...>

The command is parsed the way argparse treats repeated flags (the last value
wins). Each flag in KEYS that appears in the command is compared with the args
stored in the checkpoint. Exit 1 on any mismatch unless --allow-change.
"""
import argparse
import sys

# Settings that define the training run; changing them mid-run changes the experiment.
KEYS = [
    "batch-size", "update-freq", "lr", "weight-decay", "warmup-updates", "total-num-update",
    "seed", "arch", "task", "loss", "encoder-layers", "encoder-embed-dim",
    "encoder-ffn-embed-dim", "encoder-attention-heads", "max-atoms", "mask-prob",
    "noise-type", "noise", "only-polar", "masked-token-loss", "masked-coord-loss",
    "masked-dist-loss", "x-norm-loss", "delta-pair-repr-norm-loss", "contrastive-loss",
    "contrastive-view-mode", "adam-betas", "adam-eps", "clip-norm", "lr-scheduler", "optimizer",
]


def command_values(cmd):
    values = {}
    for i, tok in enumerate(cmd):
        if tok.startswith("--"):
            key = tok[2:]
            if "=" in key:
                key, val = key.split("=", 1)
            elif i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                val = cmd[i + 1]
            else:
                val = True
            values[key] = val
    return values


def same(stored, wanted):
    if isinstance(stored, (list, tuple)) and len(stored) == 1:  # Uni-Core stores lr / update_freq as lists
        stored = stored[0]
    if str(stored) == str(wanted):
        return True
    try:
        return float(stored) == float(wanted)
    except (TypeError, ValueError):
        return False


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        sys.exit("usage: check_resume_args.py CHECKPOINT [--allow-change] -- COMMAND ...")
    split = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--allow-change", action="store_true")
    ns = ap.parse_args(argv[:split])
    wanted = command_values(argv[split + 1:])

    import torch

    try:
        ck = torch.load(ns.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13
        ck = torch.load(ns.checkpoint, map_location="cpu")
    args = ck["args"]
    bad = []
    for key in KEYS:
        if key not in wanted:
            continue
        stored = getattr(args, key.replace("-", "_"), None)
        if not same(stored, wanted[key]):
            bad.append("{}: checkpoint {} -> command {}".format(key, stored, wanted[key]))
    for b in bad:
        print(("WARNING (allowed) " if ns.allow_change else "MISMATCH ") + b)
    if not bad:
        print("PASS  resume settings match the checkpoint ({} keys compared)".format(
            sum(k in wanted for k in KEYS)))
    return 0 if ns.allow_change or not bad else 1


if __name__ == "__main__":
    sys.exit(main())
