#!/usr/bin/env python
"""Summarise a Uni-Core training log (train/valid losses, checkpoints, resume).

Usage:
    python tools/summarize_log.py <save_dir>/train-<jobid>.log [--expect-updates 10]
                                  [--expect-resume-from 10]

Exit code 0 when every requested expectation holds and all logged losses are
finite; prints PASS/FAIL lines so the output can be pasted back for review.
"""
import argparse
import math
import re
import sys

TRAIN_RE = re.compile(r"\| train_inner \| epoch (\d+):\s+(\d+) / (\d+) (.*)$")
VALID_RE = re.compile(r"\| valid \| epoch (\d+) \| valid on '([^']+)' subset \| (.*)$")
KV_COMMA = re.compile(r"([A-Za-z_]+)=([^,]+)")
LOADED_RE = re.compile(r"Loaded checkpoint (.+) \(epoch (\d+) @ (\d+) updates\)")
SAVED_RE = re.compile(r"Saved checkpoint (.+?) \(epoch (\d+) @ (\d+) updates")
NO_CKPT_RE = re.compile(r"No existing checkpoint found")
STOP_RE = re.compile(r"Stopping training due to (.*)$")
LOSS_KEYS = [
    "loss", "masked_token_loss", "masked_acc", "masked_coord_loss", "masked_dist_loss",
    "contrastive_loss", "x_norm_loss", "delta_pair_repr_norm_loss",
]


def to_float(v):
    try:
        return float(v)
    except ValueError:
        return float("nan")


def parse(path):
    out = {"train": [], "valid": [], "loaded": [], "saved": [], "stops": [],
           "fresh_start": False, "overflow": 0, "errors": [], "done": False}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = TRAIN_RE.search(line)
            if m:
                kv = {k: to_float(v) for k, v in KV_COMMA.findall(m.group(4))}
                out["train"].append(kv)
                continue
            m = VALID_RE.search(line)
            if m:
                kv = {"subset": m.group(2)}
                for seg in m.group(3).split(" | "):
                    parts = seg.strip().split(" ", 1)
                    if len(parts) == 2:
                        kv[parts[0]] = to_float(parts[1])
                out["valid"].append(kv)
                continue
            m = LOADED_RE.search(line)
            if m:
                out["loaded"].append((m.group(1), int(m.group(3))))
            m = SAVED_RE.search(line)
            if m:
                out["saved"].append((m.group(1), int(m.group(3))))
            if NO_CKPT_RE.search(line):
                out["fresh_start"] = True
            m = STOP_RE.search(line)
            if m:
                out["stops"].append(m.group(1))
            if "gradient overflow detected" in line:
                out["overflow"] += 1
            if "done training in" in line:
                out["done"] = True
            if line.startswith("Traceback") or "Error:" in line or "CUDA out of memory" in line:
                out["errors"].append(line.strip())
    return out


def table(rows, keys):
    keys = [k for k in keys if any(k in r for r in rows)]
    head = "{:>8s} ".format("update") + " ".join("{:>13s}".format(k[:13]) for k in keys)
    lines = [head]
    for r in rows:
        upd = r.get("num_updates", float("nan"))
        lines.append("{:>8.0f} ".format(upd) + " ".join(
            "{:>13.5g}".format(r[k]) if k in r else "{:>13s}".format("-") for k in keys))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--expect-updates", type=int)
    ap.add_argument("--expect-resume-from", type=int)
    ns = ap.parse_args(argv)
    r = parse(ns.log)

    print("== train (per logged update) ==")
    print(table(r["train"], LOSS_KEYS + ["lr", "gnorm", "loss_scale", "gb_free"]))
    print("\n== valid ==")
    print(table(r["valid"], LOSS_KEYS))
    print("\n== checkpoints saved ==")
    for p, u in r["saved"]:
        print("  @{:>6d} updates  {}".format(u, p))
    print("\nstart       :", "fresh (random init)" if r["fresh_start"] else
          "resumed from " + ", ".join("{} @ {}".format(p, u) for p, u in r["loaded"]) if r["loaded"] else "unknown")
    print("stop reason :", "; ".join(r["stops"]) or "-")
    print("fp16 overflow (skipped) steps:", r["overflow"])

    checks = []
    finite = all(math.isfinite(row[k]) for row in r["train"] + r["valid"] for k in LOSS_KEYS if k in row)
    checks.append(("train losses logged", len(r["train"]) > 0))
    checks.append(("validation ran", len(r["valid"]) > 0))
    checks.append(("all logged losses finite", finite))
    checks.append(("checkpoint saved", len(r["saved"]) > 0))
    checks.append(("no Python traceback / CUDA OOM in log", not r["errors"]))
    last_upd = max([int(t.get("num_updates", 0)) for t in r["train"]] or [0])
    if ns.expect_updates is not None:
        checks.append(("reached {} optimizer updates (last logged: {})".format(ns.expect_updates, last_upd),
                       last_upd == ns.expect_updates))
    if ns.expect_resume_from is not None:
        checks.append(("resumed from checkpoint @ {} updates".format(ns.expect_resume_from),
                       any(u == ns.expect_resume_from for _, u in r["loaded"])))
    print()
    for name, ok in checks:
        print("{}  {}".format("PASS" if ok else "FAIL", name))
    for e in r["errors"][:5]:
        print("   log error:", e)
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
