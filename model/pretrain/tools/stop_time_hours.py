#!/usr/bin/env python
"""Compute Uni-Core's --stop-time-hours so a job stops cleanly before SLURM kills it.

Uni-Core compares --stop-time-hours with the *cumulative* training time, which
is carried across resumes in checkpoint_last.pt (extra_state
"previous_training_time"). A per-job budget therefore has to be added to the
time already accumulated:

    stop_time_hours = previous_training_time + (time_left - margin)

When the threshold is crossed, Uni-Core validates, writes checkpoint_last.pt
and exits normally; resubmitting the same run directory resumes from there.

Prints a float (hours); prints 0 (disabled) when the time left is unknown.
"""
import argparse
import os
import re
import sys


def parse_slurm_time(text):
    """Parse squeue %L / sbatch --time formats into seconds; None if unknown."""
    text = (text or "").strip()
    if not text or text.upper() in ("UNLIMITED", "INVALID", "NOT_SET", "N/A"):
        return None
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
        parts = [int(p) for p in text.split(":")]
        while len(parts) < 3:  # D-HH, D-HH:MM
            parts.append(0)
    else:
        parts = [int(p) for p in text.split(":")]
        while len(parts) < 3:  # MM, MM:SS
            parts.insert(0, 0)
    if not re.fullmatch(r"\d+", str(days)) or len(parts) != 3:
        return None
    h, m, s = parts
    return ((days * 24 + h) * 60 + m) * 60 + s


def previous_training_seconds(save_dir):
    path = os.path.join(save_dir, "checkpoint_last.pt")
    if not os.path.isfile(path):
        return 0.0
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only
        state = torch.load(path, map_location="cpu")
    extra = state.get("extra_state") or {}
    return float(extra.get("previous_training_time", 0.0))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-left", required=True, help="squeue -o %%L output")
    parser.add_argument("--margin-minutes", type=float, default=10.0)
    parser.add_argument("--save-dir", required=True)
    ns = parser.parse_args(argv)

    try:
        left = parse_slurm_time(ns.time_left)
    except ValueError:
        left = None
    if left is None:
        sys.stderr.write("stop_time_hours: time left {!r} unknown; disabling\n".format(ns.time_left))
        print(0)
        return 0
    budget = left - ns.margin_minutes * 60
    if budget <= 60:
        sys.stderr.write("stop_time_hours: only {}s left after margin\n".format(budget))
        budget = 60
    prev = previous_training_seconds(ns.save_dir)
    hours = (prev + budget) / 3600.0
    sys.stderr.write(
        "stop_time_hours: previous {:.3f} h + budget {:.3f} h = {:.4f} h\n".format(
            prev / 3600.0, budget / 3600.0, hours
        )
    )
    print("{:.4f}".format(hours))
    return 0


if __name__ == "__main__":
    sys.exit(main())
