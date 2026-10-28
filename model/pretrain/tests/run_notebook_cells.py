#!/usr/bin/env python
"""Execute the code cells of selected notebook parts in one namespace (kernel stand-in).

    python tests/run_notebook_cells.py notebooks/HEART_HPC_Validation.ipynb --parts 0,A,B,C,D,E --json out.json

Used by tests/test_notebook.py on the synthetic fixture (paths come from the
HEART_* environment variables the notebook reads). Cells raise only on real
errors; PASS/FAIL results are collected from the notebook's RESULTS dict.
"""
import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--parts", default="0,A,B,C,D,E")
    ap.add_argument("--json", required=True)
    ns = ap.parse_args(argv)
    parts = set(ns.parts.split(","))
    nb = json.load(open(ns.notebook, encoding="utf-8"))
    glb = {"__name__": "__heart_notebook__"}
    executed = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code" or cell["metadata"].get("heart_part") not in parts:
            continue
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        print("\n##### cell", cell.get("id"), src.splitlines()[0])
        exec(compile(src, cell.get("id", "cell"), "exec"), glb)
        executed.append(cell.get("id"))
    with open(ns.json, "w", encoding="utf-8") as fh:
        json.dump({"executed": executed, "results": glb.get("RESULTS", {})}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
