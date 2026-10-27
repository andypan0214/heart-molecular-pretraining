"""Test-only entry point equivalent to the `unicore-train` console script.

The local tests run Uni-Core from a source checkout on PYTHONPATH (no
console script installed), so they invoke `unicore_cli.train.cli_main`
through this file.

Windows-only shim: Uni-Core's checkpoint writer does
`os.rename(file + ".tmp", file)`, which cannot overwrite an existing file on
Windows (it can on Linux, where the HPC runs). On win32 only, we alias
os.rename to os.replace so local CPU tests can overwrite checkpoint_best.pt /
checkpoint_last.pt. Nothing here is used by the SLURM launcher.
"""
import os
import sys

if sys.platform == "win32":
    os.rename = os.replace

from unicore_cli.train import cli_main  # noqa: E402

if __name__ == "__main__":
    sys.argv[0] = "unicore-train"
    cli_main()
