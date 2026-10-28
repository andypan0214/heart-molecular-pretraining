"""notebooks/HEART_HPC_Validation.ipynb: structure, safety, and local execution of Parts 0-E and G.

Parts 0-E run against the synthetic fixture in a kernel stand-in
(tests/run_notebook_cells.py). Checks that need the real CMRI environment
(heart_gpu prefix, CUDA torch, installed Uni-Core, real entry counts) are
expected to FAIL here; the test asserts the data/model/pipeline checks PASS.
"""
import json
import os
import re
import subprocess
import sys

import pytest

from conftest import MODEL_ROOT, PRETRAIN_DIR, SCRIPTS_DIR, TESTS_DIR, find_bash

NB = os.path.join(PRETRAIN_DIR, "notebooks", "HEART_HPC_Validation.ipynb")
BASH = find_bash()


@pytest.fixture(scope="module")
def nb():
    with open(NB, encoding="utf-8") as fh:
        return json.load(fh)


def src(cell):
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def test_notebook_is_valid_nbformat4_with_heart_gpu_kernel(nb):
    assert nb["nbformat"] == 4 and nb["metadata"]["kernelspec"]["name"] == "heart_gpu"
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids))
    for c in nb["cells"]:
        assert c["cell_type"] in ("markdown", "code")
        if c["cell_type"] == "code":
            compile(src(c), c["id"], "exec")          # every code cell is valid Python
            assert c["outputs"] == [] and c["execution_count"] is None


def test_notebook_is_step_by_step(nb):
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    md = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    assert len(code) >= 30 and len(md) >= 8
    assert max(len(src(c).splitlines()) for c in code) <= 40     # short cells
    parts = {c["metadata"]["heart_part"] for c in code}
    assert parts == {"0", "A", "B", "C", "D", "E", "F", "G"}
    for c in code:
        first = src(c).splitlines()[0]
        assert re.match(r"# \[(CPU|GPU optional|SLURM)\]", first), first
    headings = " ".join(src(c) for c in md)
    for part in "ABCDEFG":
        assert "## Part " + part in headings


def test_notebook_is_safe(nb):
    code = "\n".join(src(c) for c in nb["cells"] if c["cell_type"] == "code")
    assert not re.search(r"^\s*[!%]", code, flags=re.M)          # no shell / magics
    assert '"pip"' not in code and "'pip'" not in code and "pip install" not in code.replace(
        'print("cd {} && bash scripts/check_heart_env.sh --fix"', "")
    assert "SUBMIT_SMOKE = False" in code and "SUBMIT_RESUME = False" in code
    assert "pretrain_heart.sh" not in code                       # never submits full training
    assert "readonly=True" in code and "lock=False" in code
    assert "/home/ypan/heart_liver/model_train" in code


def _env(fixture_dir, out_root):
    import unicore

    env = dict(os.environ)
    env.update(
        HEART_REPO=os.path.dirname(MODEL_ROOT), HEART_DATA_PATH=fixture_dir, HEART_OUTPUT_ROOT=str(out_root),
        HEART_UNICORE_SRC=os.path.dirname(os.path.dirname(os.path.abspath(unicore.__file__))),
        HEART_CONDA_ENV=sys.prefix, HEART_EXPECTED_TRAIN="6", HEART_EXPECTED_VALID="2",
        PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8",
    )
    return env


def _run_parts(parts, env, out_json):
    p = subprocess.run([sys.executable, os.path.join(TESTS_DIR, "run_notebook_cells.py"), NB,
                        "--parts", parts, "--json", str(out_json)], env=env, capture_output=True,
                       text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    return json.load(open(out_json, encoding="utf-8"))["results"]


def test_notebook_parts_A_to_E_on_fixture(fixture_dir, tmp_path):
    (tmp_path / "runs" / "slurm").mkdir(parents=True)
    res = _run_parts("0,A,B,C,D,E", _env(fixture_dir, tmp_path / "runs"), tmp_path / "r.json")
    status = {k: v[0] for k, v in res.items()}
    must_pass = [k for k in status if k.startswith(("A1", "A2", "A4", "A5", "C3", "C4", "C5", "D", "E"))
                 and not k.startswith("D6 padding")]
    assert must_pass and all(status[k] == "PASS" for k in must_pass), {k: res[k] for k in must_pass if status[k] != "PASS"}
    assert status["C5 parameter count 48,144,227"] == "PASS"
    assert status["D5 views differ (independent mode)"] == "PASS"
    assert status["E3 gradients finite (no NaN/Inf)"] == "PASS"


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_notebook_part_G_reviews_a_real_smoke_run(fixture_dir, tmp_path):
    """Simulate a finished job: run pretrain_smoke.sh (10 tiny CPU updates) with its stdout as the SLURM .out."""
    from test_scripts import cpu_env

    runs = tmp_path / "runs"
    (runs / "slurm").mkdir(parents=True)
    env = cpu_env(fixture_dir, runs)
    # the notebook reads "run dir: <path>" back with native Python: on Windows use C:/... not /c/...
    env["HEART_OUTPUT_ROOT"] = str(runs).replace("\\", "/")
    with open(runs / "slurm" / "heart-smoke-test-local.out", "w") as out, \
            open(runs / "slurm" / "heart-smoke-test-local.err", "w") as err:
        p = subprocess.run([BASH, os.path.join(SCRIPTS_DIR, "pretrain_smoke.sh")],
                           env=env, stdout=out, stderr=err, timeout=900)
    assert p.returncode == 0
    (runs / "slurm" / "last_smoke_job.txt").write_text("local\n")
    res = _run_parts("0,F,G", _env(fixture_dir, runs), tmp_path / "g.json")
    for key in ("G1 run directory exists", "G2 10 updates, finite train+valid losses, checkpoint saved",
                "G3 checkpoint_last.pt @ 10 updates, reloadable",
                "G4 no fatal CUDA OOM / data-loader / Python errors", "G4 script verdict file says PASS"):
        assert res[key][0] == "PASS", (key, res[key])


def test_notebook_part_G_without_job_records_nothing(fixture_dir, tmp_path):
    (tmp_path / "runs" / "slurm").mkdir(parents=True)
    res = _run_parts("0,F,G", _env(fixture_dir, tmp_path / "runs"), tmp_path / "g.json")
    assert not [k for k in res if k.startswith(("F", "G"))]
