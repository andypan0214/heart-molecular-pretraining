"""The HPC real-LMDB inspector, exercised on the synthetic fixture (read-only)."""
import os

import inspect_real_lmdb

from conftest import MODEL_ROOT


def test_inspector_on_fixture(fixture_dir, tmp_path, monkeypatch, unimol_pkg):
    monkeypatch.chdir(MODEL_ROOT)
    before = {f: os.path.getmtime(os.path.join(fixture_dir, f)) for f in os.listdir(fixture_dir)}
    out = tmp_path / "report.json"
    rc = inspect_real_lmdb.main([fixture_dir, "--samples", "3", "--pipeline", "--pipeline-batch", "2",
                                 "--expected-train", "6", "--expected-valid", "2", "--json", str(out)])
    assert rc == 0 and out.exists()
    after = {f: os.path.getmtime(os.path.join(fixture_dir, f)) for f in os.listdir(fixture_dir)}
    assert before == after  # nothing written to the data directory


def test_inspector_refuses_report_inside_data_dir(fixture_dir):
    rc = inspect_real_lmdb.main([fixture_dir, "--samples", "3", "--expected-train", "6",
                                 "--expected-valid", "2", "--json", os.path.join(fixture_dir, "r.json")])
    assert rc == 2
    assert not os.path.exists(os.path.join(fixture_dir, "r.json"))
