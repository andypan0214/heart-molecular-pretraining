"""tools/heart_env_check.py and tools/check_resume_args.py (pure-Python parts)."""
import os
import subprocess
import sys

import pytest

from conftest import PRETRAIN_DIR, TOOLS_DIR


def test_requirements_parse_with_install_pins_and_optional():
    import heart_env_check as h

    reqs = {r["dist"]: r for r in h.parse_requirements(os.path.join(PRETRAIN_DIR, "requirements-heart.txt"))}
    assert "torch" not in reqs and "unicore" not in reqs      # never installed by --fix
    assert reqs["numpy"]["spec"] == ">=1.24,<2" and reqs["numpy"]["install"] == "numpy==1.26.4"
    assert reqs["pyyaml"]["import_name"] == "yaml" and reqs["tensorboardX"]["import_name"] == "tensorboardX"
    assert reqs["rdkit"]["install"] == "rdkit==2024.9.6"
    for name in ("numpy", "lmdb", "rdkit", "scipy", "tokenizers", "tensorboardX", "pyyaml", "ipykernel"):
        assert not reqs[name]["optional"], name
    assert reqs["pytest"]["optional"]
    for r in reqs.values():                                   # every install pin satisfies its own range
        pin = r["install"].split("==", 1)[1]
        assert h.version_ok(pin, r["spec"]), r


@pytest.mark.parametrize("version,spec,ok", [
    ("1.26.4", ">=1.24,<2", True), ("2.0.1", ">=1.24,<2", False), ("2024.09.6", ">=2022.9,<2026", True),
    ("0.21.0", "<0.21", False), ("6.0.3", ">=6", True), ("anything", "", True),
])
def test_version_ok(version, spec, ok):
    import heart_env_check as h

    assert h.version_ok(version, spec) is ok


def test_under(tmp_path):
    import heart_env_check as h

    (tmp_path / "env" / "lib").mkdir(parents=True)
    assert h.under(str(tmp_path / "env" / "lib"), str(tmp_path / "env"))
    assert not h.under(str(tmp_path / "env2"), str(tmp_path / "env"))
    assert not h.under(None, str(tmp_path))


def test_git_head_detached_branch_and_packed(tmp_path):
    import heart_env_check as h

    sha = "ace6fae1c8479a9751f2bb1e1d6e4047427bc134"
    g = tmp_path / "repo" / ".git"
    (g / "refs" / "heads").mkdir(parents=True)
    (g / "HEAD").write_text(sha + "\n")
    assert h.git_head(str(tmp_path / "repo")) == sha
    (g / "HEAD").write_text("ref: refs/heads/main\n")
    (g / "packed-refs").write_text("# pack-refs\n{} refs/heads/main\n".format(sha))
    assert h.git_head(str(tmp_path / "repo")) == sha
    (g / "refs" / "heads" / "main").write_text("0" * 40 + "\n")
    assert h.git_head(str(tmp_path / "repo")) == "0" * 40
    assert h.git_head(str(tmp_path / "nope")) is None


def test_env_check_blocks_foreign_interpreter_and_lists_no_torch_repair(tmp_path):
    """Running with a prefix that is not this interpreter's must BLOCK (exit 2), never repair torch."""
    fix = tmp_path / "fix.txt"
    p = subprocess.run([sys.executable, os.path.join(TOOLS_DIR, "heart_env_check.py"),
                        "--env-prefix", str(tmp_path / "heart_gpu"),
                        "--requirements", os.path.join(PRETRAIN_DIR, "requirements-heart.txt"),
                        "--unicore-src", str(tmp_path / "no-unicore"),
                        "--unicore-commit", "ace6fae1c8479a9751f2bb1e1d6e4047427bc134",
                        "--emit-fix", str(fix)], capture_output=True, text=True)
    assert p.returncode == 2, p.stdout
    assert "BLOCK interpreter" in p.stdout and "BLOCK Uni-Core source checkout not found" in p.stdout
    assert not any(line.startswith("torch") for line in fix.read_text().splitlines())


def test_command_values_last_flag_wins():
    from check_resume_args import command_values, same

    v = command_values(["unicore-train", "/data", "--batch-size", "4", "--fp16", "--ddp-backend=c10d",
                        "--adam-betas", "(0.9, 0.99)", "--batch-size", "2"])
    assert v["batch-size"] == "2" and v["fp16"] is True and v["ddp-backend"] == "c10d"
    assert v["adam-betas"] == "(0.9, 0.99)"
    assert same([1e-05], "1e-5") and same([1], "1") and same("independent", "independent")
    assert not same(2, "3") and not same("identical", "independent")
