#!/usr/bin/env python
"""Check that the running interpreter is the heart_gpu env and has HEART's dependencies.

Called by scripts/check_heart_env.sh with the heart_gpu python. It only reads:
imports, package metadata, file locations. With --emit-fix FILE it also writes
the list of repairs (pip requirement specs, and/or the token UNICORE) that
check_heart_env.sh --fix may apply. Nothing is installed from here.

    python tools/heart_env_check.py --env-prefix /home/ypan/.conda/envs/heart_gpu \
        --requirements requirements-heart.txt \
        --unicore-src /home/ypan/heart_liver/model_train/third_party/Uni-Core \
        --unicore-commit ace6fae1c8479a9751f2bb1e1d6e4047427bc134 [--emit-fix FILE]

Requirement lines may carry an install pin, used only when a repair is needed:
    rdkit>=2022.9,<2026   # install: rdkit==2024.9.6
Lines whose comment contains "optional" only produce WARN.

Exit code: 0 all required checks pass, 1 something is missing/incompatible,
2 a blocker that --fix must not touch (wrong interpreter, torch/CUDA problem).
"""
import argparse
import importlib
import json
import os
import re
import shutil
import site
import sys

EXPECTED_PYTHON = (3, 11)
EXPECTED_TORCH = "2.2.2"
EXPECTED_TORCH_CUDA = "12.1"
# pip distribution name (lower case) -> import name, where they differ
IMPORT_NAMES = {"pyyaml": "yaml", "tensorboardx": "tensorboardX"}
REQ_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*([<>=!~][^#]*)?(?:#(.*))?$")


def _specifier_tools():
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
    except ImportError:  # pip always vendors packaging
        from pip._vendor.packaging.specifiers import SpecifierSet
        from pip._vendor.packaging.version import Version
    return SpecifierSet, Version


def parse_requirements(path):
    """Return [{dist, import_name, spec, install, optional}] from a requirements file."""
    reqs = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = REQ_RE.match(line)
            if not m:
                raise ValueError("cannot parse requirement line: {!r}".format(line))
            dist, spec, comment = m.group(1), (m.group(2) or "").strip(), (m.group(3) or "")
            install = re.search(r"install:\s*(\S+)", comment)
            reqs.append({
                "dist": dist,
                "import_name": IMPORT_NAMES.get(dist.lower(), dist),
                "spec": spec,
                "install": install.group(1) if install else dist + spec,
                "optional": "optional" in comment.lower(),
            })
    return reqs


def version_ok(version, spec):
    if not spec:
        return True
    SpecifierSet, Version = _specifier_tools()
    try:
        return Version(version) in SpecifierSet(spec)
    except Exception:
        return False


def dist_version(dist, module=None):
    try:
        from importlib import metadata

        return metadata.version(dist)
    except Exception:
        return getattr(module, "__version__", None)


def under(path, prefix):
    if not path:
        return False
    path, prefix = os.path.realpath(path), os.path.realpath(prefix)
    return path == prefix or path.startswith(prefix.rstrip(os.sep) + os.sep)


def git_head(src):
    """Commit checked out in a git working tree, without running git."""
    gitdir = os.path.join(src, ".git")
    if os.path.isfile(gitdir):  # worktree / submodule: "gitdir: <path>"
        with open(gitdir, encoding="utf-8") as fh:
            gitdir = os.path.join(src, fh.read().split(":", 1)[1].strip())
    head_file = os.path.join(gitdir, "HEAD")
    if not os.path.isfile(head_file):
        return None
    with open(head_file, encoding="utf-8") as fh:
        head = fh.read().strip()
    if not head.startswith("ref:"):
        return head
    ref = head.split(":", 1)[1].strip()
    ref_file = os.path.join(gitdir, ref)
    if os.path.isfile(ref_file):
        with open(ref_file, encoding="utf-8") as fh:
            return fh.read().strip()
    packed = os.path.join(gitdir, "packed-refs")
    if os.path.isfile(packed):
        with open(packed, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    return None


def unicore_direct_url():
    try:
        from importlib import metadata

        text = metadata.distribution("unicore").read_text("direct_url.json")
        return json.loads(text) if text else None
    except Exception:
        return None


class Report:
    def __init__(self):
        self.items, self.fix, self.blockers, self.failures = [], [], 0, 0

    def add(self, status, msg):
        self.items.append((status, msg))
        print("{:5s} {}".format(status, msg))
        if status == "FAIL":
            self.failures += 1
        elif status == "BLOCK":
            self.blockers += 1


def check(ns):
    r = Report()
    prefix = os.path.realpath(ns.env_prefix)

    # --- interpreter -------------------------------------------------------
    print("python      :", sys.executable)
    print("sys.prefix  :", sys.prefix)
    if under(sys.executable, prefix) and os.path.realpath(sys.prefix) == prefix:
        r.add("PASS", "interpreter belongs to {}".format(prefix))
    else:
        r.add("BLOCK", "interpreter {} is not the heart_gpu python under {}".format(sys.executable, prefix))
    if sys.version_info[:2] == EXPECTED_PYTHON:
        r.add("PASS", "Python {}".format(sys.version.split()[0]))
    else:
        r.add("BLOCK", "Python {} (heart_gpu must be {}.{})".format(sys.version.split()[0], *EXPECTED_PYTHON))
    if not site.ENABLE_USER_SITE:
        r.add("PASS", "user site-packages disabled (PYTHONNOUSERSITE=1)")
    else:
        r.add("FAIL", "user site-packages enabled: ~/.local packages can shadow heart_gpu; export PYTHONNOUSERSITE=1")
    leaks = [p for p in sys.path if p and os.path.realpath(p).startswith(os.path.realpath(os.path.expanduser("~/.local")))]
    if leaks:
        r.add("FAIL", "~/.local on sys.path: {}".format(leaks))

    # --- torch: checked, never repaired ------------------------------------
    try:
        import torch

        tv, tc = torch.__version__, torch.version.cuda
        where = torch.__file__
        ok = tv.split("+")[0] == EXPECTED_TORCH and tc == EXPECTED_TORCH_CUDA and under(where, prefix)
        r.add("PASS" if ok else "BLOCK",
              "torch {} (CUDA build {}) from {}{}".format(
                  tv, tc, where, "" if ok else "  -- expected {}+cu121 inside heart_gpu; "
                  "--fix never reinstalls torch".format(EXPECTED_TORCH)))
    except Exception as exc:
        r.add("BLOCK", "import torch failed: {!r}; --fix never installs torch".format(exc))

    # --- requirements-heart.txt -------------------------------------------
    for req in parse_requirements(ns.requirements):
        name, dist = req["import_name"], req["dist"]
        bad = "WARN" if req["optional"] else "FAIL"
        try:
            mod = importlib.import_module(name)
        except Exception as exc:
            r.add(bad, "import {:13s} missing ({}: {})".format(name, type(exc).__name__, exc))
            if not req["optional"] or ns.with_optional:
                r.fix.append(req["install"])
            continue
        where = getattr(mod, "__file__", None) or ""
        ver = dist_version(dist, mod) or "?"
        problems = []
        if not under(where, prefix):
            problems.append("imported from outside heart_gpu: " + where)
        if not version_ok(ver, req["spec"]):
            problems.append("version {} does not satisfy '{}'".format(ver, req["spec"]))
        if problems:
            r.add(bad, "import {:13s} {:12s} {}".format(name, ver, "; ".join(problems)))
            if not req["optional"] or ns.with_optional:
                r.fix.append(req["install"])
        else:
            r.add("PASS", "import {:13s} {:12s} {}".format(name, ver, where))

    # --- Uni-Core ------------------------------------------------------------
    src = os.path.realpath(ns.unicore_src)
    head = git_head(src) if os.path.isdir(src) else None
    if not os.path.isfile(os.path.join(src, "setup.py")):
        r.add("BLOCK", "Uni-Core source checkout not found: {} (expected the existing third_party checkout)".format(src))
    elif head != ns.unicore_commit:
        r.add("BLOCK", "Uni-Core source {} is at {} (pinned {}); not changing it -- "
              "inspect with: git -C {} log -1".format(src, head, ns.unicore_commit, src))
    else:
        r.add("PASS", "Uni-Core source {} @ {}".format(src, head))
    need_unicore = False
    try:
        import unicore

        where = unicore.__file__
        if under(where, prefix):
            r.add("PASS", "import unicore {} from {}".format(dist_version("unicore", unicore), where))
        else:
            r.add("FAIL", "unicore imported from outside heart_gpu: {}".format(where))
            need_unicore = True
        du = unicore_direct_url()
        url = (du or {}).get("url", "")
        if url.startswith("file://") and os.path.realpath(url[len("file://"):]) == src:
            editable = (du.get("dir_info") or {}).get("editable", False)
            r.add("PASS", "unicore installed from {}{}".format(src, " (editable)" if editable else ""))
        else:
            r.add("WARN", "unicore install origin is {!r}, not {}; --fix reinstalls it from the pinned source".format(
                url or "unknown", src))
            need_unicore = True
    except Exception as exc:
        r.add("FAIL", "import unicore failed: {!r}".format(exc))
        need_unicore = True
    exe = shutil.which("unicore-train")
    if exe and under(exe, prefix):
        r.add("PASS", "unicore-train -> {}".format(exe))
    else:
        r.add("FAIL", "unicore-train {} (must resolve inside {}/bin)".format(exe or "not on PATH", prefix))
        need_unicore = True
    if need_unicore:
        r.fix.append("UNICORE")

    if ns.emit_fix:
        with open(ns.emit_fix, "w", encoding="utf-8") as fh:
            fh.write("".join(item + "\n" for item in r.fix))
    if ns.json:
        with open(ns.json, "w", encoding="utf-8") as fh:
            json.dump({"python": sys.executable, "items": r.items, "fix": r.fix}, fh, indent=1)
    print("repairs available to --fix:", " ".join(r.fix) if r.fix else "none")
    if r.blockers:
        return 2
    return 1 if r.failures else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env-prefix", required=True)
    ap.add_argument("--requirements", required=True)
    ap.add_argument("--unicore-src", required=True)
    ap.add_argument("--unicore-commit", required=True)
    ap.add_argument("--with-optional", action="store_true", help="also list optional packages for repair")
    ap.add_argument("--emit-fix", help="write the repair list (one item per line) to this file")
    ap.add_argument("--json", help="write the report as JSON")
    return check(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
