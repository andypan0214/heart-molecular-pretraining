"""Stage A -- Uni-Core discovers the HEART/LUMI registrations through --user-dir."""
import os

from conftest import MODEL_ROOT


def test_registrations(unimol_pkg):
    from unicore.losses import CRITERION_REGISTRY
    from unicore.models import ARCH_MODEL_REGISTRY, MODEL_REGISTRY
    from unicore.tasks import TASK_REGISTRY

    assert "unimol_contrastive" in TASK_REGISTRY
    assert "unimol_contrastive" in MODEL_REGISTRY
    assert "unimol_contrastive_base" in ARCH_MODEL_REGISTRY
    assert "unimol_contrastive" in CRITERION_REGISTRY


def test_unimol_resolves_to_model_root(unimol_pkg):
    assert os.path.dirname(os.path.abspath(unimol_pkg.__file__)) == os.path.join(MODEL_ROOT, "unimol")
    # the package must not live inside pretrain/
    assert not os.path.exists(os.path.join(MODEL_ROOT, "pretrain", "unimol"))


def test_check_registry_tool(unimol_pkg, monkeypatch):
    import check_registry

    monkeypatch.chdir(MODEL_ROOT)
    assert check_registry.main(["--user-dir", "./unimol"]) == 0
