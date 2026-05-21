import importlib.util
import json
from pathlib import Path


def _load_release_gate_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "verify_release_gates.py"
    spec = importlib.util.spec_from_file_location("verify_release_gates", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mvp_release_gates_are_backed_by_artifacts_and_metrics():
    verifier = _load_release_gate_module()
    assert verifier.verify() == []


def test_release_governance_artifacts_are_verified():
    verifier = _load_release_gate_module()
    evidence = verifier.load_evidence()
    metric_names = {
        metric["name"]
        for metric in evidence["mvp_metrics"]
    }

    assert "release_governance_rollback" in metric_names
    assert verifier.DEFAULT_REGISTRY.exists()
    assert verifier.DEFAULT_ROLLBACK_RUNBOOK.exists()


def test_release_gate_verifier_allows_gitignored_model_binaries_in_ci(tmp_path):
    verifier = _load_release_gate_module()
    evidence_path = tmp_path / "release_evidence.json"
    evidence = verifier.load_evidence()
    evidence["model_artifacts"] = [
        {
            "name": "missing_in_ci",
            "path": "src/models/missing_in_ci.pt",
            "sha256": "not-used-when-file-is-absent",
            "min_size_bytes": 1,
        }
    ]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    assert verifier.verify(evidence_path) == []
    assert verifier.verify(evidence_path, strict_artifacts=True) == [
        "Missing model artifact: src/models/missing_in_ci.pt"
    ]
