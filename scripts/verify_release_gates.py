from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = REPO_ROOT / "reports" / "models" / "release_evidence.json"
DEFAULT_REGISTRY = REPO_ROOT / "reports" / "models" / "model_registry.json"
DEFAULT_ROLLBACK_RUNBOOK = REPO_ROOT / "HW" / "ROLLBACK_RUNBOOK.md"
REQUIRED_REGISTRY_MODEL_NAMES = {
    "car_detector",
    "license_plate_detector",
    "plate_ocr",
    "car_embedder",
}
REQUIRED_RUNBOOK_MARKERS = [
    "Когда запускать rollback",
    "Как откатить модели",
    "Проверка после rollback",
    "Критерий успешного rollback",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evidence(path: Path = DEFAULT_EVIDENCE) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _verify_model_registry() -> list[str]:
    failures: list[str] = []
    if not DEFAULT_REGISTRY.exists():
        return [f"Missing model registry: {DEFAULT_REGISTRY.relative_to(REPO_ROOT)}"]

    with DEFAULT_REGISTRY.open("r", encoding="utf-8") as file:
        registry = json.load(file)

    if registry.get("promotion_strategy") != "manual_approval_required":
        failures.append("Model registry must require manual promotion approval")
    if registry.get("rollback_strategy") != "previous_stable_manifest":
        failures.append("Model registry must define previous_stable_manifest rollback")

    models = registry.get("required_models", [])
    model_names = {model.get("name") for model in models}
    missing_names = REQUIRED_REGISTRY_MODEL_NAMES - model_names
    if missing_names:
        failures.append(
            "Model registry missing required models: "
            + ", ".join(sorted(missing_names))
        )

    for model in models:
        for key in ("env_var", "current_path", "local_artifact", "rollback_path"):
            if not model.get(key):
                failures.append(
                    f"Model registry entry '{model.get('name')}' misses {key}"
                )

    if not registry.get("promotion_checks"):
        failures.append("Model registry must define promotion checks")
    if not registry.get("rollback_triggers"):
        failures.append("Model registry must define rollback triggers")

    return failures


def _verify_rollback_runbook() -> list[str]:
    if not DEFAULT_ROLLBACK_RUNBOOK.exists():
        return [
            "Missing rollback runbook: "
            f"{DEFAULT_ROLLBACK_RUNBOOK.relative_to(REPO_ROOT)}"
        ]

    content = DEFAULT_ROLLBACK_RUNBOOK.read_text(encoding="utf-8")
    return [
        f"Rollback runbook misses section: {marker}"
        for marker in REQUIRED_RUNBOOK_MARKERS
        if marker not in content
    ]


def verify(
    evidence_path: Path = DEFAULT_EVIDENCE,
    *,
    strict_artifacts: bool = False,
) -> list[str]:
    evidence = load_evidence(evidence_path)
    failures: list[str] = []

    for artifact in evidence.get("model_artifacts", []):
        artifact_path = REPO_ROOT / artifact["path"]
        if not artifact_path.exists():
            if strict_artifacts:
                failures.append(f"Missing model artifact: {artifact['path']}")
            continue

        min_size = int(artifact.get("min_size_bytes", 1))
        actual_size = artifact_path.stat().st_size
        if actual_size < min_size:
            failures.append(
                f"Model artifact too small: {artifact['path']} "
                f"({actual_size} < {min_size})"
            )

        expected_sha = artifact.get("sha256")
        if expected_sha and _sha256(artifact_path) != expected_sha:
            failures.append(f"SHA256 mismatch for {artifact['path']}")

    for metric in evidence.get("mvp_metrics", []):
        observed = float(metric["observed"])
        threshold = float(metric["threshold"])
        if observed < threshold or not metric.get("passed", False):
            failures.append(
                f"MVP metric failed: {metric['name']} "
                f"({observed} < {threshold})"
            )

    failures.extend(_verify_model_registry())
    failures.extend(_verify_rollback_runbook())

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify AutobahnCV MVP release gates.")
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
        help="Path to release evidence JSON.",
    )
    parser.add_argument(
        "--strict-artifacts",
        action="store_true",
        help=(
            "Require local model binaries to exist. CI normally leaves this off "
            "because src/models/*.pt files are intentionally gitignored."
        ),
    )
    args = parser.parse_args()

    failures = verify(args.evidence, strict_artifacts=args.strict_artifacts)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    print("PASS: all MVP release gates verified")


if __name__ == "__main__":
    main()
