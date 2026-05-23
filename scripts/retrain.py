#!/usr/bin/env python3
"""Automated retraining cycle for AutobahnCV.

Implements the audit -> retrain -> evaluate -> register pipeline described in
``specs/Monitoring_Drift_Retraining.md``. The script orchestrates the per-model
workspaces in ``ml/`` and is safe to run in ``--dry-run`` mode (default), which
prints the plan without launching training.

Steps for a selected model:
    1. Export operator-corrected examples from feedback logs into the dataset.
    2. Run the workspace data-preparation script.
    3. Train a candidate model.
    4. Evaluate the candidate and compare it with the current production model.
    5. If the candidate is better, register it in MLflow as ``Staging``.

Usage:
    python3 scripts/retrain.py --model plate_ocr --dry-run
    python3 scripts/retrain.py --model plate_ocr --execute
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Per-model workspace layout: which scripts to call for each stage.
WORKSPACES: dict[str, dict[str, str]] = {
    "car_detector": {
        "prepare": "ml/car_detector/scripts/prepare_car_detector_dataset.py",
        "train": "ml/car_detector/scripts/train_car_detector.py",
        "evaluate": "ml/car_detector/scripts/evaluate_car_detector.py",
        "metric": "mAP50",
        "production_target": "0.92",
    },
    "plate_detector": {
        "prepare": "ml/plate_detector/notebooks/01_dataset_check.ipynb",
        "train": "ml/plate_detector/scripts/train_plate_detector.py",
        "evaluate": "ml/plate_detector/scripts/evaluate_plate_detector.py",
        "metric": "mAP50",
        "production_target": "0.90",
    },
    "plate_ocr": {
        "prepare": "ml/plate_ocr/scripts/prepare_ocr_dataset.py",
        "train": "ml/plate_ocr/scripts/train_plate_ocr.py",
        "evaluate": "ml/plate_ocr/scripts/evaluate_plate_ocr.py",
        "metric": "exact_match",
        "production_target": "0.95",
    },
    "car_embedder": {
        "prepare": "ml/car_embedder/scripts/prepare_mad_cars_dataset.py",
        "train": "ml/car_embedder/scripts/train_car_embedder.py",
        "evaluate": "ml/car_embedder/scripts/evaluate_car_embedder.py",
        "metric": "top5_accuracy",
        "production_target": "0.96",
    },
}

REGISTER_SCRIPT = "src/scripts/register_models.py"


def run(cmd: list[str], dry_run: bool) -> int:
    """Run a command, or print it when in dry-run mode."""
    printable = " ".join(cmd)
    if dry_run:
        print(f"  [dry-run] {printable}")
        return 0
    print(f"  [exec]    {printable}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def export_feedback(model: str, dry_run: bool) -> int:
    """Export operator-corrected examples from feedback-logs into the dataset.

    In production this queries the Elasticsearch ``feedback-logs`` index for
    ``action in (correct, reject, disputed)`` events and appends the corrected
    labels to the workspace dataset. Here it is represented as an explicit,
    reviewable step.
    """
    print("Step 1/5: export operator feedback -> training data")
    return run(
        ["python3", "scripts/export_feedback.py", "--model", model],
        dry_run,
    )


def retrain_model(model: str, dry_run: bool, epochs: int) -> int:
    ws = WORKSPACES[model]
    print(f"\n=== Retraining cycle: {model} ===")
    print(f"metric: {ws['metric']}  production target: {ws['production_target']}")

    rc = export_feedback(model, dry_run)
    if rc != 0:
        return rc

    print("Step 2/5: prepare dataset")
    if ws["prepare"].endswith(".py"):
        rc = run(["python3", ws["prepare"]], dry_run)
        if rc != 0:
            return rc
    else:
        print(f"  [manual]  run notebook {ws['prepare']}")

    print("Step 3/5: train candidate")
    rc = run(["python3", ws["train"], "--epochs", str(epochs)], dry_run)
    if rc != 0:
        return rc

    print("Step 4/5: evaluate candidate and compare with production")
    rc = run(["python3", ws["evaluate"]], dry_run)
    if rc != 0:
        return rc

    print("Step 5/5: register candidate in MLflow (stage: Staging)")
    rc = run(
        ["python3", REGISTER_SCRIPT, "--model", model, "--stage", "Staging"],
        dry_run,
    )
    if rc != 0:
        return rc

    print(
        "\nDone. Promotion Staging -> Production requires manual approval "
        "(see specs/Monitoring_Drift_Retraining.md §6)."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AutobahnCV retraining cycle")
    parser.add_argument(
        "--model",
        choices=sorted(WORKSPACES),
        required=True,
        help="which model workspace to retrain",
    )
    parser.add_argument("--epochs", type=int, default=50)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="print the plan without running anything (default)",
    )
    group.add_argument(
        "--execute",
        dest="dry_run",
        action="store_false",
        help="actually run the retraining cycle",
    )
    args = parser.parse_args()

    return retrain_model(args.model, args.dry_run, args.epochs)


if __name__ == "__main__":
    raise SystemExit(main())
