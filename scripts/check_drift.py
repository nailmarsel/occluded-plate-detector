#!/usr/bin/env python3
"""Drift detection for AutobahnCV.

Compares the distribution of inference features in a recent window against a
stored baseline using the Population Stability Index (PSI).

PSI interpretation:
    PSI < 0.10            -> no significant drift
    0.10 <= PSI < 0.25    -> moderate drift, schedule an audit
    PSI >= 0.25           -> strong drift, candidate for retraining

The baseline is stored in ``reports/data/drift_baseline.json`` and is refreshed
after every confirmed model release.

Usage:
    python3 scripts/check_drift.py --current reports/data/recent_window.json
    python3 scripts/check_drift.py --make-baseline reports/data/recent_window.json

Each window file is a JSON object mapping a feature name to a list of numeric
samples, e.g. ``{"plate_length": [8, 9, 8, ...], "car_confidence": [0.9, ...]}``.
In production these windows are exported from the Elasticsearch ``inference-logs``
index; the script itself is storage-agnostic and works on plain JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO_ROOT / "reports" / "data" / "drift_baseline.json"

MODERATE_PSI = 0.10
STRONG_PSI = 0.25
N_BINS = 10
EPS = 1e-6


def _bin_edges(values: list[float], n_bins: int = N_BINS) -> list[float]:
    """Quantile bin edges so each baseline bin holds a similar mass."""
    ordered = sorted(values)
    if not ordered:
        return [0.0, 1.0]
    edges = [ordered[0] - EPS]
    for i in range(1, n_bins):
        idx = int(i / n_bins * (len(ordered) - 1))
        edges.append(ordered[idx])
    edges.append(ordered[-1] + EPS)
    # de-duplicate while keeping order (constant features collapse to 1 bin)
    return sorted(set(edges))


def _hist(values: list[float], edges: list[float]) -> list[float]:
    counts = [0] * (len(edges) - 1)
    for v in values:
        for b in range(len(edges) - 1):
            if edges[b] <= v < edges[b + 1]:
                counts[b] += 1
                break
        else:
            counts[-1] += 1
    total = max(sum(counts), 1)
    return [c / total for c in counts]


def psi(baseline: list[float], current: list[float]) -> float:
    """Population Stability Index between two numeric samples."""
    edges = _bin_edges(baseline)
    base = _hist(baseline, edges)
    curr = _hist(current, edges)
    score = 0.0
    for b, c in zip(base, curr):
        b = max(b, EPS)
        c = max(c, EPS)
        score += (c - b) * (json_log(c) - json_log(b))
    return round(score, 4)


def json_log(x: float) -> float:
    import math

    return math.log(x)


def classify(score: float) -> str:
    if score >= STRONG_PSI:
        return "STRONG_DRIFT"
    if score >= MODERATE_PSI:
        return "MODERATE_DRIFT"
    return "STABLE"


def load(path: Path) -> dict[str, list[float]]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def make_baseline(window: Path, out: Path) -> None:
    data = load(window)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    try:
        shown = out.relative_to(REPO_ROOT)
    except ValueError:
        shown = out
    print(f"Baseline written to {shown}")


def run_check(current: Path, baseline: Path) -> int:
    if not baseline.exists():
        print(f"ERROR: baseline not found: {baseline}", file=sys.stderr)
        print("Create one with --make-baseline first.", file=sys.stderr)
        return 2

    base = load(baseline)
    curr = load(current)
    worst = "STABLE"
    print(f"{'feature':<24}{'PSI':>10}  status")
    print("-" * 48)
    for feature in sorted(base):
        if feature not in curr:
            continue
        score = psi(base[feature], curr[feature])
        status = classify(score)
        if status == "STRONG_DRIFT" or (status == "MODERATE_DRIFT" and worst == "STABLE"):
            worst = status
        print(f"{feature:<24}{score:>10.4f}  {status}")

    print("-" * 48)
    print(f"overall: {worst}")
    # exit code: 0 stable, 1 moderate (audit), 2 strong (retrain candidate)
    return {"STABLE": 0, "MODERATE_DRIFT": 1, "STRONG_DRIFT": 2}[worst]


def main() -> int:
    parser = argparse.ArgumentParser(description="AutobahnCV drift check (PSI)")
    parser.add_argument("--current", type=Path, help="recent-window JSON file")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--make-baseline",
        type=Path,
        metavar="WINDOW",
        help="store the given window JSON as the new baseline and exit",
    )
    args = parser.parse_args()

    if args.make_baseline:
        make_baseline(args.make_baseline, args.baseline)
        return 0

    if not args.current:
        parser.error("--current is required unless --make-baseline is used")

    return run_check(args.current, args.baseline)


if __name__ == "__main__":
    raise SystemExit(main())
