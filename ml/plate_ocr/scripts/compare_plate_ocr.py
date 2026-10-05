"""Compare the current and the fine-tuned plate OCR on the same test data.

Two test sets from the held-out ``test`` manifest (never used in training):
  * clean    — original plate crops;
  * degraded — the same crops with a fixed, seeded camera-like degradation
               (downscale, blur, low contrast, light change, tilt, noise).

Writes a JSON consumed by ``scripts/build_finetune_report.py``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from ocr_common import PlateOCRDataset, PlateOCRModel, repo_root, resolve_device
from ocr_metrics import DegradedOCRDataset, evaluate

METRICS = [  # (key, label, higher_is_better)
    ("exact_match", "Точность номера целиком (exact match)", True),
    ("char_accuracy", "Точность по символам (1 − CER)", True),
    ("cer", "CER (доля ошибок в символах)", False),
    ("ms_per_image", "Инференс, мс/изобр.", False),
]


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    work = root / "ml" / "plate_ocr"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, default=root / "src" / "models" / "plate_ocr.pt")
    parser.add_argument("--candidate", type=Path, default=work / "runs" / "finetune_plate_ocr" / "best.pt")
    parser.add_argument("--manifest", type=Path, default=work / "data" / "manifests" / "test.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N test samples (0 = all).")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output", type=Path, default=root / "reports" / "models" / "finetune" / "plate_ocr.json"
    )
    return parser


def load(path: Path, device) -> PlateOCRModel:
    model = PlateOCRModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device)["model"])
    return model.eval()


def main() -> None:
    args = build_parser().parse_args()
    if not args.baseline.exists():
        raise SystemExit(f"Model not found: {args.baseline}")
    if not args.candidate.exists():
        raise SystemExit(
            f"Fine-tuned model not found: {args.candidate}. Either fine-tuning has not run, "
            "or it did not beat the baseline on val (see runs/finetune_plate_ocr/finetune_summary.json)."
        )
    if not args.manifest.exists():
        raise SystemExit(f"Test manifest not found: {args.manifest}. Run `make prepare` in ml/plate_ocr.")

    device = resolve_device(args.device)
    clean = PlateOCRDataset(args.manifest)
    if args.limit:
        clean.samples = clean.samples[: args.limit]
    datasets = [
        ("clean", "Тест: исходные кропы номеров", clean),
        (
            "degraded",
            "Тест: ухудшенные кропы (размытие, шум, контраст)",
            DegradedOCRDataset(clean, args.seed),
        ),
    ]
    baseline, candidate = load(args.baseline, device), load(args.candidate, device)

    rows = []
    for key, title, dataset in datasets:
        print(f"\n=== {title} ({len(dataset)} samples) ===")
        base = evaluate(baseline, dataset, device, args.batch_size, args.workers)
        cand = evaluate(candidate, dataset, device, args.batch_size, args.workers)
        for metric, label, higher in METRICS:
            rows.append(
                {
                    "dataset": key,
                    "dataset_title": title,
                    "n": base["n"],
                    "metric": metric,
                    "metric_title": label,
                    "higher_is_better": higher,
                    "baseline": base[metric],
                    "candidate": cand[metric],
                }
            )
            print(f"{label:42s} old={base[metric]:.4f} new={cand[metric]:.4f}")

    summary_path = args.candidate.parent / "finetune_summary.json"
    result = {
        "model": "plate_ocr",
        "title": "Распознавание номера (CRNN + CTC)",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "device": str(device),
        "finetune": json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
