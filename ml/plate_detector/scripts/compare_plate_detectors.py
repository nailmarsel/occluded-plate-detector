"""Compare the current and the fine-tuned plate detector on the same test data.

Two test sets, both from the held-out ``test`` split (never used in training):
  * clean    — original images;
  * occluded — the same images with a part of every plate covered
               (deterministic, seeded), the project's target scenario.

Writes a JSON consumed by ``scripts/build_finetune_report.py``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from device import resolve_device
    from occlusion import IMAGE_EXTENSIONS, build_occluded_split
except ModuleNotFoundError:
    from scripts.device import resolve_device
    from scripts.occlusion import IMAGE_EXTENSIONS, build_occluded_split


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    work = root / "ml" / "plate_detector"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--baseline", type=Path, default=root / "src" / "models" / "license_plate_detector.pt"
    )
    parser.add_argument(
        "--candidate", type=Path, default=work / "runs" / "finetune_plate_yolo" / "weights" / "best.pt"
    )
    parser.add_argument("--data-dir", type=Path, default=work / "data" / "yolo" / "russian_plate")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=work / "data" / "yolo" / "compare",
        help="Where test lists and occluded test copies are written.",
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N test images (0 = all).")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output", type=Path, default=root / "reports" / "models" / "finetune" / "plate_detector.json"
    )
    return parser


def test_images(images_dir: Path, limit: int) -> list[Path]:
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    return images[:limit] if limit else images


def write_yaml(eval_dir: Path, name: str, images: list[Path]) -> Path:
    eval_dir.mkdir(parents=True, exist_ok=True)
    listing = eval_dir / f"{name}.txt"
    listing.write_text("\n".join(str(p.absolute()) for p in images) + "\n", encoding="utf-8")
    yaml_path = eval_dir / f"{name}.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {eval_dir.absolute()}",
                f"train: {listing.absolute()}",
                f"val: {listing.absolute()}",
                f"test: {listing.absolute()}",
                "names:",
                "  0: license_plate",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def run_val(model_path: Path, data_yaml: Path, args, device: str) -> dict:
    from ultralytics import YOLO

    metrics = YOLO(str(model_path)).val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,
        plots=False,
        verbose=False,
    )
    box = metrics.box
    p, r = float(box.mp), float(box.mr)
    return {
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "ms_per_image": float(metrics.speed.get("inference", 0.0)),
    }


METRICS = [  # (key, label, higher_is_better)
    ("mAP50", "mAP@0.5", True),
    ("mAP50-95", "mAP@0.5:0.95", True),
    ("precision", "Precision", True),
    ("recall", "Recall", True),
    ("f1", "F1", True),
    ("ms_per_image", "Инференс, мс/изобр.", False),
]


def main() -> None:
    args = build_parser().parse_args()
    for path in (args.baseline, args.candidate):
        if not path.exists():
            raise SystemExit(f"Model not found: {path}")
    test_dir = args.data_dir / "images" / "test"
    if not test_dir.is_dir():
        raise SystemExit(f"Test split not found: {test_dir}. Run `make prepare` in ml/plate_detector.")

    device = resolve_device(args.device)
    clean = test_images(test_dir, args.limit)
    occluded_root = args.eval_dir / "occluded"
    build_occluded_split(
        test_dir,
        args.data_dir / "labels" / "test",
        occluded_root,
        "test",
        fraction=1.0,
        seed=args.seed,
        limit=args.limit or None,
    )
    occluded = test_images(occluded_root / "images" / "test", 0)
    if args.limit:
        keep = {p.stem for p in clean}
        occluded = [p for p in occluded if p.stem.removesuffix("_occ") in keep]

    datasets = [
        ("clean", "Тест: исходные изображения", write_yaml(args.eval_dir, "test_clean", clean), len(clean)),
        (
            "occluded",
            "Тест: номера с перекрытием",
            write_yaml(args.eval_dir, "test_occluded", occluded),
            len(occluded),
        ),
    ]

    rows = []
    for key, title, yaml_path, n in datasets:
        print(f"\n=== {title} ({n} images) ===")
        base = run_val(args.baseline, yaml_path, args, device)
        cand = run_val(args.candidate, yaml_path, args, device)
        for metric, label, higher in METRICS:
            rows.append(
                {
                    "dataset": key,
                    "dataset_title": title,
                    "n": n,
                    "metric": metric,
                    "metric_title": label,
                    "higher_is_better": higher,
                    "baseline": base[metric],
                    "candidate": cand[metric],
                }
            )
            print(f"{label:24s} old={base[metric]:.4f} new={cand[metric]:.4f}")

    summary_path = args.candidate.parent.parent / "finetune_summary.json"
    result = {
        "model": "license_plate_detector",
        "title": "Детектор номеров (YOLO)",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "imgsz": args.imgsz,
        "device": device,
        "finetune": json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
