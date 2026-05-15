from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    work_dir = root / "ml" / "car_detector"
    parser = argparse.ArgumentParser(description="Train a YOLO vehicle detector.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml.")
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--name", default="car_detector_yolo")
    parser.add_argument("--runs-dir", type=Path, default=work_dir / "runs")
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=root / "src" / "models" / "car_detector.pt",
    )
    parser.add_argument("--publish", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        from device import resolve_device
    except ModuleNotFoundError:
        from scripts.device import resolve_device
    from ultralytics import YOLO

    device = resolve_device(args.device)
    if device != args.device:
        print(f"Resolved device '{args.device}' to '{device}'")

    model = YOLO(args.base_model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        project=str(args.runs_dir),
        name=args.name,
        exist_ok=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Best checkpoint: {best}")
    if args.publish:
        args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, args.artifact_path)
        print(f"Published detector: {args.artifact_path}")


if __name__ == "__main__":
    main()
