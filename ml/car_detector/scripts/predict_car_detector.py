from __future__ import annotations

import argparse
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run car detector visual checks.")
    parser.add_argument("--model", type=Path, default=root / "src" / "models" / "car_detector.pt")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", type=Path, default=root / "ml" / "car_detector" / "predictions")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from ultralytics import YOLO

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(args.output_dir),
        name="visual_check",
        exist_ok=True,
        save=True,
        save_txt=True,
    )


if __name__ == "__main__":
    main()
