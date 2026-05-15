from __future__ import annotations

import argparse
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Evaluate a YOLO vehicle detector.")
    parser.add_argument("--model", type=Path, default=root / "src" / "models" / "car_detector.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
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

    model = YOLO(str(args.model))
    print(model.val(data=args.data, imgsz=args.imgsz, device=device, split=args.split))


if __name__ == "__main__":
    main()
