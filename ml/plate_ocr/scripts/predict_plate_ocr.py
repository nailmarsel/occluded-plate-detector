from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from ocr_common import (
    PlateOCRModel,
    build_ocr_transform,
    decode_logits,
    repo_root,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Predict plate text from cropped plate images.")
    parser.add_argument("--model", type=Path, default=root / "src" / "models" / "plate_ocr.pt")
    parser.add_argument("--source", required=True)
    parser.add_argument("--device", default="auto")
    return parser


def iter_images(source: Path):
    if source.is_file():
        yield source
    else:
        for path in sorted(source.rglob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                yield path


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    model = PlateOCRModel().to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    transform = build_ocr_transform()

    with torch.inference_mode():
        for image_path in iter_images(Path(args.source)):
            image = Image.open(image_path).convert("RGB")
            tensor = transform.transform(image).unsqueeze(0).to(device)
            prediction = decode_logits(model(tensor))[0]
            print(f"{image_path}: {prediction}")


if __name__ == "__main__":
    main()
