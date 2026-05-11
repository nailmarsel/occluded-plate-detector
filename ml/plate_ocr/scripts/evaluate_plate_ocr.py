from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ocr_common import PlateOCRDataset, PlateOCRModel, collate_batch, decode_logits, repo_root, resolve_device


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    work_dir = root / "ml" / "plate_ocr"
    parser = argparse.ArgumentParser(description="Evaluate plate OCR model.")
    parser.add_argument("--model", type=Path, default=root / "src" / "models" / "plate_ocr.pt")
    parser.add_argument("--manifest", type=Path, default=work_dir / "data" / "manifests" / "test.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    dataset = PlateOCRDataset(args.manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_batch)
    model = PlateOCRModel().to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch in loader:
            predictions = decode_logits(model(batch.images.to(device)))
            for pred, label in zip(predictions, batch.labels, strict=False):
                correct += pred == label
                total += 1
    print(f"exact_match={correct / max(total, 1):.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
