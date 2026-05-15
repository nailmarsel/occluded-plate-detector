from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from embedder_common import ResNetEmbedder, build_transforms, repo_root, resolve_device


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Evaluate car embedder classifier head.")
    parser.add_argument("--model", type=Path, default=root / "src" / "models" / "car_embedder.pt")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    if str(device) != args.device:
        print(f"Resolved device '{args.device}' to '{device}'")
    checkpoint = torch.load(args.model, map_location=device)
    dataset = ImageFolder(args.data_dir, transform=build_transforms(train=False))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = ResNetEmbedder(num_classes=len(checkpoint["classes"]), backbone=checkpoint["backbone"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            logits, _ = model(images.to(device))
            correct += int((logits.argmax(dim=1).cpu() == labels).sum().item())
            total += labels.numel()
    print(f"accuracy={correct / max(total, 1):.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
