from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from embedder_common import ResNetEmbedder, build_transforms, repo_root, resolve_device


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    work_dir = root / "ml" / "car_embedder"
    parser = argparse.ArgumentParser(description="Train car image embedder.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet101"])
    parser.add_argument("--runs-dir", type=Path, default=work_dir / "runs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--artifact-path", type=Path, default=root / "src" / "models" / "car_embedder.pt")
    parser.add_argument("--publish", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.data_dir.exists():
        raise RuntimeError(
            f"Data directory does not exist: {args.data_dir}. "
            "Run `make prepare` or pass `--data-dir /path/to/imagefolder`."
        )

    random.seed(42)
    torch.manual_seed(42)
    device = resolve_device(args.device)
    if str(device) != args.device:
        print(f"Resolved device '{args.device}' to '{device}'")

    full_dataset = ImageFolder(args.data_dir, transform=build_transforms(train=True))
    if len(full_dataset.classes) < 2:
        raise RuntimeError("Embedder training needs at least two identity folders.")

    val_size = max(1, int(len(full_dataset) * 0.1))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    val_set.dataset.transform = build_transforms(train=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = ResNetEmbedder(num_classes=len(full_dataset.classes), backbone=args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    run_dir = args.runs_dir / "car_embedder"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for images, labels in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            logits, _ = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                logits, _ = model(images.to(device))
                predictions = logits.argmax(dim=1).cpu()
                correct += int((predictions == labels).sum().item())
                total += labels.numel()
        val_acc = correct / max(total, 1)
        print(f"epoch={epoch} loss={sum(losses) / max(len(losses), 1):.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": full_dataset.classes,
                    "backbone": args.backbone,
                    "embedding_dim": 2048,
                },
                best_path,
            )

    print(f"Best checkpoint: {best_path} val_acc={best_acc:.4f}")
    if args.publish:
        args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, args.artifact_path)
        print(f"Published embedder: {args.artifact_path}")


if __name__ == "__main__":
    main()
