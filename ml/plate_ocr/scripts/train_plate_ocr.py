from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from time import monotonic

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ocr_common import (
    BLANK_INDEX,
    PlateOCRDataset,
    PlateOCRModel,
    collate_batch,
    decode_logits,
    repo_root,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    work_dir = root / "ml" / "plate_ocr"
    parser = argparse.ArgumentParser(description="Train plate OCR model.")
    parser.add_argument("--manifest-dir", type=Path, default=work_dir / "data" / "manifests")
    parser.add_argument("--runs-dir", type=Path, default=work_dir / "runs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-minutes", type=float, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--artifact-path", type=Path, default=root / "src" / "models" / "plate_ocr.pt")
    parser.add_argument("--publish", action="store_true")
    return parser


def accuracy(model: PlateOCRModel, loader: DataLoader, device: torch.device, deadline: float) -> float | None:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch in loader:
            if monotonic() >= deadline:
                return None
            images = batch.images.to(device)
            predictions = decode_logits(model(images))
            correct += sum(pred == label for pred, label in zip(predictions, batch.labels, strict=False))
            total += len(batch.labels)
    if monotonic() >= deadline:
        return None
    return correct / max(total, 1)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 0 < args.max_minutes <= 30:
        parser.error("--max-minutes must be positive and at most 30")
    for name in ("epochs", "batch_size", "lr"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    deadline = monotonic() + args.max_minutes * 60
    for split in ("train", "val"):
        if not (args.manifest_dir / f"{split}.csv").exists():
            parser.error(
                f"Missing manifest: {args.manifest_dir / f'{split}.csv'}. "
                "Run prepare_ocr_dataset.py (make prepare in ml/plate_ocr) first."
            )

    device = resolve_device(args.device)
    if str(device) != args.device:
        print(f"Resolved device '{args.device}' to '{device}'")
    train_set = PlateOCRDataset(args.manifest_dir / "train.csv")
    val_set = PlateOCRDataset(args.manifest_dir / "val.csv")
    if not len(train_set) or not len(val_set):
        parser.error("Training and validation datasets must be non-empty")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_batch,
    )

    model = PlateOCRModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    run_dir = args.runs_dir / "plate_ocr"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    best_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        if monotonic() >= deadline:
            break
        model.train()
        losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            if monotonic() >= deadline:
                break
            images = batch.images.to(device)
            targets = batch.targets.to(device)
            target_lengths = batch.target_lengths.to(device)
            logits = model(images)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long, device=device)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        if monotonic() >= deadline:
            break
        val_acc = accuracy(model, val_loader, device, deadline)
        if val_acc is None:
            break
        mean_loss = sum(losses) / max(len(losses), 1)
        print(f"epoch={epoch} loss={mean_loss:.4f} val_exact={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "val_exact": val_acc}, best_path)

    if monotonic() >= deadline:
        print("Time budget exhausted.")
    if best_acc < 0:
        print("No completed validation checkpoint in this run; nothing to publish.")
        return
    print(f"Best checkpoint: {best_path} val_exact={best_acc:.4f}")
    if args.publish:
        args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, args.artifact_path)
        print(f"Published OCR model: {args.artifact_path}")


if __name__ == "__main__":
    main()
