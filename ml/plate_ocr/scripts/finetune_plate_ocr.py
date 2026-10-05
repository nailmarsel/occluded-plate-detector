"""Fine-tune the published plate OCR model instead of training from scratch.

Starts from ``src/models/plate_ocr.pt`` (CNN + BiLSTM + CTC used by the app),
continues training with a 10x lower learning rate, cosine schedule, gradient
clipping and camera-like augmentations (tilt, blur, low contrast, downscale,
brightness). A checkpoint is saved only when it beats the starting model on the
validation split, so the result is never worse than the baseline on val.

The published model is NOT replaced unless ``--publish`` is passed; then the
previous file is backed up to ``src/models/rollback/`` first.

Prepare manifests once with ``make prepare`` in ml/plate_ocr.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

import torch
from ocr_common import (
    BLANK_INDEX,
    PlateOCRDataset,
    PlateOCRModel,
    build_ocr_transform,
    collate_batch,
    repo_root,
    resolve_device,
)
from ocr_metrics import evaluate
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class AugmentedOCRDataset(Dataset):
    """Train split with random camera-like augmentations (label unchanged)."""

    def __init__(self, base: PlateOCRDataset, strength: float = 1.0):
        self.samples = base.samples
        self.base_transform = build_ocr_transform()
        s = strength
        self.augment = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            degrees=4 * s,
                            translate=(0.03 * s, 0.08 * s),
                            scale=(1 - 0.08 * s, 1 + 0.05 * s),
                            shear=4 * s,
                            fill=128,
                        )
                    ],
                    p=0.7,
                ),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.4 * s, contrast=0.4 * s)], p=0.7),
            ]
        )
        self.strength = s

    def __len__(self) -> int:
        return len(self.samples)

    def _random_quality(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.4 * self.strength:
            w, h = image.size
            k = random.uniform(0.45, 0.85)
            image = image.resize((max(8, int(w * k)), max(4, int(h * k))), Image.BILINEAR).resize(
                (w, h), Image.BILINEAR
            )
        if random.random() < 0.3 * self.strength:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.3)))
        return image

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = self._random_quality(self.augment(image))
        return self.base_transform(image), label


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    work = root / "ml" / "plate_ocr"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest-dir", type=Path, default=work / "data" / "manifests")
    parser.add_argument("--base-model", type=Path, default=root / "src" / "models" / "plate_ocr.pt")
    parser.add_argument("--runs-dir", type=Path, default=work / "runs")
    parser.add_argument("--name", default="finetune_plate_ocr")
    parser.add_argument("--artifact-path", type=Path, default=root / "src" / "models" / "plate_ocr.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-minutes", type=float, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4, help="10x lower than from-scratch training (1e-3).")
    parser.add_argument("--aug-strength", type=float, default=1.0, help="0 disables augmentations.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Replace src/models/plate_ocr.pt (old one goes to src/models/rollback/).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 0 < args.max_minutes <= 30:
        parser.error("--max-minutes must be positive and at most 30")
    for name in ("epochs", "batch_size", "lr"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.aug_strength < 0 or args.workers < 0:
        parser.error("--aug-strength and --workers must be non-negative")
    if not args.base_model.exists():
        parser.error(f"Base model not found: {args.base_model}")
    for split in ("train", "val"):
        if not (args.manifest_dir / f"{split}.csv").exists():
            parser.error(
                f"Missing manifest {args.manifest_dir / f'{split}.csv'}. Run `make prepare` in ml/plate_ocr first."
            )

    deadline = monotonic() + args.max_minutes * 60
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    train_base = PlateOCRDataset(args.manifest_dir / "train.csv")
    val_set = PlateOCRDataset(args.manifest_dir / "val.csv")
    if not len(train_base) or not len(val_set):
        parser.error("Training and validation datasets must be non-empty")
    train_set = AugmentedOCRDataset(train_base, args.aug_strength) if args.aug_strength > 0 else train_base
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_batch,
        drop_last=len(train_set) > args.batch_size,
    )

    model = PlateOCRModel().to(device)
    checkpoint = torch.load(args.base_model, map_location=device)
    model.load_state_dict(checkpoint["model"])

    baseline = evaluate(model, val_set, device, args.batch_size, args.workers)["exact_match"]
    print(f"Baseline val_exact={baseline:.4f} ({args.base_model})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1, anneal_strategy="cos"
    )
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    run_dir = args.runs_dir / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    if best_path.exists():
        best_path.unlink()  # never mix results of different runs
    best_acc, best_epoch, history = baseline, 0, []
    started = monotonic()

    for epoch in range(1, args.epochs + 1):
        if monotonic() >= deadline:
            break
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"finetune {epoch}/{args.epochs}"):
            if monotonic() >= deadline:
                break
            images = batch.images.to(device)
            logits = model(images)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long, device=device)
            loss = criterion(
                log_probs, batch.targets.to(device), input_lengths, batch.target_lengths.to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if scheduler.last_epoch + 1 < total_steps:
                scheduler.step()
            losses.append(float(loss.item()))

        if monotonic() >= deadline:
            print("Time budget exhausted before validation of this epoch.")
            break
        val_acc = evaluate(model, val_set, device, args.batch_size, args.workers)["exact_match"]
        mean_loss = sum(losses) / max(len(losses), 1)
        history.append({"epoch": epoch, "loss": round(mean_loss, 4), "val_exact": round(val_acc, 4)})
        print(f"epoch={epoch} loss={mean_loss:.4f} val_exact={val_acc:.4f} (best={best_acc:.4f})")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_exact": val_acc,
                    "base_model": str(args.base_model),
                    "base_val_exact": baseline,
                    "finetuned": True,
                },
                best_path,
            )

    summary = {
        "model": "plate_ocr",
        "base_model": str(args.base_model),
        "finetuned_model": str(best_path) if best_epoch else None,
        "base_val_exact": round(baseline, 4),
        "best_val_exact": round(best_acc, 4),
        "best_epoch": best_epoch,
        "epochs_requested": args.epochs,
        "lr": args.lr,
        "aug_strength": args.aug_strength,
        "batch_size": args.batch_size,
        "device": str(device),
        "train_minutes": round((monotonic() - started) / 60, 1),
        "history": history,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (run_dir / "finetune_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not best_epoch:
        print("Fine-tuned model did not beat the baseline on val; nothing saved or published.")
        return
    print(f"Best checkpoint: {best_path} val_exact={best_acc:.4f} (baseline {baseline:.4f})")
    if args.publish:
        if args.artifact_path.exists():
            rollback = args.artifact_path.parent / "rollback" / args.artifact_path.name
            rollback.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.artifact_path, rollback)
            print(f"Previous model backed up: {rollback}")
        shutil.copy2(best_path, args.artifact_path)
        print(f"Published fine-tuned OCR model: {args.artifact_path}")


if __name__ == "__main__":
    main()
