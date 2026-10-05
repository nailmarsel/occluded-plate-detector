"""Fine-tune the published plate detector instead of training from scratch.

Starts from ``src/models/license_plate_detector.pt`` (the model the app uses),
continues training on the prepared Russian plate dataset with a lower learning
rate and adds synthetic occlusions to part of the training images, so the
detector learns to find partially hidden plates.

The published model is NOT replaced unless ``--publish`` is passed; then the
previous file is backed up to ``src/models/rollback/`` first.

Prepare the dataset once with ``make prepare`` (or
``python scripts/train_plate_detector.py --prepare-only``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from device import resolve_training_device
    from occlusion import build_occluded_split
except ModuleNotFoundError:  # imported as scripts.*
    from scripts.device import resolve_training_device
    from scripts.occlusion import build_occluded_split


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    work = root / "ml" / "plate_detector"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=work / "data" / "yolo" / "russian_plate",
        help="Prepared YOLO dataset (images/{train,val,test}, labels/...).",
    )
    parser.add_argument(
        "--occluded-dir",
        type=Path,
        default=work / "data" / "yolo" / "russian_plate_occluded",
        help="Where synthetic occluded copies are written.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=root / "src" / "models" / "license_plate_detector.pt",
        help="Checkpoint to start from (the current production model).",
    )
    parser.add_argument("--runs-dir", type=Path, default=work / "runs")
    parser.add_argument("--name", default="finetune_plate_yolo")
    parser.add_argument(
        "--artifact-path", type=Path, default=root / "src" / "models" / "license_plate_detector.pt"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-minutes", type=float, default=25)
    parser.add_argument(
        "--imgsz", type=int, default=960, help="Keep equal to the base model's training size."
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.001, help="Lower than from-scratch training (0.01).")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--fraction", type=float, default=1.0, help="Share of the train split to use (0..1].")
    parser.add_argument(
        "--occlusion-fraction",
        type=float,
        default=0.3,
        help="Share of train images that get an extra occluded copy (0 disables).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Replace src/models/license_plate_detector.pt (old one goes to src/models/rollback/).",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 0 < args.max_minutes <= 30:
        parser.error("--max-minutes must be positive and at most 30")
    for name in ("epochs", "imgsz", "batch"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be in (0, 1]")
    if not 0 <= args.occlusion_fraction <= 1:
        parser.error("--occlusion-fraction must be in [0, 1]")
    if not args.base_model.exists():
        parser.error(f"Base model not found: {args.base_model}")
    for split in ("train", "val", "test"):
        if not (args.data_dir / "images" / split).is_dir():
            parser.error(
                f"Prepared dataset not found: {args.data_dir / 'images' / split}. "
                "Run `make prepare` in ml/plate_detector first."
            )


def write_finetune_yaml(args: argparse.Namespace) -> Path:
    base = args.data_dir.resolve()
    train_dirs = [str(base / "images" / "train")]
    if args.occlusion_fraction > 0:
        count = build_occluded_split(
            base / "images" / "train",
            base / "labels" / "train",
            args.occluded_dir,
            "train",
            fraction=args.occlusion_fraction,
            seed=args.seed,
        )
        print(f"Occluded train copies: {count} -> {args.occluded_dir / 'images' / 'train'}")
        if count:
            train_dirs.append(str((args.occluded_dir / "images" / "train").resolve()))

    # Model selection (best.pt) on clean + occluded val, so robustness counts too.
    val_dirs = [str(base / "images" / "val")]
    if args.occlusion_fraction > 0:
        count = build_occluded_split(
            base / "images" / "val",
            base / "labels" / "val",
            args.occluded_dir,
            "val",
            fraction=1.0,
            seed=args.seed + 1,
        )
        if count:
            val_dirs.append(str((args.occluded_dir / "images" / "val").resolve()))

    yaml_path = args.occluded_dir / "data_finetune.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"path: {base}",
        "train:",
        *[f"  - {d}" for d in train_dirs],
        "val:",
        *[f"  - {d}" for d in val_dirs],
        f"test: {base / 'images' / 'test'}",
        "names:",
        "  0: license_plate",
        "",
    ]
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return yaml_path


def publish(best: Path, artifact: Path) -> None:
    if artifact.exists():
        rollback = artifact.parent / "rollback" / artifact.name
        rollback.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, rollback)
        print(f"Previous model backed up: {rollback}")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, artifact)
    print(f"Published fine-tuned detector: {artifact}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    from ultralytics import YOLO

    device = resolve_training_device(args.device)
    data_yaml = write_finetune_yaml(args)
    print(f"Fine-tuning {args.base_model} on {data_yaml} (device={device})")

    started = time.monotonic()
    model = YOLO(str(args.base_model))

    # With `time=` Ultralytics re-computes epochs to fill the whole budget (can be
    # more than requested on small data). Keep --epochs as a hard upper bound.
    def stop_after_max_epochs(trainer) -> None:
        if trainer.epoch + 1 >= args.epochs:
            trainer.stop = True

    model.add_callback("on_fit_epoch_end", stop_after_max_epochs)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        time=args.max_minutes / 60,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=0.1,
        cos_lr=True,
        warmup_epochs=min(1.0, args.epochs / 5),
        close_mosaic=min(2, args.epochs),
        fraction=args.fraction,
        seed=args.seed,
        project=str(args.runs_dir),
        name=args.name,
        exist_ok=True,
        plots=False,
    )
    elapsed = time.monotonic() - started

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Training finished but best checkpoint is missing: {best}")

    summary = {
        "model": "license_plate_detector",
        "base_model": str(args.base_model),
        "finetuned_model": str(best),
        "data": str(data_yaml),
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "lr0": args.lr0,
        "occlusion_fraction": args.occlusion_fraction,
        "fraction": args.fraction,
        "device": str(device),
        "train_minutes": round(elapsed / 60, 1),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    summary_path = best.parent.parent / "finetune_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Best fine-tuned checkpoint: {best}")
    print(f"Summary: {summary_path}")

    if args.publish:
        publish(best, args.artifact_path)


if __name__ == "__main__":
    main()
