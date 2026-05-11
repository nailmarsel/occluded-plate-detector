from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    default_work_dir = root / "ml" / "plate_detector"
    parser = argparse.ArgumentParser(
        description="Train a YOLO detector for Russian license plates."
    )
    parser.add_argument(
        "--dataset-repo",
        default="AY000554/Car_plate_detecting_dataset",
        help="Hugging Face dataset repository id.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token. Defaults to the HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=default_work_dir / "data" / "raw",
        help="Where Hugging Face dataset files are cached.",
    )
    parser.add_argument(
        "--yolo-dir",
        type=Path,
        default=default_work_dir / "data" / "yolo" / "russian_plate",
        help="Normalized YOLO dataset output directory.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=default_work_dir / "runs",
        help="Ultralytics training output directory.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=root / "src" / "models" / "license_plate_detector.pt",
        help="Where to publish the trained detector for the app.",
    )
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--name", default="russian_plate_yolo")
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy dataset files instead of creating symlinks.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only download and normalize the dataset.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Copy best.pt to the app model artifact path after training.",
    )
    return parser


def find_split_dir(root: Path, split: str, kind: str) -> Path | None:
    names = SPLIT_ALIASES[split]
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        parts = {part.lower() for part in path.parts}
        if kind in parts and any(name in parts for name in names):
            candidates.append(path)

    if candidates:
        return min(candidates, key=lambda item: len(item.parts))

    for path in root.rglob("*"):
        if path.is_dir() and path.name.lower() in names:
            nested = path / kind
            if nested.exists():
                return nested

    return None


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def link_or_copy(source: Path, destination: Path, copy_files: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if copy_files:
        shutil.copy2(source, destination)
        return

    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def normalize_dataset(raw_root: Path, yolo_dir: Path, copy_files: bool) -> Path:
    yolo_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        images_dir = find_split_dir(raw_root, split, "images")
        labels_dir = find_split_dir(raw_root, split, "labels")
        if images_dir is None or labels_dir is None:
            raise RuntimeError(
                f"Could not find YOLO images/labels directories for split '{split}' "
                f"inside {raw_root}"
            )

        split_images = image_files(images_dir)
        if not split_images:
            raise RuntimeError(f"No images found for split '{split}' in {images_dir}")

        for image_path in split_images:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise RuntimeError(f"Missing label for {image_path}: {label_path}")

            link_or_copy(
                image_path,
                yolo_dir / "images" / split / image_path.name,
                copy_files,
            )
            link_or_copy(
                label_path,
                yolo_dir / "labels" / split / label_path.name,
                copy_files,
            )

    data_yaml = yolo_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {yolo_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: license_plate",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def download_dataset(dataset_repo: str, raw_dir: Path, hf_token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    token = hf_token or None
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Downloading Hugging Face dataset "
        f"{dataset_repo} ({'authenticated' if token else 'unauthenticated'})"
    )
    return Path(
        snapshot_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            local_dir=raw_dir / dataset_repo.replace("/", "__"),
            token=token,
        )
    )


def train(args: argparse.Namespace, data_yaml: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO(args.base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(args.runs_dir),
        name=args.name,
        exist_ok=True,
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Training finished but best checkpoint is missing: {best}")
    return best


def publish_model(best_checkpoint: Path, artifact_path: Path) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint, artifact_path)
    print(f"Published detector: {artifact_path}")


def main() -> None:
    args = build_parser().parse_args()
    raw_root = download_dataset(args.dataset_repo, args.raw_dir, args.hf_token)
    data_yaml = normalize_dataset(raw_root, args.yolo_dir, args.copy_files)
    print(f"Prepared YOLO dataset: {data_yaml}")

    if args.prepare_only:
        return

    best_checkpoint = train(args, data_yaml)
    print(f"Best checkpoint: {best_checkpoint}")
    if args.publish:
        publish_model(best_checkpoint, args.artifact_path)


if __name__ == "__main__":
    main()
