from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}
VEHICLE_CLASS_IDS = {2, 3, 5, 7}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    work_dir = root / "ml" / "car_detector"
    parser = argparse.ArgumentParser(
        description="Prepare a pseudo-labeled YOLO car dataset from plate dataset images."
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
        default=work_dir / "data" / "raw",
        help="Where Hugging Face dataset files are cached.",
    )
    parser.add_argument(
        "--source-raw-dir",
        type=Path,
        default=None,
        help="Use an existing downloaded dataset directory instead of downloading.",
    )
    parser.add_argument(
        "--yolo-dir",
        type=Path,
        default=work_dir / "data" / "yolo" / "car_from_plate_dataset",
        help="Pseudo-labeled YOLO output directory.",
    )
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy dataset files instead of creating symlinks.",
    )
    return parser


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


def safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination_root != target and destination_root not in target.parents:
                raise RuntimeError(f"Unsafe path in {archive_path}: {member.filename}")
        archive.extractall(destination)


def ensure_extracted_splits(raw_root: Path) -> Path:
    extracted_root = raw_root / "extracted"
    extracted_any = False

    for split in SPLIT_ALIASES:
        if find_split_dir(raw_root, split, "images"):
            continue

        archive_path = raw_root / f"{split}.zip"
        if not archive_path.exists():
            continue

        if find_split_dir(extracted_root, split, "images"):
            continue

        print(f"Extracting {archive_path.name}")
        safe_extract_zip(archive_path, extracted_root)
        extracted_any = True

    if extracted_any:
        print(f"Extracted dataset archives: {extracted_root}")

    return extracted_root if extracted_root.exists() else raw_root


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


def best_vehicle_box(result) -> tuple[float, float, float, float] | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    best = None
    best_confidence = -1.0
    for box in boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        if class_id not in VEHICLE_CLASS_IDS or confidence <= best_confidence:
            continue
        best = tuple(float(value) for value in box.xyxy[0].tolist())
        best_confidence = confidence

    return best


def yolo_label(
    bbox: tuple[float, float, float, float], image_width: int, image_height: int
) -> str:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(image_width), x1))
    y1 = max(0.0, min(float(image_height), y1))
    x2 = max(0.0, min(float(image_width), x2))
    y2 = max(0.0, min(float(image_height), y2))
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    x_center = x1 + width / 2
    y_center = y1 + height / 2
    return (
        "0 "
        f"{x_center / image_width:.6f} "
        f"{y_center / image_height:.6f} "
        f"{width / image_width:.6f} "
        f"{height / image_height:.6f}\n"
    )


def prepare_dataset(args: argparse.Namespace, raw_root: Path) -> Path:
    try:
        from device import resolve_device
    except ModuleNotFoundError:
        from scripts.device import resolve_device
    from PIL import Image
    from tqdm import tqdm
    from ultralytics import YOLO

    raw_root = ensure_extracted_splits(raw_root)
    device = resolve_device(args.device)
    if device != args.device:
        print(f"Resolved device '{args.device}' to '{device}'")

    model = YOLO(args.base_model)
    args.yolo_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        images_dir = find_split_dir(raw_root, split, "images")
        if images_dir is None:
            raise RuntimeError(f"Could not find images for split '{split}' in {raw_root}")

        split_images = image_files(images_dir)
        if args.max_images > 0:
            split_images = split_images[: args.max_images]

        kept = 0
        skipped = 0
        for image_path in tqdm(split_images, desc=f"pseudo-label {split}"):
            result = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.conf,
                device=device,
                verbose=False,
            )[0]
            bbox = best_vehicle_box(result)
            if bbox is None:
                skipped += 1
                continue

            with Image.open(image_path) as image:
                label = yolo_label(bbox, image.width, image.height)

            link_or_copy(
                image_path,
                args.yolo_dir / "images" / split / image_path.name,
                args.copy_files,
            )
            label_path = args.yolo_dir / "labels" / split / f"{image_path.stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(label, encoding="utf-8")
            kept += 1

        print(f"{split}: kept {kept} images, skipped {skipped} without vehicle boxes")

    data_yaml = args.yolo_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {args.yolo_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: car",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def main() -> None:
    args = build_parser().parse_args()
    if args.source_raw_dir and args.source_raw_dir.exists():
        raw_root = args.source_raw_dir
        print(f"Using existing dataset: {raw_root}")
    else:
        raw_root = download_dataset(args.dataset_repo, args.raw_dir, args.hf_token)
    data_yaml = prepare_dataset(args, raw_root)
    print(f"Prepared YOLO car dataset: {data_yaml}")


if __name__ == "__main__":
    main()
