from __future__ import annotations

import argparse
import os
from pathlib import Path

from ocr_common import PLATE_PATTERN, repo_root


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    work_dir = root / "ml" / "plate_ocr"
    parser = argparse.ArgumentParser(description="Prepare OCR manifests.")
    parser.add_argument("--dataset-repo", default="AY000554/Car_plate_OCR_dataset")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token. Defaults to the HF_TOKEN environment variable.",
    )
    parser.add_argument("--raw-dir", type=Path, default=work_dir / "data" / "raw")
    parser.add_argument("--manifest-dir", type=Path, default=work_dir / "data" / "manifests")
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


def find_split_dir(root: Path, split: str) -> Path:
    names = SPLIT_ALIASES[split]
    for path in root.rglob("*"):
        if path.is_dir() and path.name.lower() in names:
            return path
    raise RuntimeError(f"Could not find split '{split}' in {root}")


def label_from_path(path: Path) -> str | None:
    label = path.stem.upper()
    if PLATE_PATTERN.fullmatch(label):
        return label
    return None


def write_manifest(split_dir: Path, manifest_path: Path) -> int:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for path in sorted(split_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = label_from_path(path)
        if label is None:
            continue
        rows.append(f"{path.resolve()},{label}")
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def main() -> None:
    args = build_parser().parse_args()
    raw_root = download_dataset(args.dataset_repo, args.raw_dir, args.hf_token)
    for split in ("train", "val", "test"):
        split_dir = find_split_dir(raw_root, split)
        count = write_manifest(split_dir, args.manifest_dir / f"{split}.csv")
        print(f"{split}: {count} samples")


if __name__ == "__main__":
    main()
