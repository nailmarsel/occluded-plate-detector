from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_DATASET_REPO = "yandex/mad-cars"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    work_dir = root / "ml" / "car_embedder"
    parser = argparse.ArgumentParser(
        description="Prepare an ImageFolder dataset from Hugging Face MAD Cars."
    )
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token. Defaults to the HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=work_dir / "data" / "imagefolder" / "mad_cars",
    )
    parser.add_argument(
        "--max-identities",
        type=int,
        default=500,
        help="Maximum number of car_id folders to prepare.",
    )
    parser.add_argument(
        "--max-images-per-identity",
        type=int,
        default=12,
        help="Maximum images to keep for each car_id.",
    )
    parser.add_argument(
        "--min-images-per-identity",
        type=int,
        default=2,
        help="Remove identities with fewer images after preparation.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200_000,
        help="Maximum streamed rows to inspect before stopping.",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser


def safe_name(value: object) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return name or "unknown"


def image_suffix(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in IMAGE_EXTENSIONS else ".jpg"


def download_image(url: str, destination: Path, timeout: float) -> bool:
    import requests

    if destination.exists() and destination.stat().st_size > 0:
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Skipping {url}: {exc}")
        return False

    destination.write_bytes(response.content)
    return True


def remove_small_identities(output_dir: Path, min_images: int) -> None:
    for identity_dir in output_dir.iterdir():
        if not identity_dir.is_dir():
            continue
        images = [
            path
            for path in identity_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if len(images) >= min_images:
            continue
        for path in images:
            path.unlink()
        identity_dir.rmdir()


def main() -> None:
    args = build_parser().parse_args()
    from datasets import load_dataset
    from tqdm import tqdm

    token = args.hf_token or None
    dataset = load_dataset(
        args.dataset_repo,
        split=args.split,
        streaming=True,
        token=token,
    )

    counts: dict[str, int] = {}
    seen_urls: set[str] = set()
    progress = tqdm(total=args.max_identities, desc="identities")

    for row_index, row in enumerate(dataset):
        if row_index >= args.max_rows:
            break
        if len(counts) >= args.max_identities and all(
            count >= args.max_images_per_identity for count in counts.values()
        ):
            break

        car_id = row.get("car_id")
        url = row.get("url")
        if car_id is None or not url:
            continue

        identity = safe_name(car_id)
        current_count = counts.get(identity, 0)
        if identity not in counts and len(counts) >= args.max_identities:
            continue
        if current_count >= args.max_images_per_identity:
            continue
        if url in seen_urls:
            continue

        view_id = safe_name(row.get("view_id", current_count))
        destination = (
            args.output_dir
            / identity
            / f"{view_id}_{current_count:03d}{image_suffix(url)}"
        )
        if download_image(url, destination, args.timeout):
            seen_urls.add(url)
            counts[identity] = current_count + 1
            if current_count == 0:
                progress.update(1)

    progress.close()
    remove_small_identities(args.output_dir, args.min_images_per_identity)

    identities = [path for path in args.output_dir.iterdir() if path.is_dir()]
    image_count = sum(
        1
        for identity_dir in identities
        for path in identity_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"Prepared ImageFolder: {args.output_dir}")
    print(f"Identities: {len(identities)}")
    print(f"Images: {image_count}")
    if len(identities) < 2:
        raise RuntimeError("Need at least two identities for embedder training.")


if __name__ == "__main__":
    main()
