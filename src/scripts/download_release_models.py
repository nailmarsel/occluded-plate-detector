#!/usr/bin/env python3
"""Download application model artifacts from a GitHub Release."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_MODEL_FILES = (
    "car_detector.pt",
    "license_plate_detector.pt",
    "plate_ocr.pt",
    "car_embedder.pt",
)


def env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def request(url: str, token: str | None = None, accept: str = "application/json"):
    headers = {
        "Accept": accept,
        "User-Agent": "autobahncv-model-downloader",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checksums(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}

    checksums: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, filename = line.split(maxsplit=1)
        checksums[Path(filename).name] = digest
    return checksums


def fetch_release(repo: str, tag: str, token: str | None) -> dict:
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    with urllib.request.urlopen(request(url, token), timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def asset_urls(release: dict) -> dict[str, str]:
    return {
        asset["name"]: asset["url"]
        for asset in release.get("assets", [])
        if "name" in asset and "url" in asset
    }


def download_asset(url: str, destination: Path, token: str | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with urllib.request.urlopen(
            request(url, token, accept="application/octet-stream"), timeout=300
        ) as response:
            with temp_path.open("wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)
        temp_path.replace(destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def model_ok(path: Path, expected_digest: str | None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if expected_digest and sha256(path) != expected_digest:
        print(f"Checksum mismatch for {path.name}; downloading again.")
        return False
    return True


def main() -> int:
    repo = env("MODEL_RELEASE_REPO", "nailmarsel/occluded-plate-detector")
    tag = env("MODEL_RELEASE_TAG", "models-v1")
    models_dir = Path(env("MODEL_DIR", "/models"))
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    model_files = tuple(
        item.strip()
        for item in env("MODEL_FILES", ",".join(DEFAULT_MODEL_FILES)).split(",")
        if item.strip()
    )
    checksum_path = os.getenv("MODEL_SHA256_FILE")
    checksums = load_checksums(Path(checksum_path) if checksum_path else None)

    missing = [
        name
        for name in model_files
        if not model_ok(models_dir / name, checksums.get(name))
    ]
    if not missing:
        print(f"All model files are present in {models_dir}.")
        return 0

    print(f"Missing model files: {', '.join(missing)}")
    print(f"Downloading model files from {repo} release {tag}.")

    try:
        release = fetch_release(repo, tag, token)
    except urllib.error.HTTPError as error:
        print(
            f"Could not read release {repo}@{tag}: HTTP {error.code}. "
            "If the repository is private, set GITHUB_TOKEN or GH_TOKEN.",
            file=sys.stderr,
        )
        return 1

    urls = asset_urls(release)
    for name in missing:
        url = urls.get(name)
        if not url:
            print(f"Release asset not found: {name}", file=sys.stderr)
            return 1
        destination = models_dir / name
        print(f"Downloading {name}...")
        download_asset(url, destination, token)
        expected_digest = checksums.get(name)
        if expected_digest and sha256(destination) != expected_digest:
            print(f"Downloaded file failed checksum: {name}", file=sys.stderr)
            destination.unlink(missing_ok=True)
            return 1

    print(f"Model files are ready in {models_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
