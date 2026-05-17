#!/usr/bin/env python3
"""Create/update a GitHub Release with AutobahnCV model artifacts."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


MODEL_FILES = (
    "car_detector.pt",
    "license_plate_detector.pt",
    "plate_ocr.pt",
    "car_embedder.pt",
)


def token() -> str:
    value = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not value:
        raise RuntimeError("Set GITHUB_TOKEN or GH_TOKEN before uploading release assets.")
    return value


def request(
    url: str,
    api_token: str,
    method: str = "GET",
    body: bytes | None = None,
    accept: str = "application/vnd.github+json",
    content_type: str | None = None,
) -> urllib.request.Request:
    headers = {
        "Accept": accept,
        "Authorization": f"Bearer {api_token}",
        "User-Agent": "autobahncv-model-uploader",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if content_type:
        headers["Content-Type"] = content_type
    return urllib.request.Request(url, data=body, headers=headers, method=method)


def api_json(url: str, api_token: str, method: str = "GET", data: dict | None = None):
    body = json.dumps(data).encode("utf-8") if data is not None else None
    with urllib.request.urlopen(
        request(url, api_token, method=method, body=body, content_type="application/json"),
        timeout=60,
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def release_by_tag(repo: str, tag: str, api_token: str) -> dict | None:
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    try:
        return api_json(url, api_token)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def create_release(repo: str, tag: str, api_token: str) -> dict:
    print(f"Creating release {tag}.")
    return api_json(
        f"https://api.github.com/repos/{repo}/releases",
        api_token,
        method="POST",
        data={
            "tag_name": tag,
            "name": "AutobahnCV model artifacts",
            "body": (
                "Model artifacts used by docker compose when src/models is empty.\n\n"
                "Expected files: car_detector.pt, license_plate_detector.pt, "
                "plate_ocr.pt, car_embedder.pt."
            ),
            "draft": False,
            "prerelease": False,
        },
    )


def delete_asset(asset: dict, api_token: str) -> None:
    with urllib.request.urlopen(
        request(asset["url"], api_token, method="DELETE"), timeout=60
    ) as response:
        response.read()


def upload_asset(upload_url_template: str, path: Path, api_token: str) -> None:
    upload_url = upload_url_template.split("{", 1)[0]
    query = urllib.parse.urlencode({"name": path.name})
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    print(f"Uploading {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MiB).")
    with path.open("rb") as file:
        body = file.read()
    with urllib.request.urlopen(
        request(
            f"{upload_url}?{query}",
            api_token,
            method="POST",
            body=body,
            content_type=content_type,
        ),
        timeout=600,
    ) as response:
        response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="nailmarsel/occluded-plate-detector")
    parser.add_argument("--tag", default="models-v1")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--checksum-file", default="model-release.sha256")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing release assets with the same names before upload.",
    )
    args = parser.parse_args()

    api_token = token()
    models_dir = Path(args.models_dir)
    assets = [models_dir / name for name in MODEL_FILES] + [Path(args.checksum_file)]
    missing = [str(path) for path in assets if not path.exists()]
    if missing:
        print(f"Missing files: {', '.join(missing)}", file=sys.stderr)
        return 1

    release = release_by_tag(args.repo, args.tag, api_token)
    if release is None:
        release = create_release(args.repo, args.tag, api_token)

    existing = {asset["name"]: asset for asset in release.get("assets", [])}
    for path in assets:
        if path.name in existing:
            if not args.replace:
                print(f"Asset already exists, skipping: {path.name}")
                continue
            print(f"Deleting existing asset: {path.name}")
            delete_asset(existing[path.name], api_token)
        upload_asset(release["upload_url"], path, api_token)

    print(f"Release is ready: {release['html_url']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Upload failed: {error}", file=sys.stderr)
        raise SystemExit(1)
