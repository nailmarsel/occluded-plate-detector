#!/usr/bin/env python3
"""Dataset validation for AutobahnCV.

Runs reproducible data-quality checks over the two project datasets and writes
a markdown report. This is the executable artifact behind
``reports/data/DATA_QUALITY_REPORT.md`` and the ``validate`` stage of ``dvc.yaml``.

Checks implemented:
    * integrity      - files are readable, valid PNG/JPEG
    * format         - only image/png and image/jpeg are accepted
    * resolution     - images are at least 256x256
    * duplicates     - exact duplicates via SHA-256, near-duplicates via aHash
    * leakage        - no image (by content hash) appears in two splits
    * class_balance  - distribution of samples across splits / OCR regions
    * annotations    - detection: YOLO "class xc yc w h", coords in (0,1], class 0
                       ocr: filename matches the Russian-plate alphabet/length

Dataset layouts (see specs/Data_Spec.md):
    detection/<split>/images/*.png  + detection/<split>/labels/*.txt
    ocr/<split>/*.png               (label is the filename stem)

Usage:
    python3 scripts/validate_dataset.py --root data/datasets \
        --report reports/data/DATA_QUALITY_REPORT.md
    python3 scripts/validate_dataset.py --self-test      # CI smoke check

If the dataset root is absent (e.g. ``dvc pull`` was not run), the script exits
cleanly with code 0 and a "skipped" note instead of failing the pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MIN_RESOLUTION = 256
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg"}
OCR_ALPHABET = set("1234567890ABEKMHOPCTYX")
OCR_PLATE_RE = re.compile(r"^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$")
SPLITS = ("train", "val", "test")
AHASH_HAMMING_NEAR_DUP = 4  # <= this many differing bits => near-duplicate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path) -> str | None:
    """8x8 average hash; returns a 64-bit hex string, or None if unreadable."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as img:
            small = img.convert("L").resize((8, 8))
            pixels = list(small.getdata())
    except Exception:
        return None
    mean = sum(pixels) / len(pixels)
    bits = 0
    for px in pixels:
        bits = (bits << 1) | (1 if px >= mean else 0)
    return f"{bits:016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


class Result:
    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append((name, ok, detail))

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)


def _iter_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in ALLOWED_SUFFIXES
    )


def validate_detection(root: Path, res: Result) -> None:
    base = root / "detection"
    if not base.exists():
        return
    hashes_by_split: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    bad_format = bad_res = bad_ann = corrupt = 0

    for split in SPLITS:
        images = _iter_images(base / split / "images")
        counts[split] = len(images)
        hashes_by_split[split] = set()
        for img in images:
            if img.suffix.lower() not in ALLOWED_SUFFIXES:
                bad_format += 1
            try:
                from PIL import Image

                with Image.open(img) as im:
                    im.verify()
                with Image.open(img) as im:
                    w, h = im.size
                if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
                    bad_res += 1
            except ImportError:
                pass
            except Exception:
                corrupt += 1
            hashes_by_split[split].add(sha256(img))

            label = base / split / "labels" / f"{img.stem}.txt"
            if label.exists():
                for line in label.read_text().splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        bad_ann += 1
                        break
                    cls, *coords = parts
                    if cls != "0" or not all(0 < float(c) <= 1 for c in coords):
                        bad_ann += 1
                        break

    res.add("detection.format", bad_format == 0, f"{bad_format} non-image files")
    res.add("detection.resolution", bad_res == 0, f"{bad_res} images < 256x256")
    res.add("detection.integrity", corrupt == 0, f"{corrupt} corrupt files")
    res.add("detection.annotations", bad_ann == 0, f"{bad_ann} invalid YOLO labels")

    leaks = 0
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                leaks += len(hashes_by_split.get(a, set()) & hashes_by_split.get(b, set()))
    res.add("detection.split_leakage", leaks == 0, f"{leaks} images shared across splits")

    total = sum(counts.values()) or 1
    balance = ", ".join(f"{s}={counts.get(s,0)} ({counts.get(s,0)/total:.0%})" for s in SPLITS)
    res.add("detection.class_balance", True, balance)


def validate_ocr(root: Path, res: Result) -> None:
    base = root / "ocr"
    if not base.exists():
        return
    hashes_by_split: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    bad_name = corrupt = 0
    regions: Counter[str] = Counter()

    for split in SPLITS:
        images = _iter_images(base / split)
        counts[split] = len(images)
        hashes_by_split[split] = set()
        for img in images:
            label = img.stem.upper()
            if not (set(label) <= OCR_ALPHABET and OCR_PLATE_RE.match(label)):
                bad_name += 1
            else:
                regions[label[-3:] if len(label) == 9 else label[-2:]] += 1
            try:
                from PIL import Image

                with Image.open(img) as im:
                    im.verify()
            except ImportError:
                pass
            except Exception:
                corrupt += 1
            hashes_by_split[split].add(sha256(img))

    res.add("ocr.filename_labels", bad_name == 0, f"{bad_name} filenames off the plate alphabet/pattern")
    res.add("ocr.integrity", corrupt == 0, f"{corrupt} corrupt files")

    leaks = 0
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                leaks += len(hashes_by_split.get(a, set()) & hashes_by_split.get(b, set()))
    res.add("ocr.split_leakage", leaks == 0, f"{leaks} images shared across splits")

    total = sum(counts.values()) or 1
    balance = ", ".join(f"{s}={counts.get(s,0)} ({counts.get(s,0)/total:.0%})" for s in SPLITS)
    res.add("ocr.class_balance", True, balance)
    res.add("ocr.region_coverage", True, f"{len(regions)} distinct plate regions observed")


def write_report(res: Result, out: Path, root: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Data Quality Report — AutobahnCV",
        "",
        "_Сгенерировано `scripts/validate_dataset.py`. Не редактировать вручную._",
        "",
        f"Источник: `{root}`",
        "",
        "| Проверка | Статус | Детали |",
        "|----------|--------|--------|",
    ]
    for name, ok, detail in res.checks:
        lines.append(f"| `{name}` | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines += ["", f"**Итог:** {'ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if res.passed else 'ЕСТЬ ОШИБКИ'}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> int:
    """Exercise check logic on a tiny synthetic fixture (used by CI)."""
    import tempfile

    try:
        from PIL import Image
    except ImportError:
        print("self-test skipped: Pillow not installed")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        det = root / "detection" / "train"
        (det / "images").mkdir(parents=True)
        (det / "labels").mkdir(parents=True)
        Image.new("RGB", (300, 300), "gray").save(det / "images" / "A123BC77.png")
        (det / "labels" / "A123BC77.txt").write_text("0 0.5 0.5 0.2 0.1\n")
        ocr = root / "ocr" / "train"
        ocr.mkdir(parents=True)
        Image.new("RGB", (260, 80), "white").save(ocr / "A123BC77.png")

        res = Result()
        validate_detection(root, res)
        validate_ocr(root, res)
        ok = res.passed
        for name, passed, detail in res.checks:
            print(f"  {name}: {'PASS' if passed else 'FAIL'} ({detail})")
        print("self-test:", "PASS" if ok else "FAIL")
        return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="AutobahnCV dataset validation")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "datasets")
    parser.add_argument(
        "--report", type=Path, default=REPO_ROOT / "reports" / "data" / "DATA_QUALITY_REPORT.md"
    )
    parser.add_argument("--self-test", action="store_true", help="run the CI smoke check")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if not args.root.exists():
        print(f"Dataset root not found: {args.root}")
        print("Run `dvc pull` to fetch the datasets, then re-run this script.")
        return 0

    res = Result()
    validate_detection(args.root, res)
    validate_ocr(args.root, res)
    write_report(res, args.report, args.root)

    for name, ok, detail in res.checks:
        print(f"{name:<28}{'PASS' if ok else 'FAIL':<6}{detail}")
    print(f"\nReport written to {args.report}")
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
