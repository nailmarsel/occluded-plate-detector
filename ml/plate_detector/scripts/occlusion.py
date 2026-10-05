"""Synthetic plate occlusion for detector fine-tuning and robustness tests.

The project targets partially hidden plates (dirt, tow bars, other cars, masked
characters like the samples in ``tests/``). The public detection dataset has
almost only clean plates, so we create occluded copies: a part of every plate
box is covered with a rectangle (gray/black/white/noise), while the YOLO label
stays the same — the plate is still there and must be found.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from PIL import Image, ImageDraw

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_yolo_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, xc, yc, w, h = parts[:5]
        boxes.append((int(float(cls)), float(xc), float(yc), float(w), float(h)))
    return boxes


def _rng_for(path: Path, seed: int) -> random.Random:
    digest = hashlib.sha1(f"{seed}:{path.name}".encode()).hexdigest()
    return random.Random(int(digest[:12], 16))


def occlude_image(
    image: Image.Image,
    boxes: list[tuple[int, float, float, float, float]],
    rng: random.Random,
    min_cover: float = 0.25,
    max_cover: float = 0.6,
) -> Image.Image:
    """Cover ``min_cover..max_cover`` of each plate box width."""
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for _cls, xc, yc, w, h in boxes:
        x1, x2 = (xc - w / 2) * width, (xc + w / 2) * width
        y1, y2 = (yc - h / 2) * height, (yc + h / 2) * height
        box_w, box_h = x2 - x1, y2 - y1
        if box_w < 4 or box_h < 2:
            continue

        cover_w = box_w * rng.uniform(min_cover, max_cover)
        where = rng.choice(("left", "right", "middle"))
        if where == "left":
            ox1 = x1 - box_w * 0.05
        elif where == "right":
            ox1 = x2 - cover_w + box_w * 0.05
        else:
            ox1 = x1 + rng.uniform(0.15, 0.85) * (box_w - cover_w)
        ox2 = ox1 + cover_w

        # Mostly full-height blocks; sometimes only the lower/upper part.
        if rng.random() < 0.7:
            oy1, oy2 = y1 - box_h * 0.15, y2 + box_h * 0.15
        else:
            part = rng.uniform(0.5, 0.8) * box_h
            oy1, oy2 = (y2 - part, y2 + box_h * 0.1) if rng.random() < 0.5 else (y1 - box_h * 0.1, y1 + part)

        style = rng.choice(("gray", "gray", "black", "white", "noise"))
        rect = [int(ox1), int(oy1), int(ox2), int(oy2)]
        if style == "noise":
            rw, rh = max(rect[2] - rect[0], 1), max(rect[3] - rect[1], 1)
            noise = Image.effect_noise((rw, rh), 64).convert("RGB")
            image.paste(noise, (rect[0], rect[1]))
        else:
            value = {
                "gray": rng.randint(90, 150),
                "black": rng.randint(0, 30),
                "white": rng.randint(220, 255),
            }[style]
            draw.rectangle(rect, fill=(value, value, value))
    return image


def build_occluded_split(
    images_dir: Path,
    labels_dir: Path,
    out_root: Path,
    split: str,
    fraction: float = 1.0,
    seed: int = 0,
    limit: int | None = None,
) -> int:
    """Write occluded copies of ``fraction`` of images in a split.

    Output layout mirrors YOLO: ``out_root/images/<split>`` and
    ``out_root/labels/<split>``. Existing files are reused, so re-runs are cheap.
    """
    out_images = out_root / "images" / split
    out_labels = out_root / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    selector = random.Random(seed)
    if fraction < 1.0:
        images = [p for p in images if selector.random() < fraction]
    if limit:
        images = images[:limit]

    written = 0
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        boxes = read_yolo_boxes(label_path)
        if not boxes:
            continue
        target = out_images / f"{image_path.stem}_occ.jpg"
        target_label = out_labels / f"{image_path.stem}_occ.txt"
        if not target.exists():
            with Image.open(image_path) as image:
                occlude_image(image, boxes, _rng_for(image_path, seed)).save(target, quality=92)
        if not target_label.exists():
            target_label.write_text(label_path.read_text(encoding="utf-8"), encoding="utf-8")
        written += 1
    return written
