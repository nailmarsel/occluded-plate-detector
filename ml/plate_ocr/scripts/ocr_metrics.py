"""Shared OCR evaluation helpers: edit distance, degradations, metric loop."""

from __future__ import annotations

import random
import time

import torch
from ocr_common import PlateOCRDataset, build_ocr_transform, collate_batch, decode_logits
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def degrade(image: Image.Image, rng: random.Random) -> Image.Image:
    """Camera-like degradation: blur, low contrast, light change, downscale, noise, tilt."""
    image = image.convert("RGB")
    w, h = image.size
    scale = rng.uniform(0.45, 0.8)
    image = image.resize((max(8, int(w * scale)), max(4, int(h * scale))), Image.BILINEAR).resize(
        (w, h), Image.BILINEAR
    )
    image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.6)))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.45, 0.85))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.6, 1.4))
    image = image.rotate(rng.uniform(-4, 4), resample=Image.BILINEAR, expand=False, fillcolor=(128, 128, 128))
    noise = Image.effect_noise((w, h), rng.uniform(10, 30)).convert("RGB")
    return Image.blend(image, noise, rng.uniform(0.05, 0.15))


class DegradedOCRDataset(Dataset):
    """Test set with a fixed (seeded) degradation per sample."""

    def __init__(self, base: PlateOCRDataset, seed: int = 0):
        self.base = base
        self.seed = seed
        self.transform = build_ocr_transform()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image_path, label = self.base.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(degrade(image, random.Random(self.seed * 1_000_003 + index))), label


def evaluate(model, dataset, device, batch_size: int = 64, workers: int = 0) -> dict:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_batch
    )
    model.eval()
    exact = total = errors = chars = 0
    infer_seconds = 0.0
    with torch.inference_mode():
        for batch in loader:
            images = batch.images.to(device)
            start = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_seconds += time.perf_counter() - start
            for pred, label in zip(decode_logits(logits), batch.labels, strict=False):
                exact += pred == label
                errors += edit_distance(pred, label)
                chars += len(label)
                total += 1
    cer = errors / max(chars, 1)
    return {
        "n": total,
        "exact_match": exact / max(total, 1),
        "char_accuracy": max(0.0, 1 - cer),
        "cer": cer,
        "ms_per_image": 1000 * infer_seconds / max(total, 1),
    }
