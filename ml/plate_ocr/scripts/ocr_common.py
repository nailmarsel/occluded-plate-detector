from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms


ALPHABET = "0123456789ABEKMHOPCTYX"
BLANK_INDEX = 0
CHAR_TO_INDEX = {char: index + 1 for index, char in enumerate(ALPHABET)}
INDEX_TO_CHAR = {index + 1: char for index, char in enumerate(ALPHABET)}
PLATE_PATTERN = re.compile(rf"^[{ALPHABET}]+$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def encode_label(label: str) -> list[int]:
    return [CHAR_TO_INDEX[char] for char in label if char in CHAR_TO_INDEX]


def decode_logits(logits: torch.Tensor) -> list[str]:
    predictions = logits.argmax(dim=-1).cpu()
    decoded: list[str] = []
    for sequence in predictions:
        chars: list[str] = []
        previous = BLANK_INDEX
        for index in sequence.tolist():
            if index != BLANK_INDEX and index != previous:
                chars.append(INDEX_TO_CHAR.get(index, ""))
            previous = index
        decoded.append("".join(chars))
    return decoded


def build_ocr_transform():
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 192)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


class PlateOCRModel(nn.Module):
    def __init__(self, num_classes: int = len(ALPHABET) + 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.sequence = nn.LSTM(
            input_size=512 * 3,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        batch, channels, height, width = features.shape
        features = features.permute(0, 3, 1, 2).reshape(batch, width, channels * height)
        sequence, _ = self.sequence(features)
        return self.classifier(sequence)


class PlateOCRDataset(Dataset):
    def __init__(self, manifest_path: Path):
        self.samples: list[tuple[Path, str]] = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image_path, label = line.split(",", 1)
            self.samples.append((Path(image_path), label))
        self.transform = build_ocr_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


@dataclass
class Batch:
    images: torch.Tensor
    targets: torch.Tensor
    target_lengths: torch.Tensor
    labels: list[str]


def collate_batch(samples) -> Batch:
    images = torch.stack([sample[0] for sample in samples])
    labels = [sample[1] for sample in samples]
    encoded = [torch.tensor(encode_label(label), dtype=torch.long) for label in labels]
    targets = torch.cat(encoded)
    target_lengths = torch.tensor([len(item) for item in encoded], dtype=torch.long)
    return Batch(images, targets, target_lengths, labels)
