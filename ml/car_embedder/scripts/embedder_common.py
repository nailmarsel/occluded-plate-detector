from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_device(device: str) -> torch.device:
    if device.lower() != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ResNetEmbedder(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet50"):
        super().__init__()
        if backbone == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT
            model = models.resnet101(weights=weights)
        else:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        self.backbone = model
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, images: torch.Tensor):
        features = self.backbone(images)
        features = nn.functional.normalize(features, p=2, dim=1)
        logits = self.classifier(features)
        return logits, features
