from __future__ import annotations


def resolve_device(device: str) -> str:
    if device.lower() != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "0"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"
