from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from PIL import Image
from app.core.logging import logger


class BaseNeuron(ABC):
    """Abstract base class for all neural network models."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model (load weights, setup device, etc.)."""
        pass

    @abstractmethod
    async def predict(self, *args, **kwargs) -> dict:
        """Run inference on input data."""
        pass
