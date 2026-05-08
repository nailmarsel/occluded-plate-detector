from abc import ABC, abstractmethod


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
