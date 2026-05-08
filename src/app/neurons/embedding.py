from PIL import Image

from app.core.logging import logger
from app.neurons.base import BaseNeuron


class EmbeddingNeuron(BaseNeuron):
    """
    Generates normalized image embeddings from cropped car images.

    Uses TorchVision ResNet feature vectors by removing the classification
    layer. The default ResNet-50 embedding size is 2048.
    """

    def __init__(self, model_path: str, embedding_dim: int = 2048):
        self.model_path = model_path.lower()
        self.embedding_dim = embedding_dim
        self.device = None
        self.model = None
        self.preprocess = None
        self.torch = None

    async def initialize(self) -> None:
        """Load a pretrained TorchVision ResNet model."""
        try:
            import torch
            from torchvision import models
        except ImportError as exc:
            raise RuntimeError(
                "torch and torchvision are required for embeddings. "
                "Install ML dependencies with: pip install -r requirements-ml.txt"
            ) from exc

        model_name = self.model_path or "resnet50"
        if model_name == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT
            backbone = models.resnet101(weights=weights)
            self.embedding_dim = 2048
        elif model_name == "resnet152":
            weights = models.ResNet152_Weights.DEFAULT
            backbone = models.resnet152(weights=weights)
            self.embedding_dim = 2048
        else:
            weights = models.ResNet50_Weights.DEFAULT
            backbone = models.resnet50(weights=weights)
            self.embedding_dim = 2048

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = weights.transforms()
        self.torch = torch

        logger.info(
            "Embedding model initialized: %s on %s", model_name, self.device
        )

    async def predict(self, image: Image.Image) -> dict:
        """
        Generate a normalized embedding vector for the input car image.
        """
        if (
            self.model is None
            or self.preprocess is None
            or self.device is None
            or self.torch is None
        ):
            raise RuntimeError("Embedding model is not initialized")

        try:
            image = image.convert("RGB")
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with self.torch.inference_mode():
                embedding = self.model(tensor).flatten(1)
                embedding = self.torch.nn.functional.normalize(embedding, p=2, dim=1)

            vector = embedding.squeeze(0).cpu().tolist()
            return {
                "embedding": vector,
                "embedding_dim": len(vector),
            }
        except Exception as e:
            logger.error("Embedding generation failed: %s", e)
            raise
