import asyncio

from PIL import Image

from app.core.logging import logger
from app.neurons.base import BaseNeuron


class TextImageEmbeddingNeuron(BaseNeuron):
    """
    Joint text/image embedding model (CLIP) for search by free-text description.

    Car crops are encoded with the CLIP image tower, operator descriptions
    ("белая Toyota Camry седан", "red pickup truck") are encoded with a
    multilingual text encoder distilled into the same vector space, so a
    description can be matched against indexed cars with plain kNN.

    Defaults:
      - image: ``clip-ViT-B-32``
      - text:  ``sentence-transformers/clip-ViT-B-32-multilingual-v1``
        (50+ languages, including Russian; aligned with clip-ViT-B-32)
    """

    def __init__(
        self,
        image_model_name: str,
        text_model_name: str,
        embedding_dim: int = 512,
    ):
        self.image_model_name = image_model_name
        self.text_model_name = text_model_name
        self.embedding_dim = embedding_dim
        self.image_model = None
        self.text_model = None

    async def initialize(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for text search. "
                "Install ML dependencies with: pip install -r requirements-ml.txt"
            ) from exc

        self.image_model = await asyncio.to_thread(
            SentenceTransformer, self.image_model_name
        )
        if self.text_model_name == self.image_model_name:
            self.text_model = self.image_model
        else:
            self.text_model = await asyncio.to_thread(
                SentenceTransformer, self.text_model_name
            )

        probe = await self.encode_text("car")
        if len(probe) != self.embedding_dim:
            raise RuntimeError(
                f"Text embedding dim {len(probe)} does not match configured "
                f"NEURON5_CLIP_EMBEDDING_DIM={self.embedding_dim}"
            )

        logger.info(
            "Text/image embedding model initialized: image=%s, text=%s, dim=%d",
            self.image_model_name,
            self.text_model_name,
            self.embedding_dim,
        )

    async def encode_image(self, image: Image.Image) -> list[float]:
        if self.image_model is None:
            raise RuntimeError("Text/image embedding model is not initialized")
        vector = await asyncio.to_thread(
            self.image_model.encode,
            [image.convert("RGB")],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector[0].astype(float).tolist()

    async def encode_text(self, text: str) -> list[float]:
        if self.text_model is None:
            raise RuntimeError("Text/image embedding model is not initialized")
        vector = await asyncio.to_thread(
            self.text_model.encode,
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector[0].astype(float).tolist()

    async def predict(self, image: Image.Image) -> dict:
        embedding = await self.encode_image(image)
        return {"embedding": embedding, "embedding_dim": len(embedding)}
