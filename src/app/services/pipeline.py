import io
from typing import Any

from PIL import Image

from app.core.config import settings
from app.core.logging import logger
from app.neurons.car_plate_detection import CarDetectionNeuron, PlateDetectionNeuron
from app.neurons.embedding import EmbeddingNeuron
from app.neurons.ocr import OCRNeuron


class ImageProcessingPipeline:
    """
    Orchestrates the image processing pipeline:
    1. Detect and crop car (YOLO v8)
    2. Detect and crop license plate (YOLO v8)
    3. Recognize plate number (OCR)
    4. Generate embedding (ResNet-108)
    """

    def __init__(self):
        self.car_detector: CarDetectionNeuron | None = None
        self.plate_detector: PlateDetectionNeuron | None = None
        self.ocr: OCRNeuron | None = None
        self.embedding_model: EmbeddingNeuron | None = None

    def _ensure_initialized(self) -> None:
        """Ensure all neurons are initialized, raise otherwise."""
        if self.car_detector is None:
            raise RuntimeError("Car detection neuron not initialized")
        if self.plate_detector is None:
            raise RuntimeError("Plate detection neuron not initialized")
        if self.ocr is None:
            raise RuntimeError("OCR neuron not initialized")
        if self.embedding_model is None:
            raise RuntimeError("Embedding neuron not initialized")

    async def initialize(self) -> None:
        """Initialize all neural network models."""
        logger.info("Initializing image processing pipeline...")

        self.car_detector = CarDetectionNeuron(
            model_path=settings.NEURON1_CAR_DETECTION_MODEL,
            confidence_threshold=settings.NEURON1_CONFIDENCE_THRESHOLD
        )

        self.plate_detector = PlateDetectionNeuron(
            model_path=settings.NEURON2_PLATE_DETECTION_MODEL,
            confidence_threshold=settings.NEURON2_CONFIDENCE_THRESHOLD
        )

        self.ocr = OCRNeuron(
            model_path=settings.NEURON3_OCR_MODEL,
            confidence_threshold=settings.NEURON3_CONFIDENCE_THRESHOLD
        )

        self.embedding_model = EmbeddingNeuron(
            model_path=settings.NEURON4_RESNET_MODEL,
            embedding_dim=settings.NEURON4_EMBEDDING_DIM
        )

        # Initialize all models
        await self.car_detector.initialize()
        await self.plate_detector.initialize()
        await self.ocr.initialize()
        await self.embedding_model.initialize()

        logger.info("Image processing pipeline initialized successfully")

    async def process_for_search(self, image_data: bytes) -> dict[str, Any]:
        """
        Process input image for search: extract plate number and generate embedding.

        Args:
            image_data: Raw image bytes

        Returns:
            dict with keys:
                - plate_number: str - Recognized license plate text
                - embedding: List[float] - Car image embedding
                - cropped_car_image: PIL Image (for debugging/logging)
                - cropped_plate_image: PIL Image (for debugging/logging)
        """
        try:
            self._ensure_initialized()

            # Load image
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            logger.info(f"Input image loaded: {image.size}")

            # Step 1: Detect and crop car
            assert self.car_detector is not None
            car_result = await self.car_detector.predict(image)
            cropped_car = car_result["cropped_car_image"]
            logger.info(f"Car detected with confidence: {car_result['confidence']}")

            # Step 2: Detect and crop license plate
            assert self.plate_detector is not None
            plate_result = await self.plate_detector.predict(cropped_car)
            cropped_plate = plate_result["cropped_plate_image"]
            logger.info(f"Plate detected with confidence: {plate_result['confidence']}")

            # Step 3: OCR to recognize plate number
            assert self.ocr is not None
            ocr_result = await self.ocr.predict(cropped_plate)
            plate_number = ocr_result["plate_number"]
            logger.info(
                f"Plate number recognized: '{plate_number}' "
                f"(conf: {ocr_result['confidence']})"
            )

            # Step 4: Generate embedding from cropped car image
            assert self.embedding_model is not None
            embedding_result = await self.embedding_model.predict(cropped_car)
            embedding = embedding_result["embedding"]
            logger.info(f"Embedding generated: dim={len(embedding)}")

            return {
                "plate_number": plate_number,
                "embedding": embedding,
                "cropped_car_image": cropped_car,
                "cropped_plate_image": cropped_plate,
                "car_bbox": car_result["bbox"],
                "plate_bbox": plate_result["bbox"]
            }

        except Exception as e:
            logger.error(f"Image processing pipeline failed: {e}")
            raise

    async def process_for_indexing(self, image_data: bytes) -> dict[str, Any]:
        """
        Process image for indexing: same as process_for_search but optimized
        for indexing workflow.

        Args:
            image_data: Raw image bytes

        Returns:
            dict with keys:
                - plate_number: str - Recognized license plate text
                - embedding: List[float] - Car image embedding
                - cropped_car_image: PIL Image - Car image without visible plate
        """
        return await self.process_for_search(image_data)
