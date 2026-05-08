import io
import re
from typing import Any

from PIL import Image

from app.core.config import settings
from app.core.logging import logger
from app.neurons.car_plate_detection import CarDetectionNeuron, PlateDetectionNeuron
from app.neurons.embedding import EmbeddingNeuron
from app.neurons.ocr import OCRNeuron

PLATE_CHARS_PATTERN = re.compile(r"[^A-Z0-9]")
CYRILLIC_TO_LATIN_PLATE_CHARS = str.maketrans(
    {
        "А": "A",
        "В": "B",
        "Е": "E",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "Р": "P",
        "С": "C",
        "Т": "T",
        "У": "Y",
        "Х": "X",
    }
)


def normalize_plate_number(plate_number: str) -> str:
    normalized = plate_number.upper().translate(CYRILLIC_TO_LATIN_PLATE_CHARS)
    return PLATE_CHARS_PATTERN.sub("", normalized)


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
        self.initialization_errors: dict[str, str] = {}

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

        await self._initialize_neuron("car_detector", self.car_detector)
        await self._initialize_neuron("plate_detector", self.plate_detector)
        await self._initialize_neuron("ocr", self.ocr)
        await self._initialize_neuron("embedding_model", self.embedding_model)

        logger.info("Image processing pipeline initialized successfully")

    async def _initialize_neuron(self, name: str, neuron) -> None:
        try:
            await neuron.initialize()
        except Exception as exc:
            self.initialization_errors[name] = str(exc)
            if settings.ML_STRICT_STARTUP:
                raise
            logger.warning("%s disabled: %s", name, exc)

    def _ensure_neuron_ready(self, name: str) -> None:
        if name in self.initialization_errors:
            raise RuntimeError(
                f"{name} is not available: {self.initialization_errors[name]}"
            )

    async def process_for_search(
        self,
        image_data: bytes,
        plate_number: str | None = None,
        allow_missing_plate: bool = True,
    ) -> dict[str, Any]:
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
            self._ensure_neuron_ready("car_detector")
            car_result = await self.car_detector.predict(image)
            cropped_car = car_result["cropped_car_image"]
            logger.info(f"Car detected with confidence: {car_result['confidence']}")

            cropped_plate = None
            plate_bbox = None
            if plate_number:
                plate_number = normalize_plate_number(plate_number)
                if not plate_number:
                    raise ValueError("Plate number is empty after normalization")
                logger.info("Using provided plate number: '%s'", plate_number)
            else:
                # Step 2: Detect and crop license plate
                assert self.plate_detector is not None
                self._ensure_neuron_ready("plate_detector")
                plate_result = await self.plate_detector.predict(cropped_car)
                cropped_plate = plate_result["cropped_plate_image"]
                plate_bbox = plate_result["bbox"]
                logger.info(
                    f"Plate detected with confidence: {plate_result['confidence']}"
                )

                # Step 3: OCR to recognize plate number
                try:
                    assert self.ocr is not None
                    self._ensure_neuron_ready("ocr")
                    ocr_result = await self.ocr.predict(cropped_plate)
                    plate_number = ocr_result["plate_number"]
                    logger.info(
                        f"Plate number recognized: '{plate_number}' "
                        f"(conf: {ocr_result['confidence']})"
                    )
                except Exception as exc:
                    if not allow_missing_plate:
                        raise RuntimeError(
                            "Plate number was not recognized. Provide "
                            "plate_number or configure a real license plate "
                            "detector model."
                        ) from exc
                    plate_number = ""
                    logger.warning(
                        "Plate OCR failed; continuing with embedding-only "
                        "search: %s",
                        exc,
                    )

            # Step 4: Generate embedding from cropped car image
            assert self.embedding_model is not None
            self._ensure_neuron_ready("embedding_model")
            embedding_result = await self.embedding_model.predict(cropped_car)
            embedding = embedding_result["embedding"]
            logger.info(f"Embedding generated: dim={len(embedding)}")

            return {
                "plate_number": plate_number,
                "embedding": embedding,
                "cropped_car_image": cropped_car,
                "cropped_plate_image": cropped_plate,
                "car_bbox": car_result["bbox"],
                "plate_bbox": plate_bbox,
            }

        except Exception as e:
            logger.error(f"Image processing pipeline failed: {e}")
            raise

    async def process_for_indexing(
        self, image_data: bytes, plate_number: str | None = None
    ) -> dict[str, Any]:
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
        return await self.process_for_search(
            image_data,
            plate_number=plate_number,
            allow_missing_plate=False,
        )
