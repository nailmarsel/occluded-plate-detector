from pathlib import Path

# ruff: noqa: I001
from PIL import Image
from app.core.config import settings
from app.core.logging import logger
from app.neurons.base import BaseNeuron


VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # COCO: car, motorcycle, bus, truck


def _missing_dependency_message(package: str) -> str:
    return (
        f"{package} is required for neural inference. Install ML dependencies "
        "with: pip install -r requirements-ml.txt"
    )


def _clamp_bbox(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(left + 1, min(width, int(round(x2))))
    bottom = max(top + 1, min(height, int(round(y2))))
    return left, top, right, bottom


class YoloDetectionNeuron(BaseNeuron):
    """Shared Ultralytics YOLO detector wrapper."""

    allowed_class_ids: set[int] | None = None
    result_image_key = "cropped_image"
    detector_name = "object"

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None

    async def initialize(self) -> None:
        """Load Ultralytics YOLO model weights."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(_missing_dependency_message("ultralytics")) from exc

        self.model = YOLO(self.model_path)
        logger.info("%s model initialized: %s", self.detector_name, self.model_path)

    def _best_box(self, result) -> tuple[tuple[float, float, float, float], float]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            raise ValueError(f"No {self.detector_name} detected")

        best_bbox = None
        best_confidence = -1.0

        for box in boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            if confidence < self.confidence_threshold:
                continue
            if (
                self.allowed_class_ids is not None
                and class_id not in self.allowed_class_ids
            ):
                continue

            if confidence > best_confidence:
                xyxy = box.xyxy[0].tolist()
                best_bbox = tuple(float(value) for value in xyxy)
                best_confidence = confidence

        if best_bbox is None:
            raise ValueError(
                f"No {self.detector_name} matched confidence/class filters"
            )

        return best_bbox, best_confidence

    async def predict(self, image: Image.Image) -> dict:
        """Detect the highest-confidence target and return a cropped image."""
        if self.model is None:
            raise RuntimeError(f"{self.detector_name} model is not initialized")

        try:
            results = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                verbose=False,
            )
            bbox, confidence = self._best_box(results[0])
            crop_box = _clamp_bbox(bbox, image.width, image.height)

            return {
                self.result_image_key: image.crop(crop_box),
                "bbox": list(crop_box),
                "confidence": confidence,
            }
        except Exception as e:
            logger.error("%s detection failed: %s", self.detector_name, e)
            raise


class CarDetectionNeuron(YoloDetectionNeuron):
    """
    Detects and crops vehicles using Ultralytics YOLO.

    The default model is YOLO26 nano. It detects COCO vehicle classes, then
    crops the highest-confidence car-like object.
    """

    allowed_class_ids = VEHICLE_CLASS_IDS
    result_image_key = "cropped_car_image"
    detector_name = "car"

    async def initialize(self) -> None:
        await super().initialize()
        model_path = Path(self.model_path)
        if model_path.exists() and model_path.suffix == ".pt":
            self.allowed_class_ids = None
            logger.info(
                "Using custom car detector without COCO class-id filtering: %s",
                self.model_path,
            )


class PlateDetectionNeuron(YoloDetectionNeuron):
    """
    Detects and crops license plates using custom YOLO weights.

    License plates are not a built-in COCO class, so this requires a plate
    detector checkpoint configured by NEURON2_PLATE_DETECTION_MODEL.
    """

    result_image_key = "cropped_plate_image"
    detector_name = "license plate"

    async def initialize(self) -> None:
        try:
            await super().initialize()
        except FileNotFoundError:
            if not settings.ML_ALLOW_HEURISTIC_PLATE_FALLBACK:
                raise
            self.model = "heuristic"
            logger.warning(
                "License plate detector weights not found: %s. "
                "Using lower-front crop heuristic fallback.",
                self.model_path,
            )

    async def predict(self, image: Image.Image) -> dict:
        if self.model != "heuristic":
            return await super().predict(image)

        width, height = image.size
        crop_box = _clamp_bbox(
            (
                width * 0.20,
                height * 0.55,
                width * 0.80,
                height * 0.90,
            ),
            width,
            height,
        )
        return {
            self.result_image_key: image.crop(crop_box),
            "bbox": list(crop_box),
            "confidence": 0.0,
        }
