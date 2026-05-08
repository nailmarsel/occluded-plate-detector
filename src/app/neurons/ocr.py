import re

import numpy as np
from PIL import Image
from app.core.logging import logger
from app.neurons.base import BaseNeuron


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


class OCRNeuron(BaseNeuron):
    """
    Recognizes license plate text from cropped plate images using EasyOCR.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.6):
        self.languages = [
            language.strip()
            for language in model_path.split(",")
            if language.strip()
        ] or ["en"]
        self.confidence_threshold = confidence_threshold
        self.reader = None

    async def initialize(self) -> None:
        """Load EasyOCR reader."""
        try:
            import easyocr
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "easyocr and torch are required for OCR. Install ML "
                "dependencies with: pip install -r requirements-ml.txt"
            ) from exc

        use_gpu = torch.cuda.is_available()
        self.reader = easyocr.Reader(self.languages, gpu=use_gpu)
        logger.info(
            "OCR model initialized for languages: %s on %s",
            self.languages,
            "cuda" if use_gpu else "cpu",
        )

    @staticmethod
    def _normalize_plate(text: str) -> str:
        normalized = text.upper().translate(CYRILLIC_TO_LATIN_PLATE_CHARS)
        return PLATE_CHARS_PATTERN.sub("", normalized)

    async def predict(self, image: Image.Image) -> dict:
        """
        Recognize license plate number from a cropped plate image.
        """
        if self.reader is None:
            raise RuntimeError("OCR model is not initialized")

        try:
            results = self.reader.readtext(np.array(image.convert("RGB")))
            candidates = []

            for _bbox, text, confidence in results:
                normalized_text = self._normalize_plate(text)
                if not normalized_text:
                    continue
                candidates.append(
                    {
                        "text": normalized_text,
                        "raw_text": text,
                        "confidence": float(confidence),
                    }
                )

            if not candidates:
                raise ValueError("No license plate text recognized")

            best = max(candidates, key=lambda item: item["confidence"])
            if best["confidence"] < self.confidence_threshold:
                raise ValueError(
                    "License plate OCR confidence is below threshold: "
                    f"{best['confidence']:.3f}"
                )

            return {
                "plate_number": best["text"],
                "confidence": best["confidence"],
                "raw_text": best["raw_text"],
            }
        except Exception as e:
            logger.error("OCR processing failed: %s", e)
            raise
