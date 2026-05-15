# ruff: noqa: I001
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from app.core.logging import logger
from app.neurons.base import BaseNeuron


PLATE_CHARS_PATTERN = re.compile(r"[^A-Z0-9]")
RUSSIAN_PLATE_LETTERS = "ABEKMHOPCTYX"
RUSSIAN_PLATE_ALLOWLIST = "0123456789ABEKMHOPCTYXАВЕКМНОРСТУХ"
RUSSIAN_PRIVATE_PLATE_PATTERN = re.compile(
    rf"^[{RUSSIAN_PLATE_LETTERS}]\d{{3}}[{RUSSIAN_PLATE_LETTERS}]{{2}}\d{{2,3}}$"
)
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
OCR_CONFUSION_TRANSLATION = str.maketrans(
    {
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "Q": "O",
    }
)


@dataclass(frozen=True)
class PlateCandidate:
    text: str
    raw_text: str
    confidence: float
    is_valid_russian_plate: bool
    source: str = "raw"


class OCRNeuron(BaseNeuron):
    """
    Recognizes license plate text from cropped plate images using EasyOCR.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.6):
        self.model_path = model_path
        self.languages = [
            language.strip()
            for language in model_path.split(",")
            if language.strip()
        ] or ["en"]
        self.confidence_threshold = confidence_threshold
        self.reader = None
        self.model = None
        self.transform = None
        self.device = None
        self.torch = None

    async def initialize(self) -> None:
        """Load custom OCR checkpoint or EasyOCR reader."""
        model_path = Path(self.model_path)
        if model_path.exists() and model_path.suffix == ".pt":
            await self._initialize_custom_model(model_path)
            return

        await self._initialize_easyocr()

    async def _initialize_easyocr(self) -> None:
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

    async def _initialize_custom_model(self, model_path: Path) -> None:
        """Load the CRNN+CTC checkpoint produced by ml/plate_ocr."""
        try:
            import torch
            from torch import nn
            from torchvision import transforms
        except ImportError as exc:
            raise RuntimeError(
                "torch and torchvision are required for custom OCR. "
                "Install ML dependencies with: pip install -r requirements-ml.txt"
            ) from exc

        class PlateOCRModel(nn.Module):
            def __init__(self, num_classes: int = len(RUSSIAN_PLATE_LETTERS) + 11):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d((2, 1), (2, 1)),
                    nn.Conv2d(256, 512, 3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d((2, 1), (2, 1)),
                )
                self.sequence = nn.LSTM(
                    input_size=512 * 3,
                    hidden_size=256,
                    num_layers=2,
                    bidirectional=True,
                    batch_first=True,
                    dropout=0.2,
                )
                self.classifier = nn.Linear(512, num_classes)

            def forward(self, images):
                features = self.features(images)
                batch, channels, height, width = features.shape
                features = features.permute(0, 3, 1, 2).reshape(
                    batch, width, channels * height
                )
                sequence, _ = self.sequence(features)
                return self.classifier(sequence)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PlateOCRModel().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((48, 192)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        self.torch = torch
        logger.info("Custom OCR model initialized: %s on %s", model_path, self.device)

    @staticmethod
    def _decode_custom_logits(logits) -> list[str]:
        alphabet = "0123456789ABEKMHOPCTYX"
        index_to_char = {index + 1: char for index, char in enumerate(alphabet)}
        blank_index = 0
        predictions = logits.argmax(dim=-1).cpu()
        decoded: list[str] = []
        for sequence in predictions:
            chars: list[str] = []
            previous = blank_index
            for index in sequence.tolist():
                if index != blank_index and index != previous:
                    chars.append(index_to_char.get(index, ""))
                previous = index
            decoded.append("".join(chars))
        return decoded

    @staticmethod
    def _normalize_plate(text: str) -> str:
        normalized = text.upper().translate(CYRILLIC_TO_LATIN_PLATE_CHARS)
        return PLATE_CHARS_PATTERN.sub("", normalized)

    @staticmethod
    def _repair_common_ocr_confusions(text: str) -> str:
        if len(text) < 6:
            return text

        chars = list(text)
        expected_digit_positions = {1, 2, 3}
        if len(chars) in {8, 9}:
            expected_digit_positions.update(range(6, len(chars)))

        for index, char in enumerate(chars):
            if index in expected_digit_positions:
                chars[index] = char.translate(OCR_CONFUSION_TRANSLATION)

        return "".join(chars)

    @staticmethod
    def _preprocess_variants(image: Image.Image) -> list[tuple[str, Image.Image]]:
        rgb = image.convert("RGB")
        width, height = rgb.size
        scale = max(2, min(4, round(180 / max(height, 1))))
        upscaled = rgb.resize(
            (max(width * scale, width), max(height * scale, height)),
            Image.Resampling.LANCZOS,
        )

        gray = ImageOps.grayscale(upscaled)
        contrast = ImageOps.autocontrast(gray)
        sharpened = contrast.filter(ImageFilter.SHARPEN)
        threshold = sharpened.point(lambda pixel: 255 if pixel > 150 else 0)

        return [
            ("raw", rgb),
            ("upscaled", upscaled),
            ("contrast", Image.merge("RGB", (contrast, contrast, contrast))),
            ("sharpened", Image.merge("RGB", (sharpened, sharpened, sharpened))),
            ("threshold", Image.merge("RGB", (threshold, threshold, threshold))),
        ]

    @staticmethod
    def _bbox_sort_key(result) -> tuple[float, float]:
        bbox = result[0]
        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]
        return (mean(xs), mean(ys))

    @classmethod
    def _build_candidates(cls, results, source: str = "raw") -> list[PlateCandidate]:
        candidates: list[PlateCandidate] = []
        normalized_parts: list[str] = []
        raw_parts: list[str] = []
        confidences: list[float] = []

        for _bbox, text, confidence in sorted(results, key=cls._bbox_sort_key):
            normalized_text = cls._normalize_plate(text)
            if not normalized_text:
                continue

            confidence = float(confidence)
            normalized_parts.append(normalized_text)
            raw_parts.append(text)
            confidences.append(confidence)
            for candidate_text in {
                normalized_text,
                cls._repair_common_ocr_confusions(normalized_text),
            }:
                candidates.append(
                    PlateCandidate(
                        text=candidate_text,
                        raw_text=text,
                        confidence=confidence,
                        is_valid_russian_plate=bool(
                            RUSSIAN_PRIVATE_PLATE_PATTERN.fullmatch(candidate_text)
                        ),
                        source=source,
                    )
                )

        if len(normalized_parts) > 1:
            combined_text = "".join(normalized_parts)
            for candidate_text in {
                combined_text,
                cls._repair_common_ocr_confusions(combined_text),
            }:
                candidates.append(
                    PlateCandidate(
                        text=candidate_text,
                        raw_text=" ".join(raw_parts),
                        confidence=float(mean(confidences)),
                        is_valid_russian_plate=bool(
                            RUSSIAN_PRIVATE_PLATE_PATTERN.fullmatch(candidate_text)
                        ),
                        source=source,
                    )
                )

        return candidates

    @staticmethod
    def _select_best_candidate(candidates: list[PlateCandidate]) -> PlateCandidate:
        if not candidates:
            raise ValueError("No license plate text recognized")

        return max(
            candidates,
            key=lambda item: (
                item.is_valid_russian_plate,
                len(item.text),
                item.confidence,
            ),
        )

    async def predict(self, image: Image.Image) -> dict:
        """
        Recognize license plate number from a cropped plate image.
        """
        if self.model is not None:
            return await self._predict_custom(image)

        if self.reader is None:
            raise RuntimeError("OCR model is not initialized")

        try:
            all_candidates: list[PlateCandidate] = []
            for source, variant in self._preprocess_variants(image):
                results = self.reader.readtext(
                    np.array(variant),
                    allowlist=RUSSIAN_PLATE_ALLOWLIST,
                    detail=1,
                    paragraph=False,
                )
                all_candidates.extend(self._build_candidates(results, source=source))

            best = self._select_best_candidate(all_candidates)
            min_confidence = (
                min(self.confidence_threshold, 0.35)
                if best.is_valid_russian_plate
                else self.confidence_threshold
            )
            logger.info(
                "OCR best candidate: text='%s', raw='%s', source=%s, "
                "confidence=%.3f, valid_russian=%s",
                best.text,
                best.raw_text,
                best.source,
                best.confidence,
                best.is_valid_russian_plate,
            )
            if best.confidence < min_confidence:
                raise ValueError(
                    "License plate OCR confidence is below threshold: "
                    f"{best.confidence:.3f}"
                )

            return {
                "plate_number": best.text,
                "confidence": best.confidence,
                "raw_text": best.raw_text,
                "is_valid_russian_plate": best.is_valid_russian_plate,
                "ocr_source": best.source,
            }
        except Exception as e:
            logger.error("OCR processing failed: %s", e)
            raise

    async def _predict_custom(self, image: Image.Image) -> dict:
        if (
            self.model is None
            or self.transform is None
            or self.device is None
            or self.torch is None
        ):
            raise RuntimeError("Custom OCR model is not initialized")

        try:
            tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            with self.torch.inference_mode():
                prediction = self._decode_custom_logits(self.model(tensor))[0]
            prediction = self._repair_common_ocr_confusions(
                self._normalize_plate(prediction)
            )
            is_valid = bool(RUSSIAN_PRIVATE_PLATE_PATTERN.fullmatch(prediction))
            if not prediction:
                raise ValueError("No license plate text recognized")
            return {
                "plate_number": prediction,
                "confidence": 1.0,
                "raw_text": prediction,
                "is_valid_russian_plate": is_valid,
                "ocr_source": "custom",
            }
        except Exception as e:
            logger.error("Custom OCR processing failed: %s", e)
            raise
