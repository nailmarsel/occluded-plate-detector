from typing import Any

from pydantic import BaseModel, Field

from app.core.enums import ErrorCode


class ErrorResponse(BaseModel):
    error_code: ErrorCode | str
    message: str
    details: str | None = None


class HealthResponse(BaseModel):
    status: str
    elasticsearch: bool
    neurons: dict[str, str]


class IndexResponse(BaseModel):
    car_id: str
    plate_number: str
    embedding_dim: int
    status: str


class SearchResponse(BaseModel):
    car_id: str | None = None
    similarity_score: float
    plate_number: str | None = None
    image_url: str | None = None


class SearchResults(BaseModel):
    results: list[SearchResponse]
    detected_plate: str
    plate_query: str | None = None
    total_found: int
    search_mode: str = "image"
    description: str | None = None
    # Part of the description embedded with CLIP (plate fragment removed).
    text_query: str | None = None


class TextSearchRequest(BaseModel):
    description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Free-text description of the car, optionally with a plate "
            "fragment, e.g. 'белая Toyota Camry, номер А1**ВС77'."
        ),
        examples=["белая Toyota Camry седан, номер А1**ВС77"],
    )
    plate_query: str = Field(
        "",
        max_length=20,
        description=(
            "Optional plate fragment; overrides a plate found in the "
            "description. Use * or ? for hidden characters."
        ),
    )


class ClipBackfillResponse(BaseModel):
    processed: int
    updated: int
    failed: int
    remaining: int


class PipelineResult(BaseModel):
    plate_number: str
    embedding: list[float]
    cropped_car_image: Any | None = None
    cropped_plate_image: Any | None = None
    car_bbox: list[int] | None = None
    plate_bbox: list[int] | None = None
