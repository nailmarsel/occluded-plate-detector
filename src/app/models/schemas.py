from typing import Any

from pydantic import BaseModel

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
    total_found: int


class PipelineResult(BaseModel):
    plate_number: str
    embedding: list[float]
    cropped_car_image: Any | None = None
    cropped_plate_image: Any | None = None
    car_bbox: list[int] | None = None
    plate_bbox: list[int] | None = None
