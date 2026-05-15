import io
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.api.dependencies import get_processing_pipeline
from app.core.config import settings
from app.core.enums import ErrorCode
from app.core.logging import logger
from app.models.schemas import ErrorResponse, IndexResponse
from app.monitoring.metrics import (
    confidence_car,
    confidence_ocr,
    confidence_plate,
    image_size_bytes,
    images_processed_total,
    input_errors_total,
    neuron_failures_total,
    plate_length,
)
from app.services.elasticsearch_client import es_client
from app.services.pipeline import ImageProcessingPipeline, normalize_plate_number
from app.services.s3_client import s3_client

router = APIRouter(prefix="/index", tags=["index"])
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_BATCH_FILES = 500
MAX_ZIP_BYTES = 500 * 1024 * 1024


class BatchIndexResult(BaseModel):
    """Result for a single file in batch indexing."""
    car_id: str
    filename: str
    plate_number: str | None = None
    status: str  # "indexed", "failed"
    error: str | None = None


class BatchIndexResponse(BaseModel):
    """Response for batch indexing."""
    total: int
    succeeded: int
    failed: int
    results: list[BatchIndexResult]


class BatchIndexRequest(BaseModel):
    """Request model for batch indexing from a local folder."""
    folder_path: str
    prefix: str | None = None


def _infer_plate_from_zip_path(path: str) -> str:
    zip_path = PurePosixPath(path)
    return normalize_plate_number(zip_path.stem)


async def _process_and_index_single_image(
    image_data: bytes,
    car_id: str,
    pipeline: ImageProcessingPipeline,
    plate_number: str | None = None,
) -> IndexResponse:
    """
    Core logic: process image through the neural pipeline and index it.

    Args:
        image_data: Raw image bytes
        car_id: Unique identifier for the car

    Returns:
        IndexResponse with indexing details
    """
    result = await pipeline.process_for_indexing(
        image_data, plate_number=plate_number
    )

    if "car_confidence" in result:
        confidence_car.observe(result["car_confidence"])
    if "plate_confidence" in result:
        confidence_plate.observe(result["plate_confidence"])
    if "ocr_confidence" in result:
        confidence_ocr.observe(result["ocr_confidence"])

    plate_number = result.get("plate_number", "")
    plate_length.observe(len(plate_number))
    if not plate_number:
        raise RuntimeError("Cannot index a car without a plate number")

    existing_car = await es_client.get_car_by_plate(plate_number)
    old_s3_key = None
    if existing_car:
        car_id = existing_car.get("car_id") or existing_car.get("_id") or car_id
        old_s3_key = existing_car.get("s3_key")
        logger.info("Replacing existing car for plate '%s': %s", plate_number, car_id)

    embedding = result["embedding"]
    cropped_car_image = result["cropped_car_image"]

    logger.info(
        f"Indexing car {car_id} with plate: '{plate_number}', "
        f"embedding dim: {len(embedding)}"
    )

    # Upload cropped car image to S3
    car_image_bytes = io.BytesIO()
    cropped_car_image.save(car_image_bytes, format="JPEG")
    car_image_bytes = car_image_bytes.getvalue()

    s3_key = f"{settings.S3_IMAGE_PREFIX}{car_id}.jpg"
    await s3_client.upload_image(
        image_data=car_image_bytes,
        object_key=s3_key,
        content_type="image/jpeg"
    )

    # Index in Elasticsearch with S3 key
    success = await es_client.index_car(
        car_id=car_id,
        plate_number=plate_number,
        embedding=embedding,
        s3_key=s3_key
    )

    if not success:
        raise RuntimeError("Failed to index car in Elasticsearch")

    if old_s3_key and old_s3_key != s3_key:
        await s3_client.delete_image(old_s3_key)

    return IndexResponse(
        car_id=car_id,
        plate_number=plate_number,
        embedding_dim=len(embedding),
        status="indexed"
    )


@router.post(
    "",
    response_model=IndexResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    },
    summary="Index a car in Elasticsearch",
    description="""
    Upload a car photo to index it in Elasticsearch.
    The system will:
    1. Detect and crop the car
    2. Detect and crop the license plate
    3. Recognize the plate number using OCR
    4. Generate an embedding of the car image
    5. Store the car data (plate number, embedding, S3 key) in Elasticsearch
    """
)
async def index_car(
    request: Request,
    image: UploadFile = File(
        ...,
        description="Car photo to index (JPEG, PNG)"
    ),
    plate_number: str = Form(
        ...,
        description="Known plate number. Required for database indexing.",
    ),
) -> IndexResponse:
    """
    Index a car in Elasticsearch for future search queries.
    """
    try:
        # Validate file type
        if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": ErrorCode.INVALID_IMAGE,
                    "message": "Invalid image format. Only JPEG and PNG are supported."
                }
            )

        # Read image data
        image_data = await image.read()
        if not image_data:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": ErrorCode.INVALID_IMAGE,
                    "message": "Empty image file."
                }
            )

        image_size_bytes.observe(len(image_data))
        images_processed_total.labels(endpoint="/index").inc()

        car_id = str(uuid.uuid4())

        logger.info(f"Indexing car: {car_id}, image: {image.filename}")

        pipeline = get_processing_pipeline(request)
        return await _process_and_index_single_image(
            image_data, car_id, pipeline, plate_number=plate_number
        )

    except HTTPException as e:
        if "Invalid image format" in str(e.detail) or "Empty image" in str(e.detail):
            input_errors_total.labels(reason="invalid_format_or_empty").inc()
        raise
    except Exception as e:
        logger.error(f"Indexing failed for car {car_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.NEURON_ERROR,
                "message": "Failed to index car.",
                "details": str(e)
            }
        )


@router.post(
    "/batch",
    response_model=BatchIndexResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Batch index cars from a local folder",
    description="""
    Provide a local folder path containing car photos.
    The system will process all images (.jpg, .jpeg, .png) in the folder,
    extract plate numbers and embeddings, upload to S3, and index in Elasticsearch.

    Each car gets an auto-generated car_id (UUID) unless a prefix is provided.
    """
)
async def batch_index_cars(
    request: Request,
    batch_request: BatchIndexRequest
) -> BatchIndexResponse:
    """
    Batch index all car photos from a local folder.
    """
    folder = Path(batch_request.folder_path)

    if not folder.exists():
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "INVALID_FOLDER",
                "message": f"Folder does not exist: {batch_request.folder_path}"
            }
        )

    if not folder.is_dir():
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "INVALID_FOLDER",
                "message": f"Path is not a directory: {batch_request.folder_path}"
            }
        )

    image_files = sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "NO_IMAGES",
                "message": f"No images found in: {batch_request.folder_path}"
            }
        )

    logger.info(
        f"Batch indexing {len(image_files)} images from {folder}"
    )

    results: list[BatchIndexResult] = []
    succeeded = 0
    failed = 0

    for i, image_file in enumerate(image_files):
        filename = image_file.name
        prefix = batch_request.prefix or ""
        car_id = f"{prefix}_{i}" if prefix else str(uuid.uuid4())

        try:
            logger.info(
                f"[{i+1}/{len(image_files)}] Processing: {filename} -> {car_id}"
            )

            image_data = image_file.read_bytes()
            if image_data:
                image_size_bytes.observe(len(image_data))
            images_processed_total.labels(endpoint="/index").inc()

            pipeline = get_processing_pipeline(request)
            index_result = await _process_and_index_single_image(
                image_data, car_id, pipeline
            )

            results.append(BatchIndexResult(
                car_id=car_id,
                filename=filename,
                plate_number=index_result.plate_number,
                status="indexed"
            ))
            succeeded += 1

        except Exception as e:
            logger.error(f"[{i+1}/{len(image_files)}] Failed: {filename}: {e}")
            results.append(BatchIndexResult(
                car_id=car_id,
                filename=filename,
                status="failed",
                error=str(e)
            ))
            failed += 1
            neuron_failures_total.labels(stage="batch_processing").inc()

    logger.info(
        f"Batch indexing complete: {succeeded} succeeded, {failed} failed"
    )

    return BatchIndexResponse(
        total=len(image_files),
        succeeded=succeeded,
        failed=failed,
        results=results
    )


@router.post(
    "/batch/zip",
    response_model=BatchIndexResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Batch index cars from a ZIP archive",
    description="""
    Upload a ZIP archive with car photos. Plate numbers are inferred from each
    image filename, for example A864AA199.jpg.
    """
)
async def batch_index_zip(
    request: Request,
    archive: UploadFile = File(..., description="ZIP archive with car photos"),
    prefix: str | None = Form(None),
) -> BatchIndexResponse:
    if archive.content_type not in {
        "application/zip",
        "application/x-zip-compressed",
        "multipart/x-zip",
    } and not (archive.filename or "").lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "INVALID_ARCHIVE",
                "message": "Please upload a ZIP archive.",
            },
        )

    archive_data = await archive.read()
    if not archive_data:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "EMPTY_ARCHIVE",
                "message": "Uploaded ZIP archive is empty.",
            },
        )
    if len(archive_data) > MAX_ZIP_BYTES:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "ARCHIVE_TOO_LARGE",
                "message": "ZIP archive is larger than 500 MB.",
            },
        )

    try:
        zip_archive = zipfile.ZipFile(io.BytesIO(archive_data))
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "INVALID_ARCHIVE",
                "message": "Uploaded file is not a valid ZIP archive.",
            },
        ) from exc

    image_entries = [
        info for info in zip_archive.infolist()
        if not info.is_dir()
        and not info.filename.startswith("__MACOSX/")
        and PurePosixPath(info.filename).suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_entries:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "NO_IMAGES",
                "message": "No JPEG or PNG images found in ZIP archive.",
            },
        )
    if len(image_entries) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "TOO_MANY_IMAGES",
                "message": f"ZIP contains more than {MAX_BATCH_FILES} images.",
            },
        )

    pipeline = get_processing_pipeline(request)
    results: list[BatchIndexResult] = []
    succeeded = 0
    failed = 0

    for i, info in enumerate(image_entries):
        filename = info.filename
        car_id = f"{prefix}_{i}" if prefix else str(uuid.uuid4())

        try:
            plate_number = _infer_plate_from_zip_path(filename)
            if not plate_number:
                raise RuntimeError(
                    "Could not infer plate number from ZIP path. Use "
                    "the plate number as the image filename, e.g. A864AA199.jpg."
                )

            image_data = zip_archive.read(info)
            if image_data:
                image_size_bytes.observe(len(image_data))
            images_processed_total.labels(endpoint="/index").inc()

            index_result = await _process_and_index_single_image(
                image_data,
                car_id,
                pipeline,
                plate_number=plate_number,
            )

            results.append(BatchIndexResult(
                car_id=car_id,
                filename=filename,
                plate_number=index_result.plate_number,
                status="indexed"
            ))
            succeeded += 1
        except Exception as e:
            logger.error("Failed to index ZIP entry %s: %s", filename, e)
            results.append(BatchIndexResult(
                car_id=car_id,
                filename=filename,
                status="failed",
                error=str(e)
            ))
            failed += 1

    return BatchIndexResponse(
        total=len(image_entries),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )
