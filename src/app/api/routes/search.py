from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.api.dependencies import get_processing_pipeline
from app.core.config import settings
from app.core.enums import ErrorCode
from app.core.logging import logger
from app.models.schemas import ErrorResponse, SearchResponse, SearchResults
from app.services.elasticsearch_client import es_client
from app.monitoring.metrics import (
    images_processed_total,
    input_errors_total,
    neuron_failures_total,
    plate_fallback_total,
    confidence_car,
    confidence_plate,
    confidence_ocr,
    plate_length,
    image_size_bytes,
    search_similarity_score
)

router = APIRouter(prefix="/search", tags=["search"])


@router.post(
    "",
    response_model=SearchResults,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    },
    summary="Search for similar cars",
    description="""
    Upload a car photo with a partially visible license plate.
    The system will:
    1. Detect and crop the car
    2. Detect and crop the license plate
    3. Recognize the plate number using OCR
    4. Generate an embedding of the car image
    5. Search Elasticsearch for the top 5 most similar cars matching the plate number
    """
)
async def search_similar_cars(
    request: Request,
    image: UploadFile = File(
        ...,
        description="Car photo with partially visible license plate (JPEG, PNG)"
    ),
) -> SearchResults:
    """
    Search for similar cars based on uploaded image.
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

        image_size_bytes.observe(len(image_data))
        images_processed_total.labels(endpoint="/search").inc()

        if not image_data:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": ErrorCode.INVALID_IMAGE,
                    "message": "Empty image file."
                }
            )

        logger.info(f"Processing search request for image: {image.filename}")

        # Process image through the pipeline
        pipeline = get_processing_pipeline(request)
        result = await pipeline.process_for_search(image_data)

        plate_number = result["plate_number"]
        embedding = result["embedding"]

        if hasattr(result, "car_confidence"):
            confidence_car.observe(result["car_confidence"])
        if hasattr(result, "plate_confidence"):
            confidence_plate.observe(result["plate_confidence"])
        if hasattr(result, "ocr_confidence"):
            confidence_ocr.observe(result["ocr_confidence"])

        plate_number = result.get("plate_number", "")
        plate_length.observe(len(plate_number))

        if plate_number:
            logger.info(f"Searching for cars with plate: '{plate_number}'")
        else:
            logger.info("Searching for cars by embedding only")

        # Search in Elasticsearch
        search_result = await es_client.search_by_plate_and_embedding(
            plate_number=plate_number,
            query_embedding=embedding,
            top_k=settings.SEARCH_TOP_K
        )

        # Format results
        hits = search_result.get("hits", [])
        total_found = search_result.get("total", 0)

        formatted_results = []
        for hit in hits:
            source = hit.get("_source", {})
            s3_key = source.get("s3_key")
            image_url = f"/api/v1/images/{s3_key}" if s3_key else None

            formatted_results.append(SearchResponse(
                car_id=source.get("car_id"),
                similarity_score=hit.get("_score", 0.0),
                plate_number=source.get("plate_number"),
                image_url=image_url
            ))

        if formatted_results:
            search_similarity_score.observe(formatted_results[0].similarity_score)
    
        return SearchResults(
            results=formatted_results,
            detected_plate=plate_number,
            total_found=total_found
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.NEURON_ERROR,
                "message": "Failed to process search request.",
                "details": str(e)
            }
        )
