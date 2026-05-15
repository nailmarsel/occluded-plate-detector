from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.api.dependencies import get_processing_pipeline
from app.core.config import settings
from app.core.enums import ErrorCode
from app.core.logging import logger
from app.models.schemas import ErrorResponse, SearchResponse, SearchResults
from app.monitoring.metrics import (
    confidence_car,
    confidence_ocr,
    confidence_plate,
    image_size_bytes,
    images_processed_total,
    plate_length,
    search_similarity_score,
)
from app.services.elasticsearch_client import es_client
from app.services.pipeline import (
    normalize_plate_query,
    plate_query_to_elasticsearch_wildcard,
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
    plate_query: str = Form(
        "",
        description=(
            "Optional visible plate fragment. Use * or ? for each hidden "
            "character, e.g. A8**AA977."
        ),
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

        plate_number = result.get("plate_number", "")
        embedding = result["embedding"]
        normalized_plate_query = normalize_plate_query(plate_query)
        plate_filter = (
            plate_query_to_elasticsearch_wildcard(normalized_plate_query)
            if normalized_plate_query
            else plate_query_to_elasticsearch_wildcard(plate_number)
        )

        if "car_confidence" in result:
            confidence_car.observe(result["car_confidence"])
        if "plate_confidence" in result:
            confidence_plate.observe(result["plate_confidence"])
        if "ocr_confidence" in result:
            confidence_ocr.observe(result["ocr_confidence"])

        plate_length.observe(len(plate_number))

        if normalized_plate_query:
            logger.info(
                "Searching for cars with manual plate query '%s' "
                "(OCR detected '%s')",
                normalized_plate_query,
                plate_number or "<none>",
            )
        elif plate_number:
            logger.info(f"Searching for cars with OCR plate: '{plate_number}'")
        else:
            logger.info("Searching for cars by embedding only")

        # Search in Elasticsearch
        search_result = await es_client.search_by_plate_and_embedding(
            plate_number=plate_filter,
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
            plate_query=normalized_plate_query or None,
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
