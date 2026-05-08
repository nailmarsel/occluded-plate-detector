from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.core.logging import logger
from app.core.enums import ErrorCode
from app.models.schemas import SearchResults, SearchResponse, ErrorResponse
from app.services.pipeline import ImageProcessingPipeline
from app.services.elasticsearch_client import es_client
from app.services.s3_client import s3_client
from app.core.config import settings

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
    image: UploadFile = File(
        ...,
        description="Car photo with partially visible license plate (JPEG, PNG)"
    )
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
        if not image_data:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": ErrorCode.INVALID_IMAGE,
                    "message": "Empty image file."
                }
            )

        logger.info(f"Processing search request for image: {image.filename}")

        # Get pipeline instance (should be initialized on startup)
        pipeline = ImageProcessingPipeline()
        # Note: In production, use dependency injection or app state

        # Process image through the pipeline
        result = await pipeline.process_for_search(image_data)

        plate_number = result["plate_number"]
        embedding = result["embedding"]

        logger.info(f"Searching for cars with plate: '{plate_number}'")

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
            image_url = s3_client.get_presigned_url(s3_key) if s3_key else None

            formatted_results.append(SearchResponse(
                car_id=source.get("car_id"),
                similarity_score=hit.get("_score", 0.0),
                plate_number=source.get("plate_number"),
                image_url=image_url
            ))

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
