from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.api.dependencies import get_processing_pipeline
from app.core.config import settings
from app.core.enums import ErrorCode, SearchMode
from app.core.logging import logger
from app.models.schemas import (
    ErrorResponse,
    SearchResponse,
    SearchResults,
    TextSearchRequest,
)
from app.monitoring.metrics import (
    confidence_car,
    confidence_ocr,
    confidence_plate,
    image_size_bytes,
    images_processed_total,
    plate_length,
    search_requests_by_mode_total,
    search_similarity_score,
)
from app.services.description_parser import has_visual_content, parse_description
from app.services.elasticsearch_client import es_client
from app.services.pipeline import (
    normalize_plate_query,
    plate_query_to_elasticsearch_wildcard,
)

router = APIRouter(prefix="/search", tags=["search"])


def _format_hits(hits: list[dict]) -> list[SearchResponse]:
    formatted_results = []
    for hit in hits:
        source = hit.get("_source", {})
        s3_key = source.get("s3_key")
        image_url = f"/api/v1/images/{s3_key}" if s3_key else None

        formatted_results.append(SearchResponse(
            car_id=source.get("car_id"),
            similarity_score=hit.get("_score") or 0.0,
            plate_number=source.get("plate_number"),
            image_url=image_url
        ))
    return formatted_results


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

    Optionally pass `description` (e.g. "белый седан, литые диски") to rank
    candidates by both the photo and the text description. A plate fragment
    inside the description is used when `plate_query` is empty.
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
    description: str = Form(
        "",
        max_length=500,
        description=(
            "Optional text description of the car to combine with the photo, "
            "e.g. 'белая Toyota Camry'."
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

        parsed = parse_description(description)
        if not plate_query.strip() and parsed.plate_fragment:
            plate_query = parsed.plate_fragment
        text_embedding = None
        if has_visual_content(parsed.text):
            if pipeline.text_search_available:
                text_embedding = await pipeline.encode_text(parsed.text)
            else:
                logger.warning(
                    "Description ignored: text search model is not available"
                )

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
        search_kwargs = {}
        if text_embedding:
            search_kwargs = {
                "text_embedding": text_embedding,
                "text_weight": settings.HYBRID_DESCRIPTION_WEIGHT,
            }
        search_result = await es_client.search_by_plate_and_embedding(
            plate_number=plate_filter,
            query_embedding=embedding,
            top_k=settings.SEARCH_TOP_K,
            **search_kwargs,
        )

        # Format results
        formatted_results = _format_hits(search_result.get("hits", []))
        total_found = search_result.get("total", 0)

        # Hybrid scores are a weighted sum of two metrics; keep the drift
        # histogram comparable with its photo-only baseline.
        if formatted_results and not text_embedding:
            search_similarity_score.observe(formatted_results[0].similarity_score)

        mode =SearchMode.IMAGE_AND_TEXT if text_embedding else SearchMode.IMAGE
        search_requests_by_mode_total.labels(mode=mode).inc()

        return SearchResults(
            results=formatted_results,
            detected_plate=plate_number,
            plate_query=normalized_plate_query or None,
            total_found=total_found,
            search_mode=mode,
            description=parsed.original or None,
            text_query=parsed.text if text_embedding else None,
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


@router.post(
    "/text",
    response_model=SearchResults,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Search cars by text description",
    description="""
    Find cars without a photo, by a free-text description in Russian or
    English, e.g. `белая Toyota Camry седан, номер А1**ВС77` or
    `red pickup truck with a roof rack`.

    - A plate fragment in the description (after words like "номер", "г/н",
      "plate", or a token shaped like a Russian plate) becomes a wildcard
      filter; `*`, `?` or `_` mark hidden characters. An explicit
      `plate_query` overrides it.
    - The rest of the description is embedded with a multilingual CLIP text
      encoder and matched against CLIP embeddings of indexed car photos.
    - A description that contains only a plate fragment runs a plate-only
      search.
    """,
)
async def search_by_description(
    request: Request, body: TextSearchRequest
) -> SearchResults:
    parsed = parse_description(body.description)
    plate_query = normalize_plate_query(body.plate_query or parsed.plate_fragment)
    plate_filter = plate_query_to_elasticsearch_wildcard(plate_query)
    has_text = has_visual_content(parsed.text)

    if not has_text and not plate_filter:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": ErrorCode.INVALID_QUERY,
                "message": (
                    "Describe the car (color, make, body type) or give a "
                    "plate fragment."
                ),
            },
        )

    try:
        if has_text:
            pipeline = get_processing_pipeline(request)
            if not pipeline.text_search_available:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error_code": ErrorCode.TEXT_SEARCH_UNAVAILABLE,
                        "message": (
                            "Search by description is unavailable: the text "
                            "model is not loaded."
                        ),
                    },
                )
            text_embedding = await pipeline.encode_text(parsed.text)
            mode = SearchMode.TEXT
            logger.info(
                "Text search: '%s' (plate filter '%s')",
                parsed.text,
                plate_filter or "<none>",
            )
            search_result = await es_client.search_by_text_embedding(
                text_embedding=text_embedding,
                plate_number=plate_filter,
                top_k=settings.SEARCH_TOP_K,
            )
        else:
            mode = SearchMode.PLATE
            logger.info("Plate-only search: '%s'", plate_filter)
            search_result = await es_client.search_by_plate(
                plate_number=plate_filter, top_k=settings.SEARCH_TOP_K
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.ELASTICSEARCH_ERROR,
                "message": "Failed to process text search request.",
                "details": str(e),
            },
        )

    # CLIP text-image scores live on a different scale than photo-to-photo
    # scores, so they are not mixed into search_similarity_score (drift input).
    formatted_results = _format_hits(search_result.get("hits", []))
    search_requests_by_mode_total.labels(mode=mode).inc()

    return SearchResults(
        results=formatted_results,
        detected_plate="",
        plate_query=plate_query or None,
        total_found=search_result.get("total", 0),
        search_mode=mode,
        description=parsed.original,
        text_query=parsed.text if has_text else None,
    )
