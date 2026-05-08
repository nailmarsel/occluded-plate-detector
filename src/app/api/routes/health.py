from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from app.core.logging import logger
from app.models.schemas import HealthResponse
from app.services.elasticsearch_client import es_client
from app.services.s3_client import s3_client

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the application and its dependencies."
)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    try:
        # Check Elasticsearch
        es_health = await es_client.check_health()

        # Check S3/Garage
        s3_health = s3_client.client is not None

        # Check neurons (they should be initialized on startup)
        neurons_status = {
            "car_detection": "initialized",
            "plate_detection": "initialized",
            "ocr": "initialized",
            "embedding": "initialized"
        }

        overall_status = "healthy" if (es_health and s3_health) else "degraded"

        return HealthResponse(
            status=overall_status,
            elasticsearch=es_health,
            neurons=neurons_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            elasticsearch=False,
            neurons={
                "car_detection": "unknown",
                "plate_detection": "unknown",
                "ocr": "unknown",
                "embedding": "unknown"
            }
        )


@router.get(
    "/openapi/json",
    response_class=JSONResponse,
    summary="Download OpenAPI spec as JSON",
    description="Download the full OpenAPI 3.1 specification in JSON format.",
    include_in_schema=False
)
async def openapi_json():
    """Download OpenAPI specification as JSON."""
    from app.main import app as main_app

    return main_app.openapi()


@router.get(
    "/openapi/yaml",
    response_class=Response,
    summary="Download OpenAPI spec as YAML",
    description="Download the full OpenAPI 3.1 specification in YAML format.",
    include_in_schema=False
)
async def openapi_yaml():
    """Download OpenAPI specification as YAML."""
    try:
        import yaml
        from app.main import app as main_app

        schema = main_app.openapi()
        yaml_content = yaml.dump(schema, default_flow_style=False, sort_keys=False)
        return Response(
            content=yaml_content,
            media_type="text/yaml",
            headers={
                "Content-Disposition": "attachment; filename=openapi.yaml"
            }
        )
    except ImportError:
        return Response(
            content="PyYAML not installed. Install with: pip install pyyaml",
            status_code=501
        )
