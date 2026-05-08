# ruff: noqa: I001
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router
from app.core.config import settings
from app.core.logging import logger
from app.services.elasticsearch_client import es_client
from app.services.pipeline import ImageProcessingPipeline
from app.services.s3_client import s3_client


# Global pipeline instance
processing_pipeline: ImageProcessingPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    logger.info("Starting AutobahnCV application...")

    try:
        # Connect to S3/Garage
        s3_client.connect()

        # Connect to Elasticsearch
        await es_client.connect()

        # Initialize image processing pipeline (neural networks)
        global processing_pipeline
        processing_pipeline = ImageProcessingPipeline()
        await processing_pipeline.initialize()
        app.state.processing_pipeline = processing_pipeline
    except Exception:
        await es_client.close()
        s3_client.close()
        raise

    logger.info("AutobahnCV application started successfully")

    yield

    # Shutdown
    logger.info("Shutting down AutobahnCV application...")
    await es_client.close()
    s3_client.close()
    logger.info("AutobahnCV application shut down complete")


def custom_openapi_schema(app: FastAPI) -> dict:
    """Customize the OpenAPI schema with additional metadata."""
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add contact and license info
    openapi_schema["info"]["contact"] = {
        "name": "AutobahnCV Team",
        "url": "https://github.com/your-org/autobahncv",
    }
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }

    # Add external documentation reference
    openapi_schema["externalDocs"] = {
        "description": "Full project documentation",
        "url": "https://github.com/your-org/autobahncv/blob/main/src/README.md",
    }

    # Add tags metadata for better grouping in Swagger/ReDoc
    openapi_schema["tags"] = [
        {
            "name": "search",
            "description": "Search for similar cars by uploading a photo",
        },
        {
            "name": "index",
            "description": "Index cars into the system (single or batch)",
        },
        {
            "name": "health",
            "description": "Health check and system status",
        },
    ]

    # Add security scheme placeholder (for future auth implementation)
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Bearer token for API authentication (not yet enforced)",
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title=settings.APP_NAME,
        description="""
# AutobahnCV - Car License Plate Search System

This system allows you to:
- **Search** for similar cars by uploading a photo with a partially visible
  license plate
- **Index** new cars into the system for future search queries

## How it works:
1. Upload a car photo
2. The system detects and crops the car (YOLO v8)
3. Detects and crops the license plate (YOLO v8)
4. Recognizes the plate number (OCR)
5. Generates a car image embedding (ResNet-108)
6. Searches Elasticsearch for similar cars matching the plate number

## API Documentation
- **Swagger UI**: `/docs` — Interactive API explorer
- **ReDoc**: `/redoc` — Static API documentation
- **OpenAPI JSON**: `/openapi.json` — Raw OpenAPI 3.1 specification
- **Download Spec**: `/api/v1/openapi.yaml` — Download as YAML

## Authentication
Currently no authentication is required. Bearer token support is planned for
future releases.
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_version="3.1.0",
        lifespan=lifespan,
        terms_of_service="https://github.com/your-org/autobahncv",
    )

    # Custom OpenAPI schema
    app.openapi = lambda: custom_openapi_schema(app)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router, prefix=settings.API_PREFIX)

    return app


app = create_application()
