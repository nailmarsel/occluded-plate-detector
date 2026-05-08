from fastapi import Request

from app.services.pipeline import ImageProcessingPipeline


def get_processing_pipeline(request: Request) -> ImageProcessingPipeline:
    pipeline = getattr(request.app.state, "processing_pipeline", None)
    if pipeline is None:
        raise RuntimeError("Image processing pipeline is not initialized")
    return pipeline
