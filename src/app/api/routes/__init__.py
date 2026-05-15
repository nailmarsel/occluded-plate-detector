from fastapi import APIRouter

from app.api.routes import feedback, health, images, index, search

api_router = APIRouter()

# Include all route modules
api_router.include_router(search.router)
api_router.include_router(index.router)
api_router.include_router(images.router)
api_router.include_router(health.router)
api_router.include_router(feedback.router)

__all__ = ["api_router"]
