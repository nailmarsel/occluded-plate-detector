from fastapi import APIRouter
from app.api.routes import search, index, health

# Create main router that includes all route modules
api_router = APIRouter()

# Include all route modules
api_router.include_router(search.router)
api_router.include_router(index.router)
api_router.include_router(health.router)

__all__ = ["api_router"]
