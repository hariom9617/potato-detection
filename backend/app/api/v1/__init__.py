"""API v1 routes."""

from fastapi import APIRouter

from .routes import health, models, inspection


api_router = APIRouter()

# Include route modules
api_router.include_router(health.router, tags=["health"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(inspection.router, prefix="/inspection", tags=["inspection"])
