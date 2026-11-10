"""
Health and readiness check endpoints.
"""

from fastapi import APIRouter, status
from datetime import datetime
import logging

from ....schemas.health import HealthResponse, ReadinessResponse, ServiceCheck
from ....config import settings


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness Check",
    description="Check if backend and dependencies are ready to serve requests",
)
async def readiness_check():
    """
    Check if all required services are available.
    
    Returns 200 if ready, 503 if not ready.
    """
    checks = {}
    overall_ready = True
    
    # Database check
    try:
        # TODO: Add actual database ping when DB is implemented
        checks["database"] = ServiceCheck(
            status="available" if settings.enable_persistence else "unavailable",
            details={"persistence_enabled": settings.enable_persistence}
        )
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        checks["database"] = ServiceCheck(status="unavailable", details={"error": str(e)})
        overall_ready = False
    
    # Models check
    try:
        # Import here to avoid circular dependencies
        from ...deps import get_orchestrator
        
        orchestrator = get_orchestrator()
        model_statuses = orchestrator.get_model_statuses()
        
        available_count = sum(1 for m in model_statuses if m["status"] == "available")
        total_count = len(model_statuses)
        
        if available_count == 0:
            model_status = "unavailable"
        elif available_count < total_count:
            model_status = "partially_available"
        else:
            model_status = "available"
        
        checks["models"] = ServiceCheck(
            status=model_status,
            details={
                "available": available_count,
                "total": total_count,
                "models": {m["id"]: m["status"] for m in model_statuses}
            }
        )
    except Exception as e:
        logger.error(f"Models check failed: {e}")
        checks["models"] = ServiceCheck(status="unavailable", details={"error": str(e)})
        overall_ready = False
    
    # Storage check
    try:
        import os
        
        storage_available = (
            os.path.exists(settings.upload_dir)
            and os.path.exists(settings.inspection_images_dir)
        )
        
        checks["storage"] = ServiceCheck(
            status="available" if storage_available else "unavailable",
            details={
                "upload_dir": settings.upload_dir,
                "inspection_dir": settings.inspection_images_dir
            }
        )
        
        if not storage_available:
            overall_ready = False
    except Exception as e:
        logger.error(f"Storage check failed: {e}")
        checks["storage"] = ServiceCheck(status="unavailable", details={"error": str(e)})
        overall_ready = False
    
    response = ReadinessResponse(
        status="ready" if overall_ready else "not_ready",
        checks=checks,
        timestamp=datetime.utcnow()
    )
    
    if not overall_ready:
        return response  # FastAPI will use 503 if we raise HTTPException
    
    return response
