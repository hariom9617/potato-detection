"""
Model status endpoints.
"""

from fastapi import APIRouter
from datetime import datetime
import logging

from ....schemas.model import ModelsStatusResponse, ModelInfo
from ...deps import get_orchestrator


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/status",
    response_model=ModelsStatusResponse,
    summary="Get Model Status",
    description="List all models and their availability status",
)
async def get_models_status():
    """
    Get status of all AI/ML models.
    
    Returns information about model availability, versions, and reasons
    for unavailability.
    """
    orchestrator = get_orchestrator()
    model_statuses = orchestrator.get_model_statuses()
    
    models = [ModelInfo(**status) for status in model_statuses]
    
    return ModelsStatusResponse(
        models=models,
        timestamp=datetime.utcnow()
    )
