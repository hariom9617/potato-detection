"""
FastAPI dependencies for dependency injection.
"""

import logging
from typing import Optional
from functools import lru_cache

# Import ML inference orchestrator
import sys
from pathlib import Path

# Add ml directory to path
ml_dir = Path(__file__).parent.parent.parent.parent / "ml"
if str(ml_dir) not in sys.path:
    sys.path.insert(0, str(ml_dir))

from inference.orchestrator import InspectionOrchestrator
from ..config import settings


logger = logging.getLogger(__name__)


# Global orchestrator instance
_orchestrator: Optional[InspectionOrchestrator] = None


@lru_cache()
def get_orchestrator() -> InspectionOrchestrator:
    """
    Get or create inspection orchestrator singleton.
    
    Returns:
        InspectionOrchestrator instance
    """
    global _orchestrator
    
    if _orchestrator is None:
        logger.info("Initializing InspectionOrchestrator")
        _orchestrator = InspectionOrchestrator(
            model_path=settings.model_path,
            device=settings.device,
            confidence_threshold=settings.confidence_threshold,
            detection_threshold=settings.detection_threshold,
        )
        logger.info("InspectionOrchestrator initialized successfully")
    
    return _orchestrator


def reset_orchestrator() -> None:
    """Reset orchestrator (for testing)."""
    global _orchestrator
    _orchestrator = None
    get_orchestrator.cache_clear()
