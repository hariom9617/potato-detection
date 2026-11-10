"""
Potato object verification (potato vs non-potato).
Currently unavailable - returns NOT_ASSESSED.
"""

import logging
from typing import Optional
from PIL import Image

from .types import InferenceResult, ModelAvailability


logger = logging.getLogger(__name__)


class PotatoVerifier:
    """
    Verifies whether an image contains a potato.
    
    Status: Model not yet trained.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize potato verifier.
        
        Args:
            model_path: Path to trained model weights (if available)
            device: Device for inference ("cpu" or "cuda")
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.availability = ModelAvailability.UNAVAILABLE
        self.unavailable_reason = "Potato verifier model not yet trained"
        
        logger.info(
            f"PotatoVerifier initialized: model_path={model_path}, "
            f"availability={self.availability.value}"
        )
    
    def is_available(self) -> bool:
        """Check if model is available for inference."""
        return self.availability == ModelAvailability.AVAILABLE
    
    def get_status(self) -> dict:
        """Get model status information."""
        return {
            "id": "potato_verifier_v1",
            "name": "Potato Object Verifier",
            "type": "classification",
            "status": self.availability.value,
            "version": None,
            "reason": self.unavailable_reason if not self.is_available() else None,
        }
    
    def verify(self, image: Image.Image) -> InferenceResult:
        """
        Verify if image contains a potato.
        
        Args:
            image: PIL Image to verify
        
        Returns:
            InferenceResult with verification status
        """
        if not self.is_available():
            logger.debug("Potato verification requested but model unavailable")
            return InferenceResult(
                available=False,
                reason=self.unavailable_reason,
                model_version=None,
                confidence=None,
                predictions=None,
            )
        
        # Future: Implement actual inference
        # For now, this code path is never reached
        raise NotImplementedError("Potato verifier model not yet trained")
