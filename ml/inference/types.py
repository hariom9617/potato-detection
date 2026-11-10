"""
Common types for inference pipeline.
"""

from enum import Enum
from typing import Optional, Literal
from dataclasses import dataclass


class ModelAvailability(str, Enum):
    """Model availability status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    ERROR = "error"


@dataclass
class InferenceResult:
    """Generic inference result."""
    available: bool
    reason: Optional[str] = None
    model_version: Optional[str] = None
    confidence: Optional[float] = None
    predictions: Optional[dict] = None


@dataclass
class DetectionBox:
    """Single detection bounding box."""
    class_id: int
    class_name: str
    confidence: float
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    severity: Literal["critical", "high", "medium", "low"]
