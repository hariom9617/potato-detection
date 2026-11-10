"""Pydantic schemas for request/response validation."""

from .common import (
    ObjectStatus,
    AssessmentStatus,
    OverallStatus,
    Grade,
    BoundingBox,
    Detection,
    ObjectVerificationResult,
    ExternalAssessmentResult,
    InternalAssessmentResult,
    ImageInfo,
)
from .inspection import (
    InspectionResponse,
    InspectionMetadata,
)
from .model import (
    ModelInfo,
    ModelStatus,
    ModelsStatusResponse,
)
from .health import (
    HealthResponse,
    ReadinessResponse,
    ServiceCheck,
)

__all__ = [
    # Common types
    "ObjectStatus",
    "AssessmentStatus",
    "OverallStatus",
    "Grade",
    "BoundingBox",
    "Detection",
    "ObjectVerificationResult",
    "ExternalAssessmentResult",
    "InternalAssessmentResult",
    "ImageInfo",
    # Inspection
    "InspectionResponse",
    "InspectionMetadata",
    # Model
    "ModelInfo",
    "ModelStatus",
    "ModelsStatusResponse",
    # Health
    "HealthResponse",
    "ReadinessResponse",
    "ServiceCheck",
]
