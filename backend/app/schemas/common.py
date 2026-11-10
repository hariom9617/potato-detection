"""
Common schemas and enums used across the API.
"""

from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel, Field


# Enums

class ObjectStatus(str, Enum):
    """Object verification status."""
    POTATO = "POTATO"
    NON_POTATO = "NON_POTATO"
    UNCERTAIN = "UNCERTAIN"


class AssessmentStatus(str, Enum):
    """Assessment result status."""
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    UNCERTAIN = "UNCERTAIN"
    NOT_ASSESSED = "NOT_ASSESSED"


class OverallStatus(str, Enum):
    """Overall inspection status."""
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    PARTIALLY_ASSESSED = "PARTIALLY_ASSESSED"
    UNCERTAIN = "UNCERTAIN"
    REJECTED = "REJECTED"


class Grade(str, Enum):
    """Quality grade."""
    A = "A"  # Premium
    B = "B"  # Acceptable
    C = "C"  # Feed grade
    D = "D"  # Reject


class DefectSeverity(str, Enum):
    """Defect severity classification."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Models

class BoundingBox(BaseModel):
    """Bounding box for object detection."""
    x_min: int = Field(..., description="Top-left X coordinate (pixels)")
    y_min: int = Field(..., description="Top-left Y coordinate (pixels)")
    x_max: int = Field(..., description="Bottom-right X coordinate (pixels)")
    y_max: int = Field(..., description="Bottom-right Y coordinate (pixels)")
    format: Literal["xyxy"] = Field(default="xyxy", description="Coordinate format")
    coordinate_system: Literal["absolute_pixels"] = Field(
        default="absolute_pixels",
        description="Coordinate system"
    )


class Detection(BaseModel):
    """Single object detection result."""
    class_id: int = Field(..., description="Class ID (0-7)")
    class_name: str = Field(..., description="Human-readable class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    severity: DefectSeverity = Field(..., description="Defect severity")


class DefectSummary(BaseModel):
    """Summary of detected defects."""
    total_count: int = Field(..., description="Total number of defects")
    by_severity: dict[str, int] = Field(
        ...,
        description="Count by severity level",
        example={"critical": 1, "high": 0, "medium": 2, "low": 0}
    )


class ObjectVerificationResult(BaseModel):
    """Object verification (potato vs non-potato) result."""
    status: ObjectStatus | Literal["NOT_ASSESSED"] = Field(..., description="Verification status")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    model_version: Optional[str] = Field(None, description="Model version used")
    reason: Optional[str] = Field(None, description="Reason if NOT_ASSESSED")


class ExternalAssessmentResult(BaseModel):
    """External defect detection assessment result."""
    status: AssessmentStatus = Field(..., description="Assessment status")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall confidence")
    model_version: Optional[str] = Field(None, description="Model version used")
    reason: Optional[str] = Field(None, description="Reason if NOT_ASSESSED")
    detections: list[Detection] = Field(default_factory=list, description="List of detections")
    defect_summary: Optional[DefectSummary] = Field(None, description="Defect summary statistics")


class InternalAssessmentResult(BaseModel):
    """Internal defect (spectral) assessment result."""
    status: AssessmentStatus = Field(..., description="Assessment status")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    model_version: Optional[str] = Field(None, description="Model version used")
    reason: Optional[str] = Field(None, description="Reason if NOT_ASSESSED")


class ImageInfo(BaseModel):
    """Uploaded image information."""
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    format: Literal["JPEG", "PNG"] = Field(..., description="Image format")
    size_bytes: int = Field(..., description="File size in bytes")
