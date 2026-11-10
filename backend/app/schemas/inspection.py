"""Inspection request/response schemas."""

from typing import Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

from .common import (
    ImageInfo,
    ObjectVerificationResult,
    ExternalAssessmentResult,
    InternalAssessmentResult,
    OverallStatus,
    Grade,
)


class InspectionMetadata(BaseModel):
    """Optional metadata for inspection."""
    source: Optional[str] = Field(None, description="Source system or app")
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    timestamp: Optional[datetime] = Field(None, description="Client-side timestamp")
    custom_fields: Optional[dict[str, Any]] = Field(None, description="Custom metadata")


class InspectionResponse(BaseModel):
    """Complete inspection result response."""
    
    inspection_id: str = Field(..., description="Unique inspection identifier")
    timestamp: datetime = Field(..., description="Server timestamp (UTC)")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    
    image: ImageInfo = Field(..., description="Image information")
    
    object_verification: ObjectVerificationResult = Field(
        ...,
        description="Object verification result (potato vs non-potato)"
    )
    
    external_assessment: ExternalAssessmentResult = Field(
        ...,
        description="External defect assessment result"
    )
    
    internal_assessment: InternalAssessmentResult = Field(
        ...,
        description="Internal defect assessment result (spectral)"
    )
    
    overall_status: OverallStatus = Field(..., description="Overall inspection status")
    grade: Optional[Grade] = Field(None, description="Quality grade (A/B/C/D)")
    explanation: str = Field(..., description="Human-readable explanation")
    
    metadata: Optional[dict[str, Any]] = Field(None, description="User-provided metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "inspection_id": "insp_abc123def456",
                "timestamp": "2026-10-18T12:34:56.789Z",
                "processing_time_ms": 245,
                "image": {
                    "width": 1280,
                    "height": 1280,
                    "format": "JPEG",
                    "size_bytes": 245678
                },
                "object_verification": {
                    "status": "NOT_ASSESSED",
                    "confidence": None,
                    "reason": "Potato verifier model not yet trained"
                },
                "external_assessment": {
                    "status": "NOT_ASSESSED",
                    "confidence": None,
                    "reason": "External defect detector model not yet trained",
                    "detections": []
                },
                "internal_assessment": {
                    "status": "NOT_ASSESSED",
                    "confidence": None,
                    "reason": "Spectral sensor not available"
                },
                "overall_status": "PARTIALLY_ASSESSED",
                "grade": None,
                "explanation": "Inspection incomplete: no models available for assessment"
            }
        }
