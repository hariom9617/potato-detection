"""Model status schemas."""

from typing import Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class ModelStatus(str):
    """Model availability status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    ERROR = "error"


class ModelInfo(BaseModel):
    """Information about a single model."""
    
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Human-readable model name")
    type: Literal["classification", "object_detection", "segmentation"] = Field(
        ...,
        description="Model type"
    )
    status: str = Field(..., description="Model status")
    version: Optional[str] = Field(None, description="Model version if available")
    reason: Optional[str] = Field(None, description="Reason if unavailable")
    metrics: Optional[dict[str, float]] = Field(None, description="Model performance metrics")
    trained_date: Optional[str] = Field(None, description="Training completion date")


class ModelsStatusResponse(BaseModel):
    """Response with status of all models."""
    
    models: list[ModelInfo] = Field(..., description="List of model statuses")
    timestamp: datetime = Field(..., description="Response timestamp (UTC)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "models": [
                    {
                        "id": "potato_verifier_v1",
                        "name": "Potato Object Verifier",
                        "type": "classification",
                        "status": "unavailable",
                        "version": None,
                        "reason": "Model not yet trained"
                    },
                    {
                        "id": "external_detector_v1",
                        "name": "External Defect Detector",
                        "type": "object_detection",
                        "status": "unavailable",
                        "version": None,
                        "reason": "Training in progress (Milestone 3)"
                    },
                    {
                        "id": "internal_detector_v1",
                        "name": "Internal Defect Detector (Spectral)",
                        "type": "classification",
                        "status": "unavailable",
                        "version": None,
                        "reason": "Spectral sensor not available, research phase"
                    }
                ],
                "timestamp": "2026-10-18T12:34:56.789Z"
            }
        }
