"""Health check schemas."""

from typing import Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ServiceCheck(BaseModel):
    """Health check for a single service."""
    status: Literal["available", "unavailable", "degraded", "partially_available"] = Field(
        ...,
        description="Service status"
    )
    latency_ms: Optional[int] = Field(None, description="Latency in milliseconds")
    details: Optional[dict] = Field(None, description="Additional details")


class ReadinessResponse(BaseModel):
    """Readiness check response."""
    status: Literal["ready", "not_ready"] = Field(..., description="Overall readiness status")
    checks: dict[str, ServiceCheck] = Field(..., description="Individual service checks")
    timestamp: datetime = Field(..., description="Check timestamp (UTC)")


class HealthResponse(BaseModel):
    """Basic health check response."""
    status: Literal["healthy"] = Field(default="healthy", description="Service health")
    timestamp: datetime = Field(..., description="Check timestamp (UTC)")
    version: str = Field(..., description="API version")
