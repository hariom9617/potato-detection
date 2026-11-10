"""
Custom exception classes and error handling.
"""

from typing import Any, Optional
from datetime import datetime


class AppException(Exception):
    """Base exception for all application errors."""
    
    def __init__(
        self,
        message: str,
        code: str,
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        super().__init__(self.message)
    
    def to_dict(self, request_id: Optional[str] = None) -> dict[str, Any]:
        """Convert exception to error response dictionary."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "request_id": request_id,
                "timestamp": self.timestamp,
            }
        }


class ImageValidationError(AppException):
    """Image validation failures."""
    
    def __init__(self, message: str, code: str, details: Optional[dict[str, Any]] = None):
        # Map error codes to appropriate HTTP status
        status_map = {
            "MISSING_FILE": 400,
            "INVALID_IMAGE_FORMAT": 400,
            "FILE_TOO_LARGE": 413,
            "IMAGE_TOO_SMALL": 422,
            "IMAGE_TOO_LARGE": 422,
            "IMAGE_CORRUPT": 422,
            "INVALID_CONTENT_TYPE": 400,
        }
        status_code = status_map.get(code, 400)
        super().__init__(message, code, status_code, details)


class ModelNotAvailableError(AppException):
    """Model is not loaded or unavailable."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, "MODEL_NOT_AVAILABLE", 503, details)


class InspectionNotFoundError(AppException):
    """Inspection ID does not exist."""
    
    def __init__(self, inspection_id: str):
        super().__init__(
            "No inspection found with the given ID",
            "INSPECTION_NOT_FOUND",
            404,
            {"inspection_id": inspection_id}
        )


class DatabaseError(AppException):
    """Database operation failures."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, "DATABASE_ERROR", 500, details)


class ValidationError(AppException):
    """Request validation failures."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", 422, details)
