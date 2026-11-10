"""
Image validation service.
Validates uploaded images for format, size, and integrity.
"""

import io
import imghdr
from typing import Optional, Literal
from dataclasses import dataclass
from PIL import Image
import logging

from ..core.errors import ImageValidationError
from ..config import settings


logger = logging.getLogger(__name__)


@dataclass
class ValidatedImage:
    """Validated image data."""
    image: Image.Image
    format: Literal["JPEG", "PNG"]
    width: int
    height: int
    size_bytes: int


class ImageValidator:
    """
    Validates uploaded images.
    
    Requirements:
    - Only JPEG and PNG formats
    - Validate actual image decoding
    - Enforce size and dimension limits
    - Reject corrupt images
    - Do not trust client-provided metadata
    """
    
    def __init__(
        self,
        max_size_bytes: Optional[int] = None,
        min_dimension: Optional[int] = None,
        max_dimension: Optional[int] = None,
        allowed_formats: Optional[list[str]] = None,
    ):
        """
        Initialize image validator.
        
        Args:
            max_size_bytes: Maximum file size in bytes
            min_dimension: Minimum width/height in pixels
            max_dimension: Maximum width/height in pixels
            allowed_formats: List of allowed formats (default: ["JPEG", "PNG"])
        """
        self.max_size_bytes = max_size_bytes or settings.max_upload_size_bytes
        self.min_dimension = min_dimension or settings.min_image_dimension
        self.max_dimension = max_dimension or settings.max_image_dimension
        self.allowed_formats = allowed_formats or settings.allowed_image_formats
    
    def validate(self, file_data: bytes, filename: Optional[str] = None) -> ValidatedImage:
        """
        Validate uploaded image file.
        
        Args:
            file_data: Raw image file bytes
            filename: Original filename (not trusted, for logging only)
        
        Returns:
            ValidatedImage with image object and metadata
        
        Raises:
            ImageValidationError: If validation fails
        """
        # 1. Check file size
        size_bytes = len(file_data)
        if size_bytes == 0:
            raise ImageValidationError(
                "Uploaded file is empty",
                "MISSING_FILE",
                {"size_bytes": 0}
            )
        
        if size_bytes > self.max_size_bytes:
            raise ImageValidationError(
                "Uploaded file exceeds maximum size",
                "FILE_TOO_LARGE",
                {
                    "file_size_bytes": size_bytes,
                    "max_size_bytes": self.max_size_bytes,
                    "max_size_mb": self.max_size_bytes / (1024 * 1024)
                }
            )
        
        # 2. Detect actual image format using magic bytes
        format_detected = self._detect_format(file_data)
        
        if format_detected not in self.allowed_formats:
            raise ImageValidationError(
                "Unsupported image format",
                "INVALID_IMAGE_FORMAT",
                {
                    "format_detected": format_detected or "unknown",
                    "supported_formats": self.allowed_formats
                }
            )
        
        # 3. Attempt to decode image
        try:
            image = Image.open(io.BytesIO(file_data))
            image.load()  # Force loading to detect truncated/corrupt images
        except Exception as e:
            logger.warning(f"Failed to decode image '{filename}': {e}")
            raise ImageValidationError(
                "Image file is corrupt or cannot be decoded",
                "IMAGE_CORRUPT",
                {
                    "decode_error": str(e),
                    "filename": filename
                }
            )
        
        # 4. Validate dimensions
        width, height = image.size
        
        if width < self.min_dimension or height < self.min_dimension:
            raise ImageValidationError(
                "Image dimensions below minimum requirement",
                "IMAGE_TOO_SMALL",
                {
                    "width": width,
                    "height": height,
                    "min_dimension": self.min_dimension
                }
            )
        
        if width > self.max_dimension or height > self.max_dimension:
            raise ImageValidationError(
                "Image dimensions exceed maximum limit",
                "IMAGE_TOO_LARGE",
                {
                    "width": width,
                    "height": height,
                    "max_dimension": self.max_dimension
                }
            )
        
        logger.info(
            f"Image validated successfully: {format_detected}, "
            f"{width}x{height}, {size_bytes} bytes"
        )
        
        return ValidatedImage(
            image=image,
            format=format_detected,  # type: ignore
            width=width,
            height=height,
            size_bytes=size_bytes,
        )
    
    def _detect_format(self, file_data: bytes) -> Optional[str]:
        """
        Detect image format from binary data using magic bytes.
        
        Args:
            file_data: Raw file bytes
        
        Returns:
            Format string ("JPEG" or "PNG") or None if unknown
        """
        # Use imghdr to detect format from binary signature
        format_detected = imghdr.what(None, h=file_data)
        
        if format_detected == "jpeg":
            return "JPEG"
        elif format_detected == "png":
            return "PNG"
        else:
            return format_detected.upper() if format_detected else None
