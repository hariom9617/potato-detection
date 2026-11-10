"""
Tests for image validation service.
"""

import pytest
from app.services.image_validator import ImageValidator
from app.core.errors import ImageValidationError


def test_validate_jpeg_image(valid_jpeg_image):
    """Test validation of valid JPEG image."""
    validator = ImageValidator()
    result = validator.validate(valid_jpeg_image)
    
    assert result.format == "JPEG"
    assert result.width == 640
    assert result.height == 640
    assert result.size_bytes > 0


def test_validate_png_image(valid_png_image):
    """Test validation of valid PNG image."""
    validator = ImageValidator()
    result = validator.validate(valid_png_image)
    
    assert result.format == "PNG"
    assert result.width == 640
    assert result.height == 640
    assert result.size_bytes > 0


def test_reject_empty_file():
    """Test rejection of empty file."""
    validator = ImageValidator()
    
    with pytest.raises(ImageValidationError) as exc_info:
        validator.validate(b"")
    
    assert exc_info.value.code == "MISSING_FILE"
    assert exc_info.value.status_code == 400


def test_reject_file_too_large(large_image):
    """Test rejection of oversized file."""
    validator = ImageValidator()
    
    with pytest.raises(ImageValidationError) as exc_info:
        validator.validate(large_image)
    
    assert exc_info.value.code == "FILE_TOO_LARGE"
    assert exc_info.value.status_code == 413
    assert "max_size_mb" in exc_info.value.details


def test_reject_corrupt_image(corrupt_image):
    """Test rejection of corrupt image data."""
    validator = ImageValidator()
    
    with pytest.raises(ImageValidationError) as exc_info:
        validator.validate(corrupt_image)
    
    assert exc_info.value.code in ["IMAGE_CORRUPT", "INVALID_IMAGE_FORMAT"]


def test_reject_small_image(small_image):
    """Test rejection of image below minimum dimensions."""
    validator = ImageValidator()
    
    with pytest.raises(ImageValidationError) as exc_info:
        validator.validate(small_image)
    
    assert exc_info.value.code == "IMAGE_TOO_SMALL"
    assert exc_info.value.status_code == 422
    assert "min_dimension" in exc_info.value.details


def test_custom_size_limits():
    """Test validator with custom size limits."""
    validator = ImageValidator(
        max_size_bytes=1000,  # Very small limit
        min_dimension=500,
    )
    
    # Small valid image should fail size limit
    img_data = b"x" * 2000
    
    with pytest.raises(ImageValidationError) as exc_info:
        validator.validate(img_data)
    
    assert exc_info.value.code in ["FILE_TOO_LARGE", "IMAGE_CORRUPT", "INVALID_IMAGE_FORMAT"]
