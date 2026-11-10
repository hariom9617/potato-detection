"""
Tests for inspection endpoints.
"""

import pytest
import json


def test_create_inspection_valid_jpeg(client, valid_jpeg_image):
    """Test inspection with valid JPEG image."""
    response = client.post(
        "/api/v1/inspection",
        files={"file": ("test.jpg", valid_jpeg_image, "image/jpeg")}
    )
    
    assert response.status_code == 200
    
    data = response.json()
    
    # Verify response structure
    assert "inspection_id" in data
    assert data["inspection_id"].startswith("insp_")
    assert "timestamp" in data
    assert "processing_time_ms" in data
    
    # Verify image info
    assert data["image"]["format"] == "JPEG"
    assert data["image"]["width"] == 640
    assert data["image"]["height"] == 640
    
    # Verify assessments (all should be NOT_ASSESSED without models)
    assert data["object_verification"]["status"] == "NOT_ASSESSED"
    assert data["external_assessment"]["status"] == "NOT_ASSESSED"
    assert data["internal_assessment"]["status"] == "NOT_ASSESSED"
    
    # Overall status should be PARTIALLY_ASSESSED
    assert data["overall_status"] == "PARTIALLY_ASSESSED"
    assert data["grade"] is None


def test_create_inspection_valid_png(client, valid_png_image):
    """Test inspection with valid PNG image."""
    response = client.post(
        "/api/v1/inspection",
        files={"file": ("test.png", valid_png_image, "image/png")}
    )
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["image"]["format"] == "PNG"


def test_create_inspection_with_metadata(client, valid_jpeg_image):
    """Test inspection with metadata."""
    metadata = {
        "source": "mobile_app",
        "batch_id": "test_batch_001",
    }
    
    response = client.post(
        "/api/v1/inspection",
        files={"file": ("test.jpg", valid_jpeg_image, "image/jpeg")},
        data={"metadata": json.dumps(metadata)}
    )
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["metadata"] == metadata


def test_create_inspection_missing_file(client):
    """Test inspection without file upload."""
    response = client.post("/api/v1/inspection")
    
    assert response.status_code == 422  # FastAPI validation error


def test_create_inspection_invalid_format(client, corrupt_image):
    """Test inspection with invalid image format."""
    response = client.post(
        "/api/v1/inspection",
        files={"file": ("test.txt", corrupt_image, "text/plain")}
    )
    
    assert response.status_code in [400, 422]
    
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] in ["IMAGE_CORRUPT", "INVALID_IMAGE_FORMAT"]


def test_create_inspection_file_too_large(client, large_image):
    """Test inspection with oversized image."""
    response = client.post(
        "/api/v1/inspection",
        files={"file": ("large.jpg", large_image, "image/jpeg")}
    )
    
    assert response.status_code == 413
    
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "FILE_TOO_LARGE"


def test_create_inspection_image_too_small(client, small_image):
    """Test inspection with image below minimum dimensions."""
    response = client.post(
        "/api/v1/inspection",
        files={"file": ("small.jpg", small_image, "image/jpeg")}
    )
    
    assert response.status_code == 422
    
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "IMAGE_TOO_SMALL"


def test_get_inspection_not_found(client):
    """Test retrieving non-existent inspection."""
    response = client.get("/api/v1/inspection/insp_nonexistent")
    
    assert response.status_code == 404
    
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "INSPECTION_NOT_FOUND"


def test_inspection_response_schema(client, valid_jpeg_image):
    """Test that inspection response matches expected schema."""
    response = client.post(
        "/api/v1/inspection",
        files={"file": ("test.jpg", valid_jpeg_image, "image/jpeg")}
    )
    
    assert response.status_code == 200
    
    data = response.json()
    
    # Required fields
    required_fields = [
        "inspection_id", "timestamp", "processing_time_ms",
        "image", "object_verification", "external_assessment",
        "internal_assessment", "overall_status", "explanation"
    ]
    
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    
    # Image fields
    image_fields = ["width", "height", "format", "size_bytes"]
    for field in image_fields:
        assert field in data["image"]
    
    # Assessment fields
    for assessment_key in ["object_verification", "external_assessment", "internal_assessment"]:
        assessment = data[assessment_key]
        assert "status" in assessment
        assert "confidence" in assessment or assessment["confidence"] is None
        assert "reason" in assessment or assessment["reason"] is None
