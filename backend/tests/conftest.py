"""
Pytest configuration and fixtures.
"""

import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io

from app.main import app
from app.api.deps import reset_orchestrator


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_dependencies():
    """Reset global dependencies before each test."""
    reset_orchestrator()
    yield
    reset_orchestrator()


@pytest.fixture
def valid_jpeg_image():
    """Generate a valid JPEG image for testing."""
    img = Image.new("RGB", (640, 640), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def valid_png_image():
    """Generate a valid PNG image for testing."""
    img = Image.new("RGB", (640, 640), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def small_image():
    """Generate an image below minimum dimensions."""
    img = Image.new("RGB", (100, 100), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def large_image():
    """Generate an image that exceeds the 5MB file size limit but not dimension limit."""
    # Use PNG format with minimal compression to create a large file
    # 3000x3000 PNG should easily exceed 5MB while staying under 4096px dimension limit
    img = Image.new("RGB", (3000, 3000))
    
    # Fill with pattern to make it less compressible
    import random
    pixels = img.load()
    for x in range(0, img.width, 10):
        for y in range(0, img.height, 10):
            pixels[x, y] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
    
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=0)  # No compression
    buf.seek(0)
    data = buf.getvalue()
    
    # Verify it's actually over 5MB
    assert len(data) > 5 * 1024 * 1024, f"large_image fixture only {len(data)} bytes, expected > 5MB"
    
    return data


@pytest.fixture
def corrupt_image():
    """Generate corrupt image data."""
    return b"This is not a valid image file"
