"""
Tests for health check endpoints.
"""

import pytest


def test_health_endpoint(client):
    """Test basic health endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


def test_readiness_endpoint(client):
    """Test readiness check endpoint."""
    response = client.get("/api/v1/health/ready")
    
    # Should return 200 even if models unavailable
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "checks" in data
    assert "timestamp" in data
    
    # Verify expected checks
    assert "database" in data["checks"]
    assert "models" in data["checks"]
    assert "storage" in data["checks"]


def test_readiness_models_check(client):
    """Test that readiness correctly reports model status."""
    response = client.get("/api/v1/health/ready")
    
    assert response.status_code == 200
    
    data = response.json()
    models_check = data["checks"]["models"]
    
    # Models should be unavailable initially
    assert models_check["status"] in ["unavailable", "partially_available"]
    assert "details" in models_check
