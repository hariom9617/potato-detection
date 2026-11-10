"""
Tests for model status endpoints.
"""

import pytest


def test_get_models_status(client):
    """Test getting model status."""
    response = client.get("/api/v1/models/status")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "models" in data
    assert "timestamp" in data
    
    # Should have 3 models
    assert len(data["models"]) == 3
    
    # Verify model structure
    for model in data["models"]:
        assert "id" in model
        assert "name" in model
        assert "type" in model
        assert "status" in model
        
        # All should be unavailable initially
        assert model["status"] == "unavailable"
        assert "reason" in model


def test_models_include_all_types(client):
    """Test that all expected models are listed."""
    response = client.get("/api/v1/models/status")
    
    data = response.json()
    model_ids = [m["id"] for m in data["models"]]
    
    # Verify all expected models present
    assert "potato_verifier_v1" in model_ids
    assert "external_detector_v1" in model_ids
    assert "internal_detector_v1" in model_ids
