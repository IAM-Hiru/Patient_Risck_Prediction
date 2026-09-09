"""
test_api.py
-----------
Unit and endpoint integration tests for FastAPI service.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
sys.path.insert(0, _BASE)

from api import app

client = TestClient(app)


def test_get_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "disclaimer" in data
    assert "components" in data
    assert "urgency_classifier" in data["components"]
    assert "guideline_retrieval" in data["components"]


def test_post_analyze_valid():
    payload = {
        "text": "Patient received pembrolizumab 200 mg and developed severe fatigue.",
        "top_guidelines": 2
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Schema validation
    assert "text" in data
    assert "urgency" in data
    assert "entities" in data
    assert "drug_information" in data
    assert "mutation_information" in data
    assert "guideline_matches" in data
    assert "audit_flags" in data

    assert data["urgency"]["label"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert len(data["entities"]) > 0
    assert len(data["drug_information"]) > 0
    assert len(data["guideline_matches"]) <= 2


def test_post_analyze_empty_string():
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


def test_post_analyze_whitespace_only():
    response = client.post("/analyze", json={"text": "     \n\t  "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


def test_post_analyze_missing_field():
    response = client.post("/analyze", json={})
    assert response.status_code == 422  # Pydantic unprocessable entity
