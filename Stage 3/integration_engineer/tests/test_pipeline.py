"""
test_pipeline.py
----------------
Unit and integration tests for the unified OncologyNLPPipeline.
"""

import os
import sys
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_BASE, "src"))

from nlp_pipeline import OncologyNLPPipeline


@pytest.fixture(scope="module")
def pipeline():
    return OncologyNLPPipeline.load(retrain=False)


def test_pipeline_loading(pipeline):
    assert pipeline is not None
    assert pipeline.urgency_model is not None
    assert pipeline.ner_model is not None
    assert pipeline.drug_lookup is not None
    assert pipeline.gene_lookup is not None
    assert pipeline.guide_retriever is not None


def test_normal_text_analysis(pipeline):
    text = "Patient received pembrolizumab 200 mg and developed severe fatigue."
    res = pipeline.run(text)

    # Schema keys check
    expected_keys = {
        "text", "urgency", "entities", "drug_information",
        "mutation_information", "guideline_matches", "audit_flags"
    }
    assert expected_keys.issubset(set(res.keys()))

    # Text parity
    assert res["text"] == text

    # Urgency check
    assert "label" in res["urgency"]
    assert "confidence" in res["urgency"]
    assert res["urgency"]["label"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}
    assert 0.0 <= res["urgency"]["confidence"] <= 1.0

    # Entities check
    entities = res["entities"]
    assert len(entities) >= 2
    ent_texts = [e["text"].lower() for e in entities]
    assert any("pembrolizumab" in t for t in ent_texts)
    assert any("200 mg" in t for t in ent_texts)

    # Drug information check
    assert len(res["drug_information"]) >= 1
    assert res["drug_information"][0]["found"] is True
    assert res["drug_information"][0]["matched_as"] == "pembrolizumab"


def test_empty_text_handling(pipeline):
    res = pipeline.run("")
    assert res["urgency"]["label"] == "LOW"
    assert res["urgency"]["confidence"] == 0.0
    assert len(res["entities"]) == 0
    assert len(res["audit_flags"]) >= 1


def test_confidence_calibration_range(pipeline):
    texts = [
        "Routine follow-up; asymptomatic.",
        "Severe dyspnea and acute respiratory distress SpO2 86%.",
        "Moderate dizziness affecting gait."
    ]
    for t in texts:
        res = pipeline.run(t)
        conf = res["urgency"]["confidence"]
        assert 0.0 <= conf <= 1.0
        assert isinstance(res["audit_flags"], list)
