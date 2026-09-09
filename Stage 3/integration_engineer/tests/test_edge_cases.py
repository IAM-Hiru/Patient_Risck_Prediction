"""
test_edge_cases.py
------------------
Clinical edge cases and robustness tests for the unified NLP integration.
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


def test_edge_case_empty(pipeline):
    res = pipeline.run("")
    assert res["urgency"]["label"] == "LOW"
    assert res["urgency"]["confidence"] == 0.0
    assert len(res["entities"]) == 0
    assert len(res["audit_flags"]) >= 1


def test_edge_case_multi_drug_regimen(pipeline):
    text = "Combination protocol initiated with paclitaxel 175 mg/m2, carboplatin AUC 5, and bevacizumab 15 mg/kg IV."
    res = pipeline.run(text)
    entities = res["entities"]
    drugs_found = [e["text"].lower() for e in entities if e["label"] == "DRUG_NAME"]
    dosages_found = [e["text"].lower() for e in entities if e["label"] == "DOSAGE"]

    assert any("paclitaxel" in d for d in drugs_found)
    assert any("carboplatin" in d for d in drugs_found)
    assert any("bevacizumab" in d for d in drugs_found)
    assert any("175 mg/m2" in ds for ds in dosages_found)
    assert any("15 mg/kg" in ds for ds in dosages_found)


def test_edge_case_multi_severe_symptoms(pipeline):
    text = "Patient presents with persistent severe vomiting, severe diarrhea, and high fever."
    res = pipeline.run(text)
    assert res["urgency"]["label"] in {"HIGH", "CRITICAL"}
    ae_entities = [e["text"].lower() for e in res["entities"] if e["label"] == "ADVERSE_EVENT"]
    assert len(ae_entities) >= 2


def test_edge_case_explicit_negations(pipeline):
    text = "Patient denies nausea. No fever. Denies any shortness of breath or chest pain."
    res = pipeline.run(text)
    assert res["urgency"]["label"] == "LOW"
    assert any("negat" in flag.lower() or "polarity" in flag.lower() for flag in res["audit_flags"])


def test_edge_case_heavy_abbreviations(pipeline):
    text = "Pt hx of SOB, c/o severe n/v and HA after 2nd cycle of chemo."
    res = pipeline.run(text)
    assert res["urgency"]["label"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}
    ae_entities = [e["text"].lower() for e in res["entities"] if e["label"] == "ADVERSE_EVENT"]
    assert len(ae_entities) >= 2
    # Verify SOB and nausea/vomiting/headache were extracted
    all_aes = " ".join(ae_entities)
    assert ("sob" in all_aes or "breath" in all_aes)
    assert ("n/v" in all_aes or "nausea" in all_aes or "vomiting" in all_aes)


def test_edge_case_unknown_terminology(pipeline):
    text = "Patient received experimental drug XYZ-9988 450 mg and experienced pseudo-klingon arthralgia."
    res = pipeline.run(text)
    # Does not crash, extracts dosage
    assert any("450 mg" in e["text"].lower() for e in res["entities"])
    assert any("unknown" in flag.lower() or "not found" in flag.lower() or "uncertainty" in flag.lower() for flag in res["audit_flags"])


def test_edge_case_biomarker_negation(pipeline):
    text = "Patient was tested for EGFR L858R mutation and result was negative. Prescribed supportive care."
    res = pipeline.run(text)
    assert res["urgency"]["label"] == "LOW"
    assert any("negative" in flag.lower() or "biomarker" in flag.lower() for flag in res["audit_flags"])
