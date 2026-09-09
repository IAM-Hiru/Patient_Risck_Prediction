"""
api.py
------
FastAPI Service for Stage 03 Oncology NLP Decision Support.

Endpoints:
  GET  /health   - Service status, loaded components, dataset verification
  POST /analyze  - Unified NLP analysis over clinical text
"""

import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from nlp_pipeline import OncologyNLPPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Oncology NLP Decision Support API — Stage 03",
    description=(
        "**Clinical Disclaimer:** Synthetic demonstration system. "
        "Not for clinical diagnosis, treatment decisions, or emergency patient triage."
    ),
    version="3.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline Singleton
_pipeline: Optional[OncologyNLPPipeline] = None


def get_pipeline() -> OncologyNLPPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = OncologyNLPPipeline.load(retrain=False)
    return _pipeline


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        description="Clinical note, oncology consult, or patient symptom text.",
        example="Patient received pembrolizumab 200 mg and developed severe fatigue."
    )
    top_guidelines: Optional[int] = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of guideline SOP matches to retrieve."
    )


class UrgencyResult(BaseModel):
    label: str = Field(..., description="Triage priority: LOW, MODERATE, HIGH, CRITICAL")
    confidence: float = Field(..., description="Classification confidence score [0.0 - 1.0]")


class EntityResult(BaseModel):
    text: str
    label: str
    confidence: float
    negated: Optional[bool] = False


class AnalyzeResponse(BaseModel):
    text: str
    urgency: UrgencyResult
    entities: List[EntityResult]
    drug_information: List[Dict[str, Any]]
    mutation_information: List[Dict[str, Any]]
    guideline_matches: List[Dict[str, Any]]
    audit_flags: List[str]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    disclaimer: str
    timestamp: str
    components: Dict[str, str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    """Warm up pipeline on service startup."""
    try:
        get_pipeline()
    except Exception as e:
        print(f"[ERROR] Failed to load NLP Pipeline on startup: {e}")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint returning system status and component availability.
    """
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "service": "Oncology NLP Decision Support API",
        "version": "3.0.0",
        "disclaimer": "Synthetic demonstration system. Not for clinical diagnosis, treatment decisions, or medical advice.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "urgency_classifier": "loaded (Baseline TF-IDF + LR)",
            "ner_model": "loaded (Hybrid Token + Rule-based)",
            "drug_lookup": f"loaded ({len(pipeline.drug_lookup.df)} drug entries)",
            "gene_lookup": "loaded (Mutation Dictionary)",
            "guideline_retrieval": f"loaded ({len(pipeline.guide_retriever.df)} guideline chunks)",
            "audit_system": "operational"
        }
    }


@app.post("/analyze", response_model=AnalyzeResponse, tags=["NLP Analysis"])
def analyze_text(request: AnalyzeRequest):
    """
    Analyze clinical text and return structured oncology decision support JSON.
    """
    if not isinstance(request.text, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input 'text' field must be a valid string."
        )

    # Empty string validation
    if request.text.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text cannot be empty or whitespace only."
        )

    try:
        pipeline = get_pipeline()
        result = pipeline.run(request.text, top_guidelines=request.top_guidelines or 3)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
