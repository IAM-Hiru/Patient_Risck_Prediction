"""
logger.py
---------
Privacy-Preserving Audit Logger for Oncology NLP Integration.

Logs operational metrics (timestamp, input length, predicted urgency,
entity counts, latency, audit flags, errors) WITHOUT storing raw patient text
or personal health information (PHI) in compliance with HIPAA privacy standards.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR = os.path.join(_HERE, "..", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(_LOG_DIR, "integration_audit.log")

# Configure logger
logger = logging.getLogger("OncologyNLPIntegrationAudit")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # File handler with JSON formatting
    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def log_inference(
    input_text_length: int,
    predicted_urgency: str,
    urgency_confidence: float,
    num_entities: int,
    num_drugs: int,
    num_mutations: int,
    num_guidelines: int,
    audit_flags: List[str],
    latency_ms: float,
    model_errors: Optional[str] = None,
    retrieval_errors: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Log an inference event with zero retention of patient identifiable text.
    """
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_length": input_text_length,
        "predicted_urgency": predicted_urgency,
        "urgency_confidence": round(urgency_confidence, 4),
        "num_entities": num_entities,
        "num_drugs": num_drugs,
        "num_mutations": num_mutations,
        "num_guidelines": num_guidelines,
        "audit_flags_count": len(audit_flags),
        "audit_flags": audit_flags,
        "latency_ms": round(latency_ms, 2),
        "model_errors": model_errors or "NONE",
        "retrieval_errors": retrieval_errors or "NONE",
    }

    log_msg = (
        f"INFERENCE | Len={input_text_length} | Urgency={predicted_urgency} "
        f"(Conf={urgency_confidence:.2f}) | Ents={num_entities} | "
        f"Latency={latency_ms:.1f}ms | Errors={model_errors or 'None'}"
    )
    if model_errors or retrieval_errors:
        logger.error(log_msg + f" | Details: {json.dumps(event)}")
    else:
        logger.info(log_msg)

    return event
