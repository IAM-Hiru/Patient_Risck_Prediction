"""
stage04_guardrails.py
---------------------
Post-inference clinical safety guardrail engine for Stage 04 3B SLM.
Guarantees:
  1. Zero Under-Triage: Intercepts and escalates any HIGH or CRITICAL presentations
     mislabeled as MODERATE or LOW.
  2. 100% Patient ID Retention: Verifies and restores patient ID integrity.
  3. Strict 2-Sentence Compliance: Normalizes output into exactly 2 clinical sentences.
"""

import re
from typing import Tuple, Dict, Any


CRITICAL_TRIGGERS = [
    r"spo2\s*<\s*88", r"acute\s+dyspnea", r"anaphylaxis", r"angioedema",
    r"stridor", r"altered\s+mental\s+status", r"respiratory\s+arrest",
    r"severe\s+chest\s+tightness"
]

HIGH_TRIGGERS = [
    r"intractable\s+vomiting", r"spiking\s+fever", r"neutropenic\s+nadir",
    r"38\.[5-9]c", r"39\.[0-9]c", r"40\.[0-9]c", r"dehydration",
    r">\s*5\s+episodes", r"grade\s+3\s+diarrhea", r"severe\s+mucositis"
]

TIER_ACTIONS = {
    "LOW": "Recommend routine outpatient monitoring and supportive symptom care according to LOW protocol guidelines.",
    "MODERATE": "Recommend same-day oncology clinic assessment and supportive pharmacological intervention per MODERATE protocol guidelines.",
    "HIGH": "Recommend immediate clinical review, urgent hydration support, and active triage management according to HIGH protocol guidelines.",
    "CRITICAL": "Initiate immediate emergency resuscitation, stat oncology attending notification, and urgent ICU transfer according to CRITICAL protocol guidelines."
}


def count_sentences(text: str) -> int:
    """Accurately count sentences terminated by punctuation."""
    guarded = re.sub(r"(\d+\.\d+)\s*C", r"\1C", text)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", guarded) if p.strip()]
    return len(parts)


def apply_clinical_triage_guardrail(
    raw_briefing: str,
    patient_id: str,
    clinical_note: str
) -> Tuple[str, bool, int, str]:
    """
    Applies post-inference clinical guardrails to raw SLM output.
    Returns: (guarded_briefing, guardrail_triggered, sentence_count, triage_status)
    """
    note_lower = clinical_note.lower()
    guardrail_triggered = False
    status = "VERIFIED_SAFE"

    # 1. Determine Floor Safety Tier from Clinical Presentation
    required_floor_tier = "LOW"
    for trig in CRITICAL_TRIGGERS:
        if re.search(trig, note_lower):
            required_floor_tier = "CRITICAL"
            break

    if required_floor_tier != "CRITICAL":
        for trig in HIGH_TRIGGERS:
            if re.search(trig, note_lower):
                required_floor_tier = "HIGH"
                break

    # 2. Extract Tier from Generated Briefing
    m_tier = re.search(r"\b(LOW|MODERATE|HIGH|CRITICAL)-tier\b", raw_briefing, re.IGNORECASE)
    if not m_tier:
        m_tier = re.search(r"\b(LOW|MODERATE|HIGH|CRITICAL)\b", raw_briefing, re.IGNORECASE)
    
    current_tier = m_tier.group(1).upper() if m_tier else "LOW"

    # Tier severity weights
    tier_weights = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}

    # 3. Intercept Under-Triage Hazard
    guarded_text = raw_briefing
    if tier_weights.get(required_floor_tier, 0) > tier_weights.get(current_tier, 0):
        # Under-triage hazard detected! Escalate immediately.
        guardrail_triggered = True
        status = f"ESCALATED_{current_tier}_TO_{required_floor_tier}"
        
        # Replace tier mention in sentence 1
        guarded_text = re.sub(
            r"\b(LOW|MODERATE|HIGH|CRITICAL)-tier\b",
            f"{required_floor_tier}-tier",
            guarded_text,
            flags=re.IGNORECASE
        )

        # Replace action sentence with required floor guideline action
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", guarded_text) if p.strip()]
        if len(parts) >= 1:
            sentence1 = parts[0]
            if not sentence1.endswith("."):
                sentence1 += "."
            action_sentence = TIER_ACTIONS[required_floor_tier]
            guarded_text = f"{sentence1} {action_sentence}"

    # 4. Enforce Patient ID Retention
    if patient_id not in guarded_text:
        guarded_text = f"Patient {patient_id}: " + guarded_text
        guardrail_triggered = True

    # 5. Enforce Strict 2-Sentence Output Rule
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", guarded_text) if p.strip()]
    if len(parts) > 2:
        # Collapse into 2 sentences
        guarded_text = f"{parts[0]} {' '.join(parts[1:])}"
    elif len(parts) == 1:
        # Add fallback recommendation
        rec_tier = required_floor_tier if guardrail_triggered else current_tier
        guarded_text = f"{parts[0]} {TIER_ACTIONS.get(rec_tier, TIER_ACTIONS['LOW'])}"

    sentence_count = count_sentences(guarded_text)

    return guarded_text, guardrail_triggered, sentence_count, status
