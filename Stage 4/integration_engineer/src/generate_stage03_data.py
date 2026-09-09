"""
generate_stage03_nlp_outputs.py
--------------------------------
Generates 50 representative Stage 3 NLP test records for Stage 4 integration testing.
Includes:
  - Standard baseline cases across LOW, MODERATE, HIGH, and CRITICAL tiers.
  - 3 designated high-risk boundary cases (e.g. fever >38.0C / neutropenic nadir / SpO2 < 88%)
    that test guardrail escalation from under-triaged proposals.
"""

import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_INTEG_DIR = os.path.dirname(_HERE)
_OUT_DIR = os.path.join(_INTEG_DIR, "data", "processed")
os.makedirs(_OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(_OUT_DIR, "stage03_nlp_outputs.json")

CANCERS = [
    "Non-Small Cell Lung Cancer", "HER2-Positive Breast Cancer", "Colorectal Adenocarcinoma",
    "Metastatic Melanoma", "Ovarian High-Grade Serous Carcinoma", "Renal Cell Carcinoma",
    "Small Cell Lung Cancer", "Prostate Adenocarcinoma", "Triple-Negative Breast Cancer",
    "Pancreatic Ductal Adenocarcinoma"
]

BIOMARKERS = [
    "EGFR L858R", "HER2 amplification", "KRAS G12D", "BRAF V600E",
    "BRCA2 pathogenic variant", "VHL alteration", "TP53 missense variant",
    "BRCA1 pathogenic variant", "ALK fusion", "PIK3CA H1047R"
]

REGIMENS = [
    "Osimertinib 80mg oral daily", "Trastuzumab intravenous infusion",
    "Oxaliplatin with Capecitabine", "Pembrolizumab with Ipilimumab",
    "Olaparib maintenance tablets", "Nivolumab 240mg IV biweekly",
    "Atezolizumab with Carboplatin", "Docetaxel 75mg/m2 IV",
    "Doxorubicin with Cyclophosphamide", "Gemcitabine with Nab-Paclitaxel"
]

RECORDS = []

# Generate 47 Standard Cases (12 LOW, 11 MODERATE, 12 HIGH, 12 CRITICAL)
# 1-12: Standard LOW
low_notes = [
    "Mild tiredness and localized dry skin on forearms, vitals stable, appetite normal.",
    "Slight evening fatigue after daily walk, afebrile, tolerating oral regimen without nausea.",
    "Minor dry cough, room air SpO2 98%, normal oral intake, no dizziness.",
    "Grade 1 mild skin xerosis around elbows, denying nausea, vomiting or diarrhea.",
    "Mild generalized lethargy on day 5 post-cycle, blood pressure 122/78 mmHg, normal vitals.",
    "Slight intermittent anorexia, patient is active and maintaining daily physical routine.",
    "Mild fatigue 4 days post-infusion, vitals stable, appetite intact.",
    "Slight dry mouth and mild evening fatigue, patient maintaining weight and hydration.",
    "Mild localized skin dryness, bowel movements normal, tolerating oral therapy well.",
    "Minimal transient nausea after infusion, fully resolved within two hours without antiemetics.",
    "Mild tiredness noted, vital signs completely stable, afebrile.",
    "Routine oncology follow-up: mild localized fatigue, no focal complaints, ambulatory."
]

for i, note in enumerate(low_notes, 1):
    pid = f"P{i:05d}"
    RECORDS.append({
        "patient_id": pid,
        "diagnosis": CANCERS[(i - 1) % len(CANCERS)],
        "biomarker": BIOMARKERS[(i - 1) % len(BIOMARKERS)],
        "regimen": REGIMENS[(i - 1) % len(REGIMENS)],
        "clinical_note": note,
        "true_tier": "LOW",
        "is_boundary_case": False
    })

# 13-23: Standard MODERATE (11 cases)
mod_notes = [
    "Persistent moderate nausea and 3 loose stools per day, mild grade 1 peripheral tingling.",
    "Grade 2 erythematous maculopapular rash covering 15% body surface area, mild pruritus.",
    "Moderate persistent fatigue and 3 watery diarrhea episodes in past 24 hours, afebrile.",
    "Persistent grade 2 peripheral sensory neuropathy in bilateral fingertips, mild gait slowing.",
    "Moderate nausea requiring scheduled ondansetron, 4 loose bowel movements daily.",
    "Moderate mucositis grade 2 with localized oral ulcers, able to swallow liquids.",
    "Persistent dizziness upon standing, BP 105/68 mmHg, mild generalized muscular weakness.",
    "Grade 2 fatigue interfering with work, mild peripheral edema in lower extremities.",
    "Moderate anorexia with 3 kg weight loss over 3 weeks, mild postprandial cramping.",
    "Bilateral hand-foot syndrome grade 2 with erythema and mild swelling, able to perform self-care.",
    "Moderate nausea and 3 episodes of loose watery stools daily, vitals stable, afebrile."
]

for i, note in enumerate(mod_notes, 13):
    pid = f"P{i:05d}"
    RECORDS.append({
        "patient_id": pid,
        "diagnosis": CANCERS[(i - 1) % len(CANCERS)],
        "biomarker": BIOMARKERS[(i - 1) % len(BIOMARKERS)],
        "regimen": REGIMENS[(i - 1) % len(REGIMENS)],
        "clinical_note": note,
        "true_tier": "MODERATE",
        "is_boundary_case": False
    })

# 24-35: Standard HIGH (12 cases)
high_notes = [
    "Severe intractable vomiting (>5 episodes/day) with severe dehydration, unable to tolerate oral intake.",
    "Spiking fever of 38.6C with rigors during neutropenic nadir day 12 post-chemotherapy.",
    "Grade 3 diarrhea (>7 watery stools/day) with dizziness, orthostatic hypotension, and severe dry mouth.",
    "Severe oral mucositis grade 3, unable to swallow solid food or liquids, severe mucosal sloughing.",
    "Persistent temperature of 38.7C with shaking chills, absolute neutrophil count 450/uL.",
    "Intractable nausea and persistent vomiting 6 times in 12 hours, clinically dehydrated.",
    "Severe dehydration secondary to intractable grade 3 diarrhea, BP 92/58 mmHg, tachycardia 118 bpm.",
    "Spiking fever of 38.9C with rigors on day 10 of chemotherapy, neutropenic nadir confirmed.",
    "Severe persistent vomiting (>6 episodes) with ketonuria and inability to keep fluids down.",
    "High fever of 38.8C with shaking rigors, severe lethargy, and localized central line erythema.",
    "Grade 3 diarrhea with 8 watery stools today, severe cramping, dry mucous membranes.",
    "Severe intractable nausea with recurrent emesis 5 times daily, orthostasis and dry tongue."
]

for i, note in enumerate(high_notes, 24):
    pid = f"P{i:05d}"
    RECORDS.append({
        "patient_id": pid,
        "diagnosis": CANCERS[(i - 1) % len(CANCERS)],
        "biomarker": BIOMARKERS[(i - 1) % len(BIOMARKERS)],
        "regimen": REGIMENS[(i - 1) % len(REGIMENS)],
        "clinical_note": note,
        "true_tier": "HIGH",
        "is_boundary_case": False
    })

# 36-47: Standard CRITICAL (12 cases)
crit_notes = [
    "Acute severe dyspnea, SpO2 < 88% on room air with stridor and marked hypotension.",
    "Acute respiratory distress, SpO2 84%, bilateral wheezing, severe accessory muscle use.",
    "Immediate hypersensitivity reaction: severe angioedema, facial swelling, stridor, BP 76/44 mmHg.",
    "Altered mental status with severe hypotension BP 72/40 mmHg and rapid respiratory rate 36/min.",
    "Severe anaphylactic shock within 10 minutes of infusion, diffuse urticaria, stridor, collapse.",
    "Acute dyspnea, SpO2 86% on 4L nasal cannula, marked tachypnea 34/min, and diaphoresis.",
    "Stridor, severe laryngeal edema, acute dyspnea, SpO2 < 88%, emergent airway compromise.",
    "Altered mental status, severe cyanosis, SpO2 82% room air, unresponsive to verbal commands.",
    "Acute respiratory failure with profound dyspnea, SpO2 85%, marked intercostal retractions.",
    "Severe anaphylactoid reaction with bronchospasm, stridor, profound hypotension BP 70/40 mmHg.",
    "Acute severe dyspnea, SpO2 < 88% with marked central cyanosis and impending respiratory collapse.",
    "Sudden acute dyspnea with audible stridor, room air SpO2 83%, systolic BP 74 mmHg."
]

for i, note in enumerate(crit_notes, 36):
    pid = f"P{i:05d}"
    RECORDS.append({
        "patient_id": pid,
        "diagnosis": CANCERS[(i - 1) % len(CANCERS)],
        "biomarker": BIOMARKERS[(i - 1) % len(BIOMARKERS)],
        "regimen": REGIMENS[(i - 1) % len(REGIMENS)],
        "clinical_note": note,
        "true_tier": "CRITICAL",
        "is_boundary_case": False
    })

# 48-50: THREE HIGH-RISK BOUNDARY CASES (Require Guardrail Escalation)
# Boundary 1: Patient presents with moderate nausea, but has spiking fever 38.8C and neutropenic nadir!
# Raw SLM anchors on nausea -> proposes MODERATE. Guardrail must escalate to HIGH.
RECORDS.append({
    "patient_id": "P00048",
    "diagnosis": "Metastatic Colorectal Cancer",
    "biomarker": "KRAS G12C",
    "regimen": "Irinotecan with Capecitabine",
    "clinical_note": "Mild nausea accompanied by spiking fever of 38.8C during neutropenic nadir with dehydration.",
    "true_tier": "HIGH",
    "is_boundary_case": True,
    "escalation_type": "MODERATE_TO_HIGH"
})

# Boundary 2: Patient reports mild rash, but clinical notes document intractable vomiting (>5 episodes) and dehydration!
# Raw SLM anchors on rash -> proposes MODERATE. Guardrail must escalate to HIGH.
RECORDS.append({
    "patient_id": "P00049",
    "diagnosis": "Non-Small Cell Lung Cancer",
    "biomarker": "EGFR exon 19 deletion",
    "regimen": "Osimertinib 80mg oral daily",
    "clinical_note": "Mild skin rash but with intractable vomiting (>5 episodes/day) and severe clinical dehydration.",
    "true_tier": "HIGH",
    "is_boundary_case": True,
    "escalation_type": "MODERATE_TO_HIGH"
})

# Boundary 3: Patient reports mild cough, but clinical notes reveal acute dyspnea with documented SpO2 < 88% and stridor!
# Raw SLM anchors on cough -> proposes LOW. Guardrail must escalate to CRITICAL.
RECORDS.append({
    "patient_id": "P00050",
    "diagnosis": "ALK-Positive Lung Adenocarcinoma",
    "biomarker": "ALK fusion",
    "regimen": "Alectinib 600mg twice daily",
    "clinical_note": "Dry cough accompanied by acute dyspnea and documented SpO2 < 88% with audible stridor.",
    "true_tier": "CRITICAL",
    "is_boundary_case": True,
    "escalation_type": "LOW_TO_CRITICAL"
})

assert len(RECORDS) == 50, f"Expected 50 records, got {len(RECORDS)}"

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(RECORDS, f, indent=2)

print(f"Successfully generated {len(RECORDS)} Stage 3 NLP payloads at: {OUT_FILE}")
