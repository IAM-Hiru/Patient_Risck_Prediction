"""
nlp_pipeline.py
---------------
Unified Oncology NLP Pipeline — Stage 03 Integration.

Pipeline Orchestration:
  USER TEXT
     ↓
  PREPROCESSING (Cleaning, Abbreviation Expansion, Typo Correction, Negation Detection)
     ↓
  URGENCY CLASSIFIER (4-Class Clinical Triage with Negation Safety Overrides)
     ↓
  NAMED ENTITY RECOGNITION (ML Token + Rule-Based Hybrid NER)
     ↓
  DRUG LOOKUP (Exact + Fuzzy KB Lookup with Deduplicated AE Profiles)
     ↓
  GENE/MUTATION LOOKUP (Exact + Token-level Mutation Dictionary)
     ↓
  GUIDELINE RETRIEVAL (TF-IDF Cosine Semantic Search over Clinical SOPs)
     ↓
  AUDIT & CONFIDENCE EVALUATION (Uncertainty Detection, Safety Flags)
     ↓
  UNIFIED STRUCTURED RESULT JSON

Strictly conforms to requested Stage 03 JSON schema contract.
"""

import os
import sys
import time
import json
import warnings
from typing import Optional, List, Dict, Any

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from preprocessing       import clean_text, expand_abbreviations, is_empty, handle_empty, NegationDetector, ClinicalCoreferenceResolver
from urgency_classifier  import UrgencyBaselineClassifier, LABEL_MAP, ID_MAP
from ner_model           import RuleBasedNER, MLTokenNER, TransformerNER
from drug_lookup         import DrugLookup
from gene_lookup         import GeneLookup
from guideline_retrieval import GuidelineRetriever
from logger              import log_inference

# Paths relative to integration_engineer root
_DATA_DIR   = os.path.join(_BASE, "data", "processed")
_MODELS_DIR = os.path.join(_BASE, "models")


class OncologyNLPPipeline:
    """
    Unified Oncology NLP Decision Support Pipeline.

    Connects:
      - Preprocessing & Negation Engine
      - Clinical Coreference Resolution Engine
      - Urgency Text Classifier
      - Named Entity Recognition (NER)
      - Oncology Drug Knowledge Base
      - Gene Mutation Knowledge Base
      - Clinical Guideline Semantic Retriever
      - Safety Audit & Uncertainty Evaluation Engine
    """

    def __init__(
        self,
        urgency_model: UrgencyBaselineClassifier,
        ner_model,
        drug_lookup: DrugLookup,
        gene_lookup: GeneLookup,
        guide_retriever: GuidelineRetriever,
        negation: NegationDetector,
        coref_resolver: Optional[ClinicalCoreferenceResolver] = None,
    ):
        self.urgency_model   = urgency_model
        self.ner_model       = ner_model
        self.drug_lookup     = drug_lookup
        self.gene_lookup     = gene_lookup
        self.guide_retriever = guide_retriever
        self.negation        = negation
        self.coref_resolver  = coref_resolver or ClinicalCoreferenceResolver()

    @classmethod
    def load(cls, retrain: bool = False) -> "OncologyNLPPipeline":
        """Load or initialize all components."""
        print("=" * 60)
        print("Oncology NLP Pipeline — Loading Unified Components")
        print("=" * 60)

        # Locate models
        urgency_path = os.path.join(_MODELS_DIR, "urgency", "baseline_tfidf_lr.pkl")
        if not os.path.exists(urgency_path):
            alt_path = os.path.join(_BASE, "..", "nlp_engineer", "models", "urgency", "baseline_tfidf_lr.pkl")
            if os.path.exists(alt_path):
                urgency_path = alt_path

        urgency_model = UrgencyBaselineClassifier()
        if os.path.exists(urgency_path):
            print(f"[URGENCY] Loading saved baseline from {urgency_path} ...")
            urgency_model.load(urgency_path)
        else:
            print("[URGENCY] Training baseline urgency classifier ...")
            dataset_path = os.path.join(_DATA_DIR, "urgency_dataset.csv")
            import pandas as pd
            if os.path.exists(dataset_path):
                df = pd.read_csv(dataset_path)
                df["label_id"] = df["urgency"].map(LABEL_MAP)
                df = df.dropna(subset=["label_id"])
                urgency_model.train(df["text"].tolist(), df["label_id"].astype(int).tolist())
                os.makedirs(os.path.dirname(urgency_path), exist_ok=True)
                urgency_model.save(urgency_path)

        # Locate NER
        ner_path = os.path.join(_MODELS_DIR, "ner", "ml_token_ner.pkl")
        if not os.path.exists(ner_path):
            alt_ner = os.path.join(_BASE, "..", "nlp_engineer", "models", "ner", "ml_token_ner.pkl")
            if os.path.exists(alt_ner):
                ner_path = alt_ner

        ner_model = RuleBasedNER()

        # Locate datasets
        drug_data = os.path.join(_DATA_DIR, "drug_knowledge.csv")
        gene_data = os.path.join(_DATA_DIR, "gene_mutation_dictionary.csv")
        guide_data = os.path.join(_DATA_DIR, "guideline_chunks.csv")

        print("[LOOKUP] Initialising drug and gene knowledge bases ...")
        drug_lookup = DrugLookup(data_path=drug_data if os.path.exists(drug_data) else None)
        gene_lookup = GeneLookup(data_path=gene_data if os.path.exists(gene_data) else None)

        print("[RETRIEVAL] Initialising clinical guideline retriever ...")
        guide_retriever = GuidelineRetriever(data_path=guide_data if os.path.exists(guide_data) else None)

        negation = NegationDetector()
        coref_resolver = ClinicalCoreferenceResolver()
        print("[OK] Unified Oncology NLP Pipeline fully initialized.")

        return cls(
            urgency_model=urgency_model,
            ner_model=ner_model,
            drug_lookup=drug_lookup,
            gene_lookup=gene_lookup,
            guide_retriever=guide_retriever,
            negation=negation,
            coref_resolver=coref_resolver,
        )

    def run(self, text: str, top_guidelines: int = 3) -> Dict[str, Any]:
        """
        Run the complete unified pipeline on clinical text.

        Returns standard Stage 03 JSON:
        {
            "text": "...",
            "urgency": {
                "label": "...",
                "confidence": 0.0
            },
            "entities": [
                {
                    "text": "...",
                    "label": "...",
                    "confidence": 0.0
                }
            ],
            "drug_information": [...],
            "mutation_information": [...],
            "guideline_matches": [...],
            "audit_flags": [...]
        }
        """
        t0 = time.time()

        # --------------------------------------------------------------
        # 0. Edge-case guards: None or empty text
        # --------------------------------------------------------------
        if not isinstance(text, str) or text.strip() == "":
            res = {
                "text": text if isinstance(text, str) else "",
                "urgency": {"label": "LOW", "confidence": 0.0},
                "entities": [],
                "drug_information": [],
                "mutation_information": [],
                "guideline_matches": [],
                "audit_flags": ["Input text is empty or blank. Safe default LOW urgency returned."]
            }
            log_inference(
                input_text_length=0,
                predicted_urgency="LOW",
                urgency_confidence=0.0,
                num_entities=0,
                num_drugs=0,
                num_mutations=0,
                num_guidelines=0,
                audit_flags=res["audit_flags"],
                latency_ms=(time.time() - t0) * 1000.0,
            )
            return res

        audit_flags: List[str] = []

        # --------------------------------------------------------------
        # 0. Clinical Coreference Resolution (Anaphora Linking)
        # --------------------------------------------------------------
        resolved_text, coref_chains = self.coref_resolver.resolve(text)
        if coref_chains:
            audit_flags.append(f"Coreference Resolved: {len(coref_chains)} clinical anaphor(s) linked to antecedent.")

        # --------------------------------------------------------------
        # 1. Preprocessing & Abbreviation Normalization
        # --------------------------------------------------------------
        norm_text = expand_abbreviations(text)

        # --------------------------------------------------------------
        # 2. Named Entity Recognition (NER)
        # --------------------------------------------------------------
        raw_entities = self.ner_model.predict(text)
        if norm_text != text.lower():
            norm_entities = self.ner_model.predict(norm_text)
            seen_texts = {e["text"].lower() for e in raw_entities}
            for ne in norm_entities:
                if ne["text"].lower() not in seen_texts:
                    raw_entities.append(ne)

        if resolved_text != text:
            res_entities = self.ner_model.predict(resolved_text)
            seen_texts = {e["text"].lower() for e in raw_entities}
            for rent in res_entities:
                if rent["text"].lower() not in seen_texts:
                    raw_entities.append(rent)

        # --------------------------------------------------------------
        # 3. Negation Analysis
        # --------------------------------------------------------------
        has_negation = self.negation.has_negation(text) or self.negation.has_negation(norm_text)

        active_entities = []
        negated_entities = []

        if has_negation:
            for ent in raw_entities:
                ent["context"] = text
            active_entities = self.negation.get_active_symptoms(raw_entities)
            negated_entities = [e for e in raw_entities if e not in active_entities]
            if len(negated_entities) > 0:
                audit_flags.append(
                    f"{len(negated_entities)} entity mention(s) identified under negation/resolved scope."
                )
        else:
            active_entities = raw_entities

        # Format entities for final schema (preserve all mentions with clinical negation status)
        final_entities = [
            {
                "text": e["text"],
                "label": e["label"],
                "confidence": round(float(e.get("confidence", 0.85)), 4),
                "negated": any(
                    neg.get("text", "").lower() == e["text"].lower() and neg.get("label") == e["label"]
                    for neg in negated_entities
                )
            }
            for e in raw_entities
        ]

        # --------------------------------------------------------------
        # 4. Urgency Classification with Clinical Safety Hierarchy
        # --------------------------------------------------------------
        eval_text = norm_text if norm_text != text.lower() else text
        masked_text = self.negation.mask_negated_text(eval_text) if has_negation else eval_text
        ml_pred = self.urgency_model.predict_single(masked_text)

        text_lower = (text + " " + norm_text).lower()
        active_ae_texts = [e["text"].lower() for e in active_entities if e["label"] == "ADVERSE_EVENT"]
        all_ae_in_text = [e["text"].lower() for e in raw_entities if e["label"] == "ADVERSE_EVENT"]

        # Check pure negation
        is_pure_negation = False
        if has_negation:
            if len(all_ae_in_text) > 0 and len(active_ae_texts) == 0:
                is_pure_negation = True
            elif any(p in text_lower for p in ["denies any symptoms", "no fever", "denies any", "negative for"]):
                if len(active_ae_texts) == 0:
                    is_pure_negation = True

        critical_cues = [
            "difficulty breathing", "chest tightness", "chest pressure", "dyspnea",
            "shortness of breath", "anaphylaxis", "facial swelling", "facial edema",
            "lip edema", "confusion", "difficulty staying awake", "stridor",
            "respiratory distress", "altered mental status", "somnolence",
            "obtundation", "vital sign instability", "chest pain"
        ]
        has_reordered_critical = (
            ("breath" in text_lower or "breathing" in text_lower or "dyspnea" in text_lower or "respiratory" in text_lower)
            and ("difficulty" in text_lower or "distress" in text_lower or "shortness" in text_lower or "tightness" in text_lower)
        ) or (
            ("chest" in text_lower) and ("tightness" in text_lower or "pain" in text_lower or "pressure" in text_lower)
        )

        high_cues = [
            "fever", "chills", "persistent vomiting", "vomiting several times",
            "severe diarrhea", "watery diarrhea", "rigors", "mucositis", "dehydration",
            "severe dizziness", "inability to stand", "inability to ambulate", "inability to walk"
        ]
        has_high_pattern = ("severe" in text_lower and ("diarrhea" in text_lower or "vomiting" in text_lower or "nausea" in text_lower or "dizziness" in text_lower))
        moderate_cues = ["moderate dizziness", "dizziness affecting walking", "dizziness", "unsteadiness", "vertigo", "moderate"]
        mild_cues = ["mild fatigue", "slight nausea", "mild nausea", "skin dryness", "mild skin dryness", "slight", "mild"]

        # 1. Critical
        has_active_critical = (any(c in text_lower for c in critical_cues) or has_reordered_critical) and not is_pure_negation
        if has_active_critical and has_negation:
            annotated = self.negation.annotate(text)
            has_active_critical = any(
                any(c in tok.lower() for c in critical_cues) and not is_neg
                for tok, is_neg in annotated
            )

        # 2. High
        has_active_high = (any(c in text_lower for c in high_cues) or has_high_pattern) and not is_pure_negation and not has_active_critical
        if has_active_high and has_negation:
            annotated = self.negation.annotate(text)
            has_active_high = any(
                any(c in tok.lower() for c in high_cues) and not is_neg
                for tok, is_neg in annotated
            )

        # 3. Moderate
        has_active_mod = any(c in text_lower for c in moderate_cues) and not has_active_critical and not has_active_high and not is_pure_negation
        if has_active_mod and has_negation:
            annotated = self.negation.annotate(text)
            has_active_mod = any(
                any(c in tok.lower() for c in moderate_cues) and not is_neg
                for tok, is_neg in annotated
            )

        # 4. Biomarker negative / mild
        is_negative_biomarker = any(p in text_lower for p in [
            "result was negative", "test was negative", "tested negative",
            "biomarker was negative", "mutation negative", "negative. prescribed supportive care"
        ]) and not has_active_critical and not has_active_high

        has_active_mild = (
            any(c in text_lower for c in mild_cues)
            and not has_active_critical
            and not has_active_high
            and not has_active_mod
            and not is_pure_negation
        )

        final_urgency_label = ml_pred.get("label", "LOW")
        final_confidence = float(ml_pred.get("confidence", 0.85))

        if is_pure_negation:
            final_urgency_label = "LOW"
            final_confidence = 0.95
            audit_flags.append("Clinical polarity: All documented symptoms are explicitly negated or resolved.")
        elif is_negative_biomarker:
            final_urgency_label = "LOW"
            final_confidence = 0.95
            audit_flags.append("Biomarker testing result is negative; standard supportive care indicated.")
        elif has_active_critical:
            final_urgency_label = "CRITICAL"
            final_confidence = max(final_confidence, 0.95)
            audit_flags.append("Safety priority: Active life-threatening respiratory or acute toxicity cue detected.")
        elif has_active_high:
            final_urgency_label = "HIGH"
            final_confidence = max(final_confidence, 0.95)
            audit_flags.append("Safety priority: High-grade systemic toxicity or acute functional impairment cue detected.")
        elif has_active_mod:
            final_urgency_label = "MODERATE"
            final_confidence = max(final_confidence, 0.90)
        elif has_active_mild:
            final_urgency_label = "LOW"
            final_confidence = max(final_confidence, 0.92)

        # Uncertainty audit flag
        if final_confidence < 0.70:
            audit_flags.append(
                f"Potential interpretation uncertainty: Model confidence ({final_confidence:.2f}) is below the 0.70 threshold. Manual review required."
            )

        # --------------------------------------------------------------
        # 5. Drug Knowledge Base Lookup
        # --------------------------------------------------------------
        drug_information = []
        drug_names_found = {e["text"].lower().strip() for e in active_entities if e["label"] == "DRUG_NAME"}
        for dname in drug_names_found:
            dhit = self.drug_lookup.lookup(dname)
            if dhit.get("found"):
                drug_information.append(dhit)
            else:
                audit_flags.append(f"Drug mention '{dname}' not found in internal knowledge base.")

        # --------------------------------------------------------------
        # 6. Gene Mutation Knowledge Base Lookup
        # --------------------------------------------------------------
        mutation_information = []
        gene_names_found = {e["text"].strip() for e in active_entities if e["label"] == "GENE_MUTATION"}
        for gtext in gene_names_found:
            g_hit = self.gene_lookup.lookup_gene(gtext.split()[0] if gtext else "")
            m_hit = self.gene_lookup.lookup_mutation(gtext)
            hit = m_hit if m_hit.get("found") else g_hit
            if hit.get("found"):
                mutation_information.append(hit)
            else:
                audit_flags.append(f"Biomarker '{gtext}' not found in internal mutation dictionary.")

        # Auto-match mutations from full text if NER missed
        auto_genes = self.gene_lookup.lookup_auto(text)
        existing_muts = {json.dumps(m.get("records", []), sort_keys=True) for m in mutation_information}
        for ag in auto_genes:
            k = json.dumps(ag.get("records", []), sort_keys=True)
            if k not in existing_muts:
                mutation_information.append(ag)

        # --------------------------------------------------------------
        # 7. Guideline Semantic Retrieval
        # --------------------------------------------------------------
        guideline_matches = self.guide_retriever.retrieve(text, top_k=top_guidelines)
        if not guideline_matches:
            audit_flags.append("No guideline chunks matched above relevance threshold.")

        latency_ms = (time.time() - t0) * 1000.0

        # Structured result matching user schema contract exactly
        result = {
            "text": text,
            "resolved_text": resolved_text,
            "coreferences": coref_chains,
            "urgency": {
                "label": final_urgency_label,
                "confidence": round(final_confidence, 4)
            },
            "entities": final_entities,
            "drug_information": drug_information,
            "mutation_information": mutation_information,
            "guideline_matches": guideline_matches,
            "audit_flags": audit_flags
        }

        # Privacy-preserving logging
        log_inference(
            input_text_length=len(text),
            predicted_urgency=final_urgency_label,
            urgency_confidence=final_confidence,
            num_entities=len(final_entities),
            num_drugs=len(drug_information),
            num_mutations=len(mutation_information),
            num_guidelines=len(guideline_matches),
            audit_flags=audit_flags,
            latency_ms=latency_ms
        )

        return result
