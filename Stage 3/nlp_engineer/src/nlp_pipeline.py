"""
nlp_pipeline.py
---------------
Unified Oncology NLP Pipeline — Stage 3.

Integrates:
  1. Urgency Classifier     (TF-IDF + LR baseline / DistilBERT)
  2. Medical NER            (Rule-based + ML token / DistilBERT)
  3. Drug Lookup            (exact + fuzzy)
  4. Gene Mutation Lookup   (exact + token)
  5. Guideline Retrieval    (TF-IDF cosine similarity)
  6. Negation Detection     (rule-based)

Input:  clinical text string
Output: structured JSON dict

Usage:
    from src.nlp_pipeline import OncologyNLPPipeline
    pipeline = OncologyNLPPipeline.load()
    result   = pipeline.run("EGFR L858R mutation; osimertinib 80 mg prescribed.")
"""

import os
import sys
import json
import warnings
from typing import Optional, List

warnings.filterwarnings("ignore")

# Ensure src/ is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from preprocessing       import clean_text, expand_abbreviations, is_empty, handle_empty, NegationDetector, ClinicalCoreferenceResolver
from urgency_classifier  import (
    UrgencyBaselineClassifier,
    UrgencyTransformerClassifier,
    LABEL_MAP, ID_MAP,
    train_and_evaluate as train_urgency,
)
from ner_model           import (
    RuleBasedNER,
    MLTokenNER,
    TransformerNER,
    train_and_evaluate as train_ner,
)
from drug_lookup         import DrugLookup
from gene_lookup         import GeneLookup
from guideline_retrieval import GuidelineRetriever

_MODELS   = os.path.join(_HERE, "..", "models")
_DATA     = os.path.join(_HERE, "..", "..", "data_engineer", "data", "processed")


class OncologyNLPPipeline:
    """
    Full Oncology NLP Pipeline.

    Components loaded from trained model files.
    Falls back to re-training if models don't exist.

    Parameters
    ----------
    urgency_model  : trained urgency classifier (baseline preferred)
    ner_model      : best available NER model
    drug_lookup    : DrugLookup instance
    gene_lookup    : GeneLookup instance
    guide_retriever: GuidelineRetriever instance
    negation       : NegationDetector instance
    use_transformer: whether to use transformer urgency classifier if available
    coref_resolver : ClinicalCoreferenceResolver instance
    """

    def __init__(
        self,
        urgency_model,
        ner_model,
        drug_lookup:    DrugLookup,
        gene_lookup:    GeneLookup,
        guide_retriever: GuidelineRetriever,
        negation:        NegationDetector,
        transformer_urgency: Optional[UrgencyTransformerClassifier] = None,
        transformer_ner:     Optional[TransformerNER] = None,
        coref_resolver:      Optional[ClinicalCoreferenceResolver] = None,
    ):
        self.urgency_model       = urgency_model
        self.ner_model           = ner_model
        self.drug_lookup         = drug_lookup
        self.gene_lookup         = gene_lookup
        self.guide_retriever     = guide_retriever
        self.negation            = negation
        self.transformer_urgency = transformer_urgency
        self.transformer_ner     = transformer_ner
        self.coref_resolver      = coref_resolver or ClinicalCoreferenceResolver()

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, retrain: bool = False) -> "OncologyNLPPipeline":
        """
        Load or train all components.

        Parameters
        ----------
        retrain : bool
            Force re-training even if saved models exist.
        """
        print("=" * 60)
        print("Oncology NLP Pipeline — Loading Components")
        print("=" * 60)

        baseline_path   = os.path.join(_MODELS, "urgency", "baseline_tfidf_lr.pkl")
        ml_ner_path     = os.path.join(_MODELS, "ner",     "ml_token_ner.pkl")

        # --- Urgency classifier ---
        urgency_model = UrgencyBaselineClassifier()
        transformer_urgency = None

        if not retrain and os.path.exists(baseline_path):
            print("[URGENCY] Loading saved baseline model ...")
            urgency_model.load(baseline_path)
        else:
            print("[URGENCY] Training from scratch ...")
            results, urgency_model = train_urgency()

            # Check if transformer was trained
            tf_dir = os.path.join(_MODELS, "urgency", "distilbert")
            if os.path.exists(tf_dir):
                transformer_urgency = UrgencyTransformerClassifier()
                try:
                    from transformers import (
                        DistilBertTokenizerFast,
                        DistilBertForSequenceClassification,
                    )
                    transformer_urgency.tokenizer = DistilBertTokenizerFast.from_pretrained(tf_dir)
                    transformer_urgency.model     = DistilBertForSequenceClassification.from_pretrained(tf_dir)
                    transformer_urgency._is_trained = True
                    import torch
                    transformer_urgency._device = torch.device("cpu")
                    print("[URGENCY] Transformer model loaded.")
                except Exception as e:
                    print(f"[URGENCY] Could not load transformer: {e}")
                    transformer_urgency = None

        # --- NER ---
        rule_ner = RuleBasedNER()
        ml_ner   = MLTokenNER()
        transformer_ner = None

        if not retrain and os.path.exists(ml_ner_path):
            print("[NER] Loading saved ML NER model ...")
            ml_ner.load()
        else:
            print("[NER] Training NER from scratch ...")
            ner_results, rule_ner, ml_ner = train_ner()

            # Check transformer NER
            tf_ner_dir = os.path.join(_MODELS, "ner", "distilbert_ner")
            if os.path.exists(tf_ner_dir):
                transformer_ner = TransformerNER()
                try:
                    from transformers import (
                        DistilBertTokenizerFast,
                        DistilBertForTokenClassification,
                    )
                    import json as _json, torch
                    transformer_ner.tokenizer = DistilBertTokenizerFast.from_pretrained(tf_ner_dir)
                    transformer_ner.model     = DistilBertForTokenClassification.from_pretrained(tf_ner_dir)
                    label_path = os.path.join(tf_ner_dir, "label_maps.json")
                    if os.path.exists(label_path):
                        with open(label_path) as fp:
                            maps = _json.load(fp)
                        transformer_ner.id2label  = {int(k): v for k, v in maps["id2label"].items()}
                        transformer_ner.label2id  = maps["label2id"]
                    transformer_ner._is_trained = True
                    transformer_ner._device = torch.device("cpu")
                    print("[NER] Transformer NER model loaded.")
                except Exception as e:
                    print(f"[NER] Could not load transformer NER: {e}")
                    transformer_ner = None

        # --- Lookups ---
        print("[LOOKUP] Initialising drug and gene lookups ...")
        drug_lookup  = DrugLookup()
        gene_lookup  = GeneLookup()

        print("[RETRIEVAL] Initialising guideline retriever ...")
        guide_retriever = GuidelineRetriever()

        negation = NegationDetector()
        coref_resolver = ClinicalCoreferenceResolver()

        # Choose best NER
        best_ner = transformer_ner if (transformer_ner and transformer_ner._is_trained) else rule_ner

        print("\n[OK] Pipeline ready.\n")
        return cls(
            urgency_model       = urgency_model,
            ner_model           = best_ner,
            drug_lookup         = drug_lookup,
            gene_lookup         = gene_lookup,
            guide_retriever     = guide_retriever,
            negation            = negation,
            transformer_urgency = transformer_urgency,
            transformer_ner     = transformer_ner,
            coref_resolver      = coref_resolver,
        )

    # ------------------------------------------------------------------
    def run(self, text: str, top_guidelines: int = 3) -> dict:
        """
        Run the full NLP pipeline on clinical text.

        Parameters
        ----------
        text : str
            Raw clinical / patient text input.
        top_guidelines : int
            Number of guideline chunks to retrieve (default 3).

        Returns
        -------
        dict with keys:
            input_text, resolved_text, coreferences, cleaned_text,
            urgency, entities,
            drug_information, mutation_information,
            guideline_matches, negation_detected, warnings
        """
        # Empty input guard
        if is_empty(text):
            return handle_empty(text)

        # 0. Clinical Coreference Resolution (Anaphora Linking)
        resolved_text, coref_chains = self.coref_resolver.resolve(text)

        result = {
            "input_text":    text,
            "resolved_text": resolved_text,
            "coreferences":  coref_chains,
            "cleaned_text":  clean_text(text),
            "urgency":       {},
            "entities":      [],
            "drug_information":     [],
            "mutation_information": [],
            "guideline_matches":    [],
            "negation_detected":    False,
            "warnings":      [],
        }

        # ----------------------------------------------------------
        # 1. Named entity recognition & Negation analysis (FIRST)
        # ----------------------------------------------------------
        norm_text = expand_abbreviations(text)
        entities = self.ner_model.predict(text)
        if norm_text != text.lower():
            norm_entities = self.ner_model.predict(norm_text)
            existing_texts = {e["text"].lower() for e in entities}
            for ne in norm_entities:
                if ne["text"].lower() not in existing_texts:
                    entities.append(ne)

        # Also parse resolved text if anaphoric pronouns were expanded
        if resolved_text != text:
            res_entities = self.ner_model.predict(resolved_text)
            existing_texts = {e["text"].lower() for e in entities}
            for rent in res_entities:
                if rent["text"].lower() not in existing_texts:
                    entities.append(rent)

        # Also run rule-based as supplement if transformer is primary
        if isinstance(self.ner_model, TransformerNER):
            rule_ner_tmp = RuleBasedNER()
            rule_entities = rule_ner_tmp.predict(text)
            existing_spans = {(e["start"], e["end"]) for e in entities}
            for re_ent in rule_entities:
                if (re_ent["start"], re_ent["end"]) not in existing_spans:
                    entities.append(re_ent)

        # Negation detection
        has_negation = self.negation.has_negation(text) or self.negation.has_negation(norm_text)
        result["negation_detected"] = has_negation

        active_entities = []
        negated_entities = []

        if has_negation:
            for ent in entities:
                ent["context"] = text
            active_entities = self.negation.get_active_symptoms(entities)
            negated_entities = [e for e in entities if e not in active_entities]
            negated_count = len(negated_entities)
            if negated_count > 0:
                result["warnings"].append(
                    f"{negated_count} entity mention(s) detected under negation/resolved scope."
                )
            result["entities"] = active_entities
        else:
            active_entities = entities
            result["entities"] = entities

        # ----------------------------------------------------------
        # 2. Urgency classification (Negation-aware + Clinical Rules)
        # ----------------------------------------------------------
        # Mask negated spans so TF-IDF does not trigger on denied symptoms
        eval_text = norm_text if norm_text != text.lower() else text
        masked_text = self.negation.mask_negated_text(eval_text) if has_negation else eval_text
        urgency_pred = self.urgency_model.predict_single(masked_text)

        # Clinical Safety Rule Hierarchy based on ACTIVE symptoms:
        text_lower = (text + " " + norm_text).lower()
        active_ae_texts = [e["text"].lower() for e in active_entities if e["label"] == "ADVERSE_EVENT"]
        all_ae_in_text = [e["text"].lower() for e in entities if e["label"] == "ADVERSE_EVENT"]

        # Check if text is a denial / negation statement with no active symptoms
        is_pure_negation = False
        if has_negation:
            # If all detected symptoms were negated, or patient explicitly denies symptoms
            if len(all_ae_in_text) > 0 and len(active_ae_texts) == 0:
                is_pure_negation = True
            elif any(phrase in text_lower for phrase in ["patient denies any symptoms", "no fever", "denies any", "negative for"]):
                if len(active_ae_texts) == 0:
                    is_pure_negation = True

        critical_cues = [
            "difficulty breathing", "chest tightness", "chest pressure", "dyspnea",
            "shortness of breath", "anaphylaxis", "facial swelling", "facial edema",
            "lip edema", "confusion", "difficulty staying awake", "stridor",
            "respiratory distress", "altered mental status", "somnolence",
            "obtundation", "vital sign instability", "chest pain"
        ]
        # Flexible pair matching for reversed/reordered clinical word combinations
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

        # 1. Critical overrides (only if active / unnegated)
        has_active_critical = (any(c in text_lower for c in critical_cues) or has_reordered_critical) and not is_pure_negation
        if has_active_critical and has_negation:
            annotated = self.negation.annotate(text)
            has_active_critical = any(
                any(c in tok.lower() for c in critical_cues) and not is_neg
                for tok, is_neg in annotated
            )

        # 2. High overrides (only if active / unnegated)
        has_active_high = (any(c in text_lower for c in high_cues) or has_high_pattern) and not is_pure_negation and not has_active_critical
        if has_active_high and has_negation:
            annotated = self.negation.annotate(text)
            has_active_high = any(
                any(c in tok.lower() for c in high_cues) and not is_neg
                for tok, is_neg in annotated
            )

        # 3. Moderate cues
        has_active_mod = any(c in text_lower for c in moderate_cues) and not has_active_critical and not has_active_high and not is_pure_negation
        if has_active_mod and has_negation:
            annotated = self.negation.annotate(text)
            has_active_mod = any(
                any(c in tok.lower() for c in moderate_cues) and not is_neg
                for tok, is_neg in annotated
            )

        # 4. Negative biomarker testing & mild symptom cues
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

        if is_pure_negation:
            urgency_pred = {
                "label": "LOW",
                "class_id": 0,
                "confidence": 0.95,
                "all_scores": {"LOW": 0.95, "MODERATE": 0.03, "HIGH": 0.01, "CRITICAL": 0.01},
                "model": "negation_safety_override",
                "rule_applied": "All symptoms documented as negated/absent."
            }
        elif is_negative_biomarker:
            urgency_pred = {
                "label": "LOW",
                "class_id": 0,
                "confidence": 0.95,
                "all_scores": {"LOW": 0.95, "MODERATE": 0.03, "HIGH": 0.01, "CRITICAL": 0.01},
                "model": "biomarker_negation_override",
                "rule_applied": "Biomarker testing documented as negative; supportive care indicated."
            }
        elif has_active_critical:
            urgency_pred["label"] = "CRITICAL"
            urgency_pred["class_id"] = 3
            urgency_pred["confidence"] = max(urgency_pred.get("confidence", 0.0), 0.95)
            urgency_pred["rule_applied"] = "Active life-threatening respiratory/allergic cue detected."
        elif has_active_high:
            urgency_pred["label"] = "HIGH"
            urgency_pred["class_id"] = 2
            urgency_pred["confidence"] = max(urgency_pred.get("confidence", 0.0), 0.95)
            urgency_pred["rule_applied"] = "Active high-grade systemic toxicity cue detected."
        elif has_active_mod:
            urgency_pred["label"] = "MODERATE"
            urgency_pred["class_id"] = 1
            urgency_pred["confidence"] = max(urgency_pred.get("confidence", 0.0), 0.90)
            urgency_pred["rule_applied"] = "Active moderate functional impairment cue detected."
        elif has_active_mild:
            urgency_pred["label"] = "LOW"
            urgency_pred["class_id"] = 0
            urgency_pred["confidence"] = max(urgency_pred.get("confidence", 0.0), 0.92)
            urgency_pred["rule_applied"] = "Active mild baseline symptom profile without acute impairment."

        result["urgency"] = urgency_pred

        # ----------------------------------------------------------
        # 3. Drug lookup for detected DRUG_NAME entities
        # ----------------------------------------------------------
        drug_entities = [e for e in entities if e["label"] == "DRUG_NAME"]
        seen_drugs = set()
        for drug_ent in drug_entities:
            drug_name = drug_ent["text"].lower().strip()
            if drug_name in seen_drugs:
                continue
            seen_drugs.add(drug_name)
            lookup_result = self.drug_lookup.lookup(drug_name)
            if lookup_result["found"]:
                result["drug_information"].append(lookup_result)
            else:
                result["warnings"].append(
                    f"Drug '{drug_ent['text']}' not found in knowledge base."
                )

        # ----------------------------------------------------------
        # 4. Gene / mutation lookup for GENE_MUTATION entities
        # ----------------------------------------------------------
        gene_entities = [e for e in entities if e["label"] == "GENE_MUTATION"]
        seen_genes = set()
        for gene_ent in gene_entities:
            gene_text = gene_ent["text"].strip()
            if gene_text.lower() in seen_genes:
                continue
            seen_genes.add(gene_text.lower())
            # Try gene lookup first, then mutation
            g_result = self.gene_lookup.lookup_gene(gene_text.split()[0] if gene_text else "")
            m_result = self.gene_lookup.lookup_mutation(gene_text)
            if g_result["found"] or m_result["found"]:
                hit = m_result if m_result["found"] else g_result
                result["mutation_information"].append(hit)
            else:
                result["warnings"].append(
                    f"Gene/mutation '{gene_text}' not found in knowledge base."
                )

        # Also auto-detect from full text (catches things NER missed)
        auto_gene_hits = self.gene_lookup.lookup_auto(text)
        seen_auto = {json.dumps(r.get("records", []), sort_keys=True)
                     for r in result["mutation_information"]}
        for hit in auto_gene_hits:
            key = json.dumps(hit.get("records", []), sort_keys=True)
            if key not in seen_auto:
                result["mutation_information"].append(hit)

        # ----------------------------------------------------------
        # 5. Guideline retrieval
        # ----------------------------------------------------------
        guide_results = self.guide_retriever.retrieve(text, top_k=top_guidelines)
        result["guideline_matches"] = guide_results

        return result

    # ------------------------------------------------------------------
    def run_batch(self, texts: List[str]) -> List[dict]:
        """Run pipeline on a list of texts."""
        return [self.run(t) for t in texts]

    # ------------------------------------------------------------------
    def format_output(self, result: dict, indent: int = 2) -> str:
        """Return pretty-printed JSON string of the result."""
        return json.dumps(result, indent=indent, default=str)


# ===========================================================================
# Edge-case test suite
# ===========================================================================

EDGE_CASES = [
    # Normal cases
    ("normal_1",
     "EGFR L858R mutation was identified; osimertinib 80 mg was prescribed."),
    ("normal_2",
     "Patient received pembrolizumab 200 mg and developed fatigue."),
    ("normal_3",
     "BRCA1 mutation detected. Olaparib 300 mg started for ovarian cancer."),
    # Negation
    ("negation_1",  "No fever."),
    ("negation_2",  "Patient denies nausea and vomiting."),
    ("negation_3",  "History of rash, currently resolved."),
    # Empty / unknown
    ("empty",       ""),
    ("unknown_drug", "Patient is taking xyzunknowndrug 50 mg."),
    ("unknown_gene", "UNKGENE12 mutation detected."),
    # Multiple entities
    ("multi_drug",  "Paclitaxel 175 mg/m2 and cisplatin 75 mg/m2 administered."),
    ("multi_mut",   "KRAS G12D and BRAF V600E mutations both detected."),
    ("multi_sym",   "Patient has severe fatigue, persistent vomiting, and fever."),
    # Missing dosage
    ("no_dosage",   "Patient started nivolumab without documented dosage."),
    # Abbreviations
    ("abbrev",      "Pt hx of SOB and n/v after chemo."),
    # Spelling variation
    ("spelling",    "pembroluzimab 200mg prescribed for lung cancer treatment."),
    # Short text
    ("short",       "Fatigue."),
    # Urgency tests
    ("low_urgency",      "Mild fatigue, slight nausea. Appetite slightly reduced."),
    ("moderate_urgency", "Moderate dizziness affecting daily walking."),
    ("high_urgency",     "Persistent vomiting several times today, severe diarrhea."),
    ("critical_urgency", "Difficulty breathing, chest tightness, widespread rash."),
]


def run_edge_cases(pipeline: OncologyNLPPipeline):
    print("\n" + "=" * 60)
    print("EDGE CASE TEST SUITE")
    print("=" * 60)
    for name, text in EDGE_CASES:
        print(f"\n[CASE: {name}]")
        print(f"  Input: '{text[:80]}'")
        result = pipeline.run(text)
        print(f"  Urgency:   {result['urgency'].get('label', 'N/A')} "
              f"(conf={result['urgency'].get('confidence', 0):.3f})")
        print(f"  Entities:  {[(e['label'], e['text']) for e in result['entities']]}")
        print(f"  Drugs:     {[d['matched_as'] for d in result['drug_information'] if d.get('found')]}")
        print(f"  Genes:     {[r.get('gene', r.get('mutation', '?')) for r in result['mutation_information'] if r.get('found')]}")
        print(f"  Negation:  {result['negation_detected']}")
        print(f"  Guidelines:{len(result['guideline_matches'])} match(es)")
        if result["warnings"]:
            print(f"  Warnings:  {result['warnings']}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Oncology NLP Pipeline")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain all models")
    parser.add_argument("--text",    type=str, default=None,
                        help="Single text to process")
    parser.add_argument("--edge-cases", action="store_true",
                        help="Run edge-case test suite")
    args = parser.parse_args()

    pipeline = OncologyNLPPipeline.load(retrain=args.retrain)

    if args.text:
        result = pipeline.run(args.text)
        print(pipeline.format_output(result))

    elif args.edge_cases:
        run_edge_cases(pipeline)

    else:
        # Default demo
        demo_text = (
            "Patient has EGFR L858R mutation. "
            "Osimertinib 80 mg once daily was prescribed. "
            "Patient developed severe fatigue and mild rash. "
            "No fever. Patient denies nausea."
        )
        print(f"\nDemo input:\n  '{demo_text}'\n")
        result = pipeline.run(demo_text)
        print(pipeline.format_output(result))
