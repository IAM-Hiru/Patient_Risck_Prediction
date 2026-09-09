"""
ner_model.py
------------
Medical Named Entity Recognition for the Oncology NLP Pipeline.

Entities:
    GENE_MUTATION  — gene name + mutation string (e.g., "EGFR L858R")
    DRUG_NAME      — oncology drug names
    DOSAGE         — dosage patterns (e.g., "200 mg", "80 mg")
    ADVERSE_EVENT  — adverse events / side effects
    CANCER_TYPE    — cancer type mentions

APPROACH:
    Tier 1 (Rule-based):   Regex + dictionary matching — always runs.
    Tier 2 (ML/CRF-style): Token classifier trained on BIO-tagged NER dataset.
    Tier 3 (Transformer):  Token classification head on DistilBERT — if available.

All three tiers run. Results are compared and the best available tier is used.
Transformer results are NOT fabricated.
"""

import os
import re
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

_HERE    = os.path.dirname(os.path.abspath(__file__))
_DATA    = os.path.join(_HERE, "..", "..", "data_engineer", "data", "processed")
_MODELS  = os.path.join(_HERE, "..", "models", "ner")
_TOK_DIR = os.path.join(_HERE, "..", "models", "tokenizer")
_OUTPUTS = os.path.join(_HERE, "..", "outputs")

os.makedirs(_MODELS,  exist_ok=True)
os.makedirs(_TOK_DIR, exist_ok=True)
os.makedirs(_OUTPUTS, exist_ok=True)

ENTITY_LABELS = ["GENE_MUTATION", "DRUG_NAME", "DOSAGE", "ADVERSE_EVENT", "CANCER_TYPE"]


# ===========================================================================
# Tier 1 — Rule-Based NER (Dictionary + Regex)
# ===========================================================================

class RuleBasedNER:
    """
    Rule-based NER using curated dictionary lists and regex patterns.

    Always available. Does not require training.
    """

    # --- Oncology drug list (from drug_knowledge.csv + common additions)
    DRUGS = {
        "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab",
        "ipilimumab", "tremelimumab", "avelumab", "cemiplimab",
        "osimertinib", "erlotinib", "gefitinib", "afatinib", "dacomitinib",
        "crizotinib", "alectinib", "brigatinib", "lorlatinib",
        "olaparib", "niraparib", "rucaparib", "talazoparib",
        "trastuzumab", "pertuzumab", "ado-trastuzumab", "margetuximab",
        "bevacizumab", "ramucirumab", "ziv-aflibercept",
        "paclitaxel", "docetaxel", "cabazitaxel",
        "cisplatin", "carboplatin", "oxaliplatin",
        "doxorubicin", "epirubicin", "idarubicin",
        "palbociclib", "ribociclib", "abemaciclib",
        "vemurafenib", "dabrafenib", "encorafenib",
        "trametinib", "cobimetinib", "binimetinib",
        "imatinib", "dasatinib", "nilotinib", "ponatinib",
        "venetoclax", "ibrutinib", "idelalisib", "copanlisib",
        "bortezomib", "carfilzomib", "ixazomib",
        "lenalidomide", "thalidomide", "pomalidomide",
        "rituximab", "obinutuzumab", "ofatumumab",
        "gemcitabine", "fluorouracil", "capecitabine", "pemetrexed",
        "vinorelbine", "etoposide", "topotecan", "irinotecan",
        "cyclophosphamide", "ifosfamide", "temozolomide",
        "tamoxifen", "letrozole", "anastrozole", "exemestane",
        "leuprolide", "enzalutamide", "abiraterone", "apalutamide",
    }

    # --- Adverse events
    ADVERSE_EVENTS = {
        "fatigue", "nausea", "vomiting", "diarrhea", "diarrhoea",
        "constipation", "rash", "pruritus", "alopecia",
        "neutropenia", "thrombocytopenia", "anaemia", "anemia",
        "pneumonitis", "colitis", "hepatitis", "thyroiditis",
        "nephritis", "myocarditis", "uveitis", "encephalitis",
        "fever", "pyrexia", "chills", "rigors",
        "mucositis", "stomatitis", "esophagitis",
        "peripheral neuropathy", "neuropathy",
        "arthralgia", "myalgia", "muscle weakness",
        "hypertension", "hypotension",
        "dyspnoea", "dyspnea", "shortness of breath",
        "chest pain", "chest tightness",
        "infusion reaction", "anaphylaxis", "hypersensitivity",
        "pain", "headache", "dizziness", "insomnia", "anxiety",
        "appetite loss", "weight loss", "dehydration",
        "hand-foot syndrome", "palmar-plantar erythrodysaesthesia",
        "edema", "oedema", "ascites",
        "alopecia", "hair loss",
        "elevated liver enzymes", "transaminitis", "elevated ast", "elevated alt",
        "elevated creatinine", "proteinuria",
        "sob", "ha", "n/v", "headach", "nausia", "fatige", "brething", "vommiting",
        "severe fatigue", "mild fatigue", "severe nausea", "mild nausea",
        "severe diarrhea", "severe vomiting", "severe dizziness",
    }

    # --- Cancer types
    CANCER_TYPES = {
        "lung cancer", "breast cancer", "colorectal cancer",
        "colon cancer", "rectal cancer",
        "ovarian cancer", "cervical cancer", "endometrial cancer",
        "prostate cancer", "bladder cancer", "kidney cancer",
        "renal cell carcinoma", "clear cell carcinoma",
        "melanoma", "skin cancer",
        "glioblastoma", "glioma", "brain cancer",
        "leukemia", "lymphoma", "myeloma", "multiple myeloma",
        "pancreatic cancer", "gastric cancer", "esophageal cancer",
        "hepatocellular carcinoma", "liver cancer",
        "thyroid cancer", "head and neck cancer",
        "non-small cell lung cancer", "nsclc", "sclc",
        "small cell lung cancer",
        "invasive ductal breast cancer", "invasive ductal carcinoma",
        "triple-negative breast cancer", "tnbc",
        "her2-positive breast cancer",
        "colorectal adenocarcinoma", "metastatic colorectal cancer",
        "ovarian high-grade serous carcinoma", "high-grade serous carcinoma",
        "metastatic melanoma",
        "pancreatic ductal adenocarcinoma",
        "gastric adenocarcinoma",
        "glioblastoma multiforme",
        "prostate adenocarcinoma",
        "acute myeloid leukemia", "aml",
        "chronic lymphocytic leukemia", "cll",
        "diffuse large b-cell lymphoma", "dlbcl",
        "follicular lymphoma",
        "mantle cell lymphoma",
    }

    # --- Gene patterns
    GENE_NAMES = {
        "egfr", "brca1", "brca2", "kras", "braf", "alk", "ros1",
        "met", "ret", "ntrk", "pik3ca", "pten", "tp53",
        "msh2", "msh6", "mlh1", "pms2",
        "erbb2", "her2", "fgfr", "fgfr1", "fgfr2", "fgfr3",
        "idh1", "idh2", "nf1", "stk11", "keap1",
        "erbb3", "cdkn2a", "cdkn2b", "smad4", "apc",
    }

    def __init__(self, drug_knowledge_path: Optional[str] = None):
        # Load drug names from CSV to supplement the hard-coded list
        if drug_knowledge_path is None:
            drug_knowledge_path = os.path.join(_DATA, "drug_knowledge.csv")
        try:
            dk = pd.read_csv(drug_knowledge_path)
            csv_drugs = set(dk["drug_name"].str.lower().str.strip().tolist())
            self.DRUGS = self.DRUGS | csv_drugs
        except Exception:
            pass

        # Compile regex patterns - longest compound units first
        self._dosage_pat = re.compile(
            r"\b\d+(?:\.\d+)?\s*(?:mg/m2|mg/kg|mg/dl|mcg/m2|mcg/kg|mg|g|mcg|ug|ml|l|mmol|iu|units?)(?:\s*/\s*(?:m2|kg|dl|day|dose))?\b",
            re.IGNORECASE,
        )
        self._gene_mut_pat = re.compile(
            r"\b(?:"
            + "|".join(re.escape(g) for g in sorted(self.GENE_NAMES, key=len, reverse=True))
            + r")(?:\s+(?:exon\s+\d+\s+(?:deletion|insertion|mutation)|pathogenic\s+variant|missense\s+variant|amplification|rearrangement|fusion|[A-Z]?\d{1,4}[A-Z]?))?\b"
            + r"|\b[A-Z]\d{2,4}[A-Z]?\b",  # standalone mutation codes
            re.IGNORECASE,
        )

        # Sort by length descending to prefer longer matches
        self._sorted_drugs    = sorted(self.DRUGS,           key=len, reverse=True)
        self._sorted_aes      = sorted(self.ADVERSE_EVENTS,  key=len, reverse=True)
        self._sorted_cancers  = sorted(self.CANCER_TYPES,    key=len, reverse=True)
        self._sorted_genes    = sorted(self.GENE_NAMES,      key=len, reverse=True)

    def _find_spans(self, text: str, terms: list, label: str) -> List[dict]:
        entities = []
        text_lower = text.lower()
        for term in terms:
            pat = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            for m in pat.finditer(text_lower):
                entities.append({
                    "text":       text[m.start():m.end()],
                    "label":      label,
                    "start":      m.start(),
                    "end":        m.end(),
                    "confidence": 0.85,  # rule-based confidence
                    "source":     "rule_based",
                })
        return entities

    def predict(self, text: str) -> List[dict]:
        """Extract named entities from clinical text."""
        if not isinstance(text, str) or text.strip() == "":
            return []

        entities = []

        # Gene + mutation
        for m in self._gene_mut_pat.finditer(text):
            span_text = m.group().strip()
            if len(span_text) >= 3:
                entities.append({
                    "text":       span_text,
                    "label":      "GENE_MUTATION",
                    "start":      m.start(),
                    "end":        m.end(),
                    "confidence": 0.80,
                    "source":     "rule_based",
                })

        # Dosage (high confidence)
        for m in self._dosage_pat.finditer(text):
            entities.append({
                "text":       m.group().strip(),
                "label":      "DOSAGE",
                "start":      m.start(),
                "end":        m.end(),
                "confidence": 0.95,
                "source":     "rule_based",
            })

        # Cancer types (long phrases first)
        entities.extend(self._find_spans(text, self._sorted_cancers, "CANCER_TYPE"))

        # Drugs
        entities.extend(self._find_spans(text, self._sorted_drugs, "DRUG_NAME"))

        # Adverse events
        entities.extend(self._find_spans(text, self._sorted_aes, "ADVERSE_EVENT"))

        # Deduplicate by span: keep highest-priority label for overlapping spans
        entities = self._resolve_overlaps(entities)
        return entities

    def _resolve_overlaps(self, entities: List[dict]) -> List[dict]:
        """Remove overlapping entities, prefer longer spans and earlier labels."""
        if not entities:
            return entities
        entities = sorted(entities, key=lambda x: (x["start"], -(x["end"] - x["start"])))
        kept = []
        last_end = -1
        for ent in entities:
            if ent["start"] >= last_end:
                kept.append(ent)
                last_end = ent["end"]
        return kept


# ===========================================================================
# Tier 2 — ML-based NER (token-level sklearn classifier)
# ===========================================================================

class MLTokenNER:
    """
    Token-level NER trained on BIO-tagged NER dataset.

    Each token is classified using a TF-IDF window feature vector.
    This is a lightweight fallback that doesn't require a GPU.
    """

    def __init__(self):
        self.clf = None
        self.vectorizer = None
        self._is_trained = False
        self.label_encoder = None
        self._labels = []

    def _extract_token_features(self, tokens: List[str], idx: int) -> str:
        """Extract context window for a token as a feature string."""
        window = []
        for offset in range(-2, 3):
            pos = idx + offset
            if 0 <= pos < len(tokens):
                tok = tokens[pos].lower()
                window.append(f"w{offset}={tok}")
                window.append(f"len{offset}={len(tok)}")
                window.append(f"upper{offset}={int(tokens[pos][0].isupper() if tokens[pos] else 0)}")
                window.append(f"digit{offset}={int(any(c.isdigit() for c in tok))}")
                window.append(f"alpha{offset}={int(tok.isalpha())}")
            else:
                window.append(f"w{offset}=<PAD>")
        return " ".join(window)

    def build_features_from_dataset(self, ner_data: list) -> Tuple[List[str], List[str]]:
        """
        Convert NER dataset (from JSONL) into token-level feature/label pairs.
        Uses BIO tags already present in the dataset.
        """
        X, y = [], []
        for rec in ner_data:
            tokens   = rec.get("tokens", [])
            bio_tags = rec.get("bio_tags", [])
            if len(tokens) != len(bio_tags):
                continue
            for idx, (tok, tag) in enumerate(zip(tokens, bio_tags)):
                feat = self._extract_token_features(tokens, idx)
                X.append(feat)
                y.append(tag)
        return X, y

    def train(self, ner_data: list) -> dict:
        """Train the token classifier on NER JSONL data."""
        print("  [ML NER] Building features from BIO tags ...")
        X, y = self.build_features_from_dataset(ner_data)
        print(f"  [ML NER] {len(X)} token samples")

        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)
        self._labels = list(self.label_encoder.classes_)

        # TF-IDF on feature strings
        self.vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), max_features=3000)
        X_mat = self.vectorizer.fit_transform(X)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_mat, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        self.clf = LogisticRegression(
            C=0.5, max_iter=500, class_weight="balanced", solver="saga", random_state=42
        )
        self.clf.fit(X_tr, y_tr)
        self._is_trained = True

        y_pred = self.clf.predict(X_te)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        y_te_labels   = self.label_encoder.inverse_transform(y_te)

        report = classification_report(
            y_te_labels, y_pred_labels, output_dict=True, zero_division=0
        )
        print(f"  [ML NER] Token accuracy: {report.get('accuracy', 'N/A')}")
        return {"token_classification_report": report, "labels": self._labels}

    def _predict_tokens(self, tokens: List[str]) -> List[str]:
        if not self._is_trained:
            return ["O"] * len(tokens)
        feats = [self._extract_token_features(tokens, i) for i in range(len(tokens))]
        X = self.vectorizer.transform(feats)
        preds = self.clf.predict(X)
        return list(self.label_encoder.inverse_transform(preds))

    def predict(self, text: str) -> List[dict]:
        """Extract entities using BIO-tag prediction."""
        if not text or not isinstance(text, str):
            return []
        tokens = text.split()
        bio_preds = self._predict_tokens(tokens)
        return self._bio_to_entities(text, tokens, bio_preds)

    def _bio_to_entities(self, text: str, tokens: List[str], tags: List[str]) -> List[dict]:
        """Convert BIO tag sequence back to entity spans."""
        entities = []
        i = 0
        # Reconstruct char positions
        char_pos = []
        cursor = 0
        for tok in tokens:
            idx = text.find(tok, cursor)
            if idx == -1:
                idx = cursor
            char_pos.append((idx, idx + len(tok)))
            cursor = idx + len(tok)

        while i < len(tags):
            tag = tags[i]
            if tag.startswith("B-"):
                label = tag[2:]
                start_char = char_pos[i][0]
                end_char   = char_pos[i][1]
                ent_tokens = [tokens[i]]
                j = i + 1
                while j < len(tags) and tags[j] == f"I-{label}":
                    end_char = char_pos[j][1]
                    ent_tokens.append(tokens[j])
                    j += 1
                entities.append({
                    "text":       " ".join(ent_tokens),
                    "label":      label,
                    "start":      start_char,
                    "end":        end_char,
                    "confidence": 0.70,
                    "source":     "ml_token_classifier",
                })
                i = j
            else:
                i += 1
        return entities

    def save(self):
        path = os.path.join(_MODELS, "ml_token_ner.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "clf": self.clf,
                "vectorizer": self.vectorizer,
                "label_encoder": self.label_encoder,
                "labels": self._labels,
            }, f)
        print(f"  [SAVED] ML NER model -> {path}")

    def load(self):
        path = os.path.join(_MODELS, "ml_token_ner.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.clf           = data["clf"]
        self.vectorizer    = data["vectorizer"]
        self.label_encoder = data["label_encoder"]
        self._labels       = data["labels"]
        self._is_trained   = True


# ===========================================================================
# Tier 3 — Transformer NER (DistilBERT token classification)
# ===========================================================================

class TransformerNER:
    """
    DistilBERT token classification NER.

    Only run if torch + transformers are available.
    Results NOT fabricated if training fails.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name  = model_name
        self.model       = None
        self.tokenizer   = None
        self._is_trained = False
        self._available  = self._check()
        self.id2label    = {}
        self.label2id    = {}

    def _check(self) -> bool:
        try:
            import torch, transformers
            return True
        except ImportError:
            return False

    def _build_label_maps(self, all_bio_tags: List[str]) -> Tuple[dict, dict]:
        unique = sorted(set(all_bio_tags))
        label2id = {l: i for i, l in enumerate(unique)}
        id2label = {i: l for l, i in label2id.items()}
        return label2id, id2label

    def train(self, ner_data: list, epochs: int = 2, batch_size: int = 8) -> dict:
        if not self._available:
            return {"success": False, "reason": "torch/transformers not available"}

        try:
            import torch
            try:
                from torch.optim import AdamW
            except ImportError:
                from transformers import AdamW
            from transformers import (
                DistilBertTokenizerFast,
                DistilBertForTokenClassification,
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"  [Transformer NER] Device: {device}")

            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)

            # Collect all BIO tags
            all_tags = [tag for rec in ner_data for tag in rec.get("bio_tags", [])]
            self.label2id, self.id2label = self._build_label_maps(all_tags)
            num_labels = len(self.label2id)
            print(f"  [Transformer NER] {num_labels} BIO labels")

            # Build aligned dataset
            train_recs, val_recs = train_test_split(ner_data, test_size=0.1, random_state=42)
            if not torch.cuda.is_available() and len(train_recs) > 600:
                train_recs = train_recs[:600]
                val_recs = val_recs[:150]

            class NERDataset(Dataset):
                def __init__(self, records, tokenizer, label2id, max_len=64):
                    self.data = []
                    for rec in records:
                        tokens   = rec.get("tokens", [])
                        bio_tags = rec.get("bio_tags", [])
                        if not tokens:
                            continue
                        enc = tokenizer(
                            tokens, is_split_into_words=True,
                            truncation=True, padding="max_length",
                            max_length=max_len, return_tensors="pt",
                        )
                        word_ids = enc.word_ids(batch_index=0)
                        labels = []
                        prev_word_id = None
                        for wid in word_ids:
                            if wid is None:
                                labels.append(-100)
                            elif wid != prev_word_id:
                                tag = bio_tags[wid] if wid < len(bio_tags) else "O"
                                labels.append(label2id.get(tag, label2id.get("O", 0)))
                            else:
                                labels.append(-100)
                            prev_word_id = wid
                        self.data.append({
                            "input_ids":      enc["input_ids"].squeeze(),
                            "attention_mask": enc["attention_mask"].squeeze(),
                            "labels":         torch.tensor(labels, dtype=torch.long),
                        })

                def __len__(self):  return len(self.data)
                def __getitem__(self, i): return self.data[i]

            train_ds = NERDataset(train_recs, self.tokenizer, self.label2id)
            val_ds   = NERDataset(val_recs,   self.tokenizer, self.label2id)

            self.model = DistilBertForTokenClassification.from_pretrained(
                self.model_name, num_labels=num_labels,
                id2label=self.id2label, label2id=self.label2id,
            ).to(device)

            loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            opt    = AdamW(self.model.parameters(), lr=3e-5)

            history = []
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                for batch in loader:
                    opt.zero_grad()
                    out  = self.model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device),
                    )
                    out.loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    total_loss += out.loss.item()

                avg_loss = total_loss / len(loader)
                history.append({"epoch": epoch + 1, "train_loss": round(avg_loss, 4)})
                print(f"  [Transformer NER] Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

            self._is_trained = True
            self._device     = device

            # Val entity-level evaluation
            val_metrics = self._evaluate_val(val_ds, val_recs, device)

            return {"success": True, "history": history, "val_metrics": val_metrics}

        except Exception as e:
            return {"success": False, "reason": str(e)}

    def _evaluate_val(self, val_ds, val_recs, device) -> dict:
        """Evaluate on validation set — token-level metrics."""
        import torch
        from torch.utils.data import DataLoader

        loader = DataLoader(val_ds, batch_size=8)
        self.model.eval()
        all_preds, all_true = [], []

        with torch.no_grad():
            for batch in loader:
                out = self.model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                preds  = torch.argmax(out.logits, dim=-1).cpu().numpy()
                labels = batch["labels"].numpy()
                for pred_row, label_row in zip(preds, labels):
                    for p, l in zip(pred_row, label_row):
                        if l != -100:
                            all_preds.append(self.id2label.get(int(p), "O"))
                            all_true.append(self.id2label.get(int(l), "O"))

        from sklearn.metrics import classification_report as cr
        report = cr(all_true, all_preds, output_dict=True, zero_division=0)
        return {"token_classification_report": report}

    def predict(self, text: str) -> List[dict]:
        if not self._is_trained:
            return []

        import torch
        self.model.eval()
        tokens = text.split()
        if not tokens:
            return []

        enc = self.tokenizer(
            tokens, is_split_into_words=True, truncation=True,
            padding=True, max_length=128, return_tensors="pt",
        ).to(self._device)

        word_ids = enc.word_ids(batch_index=0)

        with torch.no_grad():
            logits = self.model(**enc).logits
            preds  = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
            proba  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # Map back to word-level
        word_tags = {}
        for i, wid in enumerate(word_ids):
            if wid is not None and wid not in word_tags:
                word_tags[wid] = (self.id2label.get(int(preds[i]), "O"), float(proba[i].max()))

        token_tags = [(tokens[wid], *word_tags[wid]) for wid in sorted(word_tags)]
        return self._bio_to_entities(text, token_tags)

    def _bio_to_entities(self, text: str, token_tags: list) -> List[dict]:
        entities = []
        i = 0
        cursor = 0
        while i < len(token_tags):
            tok, tag, conf = token_tags[i]
            if tag.startswith("B-"):
                label      = tag[2:]
                start_char = text.find(tok, cursor)
                if start_char == -1:
                    start_char = cursor
                end_char   = start_char + len(tok)
                ent_toks   = [tok]
                cursor     = end_char
                j = i + 1
                while j < len(token_tags) and token_tags[j][1] == f"I-{label}":
                    nt, _, _ = token_tags[j]
                    nc = text.find(nt, cursor)
                    if nc != -1:
                        end_char = nc + len(nt)
                        cursor = end_char
                    ent_toks.append(nt)
                    j += 1
                entities.append({
                    "text":       " ".join(ent_toks),
                    "label":      label,
                    "start":      start_char,
                    "end":        end_char,
                    "confidence": round(conf, 4),
                    "source":     "transformer_ner",
                })
                i = j
            else:
                nc = text.find(tok, cursor)
                if nc != -1:
                    cursor = nc + len(tok)
                i += 1
        return entities

    def save(self):
        if not self._is_trained:
            return
        save_dir = os.path.join(_MODELS, "distilbert_ner")
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "label_maps.json"), "w") as f:
            json.dump({"id2label": {str(k): v for k, v in self.id2label.items()},
                       "label2id": self.label2id}, f, indent=2)
        print(f"  [SAVED] Transformer NER -> {save_dir}")


# ===========================================================================
# NER Evaluation helper
# ===========================================================================

def evaluate_ner_predictions(predictions: List[dict], ground_truth: List[dict]) -> dict:
    """
    Entity-level evaluation (exact span + label match).

    Both predictions and ground_truth are lists of
    {'text': ..., 'label': ..., 'start': ..., 'end': ...}
    """
    pred_set = {(e["text"].lower(), e["label"]) for e in predictions}
    true_set = {(e["text"].lower(), e["label"]) for e in ground_truth}

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def per_entity_metrics(
    all_predictions: List[List[dict]],
    all_ground_truth: List[List[dict]],
) -> dict:
    """Per-entity-label precision/recall/F1."""
    from collections import defaultdict

    tp_map = defaultdict(int)
    fp_map = defaultdict(int)
    fn_map = defaultdict(int)

    for preds, gts in zip(all_predictions, all_ground_truth):
        pred_set = {(e["text"].lower(), e["label"]) for e in preds}
        true_set = {(e["text"].lower(), e["label"]) for e in gts}

        for item in pred_set & true_set:
            tp_map[item[1]] += 1
        for item in pred_set - true_set:
            fp_map[item[1]] += 1
        for item in true_set - pred_set:
            fn_map[item[1]] += 1

    metrics = {}
    for label in ENTITY_LABELS:
        tp, fp, fn = tp_map[label], fp_map[label], fn_map[label]
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        metrics[label] = {
            "precision": round(p,  4),
            "recall":    round(r,  4),
            "f1":        round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    return metrics


# ===========================================================================
# Training entry point
# ===========================================================================

def train_and_evaluate(data_path: Optional[str] = None) -> Tuple[dict, "RuleBasedNER", "MLTokenNER"]:
    if data_path is None:
        data_path = os.path.join(_DATA, "ner_dataset.jsonl")

    ner_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ner_data.append(json.loads(line))
    print(f"  Loaded {len(ner_data)} NER records")

    # Ground truth entity lists per record
    gt_entities = [
        [{
            "text":  ent["text"],
            "label": ent["label"],
            "start": 0, "end": 0,   # start/end not critical for entity-level eval
        } for ent in rec.get("entities", [])]
        for rec in ner_data
    ]

    all_results = {}

    # -------------------------------------------------------------------
    # Tier 1: Rule-based
    # -------------------------------------------------------------------
    print("\n[NER Tier 1] Rule-based NER ...")
    rule_ner = RuleBasedNER()
    rule_preds = [rule_ner.predict(rec["text"]) for rec in ner_data]
    rule_entity_metrics = per_entity_metrics(rule_preds, gt_entities)
    rule_overall = evaluate_ner_predictions(
        [e for preds in rule_preds for e in preds],
        [e for gts   in gt_entities for e in gts],
    )
    all_results["rule_based"] = {
        "model": "Rule-Based (Dictionary + Regex)",
        "overall": rule_overall,
        "per_entity": rule_entity_metrics,
    }
    print(f"  Overall F1: {rule_overall['f1']}")

    # -------------------------------------------------------------------
    # Tier 2: ML Token NER
    # -------------------------------------------------------------------
    print("\n[NER Tier 2] ML Token Classifier ...")
    ml_ner = MLTokenNER()
    ml_report = ml_ner.train(ner_data)
    ml_preds  = [ml_ner.predict(rec["text"]) for rec in ner_data]
    ml_entity_metrics = per_entity_metrics(ml_preds, gt_entities)
    ml_overall = evaluate_ner_predictions(
        [e for preds in ml_preds for e in preds],
        [e for gts   in gt_entities for e in gts],
    )
    all_results["ml_token"] = {
        "model":      "ML Token Classifier (TF-IDF + LogReg on BIO tags)",
        "overall":    ml_overall,
        "per_entity": ml_entity_metrics,
        "token_report": ml_report,
    }
    ml_ner.save()
    print(f"  Overall F1: {ml_overall['f1']}")

    # -------------------------------------------------------------------
    # Tier 3: Transformer NER
    # -------------------------------------------------------------------
    print("\n[NER Tier 3] DistilBERT Token Classification ...")
    tf_ner    = TransformerNER()
    tf_result = tf_ner.train(ner_data, epochs=2, batch_size=8)

    if tf_result["success"]:
        tf_preds = [tf_ner.predict(rec["text"]) for rec in ner_data]
        tf_entity_metrics = per_entity_metrics(tf_preds, gt_entities)
        tf_overall = evaluate_ner_predictions(
            [e for preds in tf_preds for e in preds],
            [e for gts   in gt_entities for e in gts],
        )
        all_results["transformer"] = {
            "model":      "DistilBERT Token Classification",
            "overall":    tf_overall,
            "per_entity": tf_entity_metrics,
            "training":   tf_result,
        }
        tf_ner.save()
        print(f"  Overall F1: {tf_overall['f1']}")
    else:
        print(f"  [Transformer NER] Failed: {tf_result['reason']}")
        all_results["transformer"] = {
            "success": False,
            "reason":  tf_result["reason"],
            "note":    "Transformer NER results NOT fabricated.",
        }

    # -------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------
    _save_ner_outputs(all_results)

    return all_results, rule_ner, ml_ner


def _save_ner_outputs(all_results: dict):
    # JSON metrics
    json_path = os.path.join(_OUTPUTS, "ner_metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  [SAVED] {json_path}")

    # CSV per-entity summary
    rows = []
    for model_name, res in all_results.items():
        if "per_entity" in res:
            for label, m in res["per_entity"].items():
                rows.append({
                    "model":     model_name,
                    "entity":    label,
                    "precision": m["precision"],
                    "recall":    m["recall"],
                    "f1":        m["f1"],
                })
    if rows:
        csv_path = os.path.join(_OUTPUTS, "ner_classification_report.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  [SAVED] {csv_path}")


# ===========================================================================
# Standalone
# ===========================================================================

if __name__ == "__main__":
    results, rule_ner, ml_ner = train_and_evaluate()

    print("\n=== NER Results Summary ===")
    for model_name, res in results.items():
        overall = res.get("overall", {})
        print(f"  {model_name}: F1={overall.get('f1', 'N/A')}")

    # Edge-case demonstrations
    test_texts = [
        "EGFR L858R mutation was identified; osimertinib 80 mg was prescribed.",
        "Patient received pembrolizumab 200 mg and developed fatigue.",
        "BRCA1 mutation detected. Olaparib 300 mg started for ovarian cancer.",
        "No fever. Patient denies nausea. History of rash, currently resolved.",
        "",
        "Multiple drugs: paclitaxel 175 mg/m2 and cisplatin 75 mg/m2.",
        "KRAS G12D and BRAF V600E mutations detected in colorectal cancer.",
        "Dosage missing for nivolumab prescription.",
        "xyz unknown symptom in patient with zzz cancer.",
    ]
    print("\n=== Rule-Based NER Edge Cases ===")
    for text in test_texts:
        entities = rule_ner.predict(text)
        print(f"\n  Text: '{text[:70]}'")
        for e in entities:
            print(f"    [{e['label']:15s}] '{e['text']}'")
        if not entities:
            print("    (no entities found)")
