"""
preprocessing.py
----------------
Text preprocessing utilities for the Oncology NLP Pipeline.
Handles: cleaning, tokenisation, negation detection, abbreviation expansion.
"""

import re
import string
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Abbreviation / synonym map (oncology-focused)
# ---------------------------------------------------------------------------
ABBREV_MAP = {
    r"\bpt\b":     "patient",
    r"\bpts\b":    "patients",
    r"\bhx\b":     "history",
    r"\bdx\b":     "diagnosis",
    r"\btx\b":     "treatment",
    r"\brx\b":     "prescription",
    r"\bc/o\b":    "complains of",
    r"\bsob\b":    "shortness of breath",
    r"\bn/v\b":    "nausea vomiting",
    r"\bnv\b":     "nausea vomiting",
    r"\bha\b":     "headache",
    r"\bw/\b":     "with",
    r"\bw/o\b":    "without",
    r"\byo\b":     "year old",
    r"\biv\b":     "intravenous",
    r"\bpo\b":     "oral",
    r"\bqd\b":     "once daily",
    r"\bbid\b":    "twice daily",
    r"\btid\b":    "three times daily",
    r"\bqid\b":    "four times daily",
    r"\bprn\b":    "as needed",
    r"\bae\b":     "adverse event",
    r"\birAE\b":   "immune-related adverse event",
    r"\bca\b":     "cancer",
    r"\bchemo\b":  "chemotherapy",
    r"\bimmunotx\b": "immunotherapy",
    r"\bgr\b":     "grade",
    r"\bmo\b":     "months",
    r"\bwk\b":     "weeks",
    r"\bwks\b":    "weeks",
    r"\bd/c\b":    "discontinue",
    r"\bwbc\b":    "white blood cell",
    r"\brbc\b":    "red blood cell",
    r"\bcrp\b":    "c-reactive protein",
    r"\bct\b":     "computed tomography",
    r"\bmri\b":    "magnetic resonance imaging",
    r"\becog\b":   "eastern cooperative oncology group",
    r"\bpd\b":     "progressive disease",
    r"\bpr\b":     "partial response",
    r"\bcr\b":     "complete response",
    r"\bsd\b":     "stable disease",
    r"\bnsclc\b":  "non-small cell lung cancer",
    r"\bsclc\b":   "small cell lung cancer",
    r"\btnbc\b":   "triple-negative breast cancer",
}

TYPO_MAP = {
    r"\bfatige\b":     "fatigue",
    r"\bbrething\b":   "breathing",
    r"\bvommiting\b":  "vomiting",
    r"\bvomitting\b":  "vomiting",
    r"\bnausia\b":     "nausea",
    r"\bdiarhea\b":    "diarrhea",
    r"\bdiarrhoea\b":  "diarrhea",
    r"\bdiziness\b":   "dizziness",
    r"\bdizzyness\b":  "dizziness",
    r"\bpembroluzimab\b": "pembrolizumab",
    r"\bosimertanib\b":   "osimertinib",
    r"\boxaliplatine\b":  "oxaliplatin",
    r"\bheadach\b":    "headache",
    r"\bshorness\b":   "shortness",
    r"\btightnes\b":   "tightness",
}

# ---------------------------------------------------------------------------
# Negation cue patterns
# ---------------------------------------------------------------------------
NEGATION_CUES = [
    r"\bno\b",
    r"\bnot\b",
    r"\bnever\b",
    r"\bdenies\b",
    r"\bdenied\b",
    r"\bdeny\b",
    r"\bdenying\b",
    r"\bdenial of\b",
    r"\bwithout\b",
    r"\babsence of\b",
    r"\bno evidence of\b",
    r"\bnegative for\b",
    r"\bnegative\b",
    r"\bruled out\b",
    r"\bexcludes\b",
    r"\bfree of\b",
    r"\bresolved\b",
    r"\bpreviously\b",
    r"\bhistory of\b",         # historical, not current
    r"\bhx\b",
    r"\bhx of\b",
    r"\bhistorically\b",
    r"\bcurrently resolved\b",
    r"\bnon\b",
    r"\bzero\b",
    r"\bclear of\b",
]

# Negation scope: tokens after cue to consider negated
NEGATION_SCOPE_TOKENS = 10


# ---------------------------------------------------------------------------
# Core cleaning
# ---------------------------------------------------------------------------

def correct_typos(text: str) -> str:
    """Correct common medical typographical errors."""
    for pattern, replacement in TYPO_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def expand_abbreviations(text: str) -> str:
    """Replace common medical abbreviations with their full forms and fix common typos."""
    text = text.lower()
    for pattern, replacement in ABBREV_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = correct_typos(text)
    return text


def clean_text(text: str, expand_abbrev: bool = True) -> str:
    """
    Clean clinical text for NLP models.
    - Lowercases
    - Expands abbreviations
    - Removes excess whitespace
    - Normalises punctuation
    - Preserves dosage patterns (e.g., '200 mg', '80mg')
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # Expand abbreviations before lowercasing interferes
    if expand_abbrev:
        text = expand_abbreviations(text)
    else:
        text = text.lower()

    # Remove URLs and emails
    text = re.sub(r"http\S+|www\.\S+|\S+@\S+", " ", text)

    # Normalise newlines / tabs
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Keep alphanumerics, spaces, common medical punctuation
    # Preserve patterns like "200 mg", "L858R", "G12D", "BRCA1"
    text = re.sub(r"[^a-z0-9\s\-\/\.]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenise(text: str) -> List[str]:
    """Simple whitespace tokeniser."""
    return text.split()


# ---------------------------------------------------------------------------
# Negation detection
# ---------------------------------------------------------------------------

class NegationDetector:
    """
    Rule-based negation detector using a sliding-window scope approach.

    For each token window of NEGATION_SCOPE_TOKENS following a negation cue,
    flags all tokens as negated.

    Usage:
        detector = NegationDetector()
        result = detector.annotate("Patient denies nausea and vomiting.")
        # result -> [('Patient', False), ('denies', True), ('nausea', True), ...]
    """

    def __init__(self, scope: int = NEGATION_SCOPE_TOKENS):
        self.scope = scope
        self._cue_patterns = [re.compile(p, re.IGNORECASE) for p in NEGATION_CUES]

    def _is_negation_cue(self, token: str) -> bool:
        clean = re.sub(r"[^\w\s]", "", token).strip()
        if not clean:
            return False
        for pat in self._cue_patterns:
            if pat.search(clean):
                return True
        return False

    def annotate(self, text: str) -> List[Tuple[str, bool]]:
        """
        Returns list of (token, is_negated) tuples.
        Tokens within clause or scope window following a negation cue are flagged True.
        """
        tokens = text.split()
        negated = [False] * len(tokens)
        i = 0
        while i < len(tokens):
            clean_tok = re.sub(r"[^\w]", "", tokens[i]).lower()
            is_cue = False
            for pat in self._cue_patterns:
                if pat.search(clean_tok):
                    is_cue = True
                    break

            # Also check 2-word cues like "history of", "negative for", "no evidence"
            if not is_cue and i + 1 < len(tokens):
                clean_two = clean_tok + " " + re.sub(r"[^\w]", "", tokens[i + 1]).lower()
                for pat in self._cue_patterns:
                    if pat.search(clean_two):
                        is_cue = True
                        break

            # Also check post-negation like "rash, resolved" or "result was negative"
            if clean_tok in ("resolved", "negative", "unremarkable", "normal") and i > 0:
                # Mark preceding 5 tokens as negated/resolved
                for j in range(max(0, i - 5), i + 1):
                    negated[j] = True

            if is_cue:
                # Extend negation across the clause (up to 15 tokens or sentence terminator)
                for j in range(i, min(i + 15, len(tokens))):
                    negated[j] = True
                    if tokens[j].endswith((".", ";", "!", "?")) and j > i:
                        break
            i += 1
        return list(zip(tokens, negated))

    def has_negation(self, text: str) -> bool:
        """Returns True if any negation cue is present in text."""
        for pat in self._cue_patterns:
            if pat.search(text):
                return True
        return False

    def get_active_symptoms(self, entities: list) -> list:
        """
        Filter entity list to remove negated mentions.

        Parameters
        ----------
        entities : list of dicts with keys 'text', 'label', 'start', 'end'

        Returns
        -------
        list of entities not under negation scope
        """
        active = []
        for ent in entities:
            snippet = ent.get("context", ent.get("text", ""))
            if not self.has_negation(snippet):
                ent["negated"] = False
                active.append(ent)
            else:
                # Check more carefully: is the entity itself after a cue?
                annotated = self.annotate(snippet)
                ent_tokens = set(re.sub(r"[^\w]", "", t).lower() for t in ent["text"].split())
                negated_flag = False
                for tok, neg in annotated:
                    clean_tok = re.sub(r"[^\w]", "", tok).lower()
                    if clean_tok in ent_tokens and neg:
                        negated_flag = True
                        break
                ent["negated"] = negated_flag
                if not negated_flag:
                    active.append(ent)
        return active

    def mask_negated_text(self, text: str) -> str:
        """
        Mask tokens following negation cues within scope with a neutral token
        so TF-IDF / bag-of-words models do not trigger on negated words.
        """
        if not text:
            return ""
        annotated = self.annotate(text)
        masked_tokens = []
        for (tok, is_neg) in annotated:
            if is_neg and re.search(r"[a-zA-Z]", tok):
                if self._is_negation_cue(tok):
                    masked_tokens.append(tok)
                else:
                    masked_tokens.append("negated_absent")
            else:
                masked_tokens.append(tok)
        return " ".join(masked_tokens)


# ---------------------------------------------------------------------------
# Edge-case guards
# ---------------------------------------------------------------------------

def is_empty(text: str) -> bool:
    return not isinstance(text, str) or text.strip() == ""


def handle_empty(text: str, fallback_label: str = "LOW") -> dict:
    """Return a safe fallback response for empty / None input."""
    return {
        "input_text":    text,
        "resolved_text": "",
        "coreferences":  [],
        "cleaned_text":  "",
        "error": "empty_input",
        "urgency": {"label": fallback_label, "class_id": 0, "confidence": 0.0},
        "entities": [],
        "drug_information": [],
        "mutation_information": [],
        "guideline_matches": [],
        "negation_detected": False,
        "warnings": ["empty_input: no text provided"],
    }


# ---------------------------------------------------------------------------
# Spelling normalisation (lightweight edit-distance approach)
# ---------------------------------------------------------------------------

DRUG_ALIASES = {
    "pembro":         "pembrolizumab",
    "nivo":           "nivolumab",
    "atezo":          "atezolizumab",
    "bev":            "bevacizumab",
    "trastu":         "trastuzumab",
    "herceptin":      "trastuzumab",
    "keytruda":       "pembrolizumab",
    "opdivo":         "nivolumab",
    "tagrisso":       "osimertinib",
    "lynparza":       "olaparib",
    "taxol":          "paclitaxel",
    "taxotere":       "docetaxel",
    "adriamycin":     "doxorubicin",
    "platinol":       "cisplatin",
}


def normalise_drug_name(name: str) -> str:
    """Map common brand names / abbreviations to generic names."""
    n = name.lower().strip()
    return DRUG_ALIASES.get(n, n)


# ---------------------------------------------------------------------------
# Clinical Coreference Resolution (Anaphora Resolver)
# ---------------------------------------------------------------------------

class ClinicalCoreferenceResolver:
    """
    Resolves clinical anaphoric expressions and pronouns to antecedent entities.
    Handles:
    - Drug references: 'the drug', 'the medication', 'this chemotherapy', 'the therapy', 'this agent' -> resolved to last mentioned drug
    - Pronouns: 'it', 'this' in therapeutic/tolerability context -> resolved to last mentioned drug
    - Symptom references: 'the symptom', 'the reaction', 'this toxicity', 'the adverse event' -> resolved to last mentioned adverse event
    - Mutation references: 'the mutation', 'this variant', 'the biomarker' -> resolved to last mentioned genomic alteration
    """

    DRUG_ANAPHORS = [
        r"\b(?:the|this)\s+(?:drug|medication|chemotherapy|chemo|regimen|therapy|agent|infusion|treatment)\b",
        r"\b(?:the|this)\s+oral\s+(?:drug|agent|medication)\b",
        r"\b(?:the|this)\s+iv\s+(?:drug|infusion|medication)\b"
    ]

    SYMPTOM_ANAPHORS = [
        r"\b(?:the|this)\s+(?:symptom|reaction|toxicity|adverse event|side effect|presentation|complaint)\b",
        r"\b(?:these|those)\s+(?:symptoms|reactions|toxicities|side effects)\b"
    ]

    MUTATION_ANAPHORS = [
        r"\b(?:the|this)\s+(?:mutation|biomarker|variant|alteration|genomic finding)\b"
    ]

    KNOWN_DRUGS = [
        "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab", "ipilimumab",
        "osimertinib", "gefitinib", "erlotinib", "alectinib", "lorlatinib",
        "trastuzumab", "pertuzumab", "doxorubicin", "paclitaxel", "docetaxel",
        "cisplatin", "carboplatin", "oxaliplatin", "capecitabine", "fluorouracil",
        "5-fu", "gemcitabine", "irinotecan", "etoposide", "cyclophosphamide",
        "olaparib", "rucaparib", "niraparib", "tamoxifen", "letrozole", "anastrozole"
    ]

    KNOWN_SYMPTOMS = [
        "fever", "neutropenia", "dyspnea", "shortness of breath", "chest pain",
        "nausea", "vomiting", "diarrhea", "rash", "neuropathy", "fatigue",
        "colitis", "hepatitis", "pneumonitis", "anaphylaxis", "hypotension",
        "pruritus", "stomatitis", "myalgia", "arthralgia", "dizziness"
    ]

    KNOWN_MUTATIONS = [
        "egfr", "kras", "braf", "alk", "ros1", "her2", "erbb2", "brca1", "brca2",
        "pik3ca", "tp53", "l858r", "t790m", "v600e", "g12c", "g12d", "exon 19"
    ]

    def resolve(self, text: str) -> Tuple[str, List[dict]]:
        """
        Resolves coreferences across sentences and returns:
        (resolved_text, coreference_chain)
        """
        if not text or not isinstance(text, str):
            return text, []

        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        resolved_sentences = []
        chains = []

        last_drug = None
        last_symptom = None
        last_mutation = None

        for sent in raw_sentences:
            s_lower = sent.lower()

            # 1. Update antecedent entities if present in current sentence
            for d in self.KNOWN_DRUGS:
                if re.search(r'\b' + re.escape(d) + r'\b', s_lower):
                    last_drug = d.title()
                    break

            for sym in self.KNOWN_SYMPTOMS:
                if re.search(r'\b' + re.escape(sym) + r'\b', s_lower):
                    last_symptom = sym
                    break

            for mut in self.KNOWN_MUTATIONS:
                if re.search(r'\b' + re.escape(mut) + r'\b', s_lower):
                    last_mutation = mut.upper()
                    break

            modified_sent = sent

            # 2. Resolve Drug Anaphora
            if last_drug:
                for pattern in self.DRUG_ANAPHORS:
                    matches = list(re.finditer(pattern, modified_sent, flags=re.IGNORECASE))
                    for m in matches:
                        anaphor = m.group(0)
                        chains.append({
                            "anaphor": anaphor,
                            "antecedent": last_drug,
                            "entity_type": "DRUG_NAME"
                        })
                        modified_sent = re.sub(pattern, f"{last_drug}", modified_sent, count=1, flags=re.IGNORECASE)

            # 3. Resolve Symptom Anaphora
            if last_symptom:
                for pattern in self.SYMPTOM_ANAPHORS:
                    matches = list(re.finditer(pattern, modified_sent, flags=re.IGNORECASE))
                    for m in matches:
                        anaphor = m.group(0)
                        chains.append({
                            "anaphor": anaphor,
                            "antecedent": last_symptom,
                            "entity_type": "ADVERSE_EVENT"
                        })
                        modified_sent = re.sub(pattern, f"{last_symptom}", modified_sent, count=1, flags=re.IGNORECASE)

            # 4. Resolve Mutation Anaphora
            if last_mutation:
                for pattern in self.MUTATION_ANAPHORS:
                    matches = list(re.finditer(pattern, modified_sent, flags=re.IGNORECASE))
                    for m in matches:
                        anaphor = m.group(0)
                        chains.append({
                            "anaphor": anaphor,
                            "antecedent": last_mutation,
                            "entity_type": "GENE_MUTATION"
                        })
                        modified_sent = re.sub(pattern, f"{last_mutation}", modified_sent, count=1, flags=re.IGNORECASE)

            # 5. Resolve pronoun 'it' in clinical action contexts
            if last_drug and re.search(r'\b(it)\b', modified_sent, flags=re.IGNORECASE):
                if re.search(r'\b(?:stopped|tolerated|discontinued|caused|held|reduced|administered|infused|effective)\b', modified_sent, flags=re.IGNORECASE):
                    chains.append({
                        "anaphor": "it",
                        "antecedent": last_drug,
                        "entity_type": "DRUG_NAME"
                    })
                    modified_sent = re.sub(r'\b(it)\b', f"{last_drug}", modified_sent, count=1, flags=re.IGNORECASE)

            resolved_sentences.append(modified_sent)

        resolved_text = " ".join(resolved_sentences)
        return resolved_text, chains
