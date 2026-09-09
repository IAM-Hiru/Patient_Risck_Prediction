"""
guideline_retrieval.py
----------------------
TF-IDF cosine-similarity retrieval over clinical guideline chunks.

Input:  clinical text query
Output: Top-k most relevant guideline chunks with similarity scores.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "..", "data_engineer", "data", "processed")


GUIDELINE_SYNONYMS = {
    "colitis": "Immune-related adverse events immune therapy immune-mediated reactions",
    "hepatitis": "Immune-related adverse events immune therapy immune-mediated reactions",
    "checkpoint inhibitor": "Immune-related adverse events immune therapy immune-mediated reactions",
    "immunotherapy": "Immune-related adverse events immune therapy immune-mediated reactions",
    "neuropathy": "Neuropathy peripheral neuropathy numbness tingling nerve impairment sensory neurotoxicity",
    "tingling": "Neuropathy peripheral neuropathy numbness tingling nerve impairment sensory neurotoxicity",
    "numbness": "Neuropathy peripheral neuropathy numbness tingling nerve impairment sensory neurotoxicity",
    "diarrhea": "Diarrhea severity loose stools hydration electrolyte balance loperamide",
    "abdominal pain": "Diarrhea severity loose stools hydration electrolyte balance loperamide",
    "infusion reaction": "severe infusion reaction emergency hypersensitivity",
    "dyspnea": "shortness of breath respiratory symptoms",
    "fever": "fever during systemic therapy infection neutropenia",
    "nausea": "chemotherapy nausea antiemetic hydration",
    "vomiting": "chemotherapy nausea antiemetic hydration",
}


class GuidelineRetriever:
    """
    Retrieves the most relevant oncology clinical guideline chunks
    for a given clinical text query using TF-IDF + cosine similarity.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        top_k: int = 3,
        min_score: float = 0.0,
    ):
        if data_path is None:
            data_path = os.path.join(_DATA, "guideline_chunks.csv")

        self.df = pd.read_csv(data_path)
        self.top_k = top_k
        self.min_score = min_score

        # Enrich retrieval corpus: combine section + text
        self.df["corpus_text"] = (
            self.df.get("section", pd.Series([""] * len(self.df))).fillna("").astype(str)
            + " "
            + self.df["text"].fillna("").astype(str)
        ).str.strip()

        # Fit TF-IDF on the guideline corpus
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=10000,
            stop_words="english",
        )
        self._corpus_matrix = self.vectorizer.fit_transform(
            self.df["corpus_text"].tolist()
        )

        print(f"  [GuidelineRetriever] Indexed {len(self.df)} guideline chunks.")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        k = top_k or self.top_k

        if not isinstance(query, str) or query.strip() == "":
            return []

        # Expand query using domain synonyms
        query_lower = query.lower()
        expanded_terms = [query]
        for term, expansion in GUIDELINE_SYNONYMS.items():
            if term in query_lower:
                expanded_terms.append(expansion)
        expanded_query = " ".join(expanded_terms)

        q_vec = self.vectorizer.transform([expanded_query])
        scores = cosine_similarity(q_vec, self._corpus_matrix).flatten()

        # Get top-k indices
        top_idx = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_idx:
            score = float(scores[idx])
            if score < self.min_score:
                continue
            row = self.df.iloc[idx]
            results.append({
                "document_id":    row.get("document_id", ""),
                "source":         row.get("source", ""),
                "section":        row.get("section", ""),
                "text":           row.get("text", ""),
                "similarity_score": round(score, 4),
            })

        return results

    def retrieve_for_symptoms(self, symptoms: List[str], top_k: Optional[int] = None) -> List[dict]:
        """
        Retrieve guidelines relevant to a list of symptom strings.
        Combines symptoms into a single query.
        """
        query = " ".join(symptoms)
        return self.retrieve(query, top_k=top_k)

    def retrieve_for_entities(self, entities: List[dict], top_k: Optional[int] = None) -> List[dict]:
        """
        Retrieve guidelines relevant to detected NER entities.
        Builds query from entity texts.
        """
        query = " ".join(e.get("text", "") for e in entities)
        return self.retrieve(query, top_k=top_k)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

def run_edge_case_tests(retriever: GuidelineRetriever):
    queries = [
        "patient has severe infusion reaction with fever",
        "respiratory difficulty and chest tightness",
        "immune-related adverse event grade 3",
        "dosage adjustment for osimertinib",
        "nausea and vomiting after chemotherapy",
        "EGFR mutation treatment",
        "",                             # empty query
        "xyzabc unknown random text",   # no match expected
        "fever",                        # single token
    ]
    print("\n=== Guideline Retrieval Edge Cases ===")
    for q in queries:
        results = retriever.retrieve(q, top_k=2)
        print(f"  Query: '{q[:50]}' -> {len(results)} result(s)")
        for r in results:
            print(f"    [{r['similarity_score']:.4f}] {r['section']} | {r['text'][:60]}...")


if __name__ == "__main__":
    gr = GuidelineRetriever()
    run_edge_case_tests(gr)

    print("\n--- Sample retrieval ---")
    results = gr.retrieve("Patient developed severe fever after immunotherapy")
    for r in results:
        print(f"[{r['similarity_score']:.4f}] {r['source']} / {r['section']}")
        print(f"  {r['text']}")
        print()
