"""
gene_lookup.py
--------------
Gene / mutation information lookup from gene_mutation_dictionary.csv.

Supports:
  - Gene name lookup
  - Mutation string lookup
  - Partial / fuzzy matching
  - Unknown gene / mutation handling
"""

import os
import re
import json
import pandas as pd
from typing import Optional, List


_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "..", "data_engineer", "data", "processed")


class GeneLookup:
    """
    Lookup gene -> mutation -> associated cancer from the
    gene_mutation_dictionary.csv knowledge base.

    Parameters
    ----------
    data_path : str, optional
        Path to gene_mutation_dictionary.csv.
    """

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            data_path = os.path.join(_DATA, "gene_mutation_dictionary.csv")

        self.df = pd.read_csv(data_path)
        self.df["gene_lower"]     = self.df["gene"].str.lower().str.strip()
        self.df["mutation_lower"] = self.df["mutation"].str.lower().str.strip()

        # Indices
        self._gene_index: dict     = {}
        self._mutation_index: dict = {}
        for _, row in self.df.iterrows():
            entry = {
                "gene":             row["gene"],
                "mutation":         row["mutation"],
                "associated_cancer": row["associated_cancer"],
            }
            self._gene_index.setdefault(row["gene_lower"], []).append(entry)
            # Index by mutation tokens too
            for tok in row["mutation_lower"].split():
                if len(tok) > 2:
                    self._mutation_index.setdefault(tok, []).append(entry)

        # Common abbreviation / synonym map
        self._gene_aliases = {
            "brca":    ["brca1", "brca2"],
            "her2":    ["erbb2"],
            "egfr":    ["egfr"],
            "pdl1":    ["cd274"],
            "pd-l1":   ["cd274"],
        }

    def lookup_gene(self, gene: str) -> dict:
        """Look up all entries for a gene name."""
        if not gene or not isinstance(gene, str):
            return {"found": False, "gene": gene, "error": "empty_input"}

        key = gene.lower().strip()
        records = self._gene_index.get(key, [])

        # Try aliases
        if not records:
            for alias_key, targets in self._gene_aliases.items():
                if key == alias_key:
                    for t in targets:
                        records.extend(self._gene_index.get(t, []))

        if records:
            return {
                "found":   True,
                "gene":    gene,
                "records": records,
            }

        # Partial match: gene starts with query
        partial = [
            row for gk, rows in self._gene_index.items()
            if gk.startswith(key) for row in rows
        ]
        if partial:
            return {
                "found":      True,
                "gene":       gene,
                "match_type": "partial",
                "records":    partial,
            }

        return {
            "found":   False,
            "gene":    gene,
            "records": [],
            "note":    "Gene not found in knowledge base.",
        }

    def lookup_mutation(self, mutation: str) -> dict:
        """Look up entries by mutation string (e.g., 'EGFR L858R', 'G12D')."""
        if not mutation or not isinstance(mutation, str):
            return {"found": False, "mutation": mutation, "error": "empty_input"}

        key = mutation.lower().strip()

        # Full mutation string match
        matches = []
        for _, row in self.df.iterrows():
            if key in row["mutation_lower"] or row["mutation_lower"] in key:
                matches.append({
                    "gene":             row["gene"],
                    "mutation":         row["mutation"],
                    "associated_cancer": row["associated_cancer"],
                })

        if matches:
            return {"found": True, "mutation": mutation, "records": matches}

        # Token-level match
        tokens = key.split()
        token_hits = []
        for tok in tokens:
            if len(tok) > 3:
                token_hits.extend(self._mutation_index.get(tok, []))

        # Deduplicate
        seen = set()
        unique_hits = []
        for h in token_hits:
            k = (h["gene"], h["mutation"])
            if k not in seen:
                seen.add(k)
                unique_hits.append(h)

        if unique_hits:
            return {
                "found":      True,
                "mutation":   mutation,
                "match_type": "token",
                "records":    unique_hits,
            }

        return {
            "found":   False,
            "mutation": mutation,
            "records": [],
            "note":    "Mutation not found in knowledge base.",
        }

    def lookup_auto(self, text: str) -> List[dict]:
        """
        Auto-detect gene / mutation mentions in free text and look them up.

        Scans for known gene names and mutation patterns (e.g., G12D, L858R,
        V600E, H1047R, missense, rearrangement, pathogenic).

        Returns a list of lookup results.
        """
        results = []
        text_lower = text.lower()

        # Check each known gene
        for gene_key in self._gene_index:
            pattern = r"\b" + re.escape(gene_key) + r"\b"
            if re.search(pattern, text_lower):
                r = self.lookup_gene(gene_key)
                if r["found"] and r not in results:
                    results.append(r)

        # Check mutation tokens (e.g. L858R, G12D, V600E)
        mut_pattern = re.compile(r"\b[A-Z]\d{2,4}[A-Z]?\b")
        for m in mut_pattern.finditer(text):
            r = self.lookup_mutation(m.group())
            if r["found"] and r not in results:
                results.append(r)

        return results

    def all_genes(self) -> List[str]:
        return sorted(self._gene_index.keys())

    def gene_cancer_map(self) -> dict:
        """Return dict of gene -> list of associated cancers."""
        gcm = {}
        for gene, rows in self._gene_index.items():
            gcm[gene] = list({r["associated_cancer"] for r in rows})
        return gcm


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

def run_edge_case_tests(lu: GeneLookup):
    test_cases = [
        ("gene",     "EGFR"),
        ("gene",     "BRCA1"),
        ("gene",     "BRCA2"),
        ("gene",     "KRAS"),
        ("gene",     "brca"),          # alias
        ("gene",     "UNKNOWN_GENE"),  # unknown
        ("gene",     ""),              # empty
        ("mutation", "EGFR L858R"),
        ("mutation", "KRAS G12D"),
        ("mutation", "L858R"),         # partial
        ("mutation", "V600E"),
        ("mutation", "UNKNOWN_MUT"),   # unknown
        ("auto",     "Patient has EGFR L858R mutation and started osimertinib."),
        ("auto",     "BRCA2 pathogenic variant detected. KRAS G12D also present."),
        ("auto",     "No known mutations identified."),  # negation
    ]
    print("\n=== Gene Lookup Edge Cases ===")
    for mode, query in test_cases:
        if mode == "gene":
            r = lu.lookup_gene(query)
            status = "FOUND" if r["found"] else "NOT FOUND"
            n = len(r.get("records", []))
            print(f"  [GENE]     [{status}] '{query}' -> {n} records")
        elif mode == "mutation":
            r = lu.lookup_mutation(query)
            status = "FOUND" if r["found"] else "NOT FOUND"
            n = len(r.get("records", []))
            print(f"  [MUTATION] [{status}] '{query}' -> {n} records")
        elif mode == "auto":
            r = lu.lookup_auto(query)
            print(f"  [AUTO]     '{query[:50]}...' -> {len(r)} hit(s)")


if __name__ == "__main__":
    lu = GeneLookup()
    run_edge_case_tests(lu)

    print("\nGene-Cancer Map:")
    gcm = lu.gene_cancer_map()
    for gene, cancers in gcm.items():
        print(f"  {gene.upper()}: {', '.join(cancers)}")
