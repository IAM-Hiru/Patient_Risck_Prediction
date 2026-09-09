"""
drug_lookup.py
--------------
Drug information lookup from drug_knowledge.csv.

Supports:
  - Exact match
  - Case-insensitive match
  - Brand name / alias normalisation
  - Fuzzy fallback (edit distance)
  - Unknown drug handling
"""

import os
import re
import json
import pandas as pd
from typing import Optional

# Path resolution
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "..", "data_engineer", "data", "processed")


def _edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein distance."""
    a, b = a.lower(), b.lower()
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


class DrugLookup:
    """
    Lookup drug adverse events, severity, route, and dosage
    from the oncology drug knowledge base.

    Parameters
    ----------
    data_path : str, optional
        Path to drug_knowledge.csv. Defaults to processed data directory.
    fuzzy_threshold : int
        Maximum edit distance for fuzzy matching (default 3).
    """

    def __init__(self, data_path: Optional[str] = None, fuzzy_threshold: int = 3):
        if data_path is None:
            data_path = os.path.join(_DATA, "drug_knowledge.csv")

        self.df = pd.read_csv(data_path)
        self.df["drug_name_lower"] = self.df["drug_name"].str.lower().str.strip()
        self.df = self.df.drop_duplicates(
            subset=["drug_name_lower", "adverse_event", "severity", "dosage"]
        )
        self.fuzzy_threshold = fuzzy_threshold

        # Build quick lookup index
        self._index: dict[str, list] = {}
        for _, row in self.df.iterrows():
            key = row["drug_name_lower"]
            self._index.setdefault(key, []).append(row.to_dict())

        # Alias map (brand -> generic)
        self._aliases = {
            "keytruda":     "pembrolizumab",
            "opdivo":       "nivolumab",
            "tagrisso":     "osimertinib",
            "lynparza":     "olaparib",
            "taxol":        "paclitaxel",
            "taxotere":     "docetaxel",
            "adriamycin":   "doxorubicin",
            "platinol":     "cisplatin",
            "herceptin":    "trastuzumab",
            "pembro":       "pembrolizumab",
            "nivo":         "nivolumab",
        }

    def _resolve_name(self, name: str) -> str:
        """Resolve alias to canonical name."""
        n = name.lower().strip()
        return self._aliases.get(n, n)

    def _fuzzy_match(self, name: str) -> Optional[str]:
        """Find closest drug name by edit distance."""
        best_key, best_dist = None, self.fuzzy_threshold + 1
        for key in self._index:
            d = _edit_distance(name, key)
            if d < best_dist:
                best_dist = d
                best_key = key
        return best_key if best_dist <= self.fuzzy_threshold else None

    def lookup(self, drug_name: str) -> dict:
        """
        Look up a drug by name.

        Returns
        -------
        dict with keys:
            drug, matched_as, records, found, match_type
        """
        if not drug_name or not isinstance(drug_name, str):
            return {"found": False, "drug": drug_name, "error": "empty_input"}

        resolved = self._resolve_name(drug_name)

        # Exact match
        if resolved in self._index:
            records = self._index[resolved]
            return {
                "found":      True,
                "drug":       drug_name,
                "matched_as": resolved,
                "match_type": "exact",
                "records":    self._format_records(records),
            }

        # Fuzzy match
        fuzzy_key = self._fuzzy_match(resolved)
        if fuzzy_key:
            records = self._index[fuzzy_key]
            return {
                "found":      True,
                "drug":       drug_name,
                "matched_as": fuzzy_key,
                "match_type": "fuzzy",
                "records":    self._format_records(records),
            }

        return {
            "found":      False,
            "drug":       drug_name,
            "matched_as": None,
            "match_type": None,
            "records":    [],
            "note":       "Drug not found in knowledge base. Consult clinical reference.",
        }

    def _format_records(self, rows: list) -> list:
        out = []
        for r in rows:
            out.append({
                "drug_name":     r.get("drug_name", ""),
                "adverse_event": r.get("adverse_event", ""),
                "reaction":      r.get("reaction", ""),
                "severity":      r.get("severity", ""),
                "route":         r.get("route", ""),
                "dosage":        r.get("dosage", ""),
            })
        return out

    def lookup_multiple(self, drug_names: list) -> list:
        """Batch lookup for multiple drugs."""
        return [self.lookup(d) for d in drug_names]

    def get_adverse_events(self, drug_name: str) -> list:
        """Return list of adverse events for a drug."""
        result = self.lookup(drug_name)
        if result["found"]:
            return list({r["adverse_event"] for r in result["records"]})
        return []

    def get_severity_summary(self, drug_name: str) -> dict:
        """Return severity distribution for a drug."""
        result = self.lookup(drug_name)
        if result["found"]:
            sevs = [r["severity"] for r in result["records"]]
            counts = {}
            for s in sevs:
                counts[s] = counts.get(s, 0) + 1
            return counts
        return {}

    def all_drug_names(self) -> list:
        """Return all known drug names."""
        return sorted(self._index.keys())


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

def run_edge_case_tests(lookup: DrugLookup):
    test_cases = [
        "pembrolizumab",        # exact
        "PEMBROLIZUMAB",        # case variation
        "keytruda",             # brand name alias
        "pembro",               # abbreviation alias
        "pembrolizumaab",       # spelling error (fuzzy)
        "unknown_drug_xyz",     # unknown
        "",                     # empty
        "osimertinib",          # exact
        "tagrisso",             # brand name
        "olaparib",             # exact
    ]
    print("\n=== Drug Lookup Edge Cases ===")
    for drug in test_cases:
        r = lookup.lookup(drug)
        status = "FOUND" if r["found"] else "NOT FOUND"
        matched = r.get("matched_as", "-")
        mtype = r.get("match_type", "-")
        n_records = len(r.get("records", []))
        print(f"  [{status}] '{drug}' -> '{matched}' ({mtype}) | {n_records} records")


if __name__ == "__main__":
    lu = DrugLookup()
    run_edge_case_tests(lu)

    # Sample lookup
    result = lu.lookup("pembrolizumab")
    print("\nSample lookup - pembrolizumab:")
    print(json.dumps(result["records"][:2], indent=2))
