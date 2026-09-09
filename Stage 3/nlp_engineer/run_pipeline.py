"""
run_pipeline.py
---------------
Master runner for Stage 3 — NLP Engineer.

Trains all models, generates all evaluation outputs, runs edge cases,
and produces a demo of the full pipeline.

Run from: Stage 3/nlp_engineer/
    python run_pipeline.py
    python run_pipeline.py --retrain
    python run_pipeline.py --edge-cases
    python run_pipeline.py --text "your clinical text here"
"""

import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from src.nlp_pipeline import OncologyNLPPipeline, run_edge_cases, EDGE_CASES


def main():
    parser = argparse.ArgumentParser(description="Oncology NLP Pipeline Runner")
    parser.add_argument("--retrain",    action="store_true", help="Force retrain all models")
    parser.add_argument("--text",       type=str,            help="Process single text")
    parser.add_argument("--edge-cases", action="store_true", help="Run edge-case test suite")
    parser.add_argument("--demo",       action="store_true", help="Run standard demo (default)")
    args = parser.parse_args()

    # Build / load pipeline
    pipeline = OncologyNLPPipeline.load(retrain=args.retrain)

    if args.text:
        result = pipeline.run(args.text)
        print(pipeline.format_output(result))
        return

    if args.edge_cases:
        run_edge_cases(pipeline)
        return

    # Default: run demo + edge cases
    demo_texts = [
        "Patient has EGFR L858R mutation. Osimertinib 80 mg once daily was prescribed. "
        "Patient developed severe fatigue and mild rash. No fever. Patient denies nausea.",

        "BRCA2 pathogenic variant detected. Olaparib 300 mg BID started for ovarian cancer. "
        "Patient reports persistent nausea and vomiting several times today.",

        "Difficulty breathing and chest tightness. Widespread rash. Confusion. "
        "Pembrolizumab 200 mg was last administered 3 days ago.",

        "Mild fatigue, slight nausea. Appetite slightly reduced. No other symptoms.",

        "KRAS G12D and TP53 mutations found. Paclitaxel 175 mg/m2 and cisplatin 75 mg/m2 "
        "planned for colorectal cancer.",
    ]

    print("\n" + "=" * 70)
    print("ONCOLOGY NLP PIPELINE — DEMO OUTPUTS")
    print("=" * 70)

    output_dir = os.path.join(_HERE, "outputs")
    demo_results = []

    for i, text in enumerate(demo_texts, 1):
        print(f"\n{'='*70}")
        print(f"[DEMO {i}] {text[:90]}{'...' if len(text) > 90 else ''}")
        print('='*70)
        result = pipeline.run(text)

        print(f"  URGENCY:  {result['urgency'].get('label')}  "
              f"(confidence: {result['urgency'].get('confidence', 0):.3f})")
        print(f"  MODEL:    {result['urgency'].get('model', 'N/A')}")

        print(f"\n  ENTITIES ({len(result['entities'])}):")
        for ent in result["entities"]:
            print(f"    [{ent['label']:15s}] '{ent['text']}' (conf={ent.get('confidence', 0):.2f})")

        print(f"\n  DRUG INFORMATION ({len(result['drug_information'])}):")
        for di in result["drug_information"]:
            print(f"    Drug: {di.get('matched_as', '?')} | {len(di.get('records', []))} record(s)")
            for r in di.get("records", [])[:1]:
                print(f"      AE: {r['adverse_event']} | Severity: {r['severity']} | Route: {r['route']}")

        print(f"\n  GENE/MUTATION INFO ({len(result['mutation_information'])}):")
        for mi in result["mutation_information"]:
            for rec in mi.get("records", [])[:1]:
                print(f"    {rec['gene']} | {rec['mutation']} | {rec['associated_cancer']}")

        print(f"\n  GUIDELINE MATCHES ({len(result['guideline_matches'])}):")
        for gm in result["guideline_matches"]:
            print(f"    [{gm['similarity_score']:.4f}] {gm['section']} — {gm['text'][:60]}...")

        if result["warnings"]:
            print(f"\n  WARNINGS: {result['warnings']}")
        if result["negation_detected"]:
            print("  [!] Negation detected in text")

        demo_results.append(result)

    # Save demo output
    demo_out_path = os.path.join(output_dir, "demo_pipeline_results.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(demo_out_path, "w") as f:
        json.dump(demo_results, f, indent=2, default=str)
    print(f"\n[SAVED] Demo results -> {demo_out_path}")

    # Run edge cases
    run_edge_cases(pipeline)

    print("\n" + "="*70)
    print("[OK] Stage 3 NLP Engineer — Complete")
    print(f"     Outputs in: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
