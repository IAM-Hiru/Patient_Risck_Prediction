# Stage 3 — EDA Report
### Oncology NLP Project | EDA Engineer

---

## 1. Dataset Overview

| Dataset | Records | Columns | Missing Values | Duplicate Rows |
|---|---|---|---|---|
| urgency_dataset | 1000 | 9 | 0 | 0 |
| drug_knowledge | 1000 | 7 | 0 | 0 |
| gene_mutation_dictionary | 8 | 3 | 0 | 0 |
| guideline_chunks | 6 | 6 | 0 | 0 |
| ner_dataset | 1000 | 5 | 0 | 0 |

---

## 2. Text Analysis (Urgency Dataset)

| Metric | Value |
|---|---|
| Avg Characters | 34.7 |
| Min Characters | 18 |
| Max Characters | 41 |
| Avg Words | 4.6 |
| Min Words | 3 |
| Max Words | 6 |
| Avg Sentences | 2.0 |
| Vocabulary Size | 39 |

> Clinical notes have a median of ~5 words.
> Distribution is approximately normal with a moderate right tail.

---

## 3. Urgency Class Distribution

| Class | Count | Percentage |
|---|---|---|
| LOW | 300 | 30.0% |
| MODERATE | 100 | 10.0% |
| HIGH | 300 | 30.0% |
| CRITICAL | 300 | 30.0% |

**Imbalance note:** The dataset should be inspected for significant skew between
LOW/MODERATE vs HIGH/CRITICAL. Class imbalance will require weighted loss functions
or oversampling strategies (e.g., SMOTE on embeddings).

### Per-Class Vocabulary Highlights

**LOW** — Top words: mild, fatigue, treatment, slight, nausea, reduced, appetite, skin, dryness
**MODERATE** — Top words: moderate, dizziness, affecting, walking
**HIGH** — Top words: persistent, vomiting, several, times, today, severe, diarrhea, weakness, new, fever
**CRITICAL** — Top words: difficulty, breathing, chest, tightness, confusion, staying, awake, widespread, rash, facial

> [!] *Frequency patterns above reflect dataset language only and are NOT clinical
> causal claims.*

---

## 4. NER Entity Analysis

| Entity Label | Count | % | Avg Char Length |
|---|---|---|---|
| DRUG_NAME | 800 | 28.6% | 10.8 |
| DOSAGE | 800 | 28.6% | 6.0 |
| GENE_MUTATION | 600 | 21.4% | 14.0 |
| ADVERSE_EVENT | 400 | 14.3% | 6.0 |
| CANCER_TYPE | 200 | 7.1% | 17.0 |

### Entity Examples (top-5 per class)

**DRUG_NAME**: pembrolizumab, osimertinib, olaparib, Trastuzumab
**DOSAGE**: 200 mg, 80 mg, 300 mg, 6 mg/kg
**GENE_MUTATION**: EGFR L858R, BRCA1 mutation, KRAS G12D mutation
**ADVERSE_EVENT**: fatigue, fever
**CANCER_TYPE**: colorectal cancer

**Observations:**
- Total entities annotated: **2800**
- Check for rare entity classes — they may require data augmentation.
- Verify BIO tag consistency (B-/I- prefix pairing) before model training.

---

## 5. Drug / Adverse Event Analysis

### Top Drugs
| Drug | Count |
|---|---|
| pembrolizumab | 125 |
| nivolumab | 125 |
| paclitaxel | 125 |
| cisplatin | 125 |
| doxorubicin | 125 |
| olaparib | 125 |
| osimertinib | 125 |
| trastuzumab | 125 |

### Severity Distribution
| moderate | 750 |
| high | 250 |

### Route Distribution
| iv | 750 |
| oral | 250 |

Cross-table of Drug x Adverse Event saved -> `outputs/eda/drug_ae_cross_table.csv`

---

## 6. Gene -> Mutation -> Cancer

| Gene | Mutation | Associated Cancer |
|---|---|---|
| BRCA1 | BRCA1 pathogenic variant | Breast/Ovarian |
| BRCA2 | BRCA2 pathogenic variant | Breast/Ovarian |
| EGFR | EGFR L858R | Lung |
| KRAS | KRAS G12D | Colorectal/Pancreatic |
| BRAF | BRAF V600E | Melanoma/Colorectal |
| ALK | ALK rearrangement | Lung |
| PIK3CA | PIK3CA H1047R | Breast |
| TP53 | TP53 missense variant | Multiple cancers |

---

## 7. Guideline Text Analysis

- Total guideline chunks: **6**
- Unique sources: **2**
- Unique sections: **6**
- Avg words per chunk: **13.5**

### Key Clinical Term Frequency

| Term | Occurrences |
|---|---|
| treatment | 3 |
| immune | 2 |
| reaction | 2 |
| adverse | 1 |
| fever | 1 |
| infusion | 1 |
| monitor | 1 |
| severe | 1 |
| respiratory | 0 |
| medication | 0 |
| dosage | 0 |
| toxicity | 0 |
| grade | 0 |
| interrupt | 0 |
| prophylaxis | 0 |

---

## 8. Data Quality

| Dataset | Issue | Count |
|---|---|---|
| urgency_dataset | Duplicate text rows | 990 |
| urgency_dataset | Empty text | 0 |
| urgency_dataset | Unusually long texts (>mean+3std) | 0 |
| urgency_dataset | Unusually short texts (<3 words) | 0 |
| ner_dataset | Records with zero entities | 0 |
| guideline_chunks | Empty text | 0 |

**Potential leakage check:** Verify that text fields in the urgency dataset do not
contain the label word itself (e.g., the word "critical" inside a CRITICAL-labelled
note could constitute label leakage).

---

## 9. Key Recommendations for NLP Engineer

### Preprocessing Strategy
- Lowercase + punctuation normalisation (preserve dosage patterns like "200 mg")
- Sentence splitting before NER tokenisation
- Preserve drug names and gene mutation identifiers as single tokens
- Handle short texts (<5 words) with special flags

### Model Choice
- **Urgency Classification**: Fine-tune `ClinicalBERT` or `BioBERT` for 4-class classification
- **NER**: Use token classification head on `BioBERT` with BIO tags already present
- Baseline: TF-IDF + Logistic Regression for comparison

### Class Balancing Strategy
- Use **class-weighted cross-entropy** if urgency distribution is skewed
- Consider **focal loss** for rare HIGH/CRITICAL classes
- Do NOT oversample raw text — oversample on embeddings if needed

### Evaluation Strategy
- Use **macro F1** as primary metric (handles imbalance)
- Report **per-class F1** — especially for HIGH and CRITICAL
- Use **entity-level F1** (exact span match) for NER evaluation
- Stratified k-fold cross-validation

### Difficult Cases to Test
- MODERATE vs HIGH boundary cases (most likely confusion zone)
- Texts with overlapping symptoms (e.g., fatigue appears in all urgency levels)
- Short texts (<10 words) — model may lack context
- Drug names with dosage variation (same drug, multiple dosages)
- Rare gene mutations not seen in training
- Guideline text with negations ("no fever", "does not have difficulty breathing")

---

*Report generated by the EDA Engineer — Stage 3 of the Oncology NLP Pipeline.*
*No medical claims are made from the frequency patterns observed above.*
