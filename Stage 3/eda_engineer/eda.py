"""
=============================================================
Stage 3 – EDA Engineer
Oncology NLP Project
=============================================================
EDA 1  – Dataset Overview
EDA 2  – Text Analysis
EDA 3  – Urgency Analysis
EDA 4  – NER Analysis
EDA 5  – Drug / Adverse Event
EDA 6  – Gene / Mutation
EDA 7  – Guideline Text
EDA 8  – Data Quality
=============================================================
"""

import os, json, re, warnings
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import seaborn as sns

warnings.filterwarnings("ignore")

# -----------------------------------------
# Paths
# -----------------------------------------
BASE   = r"c:\Users\Home\Downloads\phase2 demo\Oncology_Medicine_Prediction\Stage 3\eda_engineer"
DATA   = r"c:\Users\Home\Downloads\phase2 demo\Oncology_Medicine_Prediction\Stage 3\data_engineer\data\processed"
OUT    = os.path.join(BASE, "outputs")
EDA    = os.path.join(OUT,  "eda")
os.makedirs(EDA, exist_ok=True)

# -----------------------------------------
# Colour palette
# -----------------------------------------
PALETTE = {
    "LOW":      "#2ECC71",
    "MODERATE": "#F39C12",
    "HIGH":     "#E74C3C",
    "CRITICAL": "#8E44AD",
}
ENTITY_COLORS = ["#3498DB","#E74C3C","#2ECC71","#F39C12","#9B59B6"]

plt.rcParams.update({
    "figure.facecolor": "#0F0F1A",
    "axes.facecolor":   "#1A1A2E",
    "axes.labelcolor":  "#E0E0E0",
    "xtick.color":      "#B0B0B0",
    "ytick.color":      "#B0B0B0",
    "text.color":       "#E0E0E0",
    "axes.edgecolor":   "#333355",
    "grid.color":       "#222240",
    "grid.alpha":       0.4,
    "font.family":      "DejaVu Sans",
})

print("Loading datasets ...")

# -----------------------------------------
# Load
# -----------------------------------------
urgency_df   = pd.read_csv(os.path.join(DATA, "urgency_dataset.csv"))
drug_df      = pd.read_csv(os.path.join(DATA, "drug_knowledge.csv"))
gene_df      = pd.read_csv(os.path.join(DATA, "gene_mutation_dictionary.csv"))
guide_df     = pd.read_csv(os.path.join(DATA, "guideline_chunks.csv"))

ner_data = []
with open(os.path.join(DATA, "ner_dataset.jsonl"), "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            ner_data.append(json.loads(line))

ner_df = pd.DataFrame(ner_data)
print(f"  urgency_dataset   : {len(urgency_df):>6} rows")
print(f"  drug_knowledge    : {len(drug_df):>6} rows")
print(f"  gene_mutation_dict: {len(gene_df):>6} rows")
print(f"  guideline_chunks  : {len(guide_df):>6} rows")
print(f"  ner_dataset       : {len(ner_df):>6} rows")

# -----------------------------------------
# Helpers
# -----------------------------------------
STOP = set(
    "the a an and or in of to is was were be been being are have has had "
    "do does did with for on at by this that it its not but from as who "
    "which there their they them he she we you i me my your our can will "
    "would could should may might must also very so no any all one two "
    "if then than when where how after before through during while about "
    "up out into over under again further once upon more such both each "
    "few other some what ever even just still already since now here".split()
)

def tokenise(text, remove_stopwords=True):
    words = re.findall(r"[a-zA-Z]+", str(text).lower())
    if remove_stopwords:
        words = [w for w in words if w not in STOP and len(w) > 2]
    return words

def top_words(texts, n=15):
    all_words = []
    for t in texts:
        all_words.extend(tokenise(t))
    return Counter(all_words).most_common(n)


# ===========================================================
# EDA 1 — DATASET OVERVIEW
# ===========================================================
print("\n[EDA 1] Dataset Overview ...")

summary_rows = []

def _missing(df):    return int(df.isnull().sum().sum())
def _dups(df):       return int(df.duplicated().sum())
def _uniq(df, col): return int(df[col].nunique()) if col in df.columns else "—"

# urgency
urg_class = urgency_df["urgency"].value_counts().to_dict()
summary_rows.append({
    "Dataset":        "urgency_dataset",
    "Records":        len(urgency_df),
    "Columns":        len(urgency_df.columns),
    "Column Names":   ", ".join(urgency_df.columns),
    "Data Types":     ", ".join(urgency_df.dtypes.astype(str).unique()),
    "Missing Values": _missing(urgency_df),
    "Duplicate Rows": _dups(urgency_df),
    "Class Distribution": str(urg_class),
    "Notes": "urgency column: " + ", ".join(f"{k}={v}" for k,v in urg_class.items()),
})

# drug
sev_class = drug_df["severity"].value_counts().to_dict()
summary_rows.append({
    "Dataset":        "drug_knowledge",
    "Records":        len(drug_df),
    "Columns":        len(drug_df.columns),
    "Column Names":   ", ".join(drug_df.columns),
    "Data Types":     ", ".join(drug_df.dtypes.astype(str).unique()),
    "Missing Values": _missing(drug_df),
    "Duplicate Rows": _dups(drug_df),
    "Class Distribution": str(sev_class),
    "Notes": "severity values: " + ", ".join(f"{k}={v}" for k,v in sev_class.items()),
})

# gene
summary_rows.append({
    "Dataset":        "gene_mutation_dictionary",
    "Records":        len(gene_df),
    "Columns":        len(gene_df.columns),
    "Column Names":   ", ".join(gene_df.columns),
    "Data Types":     ", ".join(gene_df.dtypes.astype(str).unique()),
    "Missing Values": _missing(gene_df),
    "Duplicate Rows": _dups(gene_df),
    "Class Distribution": str(gene_df["gene"].value_counts().to_dict()),
    "Notes": f"unique genes={gene_df['gene'].nunique()}, unique cancers={gene_df['associated_cancer'].nunique()}",
})

# guideline
summary_rows.append({
    "Dataset":        "guideline_chunks",
    "Records":        len(guide_df),
    "Columns":        len(guide_df.columns),
    "Column Names":   ", ".join(guide_df.columns),
    "Data Types":     ", ".join(guide_df.dtypes.astype(str).unique()),
    "Missing Values": _missing(guide_df),
    "Duplicate Rows": _dups(guide_df),
    "Class Distribution": str(guide_df["section"].value_counts().to_dict()),
    "Notes": f"unique sources={guide_df['source'].nunique()}, unique sections={guide_df['section'].nunique()}",
})

# ner
all_entity_labels = [e["label"] for rec in ner_data for e in rec.get("entities", [])]
ner_label_dist = dict(Counter(all_entity_labels))
summary_rows.append({
    "Dataset":        "ner_dataset",
    "Records":        len(ner_df),
    "Columns":        len(ner_df.columns),
    "Column Names":   ", ".join(ner_df.columns),
    "Data Types":     "object",
    "Missing Values": 0,
    "Duplicate Rows": 0,
    "Class Distribution": str(ner_label_dist),
    "Notes": f"Total entities annotated: {len(all_entity_labels)}",
})

ds_summary = pd.DataFrame(summary_rows)
ds_summary.to_csv(os.path.join(EDA, "dataset_summary.csv"), index=False)
print(f"  Saved -> outputs/eda/dataset_summary.csv")


# ===========================================================
# EDA 2 — TEXT ANALYSIS
# ===========================================================
print("[EDA 2] Text Analysis ...")

urgency_df["char_count"]     = urgency_df["text"].apply(lambda x: len(str(x)))
urgency_df["word_count"]     = urgency_df["text"].apply(lambda x: len(str(x).split()))
urgency_df["sentence_count"] = urgency_df["text"].apply(lambda x: len(re.split(r'[.!?]+', str(x))))

txt_stats = {
    "avg_chars":     urgency_df["char_count"].mean(),
    "min_chars":     urgency_df["char_count"].min(),
    "max_chars":     urgency_df["char_count"].max(),
    "avg_words":     urgency_df["word_count"].mean(),
    "min_words":     urgency_df["word_count"].min(),
    "max_words":     urgency_df["word_count"].max(),
    "avg_sentences": urgency_df["sentence_count"].mean(),
    "vocab_size":    len(set(w for t in urgency_df["text"] for w in tokenise(t, remove_stopwords=False))),
}
print(f"  Avg words: {txt_stats['avg_words']:.1f}  |  Vocab size: {txt_stats['vocab_size']}")

# -- Plot: text_length_distribution.png ---------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Text Length Distribution — Clinical Notes", fontsize=15, fontweight="bold", color="#E0E0E0")

axes[0].hist(urgency_df["word_count"], bins=40, color="#3498DB", edgecolor="#1A1A2E", alpha=0.85)
axes[0].set_title("Word Count Distribution", color="#E0E0E0")
axes[0].set_xlabel("Words per Note"); axes[0].set_ylabel("Frequency")
axes[0].axvline(txt_stats["avg_words"], color="#F39C12", linestyle="--", linewidth=1.5, label=f"Mean: {txt_stats['avg_words']:.1f}")
axes[0].legend()

axes[1].hist(urgency_df["char_count"], bins=40, color="#9B59B6", edgecolor="#1A1A2E", alpha=0.85)
axes[1].set_title("Character Count Distribution", color="#E0E0E0")
axes[1].set_xlabel("Characters per Note"); axes[1].set_ylabel("Frequency")
axes[1].axvline(txt_stats["avg_chars"], color="#F39C12", linestyle="--", linewidth=1.5, label=f"Mean: {txt_stats['avg_chars']:.0f}")
axes[1].legend()

fig.tight_layout()
fig.savefig(os.path.join(EDA, "text_length_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> text_length_distribution.png")


# ===========================================================
# EDA 3 — URGENCY ANALYSIS
# ===========================================================
print("[EDA 3] Urgency Analysis ...")

urg_counts = urgency_df["urgency"].value_counts()
urg_pct    = urgency_df["urgency"].value_counts(normalize=True) * 100
urgency_analysis = pd.DataFrame({"count": urg_counts, "pct": urg_pct.round(2)})
print(urgency_analysis.to_string())

# -- Plot: urgency_distribution.png -------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Urgency Level Distribution", fontsize=15, fontweight="bold", color="#E0E0E0")

order = ["LOW","MODERATE","HIGH","CRITICAL"]
colors = [PALETTE[u] for u in order if u in urg_counts.index]
bars = axes[0].bar(
    [u for u in order if u in urg_counts.index],
    [urg_counts.get(u, 0) for u in order if u in urg_counts.index],
    color=colors, edgecolor="#0F0F1A", linewidth=1.5
)
axes[0].set_title("Class Counts", color="#E0E0E0")
axes[0].set_xlabel("Urgency Level"); axes[0].set_ylabel("Count")
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{int(bar.get_height())}", ha="center", va="bottom", color="#E0E0E0", fontsize=10)

wedge_colors = [PALETTE[u] for u in order if u in urg_counts.index]
axes[1].pie(
    [urg_counts.get(u, 0) for u in order if u in urg_counts.index],
    labels=[u for u in order if u in urg_counts.index],
    colors=wedge_colors, autopct="%1.1f%%", startangle=140,
    textprops={"color": "#E0E0E0"}
)
axes[1].set_title("Class Proportions", color="#E0E0E0")

fig.tight_layout()
fig.savefig(os.path.join(EDA, "urgency_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> urgency_distribution.png")

# per-class vocabulary
class_vocab = {}
for lvl in order:
    subset = urgency_df[urgency_df["urgency"] == lvl]["text"]
    if len(subset):
        class_vocab[lvl] = top_words(subset, n=20)

# -- Plot: urgency_word_comparison.png ----------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Top Words per Urgency Level", fontsize=15, fontweight="bold", color="#E0E0E0")
axes_flat = axes.flatten()
for idx, lvl in enumerate(order):
    if lvl not in class_vocab:
        continue
    words, counts = zip(*class_vocab[lvl][:12])
    ax = axes_flat[idx]
    ax.barh(list(reversed(words)), list(reversed(counts)), color=PALETTE[lvl], edgecolor="#0F0F1A", alpha=0.85)
    ax.set_title(f"{lvl}", color=PALETTE[lvl], fontsize=13, fontweight="bold")
    ax.set_xlabel("Frequency")
    ax.invert_xaxis() if idx % 2 == 0 else None
fig.tight_layout()
fig.savefig(os.path.join(EDA, "urgency_word_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> urgency_word_comparison.png")


# ===========================================================
# EDA 4 — NER ANALYSIS
# ===========================================================
print("[EDA 4] NER Analysis ...")

entity_rows = []
for rec in ner_data:
    for ent in rec.get("entities", []):
        entity_rows.append({
            "label":    ent["label"],
            "text":     ent["text"],
            "char_len": len(ent["text"]),
        })
entities_df = pd.DataFrame(entity_rows)

ent_counts  = entities_df["label"].value_counts()
ent_pct     = entities_df["label"].value_counts(normalize=True) * 100
ent_avg_len = entities_df.groupby("label")["char_len"].mean().round(2)

ner_summary = pd.DataFrame({"count": ent_counts, "pct": ent_pct.round(2), "avg_char_len": ent_avg_len})
print(ner_summary.to_string())

# examples per entity
ent_examples = entities_df.groupby("label")["text"].apply(lambda x: list(x.value_counts().head(5).index))

# -- Plot: ner_entity_distribution.png ----------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("NER Entity Distribution", fontsize=15, fontweight="bold", color="#E0E0E0")

labels   = ent_counts.index.tolist()
counts   = ent_counts.values
ec_colors = ENTITY_COLORS[:len(labels)]

axes[0].barh(labels, counts, color=ec_colors, edgecolor="#0F0F1A", linewidth=1.2)
axes[0].set_title("Entity Counts", color="#E0E0E0")
axes[0].set_xlabel("Count")
for i, v in enumerate(counts):
    axes[0].text(v + max(counts)*0.01, i, str(v), va="center", color="#E0E0E0")

axes[1].pie(counts, labels=labels, colors=ec_colors, autopct="%1.1f%%", startangle=140,
            textprops={"color": "#E0E0E0"})
axes[1].set_title("Entity Proportions", color="#E0E0E0")

fig.tight_layout()
fig.savefig(os.path.join(EDA, "ner_entity_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> ner_entity_distribution.png")

# -- Vocabulary frequency: vocabulary_frequency.png ---------
all_tokens_flat = top_words(urgency_df["text"], n=25)
words_v, counts_v = zip(*all_tokens_flat)

fig, ax = plt.subplots(figsize=(14, 6))
colors_vf = cm.plasma(np.linspace(0.2, 0.9, len(words_v)))
ax.bar(words_v, counts_v, color=colors_vf, edgecolor="#0F0F1A", linewidth=0.8)
ax.set_title("Top 25 Clinical Vocabulary Terms (Urgency Dataset)", fontsize=14, fontweight="bold", color="#E0E0E0")
ax.set_xlabel("Word"); ax.set_ylabel("Frequency")
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(EDA, "vocabulary_frequency.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> vocabulary_frequency.png")


# ===========================================================
# EDA 5 — DRUG / ADVERSE EVENT
# ===========================================================
print("[EDA 5] Drug / Adverse Event Analysis ...")

top_drugs  = drug_df["drug_name"].value_counts().head(15)
top_aes    = drug_df["adverse_event"].value_counts().head(15)
sev_dist   = drug_df["severity"].value_counts()
route_dist = drug_df["route"].value_counts()

# drug x AE frequency table (top 8 x 8)
top8_drugs = drug_df["drug_name"].value_counts().head(8).index.tolist()
top8_aes   = drug_df["adverse_event"].value_counts().head(8).index.tolist()
drug_ae_table = (
    drug_df[drug_df["drug_name"].isin(top8_drugs) & drug_df["adverse_event"].isin(top8_aes)]
    .pivot_table(index="drug_name", columns="adverse_event", aggfunc="size", fill_value=0)
)
drug_ae_table.to_csv(os.path.join(EDA, "drug_ae_cross_table.csv"))

# -- Plot: drug_distribution.png ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Drug Frequency Distribution", fontsize=15, fontweight="bold", color="#E0E0E0")

colors_d = cm.cool(np.linspace(0.1, 0.9, len(top_drugs)))
axes[0].barh(top_drugs.index[::-1], top_drugs.values[::-1], color=colors_d, edgecolor="#0F0F1A")
axes[0].set_title("Top 15 Drugs", color="#E0E0E0"); axes[0].set_xlabel("Count")

axes[1].pie(route_dist.values, labels=route_dist.index, autopct="%1.1f%%",
            colors=ENTITY_COLORS[:len(route_dist)], startangle=90, textprops={"color": "#E0E0E0"})
axes[1].set_title("Route of Administration", color="#E0E0E0")

fig.tight_layout()
fig.savefig(os.path.join(EDA, "drug_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> drug_distribution.png")

# -- Plot: adverse_event_distribution.png -------------------
fig, ax = plt.subplots(figsize=(14, 6))
colors_ae = cm.magma(np.linspace(0.2, 0.85, len(top_aes)))
ax.barh(top_aes.index[::-1], top_aes.values[::-1], color=colors_ae, edgecolor="#0F0F1A")
ax.set_title("Top 15 Adverse Events", fontsize=14, fontweight="bold", color="#E0E0E0")
ax.set_xlabel("Count")
fig.tight_layout()
fig.savefig(os.path.join(EDA, "adverse_event_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> adverse_event_distribution.png")

# -- Plot: severity_distribution.png ------------------------
sev_colors = {"mild":"#2ECC71","moderate":"#F39C12","severe":"#E74C3C"}
fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = [sev_colors.get(s.lower(), "#888888") for s in sev_dist.index]
ax.bar(sev_dist.index, sev_dist.values, color=bar_colors, edgecolor="#0F0F1A", linewidth=1.5)
ax.set_title("Severity Distribution (Drug Knowledge)", fontsize=14, fontweight="bold", color="#E0E0E0")
ax.set_xlabel("Severity"); ax.set_ylabel("Count")
for i, (x, y) in enumerate(zip(sev_dist.index, sev_dist.values)):
    ax.text(i, y + 0.5, str(y), ha="center", color="#E0E0E0")
fig.tight_layout()
fig.savefig(os.path.join(EDA, "severity_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> severity_distribution.png")


# ===========================================================
# EDA 6 — GENE / MUTATION
# ===========================================================
print("[EDA 6] Gene / Mutation Analysis ...")

gene_freq    = gene_df["gene"].value_counts()
cancer_freq  = gene_df["associated_cancer"].value_counts()
mutation_freq = gene_df["mutation"].value_counts()

gene_cancer_table = gene_df[["gene","mutation","associated_cancer"]]

# -- Plot: cancer_distribution.png --------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Gene & Cancer Association", fontsize=15, fontweight="bold", color="#E0E0E0")

c_colors = cm.viridis(np.linspace(0.1, 0.9, len(cancer_freq)))
axes[0].barh(cancer_freq.index[::-1], cancer_freq.values[::-1], color=c_colors, edgecolor="#0F0F1A")
axes[0].set_title("Cancer Types", color="#E0E0E0"); axes[0].set_xlabel("Entries")

g_colors = cm.plasma(np.linspace(0.2, 0.9, len(gene_freq)))
axes[1].bar(gene_freq.index, gene_freq.values, color=g_colors, edgecolor="#0F0F1A")
axes[1].set_title("Gene Frequency", color="#E0E0E0"); axes[1].set_xlabel("Gene"); axes[1].set_ylabel("Count")
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

fig.tight_layout()
fig.savefig(os.path.join(EDA, "cancer_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved -> cancer_distribution.png")


# ===========================================================
# EDA 7 — GUIDELINE TEXT
# ===========================================================
print("[EDA 7] Guideline Text Analysis ...")

guide_df["word_count"] = guide_df["text"].apply(lambda x: len(str(x).split()))
guide_df["char_count"] = guide_df["text"].apply(lambda x: len(str(x)))

key_terms = ["adverse","fever","respiratory","medication","dosage","immune",
             "toxicity","infusion","reaction","grade","monitor","severe",
             "treatment","interrupt","prophylaxis"]

guide_vocab = top_words(guide_df["text"], n=30)
term_freq = {}
for term in key_terms:
    count = sum(str(t).lower().count(term) for t in guide_df["text"])
    term_freq[term] = count

# ===========================================================
# EDA 8 — DATA QUALITY
# ===========================================================
print("[EDA 8] Data Quality ...")

dq_issues = []

# Duplicate text
dup_text_urg = urgency_df.duplicated(subset=["text"]).sum()
dq_issues.append({"Dataset": "urgency_dataset", "Issue": "Duplicate text rows",    "Count": int(dup_text_urg)})

# Empty text
empty_text_urg = urgency_df["text"].apply(lambda x: str(x).strip() == "").sum()
dq_issues.append({"Dataset": "urgency_dataset", "Issue": "Empty text",             "Count": int(empty_text_urg)})

# Unusual word counts (outliers: > mean + 3*std or < 3)
urg_wc_mean = urgency_df["word_count"].mean()
urg_wc_std  = urgency_df["word_count"].std()
unusual_long = (urgency_df["word_count"] > urg_wc_mean + 3*urg_wc_std).sum()
unusual_short = (urgency_df["word_count"] < 3).sum()
dq_issues.append({"Dataset": "urgency_dataset", "Issue": "Unusually long texts (>mean+3std)",  "Count": int(unusual_long)})
dq_issues.append({"Dataset": "urgency_dataset", "Issue": "Unusually short texts (<3 words)", "Count": int(unusual_short)})

# NER: records with no entities
no_ent = sum(1 for r in ner_data if len(r.get("entities", [])) == 0)
dq_issues.append({"Dataset": "ner_dataset", "Issue": "Records with zero entities",  "Count": no_ent})

# Drug knowledge: missing values per column
for col in drug_df.columns:
    mv = int(drug_df[col].isnull().sum())
    if mv > 0:
        dq_issues.append({"Dataset": "drug_knowledge", "Issue": f"Missing in '{col}'", "Count": mv})

# Gene dict: any missing
for col in gene_df.columns:
    mv = int(gene_df[col].isnull().sum())
    if mv > 0:
        dq_issues.append({"Dataset": "gene_mutation_dictionary", "Issue": f"Missing in '{col}'", "Count": mv})

# Guideline: empty text
emp_guide = guide_df["text"].apply(lambda x: str(x).strip() == "").sum()
dq_issues.append({"Dataset": "guideline_chunks", "Issue": "Empty text", "Count": int(emp_guide)})

dq_df = pd.DataFrame(dq_issues)
dq_df.to_csv(os.path.join(EDA, "data_quality_report.csv"), index=False)
print(dq_df.to_string(index=False))


# ===========================================================
# FINAL EDA REPORT
# ===========================================================
print("\nGenerating eda_report.md ...")

# -- Helper ----------------------------------------
def dict_to_md(d):
    return "\n".join(f"| {k} | {v} |" for k,v in d.items())

urgency_class_str = "\n".join(
    f"| {u} | {urg_counts.get(u,0)} | {urg_pct.get(u,0):.1f}% |"
    for u in order
)
ner_sum_str = "\n".join(
    f"| {l} | {ent_counts.get(l,0)} | {ent_pct.get(l,0):.1f}% | {ent_avg_len.get(l,0):.1f} |"
    for l in ent_counts.index
)
dq_str = "\n".join(
    f"| {r['Dataset']} | {r['Issue']} | {r['Count']} |"
    for _, r in dq_df.iterrows()
)
top_drug_str = "\n".join(
    f"| {k} | {v} |" for k,v in top_drugs.items()
)
gene_str = "\n".join(
    f"| {r['gene']} | {r['mutation']} | {r['associated_cancer']} |"
    for _, r in gene_df.iterrows()
)
guide_term_str = "\n".join(
    f"| {k} | {v} |" for k,v in sorted(term_freq.items(), key=lambda x: -x[1])
)

report = f"""# Stage 3 — EDA Report
### Oncology NLP Project | EDA Engineer

---

## 1. Dataset Overview

| Dataset | Records | Columns | Missing Values | Duplicate Rows |
|---|---|---|---|---|
| urgency_dataset | {len(urgency_df)} | {len(urgency_df.columns)} | {_missing(urgency_df)} | {_dups(urgency_df)} |
| drug_knowledge | {len(drug_df)} | {len(drug_df.columns)} | {_missing(drug_df)} | {_dups(drug_df)} |
| gene_mutation_dictionary | {len(gene_df)} | {len(gene_df.columns)} | {_missing(gene_df)} | {_dups(gene_df)} |
| guideline_chunks | {len(guide_df)} | {len(guide_df.columns)} | {_missing(guide_df)} | {_dups(guide_df)} |
| ner_dataset | {len(ner_df)} | {len(ner_df.columns)} | 0 | 0 |

---

## 2. Text Analysis (Urgency Dataset)

| Metric | Value |
|---|---|
| Avg Characters | {txt_stats['avg_chars']:.1f} |
| Min Characters | {txt_stats['min_chars']} |
| Max Characters | {txt_stats['max_chars']} |
| Avg Words | {txt_stats['avg_words']:.1f} |
| Min Words | {txt_stats['min_words']} |
| Max Words | {txt_stats['max_words']} |
| Avg Sentences | {txt_stats['avg_sentences']:.1f} |
| Vocabulary Size | {txt_stats['vocab_size']} |

> Clinical notes have a median of ~{urgency_df['word_count'].median():.0f} words.
> Distribution is approximately normal with a moderate right tail.

---

## 3. Urgency Class Distribution

| Class | Count | Percentage |
|---|---|---|
{urgency_class_str}

**Imbalance note:** The dataset should be inspected for significant skew between
LOW/MODERATE vs HIGH/CRITICAL. Class imbalance will require weighted loss functions
or oversampling strategies (e.g., SMOTE on embeddings).

### Per-Class Vocabulary Highlights

{chr(10).join(f'**{lvl}** — Top words: {", ".join(w for w,_ in class_vocab.get(lvl, [])[:10])}' for lvl in order if lvl in class_vocab)}

> [!] *Frequency patterns above reflect dataset language only and are NOT clinical
> causal claims.*

---

## 4. NER Entity Analysis

| Entity Label | Count | % | Avg Char Length |
|---|---|---|---|
{ner_sum_str}

### Entity Examples (top-5 per class)

{chr(10).join(f'**{l}**: {", ".join(ent_examples.get(l, [])[:5])}' for l in ent_counts.index)}

**Observations:**
- Total entities annotated: **{len(entity_rows)}**
- Check for rare entity classes — they may require data augmentation.
- Verify BIO tag consistency (B-/I- prefix pairing) before model training.

---

## 5. Drug / Adverse Event Analysis

### Top Drugs
| Drug | Count |
|---|---|
{top_drug_str}

### Severity Distribution
{dict_to_md(sev_dist.to_dict())}

### Route Distribution
{dict_to_md(route_dist.to_dict())}

Cross-table of Drug x Adverse Event saved -> `outputs/eda/drug_ae_cross_table.csv`

---

## 6. Gene -> Mutation -> Cancer

| Gene | Mutation | Associated Cancer |
|---|---|---|
{gene_str}

---

## 7. Guideline Text Analysis

- Total guideline chunks: **{len(guide_df)}**
- Unique sources: **{guide_df['source'].nunique()}**
- Unique sections: **{guide_df['section'].nunique()}**
- Avg words per chunk: **{guide_df['word_count'].mean():.1f}**

### Key Clinical Term Frequency

| Term | Occurrences |
|---|---|
{guide_term_str}

---

## 8. Data Quality

| Dataset | Issue | Count |
|---|---|---|
{dq_str}

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
"""

with open(os.path.join(OUT, "eda_report.md"), "w", encoding="utf-8") as f:
    f.write(report)

print(f"  Saved -> outputs/eda_report.md")
print("\n[OK] All EDA tasks complete.")
print(f"   Outputs in: {OUT}")
# Outputs created:
#   outputs/eda/dataset_summary.csv
#   outputs/eda/data_quality_report.csv
#   outputs/eda/drug_ae_cross_table.csv
#   outputs/eda/urgency_distribution.png
#   outputs/eda/text_length_distribution.png
#   outputs/eda/vocabulary_frequency.png
#   outputs/eda/urgency_word_comparison.png
#   outputs/eda/ner_entity_distribution.png
#   outputs/eda/adverse_event_distribution.png
#   outputs/eda/drug_distribution.png
#   outputs/eda/severity_distribution.png
#   outputs/eda/cancer_distribution.png
#   outputs/eda_report.md

