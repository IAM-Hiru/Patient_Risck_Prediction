"""
urgency_classifier.py
---------------------
Urgency text classifier for the Oncology NLP Pipeline.

MODEL 1 — BASELINE:   TF-IDF + Logistic Regression (always runs)
MODEL 2 — TRANSFORMER: DistilBERT fine-tuned classifier (runs if torch+transformers available)

The baseline model is the production fallback.
Transformer results are clearly labelled and only reported if training succeeds.
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Optional

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "..", "data_engineer", "data", "processed")
_MODELS = os.path.join(_HERE, "..", "models", "urgency")
_OUTPUTS = os.path.join(_HERE, "..", "outputs")

LABEL_MAP = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
ID_MAP    = {0: "LOW", 1: "MODERATE", 2: "HIGH", 3: "CRITICAL"}

os.makedirs(_MODELS, exist_ok=True)
os.makedirs(_OUTPUTS, exist_ok=True)


# ===========================================================================
# Preprocessing (inline, no circular import)
# ===========================================================================

import re

def _clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    try:
        from preprocessing import expand_abbreviations
        text = expand_abbreviations(text)
    except ImportError:
        text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-\/\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ===========================================================================
# BASELINE — TF-IDF + Logistic Regression
# ===========================================================================

class UrgencyBaselineClassifier:
    """
    TF-IDF + Logistic Regression urgency classifier.

    Handles class imbalance via class_weight='balanced'.
    Supports 4-class prediction: LOW, MODERATE, HIGH, CRITICAL.
    """

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.label_map = LABEL_MAP
        self.id_map    = ID_MAP
        self.classes_  = list(LABEL_MAP.keys())
        self._is_trained = False

    def build(self) -> Pipeline:
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=8000,
                analyzer="word",
            )),
            ("clf", LogisticRegression(
                C=2.5,
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="multinomial",
                random_state=42,
            )),
        ])
        return self.pipeline

    def train(self, X_train: list, y_train: list) -> "UrgencyBaselineClassifier":
        if self.pipeline is None:
            self.build()
        self.pipeline.fit(X_train, y_train)
        self._is_trained = True
        return self

    def predict(self, texts: list) -> np.ndarray:
        cleaned = [_clean(t) for t in texts]
        return self.pipeline.predict(cleaned)

    def predict_proba(self, texts: list) -> np.ndarray:
        cleaned = [_clean(t) for t in texts]
        return self.pipeline.predict_proba(cleaned)

    def predict_single(self, text: str) -> dict:
        """
        Predict urgency for a single clinical text.

        Returns
        -------
        dict with keys: label, class_id, confidence, all_scores
        """
        if not isinstance(text, str) or text.strip() == "":
            return {
                "label": "LOW", "class_id": 0,
                "confidence": 0.0, "all_scores": {},
                "warning": "empty_input",
            }

        cleaned  = [_clean(text)]
        pred     = self.pipeline.predict(cleaned)[0]
        proba    = self.pipeline.predict_proba(cleaned)[0]
        classes  = self.pipeline.classes_

        scores = {self.id_map.get(int(c), str(c)): round(float(p), 4)
                  for c, p in zip(classes, proba)}

        return {
            "label":      self.id_map.get(int(pred), str(pred)),
            "class_id":   int(pred),
            "confidence": round(float(max(proba)), 4),
            "all_scores": scores,
            "model":      "baseline_tfidf_lr",
        }

    def evaluate(self, X_test: list, y_test: list) -> dict:
        y_pred  = self.predict(X_test)
        labels  = sorted(set(y_test) | set(y_pred))
        id_names = [self.id_map.get(l, str(l)) for l in labels]

        metrics = {
            "model":        "TF-IDF + Logistic Regression (Baseline)",
            "accuracy":     round(accuracy_score(y_test, y_pred), 4),
            "precision_macro":   round(precision_score(y_test, y_pred, average="macro",  zero_division=0), 4),
            "recall_macro":      round(recall_score   (y_test, y_pred, average="macro",  zero_division=0), 4),
            "f1_macro":          round(f1_score       (y_test, y_pred, average="macro",  zero_division=0), 4),
            "precision_weighted":round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "recall_weighted":   round(recall_score   (y_test, y_pred, average="weighted", zero_division=0), 4),
            "f1_weighted":       round(f1_score       (y_test, y_pred, average="weighted", zero_division=0), 4),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=id_names,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
            "label_names": id_names,
        }
        return metrics

    def save(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(_MODELS, "baseline_tfidf_lr.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)
        cfg = {
            "model_type": "TF-IDF+LogisticRegression",
            "label_map":  LABEL_MAP,
            "id_map":     ID_MAP,
            "path":       path,
        }
        with open(os.path.join(_MODELS, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  [SAVED] Baseline model -> {path}")

    def load(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(_MODELS, "baseline_tfidf_lr.pkl")
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)
        self._is_trained = True
        return self


# ===========================================================================
# TRANSFORMER — DistilBERT Fine-tuned
# ===========================================================================

class UrgencyTransformerClassifier:
    """
    DistilBERT-based urgency classifier (fine-tuned on urgency_dataset.csv).

    NOTE: Only trained and reported if torch + transformers are available
    and training completes without error. Results are NOT fabricated.
    If training fails, the fallback is the baseline classifier.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name  = model_name
        self.model       = None
        self.tokenizer   = None
        self.label_map   = LABEL_MAP
        self.id_map      = ID_MAP
        self._is_trained = False
        self._available  = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            import torch
            from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
            return True
        except ImportError:
            return False

    def train(
        self,
        X_train: list,
        y_train: list,
        X_val: list,
        y_val: list,
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        max_length: int = 64,
    ) -> dict:
        """
        Fine-tune DistilBERT on the urgency dataset.

        Returns a dict with training info and val metrics.
        If training fails for any reason, returns a failure dict.
        """
        if not self._available:
            return {"success": False, "reason": "torch/transformers not available"}

        try:
            import torch
            from torch.utils.data import Dataset, DataLoader
            try:
                from torch.optim import AdamW
            except ImportError:
                from transformers import AdamW
            from transformers import (
                DistilBertTokenizerFast,
                DistilBertForSequenceClassification,
                get_linear_schedule_with_warmup,
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"  [Transformer] Device: {device}")
            print(f"  [Transformer] Loading tokenizer: {self.model_name}")

            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)

            # --- Dataset ---
            class UrgencyDataset(Dataset):
                def __init__(self, texts, labels, tokenizer, max_len):
                    self.encodings = tokenizer(
                        texts, truncation=True, padding=True,
                        max_length=max_len, return_tensors="pt"
                    )
                    self.labels = torch.tensor(labels, dtype=torch.long)

                def __len__(self):
                    return len(self.labels)

                def __getitem__(self, idx):
                    item = {k: v[idx] for k, v in self.encodings.items()}
                    item["labels"] = self.labels[idx]
                    return item

            num_labels = len(LABEL_MAP)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=num_labels
            ).to(device)

            # Class weights for imbalance
            unique_labels = sorted(set(y_train))
            class_w = compute_class_weight(
                "balanced", classes=np.array(unique_labels), y=np.array(y_train)
            )
            class_weights_tensor = torch.tensor(class_w, dtype=torch.float).to(device)

            train_dataset = UrgencyDataset(
                [_clean(t) for t in X_train], y_train, self.tokenizer, max_length
            )
            val_dataset = UrgencyDataset(
                [_clean(t) for t in X_val], y_val, self.tokenizer, max_length
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

            optimizer = AdamW(self.model.parameters(), lr=lr)
            total_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps,
            )

            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

            history = []

            for epoch in range(epochs):
                # Train
                self.model.train()
                total_loss = 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    input_ids      = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels_b       = batch["labels"].to(device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss    = loss_fn(outputs.logits, labels_b)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    total_loss += loss.item()

                avg_train_loss = total_loss / len(train_loader)

                # Validate
                self.model.eval()
                all_preds, all_true = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids      = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels_b       = batch["labels"].to(device)
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        preds   = torch.argmax(outputs.logits, dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_true.extend(labels_b.cpu().numpy())

                val_f1  = f1_score(all_true, all_preds, average="macro", zero_division=0)
                val_acc = accuracy_score(all_true, all_preds)
                epoch_info = {
                    "epoch":          epoch + 1,
                    "train_loss":     round(avg_train_loss, 4),
                    "val_accuracy":   round(val_acc, 4),
                    "val_f1_macro":   round(val_f1, 4),
                }
                history.append(epoch_info)
                print(f"  Epoch {epoch+1}/{epochs}: loss={avg_train_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

            self._is_trained = True
            self._device = device

            # Final val metrics
            final_metrics = {
                "model":      "DistilBERT (fine-tuned)",
                "epochs":     epochs,
                "history":    history,
                "val_accuracy": history[-1]["val_accuracy"],
                "val_f1_macro": history[-1]["val_f1_macro"],
            }
            return {"success": True, "metrics": final_metrics}

        except Exception as e:
            return {"success": False, "reason": str(e)}

    def predict_single(self, text: str) -> dict:
        if not self._is_trained:
            return {"error": "model_not_trained"}

        import torch
        self.model.eval()
        encoding = self.tokenizer(
            _clean(text), truncation=True, padding=True,
            max_length=64, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            logits = self.model(**encoding).logits
            proba  = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred   = int(np.argmax(proba))

        scores = {ID_MAP[i]: round(float(p), 4) for i, p in enumerate(proba)}
        return {
            "label":      ID_MAP[pred],
            "class_id":   pred,
            "confidence": round(float(max(proba)), 4),
            "all_scores": scores,
            "model":      "distilbert_finetuned",
        }

    def save(self, path: Optional[str] = None):
        if not self._is_trained:
            return
        save_dir = path or os.path.join(_MODELS, "distilbert")
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"  [SAVED] DistilBERT model -> {save_dir}")


# ===========================================================================
# Training entry point
# ===========================================================================

def train_and_evaluate(data_path: Optional[str] = None) -> dict:
    """
    Load data, train baseline (and transformer if available),
    evaluate, save models and artefacts.

    Returns dict with all metrics.
    """
    if data_path is None:
        data_path = os.path.join(_DATA, "urgency_dataset.csv")

    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} rows from {os.path.basename(data_path)}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Class distribution:\n{df['urgency'].value_counts()}")

    # Encode labels on full dataset
    df["label_id"] = df["urgency"].map(LABEL_MAP)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    X = df["text"].tolist()
    y = df["label_id"].tolist()

    # Stratified split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    all_results = {}

    # -------------------------------------------------------------------
    # BASELINE
    # -------------------------------------------------------------------
    print("\n[BASELINE] Training TF-IDF + Logistic Regression ...")
    baseline = UrgencyBaselineClassifier()
    baseline.build()
    baseline.train(X_train, y_train)
    baseline_metrics = baseline.evaluate(X_test, y_test)
    baseline.save()

    all_results["baseline"] = baseline_metrics
    print(f"  Accuracy:  {baseline_metrics['accuracy']}")
    print(f"  F1 (macro):{baseline_metrics['f1_macro']}")

    # Save metrics
    _save_metrics(baseline_metrics, all_results)

    # -------------------------------------------------------------------
    # TRANSFORMER (DistilBERT)
    # -------------------------------------------------------------------
    print("\n[TRANSFORMER] Attempting DistilBERT fine-tuning ...")
    transformer = UrgencyTransformerClassifier()
    X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    try:
        import torch
        if not torch.cuda.is_available() and len(X_tr2) > 800:
            X_tr2, _, y_tr2, _ = train_test_split(
                X_tr2, y_tr2, train_size=800, random_state=42, stratify=y_tr2
            )
            X_val2 = X_val2[:200]
            y_val2 = y_val2[:200]
    except Exception:
        pass
    tf_result = transformer.train(
        X_tr2, y_tr2, X_val2, y_val2,
        epochs=2, batch_size=16, max_length=64,
    )

    if tf_result["success"]:
        print("  [TRANSFORMER] Training succeeded.")
        transformer.save()
        # Full test evaluation
        tf_test_metrics = _evaluate_transformer(transformer, X_test, y_test)
        all_results["transformer"] = {
            "training_info":  tf_result["metrics"],
            "test_metrics":   tf_test_metrics,
        }
        _save_metrics(tf_test_metrics, all_results, prefix="transformer_")
    else:
        print(f"  [TRANSFORMER] Training failed: {tf_result['reason']}")
        print("  [TRANSFORMER] Only baseline metrics are reported.")
        all_results["transformer"] = {
            "success": False,
            "reason":  tf_result["reason"],
            "note":    "Transformer results NOT fabricated. Baseline is the reported model.",
        }

    return all_results, baseline


def _evaluate_transformer(transformer, X_test, y_test) -> dict:
    """Evaluate transformer on test set."""
    preds = []
    for text in X_test:
        r = transformer.predict_single(text)
        preds.append(r["class_id"])

    id_names = [ID_MAP[l] for l in sorted(set(y_test) | set(preds))]
    labels   = sorted(set(y_test) | set(preds))

    return {
        "model":             "DistilBERT (fine-tuned) — TEST SET",
        "accuracy":          round(accuracy_score(y_test, preds), 4),
        "f1_macro":          round(f1_score(y_test, preds, average="macro",     zero_division=0), 4),
        "f1_weighted":       round(f1_score(y_test, preds, average="weighted",  zero_division=0), 4),
        "precision_macro":   round(precision_score(y_test, preds, average="macro",    zero_division=0), 4),
        "recall_macro":      round(recall_score   (y_test, preds, average="macro",    zero_division=0), 4),
        "classification_report": classification_report(
            y_test, preds, target_names=id_names, output_dict=True, zero_division=0
        ),
        "confusion_matrix":  confusion_matrix(y_test, preds, labels=labels).tolist(),
        "label_names":       id_names,
    }


def _save_metrics(metrics: dict, all_results: dict, prefix: str = ""):
    """Save metrics to JSON, CSV, and PNG confusion matrix."""

    # JSON
    json_path = os.path.join(_OUTPUTS, f"{prefix}urgency_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [SAVED] {json_path}")

    # CSV classification report
    cr = metrics.get("classification_report", {})
    if cr:
        rows = []
        for label, vals in cr.items():
            if isinstance(vals, dict):
                rows.append({"class": label, **vals})
        if rows:
            cr_path = os.path.join(_OUTPUTS, f"{prefix}urgency_classification_report.csv")
            pd.DataFrame(rows).to_csv(cr_path, index=False)
            print(f"  [SAVED] {cr_path}")

    # Confusion matrix PNG
    cm = metrics.get("confusion_matrix")
    label_names = metrics.get("label_names", [])
    if cm:
        _plot_confusion_matrix(
            np.array(cm), label_names,
            title=f"Urgency Classifier — Confusion Matrix\n({metrics.get('model', '')})",
            save_path=os.path.join(_OUTPUTS, f"{prefix}urgency_confusion_matrix.png"),
        )


def _plot_confusion_matrix(cm: np.ndarray, labels: list, title: str, save_path: str):
    plt.rcParams.update({
        "figure.facecolor": "#0F0F1A",
        "axes.facecolor":   "#1A1A2E",
        "axes.labelcolor":  "#E0E0E0",
        "xtick.color":      "#B0B0B0",
        "ytick.color":      "#B0B0B0",
        "text.color":       "#E0E0E0",
    })
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax, annot_kws={"size": 12},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=12, fontweight="bold", color="#E0E0E0")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


# ===========================================================================
# Standalone execution
# ===========================================================================

if __name__ == "__main__":
    results, baseline_model = train_and_evaluate()

    print("\n=== Baseline Metrics Summary ===")
    bm = results["baseline"]
    for k, v in bm.items():
        if k not in ("classification_report", "confusion_matrix", "label_names"):
            print(f"  {k}: {v}")

    # Edge-case predictions
    edge_cases = [
        "Mild fatigue, slight nausea.",
        "Patient denies any symptoms.",
        "No fever.",
        "Severe difficulty breathing and chest tightness.",
        "Persistent vomiting several times today.",
        "",
        "EGFR L858R mutation detected.",
        "xyz unknown text zzz",
    ]
    print("\n=== Edge-Case Predictions (Baseline) ===")
    for text in edge_cases:
        result = baseline_model.predict_single(text)
        print(f"  [{result['label']:8s}] conf={result['confidence']:.3f} | '{text[:55]}'")
