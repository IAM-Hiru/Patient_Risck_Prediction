"""
================================================================================
STAGE 02 MULTI-MODAL CANCER PROGRESSION PIPELINE - EVALUATION MODULE
File: 04_model_evaluation.py
Role: Evaluation Engineer
================================================================================
Quantitative model evaluation and metric validation module.
Loads trained model weights ('best_stage02_multimodal_lstm.pth') and test DataLoader.
Calculates:
  - Global Metrics: ROC-AUC, PR-AUC, F1-Score, Precision, Recall, Confusion Matrix
  - Cancer Subgroup Breakdown: Independent ROC-AUC & F1-Score across 8 cancer types
    ('BRCA', 'LUAD', 'LUSC', 'COAD', 'PRAD', 'STAD', 'SKCM', 'BLCA')
Saves results to 'evaluation_results.json'.
"""

import os
import json
import importlib.util
from typing import Dict, Any, List, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve
)


def load_modules(script_dir: str):
    """Dynamically imports data_loader and deep_learning modules."""
    dl_path = os.path.join(script_dir, "01_data_loader.py")
    model_path = os.path.join(script_dir, "03_deep_learning_pipeline.py")

    spec_dl = importlib.util.spec_from_file_location("dl_mod", dl_path)
    dl_mod = importlib.util.module_from_spec(spec_dl)
    spec_dl.loader.exec_module(dl_mod)

    spec_model = importlib.util.spec_from_file_location("model_mod", model_path)
    model_mod = importlib.util.module_from_spec(spec_model)
    spec_model.loader.exec_module(model_mod)

    return dl_mod, model_mod


def evaluate_model(
    model_path: str,
    test_loader: DataLoader,
    output_json_path: str = "evaluation_results.json",
    device: str = None
) -> Dict[str, Any]:
    """
    Evaluates trained MultimodalLSTM on test set, computing global and subgroup metrics.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dl_mod, model_mod = load_modules(script_dir)

    print(f"============================================================")
    print(f"STARTING STAGE 02 QUANTITATIVE EVALUATION PIPELINE")
    print(f"============================================================")
    print(f"Model Checkpoint : {model_path}")
    print(f"Device           : {device}")
    print(f"Output File      : {output_json_path}\n")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    # Load Model Architecture & Checkpoint Weights
    model = model_mod.MultimodalLSTM().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    all_y_true = []
    all_y_prob = []
    all_cancer_types = []

    with torch.no_grad():
        for batch in test_loader:
            h_seq = batch["pathology_seq"].to(device)
            c_seq = batch["ct_seq"].to(device)
            tab_seq = batch["tabular_seq"].to(device)
            labels = batch["labels_seq"] # CPU FloatTensor [B, T]
            c_types = batch["cancer_type"] # List of strings size B

            logits = model(h_seq, c_seq, tab_seq) # [B, T]
            probs = torch.sigmoid(logits).cpu()

            B, T = labels.shape
            for i in range(B):
                c_type = c_types[i]
                for t in range(T):
                    all_y_true.append(float(labels[i, t]))
                    all_y_prob.append(float(probs[i, t]))
                    all_cancer_types.append(c_type)

    y_true = np.array(all_y_true)
    y_prob = np.array(all_y_prob)
    c_types_arr = np.array(all_cancer_types)

    # --- OPTIMAL THRESHOLD via PR Curve (maximises F1) ---
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_prob)
    f1_arr = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0.0
    )
    best_thresh = float(thresholds_pr[np.argmax(f1_arr[:-1])])  # last element has no threshold
    print(f"Optimal Classification Threshold (max-F1): {best_thresh:.4f}")
    y_pred = (y_prob >= best_thresh).astype(int)

    # 1. GLOBAL METRICS
    global_roc_auc = float(roc_auc_score(y_true, y_prob))
    global_pr_auc = float(average_precision_score(y_true, y_prob))
    global_f1 = float(f1_score(y_true, y_pred, zero_division=0))
    global_precision = float(precision_score(y_true, y_pred, zero_division=0))
    global_recall = float(recall_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    global_metrics = {
        "Threshold": round(best_thresh, 4),
        "ROC_AUC": round(global_roc_auc, 4),
        "PR_AUC": round(global_pr_auc, 4),
        "F1_Score": round(global_f1, 4),
        "Precision": round(global_precision, 4),
        "Recall": round(global_recall, 4),
        "Confusion_Matrix": {
            "TN": tn, "FP": fp, "FN": fn, "TP": tp
        }
    }

    # 2. CANCER SUBGROUP METRICS (8 Types)
    target_cancers = ["BRCA", "LUAD", "LUSC", "COAD", "PRAD", "STAD", "SKCM", "BLCA"]
    subgroup_metrics = {}

    for c in target_cancers:
        idx = (c_types_arr == c)
        if np.sum(idx) > 0:
            yt = y_true[idx]
            yp = y_prob[idx]
            ypr = y_pred[idx]

            try:
                sub_auc = float(roc_auc_score(yt, yp))
            except ValueError:
                sub_auc = 0.0

            sub_f1 = float(f1_score(yt, ypr, zero_division=0))
            subgroup_metrics[c] = {
                "sample_count": int(np.sum(idx)),
                "ROC_AUC": round(sub_auc, 4),
                "F1_Score": round(sub_f1, 4)
            }

    # PRINT SUMMARY TABLE
    print("============================================================")
    print("GLOBAL MODEL PERFORMANCE METRICS")
    print("============================================================")
    print(f"Optimal Threshold : {best_thresh:.4f}")
    print(f"ROC-AUC     : {global_roc_auc:.4f}")
    print(f"PR-AUC      : {global_pr_auc:.4f}")
    print(f"F1-Score    : {global_f1:.4f}")
    print(f"Precision   : {global_precision:.4f}")
    print(f"Recall      : {global_recall:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")

    print("============================================================")
    print("SUBGROUP EVALUATION BY CANCER TYPE")
    print("============================================================")
    print(f"{'Cancer Type':<12} | {'Samples':<10} | {'ROC-AUC':<12} | {'F1-Score':<12}")
    print("-" * 52)
    for c, metrics in subgroup_metrics.items():
        print(f"{c:<12} | {metrics['sample_count']:<10} | {metrics['ROC_AUC']:<12.4f} | {metrics['F1_Score']:<12.4f}")
    print("-" * 52 + "\n")

    results_dict = {
        "global_metrics": global_metrics,
        "subgroup_metrics": subgroup_metrics
    }

    # SAVE TO JSON
    out_dir = os.path.dirname(output_json_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)

    print(f"Evaluation metrics saved to '{output_json_path}'.")
    return results_dict


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dl_mod, _ = load_modules(script_dir)

    csv_file = os.path.join(script_dir, "data", "stage02_lstm_8types_3000samples.csv")
    if not os.path.exists(csv_file):
        csv_file = os.path.join(script_dir, "stage02_lstm_8types_3000samples.csv")
    images_dir = os.path.join(script_dir, "2d_images_large")

    _, _, test_ldr, scaler = dl_mod.get_dataloaders(csv_file, images_dir, batch_size=8)
    ckpt_path = os.path.join(script_dir, "best_stage02_multimodal_lstm.pth")
    json_path = os.path.join(script_dir, "evaluation_results.json")

    if os.path.exists(ckpt_path):
        evaluate_model(ckpt_path, test_ldr, json_path)
    else:
        print(f"Model checkpoint '{ckpt_path}' not found. Run '03_deep_learning_pipeline.py' first.")
