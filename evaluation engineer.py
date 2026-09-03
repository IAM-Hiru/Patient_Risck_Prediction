import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance

def main():
    print("=======================================")
    print("EVALUATION ENGINEER - MODEL STRESS TEST")
    print("=======================================\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Best Model
    best_model_path = os.path.join(script_dir, "best_toxicity_model.pkl")
    print(f"Loading best model from: {best_model_path}")
    model = joblib.load(best_model_path)
    
    # 2. Load Unseen Dataset (X_test and y_test)
    X_test_path = os.path.join(script_dir, "X_test.csv")
    y_test_path = os.path.join(script_dir, "y_test.csv")
    
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        print(f"Error: Datasets not found at {X_test_path} and {y_test_path}")
        print("Please ensure ml_engineer.py has run and saved X_test.csv and y_test.csv")
        return
        
    print(f"Loading unseen dataset from:\n  {X_test_path}\n  {y_test_path}\n")
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()
    
    labels = sorted(y_test.unique())
    target_names = [str(l) for l in labels]
    
    print("--- 1. PREDICTING ON UNSEEN TEST ROWS ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    print("\n--- 2-5. CORE METRICS ---")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    try:
        if len(labels) > 2:
            roc = roc_auc_score(y_test, y_prob, multi_class="ovr")
        else:
            roc = roc_auc_score(y_test, y_prob[:, 1])
    except:
        roc = np.nan
        
    print(f"[01] Accuracy : {acc:.4f}")
    print(f"[02] Precision: {prec:.4f}")
    print(f"[03] Recall   : {rec:.4f}")
    print(f"[04] F1 Score : {f1:.4f}")
    print(f"[05] ROC-AUC  : {roc:.4f}")
    
    print("\n--- [06] CONFUSION MATRIX ---")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels), index=labels, columns=labels))
    
    print("\n--- [08 & 09] ERROR ANALYSIS & SEVERE-RISK FALSE NEGATIVES ---")
    severe_fn = 0
    high_class = max(labels) # Assume the highest numerical class is the highest risk
    low_class = min(labels)
    
    for i, (actual, pred) in enumerate(zip(y_test, y_pred)):
        if actual == high_class and pred == low_class:
            severe_fn += 1
            
    print(f"Severe-Risk False Negatives (Actual HIGH, Predicted LOW): {severe_fn}")
    if severe_fn > 0:
        print(">> WARNING: Model missed high-risk patients. Very dangerous in real-world scenarios.")
    else:
        print(">> Good! No severe-risk false negatives detected in test set.")
    
    print("\n--- [13] STRESS TESTING: FLAGGING OVERCONFIDENCE ---")
    probabilities = y_prob
    max_confidence = probabilities.max(axis=1)
    overconfident_errors = 0
    
    y_test_arr = y_test.values
    for i in range(len(y_test_arr)):
        actual = y_test_arr[i]
        pred = y_pred[i]
        conf = max_confidence[i]
        
        flag = "  <-- OVERCONFIDENT & WRONG!" if (pred != actual and conf > 0.8) else ""
        if flag:
            overconfident_errors += 1
            print(f"Row {i}: actual={actual}, predicted={pred}, confidence={conf:.0%}{flag}")
            
    print(f"Total overconfident and wrong predictions: {overconfident_errors} out of {len(y_test_arr)}")
    
    print("\n--- [11] MODEL COMPARISON ---")
    report_path = os.path.join(script_dir, "model_comparison_tuned.csv")
    if os.path.exists(report_path):
        df_report = pd.read_csv(report_path)
        print(df_report.to_string(index=False))
        
    print("\n--- [12] FEATURE IMPORTANCE ---")
    print("Calculating permutation importance (this might take a moment)...")
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    
    importances = pd.Series(result.importances_mean, index=X_test.columns)
    importances = importances.sort_values(ascending=False).head(10)
    print("Top 10 Most Important Features:")
    for feature, imp in importances.items():
        print(f"  {feature}: {imp:.4f}")
        
    print("\n--- FINAL EVALUATION REPORT ---")
    print("Detailed report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    print("\nEvaluation Stress-Test Pipeline Completed Successfully.")

if __name__ == "__main__":
    main()
