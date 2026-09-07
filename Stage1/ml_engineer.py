# ============================================================
# ML ENGINEER - ONCOLOGY TOXICITY & RISK PREDICTION PIPELINE
# ============================================================

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import glob
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. DATASET LOCATOR & AUTOMATIC SELECTION
# ============================================================

def locate_data_file():
    """Locate the best available dataset for oncology risk prediction."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    cwd = os.getcwd()

    search_dirs = [cwd, script_dir, parent_dir]

    candidates = []
    for d in search_dirs:
        for pattern in [
            "*oncology_risk_dataset_2000*.csv",
            "*Cleaned_oncology_dataset_10k*.csv",
            "*oncology_dataset_10k_CLEANED*.csv",
            "eda_processed_data.csv*"
        ]:
            matches = glob.glob(os.path.join(d, pattern))
            candidates.extend(matches)

    r2k = [f for f in candidates if "2000" in os.path.basename(f)]
    if r2k:
        return r2k[0]

    r10k = [f for f in candidates if "10k" in os.path.basename(f)]
    if r10k:
        return r10k[0]

    if candidates:
        return candidates[0]

    raise FileNotFoundError("No valid dataset file found in workspace directories.")


FILE_PATH = locate_data_file()
print(f"Loading dataset from: {FILE_PATH}")

df = pd.read_csv(FILE_PATH)
print("Dataset shape:", df.shape)


# ============================================================
# 2. TARGET & CLINICAL FEATURE ENGINEERING
# ============================================================

df_processed = df.copy()

if "toxicity_risk" in df_processed.columns:
    TARGET = "toxicity_risk"
    IS_MULTICLASS = True
    print("\nDataset Type: Oncology Clinical Risk & Toxicity (Multiclass Target)")

    label_encoder = LabelEncoder()
    df_processed[TARGET] = label_encoder.fit_transform(df_processed[TARGET])
    target_classes = label_encoder.classes_
    print("Target classes:", list(target_classes))

    # Feature Engineering for Clinical Risk
    if "dosage_mg" in df_processed and "previous_dose_mg" in df_processed:
        df_processed["dose_ratio"] = df_processed["dosage_mg"] / (df_processed["previous_dose_mg"] + 1.0)

    if "systolic_bp" in df_processed and "diastolic_bp" in df_processed:
        df_processed["pulse_pressure"] = df_processed["systolic_bp"] - df_processed["diastolic_bp"]

    if "ALT" in df_processed and "AST" in df_processed:
        df_processed["liver_enzyme_sum"] = df_processed["ALT"] + df_processed["AST"]

    DROP_COLS = ["patient_id", "timestamp", "adverse_event_count", TARGET]
    if "toxicity_score" in df_processed.columns:
        DROP_COLS.append("toxicity_score")
else:
    TARGET = "grade3_4_ae_flag"
    IS_MULTICLASS = False
    target_classes = [0, 1]
    print("\nDataset Type: 10k Synthetic Oncology Dataset (Binary Target)")

    if "ctdna_3mo_vaf_pct" in df_processed and "ctdna_baseline_vaf_pct" in df_processed:
        df_processed["ctdna_change_3m"] = df_processed["ctdna_3mo_vaf_pct"] - df_processed["ctdna_baseline_vaf_pct"]

    if "ct_3mo_tumor_mm" in df_processed and "baseline_tumor_diameter_mm" in df_processed:
        df_processed["tumor_change_3m"] = df_processed["ct_3mo_tumor_mm"] - df_processed["baseline_tumor_diameter_mm"]

    DROP_COLS = ["patient_id", TARGET]


# Filter features
feature_cols = [c for c in df_processed.columns if c not in DROP_COLS]

X = df_processed[feature_cols].copy()
y = df_processed[TARGET].copy()

print("\nTarget distribution:")
print(y.value_counts())
print("\nNumber of features utilized:", len(feature_cols))


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 4. PREPROCESSING PIPELINE
# ============================================================

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ============================================================
# 5. HIGH-PERFORMANCE MODEL INITIALIZATION & TRAINING
# ============================================================

xgb_obj = "multi:softprob" if IS_MULTICLASS else "binary:logistic"
xgb_eval = "mlogloss" if IS_MULTICLASS else "logloss"

base_models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=1),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, objective=xgb_obj, eval_metric=xgb_eval, random_state=42, n_jobs=1)
}

trained_pipelines = {}

print("\n==============================")
print("TRAINING BASE MODELS")
print("==============================")

for name, clf in base_models.items():
    print(f"Training {name}...")
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", clf)
    ])
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe

# Build Voting Ensemble
print("\n==============================")
print("TRAINING ENSEMBLE CLASSIFIER")
print("==============================")

voting_clf = VotingClassifier(
    estimators=[
        ("rf", trained_pipelines["Random Forest"]),
        ("et", trained_pipelines["Extra Trees"]),
        ("xgb", trained_pipelines["XGBoost"])
    ],
    voting="soft",
    n_jobs=1
)

voting_clf.fit(X_train, y_train)
trained_pipelines["Voting Ensemble"] = voting_clf
print("Voting Ensemble trained successfully.")


# ============================================================
# 6. MODEL METRICS CALCULATION
# ============================================================

def calculate_metrics(name, model):
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, prediction)

    if IS_MULTICLASS:
        precision = precision_score(y_test, prediction, average="weighted", zero_division=0)
        recall = recall_score(y_test, prediction, average="weighted", zero_division=0)
        f1 = f1_score(y_test, prediction, average="weighted", zero_division=0)
        try:
            auc = roc_auc_score(y_test, probability, multi_class="ovr")
        except Exception:
            auc = np.nan
    else:
        precision = precision_score(y_test, prediction, zero_division=0)
        recall = recall_score(y_test, prediction, zero_division=0)
        f1 = f1_score(y_test, prediction, zero_division=0)
        auc = roc_auc_score(y_test, probability[:, 1])

    print("\n================================")
    print(name)
    print("================================")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4) if not np.isnan(auc) else "N/A")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, prediction))

    print("\nClassification Report:")
    target_names = [str(c) for c in target_classes] if IS_MULTICLASS else None
    print(classification_report(y_test, prediction, target_names=target_names, zero_division=0))

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": auc
    }


# ============================================================
# 7. COMPARE MODELS & SELECT BEST
# ============================================================

results_list = []
for name, pipe in trained_pipelines.items():
    res = calculate_metrics(name, pipe)
    results_list.append(res)

results = pd.DataFrame(results_list)
results = results.sort_values(by="Accuracy", ascending=False)

print("\n\n==============================================")
print("FINAL MODEL COMPARISON (SORTED BY ACCURACY)")
print("==============================================")
print(results.to_string(index=False))

best_model_name = results.iloc[0]["Model"]
best_accuracy = results.iloc[0]["Accuracy"]
final_model = trained_pipelines[best_model_name]

print(f"\nFinal selected top model: {best_model_name} (Accuracy: {best_accuracy * 100:.2f}%)")


# ============================================================
# 8. SAVE MODEL ARTIFACTS & REPORTS
# ============================================================

output_dir = os.path.dirname(os.path.abspath(__file__))

best_model_path = os.path.join(output_dir, "best_toxicity_model.pkl")
joblib.dump(final_model, best_model_path)

joblib.dump(trained_pipelines["Decision Tree"], os.path.join(output_dir, "tuned_decision_tree.pkl"))
joblib.dump(trained_pipelines["Random Forest"], os.path.join(output_dir, "tuned_random_forest.pkl"))
joblib.dump(trained_pipelines["Extra Trees"], os.path.join(output_dir, "tuned_extra_trees.pkl"))
joblib.dump(trained_pipelines["XGBoost"], os.path.join(output_dir, "tuned_xgboost.pkl"))
joblib.dump(trained_pipelines["Voting Ensemble"], os.path.join(output_dir, "tuned_stacking_model.pkl"))

results_path = os.path.join(output_dir, "model_comparison_tuned.csv")
results.to_csv(results_path, index=False)
results.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

print("\nFiles saved successfully:")
print(f"1. {best_model_path}")
print("2. tuned_decision_tree.pkl")
print("3. tuned_random_forest.pkl")
print("4. tuned_extra_trees.pkl")
print("5. tuned_xgboost.pkl")
print("6. tuned_stacking_model.pkl")
print(f"7. {results_path}")
print("8. model_comparison.csv")

print("\n" + "=" * 70)
print("ML ENGINEER PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)