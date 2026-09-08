import pandas as pd
import numpy as np

file_path = "c:/Users/aruni/Desktop/New folder/eda_processed_data.csv"
df = pd.read_csv(file_path)

issues = []
warnings = []
passed_checks = []

# 1. Duplicate checks
dup_rows = df.duplicated().sum()
dup_patients = df['patient_id'].duplicated().sum()
if dup_rows > 0:
    issues.append(f"Found {dup_rows} duplicate rows in the dataset.")
else:
    passed_checks.append("No duplicate rows found.")

if dup_patients > 0:
    issues.append(f"Found {dup_patients} duplicate patient IDs.")
else:
    passed_checks.append("All patient_ids are unique.")

# 2. Missing Value Analysis
null_summary = df.isnull().sum()
cols_with_nulls = null_summary[null_summary > 0]
null_details = {}
for col, count in cols_with_nulls.items():
    pct = round((count / len(df)) * 100, 2)
    null_details[col] = f"{count} nulls ({pct}%)"

# Categorize nulls as expected (e.g. 2nd line therapy for patients who didn't progress) vs unexpected
expected_null_cols = ['primary_driver_mutation', 'mutation_1', 'mutation_2', 'mutation_3', 
                      'treatment_line_2', 'treatment_line_3', 'time_to_response_months', 
                      'duration_of_response_months', 'metastatic_sites']

unexpected_nulls = {col: detail for col, detail in null_details.items() if col not in expected_null_cols}
if unexpected_nulls:
    issues.append(f"Unexpected missing values in columns: {unexpected_nulls}")
else:
    passed_checks.append("All missing values are in expected optional/conditional fields (e.g., driver mutations, 2nd/3rd line tx, response duration).")

# 3. Numeric Range & Boundary Checks
# Age
if (df['age_at_diagnosis'] < 0).any() or (df['age_at_diagnosis'] > 120).any():
    issues.append("Invalid age values found (out of range 0-120).")
else:
    passed_checks.append(f"Age range valid: [{df['age_at_diagnosis'].min()}, {df['age_at_diagnosis'].max()}] years.")

# BMI
if (df['bmi'] < 10).any() or (df['bmi'] > 100).any():
    issues.append("Invalid BMI values found (out of range 10-100).")
else:
    passed_checks.append(f"BMI range valid: [{df['bmi'].min():.1f}, {df['bmi'].max():.1f}].")

# Pack years
if (df['pack_years'] < 0).any():
    issues.append("Negative pack_years values found.")
else:
    passed_checks.append(f"Pack years range valid: [{df['pack_years'].min()}, {df['pack_years'].max()}].")

# PD-L1 TPS %
if (df['pd_l1_tps_pct'] < 0).any() or (df['pd_l1_tps_pct'] > 100).any():
    issues.append("PD-L1 TPS % out of range [0, 100].")
else:
    passed_checks.append(f"PD-L1 TPS % range valid: [{df['pd_l1_tps_pct'].min():.1f}%, {df['pd_l1_tps_pct'].max():.1f}%].")

# Survival Times
if (df['overall_survival_months'] < 0).any():
    issues.append("Negative overall_survival_months found.")
else:
    passed_checks.append(f"Overall survival range valid: [{df['overall_survival_months'].min():.1f}, {df['overall_survival_months'].max():.1f}] months.")

if (df['progression_free_survival_months'] < 0).any():
    issues.append("Negative progression_free_survival_months found.")
else:
    passed_checks.append(f"PFS range valid: [{df['progression_free_survival_months'].min():.1f}, {df['progression_free_survival_months'].max():.1f}] months.")

# Logical consistency: PFS <= OS
pfs_gt_os = (df['progression_free_survival_months'] > df['overall_survival_months']).sum()
if pfs_gt_os > 0:
    issues.append(f"Logical error: {pfs_gt_os} patients have PFS > OS (progression after death date).")
else:
    passed_checks.append("Logical consistency passed: PFS <= OS for all patients.")

# Tumor diameter measurements
tumor_cols = ['baseline_tumor_diameter_mm', 'ct_3mo_tumor_mm', 'ct_6mo_tumor_mm', 'ct_12mo_tumor_mm', 'ct_24mo_tumor_mm']
neg_tumors = 0
for col in tumor_cols:
    if (df[col] < 0).any():
        neg_tumors += (df[col] < 0).sum()
if neg_tumors > 0:
    issues.append(f"Negative tumor diameter measurements found ({neg_tumors} occurrences).")
else:
    passed_checks.append("All tumor diameter measurements are non-negative.")

# 4. Binary Flags Validation
flag_cols = [col for col in df.columns if col.endswith('_flag')]
invalid_flags = {}
for col in flag_cols:
    unique_vals = set(df[col].dropna().unique())
    if not unique_vals.issubset({0, 1}):
        invalid_flags[col] = list(unique_vals)
if invalid_flags:
    issues.append(f"Binary flag columns contain non-binary values: {invalid_flags}")
else:
    passed_checks.append(f"All {len(flag_cols)} binary flag columns contain strictly 0 or 1 values.")

# 5. Categorical Encoding & Typo Check
cat_checks = {
    'sex': ['F', 'M'],
    'cancer_type': ['NSCLC', 'CRC', 'BREAST', 'PANCREATIC', 'HCC', 'PROSTATE', 'OVARIAN', 'MELANOMA', 'BLADDER', 'GBM'],
    'msi_status': ['MSS', 'MSI-H'],
    'best_overall_response': ['CR', 'PR', 'SD', 'PD', 'NE'],
    'tumor_mutational_burden_class': ['Low', 'Intermediate', 'High'],
    'hrd_status': ['HRD_pos', 'HRD_neg', 'Unknown', 'Negative', 'Positive']
}
typos = {}
for col, valid_set in cat_checks.items():
    actual_set = set(df[col].dropna().unique())
    diff = actual_set - set(valid_set)
    if diff:
        typos[col] = list(diff)
if typos:
    warnings.append(f"Unexpected categorical values found: {typos}")
else:
    passed_checks.append("All categorical columns contain clean, standardized categories with zero typos.")

print("==================================================")
print(" DATA QUALITY AUDIT REPORT")
print("==================================================")
print(f"Total Rows Checked   : {len(df)}")
print(f"Total Columns Checked: {len(df.columns)}")
print(f"Passed Checks        : {len(passed_checks)}")
print(f"Warnings Found       : {len(warnings)}")
print(f"Critical Errors Found: {len(issues)}\n")

print("--- PASSED INTEGRITY CHECKS ---")
for check in passed_checks:
    print(f"[PASS] {check}")

if warnings:
    print("\n--- WARNINGS ---")
    for w in warnings:
        print(f"[WARN] {w}")

if issues:
    print("\n--- CRITICAL ERRORS ---")
    for err in issues:
        print(f"[FAIL] {err}")
else:
    print("\n>>> DATASET STATUS: CLEAN & ERROR-FREE <<<")
