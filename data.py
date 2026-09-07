import pandas as pd

# ============================================================
# 1. LOAD DATASET
# ============================================================

INPUT_FILE = "oncology_dataset_10k.csv"
OUTPUT_FILE = "master_dataset.csv"

df = pd.read_csv(INPUT_FILE)

print("Raw Dataset Shape:", df.shape)


# ============================================================
# 2. REMOVE DUPLICATE ROWS
# ============================================================

before = len(df)

df = df.drop_duplicates()

print("Duplicate rows removed:", before - len(df))


# ============================================================
# 3. REMOVE DUPLICATE PATIENT RECORDS
# ============================================================

if "patient_id" in df.columns:
    df = df.drop_duplicates(
        subset="patient_id",
        keep="first"
    )


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

numeric_cols = df.select_dtypes(
    include="number"
).columns

categorical_cols = df.select_dtypes(
    include="object"
).columns


# Numerical → Median
for col in numeric_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(
            df[col].median()
        )


# Categorical → Mode
for col in categorical_cols:
    if df[col].isnull().any():

        mode = df[col].mode()

        if not mode.empty:
            df[col] = df[col].fillna(
                mode.iloc[0]
            )
        else:
            df[col] = df[col].fillna("UNKNOWN")


# ============================================================
# 5. STANDARDIZE TEXT DATA
# ============================================================

for col in categorical_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.upper()
    )


# ============================================================
# 6. CLEAN PATIENT ID
# ============================================================

if "patient_id" in df.columns:
    df["patient_id"] = (
        df["patient_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )


# ============================================================
# 7. BASIC VALIDATION
# ============================================================

if "age_at_diagnosis" in df.columns:

    invalid_age = (
        (df["age_at_diagnosis"] < 0) |
        (df["age_at_diagnosis"] > 120)
    )

    df.loc[
        invalid_age,
        "age_at_diagnosis"
    ] = df["age_at_diagnosis"].median()


if "bmi" in df.columns:

    invalid_bmi = (
        (df["bmi"] <= 0) |
        (df["bmi"] > 100)
    )

    df.loc[
        invalid_bmi,
        "bmi"
    ] = df["bmi"].median()


# ============================================================
# 8. FINAL CHECK
# ============================================================

print("Remaining missing values:",
      df.isnull().sum().sum())

print("Final Dataset Shape:",
      df.shape)


# ============================================================
# 9. SAVE MASTER DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 10. SUMMARY
# ============================================================

print("\n======================================")
print("DATA ENGINEERING COMPLETED")
print("======================================")

print("Rows       :", len(df))
print("Columns    :", len(df.columns))

if "patient_id" in df.columns:
    print(
        "Patients   :",
        df["patient_id"].nunique()
    )

print("Output     :", OUTPUT_FILE)

print("\nFirst 5 rows:")
print(df.head())