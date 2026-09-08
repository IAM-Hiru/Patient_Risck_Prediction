"""
===================================================================
EXPLORATORY DATA ANALYSIS (EDA) - STEP-BY-STEP QUICKSTART TEMPLATE
===================================================================
This script provides the standard step-by-step EDA workflow commonly 
used in Jupyter Notebooks, Google Colab, and data science scripts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------
# 1. LOAD DATASET
# -----------------------------------------------------------------
# Specify the path to your dataset (CSV, Excel, TSV, etc.)
FILE_PATH = "eda_processed_data.csv"  # <-- Updated dataset file path

try:
    if FILE_PATH.endswith('.csv'):
        df = pd.read_csv(FILE_PATH)
    elif FILE_PATH.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(FILE_PATH)
    else:
        df = pd.read_csv(FILE_PATH)
    print(f"Successfully loaded dataset from '{FILE_PATH}'\n")
except FileNotFoundError:
    print(f"Error: Could not find file '{FILE_PATH}'. Please update FILE_PATH with your dataset file path.")
    exit(1)

# -----------------------------------------------------------------
# 2. INITIAL INSPECTION
# -----------------------------------------------------------------
print("=== 1. HEAD (FIRST 5 ROWS) ===")
print(df.head())

print("\n=== 2. DATASET SHAPE ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n=== 3. DATA TYPES & NON-NULL COUNTS ===")
print(df.info())

print("\n=== 4. MISSING VALUES COUNT ===")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])

print("\n=== 5. DUPLICATE ROWS ===")
print(f"Total Duplicates: {df.duplicated().sum()}")

# -----------------------------------------------------------------
# 3. STATISTICAL SUMMARIES
# -----------------------------------------------------------------
print("\n=== 6. NUMERICAL SUMMARY ===")
print(df.describe().T)

print("\n=== 7. CATEGORICAL SUMMARY ===")
print(df.describe(include=['object', 'category', 'str']))

# -----------------------------------------------------------------
# 4. UNIVARIATE ANALYSIS (DISTRIBUTIONS)
# -----------------------------------------------------------------
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object', 'category', 'str']).columns

# Plot numerical distributions
fig, axes = plt.subplots(len(num_cols), 2, figsize=(12, 3 * len(num_cols)))
for i, col in enumerate(num_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=axes[i, 0], color='skyblue')
    axes[i, 0].set_title(f'{col} - Histogram & KDE')
    
    sns.boxplot(x=df[col].dropna(), ax=axes[i, 1], color='lightgreen')
    axes[i, 1].set_title(f'{col} - Boxplot (Outlier Check)')

plt.tight_layout()
plt.savefig("univariate_analysis.png")
print("\nSaved univariate analysis plots to 'univariate_analysis.png'")

# -----------------------------------------------------------------
# 5. BIVARIATE & MULTIVARIATE ANALYSIS
# -----------------------------------------------------------------
# Correlation Heatmap for Numerical Columns
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Numerical Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
print("Saved correlation heatmap to 'correlation_heatmap.png'")

# Categorical vs Numerical Relationship
if len(cat_cols) > 0 and len(num_cols) > 0:
    first_cat = cat_cols[0] if cat_cols[0] != 'patient_id' else (cat_cols[1] if len(cat_cols) > 1 else cat_cols[0])
    first_num = num_cols[0]
    
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=first_cat, y=first_num, data=df, palette='Set2')
    plt.title(f"{first_num} by {first_cat}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("bivariate_analysis.png")
    print("Saved bivariate plot to 'bivariate_analysis.png'")

print("\nEDA script completed successfully!")
