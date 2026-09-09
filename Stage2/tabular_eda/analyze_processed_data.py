import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set plot design parameters
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.dpi'] = 150

file_path = "c:/Users/aruni/Desktop/New folder/eda_processed_data.csv"
df = pd.read_csv(file_path)

results = {}

# 1. Dataset Dimensions & Basic Metadata
results['n_patients'] = len(df)
results['n_features'] = df.shape[1]
results['cancer_type_counts'] = df['cancer_type'].value_counts().to_dict()
results['sex_distribution'] = df['sex'].value_counts(normalize=True).mul(100).round(1).to_dict()
results['median_age'] = float(df['age_at_diagnosis'].median())
results['age_iqr'] = [float(df['age_at_diagnosis'].quantile(0.25)), float(df['age_at_diagnosis'].quantile(0.75))]

# 2. Key Biomarkers Breakdown
results['pd_l1_median'] = float(df['pd_l1_tps_pct'].median())
results['tmb_median'] = float(df['tmb_mut_per_mb'].median())
results['tmb_class_counts'] = df['tumor_mutational_burden_class'].value_counts().to_dict()
results['msi_status_counts'] = df['msi_status'].value_counts().to_dict()
results['top_driver_mutations'] = df['primary_driver_mutation'].value_counts().head(10).to_dict()

# 3. Clinical Outcomes & Survival
results['median_os_months'] = float(df['overall_survival_months'].median())
results['os_event_rate'] = float(df['os_event_flag'].mean() * 100)
results['median_pfs_months'] = float(df['progression_free_survival_months'].median())
results['pfs_event_rate'] = float(df['pfs_event_flag'].mean() * 100)

# Survival by Cancer Type
os_by_cancer = df.groupby('cancer_type')['overall_survival_months'].agg(['median', 'mean']).round(1).to_dict(orient='index')
results['os_by_cancer'] = os_by_cancer

# Objective Response Rate (ORR = CR + PR)
orr_count = df['best_overall_response'].isin(['CR', 'PR']).sum()
results['orr_pct'] = round((orr_count / len(df)) * 100, 1)
results['bor_counts'] = df['best_overall_response'].value_counts().to_dict()

# Treatment modalities breakdown
results['treatment_flags'] = {
    'Immunotherapy': int(df['immunotherapy_flag'].sum()),
    'Targeted Therapy': int(df['targeted_therapy_flag'].sum()),
    'Chemotherapy': int(df['chemo_flag'].sum()),
    'Surgery': int(df['surgery_flag'].sum()),
    'Radiation': int(df['radiation_flag'].sum())
}

# Adverse Events & Safety
results['grade3_4_ae_rate'] = round(float(df['grade3_4_ae_flag'].mean() * 100), 1)
results['dose_reduction_rate'] = round(float(df['dose_reduction_flag'].mean() * 100), 1)
results['discontinuation_rate'] = round(float(df['tx_discontinuation_flag'].mean() * 100), 1)

# Save json results
with open('analysis_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print("JSON summary created successfully.")

# Create Visual Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cancer Type Distribution
sns.countplot(y='cancer_type', data=df, order=df['cancer_type'].value_counts().index, ax=axes[0,0], palette='Blues_r')
axes[0,0].set_title('Patient Count by Cancer Type', fontweight='bold', fontsize=12)
axes[0,0].set_xlabel('Patient Count')
axes[0,0].set_ylabel('Cancer Type')

# Plot 2: OS by Cancer Type (Boxplot)
sns.boxplot(x='overall_survival_months', y='cancer_type', data=df, ax=axes[0,1], palette='crest')
axes[0,1].set_title('Overall Survival (Months) by Cancer Type', fontweight='bold', fontsize=12)
axes[0,1].set_xlabel('OS (Months)')
axes[0,1].set_ylabel('')

# Plot 3: Best Overall Response (BOR) Distribution
sns.countplot(x='best_overall_response', data=df, order=['CR', 'PR', 'SD', 'PD', 'NE'], ax=axes[1,0], palette='viridis')
axes[1,0].set_title('Best Overall Response (RECIST 1.1)', fontweight='bold', fontsize=12)
axes[1,0].set_xlabel('Response Category')
axes[1,0].set_ylabel('Count')

# Plot 4: PD-L1 TPS vs TMB (Scatter / Hexbin)
sns.scatterplot(x='pd_l1_tps_pct', y='tmb_mut_per_mb', hue='best_overall_response', data=df, ax=axes[1,1], alpha=0.7)
axes[1,1].set_title('Biomarker Co-expression: PD-L1 TPS vs TMB', fontweight='bold', fontsize=12)
axes[1,1].set_xlabel('PD-L1 TPS (%)')
axes[1,1].set_ylabel('TMB (mut/Mb)')

plt.tight_layout()
plt.savefig('cohort_analysis_overview.png')
plt.close()
print("Saved visualization overview plot: cohort_analysis_overview.png")
