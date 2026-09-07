import pandas as pd
import numpy as np
import uuid

def generate_synthetic_data(num_records=1000, seed=42):
    np.random.seed(seed)
    
    data = {}
    
    # IDs
    data['patient_id'] = [str(uuid.uuid4()) for _ in range(num_records)]
    
    # Demographics
    data['age'] = np.random.randint(20, 90, size=num_records)
    data['bmi'] = np.random.normal(25, 5, size=num_records)
    data['gender'] = np.random.choice(['Male', 'Female'], size=num_records)
    data['ethnicity'] = np.random.choice(['Caucasian', 'Asian', 'African', 'Hispanic', 'Other'], size=num_records)
    data['smoking_status'] = np.random.choice(['Never', 'Former', 'Current'], size=num_records)
    data['alcohol_use'] = np.random.choice(['None', 'Moderate', 'Heavy'], size=num_records)
    data['family_history'] = np.random.choice(['Yes', 'No'], size=num_records)
    
    # Clinical Status
    data['comorbidities_count'] = np.random.poisson(1, size=num_records)
    data['ecog_performance'] = np.random.choice([0, 1, 2, 3, 4], size=num_records, p=[0.4, 0.3, 0.2, 0.08, 0.02])
    data['stage_numeric'] = np.random.choice([1, 2, 3, 4], size=num_records, p=[0.1, 0.2, 0.3, 0.4])
    data['tumor_grade'] = np.random.choice(['G1', 'G2', 'G3', 'G4', 'Unknown'], size=num_records)
    data['histology'] = np.random.choice(['Adenocarcinoma', 'Squamous', 'Large Cell', 'Small Cell', 'Other'], size=num_records)
    data['lesion_count'] = np.random.poisson(3, size=num_records)
    data['tumor_burden_mm'] = np.random.normal(50, 20, size=num_records).clip(5, 200)
    
    # Mutations & Biomarkers
    mutations = ['egfr', 'alk', 'kras', 'braf', 'ros1', 'brca1', 'brca2', 'pik3ca']
    for mut in mutations:
        data[f'{mut}_mutation'] = np.random.choice(['Positive', 'Negative', 'Unknown'], size=num_records, p=[0.15, 0.75, 0.1])
        
    data['pdl1_expression'] = np.random.normal(30, 25, size=num_records).clip(0, 100)
    data['tmb_score'] = np.random.normal(10, 5, size=num_records).clip(0, 50)
    data['msi_status'] = np.random.choice(['MSS', 'MSI-H', 'Unknown'], size=num_records)
    data['met_amplification'] = np.random.choice([0, 1], size=num_records, p=[0.9, 0.1])
    data['ret_fusion'] = np.random.choice([0, 1], size=num_records, p=[0.95, 0.05])
    data['ntrk_fusion'] = np.random.choice([0, 1], size=num_records, p=[0.98, 0.02])
    data['her2_status'] = np.random.choice(['Positive', 'Negative', 'Equivocal'], size=num_records)
    
    # Metastasis Flags
    meta_sites = ['liver', 'lung', 'brain', 'bone', 'peritoneal']
    for site in meta_sites:
        data[f'{site}_metastasis'] = np.random.choice([0, 1], size=num_records, p=[0.7, 0.3])
    data['lymph_node_involvement'] = np.random.choice(['Yes', 'No'], size=num_records)
    
    # Treatment History
    data['prior_chemo'] = np.random.choice([0, 1], size=num_records)
    data['prior_radiation'] = np.random.choice([0, 1], size=num_records)
    data['prior_surgery'] = np.random.choice([0, 1], size=num_records)
    data['prior_immunotherapy'] = np.random.choice([0, 1], size=num_records)
    data['prior_targeted_therapy'] = np.random.choice([0, 1], size=num_records)
    data['line_of_therapy'] = np.random.choice([1, 2, 3, 4], size=num_records)
    data['treatment_type'] = np.random.choice(['Chemo', 'Immuno', 'Targeted', 'Combo'], size=num_records)
    
    # Baseline Labs
    data['baseline_wbc'] = np.random.normal(7, 2, size=num_records)
    data['baseline_hgb'] = np.random.normal(12, 2, size=num_records)
    data['baseline_plt'] = np.random.normal(250, 50, size=num_records)
    data['baseline_alt'] = np.random.normal(40, 20, size=num_records)
    data['baseline_ast'] = np.random.normal(35, 15, size=num_records)
    data['baseline_alp'] = np.random.normal(100, 30, size=num_records)
    data['baseline_bilirubin'] = np.random.normal(1.0, 0.5, size=num_records)
    data['baseline_creatinine'] = np.random.normal(1.0, 0.3, size=num_records)
    data['baseline_albumin'] = np.random.normal(4.0, 0.5, size=num_records)
    data['baseline_ldh'] = np.random.normal(250, 100, size=num_records)
    data['baseline_crp'] = np.random.normal(10, 10, size=num_records).clip(0, 100)
    data['baseline_cea'] = np.random.normal(5, 5, size=num_records).clip(0, 50)
    data['baseline_ca125'] = np.random.normal(30, 20, size=num_records).clip(0, 200)
    data['baseline_cdna'] = np.random.exponential(100, size=num_records)
    data['week4_cdna'] = data['baseline_cdna'] * np.random.uniform(0.1, 1.5, size=num_records) # Legitimate before 3mo prediction
    
    # Create the Target
    # Generate some structure so it's learnable but not perfectly
    target_prob = 1 / (1 + np.exp(-(-2 + 0.5 * data['stage_numeric'] + 0.05 * data['age'] - 0.2 * data['baseline_albumin'] + (data['egfr_mutation']=='Positive')*1.5)))
    data['treatment_response_flag'] = np.random.binomial(1, target_prob)
    
    # Post-treatment / Leakage Variables (correlated with target to look realistic)
    data['overall_survival_months'] = np.where(data['treatment_response_flag'] == 1, 
                                               np.random.normal(36, 12, size=num_records), 
                                               np.random.normal(12, 6, size=num_records)).clip(1, 120)
    data['os_event_flag'] = np.random.choice([0, 1], size=num_records, p=[0.4, 0.6])
    
    data['progression_free_survival_months'] = data['overall_survival_months'] * np.random.uniform(0.3, 0.9, size=num_records)
    data['pfs_event_flag'] = np.random.choice([0, 1], size=num_records, p=[0.3, 0.7])
    
    data['time_to_response_months'] = np.where(data['treatment_response_flag'] == 1, np.random.uniform(1, 6, size=num_records), np.nan)
    data['duration_of_response_months'] = np.where(data['treatment_response_flag'] == 1, np.random.uniform(3, 24, size=num_records), np.nan)
    data['best_overall_response'] = np.where(data['treatment_response_flag'] == 1, 
                                             np.random.choice(['CR', 'PR'], size=num_records),
                                             np.random.choice(['SD', 'PD'], size=num_records))
    
    data['ct_3mo_tumor_mm'] = data['tumor_burden_mm'] * np.where(data['treatment_response_flag'] == 1, np.random.uniform(0.1, 0.8), np.random.uniform(1.0, 1.5))
    data['ct_3mo_recist'] = np.where(data['treatment_response_flag'] == 1, 'PR', 'PD')
    data['ct_6mo_tumor_mm'] = data['ct_3mo_tumor_mm'] * np.random.uniform(0.8, 1.2, size=num_records)
    data['ct_6mo_recist'] = data['ct_3mo_recist']
    data['ct_12mo_tumor_mm'] = data['ct_6mo_tumor_mm'] * np.random.uniform(0.8, 1.2, size=num_records)
    data['ct_12mo_recist'] = data['ct_6mo_recist']
    data['ct_24mo_tumor_mm'] = data['ct_12mo_tumor_mm'] * np.random.uniform(0.8, 1.2, size=num_records)
    data['ct_24mo_recist'] = data['ct_12mo_recist']
    data['adverse_event_grade'] = np.random.choice([0, 1, 2, 3, 4, 5], size=num_records)
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values randomly
    for col in df.columns:
        if col not in ['patient_id', 'treatment_response_flag'] and 'leakage' not in col:
            mask = np.random.rand(num_records) < 0.05
            df.loc[mask, col] = np.nan
            
    # Add some duplicate rows as per request
    duplicates = df.sample(n=20)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    print(f"Generated {df.shape[0]} rows and {df.shape[1]} columns.")
    df.to_csv('oncology_dataset.csv', index=False)
    print("Saved to oncology_dataset.csv")
    
if __name__ == "__main__":
    generate_synthetic_data()
