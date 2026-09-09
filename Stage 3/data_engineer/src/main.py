import os
import json
import numpy as np
import pandas as pd
from data_loader import DataLoader
from data_cleaning import DataCleaner
from data_validation import DataValidator
from data_split import DataSplitter

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    outputs_dir = os.path.join(base_dir, "outputs")

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    loader = DataLoader(raw_dir)
    cleaner = DataCleaner()
    validator = DataValidator()
    splitter = DataSplitter()

    # 1. Load Data
    raw_data = loader.load_all_datasets()

    # 2. Process Urgency Data
    print("Processing urgency data...")
    urgency_df = raw_data['urgency_symptom_logs']
    original_urgency_count = len(urgency_df)
    clean_urgency_df = cleaner.clean_urgency_data(urgency_df)
    valid_urgency_df = validator.validate_urgency_data(clean_urgency_df, original_urgency_count)
    
    # Check leakage
    u_train, u_val, u_test = splitter.split_dataframe(valid_urgency_df, patient_col="patient_id")
    leakage = validator.check_leakage(u_train, u_test, "patient_id")
    validator.quality_report["datasets"]["urgency_symptom_logs"]["leakage_findings"] = leakage
    
    # Save processed dataset
    valid_urgency_df.to_csv(os.path.join(processed_dir, "urgency_dataset.csv"), index=False)
    
    # 3. Process NER Data
    print("Processing NER data...")
    ner_data = raw_data['ner_training_data']
    valid_ner_data = validator.validate_and_convert_ner_data(ner_data)
    save_jsonl(valid_ner_data, os.path.join(processed_dir, "ner_dataset.jsonl"))

    # 4. Process Drug Data
    print("Processing drug data...")
    drug_df = raw_data['drug_adverse_events']
    clean_drug_df = cleaner.clean_drug_data(drug_df)
    valid_drug_df = validator.validate_generic_dataframe(clean_drug_df, len(drug_df), "drug_adverse_events")
    valid_drug_df.to_csv(os.path.join(processed_dir, "drug_knowledge.csv"), index=False)

    # 5. Process Gene Data
    print("Processing gene data...")
    gene_df = raw_data['gene_mutation_dictionary']
    clean_gene_df = cleaner.clean_gene_data(gene_df)
    valid_gene_df = validator.validate_generic_dataframe(clean_gene_df, len(gene_df), "gene_mutation_dictionary")
    valid_gene_df.to_csv(os.path.join(processed_dir, "gene_mutation_dictionary.csv"), index=False)

    # 6. Process Guideline Data
    print("Processing guideline data...")
    guide_df = raw_data['oncology_guideline_chunks']
    clean_guide_df = cleaner.clean_guideline_data(guide_df)
    valid_guide_df = validator.validate_generic_dataframe(clean_guide_df, len(guide_df), "oncology_guideline_chunks")
    valid_guide_df.to_csv(os.path.join(processed_dir, "guideline_chunks.csv"), index=False)

    # Save Quality Report
    print("Saving quality reports...")
    report = validator.quality_report
    with open(os.path.join(outputs_dir, "data_quality_report.json"), "w") as f:
        json.dump(report, f, indent=4, cls=NpEncoder)
        
    # Convert to CSV and save
    report_records = []
    for dataset_name, metrics in report["datasets"].items():
        metrics["dataset_name"] = dataset_name
        report_records.append(metrics)
        
    report_df = pd.DataFrame(report_records)
    # Move dataset_name to front
    cols = ['dataset_name'] + [c for c in report_df.columns if c != 'dataset_name']
    report_df = report_df[cols]
    report_df.to_csv(os.path.join(outputs_dir, "data_quality_report.csv"), index=False)
    
    print("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
