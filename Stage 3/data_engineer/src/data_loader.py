import pandas as pd
import json
import os

class DataLoader:
    def __init__(self, raw_dir):
        self.raw_dir = raw_dir

    def load_csv(self, filename):
        file_path = os.path.join(self.raw_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_csv(file_path)

    def load_jsonl(self, filename):
        file_path = os.path.join(self.raw_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def load_all_datasets(self):
        print("Loading datasets...")
        datasets = {}
        datasets['pathology_clinical_notes'] = self.load_csv("pathology_clinical_notes.csv")
        datasets['urgency_symptom_logs'] = self.load_csv("urgency_symptom_logs.csv")
        datasets['drug_adverse_events'] = self.load_csv("drug_adverse_events.csv")
        datasets['oncology_guideline_chunks'] = self.load_csv("oncology_guideline_chunks.csv")
        datasets['gene_mutation_dictionary'] = self.load_csv("gene_mutation_dictionary.csv")
        datasets['ner_training_data'] = self.load_jsonl("ner_training_data.jsonl")
        print("All datasets loaded successfully.")
        return datasets
