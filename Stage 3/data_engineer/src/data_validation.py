import pandas as pd
import json

class DataValidator:
    def __init__(self):
        self.quality_report = {
            "datasets": {}
        }

    def init_dataset_metrics(self, name, original_count):
        self.quality_report["datasets"][name] = {
            "original_record_count": original_count,
            "final_record_count": 0,
            "duplicates_removed": 0,
            "missing_values": 0,
            "invalid_records": 0,
            "invalid_labels": 0,
            "empty_texts": 0,
            "entity_annotation_errors": 0,
            "leakage_findings": "None"
        }

    def validate_urgency_data(self, df, original_count):
        name = "urgency_symptom_logs"
        self.init_dataset_metrics(name, original_count)
        metrics = self.quality_report["datasets"][name]
        
        # Check empty texts
        empty_texts = df['text'].isna() | (df['text'] == "")
        metrics["empty_texts"] = empty_texts.sum()
        
        # Check missing values
        metrics["missing_values"] = df.isna().sum().sum() - metrics["empty_texts"]
        
        # Check invalid labels (expecting 0, 1, 2, 3)
        if 'urgency_encoded' in df.columns:
            invalid_labels = df['urgency_encoded'].isna()
            metrics["invalid_labels"] = invalid_labels.sum()
        
        # Remove invalid records
        valid_df = df[~empty_texts & ~invalid_labels].copy()
        metrics["invalid_records"] = original_count - len(valid_df)
        
        # Check duplicates
        duplicates = valid_df.duplicated().sum()
        metrics["duplicates_removed"] = duplicates
        valid_df = valid_df.drop_duplicates()
        
        metrics["final_record_count"] = len(valid_df)
        return valid_df

    def validate_and_convert_ner_data(self, data):
        name = "ner_training_data"
        self.init_dataset_metrics(name, len(data))
        metrics = self.quality_report["datasets"][name]
        
        valid_entities = {"GENE_MUTATION", "DRUG_NAME", "DOSAGE", "ADVERSE_EVENT", "CANCER_TYPE"}
        
        processed_data = []
        seen_texts = set()
        for record in data:
            text = record.get("text", "")
            if not text.strip():
                metrics["empty_texts"] += 1
                metrics["invalid_records"] += 1
                continue
            
            if text in seen_texts:
                metrics["duplicates_removed"] += 1
                continue
            seen_texts.add(text)
                
            entities = record.get("entities", [])
            valid_record_entities = []
            has_error = False
            
            # Simple sorting by starting index of entity string in text to check overlap
            # Since positions might not be provided, we have to find them
            # For simplicity, if start/end not provided, we find the first occurrence.
            # But overlapping is tricky without predefined start/end.
            found_intervals = []
            
            for ent in entities:
                ent_text = ent.get("text", "")
                ent_label = ent.get("label", "")
                
                if ent_label not in valid_entities:
                    metrics["invalid_labels"] += 1
                    has_error = True
                    continue
                    
                start_idx = ent.get("start", -1)
                end_idx = ent.get("end", -1)
                
                if start_idx == -1 or end_idx == -1:
                    start_idx = text.find(ent_text)
                    if start_idx != -1:
                        end_idx = start_idx + len(ent_text)
                    else:
                        metrics["entity_annotation_errors"] += 1
                        has_error = True
                        continue
                
                # Check overlap
                overlap = False
                for s, e in found_intervals:
                    if max(start_idx, s) < min(end_idx, e):
                        overlap = True
                        break
                
                if overlap:
                    metrics["entity_annotation_errors"] += 1
                    has_error = True
                    continue
                    
                found_intervals.append((start_idx, end_idx))
                valid_record_entities.append({"text": ent_text, "label": ent_label, "start": start_idx, "end": end_idx})
            
            if has_error:
                metrics["invalid_records"] += 1
                continue
                
            # Convert to BIO
            tokens = text.split()
            bio_tags = ["O"] * len(tokens)
            
            # Very simplistic tokenization mapping based on character offsets
            # In a real scenario, standard tokenizers (like spacy) would be used.
            # We map tokens to offsets
            token_offsets = []
            curr_pos = 0
            for token in tokens:
                start = text.find(token, curr_pos)
                end = start + len(token)
                token_offsets.append((start, end, token))
                curr_pos = end
                
            for ent in valid_record_entities:
                ent_start = ent["start"]
                ent_end = ent["end"]
                ent_label = ent["label"]
                
                first_token = True
                for i, (t_start, t_end, t_text) in enumerate(token_offsets):
                    # If token overlaps with entity
                    if max(ent_start, t_start) < min(ent_end, t_end):
                        if first_token:
                            bio_tags[i] = f"B-{ent_label}"
                            first_token = False
                        else:
                            bio_tags[i] = f"I-{ent_label}"
            
            record["tokens"] = tokens
            record["bio_tags"] = bio_tags
            processed_data.append(record)
            
        metrics["final_record_count"] = len(processed_data)
        return processed_data

    def validate_generic_dataframe(self, df, original_count, name):
        self.init_dataset_metrics(name, original_count)
        metrics = self.quality_report["datasets"][name]
        
        metrics["missing_values"] = df.isna().sum().sum()
        
        # Check duplicates
        duplicates = df.duplicated().sum()
        metrics["duplicates_removed"] = duplicates
        valid_df = df.drop_duplicates().dropna()
        
        metrics["invalid_records"] = original_count - len(valid_df) - duplicates
        metrics["final_record_count"] = len(valid_df)
        
        return valid_df

    def check_leakage(self, train_df, test_df, patient_col="patient_id"):
        if patient_col in train_df.columns and patient_col in test_df.columns:
            train_patients = set(train_df[patient_col].unique())
            test_patients = set(test_df[patient_col].unique())
            leakage = train_patients.intersection(test_patients)
            if leakage:
                return f"Leakage detected: {len(leakage)} patients in both train and test."
            return "No leakage detected."
        return "Not applicable (no patient_id)."
