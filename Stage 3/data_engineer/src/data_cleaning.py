import pandas as pd
import unicodedata
import re

class DataCleaner:
    def __init__(self):
        pass

    def clean_text(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return text
        
        # 1. Unicode normalization (NFKC to ensure consistency)
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Removal of accidental control characters, keep newlines/tabs
        # Keeps printable characters and standard whitespace
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in ['\n', '\r', '\t'])
        
        # 3. Whitespace normalization (replace multiple spaces with single space, strip)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()
        
        return text

    def clean_urgency_data(self, df):
        df = df.copy()
        df['text'] = df['text'].apply(self.clean_text)
        
        # Map labels to integer encoding
        label_map = {'LOW': 0, 'MODERATE': 1, 'HIGH': 2, 'CRITICAL': 3}
        # Keep track of mapping for later
        self.urgency_label_map = label_map
        
        # Validate/clean labels - convert to uppercase to be safe
        df['urgency'] = df['urgency'].str.upper()
        # Drop rows with invalid or missing urgency to be safe, or just map what exists
        df['urgency_encoded'] = df['urgency'].map(label_map)
        
        return df

    def clean_drug_data(self, df):
        df = df.copy()
        # Normalize fields to lowercase and strip whitespace for consistency
        for col in ['drug_name', 'adverse_event', 'severity', 'route', 'dosage']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
                df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x))
        
        # Remove exact duplicates
        df = df.drop_duplicates()
        return df

    def clean_gene_data(self, df):
        df = df.copy()
        # Normalize gene symbols (usually uppercase), mutation, associated_cancer
        if 'gene' in df.columns:
            df['gene'] = df['gene'].astype(str).str.upper().str.strip()
        if 'mutation' in df.columns:
            df['mutation'] = df['mutation'].astype(str).str.strip().apply(lambda x: re.sub(r'\s+', ' ', x))
        if 'associated_cancer' in df.columns:
            df['associated_cancer'] = df['associated_cancer'].astype(str).str.strip()
        
        df = df.drop_duplicates()
        return df

    def clean_guideline_data(self, df):
        df = df.copy()
        if 'source' in df.columns:
            df['source'] = df['source'].astype(str).str.strip()
        if 'section' in df.columns:
            df['section'] = df['section'].astype(str).str.strip()
        if 'text' in df.columns:
            df['text'] = df['text'].apply(self.clean_text)
        
        df = df.drop_duplicates()
        return df
