import pandas as pd
from sklearn.model_selection import train_test_split
import random

class DataSplitter:
    def __init__(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        random.seed(random_state)

    def split_dataframe(self, df, patient_col="patient_id"):
        if patient_col in df.columns:
            # Patient-level split
            patients = list(df[patient_col].dropna().unique())
            train_val_patients, test_patients = train_test_split(
                patients, test_size=self.test_ratio, random_state=self.random_state
            )
            
            # Adjusted validation ratio for the remaining training set
            val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
            train_patients, val_patients = train_test_split(
                train_val_patients, test_size=val_ratio_adjusted, random_state=self.random_state
            )
            
            train_df = df[df[patient_col].isin(train_patients)].copy()
            val_df = df[df[patient_col].isin(val_patients)].copy()
            test_df = df[df[patient_col].isin(test_patients)].copy()
            
        else:
            # Random split
            train_val_df, test_df = train_test_split(
                df, test_size=self.test_ratio, random_state=self.random_state
            )
            val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
            train_df, val_df = train_test_split(
                train_val_df, test_size=val_ratio_adjusted, random_state=self.random_state
            )
            
        return train_df, val_df, test_df

    def split_list(self, data):
        # Assuming no patient_id in ner_training_data unless specified in JSON (but our data might not have it)
        # We will do a random split
        train_val, test = train_test_split(data, test_size=self.test_ratio, random_state=self.random_state)
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        train, val = train_test_split(train_val, test_size=val_ratio_adjusted, random_state=self.random_state)
        
        return train, val, test
