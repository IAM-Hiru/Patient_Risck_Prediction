import pandas as pd
import joblib

def load_and_evaluate(model_path='final_model.pkl', data_path='oncology_dataset.csv'):
    # Load the trained pipeline
    print(f"Loading model from {model_path}...")
    pipeline = joblib.load(model_path)
    
    # Load some new data (for demonstration, using a sample from the dataset)
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Simulate receiving "new" data (e.g. first 5 records)
    new_data = df.head(5).copy()
    
    # Ensure any feature engineering done during training is also applied here
    # (Since we didn't include feature_engineering in the scikit-learn Pipeline directly,
    # we must call the same function before predicting).
    from train_pipeline import feature_engineering
    new_data = feature_engineering(new_data)
    
    # Drop target and leakage columns (as would be the case in production)
    leakage_cols = [
        'overall_survival_months', 'os_event_flag',
        'progression_free_survival_months', 'pfs_event_flag',
        'time_to_response_months', 'duration_of_response_months',
        'best_overall_response', 'ct_3mo_tumor_mm', 'ct_3mo_recist',
        'ct_6mo_tumor_mm', 'ct_6mo_recist', 'ct_12mo_tumor_mm',
        'ct_12mo_recist', 'ct_24mo_tumor_mm', 'ct_24mo_recist',
        'adverse_event_grade', 'patient_id', 'treatment_response_flag'
    ]
    cols_to_drop = [c for c in leakage_cols if c in new_data.columns]
    X_new = new_data.drop(columns=cols_to_drop)
    
    # Predict
    print("Running predictions on new data...")
    predictions = pipeline.predict(X_new)
    probabilities = pipeline.predict_proba(X_new)[:, 1] if hasattr(pipeline, 'predict_proba') else pipeline.decision_function(X_new)
    
    results = pd.DataFrame({
        'PatientIndex': range(len(new_data)),
        'PredictedResponseFlag': predictions,
        'Probability': probabilities
    })
    
    print("\nPrediction Results:")
    print(results)
    
if __name__ == "__main__":
    load_and_evaluate()
