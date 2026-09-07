import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

def create_data_quality_report(df, filename='data_quality_report.csv'):
    report = pd.DataFrame({
        'DataType': df.dtypes,
        'MissingValues': df.isnull().sum(),
        'MissingPercentage': (df.isnull().sum() / len(df)) * 100,
        'UniqueValues': df.nunique()
    })
    report.to_csv(filename)
    return report

def feature_engineering(df):
    df_engineered = df.copy()
    
    # 1. Age groups
    if 'age' in df_engineered.columns:
        df_engineered['age_group'] = pd.cut(df_engineered['age'], bins=[0, 40, 60, 80, 120], labels=['<40', '40-60', '60-80', '>80'])
        
    # 2. BMI categories
    if 'bmi' in df_engineered.columns:
        df_engineered['bmi_category'] = pd.cut(df_engineered['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
    # 3. Metastasis count
    meta_cols = [c for c in df_engineered.columns if 'metastasis' in c]
    df_engineered['metastasis_count'] = df_engineered[meta_cols].sum(axis=1)
    
    return df_engineered

def main():
    print("Loading data...")
    df = pd.read_csv('oncology_dataset.csv')
    
    print(f"Original shape: {df.shape}")
    
    # 1. Remove duplicate patient records
    df = df.drop_duplicates(subset=['patient_id'], keep='first')
    print(f"Shape after duplicate removal: {df.shape}")
    
    # 2. Data Audit
    report = create_data_quality_report(df)
    
    # 3. Prevent Data Leakage
    # Drop variables collected post-treatment
    leakage_cols = [
        'overall_survival_months', 'os_event_flag',
        'progression_free_survival_months', 'pfs_event_flag',
        'time_to_response_months', 'duration_of_response_months',
        'best_overall_response', 'ct_3mo_tumor_mm', 'ct_3mo_recist',
        'ct_6mo_tumor_mm', 'ct_6mo_recist', 'ct_12mo_tumor_mm',
        'ct_12mo_recist', 'ct_24mo_tumor_mm', 'ct_24mo_recist',
        'adverse_event_grade'
    ]
    
    target_col = 'treatment_response_flag'
    id_col = 'patient_id'
    
    cols_to_drop = leakage_cols + [id_col]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=cols_to_drop + [target_col])
    y = df[target_col]
    
    # 4. Feature Engineering
    X = feature_engineering(X)
    
    # Identify numerical and categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")
    
    # 5. Missing Values & Encoding within a ColumnTransformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # 6. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Class distribution in training: {y_train.value_counts(normalize=True).to_dict()}")
    
    # 7. Model Comparison & Hyperparameter Tuning
    models = {
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42)
    }
    
    param_grids = {
        'LogisticRegression': {
            'classifier__C': [0.1, 1, 10]
        },
        'RandomForest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5, 10]
        },
        'HistGradientBoosting': {
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_iter': [50, 100, 200],
            'classifier__max_depth': [None, 5, 10]
        }
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_models = {}
    model_results = []
    
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        
        search = RandomizedSearchCV(
            pipeline, 
            param_distributions=param_grids[name], 
            n_iter=5, 
            cv=cv, 
            scoring='roc_auc', 
            n_jobs=-1, 
            random_state=42
        )
        search.fit(X_train, y_train)
        
        best_models[name] = search.best_estimator_
        
        y_train_pred = search.predict(X_train)
        y_test_pred = search.predict(X_test)
        y_test_proba = search.predict_proba(X_test)[:, 1] if hasattr(search, 'predict_proba') else search.decision_function(X_test)
        
        model_results.append({
            'Model': name,
            'Train_Accuracy': accuracy_score(y_train, y_train_pred),
            'Test_Accuracy': accuracy_score(y_test, y_test_pred),
            'Test_Balanced_Accuracy': balanced_accuracy_score(y_test, y_test_pred),
            'Test_Precision': precision_score(y_test, y_test_pred),
            'Test_Recall': recall_score(y_test, y_test_pred),
            'Test_F1': f1_score(y_test, y_test_pred),
            'Test_ROC_AUC': roc_auc_score(y_test, y_test_proba),
            'CV_ROC_AUC_Mean': search.cv_results_['mean_test_score'][search.best_index_],
            'CV_ROC_AUC_Std': search.cv_results_['std_test_score'][search.best_index_]
        })
        
    results_df = pd.DataFrame(model_results)
    results_df.to_csv('model_comparison.csv', index=False)
    print("Model comparison saved to model_comparison.csv")
    
    # Select Best Model based on CV ROC_AUC
    best_model_name = results_df.sort_values(by='CV_ROC_AUC_Mean', ascending=False).iloc[0]['Model']
    print(f"\nBest Model: {best_model_name}")
    final_model = best_models[best_model_name]
    
    # 8. Evaluation
    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)[:, 1] if hasattr(final_model, 'predict_proba') else final_model.decision_function(X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Feature Importance (Permutation)
    print("Calculating feature importance...")
    result = permutation_importance(final_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
    top_features = importances.head(15)
    
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title('Top 15 Feature Importances (Permutation)')
    plt.xlabel('Mean Accuracy Decrease')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Error Analysis
    false_negatives = X_test[(y_test == 1) & (y_test_pred == 0)]
    false_positives = X_test[(y_test == 0) & (y_test_pred == 1)]
    
    with open('evaluation_report.txt', 'w') as f:
        f.write("=== Evaluation Report ===\n\n")
        f.write(f"Best Model Selected: {best_model_name}\n")
        f.write(f"Classification Report:\n{classification_report(y_test, y_test_pred)}\n")
        f.write(f"\nError Analysis:\n")
        f.write(f"False Negatives (Missed Responders): {len(false_negatives)}\n")
        f.write(f"False Positives (Incorrectly Predicted Responders): {len(false_positives)}\n")
        
        train_acc = results_df[results_df['Model'] == best_model_name]['Train_Accuracy'].values[0]
        test_acc = results_df[results_df['Model'] == best_model_name]['Test_Accuracy'].values[0]
        
        f.write(f"\nOverfitting Check:\n")
        f.write(f"Train Accuracy: {train_acc:.4f}\n")
        f.write(f"Test Accuracy:  {test_acc:.4f}\n")
        if train_acc - test_acc > 0.1:
            f.write("Warning: Model might be overfitting (Train Acc significantly higher than Test Acc).\n")
        else:
            f.write("Model does not show signs of severe overfitting.\n")
            
        f.write("\nFeatures Dropped (Leakage/IDs):\n")
        f.write(", ".join(cols_to_drop) + "\n")
        
    # Save the pipeline
    joblib.dump(final_model, 'final_model.pkl')
    
    # Save feature list
    with open('feature_list.txt', 'w') as f:
        f.write(",\n".join(X.columns.tolist()))
        
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
