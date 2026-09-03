# Simple Evaluation Report

**01. Accuracy**: 99.25%
**02. Precision**: 99.25%
**03. Recall**: 99.25%
**04. F1 Score**: 99.25%
**05. ROC-AUC**: 0.9999

**06. Confusion Matrix**
- **High Risk**: 99 correctly predicted (1 missed)
- **Low Risk**: 161 correctly predicted (0 missed)
- **Moderate Risk**: 137 correctly predicted (2 missed)

**07. Cross-validation results**
- Average Accuracy: **98.85%** (5 folds).

**08 & 09. Error Analysis & Severe-risk False Negatives**
- Severe false negatives: **0**. The model never predicted a high-risk patient as low-risk.

**10. Overfitting check and stress test**
- Train Accuracy: 100% | Test Accuracy: 99.25%. 
- No significant overfitting detected.

**11. Model comparison**
- The **Voting Ensemble (99.25%)** was the best, slightly beating XGBoost and Random Forest (98.50%).

**12. Feature importance**
- Top 3 predictors: `mutation_count`, `dosage_mg`, and `WBC`.

**13. Final evaluation report**
- **Stress Test**: 0 overconfident errors (where confidence >80% but wrong).
- **Conclusion**: The model is exceptionally safe, highly accurate, and ready for use.
