# Stage 03 Oncology NLP Data Engineering

This directory contains the data engineering and preparation pipeline for Stage 03 of the oncology NLP project. 
The pipeline processes raw, synthetic oncology text datasets into a clean, validated, and model-ready format for the NLP Engineer.

## Datasets Processed
1. `pathology_clinical_notes.csv`
2. `urgency_symptom_logs.csv`
3. `drug_adverse_events.csv`
4. `oncology_guideline_chunks.csv`
5. `gene_mutation_dictionary.csv`
6. `ner_training_data.jsonl`

## Pipeline Overview
The pipeline consists of the following modules in `src/`:
- `data_loader.py`: Loads CSV and JSONL files.
- `data_cleaning.py`: Performs whitespace normalization, unicode normalization, removes control characters, and standardizes capitalization without aggressive lemmatization.
- `data_validation.py`: Validates labels, filters invalid records, verifies NER overlapping/boundaries, converts NER data into BIO format, and generates metrics for the data quality report.
- `data_split.py`: Performs 70/15/15 Train/Val/Test splits, maintaining patient-level grouping to prevent data leakage.
- `main.py`: Orchestrator that runs the entire pipeline end-to-end.

## How to Run
Ensure you have the required dependencies:
```bash
pip install pandas scikit-learn
```

Run the pipeline from the project root:
```bash
python "Stage 3/src/main.py"
```

## Outputs
- Clean datasets are stored in `data/processed/`.
- `outputs/data_quality_report.csv` and `.json` summarize the number of records dropped due to invalid labels, duplicates, missing values, empty text, or entity annotation errors.
