# Pipeline design — Astro ML Pipeline Demo

Author: Kuldeep Pawar

## Overview
This repository demonstrates an end-to-end ML pipeline for astronomical tabular data (galaxy classification).
The pipeline includes:
- Data generation (small and large synthetic datasets)
- Preprocessing and feature engineering
- Model training (RandomForest) with MLflow tracking
- Optional dataset versioning with DVC
- Orchestration using Apache Airflow (DAG provided)

## Steps
1. Generate or place your raw dataset in `data/objects.csv`.
2. Run `src/data_pipeline.py` to preprocess and produce `data/processed_objects.csv`.
3. Run `src/model_pipeline.py` to train and log experiments to MLflow.
4. (Optional) Initialize DVC and `dvc add data/objects.csv` to version the dataset.
5. (Optional) Use Airflow to orchestrate preprocessing + training via the provided DAG.
