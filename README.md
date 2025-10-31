# 🚀 Astro ML Pipeline Demo

An end-to-end **Machine Learning pipeline for astronomical data** — built with Python, DVC, MLflow, and Airflow-ready structure.  
This project demonstrates how to automate data preparation, model training, experiment tracking, and model comparison.

---

## 🌌 Project Overview

| Stage | Description |
|--------|--------------|
| **1. Data Pipeline** | Generates and preprocesses synthetic astronomical data for regression tasks. |
| **2. Model Training** | Trains baseline models (Linear Regression, Decision Tree, Random Forest). |
| **3. Experiment Tracking** | Logs metrics, parameters, and artifacts with **MLflow**. |
| **4. Model Comparison** | Compares models using R² and MSE, visualized in MLflow UI. |
| **5. Auto-Registration** | Automatically registers the best model in the MLflow Model Registry. |

---

## 🧠 Tech Stack

- **Python 3.11+**
- **MLflow** — experiment tracking & model registry  
- **scikit-learn** — machine learning models  
- **pandas**, **numpy**, **matplotlib** — data handling  
- *(Optional)* **Airflow / DVC** — for pipeline orchestration & version control

---

source venv/Scripts/activate
pip install -r requirements.txt
python src/model_compare_pipeline.py
python src/auto_register_best.py
python -m mlflow ui --backend-store-uri ./mlruns --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
# Open http://localhost:5000


## 📊 MLflow Experiment Tracking

Below is the MLflow dashboard showing multiple tracked experiments during model training and comparison.

![MLflow Experiments](docs/images/mlflow_experiments.png)


## 🧠 Registered Model in MLflow
![Registered Model](docs/images/mlflow_registered_model.png)



