# 🚀 Astro ML Pipeline Demo

**Author:** Kuldeep Pawar  
**Email:** [kuldeepawar01@gmail.com]  
**GitHub:** [https://github.com/kuldeepawar/astro-ml-pipeline-demo](https://github.com/kuldeepawar/astro-ml-pipeline-demo)

---

## 🧠 Overview
This project demonstrates an **end-to-end Machine Learning pipeline** for astronomical data using **Python, MLflow, and Airflow-style modularization**.

It covers:
- Data ingestion & preprocessing
- Model training, comparison, and selection
- Experiment tracking with MLflow
- Model registry automation (best model auto-registration)
- Reproducibility via environment & version control

---

## ⚙️ Tech Stack
- **Language:** Python 3.10+
- **ML Frameworks:** scikit-learn, pandas, numpy
- **MLOps Tools:** MLflow, DVC, Git
- **Pipeline Scripts:** Modular `src/` folder
- **Visualization:** Jupyter notebooks

---

## 🧩 Project Structure

astro-ml-pipeline-demo/
├── src/
│ ├── data_pipeline.py
│ ├── model_pipeline.py
│ ├── model_compare_pipeline.py
│ └── auto_register_best.py
├── notebooks/
│ └── exploratory_analysis.ipynb
├── docs/
│ ├── pipeline_design.md
│ └── data_description.md
├── scripts/
│ └── generate_sample_data.py
├── requirements.txt
└── README.md


---

## 📊 MLflow Integration
MLflow tracks every model training run, records metrics, and manages model registry.

**Experiment View:**
![MLflow Experiments](docs/images/mlflow_experiments.png)

**Registered Model:**
![Registered Model](docs/images/mlflow_registered_model.png)

---

## 📦 Artifacts & Reproducibility
All MLflow artifacts (`mlruns/`, `mlartifacts/`) are backed up separately in this release:  
👉 [📁 Download Release: Astro ML Artifacts Backup](../../releases)

---

## 🧠 Key Features
- Modular, reusable ML pipeline design
- Automated model comparison and best-model registration
- MLflow tracking & registry integration
- Ready for scaling or Docker/Airflow integration

---

## 🧾 License
Licensed under the [MIT License](LICENSE).

---

## 👥 Contributing
Pull requests and feedback are welcome!

---

## 📄 Project Documentation
See:
- [`docs/pipeline_design.md`](docs/pipeline_design.md)
- [`docs/data_description.md`](docs/data_description.md)


