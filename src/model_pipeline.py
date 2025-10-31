# src/model_pipeline.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed_objects.csv"
MODEL_DIR = ROOT / "models"

def train(infile=None, n_estimators=50):
    infile = Path(infile or PROC)
    df = pd.read_csv(infile)
    X = df[['magnitude_norm','color_index','size','ellipticity']]
    y = df['label_num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metrics({f"class_{k}_f1": v['f1-score'] for k,v in report.items() if k in ['0','1']})

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.sklearn.log_model(clf, "model")
        print(f'Finished training. Accuracy: {acc:.4f}')

if __name__ == '__main__':
    train()
