"""
Auto-select best model by metric and register it in MLflow Model Registry.

Assumptions:
- MLflow tracking server is available at http://localhost:5000
- The experiment name used for training is "astro-ml-comparison" (change if needed)
- Metric used for selection is 'r2' (higher-is-better). For lower-is-better metrics use logic below.
"""

import mlflow
from mlflow.exceptions import MlflowException

# CONFIG — adjust if necessary
TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "astro-ml-model-comparison"    # change if your runs are in another experiment
SELECTION_METRIC = "r2"                    # metric to use for selection
HIGHER_IS_BETTER = True                    # set False if lower is better (e.g., mse)
REGISTERED_MODEL_NAME = "astro-ml-model-comparison"

def main():
    # Connect to server
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Resolve experiment
    try:
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            raise SystemExit(f"Experiment '{EXPERIMENT_NAME}' not found. Check experiment name.")
        experiment_id = exp.experiment_id
    except MlflowException as e:
        raise SystemExit(f"Failed to get experiment: {e}")

    # Search runs for experiment and pick the best by metric
    runs = client.search_runs(experiment_ids=[experiment_id], max_results=1000)
    if not runs:
        raise SystemExit("No runs found in the experiment.")

    # Filter runs that have the selection metric
    runs_with_metric = []
    for r in runs:
        metrics = r.data.metrics
        if SELECTION_METRIC in metrics:
            runs_with_metric.append((r.info.run_id, metrics[SELECTION_METRIC]))

    if not runs_with_metric:
        raise SystemExit(f"No runs have metric '{SELECTION_METRIC}' logged.")

    # Select best
    if HIGHER_IS_BETTER:
        best_run = max(runs_with_metric, key=lambda x: x[1])
    else:
        best_run = min(runs_with_metric, key=lambda x: x[1])

    best_run_id, best_metric_value = best_run
    print(f"Best run id: {best_run_id}  {SELECTION_METRIC} = {best_metric_value}")

    # Register the model artifact from the best run
    # We assume the model artifact is logged under artifact_path 'model' (common), 
    # or you can point to the run artifact root directly.
    # If your model artifact name is different, adjust `artifact_path` accordingly.
    artifact_path = "model"  # change if your models were logged with a different artifact path

    model_source = f"runs:/{best_run_id}/{artifact_path}"
    print(f"Registering model from: {model_source} as '{REGISTERED_MODEL_NAME}'")

    try:
        # Check if model exists
        try:
            client.get_registered_model(REGISTERED_MODEL_NAME)
            model_exists = True
        except MlflowException:
            model_exists = False

        # Register new model version
        mv = client.create_model_version(name=REGISTERED_MODEL_NAME, source=model_source, run_id=best_run_id)
        print(f"Created model version: {mv.version}")

        # Optionally transition to 'Staging' automatically
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=False
        )
        print(f"Model version {mv.version} transitioned to 'Staging'")

    except MlflowException as e:
        raise SystemExit(f"Failed to register model: {e}")

if __name__ == "__main__":
    main()
