import os
import mlflow

def set_tracking():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "imgcls-baseline")
    mlflow.set_experiment(exp_name)

def log_params(params: dict):
    for k, v in params.items():
        mlflow.log_param(k, v)

def log_metrics(metrics: dict, step: int | None = None):
    for k, v in metrics.items():
        mlflow.log_metric(k, v, step=step)

def register_model(run_id: str, artifact_path: str, model_name: str):
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    return mv
