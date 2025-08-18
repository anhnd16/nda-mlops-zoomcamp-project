from datetime import datetime
import os, sys

from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_ROOT = "/app"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models import train
import mlflow

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "adult_income_classifier")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def retrain_and_register(**_):
    model, metrics, run_id = train.train_model()
    mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    return metrics


def deploy_latest(**_):
    client = mlflow.tracking.MlflowClient()
    latest = client.get_latest_versions(MODEL_NAME, stages=["None"])[-1]
    
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest,
        stage="Production"
    )
    print(f"Latest model version {latest.version} registered.")
    print(f"Deployed model version {latest.version} of '{MODEL_NAME}' to Production.")


with DAG(
    dag_id="train_register_deploy",
    start_date=datetime(2025,1,1),
    schedule_interval=None,
    catchup=False,
    tags=["training","mlflow"],
) as dag:

    retrain = PythonOperator(task_id="retrain_and_register", python_callable=retrain_and_register)
    deploy = PythonOperator(task_id="deploy", python_callable=deploy_latest)

    retrain >> deploy