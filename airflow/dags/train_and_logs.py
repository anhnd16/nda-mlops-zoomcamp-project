from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import sys

# Make project code visible to Airflow
PROJECT_ROOT = "/app"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.train import main as train_main

with DAG(
    dag_id="train_and_log",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops", "adult"],
) as dag:

    # Ensure the tracking URI is correct inside the container
    def set_tracking_uri():
        uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        os.environ["MLFLOW_TRACKING_URI"] = uri
        return uri

    init = PythonOperator(
        task_id="init_env",
        python_callable=set_tracking_uri,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_main,
    )

    init >> train