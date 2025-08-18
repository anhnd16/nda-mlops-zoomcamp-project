from datetime import datetime
import os

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator

# Env defaults
REF_DATA = os.getenv("REF_DATA", "/app/data/splits/train.csv")
CUR_DATA = os.getenv("CUR_DATA", "/app/data/current.csv")
REPORT_DIR = os.getenv("REPORT_DIR", "/app/reports")
WINDOW_SECS = int(os.getenv("MONITOR_WINDOW_SECS", str(24*3600)))

# Make project code importable
import sys
PROJECT_ROOT = "/app"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.monitoring.make_windows import build_current_window
from src.monitoring.drift_check import run_evidently


def make_window_task(**_):
    return build_current_window()


def run_evidently_task(**_):
    return run_evidently(REF_DATA, CUR_DATA, REPORT_DIR)


def branch_on_drift(**ctx):
    drift = ctx["ti"].xcom_pull(task_ids="evidently")
    return "trigger_retrain" if drift else "skip"

with DAG(
    dag_id="monitor_and_retrain",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@hourly",
    catchup=False,
    tags=["monitoring", "evidently"],
) as dag:

    build_window = PythonOperator(task_id="build_window", python_callable=make_window_task)
    evidently = PythonOperator(task_id="evidently", python_callable=run_evidently_task)

    decide = BranchPythonOperator(task_id="decide", python_callable=branch_on_drift)

    trigger = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="train_register_deploy",
        reset_dag_run=True,
        wait_for_completion=False,
    )

    skip = EmptyOperator(task_id="skip")
    done = EmptyOperator(task_id="done", trigger_rule="none_failed_min_one_success")

    build_window >> evidently >> decide
    decide >> trigger >> done
    decide >> skip >> done