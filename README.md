# GCP MLOps Project: Adult Income Prediction

## Problem Description
This project implements a complete **MLOps workflow** on the **Adult Income dataset**. The dataset predicts whether an individual earns **>50K annually** based on demographic and employment attributes. The goal is to demonstrate an end-to-end lifecycle including data ingestion, training, experiment tracking, orchestration, deployment, monitoring, and retraining.

The project covers:
- **Data ingestion** from GCP bucket.
- **Experiment tracking & model registry** with MLflow.
- **Workflow orchestration** with Apache Airflow.
- **Containerization** using Docker & Docker Compose.
- **Model deployment** with FastAPI service.
- **Monitoring** using Evidently (drift detection).
- **Conditional retraining** when drift is detected.
- Follows **best practices** for reproducibility and modularity (TBA).

## Objectives

+ Task: Binary classification — predict whether an adult’s income exceeds $50K/yr based on census features.
+ Why: Useful proxy for understanding socio‑economic patterns, with classic tabular ML challenges (categoricals, imbalance, leakage, drift).
+ Metric: 
   + Primary — AUROC; 
   + Secondary — F1, precision/recall; business metric — false‑positive rate (flagging high‑income when not) kept below threshold.
+ Deployment: Real‑time FastAPI HTTP service for inference; batch scoring job included.
+ Monitoring: Data drift, target drift (if labels arrive), and performance degradation with Evidently;conditionally trigger auto‑retrain DAG.
---

## Project Structure
```
.
├─ airflow/
│  ├─ dags/
│  │  ├─ monitor_and_retrain.py      # Drift detection + conditional retrain
│  │  ├─ train_and_log.py            # Simple DAG to train and log model to MLFlow
│  │  └─ train_register_deploy.py    # Retrain + register model
│  ├─ requirements.txt
├─ src/
│  ├─ ingest/
│  │  └─ ingest.py                   # Load dataset from GCP bucket
│  ├─ training/
│  │  └─ train.py                    # Train + log model to MLflow
│  ├─ serving/
│  │  └─ app.py                      # FastAPI serving + request capture
│  ├─ monitoring/
│  │  ├─ drift_check.py              # Run Evidently drift report
│  │  └─ make_windows.py             # Build current window dataset
├─ data/                             # Data splits, captures, reports
│  ├─ raw/
│  ├─ splits/
│  ├─ capture/
│  └─ reports/
├─ docker-compose.yml
├─ requirements.txt
└─ README.md
```


## Architecture: 

```
            ┌────────────┐      track/register      ┌──────────────┐
raw data ──▶│  training  │─────────────────────────▶│   MLflow     │
(GCS)       │  pipeline  │                          │  Tracking &  │
            └─────┬──────┘                          │   Registry   │
                  │                                 └──────────────┘
                  │ register prod model URI
                  ▼
            ┌────────────┐   serve        ┌───────────────┐
            │   Docker   │───────────────▶│  FastAPI svc  │ (local or Cloud Run)
            └─────┬──────┘                └─────┬─────────┘
                  │                              │
               CI/CD                         monitor
                  │                              │
                  ▼                              ▼
            ┌────────────┐   drift/alerts   ┌──────────────┐
            │  Artifact  │◀────────────────▶│  Evidently   │
            │  Registry  │                  │  reports     │
            └────────────┘                  └──────────────┘

```

---

## Workflow

### Step 1 — Data Ingest & Training
- Download Adult Income dataset from GCP bucket.
- Preprocess and split into train/test.
- Train a scikit-learn pipeline (logistic regression).
- Track experiments in MLflow.

### Step 1.5 — Containerization & Orchestration
- Define **docker-compose.yml** for MLflow, Postgres (Airflow backend), Airflow webserver/scheduler, and FastAPI service.
- Add Airflow DAG to orchestrate training workflow.

### Step 2 — Model Deployment
- Serve trained model using FastAPI.
- Model loaded from MLflow Registry.
- `/predict` endpoint exposed for scoring.
- Inputs captured to CSV for monitoring.

### Step 3 — Monitoring with Evidently
- Capture prediction requests into `data/capture/events.csv`.
- Airflow DAG `monitor_and_retrain` builds rolling `current.csv` window.
- Run Evidently drift detection comparing against training `reference.csv`.
- Generate HTML/JSON reports in `data/reports/`.
- If drift detected → trigger retraining DAG.

### Phase 3.1 — Conditional Retraining
- `train_register_deploy` DAG retrains the model.
- Logs new run to MLflow.
- Registers new version in MLflow Model Registry.
- Placeholder deploy step (can be extended to redeploy FastAPI container or Cloud Run service).

### Phase 4 - Cloud deployment
- Deploy the components (Airflow, MLflow, training, serving, postgres) to an GCE VM.

---

## Setup Guide

### Prerequisites
- Docker & Docker Compose
- Python 3.10+ (prefer 3.12+)
- A created project on GCP
- GCP bucket containing the Adult Income dataset
- Service account key JSON file

### Environment Variables
Create `.env` file:
```dotenv
# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_MODEL_NAME=adult_income_classifier

# Monitoring paths
REF_DATA=/app/data/splits/train.csv
CUR_DATA=/app/data/current.csv
REPORT_DIR=/app/reports
MONITOR_WINDOW_SECS=86400

# Capture settings
CAPTURE_ENABLED=true
CAPTURE_PATH=data/capture/events.csv

# GCP credentials
GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa.json
```

### Start Services
```bash
docker compose up -d postgres mlflow airflow
```
Local access:
- MLflow UI → http://localhost:5000
- Airflow UI → http://localhost:8081 (user: admin@example.com / password: airflow)
- FastAPI service → http://localhost:8080/

### Train Model
```bash
docker compose run --rm app python src/training/train.py
```
- Check MLflow UI for runs.

### Serve Model
```bash
docker compose up -d serving
```
Send request:
```bash
curl -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d  '{
    "age": 31,
    "workclass": "Self-emp-inc",
    "fnlwgt": 117963,
    "education": "Doctorate",
    "education_num": 16,
    "marital_status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Own-child",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States",
    "income": 0
}'
```

### Monitor & Retrain
1. Generate prediction traffic. Simulate inference data with scripts:
```

```

2. Trigger monitoring DAG:
```bash
airflow dags trigger monitor_and_retrain
```
3. Open `data/reports/evidently_report.html` for drift dashboard.
4. If drift detected → `train_register_deploy` DAG runs to retrain & register.

### Deploy docker-compose on a GCE VM

1. **Create VM** (e.g., Ubuntu 22.04, `e2-standard-2`) and allow HTTP/HTTPS.
   ````bash
   gcloud compute instances create my-vm \
   --zone=asia-southeast1-a \
   --machine-type=e2-standard-2 \
   --image-family=ubuntu-2204-lts \
   --image-project=ubuntu-os-cloud \
   --network-interface=network-tier=PREMIUM,subnet=default,access-config=natIP=$(gcloud compute addresses describe my-static-ip --region=asia-southeast1 --format="get(address)")

   ````

2. **SSH in and install Docker & Compose**:
   ```bash
   sudo apt-get update && sudo apt-get install -y ca-certificates curl gnupg
   sudo install -m 0755 -d /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
     https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
     | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io
   sudo usermod -aG docker $USER  # log out/in after this
   ```
3. **Clone repo & set env**:
   ```bash
   git clone https://github.com/anhnd16/nda-mlops-zoomcamp-project.git mlops-adult && cd mlops-adult
   cp .env.example .env  # or create .env
   # Set GCS + MLflow vars inside .env as documented above
   ```
4. **Open firewall (if needed)**: allow port **5000** (MLflow), **8081** (Airflow), **8080** (FastAPI). 

5. **Create a public external IP address** and assign to the VM (from above command).

6. **Start services**:
   ```bash
   docker compose up -d postgres mlflow airflow
   # optional: also bring up serving
   docker compose up -d serving
   ```
7. **Access**:
   - MLflow → `http://<VM_EXTERNAL_IP>:5000`
   - Airflow → `http://<VM_EXTERNAL_IP>:8081`
   - FastAPI → `http://<VM_EXTERNAL_IP>:8080`

---

## Best Practices Implemented
- Experiment tracking & registry via MLflow.
- Orchestration with Airflow DAGs.
- Containerization for reproducibility.
- Monitoring with Evidently.
- Modular codebase: `ingest`, `training`, `serving`, `monitoring`.
- Environment variables managed with `.env`.

## Future Improvements
- Add Terraform for IaC provisioning (GCP buckets, Cloud Run, etc.).
- Add CI/CD pipeline (GitHub Actions).
- Extend deployment step for automated redeploy.
- Integrate performance monitoring (requires ground-truth labels).
- Push drift reports & logs to BigQuery or GCS.

---