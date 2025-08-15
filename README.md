# nda-mlops-zoomcamp-project

## Project description:
#TODO:


## Objectives

+ Task: Binary classification — predict whether an adult’s income exceeds $50K/yr based on census features.
+ Why: Useful proxy for understanding socio‑economic patterns, with classic tabular ML challenges (categoricals, imbalance, leakage, drift).
+ Metric: Primary — AUROC; Secondary — F1, precision/recall; business metric — false‑positive rate (flagging high‑income when not) kept below threshold.
+ Deployment: Real‑time FastAPI HTTP service for inference; batch scoring job included.
+ Monitoring: Data drift, target drift (if labels arrive), and performance degradation with Evidently; alerts via Pub/Sub or Slack webhook; optional auto‑retrain DAG.



## Architecture: 

```
            ┌────────────┐      track/register      ┌──────────────┐
raw data ──▶│  training  │─────────────────────────▶│   MLflow     │
            │  pipeline  │                          │  Tracking &  │
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
            │  Registry  │  Pub/Sub/Slack   │  reports     │
            └────────────┘                  └──────────────┘

```

## Repository structure:

```
mlops-adult-income/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ Makefile
├─ .env.example
├─ .pre-commit-config.yaml
├─ .flake8
├─ docker/
│  ├─ Dockerfile.serving
│  ├─ Dockerfile.training
│  └─ gunicorn_conf.py
├─ docker-compose.yml
├─ airflow/
│  ├─ dags/
│  │  ├─ train_register_deploy.py
│  │  └─ monitor_and_retrain.py
│  └─ requirements.txt
├─ infra/terraform/
│  ├─ main.tf
│  ├─ variables.tf
│  ├─ outputs.tf
│  └─ cloud_run.tf
├─ src/
│  ├─ data/
│  │  ├─ ingest.py
│  │  └─ schema.py
│  ├─ features/
│  │  └─ build_features.py
│  ├─ models/
│  │  ├─ train.py
│  │  ├─ evaluate.py
│  │  ├─ registry.py
│  │  └─ infer.py
│  ├─ pipeline/
│  │  ├─ training_pipeline.py
│  │  └─ batch_scoring.py
│  ├─ serving/
│  │  ├─ app.py
│  │  └─ utils.py
│  └─ monitoring/
│     ├─ generate_reports.py
│     ├─ drift_checks.py
│     └─ alerting.py
├─ tests/
│  ├─ test_features.py
│  └─ test_training.py
└─ ci/
   └─ github/
      └─ workflows/
         └─ ci_cd.yml
```

