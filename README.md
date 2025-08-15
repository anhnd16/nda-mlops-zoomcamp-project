# nda-mlops-zoomcamp-project


+ Task: Binary classification — predict whether an adult’s income exceeds $50K/yr based on census features.
+ Why: Useful proxy for understanding socio‑economic patterns, with classic tabular ML challenges (categoricals, imbalance, leakage, drift).
+ Metric: Primary — AUROC; Secondary — F1, precision/recall; business metric — false‑positive rate (flagging high‑income when not) kept below threshold.
+ Deployment: Real‑time FastAPI HTTP service for inference; batch scoring job included.
+ Monitoring: Data drift, target drift (if labels arrive), and performance degradation with Evidently; alerts via Pub/Sub or Slack webhook; optional auto‑retrain DAG.