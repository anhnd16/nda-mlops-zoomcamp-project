import os, json
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def run_drift_check(ref_path: str, cur_path: str, out_html: str, threshold: float = 0.3) -> bool:
    """
    Run Evidently drift check. Returns True if drift is detected above threshold.
    """
    ref = pd.read_csv(ref_path)
    cur = pd.read_csv(cur_path)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    report.save_html(out_html)

    summary = report.as_dict()
    drift_score = summary["metrics"][0]["result"]["dataset_drift"]
    print(f"[Drift Check] dataset_drift={drift_score}")

    return drift_score > threshold


def run_evidently(ref_path: str, cur_path: str, out_dir: str = "reports") -> bool:
    os.makedirs(out_dir, exist_ok=True)
    ref = pd.read_csv(ref_path)
    cur = pd.read_csv(cur_path)
    # Keep only feature columns known to training (ignore label/prediction/ts)
    drop_cols = [c for c in ["income", "prediction", "ts"] if c in cur.columns]
    cur = cur.drop(columns=drop_cols, errors="ignore")
    ref = ref.drop(columns=[c for c in ["income", "prediction", "ts"] if c in ref.columns], errors="ignore")

    report = Report([DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    html_path = os.path.join(out_dir, "evidently_report.html")
    json_path = os.path.join(out_dir, "evidently_report.json")
    report.save_html(html_path)
    data = report.as_dict()
    with open(json_path, "w") as f:
        json.dump(data, f)

    # dataset_drift is a boolean in the preset result
    drift = False
    try:
        drift = data["metrics"][0]["result"].get("dataset_drift", False)
    except Exception:
        pass
    return bool(drift)