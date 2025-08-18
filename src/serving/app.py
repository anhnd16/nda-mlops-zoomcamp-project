import os, mlflow, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os, csv, time
from pathlib import Path

CAPTURE_ENABLED = os.getenv("CAPTURE_ENABLED", "true").lower() == "true"
CAPTURE_PATH = os.getenv("CAPTURE_PATH", "data/capture/events.csv")
CAPTURE_DIR = Path(CAPTURE_PATH).parent
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

        
MODEL_URI = os.getenv("MODEL_URI") or f"models:/{os.getenv('MLFLOW_MODEL_NAME','adult_income_classifier')}/Production"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="Adult Income Service", version="1.0.0")
_model = None
def get_model():
    global _model
    if _model is None:
        _model = mlflow.pyfunc.load_model(MODEL_URI)
    return _model

class Item(BaseModel):
    # Full set expected by the training pipeline
    age: int
    workclass: str | None = None
    fnlwgt: int = 0
    education: str | None = None
    education_num: int = 0
    marital_status: str | None = None
    occupation: str | None = None
    relationship: str | None = None
    race: str | None = None
    sex: str | None = None
    capital_gain: int = 0
    capital_loss: int = 0
    hours_per_week: int = 40
    native_country: str | None = None

@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI}

@app.post("/predict")
def predict(item: Item):
    df = pd.DataFrame([item.dict()])
    y = get_model().predict(df)
    score = float(y[0]) if hasattr(y, "__getitem__") else float(y)

    record = item.dict() | {"prediction": float(score), "ts": int(time.time())}
    if CAPTURE_ENABLED:
        file_exists = Path(CAPTURE_PATH).exists()
        with open(CAPTURE_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(record.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    return {"probability_gt_50k": score}
