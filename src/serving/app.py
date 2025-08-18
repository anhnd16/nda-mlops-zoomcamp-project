import os, mlflow, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

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
    return {"probability_gt_50k": score}
