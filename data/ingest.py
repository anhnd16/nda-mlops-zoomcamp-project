import io
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from google.cloud import storage

# Column schema for the classic Adult Income dataset (UCI)
COLUMNS = [
    "age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","sex","capital-gain","capital-loss",
    "hours-per-week","native-country","income",
]

GCP_PROJECT_ID = "nda-de-zoomcamp"
GCS_BUCKET = "nda-mlops-zoomcamp-bucket"
BLOB_NAME = "uci/adult.data"
SA_KEY = ""
    

def _download_gcs_bytes(bucket: str, blob_name: str, project: Optional[str] = None) -> bytes:
    """Download an object from GCS and return its bytes."""
    sa_json_key = SA_KEY
    client = storage.Client(project=project)  # relies on ADC; set GOOGLE_APPLICATION_CREDENTIALS if needed
    blob = client.bucket(bucket).blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{blob_name} not found")
    return blob.download_as_bytes()


def load_adult_from_gcs(
    bucket: Optional[str] = None,
    blob_name: Optional[str] = None,
    *,
    project: Optional[str] = None,
    has_header: bool = False,
    to_csv: bool = True,
    out_dir: str = "data/raw",
    fname: str = "adult_train.csv",
) -> pd.DataFrame:
    """
    Load the Adult Income dataset from a GCS bucket.

    Environment fallbacks:
      - GCP_PROJECT_ID -> project
      - GCS_BUCKET -> bucket
      - GCS_BLOB -> blob_name (e.g., "adult/adult.data" or "adult_train.csv")

    If your object already contains headers, set has_header=True.
    """
    project = project or os.getenv("GCP_PROJECT_ID", None) or GCP_PROJECT_ID
    bucket = bucket or os.getenv("GCS_BUCKET")
    blob_name = blob_name or os.getenv("GCS_BLOB")

    if not bucket or not blob_name:
        raise ValueError("bucket and blob_name are required (or set GCS_BUCKET and GCS_BLOB env vars)")

    raw_bytes = _download_gcs_bytes(bucket, blob_name, project)
    header = 0 if has_header else None
    df = pd.read_csv(io.BytesIO(raw_bytes), header=header, names=None if has_header else COLUMNS, na_values="?", skipinitialspace=True)

    if to_csv:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        df.to_csv(path / fname, index=False)
    return df


# Backwards-compatible wrapper name used by training script
load_adult = load_adult_from_gcs


if __name__ == "__main__":
    # Example manual run:
    #   export GCS_BUCKET=your-bucket
    #   export GCS_BLOB=adult/adult.data
    #   python -m src.data.ingest
    df = load_adult_from_gcs(
        bucket=GCS_BUCKET,
        blob_name=BLOB_NAME,
    )
    print(df.head())
    
