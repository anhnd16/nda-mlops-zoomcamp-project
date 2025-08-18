import os
import mlflow
import mlflow.sklearn  # ensures sklearn flavor available
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

from data.ingest import load_adult
from data.preprocess import clean, build_preprocessor, TARGET



EXP_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "adult-income-phase1")
RANDOM_STATE = int(os.getenv("SEED", 42))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))


def _log_confusion_matrix(y_true, y_pred, out_dir: str = "artifacts"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in zip([(0,0),(0,1),(1,0),(1,1)], cm.flatten()):
        plt.text(j, i, str(v), ha="center", va="center")
    fig_path = Path(out_dir) / "confusion_matrix.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(fig_path), artifact_path="plots")


def main():
    # Load environment variables from .env file
    if load_dotenv(): 
        print("Loaded .env file.")
    
    # Optional: respect a user-provided tracking URI, else default local ./mlruns
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    mlflow.set_experiment(EXP_NAME)

    with mlflow.start_run() as run:
        # Load & clean
        raw = load_adult(to_csv=True)
        df = clean(raw)

        print("Start training data....")
        train_df, test_df = train_test_split(
            df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
        )

        X_train = train_df.drop(columns=[TARGET])
        y_train = train_df[TARGET]
        X_test = test_df.drop(columns=[TARGET])
        y_test = test_df[TARGET]

        # Model pipeline
        pre = build_preprocessor()
        clf = LogisticRegression(max_iter=1000)
        pipe = Pipeline([("pre", pre), ("clf", clf)])

        # Log basic params for reproducibility
        mlflow.log_params({
            "model": "LogisticRegression",
            "max_iter": 1000,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        })

        # Train & evaluate
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        auroc = roc_auc_score(y_test, proba)
        f1 = f1_score(y_test, pred)

        mlflow.log_metric("auroc", float(auroc))
        mlflow.log_metric("f1", float(f1))

        # Artifacts: model + confusion matrix
        mlflow.sklearn.log_model(pipe, artifact_path="model")
        _log_confusion_matrix(y_test, pred)

        # Save the exact train/test CSVs (helps with reproducibility)
        outp = Path("data/splits")
        outp.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(outp / "train.csv", index=False)
        test_df.to_csv(outp / "test.csv", index=False)
        mlflow.log_artifact(str(outp / "train.csv"), artifact_path="data")
        mlflow.log_artifact(str(outp / "test.csv"), artifact_path="data")

        print({"auroc": round(float(auroc), 4), "f1": round(float(f1), 4)})


if __name__ == "__main__":
    main()