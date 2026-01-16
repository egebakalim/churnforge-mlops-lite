# src/train.py
from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import basic_normalize, ge_validate, load_raw, split_xy
from src.features import build_preprocessor

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"


def train() -> dict:
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

    df = basic_normalize(load_raw())

    # Data validation gate
    if not ge_validate(df):
        raise RuntimeError("Great Expectations validation failed. See great_expectations/validation_result.json")

    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    fb = build_preprocessor(X_train)

    clf = LogisticRegression(max_iter=1000)

    pipe = Pipeline(
        steps=[
            ("preprocess", fb.preprocessor),
            ("model", clf),
        ]
    )

    mlflow.set_experiment("churnforge-mlops-lite")

    with mlflow.start_run() as run:
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1", float(f1))
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("num_cols", len(fb.numeric_cols))
        mlflow.log_param("cat_cols", len(fb.categorical_cols))

        # Save model locally + log artifact
        import joblib
        joblib.dump(pipe, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH))

        result = {
            "run_id": run.info.run_id,
            "accuracy": float(acc),
            "f1": float(f1),
            "model_path": str(MODEL_PATH),
        }
        return result


def main():
    res = train()
    print("Training complete:")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("\nTip: run `mlflow ui` in this repo to view experiments.")


if __name__ == "__main__":
    main()
