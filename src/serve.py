# src/serve.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "artifacts" / "model.pkl"

app = FastAPI(title="churnforge-mlops-lite", version="0.1.0")

_model = None


def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run: make train"
            )
        import joblib
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_present": MODEL_PATH.exists()}


@app.post("/predict")
def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload should be a JSON dict representing a single row of features.
    """
    try:
        model = _load_model()
        X = pd.DataFrame([payload])
        pred = int(model.predict(X)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0][1])
        return {"churn_pred": pred, "churn_proba": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
