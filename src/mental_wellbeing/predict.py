from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .config import (
    DEPRESSION_RISK_MODEL,
    DEPRESSION_RISK_PROFILE,
    MENTAL_STATUS_MODEL,
    MENTAL_STATUS_PROFILE,
)
from .explain import explain_prediction
from .features import LIFESTYLE_FEATURES, MENTAL_STATUS_FEATURES

MENTAL_STATUS_LABELS = {
    0: "Low risk",
    1: "Moderate risk",
    2: "High risk",
}


def _load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python -m mental_wellbeing.train` first."
        )
    return joblib.load(path)


def _load_profile(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Explanation profile not found at {path}. Run `py run_train.py` again."
        )
    return joblib.load(path)


def _predict_single(model_path: Path, profile_path: Path, payload: dict, features: list[str]) -> dict:
    model = _load_model(model_path)
    profile = _load_profile(profile_path)
    row = pd.DataFrame([{feature: payload.get(feature) for feature in features}])
    prediction = int(model.predict(row)[0])
    probabilities = {}
    if hasattr(model, "predict_proba"):
        classes = model.classes_
        values = model.predict_proba(row)[0]
        probabilities = {str(cls): round(float(prob), 4) for cls, prob in zip(classes, values)}
    explanation = explain_prediction(payload, profile, prediction)
    return {
        "prediction": prediction,
        "probabilities": probabilities,
        "reasons": explanation["reasons"],
        "recommendations": explanation["recommendations"],
    }


def predict_mental_status(payload: dict) -> dict:
    result = _predict_single(MENTAL_STATUS_MODEL, MENTAL_STATUS_PROFILE, payload, MENTAL_STATUS_FEATURES)
    result["label"] = MENTAL_STATUS_LABELS.get(result["prediction"], "Unknown")
    return result


def predict_depression_risk(payload: dict) -> dict:
    result = _predict_single(DEPRESSION_RISK_MODEL, DEPRESSION_RISK_PROFILE, payload, LIFESTYLE_FEATURES)
    result["label"] = "Depression risk" if result["prediction"] == 1 else "No depression risk"
    return result
