from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import INTERVENTION_CSV, PREVENTION_MODEL, REASON_MODEL, RANDOM_STATE
from .features import INTERVENTION_FEATURES, PREVENTION_TARGET, REASON_TARGET

CATEGORICAL_FEATURES = ["Gender", "Department", "Mood_Description"]
NUMERIC_FEATURES = [feature for feature in INTERVENTION_FEATURES if feature not in CATEGORICAL_FEATURES]


def _build_text_label_classifier() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _prepare_intervention_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_targets = {REASON_TARGET, PREVENTION_TARGET}
    missing_targets = required_targets - set(df.columns)
    if missing_targets:
        missing = ", ".join(sorted(missing_targets))
        raise ValueError(f"Intervention dataset is missing target column(s): {missing}")

    for feature in INTERVENTION_FEATURES:
        if feature not in df.columns:
            df[feature] = None

    df = df.dropna(subset=[REASON_TARGET, PREVENTION_TARGET])
    df[REASON_TARGET] = df[REASON_TARGET].astype(str)
    df[PREVENTION_TARGET] = df[PREVENTION_TARGET].astype(str)
    return df


def train_intervention_models_if_available(path: Path = INTERVENTION_CSV) -> dict:
    if not path.exists():
        return {
            "trained": False,
            "message": (
                "No supervised reason/prevention dataset found. Add data/raw/wellbeing_interventions.csv "
                "with Reason and Prevention_Method columns to train these models."
            ),
        }

    df = _prepare_intervention_data(path)
    if len(df) < 20:
        return {
            "trained": False,
            "message": "Intervention dataset needs at least 20 labeled rows for training.",
        }

    x = df[INTERVENTION_FEATURES]

    reason_model = _build_text_label_classifier()
    reason_model.fit(x, df[REASON_TARGET])
    joblib.dump(reason_model, REASON_MODEL)

    prevention_model = _build_text_label_classifier()
    prevention_model.fit(x, df[PREVENTION_TARGET])
    joblib.dump(prevention_model, PREVENTION_MODEL)

    return {
        "trained": True,
        "rows": int(len(df)),
        "reason_labels": int(df[REASON_TARGET].nunique()),
        "prevention_labels": int(df[PREVENTION_TARGET].nunique()),
    }


def predict_reason_and_prevention(payload: dict) -> dict:
    if not REASON_MODEL.exists() or not PREVENTION_MODEL.exists():
        return {
            "available": False,
            "reason": None,
            "prevention_method": None,
            "message": (
                "Reason and prevention predictions are not available because no labeled "
                "reason/prevention model has been trained."
            ),
        }

    row = pd.DataFrame([{feature: payload.get(feature) for feature in INTERVENTION_FEATURES}])
    reason_model = joblib.load(REASON_MODEL)
    prevention_model = joblib.load(PREVENTION_MODEL)

    return {
        "available": True,
        "reason": str(reason_model.predict(row)[0]),
        "prevention_method": str(prevention_model.predict(row)[0]),
        "message": "Predicted by supervised reason and prevention models.",
    }
