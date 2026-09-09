"""Independent training and prediction workflow for the supplementary dataset."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MENTAL_HEALTH_ARTIFACT_PATH, MENTAL_HEALTH_CATEGORICAL_FEATURES, MENTAL_HEALTH_CSV, MENTAL_HEALTH_FEATURES, MENTAL_HEALTH_METRICS_PATH, MENTAL_HEALTH_NUMERIC_FEATURES, MENTAL_HEALTH_TARGET, MENTAL_HEALTH_TEXT_FEATURE, MODEL_DIR, PROCESSED_DATA_DIR, RANDOM_STATE
from .data import clean_mental_health, load_lifestyle
from .predict import _shap_summary
from .xai import lime_explanation, shap_explanation


MENTAL_HEALTH_LABELS = {0: "Healthy", 1: "At-risk", 2: "Struggling"}


def _flatten_text(values):
    return np.asarray(values).ravel()


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), MENTAL_HEALTH_NUMERIC_FEATURES),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), MENTAL_HEALTH_CATEGORICAL_FEATURES),
            ("reflections", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="")), ("flatten", FunctionTransformer(_flatten_text, validate=False, feature_names_out="one-to-one")), ("tfidf", TfidfVectorizer(max_features=150, stop_words="english"))]), [MENTAL_HEALTH_TEXT_FEATURE]),
        ]
    )


def classifier_candidates() -> dict[str, Pipeline]:
    def pipeline(model):
        return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])

    return {
        "Random Forest": pipeline(RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
        "Extra Trees": pipeline(ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    }


def train_mental_health() -> dict:
    """Train only on mental_health_dataset.csv; it is never merged with lifestyle data."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = clean_mental_health(load_lifestyle(MENTAL_HEALTH_CSV))
    x_train, x_test, y_train, y_test = train_test_split(data[MENTAL_HEALTH_FEATURES], data[MENTAL_HEALTH_TARGET], test_size=0.2, stratify=data[MENTAL_HEALTH_TARGET], random_state=RANDOM_STATE)
    results, fitted = [], {}
    for name, model in classifier_candidates().items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        results.append({"model": name, "accuracy": round(float(accuracy_score(y_test, prediction)), 4), "macro_precision": round(float(precision_score(y_test, prediction, average="macro", zero_division=0)), 4), "macro_recall": round(float(recall_score(y_test, prediction, average="macro", zero_division=0)), 4), "macro_f1": round(float(f1_score(y_test, prediction, average="macro", zero_division=0)), 4)})
        fitted[name] = model
    results_df = pd.DataFrame(results).sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)
    best_name = str(results_df.iloc[0]["model"])
    best_model = fitted[best_name]
    reference = best_model.named_steps["preprocessor"].transform(x_train.iloc[: min(300, len(x_train))])
    reference = reference.toarray() if hasattr(reference, "toarray") else reference
    artifact = {"model": best_model, "features": MENTAL_HEALTH_FEATURES, "training_reference": reference, "transformed_feature_names": list(best_model.named_steps["preprocessor"].get_feature_names_out()), "best_model_name": best_name, "target": MENTAL_HEALTH_TARGET}
    joblib.dump(artifact, MENTAL_HEALTH_ARTIFACT_PATH)
    metrics = {"generated_at": datetime.now(timezone.utc).isoformat(), "dataset_rows": int(len(data)), "target": MENTAL_HEALTH_TARGET, "target_classes": sorted(int(value) for value in data[MENTAL_HEALTH_TARGET].unique()), "prediction_models": results_df.to_dict(orient="records"), "selected_prediction_model": best_name, "classification_report": classification_report(y_test, best_model.predict(x_test), output_dict=True)}
    MENTAL_HEALTH_METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    return metrics


def predict_mental_health(payload: dict, include_explanations: bool = True) -> dict:
    if not MENTAL_HEALTH_ARTIFACT_PATH.exists():
        raise FileNotFoundError("Mental-health model not found. Run `py run_train_mental_health.py` first.")
    artifact = joblib.load(MENTAL_HEALTH_ARTIFACT_PATH)
    row = pd.DataFrame([{feature: payload.get(feature) for feature in artifact["features"]}])
    model = artifact["model"]
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    class_index = classes.index(prediction)
    output = {"system": "mental_health", "prediction": prediction, "label": f"Mental health status: {MENTAL_HEALTH_LABELS.get(prediction, f'Class {prediction}')}", "status_label": MENTAL_HEALTH_LABELS.get(prediction, f"Class {prediction}"), "class_labels": {str(label): MENTAL_HEALTH_LABELS.get(int(label), f"Class {label}") for label in classes}, "probabilities": {str(label): round(float(value), 4) for label, value in zip(classes, probabilities)}, "model": artifact["best_model_name"]}
    if include_explanations:
        transformed = model.named_steps["preprocessor"].transform(row)
        transformed = transformed.toarray() if hasattr(transformed, "toarray") else transformed
        shap_values = shap_explanation(model, transformed, artifact["transformed_feature_names"], class_index)
        lime_values = lime_explanation(model, artifact["training_reference"], transformed, artifact["transformed_feature_names"], class_index)
        output["shap"] = shap_values.head(10).round(5).to_dict(orient="records")
        output["lime"] = lime_values.head(10).round(5).to_dict(orient="records")
        summary, details = _shap_summary(shap_values, payload, prediction, float(probabilities[class_index]), outcome_name=MENTAL_HEALTH_LABELS.get(prediction, f"class {prediction}"))
        output["reason"] = summary
        output["shap_details"] = details
    return output
