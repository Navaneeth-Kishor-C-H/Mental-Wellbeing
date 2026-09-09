from __future__ import annotations

import joblib
import pandas as pd

from .config import ARTIFACT_PATH
from .xai import lime_explanation, shap_explanation


DISPLAY_NAMES = {
    "CGPA": "CGPA",
    "Sleep_Duration": "sleep duration",
    "Study_Hours": "study hours",
    "Social_Media_Hours": "social-media time",
    "Physical_Activity": "physical activity",
    "Stress_Level": "stress level",
    "Age": "age",
    "Gender": "gender",
    "Department": "department",
    "GPA": "GPA",
    "Anxiety_Score": "anxiety score",
    "Depression_Score": "depression score",
    "Sleep_Hours": "sleep hours",
    "Steps_Per_Day": "daily steps",
    "Mood_Description": "mood",
    "Cluster": "student segment",
}


def _artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError("Trained system not found. Run `py run_train.py` first.")
    return joblib.load(ARTIFACT_PATH)


def _display_factor(transformed_name: str, payload: dict) -> str:
    """Turn a pipeline feature name into a readable, student-specific factor."""
    original = transformed_name.split("__", 1)[-1]
    if original in payload:
        return f"{DISPLAY_NAMES.get(original, original)} ({payload[original]})"
    for field in ("Gender", "Department"):
        prefix = f"{field}_"
        if original.startswith(prefix):
            return f"{DISPLAY_NAMES[field]} ({original.removeprefix(prefix)})"
    return DISPLAY_NAMES.get(original, original.replace("_", " "))


def _shap_summary(shap_values: pd.DataFrame, payload: dict, predicted_class: int, probability: float, outcome_name: str | None = None) -> tuple[str, list[str]]:
    """Create a concise explanation of the strongest local SHAP contributions."""
    outcome = outcome_name or ("at-risk" if predicted_class else "lower-risk")
    details = []
    for item in shap_values.head(3).itertuples(index=False):
        direction = "increased" if item.shap_value > 0 else "reduced"
        factor = _display_factor(item.feature, payload)
        details.append(f"{factor[:1].upper() + factor[1:]} {direction} the model's {outcome} score.")
    summary = f"Predicted {outcome} probability: {probability:.1%}. " + " ".join(details)
    summary += " These are model-based associations, not a clinical diagnosis or proof that any factor caused depression."
    return summary, details


def predict_student(payload: dict, include_explanations: bool = True) -> dict:
    artifact = _artifact()
    row = pd.DataFrame([{feature: payload.get(feature) for feature in artifact["features"]}])
    row["Cluster"] = artifact["clusterer"].predict(row[artifact["cluster_features"]])
    model = artifact["model"]
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    class_index = classes.index(prediction)
    transformed = model.named_steps["preprocessor"].transform(row)
    transformed = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    output = {"prediction": prediction, "label": "At-risk of depression" if prediction else "Lower depression risk", "cluster": int(row["Cluster"].iloc[0]), "cluster_algorithm": artifact["clusterer"].algorithm, "probabilities": {str(label): round(float(value), 4) for label, value in zip(classes, probabilities)}, "model": artifact["best_model_name"]}
    if include_explanations:
        shap_values = shap_explanation(model, transformed, artifact["transformed_feature_names"], class_index)
        lime_values = lime_explanation(model, artifact["training_reference"], transformed, artifact["transformed_feature_names"], class_index)
        output["shap"] = shap_values.head(10).round(5).to_dict(orient="records")
        output["lime"] = lime_values.head(10).round(5).to_dict(orient="records")
        summary, details = _shap_summary(shap_values, payload, prediction, float(probabilities[class_index]))
        output["reason"] = summary
        output["shap_details"] = details
    return output


# Kept as aliases so existing integrations do not break.
predict_depression_risk = predict_student
