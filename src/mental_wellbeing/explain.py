from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureProfile:
    mean: float
    low_risk_mean: float
    high_risk_mean: float
    std: float
    importance: float


HIGHER_IS_BETTER = {
    "GPA",
    "CGPA",
    "Sleep_Hours",
    "Sleep_Duration",
    "Steps_Per_Day",
    "Physical_Activity",
    "Sentiment_Score",
}

LOWER_IS_BETTER = {
    "Stress_Level",
    "Anxiety_Score",
    "Depression_Score",
    "Social_Media_Hours",
}

FRIENDLY_NAMES = {
    "GPA": "GPA",
    "CGPA": "CGPA",
    "Sleep_Hours": "sleep hours",
    "Sleep_Duration": "sleep duration",
    "Steps_Per_Day": "daily steps",
    "Physical_Activity": "physical activity",
    "Sentiment_Score": "reflection sentiment",
    "Stress_Level": "stress level",
    "Anxiety_Score": "anxiety score",
    "Depression_Score": "depression score",
    "Social_Media_Hours": "social media time",
    "Study_Hours": "study hours",
    "Age": "age",
}


def _feature_importance_by_original_column(model, numeric_features: list[str], categorical_features: list[str]) -> dict[str, float]:
    classifier = model.named_steps["model"]
    importances = classifier.feature_importances_
    preprocessor = model.named_steps["preprocessor"]
    transformed_names = preprocessor.get_feature_names_out()

    scores = {feature: 0.0 for feature in numeric_features + categorical_features}
    for transformed_name, importance in zip(transformed_names, importances):
        original = transformed_name.split("__", 1)[-1]
        for feature in scores:
            if original == feature or original.startswith(f"{feature}_"):
                scores[feature] += float(importance)
                break
    return scores


def build_profile(
    df: pd.DataFrame,
    target: str,
    low_risk_value: int,
    high_risk_values: set[int],
    numeric_features: list[str],
    categorical_features: list[str],
    model,
) -> dict:
    importances = _feature_importance_by_original_column(model, numeric_features, categorical_features)
    low_risk_df = df[df[target] == low_risk_value]
    high_risk_df = df[df[target].isin(high_risk_values)]

    feature_profiles = {}
    for feature in numeric_features:
        feature_profiles[feature] = {
            "mean": float(df[feature].mean()),
            "low_risk_mean": float(low_risk_df[feature].mean()),
            "high_risk_mean": float(high_risk_df[feature].mean()),
            "std": float(df[feature].std() or 1.0),
            "importance": float(importances.get(feature, 0.0)),
        }

    category_profiles = {}
    for feature in categorical_features:
        category_profiles[feature] = (
            low_risk_df[feature].astype(str).value_counts(normalize=True).head(5).to_dict()
        )

    return {
        "target": target,
        "low_risk_value": low_risk_value,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_profiles": feature_profiles,
        "category_profiles": category_profiles,
    }


def explain_prediction(payload: dict, profile: dict, predicted_risk: int) -> dict:
    reason_rows = []
    for feature, stats in profile["feature_profiles"].items():
        value = payload.get(feature)
        if value is None:
            continue
        value = float(value)
        low_risk_mean = float(stats["low_risk_mean"])
        std = max(float(stats["std"]), 0.01)
        importance = float(stats["importance"])

        if feature in HIGHER_IS_BETTER:
            risk_gap = max(0.0, low_risk_mean - value)
            direction = "below"
        elif feature in LOWER_IS_BETTER:
            risk_gap = max(0.0, value - low_risk_mean)
            direction = "above"
        else:
            risk_gap = abs(value - low_risk_mean)
            direction = "different from"

        impact = (risk_gap / std) * (importance + 0.01)
        if impact <= 0:
            continue

        name = FRIENDLY_NAMES.get(feature, feature.replace("_", " ").lower())
        reason_rows.append(
            {
                "feature": feature,
                "reason": f"{name} is {direction} the low-risk dataset pattern ({value:.2f} vs {low_risk_mean:.2f}).",
                "recommendation": f"Move {name} closer to the low-risk dataset average of {low_risk_mean:.2f}.",
                "impact": impact,
            }
        )

    reason_rows.sort(key=lambda row: row["impact"], reverse=True)
    selected = reason_rows[:4]

    if not selected and predicted_risk == profile["low_risk_value"]:
        return {
            "reasons": ["The input pattern is close to the low-risk group in the training data."],
            "recommendations": ["Maintain the current low-risk pattern and keep monitoring changes."],
        }

    return {
        "reasons": [row["reason"] for row in selected],
        "recommendations": [row["recommendation"] for row in selected],
    }
