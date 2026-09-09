from __future__ import annotations

import pandas as pd


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

NON_ACTIONABLE_FEATURES = {"Age"}


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


def _class_probability(model, row: pd.DataFrame, class_value: int) -> float:
    if not hasattr(model, "predict_proba"):
        return 0.0
    classes = list(model.classes_)
    if class_value not in classes:
        return 0.0
    class_index = classes.index(class_value)
    return float(model.predict_proba(row)[0][class_index])


def _toward_low_risk_value(feature: str, current_value: float, low_risk_value: float) -> float:
    if feature in HIGHER_IS_BETTER:
        return max(current_value, low_risk_value)
    if feature in LOWER_IS_BETTER:
        return min(current_value, low_risk_value)
    return low_risk_value


def _driver_reason(feature: str, risk_probability: float, improved_probability: float) -> str:
    name = FRIENDLY_NAMES.get(feature, feature.replace("_", " ").lower())
    change = max(0.0, risk_probability - improved_probability)
    return (
        f"The trained model found {name} to be an important risk driver for this input. "
        f"When only this factor is shifted toward patterns learned from low-risk students, "
        f"the model-estimated risk drops by {change:.1%}."
    )


def _model_action(feature: str, current_value: float, tested_value: float) -> str:
    name = FRIENDLY_NAMES.get(feature, feature.replace("_", " ").lower())
    if tested_value > current_value:
        direction = "increase"
    elif tested_value < current_value:
        direction = "reduce"
    else:
        direction = "stabilize"
    return (
        f"Model-derived prevention focus: {direction} {name}. "
        "This is selected because the trained model predicts lower risk under that changed input pattern."
    )


def explain_prediction(model, payload: dict, profile: dict, predicted_risk: int, features: list[str]) -> dict:
    row = pd.DataFrame([{feature: payload.get(feature) for feature in features}])
    low_risk_value = int(profile["low_risk_value"])
    risk_class = predicted_risk
    if predicted_risk == low_risk_value:
        risk_class = max((int(cls) for cls in model.classes_), default=predicted_risk)

    base_risk_probability = _class_probability(model, row, risk_class)
    counterfactuals = []

    for feature, stats in profile["feature_profiles"].items():
        if feature in NON_ACTIONABLE_FEATURES or feature not in payload:
            continue

        current_value = payload.get(feature)
        if current_value is None:
            continue

        current_value = float(current_value)
        tested_value = _toward_low_risk_value(feature, current_value, float(stats["low_risk_mean"]))
        if tested_value == current_value:
            continue

        changed_payload = dict(payload)
        changed_payload[feature] = tested_value
        changed_row = pd.DataFrame([{column: changed_payload.get(column) for column in features}])
        changed_risk_probability = _class_probability(model, changed_row, risk_class)
        improvement = base_risk_probability - changed_risk_probability
        if improvement <= 0:
            continue

        counterfactuals.append(
            {
                "feature": feature,
                "current_value": current_value,
                "tested_value": tested_value,
                "importance": float(stats["importance"]),
                "improvement": improvement,
                "changed_risk_probability": changed_risk_probability,
            }
        )

    counterfactuals.sort(key=lambda item: (item["improvement"], item["importance"]), reverse=True)
    selected = counterfactuals[:4]

    if not selected:
        return {
            "reasons": [
                "The model did not find a single modifiable factor that clearly reduced the predicted risk when tested alone."
            ],
            "recommendations": [
                "No model-derived prevention focus was found from single-factor counterfactual testing."
            ],
        }

    return {
        "reasons": [
            _driver_reason(item["feature"], base_risk_probability, item["changed_risk_probability"])
            for item in selected
        ],
        "recommendations": [
            _model_action(item["feature"], item["current_value"], item["tested_value"])
            for item in selected
        ],
    }
