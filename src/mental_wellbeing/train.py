from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from .config import (
    DEPRESSION_RISK_MODEL,
    DEPRESSION_RISK_PROFILE,
    MENTAL_STATUS_MODEL,
    MENTAL_STATUS_PROFILE,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)
from .data import clean_lifestyle, clean_mental_health, load_lifestyle, load_mental_health
from .explain import build_profile
from .features import (
    DEPRESSION_TARGET,
    LIFESTYLE_FEATURES,
    MENTAL_STATUS_FEATURES,
    MENTAL_STATUS_TARGET,
    build_depression_risk_pipeline,
    build_mental_status_pipeline,
)


def _train_and_save(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    pipeline,
    model_path: Path,
) -> dict:
    x_train, x_test, y_train, y_test = train_test_split(
        df[features],
        df[target],
        test_size=0.2,
        stratify=df[target],
        random_state=RANDOM_STATE,
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    return {
        "model_path": str(model_path),
        "rows": int(len(df)),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "weighted_f1": round(float(f1_score(y_test, predictions, average="weighted")), 4),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }


def _save_profile(
    df: pd.DataFrame,
    target: str,
    low_risk_value: int,
    high_risk_values: set[int],
    numeric_features: list[str],
    categorical_features: list[str],
    model_path: Path,
    profile_path: Path,
) -> None:
    model = joblib.load(model_path)
    profile = build_profile(
        df=df,
        target=target,
        low_risk_value=low_risk_value,
        high_risk_values=high_risk_values,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model=model,
    )
    joblib.dump(profile, profile_path)


def train_all() -> dict:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    mental_df = clean_mental_health(load_mental_health())
    lifestyle_df = clean_lifestyle(load_lifestyle())

    mental_df.to_csv(PROCESSED_DATA_DIR / "mental_health_clean.csv", index=False)
    lifestyle_df.to_csv(PROCESSED_DATA_DIR / "student_lifestyle_clean.csv", index=False)

    results = {
        "mental_status": _train_and_save(
            mental_df,
            MENTAL_STATUS_FEATURES,
            MENTAL_STATUS_TARGET,
            build_mental_status_pipeline(),
            MENTAL_STATUS_MODEL,
        ),
        "depression_risk": _train_and_save(
            lifestyle_df,
            LIFESTYLE_FEATURES,
            DEPRESSION_TARGET,
            build_depression_risk_pipeline(),
            DEPRESSION_RISK_MODEL,
        ),
    }

    _save_profile(
        mental_df,
        MENTAL_STATUS_TARGET,
        low_risk_value=0,
        high_risk_values={1, 2},
        numeric_features=[
            "Age",
            "GPA",
            "Stress_Level",
            "Anxiety_Score",
            "Depression_Score",
            "Sleep_Hours",
            "Steps_Per_Day",
            "Sentiment_Score",
        ],
        categorical_features=["Gender", "Mood_Description"],
        model_path=MENTAL_STATUS_MODEL,
        profile_path=MENTAL_STATUS_PROFILE,
    )
    _save_profile(
        lifestyle_df,
        DEPRESSION_TARGET,
        low_risk_value=0,
        high_risk_values={1},
        numeric_features=[
            "Age",
            "CGPA",
            "Sleep_Duration",
            "Study_Hours",
            "Social_Media_Hours",
            "Physical_Activity",
            "Stress_Level",
        ],
        categorical_features=["Gender", "Department"],
        model_path=DEPRESSION_RISK_MODEL,
        profile_path=DEPRESSION_RISK_PROFILE,
    )

    report_path = PROCESSED_DATA_DIR / "training_metrics.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    results = train_all()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
