from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE

MENTAL_STATUS_FEATURES = [
    "Age",
    "Gender",
    "GPA",
    "Stress_Level",
    "Anxiety_Score",
    "Depression_Score",
    "Sleep_Hours",
    "Steps_Per_Day",
    "Mood_Description",
    "Sentiment_Score",
]

LIFESTYLE_FEATURES = [
    "Age",
    "Gender",
    "Department",
    "CGPA",
    "Sleep_Duration",
    "Study_Hours",
    "Social_Media_Hours",
    "Physical_Activity",
    "Stress_Level",
]

MENTAL_STATUS_TARGET = "Mental_Health_Status"
DEPRESSION_TARGET = "Depression"


def build_classifier(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
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
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=180,
                    max_depth=12,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_mental_status_pipeline() -> Pipeline:
    return build_classifier(
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
    )


def build_depression_risk_pipeline() -> Pipeline:
    return build_classifier(
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
    )
