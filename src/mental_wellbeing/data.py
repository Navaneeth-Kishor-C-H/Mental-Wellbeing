from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import LIFESTYLE_CSV, MENTAL_HEALTH_CSV


def load_mental_health(path: Path = MENTAL_HEALTH_CSV) -> pd.DataFrame:
    """Load the mental health score dataset."""
    return pd.read_csv(path)


def load_lifestyle(path: Path = LIFESTYLE_CSV) -> pd.DataFrame:
    """Load the lifestyle and depression dataset."""
    return pd.read_csv(path)


def clean_mental_health(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates(subset=["Student_ID"])
    cleaned["Gender"] = cleaned["Gender"].fillna("Unknown")
    cleaned["Mood_Description"] = cleaned["Mood_Description"].fillna("Unknown")
    cleaned["Daily_Reflections"] = cleaned["Daily_Reflections"].fillna("")

    numeric_columns = [
        "Age",
        "GPA",
        "Stress_Level",
        "Anxiety_Score",
        "Depression_Score",
        "Sleep_Hours",
        "Steps_Per_Day",
        "Sentiment_Score",
        "Mental_Health_Status",
    ]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=["Mental_Health_Status"])
    for column in numeric_columns:
        if column != "Mental_Health_Status":
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    cleaned["Mental_Health_Status"] = cleaned["Mental_Health_Status"].astype(int)
    return cleaned


def clean_lifestyle(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates(subset=["Student_ID"])
    cleaned["Gender"] = cleaned["Gender"].fillna("Unknown")
    cleaned["Department"] = cleaned["Department"].fillna("Unknown")

    numeric_columns = [
        "Age",
        "CGPA",
        "Sleep_Duration",
        "Study_Hours",
        "Social_Media_Hours",
        "Physical_Activity",
        "Stress_Level",
    ]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    cleaned["Depression"] = cleaned["Depression"].astype(str).str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0}
    )
    cleaned = cleaned.dropna(subset=["Depression"])
    cleaned["Depression"] = cleaned["Depression"].astype(int)
    return cleaned
