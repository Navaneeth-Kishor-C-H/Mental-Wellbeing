"""Dataset loading and validation for the student wellbeing system."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import CATEGORICAL_FEATURES, FEATURES, LIFESTYLE_CSV, NUMERIC_FEATURES, TARGET


def load_lifestyle(path: Path = LIFESTYLE_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Student dataset not found: {path}")
    return pd.read_csv(path)


def clean_lifestyle(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the SRS lifestyle dataset without silently inventing target labels."""
    required = set(FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    cleaned = df.copy()
    if "Student_ID" in cleaned:
        cleaned = cleaned.drop_duplicates(subset=["Student_ID"])
    else:
        cleaned = cleaned.drop_duplicates()
    for column in NUMERIC_FEATURES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())
    for column in CATEGORICAL_FEATURES:
        cleaned[column] = cleaned[column].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    cleaned[TARGET] = cleaned[TARGET].astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
    )
    cleaned = cleaned.dropna(subset=[TARGET])
    cleaned[TARGET] = cleaned[TARGET].astype(int)
    return cleaned.reset_index(drop=True)


def clean_mental_health(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible cleaner for the supplementary mental-health data."""
    cleaned = df.copy().drop_duplicates(subset=["Student_ID"])
    for column in ["Gender", "Mood_Description"]:
        cleaned[column] = cleaned[column].fillna("Unknown")
    cleaned["Daily_Reflections"] = cleaned["Daily_Reflections"].fillna("")
    numeric = ["Age", "GPA", "Stress_Level", "Anxiety_Score", "Depression_Score", "Sleep_Hours", "Steps_Per_Day", "Sentiment_Score", "Mental_Health_Status"]
    for column in numeric:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=["Mental_Health_Status"])
    for column in numeric[:-1]:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())
    return cleaned.assign(Mental_Health_Status=cleaned["Mental_Health_Status"].astype(int))
