from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
DATABASE_PATH = PROJECT_ROOT / "data" / "wellbeing_portal.db"

LIFESTYLE_CSV = RAW_DATA_DIR / "student_lifestyle_100k.csv"
STUDENT_DIRECTORY_CSV = RAW_DATA_DIR / "student_details_dummy_100k.csv"
MENTAL_HEALTH_CSV = RAW_DATA_DIR / "mental_health_dataset.csv"
ARTIFACT_PATH = MODEL_DIR / "wellbeing_system.joblib"
METRICS_PATH = PROCESSED_DATA_DIR / "training_metrics.json"
CLUSTERED_DATA_PATH = PROCESSED_DATA_DIR / "student_lifestyle_clustered.csv"
MENTAL_HEALTH_ARTIFACT_PATH = MODEL_DIR / "mental_health_system.joblib"
MENTAL_HEALTH_METRICS_PATH = PROCESSED_DATA_DIR / "mental_health_training_metrics.json"
RANDOM_STATE = 42

FEATURES = ["Age", "Gender", "Department", "CGPA", "Sleep_Duration", "Study_Hours", "Social_Media_Hours", "Physical_Activity", "Stress_Level"]
NUMERIC_FEATURES = ["Age", "CGPA", "Sleep_Duration", "Study_Hours", "Social_Media_Hours", "Physical_Activity", "Stress_Level"]
CATEGORICAL_FEATURES = ["Gender", "Department"]
TARGET = "Depression"
CLUSTER_FEATURES = NUMERIC_FEATURES

MENTAL_HEALTH_FEATURES = ["Age", "Gender", "GPA", "Stress_Level", "Anxiety_Score", "Depression_Score", "Daily_Reflections", "Sleep_Hours", "Steps_Per_Day", "Mood_Description", "Sentiment_Score"]
MENTAL_HEALTH_NUMERIC_FEATURES = ["Age", "GPA", "Stress_Level", "Anxiety_Score", "Depression_Score", "Sleep_Hours", "Steps_Per_Day", "Sentiment_Score"]
MENTAL_HEALTH_CATEGORICAL_FEATURES = ["Gender", "Mood_Description"]
MENTAL_HEALTH_TEXT_FEATURE = "Daily_Reflections"
MENTAL_HEALTH_TARGET = "Mental_Health_Status"
