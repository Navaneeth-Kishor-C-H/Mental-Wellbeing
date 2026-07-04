from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MENTAL_HEALTH_CSV = RAW_DATA_DIR / "mental_health_dataset.csv"
LIFESTYLE_CSV = RAW_DATA_DIR / "student_lifestyle_100k.csv"

MENTAL_STATUS_MODEL = MODEL_DIR / "mental_status_model.joblib"
DEPRESSION_RISK_MODEL = MODEL_DIR / "depression_risk_model.joblib"
MENTAL_STATUS_PROFILE = MODEL_DIR / "mental_status_profile.joblib"
DEPRESSION_RISK_PROFILE = MODEL_DIR / "depression_risk_profile.joblib"

RANDOM_STATE = 42
