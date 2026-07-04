import pandas as pd

from mental_wellbeing.data import clean_lifestyle, clean_mental_health


def test_clean_lifestyle_converts_depression_to_integer():
    df = pd.DataFrame(
        {
            "Student_ID": [1, 1, 2],
            "Age": [20, 20, None],
            "Gender": ["Female", "Female", None],
            "Department": ["Science", "Science", None],
            "CGPA": [3.2, 3.2, 2.8],
            "Sleep_Duration": [7, 7, 5],
            "Study_Hours": [4, 4, 8],
            "Social_Media_Hours": [2, 2, 5],
            "Physical_Activity": [60, 60, 10],
            "Stress_Level": [3, 3, 8],
            "Depression": ["False", "False", "True"],
        }
    )

    cleaned = clean_lifestyle(df)

    assert len(cleaned) == 2
    assert cleaned["Depression"].tolist() == [0, 1]


def test_clean_mental_health_keeps_required_target():
    df = pd.DataFrame(
        {
            "Student_ID": [1, 2],
            "Age": [21, None],
            "Gender": ["Male", None],
            "GPA": [3.1, 2.5],
            "Stress_Level": [4, 7],
            "Anxiety_Score": [8, 12],
            "Depression_Score": [6, 18],
            "Daily_Reflections": [None, "Rough day"],
            "Sleep_Hours": [7.5, None],
            "Steps_Per_Day": [8000, 3000],
            "Mood_Description": ["Happy", None],
            "Sentiment_Score": [0.4, -0.2],
            "Mental_Health_Status": [0, 2],
        }
    )

    cleaned = clean_mental_health(df)

    assert cleaned["Mental_Health_Status"].dtype.kind in {"i", "u"}
    assert cleaned.isna().sum().sum() == 0
