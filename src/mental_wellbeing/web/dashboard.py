from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from mental_wellbeing.config import PROCESSED_DATA_DIR
from mental_wellbeing.data import clean_lifestyle, clean_mental_health, load_lifestyle, load_mental_health
from mental_wellbeing.predict import predict_depression_risk, predict_mental_status

st.set_page_config(page_title="Student Mental Wellbeing", layout="wide")


def show_prediction_result(result: dict) -> None:
    st.metric("Prediction", result["label"])
    if result.get("probabilities"):
        probabilities = pd.DataFrame(
            {
                "Class": list(result["probabilities"].keys()),
                "Probability": list(result["probabilities"].values()),
            }
        )
        st.plotly_chart(px.bar(probabilities, x="Class", y="Probability"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Reasons")
        for reason in result.get("reasons", []):
            st.write(f"- {reason}")
    with col2:
        st.subheader("Data-Driven Recommendations")
        for recommendation in result.get("recommendations", []):
            st.write(f"- {recommendation}")


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return clean_mental_health(load_mental_health()), clean_lifestyle(load_lifestyle())


def show_overview(mental_df: pd.DataFrame, lifestyle_df: pd.DataFrame) -> None:
    st.title("Student Mental Wellbeing")
    left, middle, right = st.columns(3)
    left.metric("Mental health records", f"{len(mental_df):,}")
    middle.metric("Lifestyle records", f"{len(lifestyle_df):,}")
    right.metric("Average stress", f"{lifestyle_df['Stress_Level'].mean():.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            px.histogram(lifestyle_df, x="Sleep_Duration", color="Depression", barmode="overlay"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            px.box(mental_df, x="Mental_Health_Status", y="Stress_Level", color="Mental_Health_Status"),
            use_container_width=True,
        )

    st.subheader("Sample data")
    st.dataframe(lifestyle_df.head(100), use_container_width=True)


def show_mental_prediction() -> None:
    st.subheader("Predict Mental Health Status")
    with st.form("mental_status_form"):
        col1, col2, col3 = st.columns(3)
        payload = {
            "Age": col1.number_input("Age", 15, 40, 21),
            "Gender": col2.selectbox("Gender", ["Female", "Male", "Other"]),
            "GPA": col3.number_input("GPA", 0.0, 4.0, 3.0, step=0.1),
            "Stress_Level": col1.slider("Stress level", 1, 10, 5),
            "Anxiety_Score": col2.slider("Anxiety score", 0, 30, 10),
            "Depression_Score": col3.slider("Depression score", 0, 30, 10),
            "Sleep_Hours": col1.number_input("Sleep hours", 0.0, 12.0, 7.0, step=0.5),
            "Steps_Per_Day": col2.number_input("Steps per day", 0, 30000, 7000, step=500),
            "Mood_Description": col3.selectbox("Mood", ["Happy", "Sad", "Tired", "Neutral", "Anxious"]),
            "Sentiment_Score": col1.slider("Sentiment score", -1.0, 1.0, 0.2),
        }
        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            show_prediction_result(predict_mental_status(payload))
        except FileNotFoundError as exc:
            st.error(str(exc))


def show_depression_prediction() -> None:
    st.subheader("Predict Depression Risk")
    with st.form("depression_risk_form"):
        col1, col2, col3 = st.columns(3)
        payload = {
            "Age": col1.number_input("Student age", 15, 40, 21),
            "Gender": col2.selectbox("Student gender", ["Female", "Male", "Other"]),
            "Department": col3.selectbox("Department", ["Engineering", "Science", "Medical", "Arts", "Commerce"]),
            "CGPA": col1.number_input("CGPA", 0.0, 4.0, 3.1, step=0.1),
            "Sleep_Duration": col2.number_input("Sleep duration", 0.0, 12.0, 6.5, step=0.5),
            "Study_Hours": col3.number_input("Study hours", 0.0, 16.0, 5.0, step=0.5),
            "Social_Media_Hours": col1.number_input("Social media hours", 0.0, 16.0, 3.0, step=0.5),
            "Physical_Activity": col2.number_input("Physical activity minutes", 0, 240, 45, step=5),
            "Stress_Level": col3.slider("Lifestyle stress level", 1, 10, 5),
        }
        submitted = st.form_submit_button("Predict risk")

    if submitted:
        try:
            show_prediction_result(predict_depression_risk(payload))
        except FileNotFoundError as exc:
            st.error(str(exc))


def show_metrics() -> None:
    metrics_path = PROCESSED_DATA_DIR / "training_metrics.json"
    st.subheader("Model Metrics")
    if metrics_path.exists():
        st.json(json.loads(metrics_path.read_text(encoding="utf-8")))
    else:
        st.info("Train the models first to generate metrics.")


def main() -> None:
    mental_df, lifestyle_df = load_data()
    page = st.sidebar.radio(
        "View",
        ["Overview", "Mental Status Prediction", "Depression Risk Prediction", "Model Metrics"],
    )
    if page == "Overview":
        show_overview(mental_df, lifestyle_df)
    elif page == "Mental Status Prediction":
        show_mental_prediction()
    elif page == "Depression Risk Prediction":
        show_depression_prediction()
    else:
        show_metrics()


if __name__ == "__main__":
    main()
