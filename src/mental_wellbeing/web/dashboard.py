from __future__ import annotations

import json
import pandas as pd
import plotly.express as px
import streamlit as st

from mental_wellbeing.config import CLUSTERED_DATA_PATH, MENTAL_HEALTH_METRICS_PATH, METRICS_PATH
from mental_wellbeing.data import clean_lifestyle, load_lifestyle
from mental_wellbeing.mental_health import predict_mental_health
from mental_wellbeing.predict import predict_student
from mental_wellbeing.storage import admin_exists, authenticate, change_password, create_admin, create_counsellor, initialise_database, recent_students, save_assessment, student_history

st.set_page_config(page_title="Student Wellbeing Intelligence", page_icon="🧠", layout="wide")
initialise_database()

@st.cache_data
def dataset() -> pd.DataFrame:
    return clean_lifestyle(load_lifestyle())

def login() -> None:
    st.title("🧠 Student Wellbeing Intelligence")
    st.caption("College wellbeing decision support — not a clinical diagnosis.")
    if not admin_exists():
        st.subheader("First-time setup: create the administrator account")
        with st.form("admin_setup"):
            username = st.text_input("Admin username")
            password = st.text_input("Create admin password (at least 8 characters)", type="password")
            confirm = st.text_input("Confirm admin password", type="password")
            setup = st.form_submit_button("Create administrator account")
        if setup:
            if password != confirm:
                st.error("The passwords do not match.")
            else:
                try:
                    create_admin(username, password)
                    st.success("Administrator account created. You can now sign in as Admin.")
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
        return
    # This selector must stay outside the form so its choice updates the fields
    # and instructions immediately, before the user submits credentials.
    account_type = st.radio("I am logging in as", ["Counsellor", "Admin"], horizontal=True)
    if account_type == "Counsellor":
        st.caption("Counsellor login: use the email address and temporary password provided by your administrator.")
    else:
        st.caption("Admin login: sign in to create counsellor accounts. Administrators cannot access student records.")
    with st.form("login"):
        if account_type == "Counsellor":
            username = st.text_input("Counsellor email address")
        else:
            username = st.text_input("Admin username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        expected_role = "counsellor" if account_type == "Counsellor" else "admin"
        user = authenticate(username, password, expected_role)
        if user:
            st.session_state.user = user
            st.rerun()
        st.error("Incorrect username or password.")

def render_result(result: dict) -> None:
    st.metric("Prediction", result["label"])
    if result.get("system") == "mental_health":
        st.caption(f"Supplementary mental-health dataset · {result['model']}")
        labels = result.get("class_labels", {})
        probabilities = pd.DataFrame({"Outcome": [labels.get(label, f"Status class {label}") for label in result["probabilities"]], "Probability": list(result["probabilities"].values())})
    else:
        st.caption(f"Student segment {result['cluster']} · {result['cluster_algorithm']} · {result['model']}")
        probabilities = pd.DataFrame({"Outcome": ["Lower risk", "At-risk"], "Probability": [result["probabilities"].get("0", 0), result["probabilities"].get("1", 0)]})
    st.plotly_chart(px.bar(probabilities, x="Outcome", y="Probability", range_y=[0, 1]), use_container_width=True)
    st.subheader("Model explanation")
    st.info(result["reason"])
    if result.get("shap_details"):
        st.markdown("**Main SHAP factors**")
        for detail in result["shap_details"]:
            st.write(f"- {detail}")
    one, two = st.columns(2)
    one.plotly_chart(px.bar(pd.DataFrame(result["shap"]), x="shap_value", y="feature", orientation="h", color="shap_value", title="SHAP contributions"), use_container_width=True)
    two.dataframe(pd.DataFrame(result["lime"]), use_container_width=True, hide_index=True)

def assessment() -> None:
    st.title("Student Assessment")
    selected_system = st.radio("Choose prediction dataset", ["Student lifestyle depression model", "Supplementary mental-health status model"], horizontal=True)
    if selected_system.startswith("Supplementary"):
        st.caption("This is a separate three-class model trained only on the supplementary mental-health dataset. Its status classes are dataset labels, not clinical diagnoses.")
    with st.form("assessment"):
        st.subheader("Student record")
        a, b, c = st.columns(3)
        student_code = a.text_input("Student ID *")
        name = b.text_input("Student name *")
        programme = c.text_input("Programme / course")
        st.subheader("Wellbeing inputs")
        if selected_system.startswith("Student lifestyle"):
            payload = {"Age": a.number_input("Age", 15, 50, 21), "Gender": b.selectbox("Gender", ["Female", "Male", "Other"]), "Department": c.selectbox("Department", ["Engineering", "Science", "Medical", "Arts", "Commerce"]), "CGPA": a.number_input("CGPA", 0.0, 4.0, 3.0, 0.1), "Sleep_Duration": b.number_input("Sleep duration", 0.0, 16.0, 7.0, 0.5), "Study_Hours": c.number_input("Study hours", 0.0, 18.0, 5.0, 0.5), "Social_Media_Hours": a.number_input("Social media hours", 0.0, 18.0, 3.0, 0.5), "Physical_Activity": b.number_input("Physical activity (minutes)", 0, 300, 60, 5), "Stress_Level": c.slider("Stress level", 1, 10, 5)}
        else:
            payload = {"Age": a.number_input("Age", 15, 80, 21), "Gender": b.selectbox("Gender", ["Female", "Male", "Other"]), "GPA": c.number_input("GPA", 0.0, 4.0, 3.0, 0.1), "Stress_Level": a.slider("Stress level", 1, 5, 3), "Anxiety_Score": b.number_input("Anxiety score", 0, 30, 10), "Depression_Score": c.number_input("Depression score", 0, 30, 10), "Sleep_Hours": a.number_input("Sleep hours", 0.0, 16.0, 7.0, 0.1), "Steps_Per_Day": b.number_input("Steps per day", 0, 30000, 5000, 100), "Mood_Description": c.text_input("Mood description", "Neutral"), "Sentiment_Score": a.number_input("Sentiment score", -1.0, 1.0, 0.0, 0.01), "Daily_Reflections": st.text_area("Daily reflections", "")}
        submitted = st.form_submit_button("Predict and save assessment")
    if submitted:
        try:
            result = predict_student(payload) if selected_system.startswith("Student lifestyle") else predict_mental_health(payload)
            save_assessment(student_code, name, programme, payload, result, st.session_state.user["id"])
            st.success("Assessment saved to the student history.")
            render_result(result)
        except (ValueError, FileNotFoundError, ImportError) as error:
            st.error(str(error))

def history() -> None:
    st.title("Student History")
    students = recent_students()
    if not students:
        st.info("No student assessments have been saved yet.")
        return
    labels = {f"{item['student_id']} — {item['full_name']}": item['student_id'] for item in students}
    selected = st.selectbox("Find a saved student", list(labels))
    records = student_history(labels[selected])
    st.caption(f"{records[0]['full_name']} · {records[0]['programme'] or 'Programme not recorded'}")
    for index, record in enumerate(records, start=1):
        result = record["result"]
        with st.expander(f"Assessment {index}: {record['created_at'][:19].replace('T', ' ')} · {result['label']}", expanded=index == 1):
            st.write(f"Saved by: {record['counsellor']}")
            st.json({"input": record["input"], "prediction": {key: value for key, value in result.items() if key not in {"shap", "lime"}}})
            if "shap" in result:
                st.dataframe(pd.DataFrame(result["shap"]), hide_index=True, use_container_width=True)

def overview() -> None:
    data = dataset()
    st.title("Wellbeing Overview")
    cols = st.columns(4)
    cols[0].metric("Students in dataset", f"{len(data):,}")
    cols[1].metric("Depression rate", f"{data.Depression.mean():.1%}")
    cols[2].metric("Average sleep", f"{data.Sleep_Duration.mean():.1f} h")
    cols[3].metric("Saved student records", len(recent_students()))
    st.plotly_chart(px.histogram(data, x="Sleep_Duration", color="Depression", barmode="overlay", title="Sleep duration by outcome"), use_container_width=True)

def clustering() -> None:
    st.title("Clustering Analysis")
    if not METRICS_PATH.exists() or not CLUSTERED_DATA_PATH.exists():
        st.warning("Train the system first: `py run_train.py`"); return
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    selected = metrics["selected_clusterer"]
    st.success(f"Selected: {selected['algorithm']} with {selected['n_clusters']} clusters.")
    st.dataframe(pd.DataFrame(metrics["cluster_selection"]), hide_index=True, use_container_width=True)
    data = pd.read_csv(CLUSTERED_DATA_PATH)
    st.plotly_chart(px.scatter(data, x="Sleep_Duration", y="Stress_Level", color="Cluster", symbol="Depression"), use_container_width=True)


def _evaluation_panel(metrics: dict, title: str, metric_columns: list[tuple[str, str]]) -> None:
    st.subheader(title)
    selected = metrics.get("selected_prediction_model", "")
    st.caption(f"Selected model: {selected} · Test results are from a fixed stratified holdout set.")
    models = pd.DataFrame(metrics.get("prediction_models", []))
    if models.empty:
        st.info("No model evaluation results are available.")
        return
    cards = st.columns(len(metric_columns))
    selected_row = models[models["model"] == selected]
    selected_row = selected_row.iloc[0] if not selected_row.empty else models.iloc[0]
    for card, (column, label) in zip(cards, metric_columns):
        card.metric(label, f"{float(selected_row[column]):.2%}")
    display_columns = ["model"] + [column for column, _ in metric_columns]
    display = models[display_columns].copy()
    display.columns = ["Model"] + [label for _, label in metric_columns]
    for column in display.columns[1:]:
        display[column] = display[column].map(lambda value: f"{float(value):.2%}")
    st.dataframe(display, hide_index=True, use_container_width=True)
    chart = models.melt(id_vars="model", value_vars=[column for column, _ in metric_columns], var_name="Metric", value_name="Score")
    chart["Metric"] = chart["Metric"].map(dict(metric_columns))
    st.plotly_chart(px.bar(chart, x="model", y="Score", color="Metric", barmode="group", range_y=[0, 1], title="Model metric comparison"), use_container_width=True)
def evaluation() -> None:
    st.title("Model Evaluation")
    st.caption("Metrics are shown separately because the two datasets have different targets and label definitions.")
    lifestyle_exists = METRICS_PATH.exists()
    mental_exists = MENTAL_HEALTH_METRICS_PATH.exists()
    if not lifestyle_exists and not mental_exists:
        st.warning("Train at least one model before opening this page.")
        return
    tabs = st.tabs([label for label, exists in [("Lifestyle depression model", lifestyle_exists), ("Mental-health status model", mental_exists)] if exists])
    tab_index = 0
    if lifestyle_exists:
        with tabs[tab_index]:
            _evaluation_panel(json.loads(METRICS_PATH.read_text(encoding="utf-8")), "Student lifestyle depression model", [("accuracy", "Accuracy"), ("precision", "At-risk precision"), ("recall", "At-risk recall"), ("f1", "At-risk F1")])
        tab_index += 1
    if mental_exists:
        with tabs[tab_index]:
            _evaluation_panel(json.loads(MENTAL_HEALTH_METRICS_PATH.read_text(encoding="utf-8")), "Supplementary mental-health status model", [("accuracy", "Accuracy"), ("macro_precision", "Macro precision"), ("macro_recall", "Macro recall"), ("macro_f1", "Macro F1")])

def admin() -> None:
    st.title("Admin: Counsellor Accounts")
    with st.form("new_counsellor"):
        username = st.text_input("Counsellor email address")
        password = st.text_input("Temporary password", type="password")
        submit = st.form_submit_button("Create counsellor account")
    if submit:
        try:
            create_counsellor(username, password)
            st.success("Counsellor account created. Give these credentials to the counsellor securely.")
        except ValueError as error:
            st.error(str(error))

def password_page() -> None:
    st.title("Change Password")
    with st.form("password"):
        current = st.text_input("Current password", type="password")
        new = st.text_input("New password (at least 8 characters)", type="password")
        confirm = st.text_input("Confirm new password", type="password")
        submit = st.form_submit_button("Update password")
    if submit:
        if new != confirm: st.error("The new passwords do not match.")
        else:
            try:
                if change_password(st.session_state.user["id"], current, new): st.success("Password updated.")
                else: st.error("Current password is incorrect.")
            except ValueError as error: st.error(str(error))

def main() -> None:
    if "user" not in st.session_state:
        login(); return
    user = st.session_state.user
    st.sidebar.title(f"Signed in: {user['username']}")
    # Administrators manage access only. Student information is available solely
    # to counsellor accounts.
    if user["role"] == "admin":
        pages = ["Admin Accounts"]
    else:
        pages = ["Overview", "Student Assessment", "Student History", "Model Evaluation", "Clustering", "Change Password"]
    page = st.sidebar.radio("Navigate", pages)
    if st.sidebar.button("Sign out"):
        del st.session_state.user; st.rerun()
    {"Overview": overview, "Student Assessment": assessment, "Student History": history, "Model Evaluation": evaluation, "Clustering": clustering, "Change Password": password_page, "Admin Accounts": admin}[page]()

if __name__ == "__main__": main()
