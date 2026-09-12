# Student Wellbeing Intelligence System

Python-only student wellbeing analytics system built with Streamlit, Flask, scikit-learn, SHAP, LIME, Plotly, Matplotlib, and Seaborn. It is an educational screening tool, not a clinical diagnostic system.

## What it does

1. Cleans and validates the student lifestyle dataset.
2. Clusters students with K-Means, Agglomerative Clustering, and Gaussian Mixture Models over 2–6 segments.
3. Selects the strongest clustering candidate using silhouette score, Davies–Bouldin index, and Calinski–Harabasz score.
4. Adds the selected student segment to the prediction features.
5. Benchmarks Random Forest and Extra Trees classifiers, selecting the best with F1 score then accuracy.
6. Predicts an individual student's depression-risk outcome, cluster, class probabilities, SHAP contributions, and LIME explanation.

## Setup and use

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
py run_train.py
py run_dashboard.py
```

Run the Flask API with `py run_api.py`. Send an `application/json` POST to `/predict` using the fields shown in the dashboard. Training creates `models/wellbeing_system.joblib`, `data/processed/student_lifestyle_clustered.csv`, and `data/processed/training_metrics.json`.

The required raw dataset is `data/raw/student_lifestyle_100k.csv`, with `Age`, `Gender`, `Department`, `CGPA`, `Sleep_Duration`, `Study_Hours`, `Social_Media_Hours`, `Physical_Activity`, `Stress_Level`, and `Depression` columns.

The counsellor portal uses the generated student directory at `data/raw/student_details_dummy_100k.csv`. It contains 100,000 dummy student records with roll number, name, programme, age, gender, department, and lifestyle details. Regenerate it with `py scripts/generate_student_directory.py` when the source lifestyle dataset changes. Counsellors enter the roll number in the assessment screen; the matching profile and lifestyle fields are then filled from this directory.

## Counsellor portal

The Streamlit dashboard includes a login-protected counsellor portal. On its first start, it creates a local SQLite database at `data/wellbeing_portal.db` and displays a **Create administrator account** screen. Choose the admin username and password there; no preset password is used.

After setup, choose **Admin** or **Counsellor** on the login screen. Admins can add approved campuses and create counsellor accounts using the counsellor's email address, assigned campus, and temporary password; they cannot see student information. Counsellors select their campus at login, but the server accepts only the campus assigned to their account. They can save student assessments, view student history, and change their own password.

Student records are scoped to the counsellor's assigned campus. A counsellor can only see the overview, assessments, and history for that campus. Existing databases are migrated automatically; records created before campus support are assigned to `Main Campus`. Student IDs remain globally unique, so the same ID cannot currently be used for two different campuses.
