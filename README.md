# Student Mental Wellbeing Prediction

Semester project for analyzing student lifestyle patterns and predicting mental wellbeing risk using machine learning.

## Project Features

- Cleans and validates two student wellbeing datasets.
- Trains two models:
  - `mental_status`: predicts mental health status from scores, sleep, GPA, activity, mood, and reflection sentiment.
  - `depression_risk`: predicts depression risk from student lifestyle factors.
- Provides a Streamlit dashboard for dataset exploration and predictions.
- Explains predictions using model feature importance and low-risk patterns learned from the dataset.
- Generates recommendations by comparing student input with the dataset's low-risk profile.
- Provides a small Flask API for programmatic predictions.
- Includes tests and a simple project report outline.

## Directory Structure

```text
Mental-Wellbeing/
  data/
    raw/                 Original CSV files
    processed/           Cleaned/generated datasets
  docs/                  Proposal and synopsis PDFs
  models/                Trained model files
  notebooks/             Notebook workspace
  reports/               Project report material
  src/mental_wellbeing/  Application source code
  tests/                 Automated tests
```

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m mental_wellbeing.train
streamlit run src\mental_wellbeing\web\dashboard.py
```

If Python cannot find the package, install it in editable mode:

```powershell
pip install -e .
```

## Run The API

```powershell
python -m mental_wellbeing.web.api
```

Example request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict/depression-risk -ContentType "application/json" -Body '{
  "Age": 21,
  "Gender": "Female",
  "Department": "Engineering",
  "CGPA": 3.2,
  "Sleep_Duration": 6.5,
  "Study_Hours": 5.0,
  "Social_Media_Hours": 3.0,
  "Physical_Activity": 60,
  "Stress_Level": 6
}'
```

## Academic Note

This system is for educational screening and analysis only. It must not be used as a clinical diagnosis tool.
