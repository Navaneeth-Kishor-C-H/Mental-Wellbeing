"""Generate the dummy student master dataset used by the counsellor lookup."""
from pathlib import Path
import csv
import hashlib

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
source_path = ROOT / "data" / "raw" / "student_lifestyle_100k.csv"
output_path = ROOT / "data" / "raw" / "student_details_dummy_100k.csv"
first_names = ["Aarav", "Aisha", "Daniel", "Fatima", "Ishaan", "Maya", "Noah", "Priya", "Sofia", "Zain"]
last_names = ["Adams", "Bennett", "Das", "Khan", "Mensah", "Patel", "Rossi", "Sharma", "Smith", "Wilson"]
programmes = {
    "Engineering": "B.Tech Engineering",
    "Science": "B.Sc Science",
    "Medical": "B.Sc Medical Sciences",
    "Arts": "B.A. Arts",
    "Commerce": "B.Com Commerce",
}


def name_for(student_id: str) -> str:
    digest = hashlib.sha256(student_id.encode()).digest()
    return f"{first_names[digest[0] % len(first_names)]} {last_names[digest[1] % len(last_names)]}"


def main() -> None:
    source = pd.read_csv(source_path, dtype={"Student_ID": str})
    source["Student_ID"] = source["Student_ID"].str.strip()
    source = source.drop_duplicates("Student_ID")
    source.insert(1, "Student_Name", source["Student_ID"].map(name_for))
    source.insert(2, "Programme", source["Department"].map(programmes).fillna("General Studies"))
    columns = ["Student_ID", "Student_Name", "Programme", "Age", "Gender", "Department", "CGPA", "Sleep_Duration", "Study_Hours", "Social_Media_Hours", "Physical_Activity", "Stress_Level"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source[columns].to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Generated {len(source):,} dummy student records at {output_path}")


if __name__ == "__main__":
    main()