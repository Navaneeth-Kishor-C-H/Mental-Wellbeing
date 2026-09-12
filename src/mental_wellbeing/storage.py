"""SQLite storage for counsellor accounts and student assessment history."""
from __future__ import annotations

import json
import csv
import re
import sqlite3
from datetime import datetime, timezone

from werkzeug.security import check_password_hash, generate_password_hash

from .config import DATABASE_PATH, STUDENT_DIRECTORY_CSV


def _connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    return con


def initialise_database() -> None:
    with _connection() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS campuses (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('admin', 'counsellor')),
                campus TEXT NOT NULL DEFAULT 'Main Campus', created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL, programme TEXT, campus TEXT NOT NULL DEFAULT 'Main Campus', created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS student_directory (
                student_id TEXT PRIMARY KEY, full_name TEXT NOT NULL, programme TEXT NOT NULL,
                age INTEGER NOT NULL, gender TEXT NOT NULL, department TEXT NOT NULL,
                cgpa REAL NOT NULL, sleep_duration REAL NOT NULL, study_hours REAL NOT NULL,
                social_media_hours REAL NOT NULL, physical_activity INTEGER NOT NULL,
                stress_level INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT, student_id INTEGER NOT NULL,
                counsellor_id INTEGER NOT NULL, created_at TEXT NOT NULL,
                input_json TEXT NOT NULL, result_json TEXT NOT NULL,
                FOREIGN KEY(student_id) REFERENCES students(id), FOREIGN KEY(counsellor_id) REFERENCES users(id)
            );
        """)
        user_columns = {row["name"] for row in con.execute("PRAGMA table_info(users)")}
        if "campus" not in user_columns:
            con.execute("ALTER TABLE users ADD COLUMN campus TEXT NOT NULL DEFAULT 'Main Campus'")
        student_columns = {row["name"] for row in con.execute("PRAGMA table_info(students)")}
        if "campus" not in student_columns:
            con.execute("ALTER TABLE students ADD COLUMN campus TEXT NOT NULL DEFAULT 'Main Campus'")
        con.execute("INSERT OR IGNORE INTO campuses (name, created_at) VALUES ('Main Campus', ?)", (_now(),))
        con.execute("INSERT OR IGNORE INTO campuses (name, created_at) SELECT DISTINCT campus, ? FROM users WHERE campus IS NOT NULL AND TRIM(campus) <> ''", (_now(),))
        con.execute("INSERT OR IGNORE INTO campuses (name, created_at) SELECT DISTINCT campus, ? FROM students WHERE campus IS NOT NULL AND TRIM(campus) <> ''", (_now(),))
        if not con.execute("SELECT 1 FROM student_directory LIMIT 1").fetchone() and STUDENT_DIRECTORY_CSV.exists():
            with STUDENT_DIRECTORY_CSV.open(newline="", encoding="utf-8") as directory_file:
                rows = csv.DictReader(directory_file)
                con.executemany("""INSERT OR IGNORE INTO student_directory
                    (student_id, full_name, programme, age, gender, department, cgpa, sleep_duration,
                     study_hours, social_media_hours, physical_activity, stress_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    ((row["Student_ID"], row["Student_Name"], row["Programme"], int(row["Age"]), row["Gender"],
                      row["Department"], float(row["CGPA"]), float(row["Sleep_Duration"]), float(row["Study_Hours"]),
                      float(row["Social_Media_Hours"]), int(row["Physical_Activity"]), int(row["Stress_Level"]))
                     for row in rows))
        # The first administrator is explicitly created from the Streamlit setup
        # screen; no default password is stored in the application.


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def admin_exists() -> bool:
    with _connection() as con:
        return con.execute("SELECT 1 FROM users WHERE role = 'admin'").fetchone() is not None


def list_campuses() -> list[str]:
    with _connection() as con:
        rows = con.execute("SELECT name FROM campuses ORDER BY name").fetchall()
    return [row["name"] for row in rows]


def create_campus(name: str) -> None:
    name = name.strip()
    if len(name) < 2:
        raise ValueError("Campus name must have at least 2 characters.")
    try:
        with _connection() as con:
            con.execute("INSERT INTO campuses (name, created_at) VALUES (?, ?)", (name, _now()))
    except sqlite3.IntegrityError as error:
        raise ValueError("That campus already exists.") from error


def create_admin(username: str, password: str) -> None:
    username = username.strip()
    if len(username) < 3:
        raise ValueError("Admin username must have at least 3 characters.")
    if len(password) < 8:
        raise ValueError("Admin password must have at least 8 characters.")
    with _connection() as con:
        if con.execute("SELECT 1 FROM users WHERE role = 'admin'").fetchone():
            raise ValueError("An administrator account already exists.")
        con.execute("INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, 'admin', ?)", (username, generate_password_hash(password), _now()))


def authenticate(username: str, password: str, role: str | None = None, campus: str | None = None) -> dict | None:
    with _connection() as con:
        row = con.execute("SELECT id, username, password_hash, role, campus FROM users WHERE username = ?", (username.strip().lower(),)).fetchone()
    campus_matches = campus is None or row and row["campus"].casefold() == campus.strip().casefold()
    if row and (role is None or row["role"] == role) and campus_matches and check_password_hash(row["password_hash"], password):
        return {"id": row["id"], "username": row["username"], "role": row["role"], "campus": row["campus"]}
    return None


def create_counsellor(username: str, password: str, campus: str = "Main Campus") -> None:
    username = username.strip().lower()
    campus = campus.strip()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", username):
        raise ValueError("Enter a valid counsellor email address.")
    if not campus:
        raise ValueError("Campus is required.")
    if len(password) < 8:
        raise ValueError("Password must have at least 8 characters.")
    try:
        with _connection() as con:
            if not con.execute("SELECT 1 FROM campuses WHERE name = ? COLLATE NOCASE", (campus,)).fetchone():
                raise ValueError("Select a campus from the approved campus list.")
            con.execute("INSERT INTO users (username, password_hash, role, campus, created_at) VALUES (?, ?, 'counsellor', ?, ?)", (username, generate_password_hash(password), campus, _now()))
    except sqlite3.IntegrityError as error:
        raise ValueError("That username is already in use.") from error


def change_password(user_id: int, current_password: str, new_password: str) -> bool:
    if len(new_password) < 8:
        raise ValueError("New password must have at least 8 characters.")
    with _connection() as con:
        user = con.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user or not check_password_hash(user["password_hash"], current_password):
            return False
        con.execute("UPDATE users SET password_hash = ? WHERE id = ?", (generate_password_hash(new_password), user_id))
    return True


def save_assessment(student_code: str, name: str, programme: str, payload: dict, result: dict, counsellor_id: int, campus: str = "Main Campus") -> None:
    if not student_code.strip() or not name.strip():
        raise ValueError("Student ID and student name are required to save an assessment.")
    campus = campus.strip()
    if not campus:
        raise ValueError("A campus must be assigned before saving an assessment.")
    with _connection() as con:
        existing = con.execute("SELECT id, campus FROM students WHERE student_id = ?", (student_code.strip(),)).fetchone()
        if existing and existing["campus"] != campus:
            raise ValueError("This student ID belongs to another campus and cannot be accessed here.")
        con.execute("INSERT INTO students (student_id, full_name, programme, campus, created_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(student_id) DO UPDATE SET full_name=excluded.full_name, programme=excluded.programme", (student_code.strip(), name.strip(), programme.strip(), campus, _now()))
        student = con.execute("SELECT id FROM students WHERE student_id = ? AND campus = ?", (student_code.strip(), campus)).fetchone()
        con.execute("INSERT INTO assessments (student_id, counsellor_id, created_at, input_json, result_json) VALUES (?, ?, ?, ?, ?)", (student["id"], counsellor_id, _now(), json.dumps(payload), json.dumps(result)))


def student_history(student_code: str, campus: str = "Main Campus") -> list[dict]:
    with _connection() as con:
        rows = con.execute("""SELECT s.student_id, s.full_name, s.programme, a.created_at, u.username AS counsellor, a.input_json, a.result_json
            FROM assessments a JOIN students s ON s.id=a.student_id JOIN users u ON u.id=a.counsellor_id
            WHERE s.student_id = ? AND s.campus = ? ORDER BY a.created_at DESC""", (student_code.strip(), campus.strip())).fetchall()
    return [{**dict(row), "input": json.loads(row["input_json"]), "result": json.loads(row["result_json"])} for row in rows]


def recent_students(campus: str | None = None) -> list[dict]:
    with _connection() as con:
        if campus is None:
            rows = con.execute("SELECT student_id, full_name, programme, campus FROM students ORDER BY full_name").fetchall()
        else:
            rows = con.execute("SELECT student_id, full_name, programme, campus FROM students WHERE campus = ? ORDER BY full_name", (campus.strip(),)).fetchall()
    return [dict(row) for row in rows]


def find_directory_student(student_code: str) -> dict | None:
    """Return the seeded student profile for a roll number."""
    with _connection() as con:
        row = con.execute("SELECT * FROM student_directory WHERE student_id = ?", (student_code.strip(),)).fetchone()
    return dict(row) if row else None
