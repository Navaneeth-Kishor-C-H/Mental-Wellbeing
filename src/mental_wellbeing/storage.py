"""SQLite storage for counsellor accounts and student assessment history."""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone

from werkzeug.security import check_password_hash, generate_password_hash

from .config import DATABASE_PATH


def _connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    return con


def initialise_database() -> None:
    with _connection() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('admin', 'counsellor')),
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL, programme TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT, student_id INTEGER NOT NULL,
                counsellor_id INTEGER NOT NULL, created_at TEXT NOT NULL,
                input_json TEXT NOT NULL, result_json TEXT NOT NULL,
                FOREIGN KEY(student_id) REFERENCES students(id), FOREIGN KEY(counsellor_id) REFERENCES users(id)
            );
        """)
        # The first administrator is explicitly created from the Streamlit setup
        # screen; no default password is stored in the application.


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def admin_exists() -> bool:
    with _connection() as con:
        return con.execute("SELECT 1 FROM users WHERE role = 'admin'").fetchone() is not None


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


def authenticate(username: str, password: str, role: str | None = None) -> dict | None:
    with _connection() as con:
        row = con.execute("SELECT id, username, password_hash, role FROM users WHERE username = ?", (username.strip().lower(),)).fetchone()
    if row and (role is None or row["role"] == role) and check_password_hash(row["password_hash"], password):
        return {"id": row["id"], "username": row["username"], "role": row["role"]}
    return None


def create_counsellor(username: str, password: str) -> None:
    username = username.strip().lower()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", username):
        raise ValueError("Enter a valid counsellor email address.")
    if len(password) < 8:
        raise ValueError("Password must have at least 8 characters.")
    try:
        with _connection() as con:
            con.execute("INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, 'counsellor', ?)", (username, generate_password_hash(password), _now()))
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


def save_assessment(student_code: str, name: str, programme: str, payload: dict, result: dict, counsellor_id: int) -> None:
    if not student_code.strip() or not name.strip():
        raise ValueError("Student ID and student name are required to save an assessment.")
    with _connection() as con:
        con.execute("INSERT INTO students (student_id, full_name, programme, created_at) VALUES (?, ?, ?, ?) ON CONFLICT(student_id) DO UPDATE SET full_name=excluded.full_name, programme=excluded.programme", (student_code.strip(), name.strip(), programme.strip(), _now()))
        student = con.execute("SELECT id FROM students WHERE student_id = ?", (student_code.strip(),)).fetchone()
        con.execute("INSERT INTO assessments (student_id, counsellor_id, created_at, input_json, result_json) VALUES (?, ?, ?, ?, ?)", (student["id"], counsellor_id, _now(), json.dumps(payload), json.dumps(result)))


def student_history(student_code: str) -> list[dict]:
    with _connection() as con:
        rows = con.execute("""SELECT s.student_id, s.full_name, s.programme, a.created_at, u.username AS counsellor, a.input_json, a.result_json
            FROM assessments a JOIN students s ON s.id=a.student_id JOIN users u ON u.id=a.counsellor_id
            WHERE s.student_id = ? ORDER BY a.created_at DESC""", (student_code.strip(),)).fetchall()
    return [{**dict(row), "input": json.loads(row["input_json"]), "result": json.loads(row["result_json"])} for row in rows]


def recent_students() -> list[dict]:
    with _connection() as con:
        rows = con.execute("SELECT student_id, full_name, programme FROM students ORDER BY full_name").fetchall()
    return [dict(row) for row in rows]
