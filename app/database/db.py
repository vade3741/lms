import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH = Path("app/database/interviews.db")


COLUMNS = {
    "role": "TEXT",
    "resume": "TEXT",
    "job_description": "TEXT",
    "question": "TEXT",
    "answer": "TEXT",
    "score": "INTEGER DEFAULT 0",
    "technical_feedback": "TEXT DEFAULT ''",
    "communication_feedback": "TEXT DEFAULT ''",
    "missing_points": "TEXT DEFAULT ''",
    "improved_answer": "TEXT DEFAULT ''",
    "final_suggestion": "TEXT DEFAULT ''",
    "raw_evaluation": "TEXT DEFAULT ''",
    "created_at": "TEXT DEFAULT ''"
}


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT
        )
    """)

    cursor.execute("PRAGMA table_info(interviews)")
    existing_columns = [column[1] for column in cursor.fetchall()]

    for column_name, column_type in COLUMNS.items():
        if column_name not in existing_columns:
            cursor.execute(
                f"ALTER TABLE interviews ADD COLUMN {column_name} {column_type}"
            )

    conn.commit()
    conn.close()


def save_interview(
    role,
    resume,
    job_description,
    question,
    answer,
    score,
    technical_feedback,
    communication_feedback,
    missing_points,
    improved_answer,
    final_suggestion,
    raw_evaluation
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO interviews (
            role,
            resume,
            job_description,
            question,
            answer,
            score,
            technical_feedback,
            communication_feedback,
            missing_points,
            improved_answer,
            final_suggestion,
            raw_evaluation,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        role,
        resume,
        job_description,
        question,
        answer,
        score,
        technical_feedback,
        communication_feedback,
        missing_points,
        improved_answer,
        final_suggestion,
        raw_evaluation,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_interviews():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            role,
            question,
            answer,
            score,
            technical_feedback,
            communication_feedback,
            missing_points,
            improved_answer,
            final_suggestion,
            created_at
        FROM interviews
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return rows