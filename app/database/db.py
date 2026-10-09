import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from app.config import get_settings


DB_PATH = get_settings().sqlite_db_path


def _connect(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def _connection(db_path=DB_PATH):
    conn = _connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


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


def init_db(db_path=DB_PATH):
    db_path = db_path.expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with _connection(db_path) as conn:
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

        cursor.executescript("""
        CREATE TABLE IF NOT EXISTS study_documents (
            id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL UNIQUE,
            filename TEXT NOT NULL,
            sha256 TEXT NOT NULL DEFAULT '',
            page_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS study_sources (
            id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL DEFAULT '',
            publisher TEXT NOT NULL DEFAULT '',
            retrieved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS study_notes (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            note_text TEXT NOT NULL,
            source_evidence_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS study_questions (
            id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            topic TEXT NOT NULL DEFAULT '',
            question_type TEXT NOT NULL DEFAULT 'fundamentals',
            difficulty TEXT NOT NULL DEFAULT 'medium',
            provenance TEXT NOT NULL,
            expected_concepts_json TEXT NOT NULL DEFAULT '[]',
            hint TEXT NOT NULL DEFAULT '',
            sample_answer TEXT NOT NULL DEFAULT '',
            source_id TEXT REFERENCES study_sources(id) ON DELETE SET NULL,
            document_id TEXT REFERENCES study_documents(id) ON DELETE SET NULL,
            page_number INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS practice_sessions (
            id TEXT PRIMARY KEY,
            mode TEXT NOT NULL DEFAULT 'single_question',
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            ended_at TEXT
        );

        CREATE TABLE IF NOT EXISTS practice_attempts (
            id TEXT PRIMARY KEY,
            question_id TEXT NOT NULL
                REFERENCES study_questions(id) ON DELETE CASCADE,
            session_id TEXT REFERENCES practice_sessions(id) ON DELETE SET NULL,
            answer TEXT NOT NULL,
            score INTEGER,
            score_breakdown_json TEXT NOT NULL DEFAULT '{}',
            feedback_json TEXT NOT NULL DEFAULT '{}',
            source_evidence_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS revision_schedules (
            question_id TEXT PRIMARY KEY
                REFERENCES study_questions(id) ON DELETE CASCADE,
            last_attempted_at TEXT,
            last_score INTEGER,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            next_review_at TEXT,
            reason TEXT NOT NULL DEFAULT '',
            extra_revision INTEGER NOT NULL DEFAULT 0
                CHECK (extra_revision IN (0, 1))
        );

        CREATE INDEX IF NOT EXISTS idx_study_questions_topic
            ON study_questions(topic);
        CREATE INDEX IF NOT EXISTS idx_study_questions_provenance
            ON study_questions(provenance);
        CREATE INDEX IF NOT EXISTS idx_practice_attempts_question_date
            ON practice_attempts(question_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_revision_schedules_due
            ON revision_schedules(next_review_at);
        """)


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
    with _connection() as conn:
        conn.execute("""
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


def get_interviews():
    with _connection() as conn:
        rows = conn.execute("""
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
        """).fetchall()

    return rows


def save_study_document(document, db_path=DB_PATH):
    with _connection(db_path) as conn:
        conn.execute("""
            INSERT INTO study_documents (
                id, file_path, filename, sha256, page_count, status, error,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(file_path) DO UPDATE SET
                filename = excluded.filename,
                sha256 = excluded.sha256,
                page_count = excluded.page_count,
                status = excluded.status,
                error = excluded.error,
                updated_at = CURRENT_TIMESTAMP
        """, (
            document["id"],
            document["file_path"],
            document["filename"],
            document.get("sha256", ""),
            document.get("page_count", 0),
            document.get("status", "pending"),
            document.get("error", "")
        ))
        row = conn.execute(
            "SELECT id FROM study_documents WHERE file_path = ?",
            (document["file_path"],)
        ).fetchone()
        return row[0]


def get_study_documents(db_path=DB_PATH):
    with _connection(db_path) as conn:
        rows = conn.execute("""
            SELECT id, file_path, filename, sha256, page_count, status, error
            FROM study_documents
            ORDER BY filename COLLATE NOCASE
        """).fetchall()
    return [
        {
            "id": row[0],
            "file_path": row[1],
            "filename": row[2],
            "sha256": row[3],
            "page_count": row[4],
            "status": row[5],
            "error": row[6]
        }
        for row in rows
    ]


def delete_study_document(document_id, db_path=DB_PATH):
    with _connection(db_path) as conn:
        conn.execute(
            "DELETE FROM study_questions WHERE document_id = ?",
            (document_id,)
        )
        conn.execute("DELETE FROM study_documents WHERE id = ?", (document_id,))


def save_study_source(source, db_path=DB_PATH):
    with _connection(db_path) as conn:
        conn.execute("""
            INSERT INTO study_sources (
                id, source_type, title, url, publisher, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                url = excluded.url,
                publisher = excluded.publisher,
                metadata_json = excluded.metadata_json
        """, (
            source["id"],
            source["source_type"],
            source["title"],
            source.get("url", ""),
            source.get("publisher", ""),
            json.dumps(source.get("metadata", {}), ensure_ascii=False)
        ))


def get_study_source(source_id, db_path=DB_PATH):
    with _connection(db_path) as conn:
        row = conn.execute("""
            SELECT id, source_type, title, url, publisher, metadata_json
            FROM study_sources
            WHERE id = ?
        """, (source_id,)).fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "source_type": row[1],
        "title": row[2],
        "url": row[3],
        "publisher": row[4],
        "metadata": json.loads(row[5])
    }


def get_study_sources(source_type=None, db_path=DB_PATH):
    with _connection(db_path) as conn:
        if source_type:
            rows = conn.execute("""
                SELECT id, source_type, title, url, publisher, metadata_json
                FROM study_sources
                WHERE source_type = ?
                ORDER BY retrieved_at DESC
            """, (source_type,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, source_type, title, url, publisher, metadata_json
                FROM study_sources
                ORDER BY retrieved_at DESC
            """).fetchall()
    return [
        {
            "id": row[0],
            "source_type": row[1],
            "title": row[2],
            "url": row[3],
            "publisher": row[4],
            "metadata": json.loads(row[5])
        }
        for row in rows
    ]


def save_study_note(note, db_path=DB_PATH):
    with _connection(db_path) as conn:
        conn.execute("""
            INSERT INTO study_notes (id, topic, note_text, source_evidence_json)
            VALUES (?, ?, ?, ?)
        """, (
            note["id"],
            note["topic"],
            note["note_text"],
            json.dumps(note.get("source_evidence", []), ensure_ascii=False)
        ))


def get_study_notes(limit=20, db_path=DB_PATH):
    with _connection(db_path) as conn:
        rows = conn.execute("""
            SELECT topic, note_text, source_evidence_json, created_at
            FROM study_notes
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [
        {
            "topic": row[0],
            "note_text": row[1],
            "source_evidence": json.loads(row[2]),
            "created_at": row[3]
        }
        for row in rows
    ]


def save_study_question(question, db_path=DB_PATH):
    with _connection(db_path) as conn:
        conn.execute("""
            INSERT OR IGNORE INTO study_questions (
                id, question, topic, question_type, difficulty, provenance,
                expected_concepts_json, hint, sample_answer, source_id,
                document_id, page_number
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            question["id"],
            question["question"],
            question.get("topic", ""),
            question.get("question_type", "fundamentals"),
            question.get("difficulty", "medium"),
            question["provenance"],
            json.dumps(question.get("expected_concepts", []), ensure_ascii=False),
            question.get("hint", ""),
            question.get("sample_answer", ""),
            question.get("source_id"),
            question.get("document_id"),
            question.get("page_number")
        ))
        conn.execute("""
            INSERT OR IGNORE INTO revision_schedules (question_id, next_review_at)
            VALUES (?, CURRENT_TIMESTAMP)
        """, (question["id"],))


def get_study_questions(due_only=False, db_path=DB_PATH):
    due_filter = (
        "AND COALESCE(rs.next_review_at, CURRENT_TIMESTAMP) <= CURRENT_TIMESTAMP"
        if due_only else ""
    )
    with _connection(db_path) as conn:
        rows = conn.execute(f"""
            SELECT q.id, q.question, q.topic, q.question_type, q.difficulty,
                   q.provenance, q.expected_concepts_json, q.hint,
                   q.sample_answer, q.source_id, q.document_id, q.page_number,
                   s.title, s.url, s.metadata_json, d.filename, rs.next_review_at,
                   rs.last_score, rs.attempt_count, rs.reason
            FROM study_questions q
            LEFT JOIN study_sources s ON s.id = q.source_id
            LEFT JOIN study_documents d ON d.id = q.document_id
            LEFT JOIN revision_schedules rs ON rs.question_id = q.id
            WHERE 1 = 1 {due_filter}
            ORDER BY COALESCE(rs.next_review_at, q.created_at), q.created_at
        """).fetchall()
    return [
        {
            "id": row[0],
            "question": row[1],
            "topic": row[2],
            "question_type": row[3],
            "difficulty": row[4],
            "provenance": row[5],
            "expected_concepts": json.loads(row[6]),
            "hint": row[7],
            "sample_answer": row[8],
            "source_id": row[9],
            "document_id": row[10],
            "page_number": row[11],
            "source_title": row[12],
            "source_url": row[13],
            "source_metadata": json.loads(row[14] or "{}"),
            "filename": row[15],
            "next_review_at": row[16],
            "last_score": row[17],
            "attempt_count": row[18] or 0,
            "review_reason": row[19] or ""
        }
        for row in rows
    ]


def save_practice_attempt(attempt, db_path=DB_PATH):
    with _connection(db_path) as conn:
        if attempt.get("session_id"):
            conn.execute("""
                INSERT OR IGNORE INTO practice_sessions (id, mode, ended_at)
                VALUES (?, 'single_question', CURRENT_TIMESTAMP)
            """, (attempt["session_id"],))
        conn.execute("""
            INSERT INTO practice_attempts (
                id, question_id, session_id, answer, score,
                score_breakdown_json, feedback_json, source_evidence_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            attempt["id"],
            attempt["question_id"],
            attempt.get("session_id"),
            attempt["answer"],
            attempt["score"],
            json.dumps(attempt.get("score_breakdown", {}), ensure_ascii=False),
            json.dumps(attempt.get("feedback", {}), ensure_ascii=False),
            json.dumps(attempt.get("source_evidence", []), ensure_ascii=False)
        ))
        conn.execute("""
            INSERT INTO revision_schedules (
                question_id, last_attempted_at, last_score, attempt_count,
                next_review_at, reason, extra_revision
            )
            VALUES (?, CURRENT_TIMESTAMP, ?, 1, ?, ?, ?)
            ON CONFLICT(question_id) DO UPDATE SET
                last_attempted_at = CURRENT_TIMESTAMP,
                last_score = excluded.last_score,
                attempt_count = revision_schedules.attempt_count + 1,
                next_review_at = excluded.next_review_at,
                reason = excluded.reason,
                extra_revision = excluded.extra_revision
        """, (
            attempt["question_id"],
            attempt["score"],
            attempt["next_review_at"],
            attempt.get("reason", ""),
            int(attempt.get("extra_revision", False))
        ))


def get_practice_history(limit=20, db_path=DB_PATH):
    with _connection(db_path) as conn:
        rows = conn.execute("""
            SELECT a.created_at, q.question, q.topic, a.answer, a.score,
                   a.feedback_json, a.source_evidence_json
            FROM practice_attempts a
            JOIN study_questions q ON q.id = a.question_id
            ORDER BY a.created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [
        {
            "created_at": row[0],
            "question": row[1],
            "topic": row[2],
            "answer": row[3],
            "score": row[4],
            "feedback": json.loads(row[5]),
            "source_evidence": json.loads(row[6])
        }
        for row in rows
    ]