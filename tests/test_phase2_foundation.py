import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from app.config import get_settings
from app.database.db import init_db
from app.llm.ollama_client import check_ollama_status


class StudyCollectionTests(unittest.TestCase):
    @patch("app.rag.rag_engine.get_chroma_client")
    @patch("app.rag.rag_engine.get_settings")
    def test_embedding_models_use_distinct_study_collections(
        self,
        mock_settings,
        mock_get_chroma_client
    ):
        from app.rag.rag_engine import get_study_collection

        first_model = "embedding-model-a"
        second_model = "embedding-model-b"
        first_collection = Mock(metadata={"embedding_model": first_model})
        second_collection = Mock(metadata={"embedding_model": second_model})
        client = Mock()
        client.get_or_create_collection.side_effect = [
            first_collection,
            second_collection
        ]
        mock_get_chroma_client.return_value = client
        mock_settings.side_effect = [
            get_settings({"OLLAMA_EMBEDDING_MODEL": first_model}),
            get_settings({"OLLAMA_EMBEDDING_MODEL": second_model})
        ]

        self.assertIs(get_study_collection(), first_collection)
        self.assertIs(get_study_collection(), second_collection)
        first_name = client.get_or_create_collection.call_args_list[0].kwargs["name"]
        second_name = client.get_or_create_collection.call_args_list[1].kwargs["name"]
        self.assertNotEqual(first_name, second_name)


class SettingsTests(unittest.TestCase):
    def test_reads_model_and_data_path_overrides(self):
        settings = get_settings({
            "OLLAMA_BASE_URL": "http://127.0.0.1:11435/",
            "OLLAMA_CHAT_MODEL": "local-chat:latest",
            "OLLAMA_EMBEDDING_MODEL": "local-embed:latest",
            "AI_INTERVIEW_DB_PATH": "~/study/interviews.db",
            "AI_INTERVIEW_CHROMA_PATH": "~/study/vectors",
            "AI_INTERVIEW_PDF_DIRECTORY": "~/study/pdfs",
            "OLLAMA_TIMEOUT_SECONDS": "45",
            "OLLAMA_EMBEDDING_TIMEOUT_SECONDS": "30"
        })

        self.assertEqual(settings.ollama_base_url, "http://127.0.0.1:11435")
        self.assertEqual(settings.chat_model, "local-chat:latest")
        self.assertEqual(settings.embedding_model, "local-embed:latest")
        self.assertEqual(settings.sqlite_db_path, Path("~/study/interviews.db").expanduser())
        self.assertEqual(settings.chroma_path, Path("~/study/vectors").expanduser())
        self.assertEqual(settings.pdf_directory, Path("~/study/pdfs").expanduser())
        self.assertEqual(settings.ollama_timeout_seconds, 45)
        self.assertEqual(settings.embedding_timeout_seconds, 30)


class OllamaStatusTests(unittest.TestCase):
    @patch("app.llm.ollama_client.get_settings")
    @patch("app.llm.ollama_client.requests.get")
    def test_reports_installed_configured_models(self, mock_get, mock_settings):
        mock_settings.return_value = get_settings({
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "OLLAMA_CHAT_MODEL": "chat:latest",
            "OLLAMA_EMBEDDING_MODEL": "embed:latest"
        })
        response = Mock()
        response.json.return_value = {
            "models": [{"name": "chat:latest"}, {"name": "embed:latest"}]
        }
        mock_get.return_value = response

        status = check_ollama_status()

        self.assertTrue(status["service_available"])
        self.assertTrue(status["chat_model_available"])
        self.assertTrue(status["embedding_model_available"])
        mock_get.assert_called_once_with(
            "http://localhost:11434/api/tags",
            timeout=5
        )

    @patch("app.llm.ollama_client.get_settings")
    @patch("app.llm.ollama_client.requests.get")
    def test_reports_missing_models(self, mock_get, mock_settings):
        mock_settings.return_value = get_settings({
            "OLLAMA_CHAT_MODEL": "not-installed:latest",
            "OLLAMA_EMBEDDING_MODEL": "also-missing:latest"
        })
        response = Mock()
        response.json.return_value = {"models": [{"name": "another:latest"}]}
        mock_get.return_value = response

        status = check_ollama_status()

        self.assertTrue(status["service_available"])
        self.assertFalse(status["chat_model_available"])
        self.assertFalse(status["embedding_model_available"])

    @patch("app.llm.ollama_client.get_settings")
    @patch("app.llm.ollama_client.requests.get")
    def test_reports_ollama_connection_failure(self, mock_get, mock_settings):
        mock_settings.return_value = get_settings({})
        mock_get.side_effect = requests.exceptions.ConnectionError("offline")

        status = check_ollama_status()

        self.assertFalse(status["service_available"])
        self.assertIn("ollama serve", status["error"])


class DatabaseSchemaTests(unittest.TestCase):
    def test_initializes_additive_schema_and_preserves_interview_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "nested" / "interviews.db"
            db_path.parent.mkdir(parents=True)
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    CREATE TABLE interviews (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        question TEXT
                    )
                """)
                conn.execute(
                    "INSERT INTO interviews (question) VALUES (?)",
                    ("Existing interview question",)
                )

            init_db(db_path)
            init_db(db_path)

            with sqlite3.connect(db_path) as conn:
                existing_question = conn.execute(
                    "SELECT question FROM interviews WHERE id = 1"
                ).fetchone()[0]
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                interview_columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(interviews)")
                }

            self.assertEqual(existing_question, "Existing interview question")
            self.assertTrue({
                "study_documents",
                "study_sources",
                "study_questions",
                "practice_sessions",
                "practice_attempts",
                "revision_schedules"
            }.issubset(tables))
            self.assertTrue({
                "role",
                "resume",
                "job_description",
                "raw_evaluation",
                "created_at"
            }.issubset(interview_columns))

    def test_study_relations_enforce_foreign_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "interviews.db"
            init_db(db_path)

            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute("""
                        INSERT INTO practice_attempts (id, question_id, answer)
                        VALUES ('attempt-1', 'missing-question', 'answer')
                    """)


if __name__ == "__main__":
    unittest.main()
