import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pymupdf
import requests

from app.database.db import (
    _connection,
    get_practice_history,
    get_study_documents,
    get_study_notes,
    get_study_questions,
    init_db,
    save_practice_attempt,
    save_study_document,
    save_study_note,
    save_study_question,
    save_study_source
)
from app.rag.text_splitter import split_text
from app.study_workflow import (
    MAX_PDF_SIZE_BYTES,
    _local_json_response,
    add_manual_question,
    add_pdf_files,
    evaluate_study_answer,
    generate_pdf_questions,
    import_web_source,
    search_public_questions
)


class FakeUpload:
    def __init__(self, name, content):
        self.name = name
        self.content = content

    def getvalue(self):
        return self.content


def make_pdf_bytes():
    document = pymupdf.open()
    document.new_page().insert_text(
        (72, 72),
        "Kafka producer retries use sequence numbers to avoid duplicate records."
    )
    contents = document.tobytes()
    document.close()
    return contents


class StudyUploadTests(unittest.TestCase):
    @patch("app.study_workflow.sync_pdf_library")
    def test_upload_sanitizes_filename_and_deduplicates_bytes(self, mock_sync):
        mock_sync.side_effect = lambda directory: {
            "indexed": 1,
            "unchanged": 0,
            "removed": 0,
            "pdf_count": 1,
            "errors": [],
            "documents": []
        }
        pdf_data = make_pdf_bytes()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = add_pdf_files(
                [FakeUpload("../Kafka notes.pdf", pdf_data)],
                directory=temp_dir
            )
            duplicate_result = add_pdf_files(
                [FakeUpload("renamed.pdf", pdf_data)],
                directory=temp_dir
            )
            names = [path.name for path in Path(temp_dir).iterdir()]

        self.assertEqual(result["uploaded"], ["Kafka notes.pdf"])
        self.assertEqual(duplicate_result["duplicates"], ["Kafka notes.pdf"])
        self.assertEqual(names, ["Kafka notes.pdf"])
        self.assertEqual(mock_sync.call_count, 2)

    def test_upload_rejects_non_pdf_and_oversized_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, r"\.pdf"):
                add_pdf_files(
                    [FakeUpload("notes.txt", b"not a PDF")],
                    directory=temp_dir
                )
            with patch(
                "app.study_workflow.MAX_PDF_SIZE_BYTES",
                5
            ):
                with self.assertRaisesRegex(ValueError, "50 MB"):
                    add_pdf_files(
                        [FakeUpload("large.pdf", make_pdf_bytes())],
                        directory=temp_dir
                    )


class ResearchTests(unittest.TestCase):
    @patch("app.study_workflow.DDGS")
    def test_public_search_filters_non_http_links(self, mock_ddgs):
        mock_ddgs.return_value.text.return_value = [
            {"title": "Kafka interview", "href": "https://example.com/a", "body": "Snippet"},
            {"title": "Invalid", "href": "javascript:alert(1)", "body": "Ignore"}
        ]

        results = search_public_questions("Kafka interview")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["publisher"], "example.com")
        mock_ddgs.return_value.text.assert_called_once_with(
            "Kafka interview",
            max_results=5
        )

    @patch("app.study_workflow.save_study_source")
    @patch("app.study_workflow.trafilatura.extract_metadata")
    @patch("app.study_workflow.trafilatura.extract")
    @patch("app.study_workflow.requests.get")
    def test_import_saves_only_bounded_article_excerpt(
        self,
        mock_get,
        mock_extract,
        mock_metadata,
        mock_save_source
    ):
        mock_get.return_value = SimpleNamespace(
            content=b"<html>article</html>",
            text="<html>article</html>",
            raise_for_status=Mock()
        )
        mock_extract.return_value = "Useful article text " * 1000
        mock_metadata.return_value = SimpleNamespace(
            title="Article title",
            sitename="Publisher"
        )

        source = import_web_source("https://example.com/article")

        self.assertEqual(source["title"], "Article title")
        self.assertEqual(source["publisher"], "Publisher")
        self.assertEqual(len(source["metadata"]["excerpt"]), 12000)
        mock_save_source.assert_called_once_with(source)

    def test_import_rejects_non_http_urls(self):
        with self.assertRaisesRegex(ValueError, "http"):
            import_web_source("file:///private/document.pdf")

    @patch("app.study_workflow.save_study_source")
    @patch("app.study_workflow.requests.get")
    def test_import_uses_attributed_search_snippet_when_source_returns_403(
        self,
        mock_get,
        mock_save_source
    ):
        forbidden_response = SimpleNamespace(status_code=403)
        error = requests.HTTPError("403 Forbidden", response=forbidden_response)
        mock_get.return_value.raise_for_status.side_effect = error
        snippet = (
            "Kafka producer idempotence uses producer IDs and sequence numbers "
            "to prevent duplicate writes during retries."
        )

        source = import_web_source(
            "https://medium.com/@author/kafka",
            title="Kafka article",
            publisher="Medium",
            search_snippet=snippet
        )

        self.assertEqual(source["metadata"]["excerpt"], snippet)
        self.assertEqual(
            source["metadata"]["extraction_mode"],
            "search_result_snippet"
        )
        self.assertEqual(source["url"], "https://medium.com/@author/kafka")
        mock_save_source.assert_called_once_with(source)

    @patch("app.study_workflow.requests.get")
    def test_import_does_not_use_snippet_for_non_403_http_errors(self, mock_get):
        response = SimpleNamespace(status_code=500)
        error = requests.HTTPError("500 Server Error", response=response)
        mock_get.return_value.raise_for_status.side_effect = error

        with self.assertRaisesRegex(RuntimeError, "Could not download"):
            import_web_source(
                "https://example.com/article",
                search_snippet="A search-result snippet that would be usable."
            )


class ManualQuestionTests(unittest.TestCase):
    @patch("app.study_workflow.save_study_question")
    @patch("app.study_workflow.save_study_source")
    def test_manual_question_has_local_reference_provenance(
        self,
        mock_save_source,
        mock_save_question
    ):
        question = add_manual_question(
            "Explain idempotent producer retries.",
            "Kafka",
            "Sequence numbers let the broker detect retries and prevent duplicates.",
            expected_concepts=["sequence numbers", "duplicate prevention"],
            hint="Think about what identifies a producer's request."
        )

        source = mock_save_source.call_args.args[0]
        self.assertEqual(question["provenance"], "manual")
        self.assertEqual(question["source_id"], source["id"])
        self.assertEqual(
            source["metadata"]["excerpt"],
            question["sample_answer"]
        )
        mock_save_question.assert_called_once_with(question)

    def test_manual_question_requires_a_reference_answer(self):
        with self.assertRaisesRegex(ValueError, "reference answer"):
            add_manual_question("Question?", "Topic", "")


class GroundedPracticeTests(unittest.TestCase):
    def test_local_json_response_accepts_array_output(self):
        with patch(
            "app.study_workflow.ask_ollama",
            return_value='{"data": [{"question": "Q"}]}'
        ):
            result = _local_json_response("prompt", list)
        self.assertEqual(result, [{"question": "Q"}])

    @patch("app.study_workflow.save_study_question")
    @patch("app.study_workflow.get_study_documents")
    @patch("app.study_workflow.ask_ollama")
    @patch("app.study_workflow.retrieve_study_passages")
    def test_generated_pdf_question_keeps_page_provenance(
        self,
        mock_retrieve,
        mock_ask,
        mock_documents,
        mock_save_question
    ):
        mock_retrieve.return_value = [{
            "text": "Producer retries use sequence numbers.",
            "file_path": "/study/kafka.pdf",
            "filename": "kafka.pdf",
            "page": 6
        }]
        mock_documents.return_value = [{
            "id": "document-id",
            "file_path": "/study/kafka.pdf"
        }]
        mock_ask.return_value = json.dumps([{
            "question": "Why are sequence numbers useful?",
            "topic": "idempotence",
            "difficulty": "medium",
            "expected_concepts": ["duplicate prevention"],
            "sample_answer": "They detect retries.",
            "evidence_id": "E1"
        }])

        generated = generate_pdf_questions("producer idempotence")

        self.assertEqual(generated[0]["document_id"], "document-id")
        self.assertEqual(generated[0]["page_number"], 6)
        self.assertEqual(generated[0]["provenance"], "pdf")
        mock_save_question.assert_called_once_with(generated[0])

    @patch("app.study_workflow.save_practice_attempt")
    @patch("app.study_workflow.get_study_questions", return_value=[])
    @patch("app.study_workflow.get_study_documents")
    @patch("app.study_workflow.retrieve_study_passages")
    @patch("app.study_workflow.ask_ollama")
    def test_answer_feedback_is_grounded_and_persisted(
        self,
        mock_ask,
        mock_retrieve,
        mock_documents,
        _mock_questions,
        mock_save_attempt
    ):
        mock_documents.return_value = [{
            "id": "doc-id",
            "file_path": "/study/kafka.pdf"
        }]
        mock_retrieve.return_value = [{
            "text": "Idempotence prevents duplicates after retries.",
            "filename": "kafka.pdf",
            "file_path": "/study/kafka.pdf",
            "page": 6
        }]
        mock_ask.return_value = json.dumps({
            "score": 82,
            "strengths": ["Explained retry safety"],
            "missing_concepts": ["sequence numbers"],
            "unsupported_claims": [],
            "feedback": "Good explanation.",
            "next_step": "Review sequence numbers."
        })
        question = {
            "id": "question-id",
            "question": "What does idempotence provide?",
            "provenance": "pdf",
            "document_id": "doc-id",
            "expected_concepts": ["duplicates"]
        }

        feedback = evaluate_study_answer(question, "It avoids duplicate messages.")

        self.assertEqual(feedback["score"], 82)
        self.assertIn("kafka.pdf, page 6", feedback["source_evidence"][0]["citation"])
        mock_save_attempt.assert_called_once()
        attempt = mock_save_attempt.call_args.args[0]
        self.assertEqual(attempt["score"], 82)
        self.assertTrue(attempt["next_review_at"])


class StudyDatabaseTests(unittest.TestCase):
    def test_database_connection_closes_and_rolls_back_on_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "connection.db"
            with _connection(db_path) as conn:
                conn.execute("CREATE TABLE values_table (value TEXT)")
                conn.execute("INSERT INTO values_table VALUES ('committed')")
            with self.assertRaises(sqlite3.ProgrammingError):
                conn.execute("SELECT 1")

            with self.assertRaisesRegex(RuntimeError, "rollback"):
                with _connection(db_path) as conn:
                    conn.execute(
                        "INSERT INTO values_table VALUES ('rolled back')"
                    )
                    raise RuntimeError("rollback")
            with sqlite3.connect(db_path) as conn:
                values = conn.execute(
                    "SELECT value FROM values_table"
                ).fetchall()

        self.assertEqual(values, [("committed",)])

    def test_documents_questions_notes_and_attempts_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "study.db"
            init_db(db_path)
            save_study_document({
                "id": "doc",
                "file_path": "/study/kafka.pdf",
                "filename": "kafka.pdf",
                "sha256": "hash",
                "page_count": 12,
                "status": "indexed"
            }, db_path)
            save_study_source({
                "id": "source",
                "source_type": "web",
                "title": "Source",
                "url": "https://example.com",
                "metadata": {"excerpt": "Article text"}
            }, db_path)
            save_study_question({
                "id": "question",
                "question": "Explain idempotence",
                "topic": "Kafka",
                "provenance": "pdf",
                "expected_concepts": ["retry safety"],
                "document_id": "doc",
                "page_number": 6
            }, db_path)
            due_before_answer = get_study_questions(
                due_only=True,
                db_path=db_path
            )
            save_study_note({
                "id": "note",
                "topic": "Kafka",
                "note_text": "Short cited notes.",
                "source_evidence": [{"citation": "kafka.pdf, page 6"}]
            }, db_path)
            save_practice_attempt({
                "id": "attempt",
                "question_id": "question",
                "session_id": "session",
                "answer": "It makes retries safe.",
                "score": 82,
                "feedback": {"feedback": "Good"},
                "source_evidence": [{"citation": "kafka.pdf, page 6"}],
                "next_review_at": "2999-01-01 00:00:00",
                "reason": "Spaced review"
            }, db_path)

            documents = get_study_documents(db_path)
            questions = get_study_questions(db_path=db_path)
            due_after_answer = get_study_questions(
                due_only=True,
                db_path=db_path
            )
            history = get_practice_history(db_path=db_path)
            notes = get_study_notes(db_path=db_path)
            with sqlite3.connect(db_path) as conn:
                session_count = conn.execute(
                    "SELECT COUNT(*) FROM practice_sessions"
                ).fetchone()[0]

        self.assertEqual(documents[0]["page_count"], 12)
        self.assertEqual([item["id"] for item in due_before_answer], ["question"])
        self.assertEqual(questions[0]["page_number"], 6)
        self.assertEqual(questions[0]["attempt_count"], 1)
        self.assertEqual(due_after_answer, [])
        self.assertEqual(history[0]["score"], 82)
        self.assertEqual(notes[0]["source_evidence"][0]["citation"], "kafka.pdf, page 6")
        self.assertEqual(session_count, 1)


class SplitterTests(unittest.TestCase):
    def test_splits_at_readable_boundaries_and_caps_chunk_size(self):
        text = "First sentence. " + ("word " * 200) + "Final sentence."

        chunks = split_text(text, chunk_size=100, overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertTrue(all(chunk.strip() for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
