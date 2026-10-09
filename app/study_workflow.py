import hashlib
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
import pymupdf
import trafilatura
from ddgs import DDGS

from app.config import get_settings
from app.database.db import (
    delete_study_document,
    get_study_documents,
    get_study_questions,
    get_study_source,
    get_study_sources,
    save_practice_attempt,
    save_study_document,
    save_study_note,
    save_study_question,
    save_study_source
)
from app.llm.ollama_client import ask_ollama, is_ollama_error
from app.rag.rag_engine import (
    get_study_collection,
    index_study_pdfs,
    retrieve_study_passages
)
from app.utils.json_utils import extract_json


MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024
MAX_SEARCH_RESULTS = 8
MAX_SOURCE_EXCERPT = 12000


def _new_id():
    return str(uuid.uuid4())


def _safe_pdf_name(filename):
    name = Path(str(filename)).name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    if not name or Path(name).suffix.lower() != ".pdf":
        raise ValueError("Only files with a .pdf extension can be added.")
    return name


def add_pdf_files(uploaded_files, directory=None):
    directory = Path(directory or get_settings().pdf_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    added = []
    unchanged = []

    for uploaded_file in uploaded_files:
        name = _safe_pdf_name(uploaded_file.name)
        file_bytes = uploaded_file.getvalue()
        if not file_bytes:
            raise ValueError(f"'{name}' is empty.")
        if len(file_bytes) > MAX_PDF_SIZE_BYTES:
            raise ValueError(
                f"'{name}' is larger than the 50 MB per-PDF limit."
            )
        try:
            with pymupdf.open(stream=file_bytes, filetype="pdf") as document:
                if document.needs_pass:
                    raise ValueError(f"'{name}' is password-protected.")
                if document.page_count == 0:
                    raise ValueError(f"'{name}' does not contain any pages.")
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"'{name}' is not a readable PDF: {error}") from error

        file_hash = hashlib.sha256(file_bytes).hexdigest()
        duplicate = next(
            (
                path for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() == ".pdf"
                and hashlib.sha256(path.read_bytes()).hexdigest() == file_hash
            ),
            None
        )
        if duplicate:
            unchanged.append(duplicate.name)
            continue

        target = directory / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != file_hash:
            stem = target.stem
            target = directory / f"{stem}-{file_hash[:8]}.pdf"
        temporary_target = target.with_name(f".{target.name}.uploading")
        try:
            temporary_target.write_bytes(file_bytes)
            temporary_target.replace(target)
        finally:
            temporary_target.unlink(missing_ok=True)
        added.append(target)

    result = sync_pdf_library(directory)
    result["uploaded"] = [path.name for path in added]
    result["duplicates"] = unchanged
    return result


def sync_pdf_library(directory=None):
    directory = Path(directory or get_settings().pdf_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    result = index_study_pdfs(directory)
    indexed_by_path = {
        item["file_path"]: item for item in result.get("documents", [])
    }

    for document in indexed_by_path.values():
        save_study_document({
            **document,
            "id": _new_id()
        })

    present = {
        str(path.resolve())
        for path in directory.iterdir()
        if path.suffix.lower() == ".pdf"
        and path.is_file()
    }
    for document in get_study_documents():
        if (
            Path(document["file_path"]).resolve().parent == directory
            and document["file_path"] not in present
        ):
            delete_study_document(document["id"])
    return result


def remove_pdf(document_id):
    document = next(
        (item for item in get_study_documents() if item["id"] == document_id),
        None
    )
    if document is None:
        raise ValueError("The selected PDF is no longer in the study library.")

    library_dir = Path(get_settings().pdf_directory).expanduser().resolve()
    file_path = Path(document["file_path"]).resolve()
    if file_path.parent != library_dir or file_path.suffix.lower() != ".pdf":
        raise RuntimeError("Refusing to remove a PDF outside the configured library.")

    collection = get_study_collection()
    stored = collection.get(
        where={"file_path": str(file_path)},
        include=[]
    )
    if stored.get("ids"):
        collection.delete(ids=stored["ids"])
    file_path.unlink(missing_ok=True)
    delete_study_document(document_id)


def search_public_questions(query, max_results=5):
    if not query or not query.strip():
        raise ValueError("Enter a topic to search.")
    result_count = max(1, min(int(max_results), MAX_SEARCH_RESULTS))
    try:
        raw_results = DDGS().text(query, max_results=result_count)
    except Exception as error:
        raise RuntimeError(
            f"Public web search failed. Your saved library remains available offline: {error}"
        ) from error

    results = []
    for item in raw_results:
        url = item.get("href") or item.get("url") or ""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        results.append({
            "title": item.get("title") or url,
            "url": url,
            "publisher": parsed.netloc,
            "snippet": item.get("body") or item.get("snippet") or ""
        })
    return results


def import_web_source(url, title="", publisher="", search_snippet=""):
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Use a complete public http:// or https:// URL.")
    existing_source = next(
        (source for source in get_study_sources("web")
         if source["url"] == parsed.geturl()),
        None
    )
    if existing_source:
        return existing_source

    extraction_mode = "article"
    try:
        response = requests.get(
            parsed.geturl(),
            headers={"User-Agent": "LocalInterviewStudyAssistant/1.0"},
            timeout=(5, 20)
        )
        response.raise_for_status()
    except requests.HTTPError as error:
        snippet = (search_snippet or "").strip()
        if (
            error.response is None
            or error.response.status_code != 403
            or len(snippet) < 40
        ):
            raise RuntimeError(
                f"Could not download the web source: {error}"
            ) from error
        extracted = snippet
        metadata = None
        extraction_mode = "search_result_snippet"
    except requests.RequestException as error:
        raise RuntimeError(f"Could not download the web source: {error}") from error

    if extraction_mode == "article":
        if len(response.content) > 5 * 1024 * 1024:
            raise ValueError("The page is larger than the 5 MB import limit.")
        extracted = trafilatura.extract(
            response.text,
            url=parsed.geturl(),
            include_comments=False,
            include_tables=False
        )
        if not extracted or len(extracted.strip()) < 150:
            raise ValueError(
                "Could not extract enough readable article text from this URL."
            )
        metadata = trafilatura.extract_metadata(
            response.text,
            default_url=parsed.geturl()
        )

    source_title = title or (metadata.title if metadata else "") or parsed.netloc
    source_publisher = (
        publisher or (metadata.sitename if metadata else "") or parsed.netloc
    )
    source = {
        "id": _new_id(),
        "source_type": "web",
        "title": source_title,
        "url": parsed.geturl(),
        "publisher": source_publisher,
        "metadata": {
            "excerpt": extracted[:MAX_SOURCE_EXCERPT],
            "extraction_mode": extraction_mode,
            "retrieved_at": datetime.now(timezone.utc).isoformat()
        }
    }
    save_study_source(source)
    return source


def add_manual_question(question_text, topic, reference_answer, expected_concepts=None, hint=""):
    question_text = (question_text or "").strip()
    topic = (topic or "").strip()
    reference_answer = (reference_answer or "").strip()
    if not question_text or not topic or not reference_answer:
        raise ValueError(
            "A question, topic, and reference answer are required."
        )

    source_id = _new_id()
    source = {
        "id": source_id,
        "source_type": "manual",
        "title": topic,
        "publisher": "Personal notes",
        "metadata": {"excerpt": reference_answer}
    }
    save_study_source(source)
    question = {
        "id": _new_id(),
        "question": question_text,
        "topic": topic,
        "question_type": "fundamentals",
        "difficulty": "medium",
        "provenance": "manual",
        "expected_concepts": expected_concepts or [],
        "hint": hint.strip(),
        "sample_answer": reference_answer,
        "source_id": source_id
    }
    save_study_question(question)
    return question


def _format_evidence(passages):
    formatted = []
    for index, passage in enumerate(passages, start=1):
        evidence = dict(passage)
        evidence["evidence_id"] = f"E{index}"
        evidence["citation"] = (
            f"{passage['filename']}, page {passage['page']}"
        )
        formatted.append(evidence)
    return formatted


def _local_json_response(prompt, expected_type):
    response = ask_ollama(prompt)
    if is_ollama_error(response):
        raise RuntimeError(response)
    if response.strip() == "No response received from Ollama.":
        raise RuntimeError("The local model returned no response.")
    try:
        parsed = json.loads(response)
    except (TypeError, json.JSONDecodeError):
        array_match = re.search(r"\[.*\]", response, re.DOTALL)
        if array_match:
            try:
                parsed = json.loads(array_match.group())
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None
    if parsed is None:
        parsed = extract_json(response)
    if expected_type is list and isinstance(parsed, dict):
        parsed = next(
            (
                parsed[key] for key in ("questions", "data", "items")
                if isinstance(parsed.get(key), list)
            ),
            parsed
        )
    if not isinstance(parsed, expected_type):
        raise RuntimeError("The local model returned an invalid JSON response.")
    return parsed


def generate_pdf_questions(topic, file_paths=None, count=5):
    passages = retrieve_study_passages(topic, top_k=8, file_paths=file_paths)
    evidence = _format_evidence(passages)
    if not evidence:
        raise ValueError("No matching PDF passages were found for this topic.")

    prompt = f"""
Create {max(1, min(int(count), 8))} distinct interview practice questions about
{topic!r}, using only the supplied PDF evidence. Do not use outside knowledge.
For every question choose one evidence_id from the supplied evidence list.
Return only a JSON array. Each item must contain:
question, topic, question_type, difficulty (easy/medium/hard),
expected_concepts (array of short strings), hint, sample_answer, evidence_id.
Questions should test understanding and application, not ask the learner to
recite the passage. Keep sample answers concise and grounded.

PDF evidence:
{json.dumps(evidence, ensure_ascii=False)}
""".strip()
    generated = _local_json_response(prompt, list)
    documents = {item["file_path"]: item for item in get_study_documents()}
    saved = []
    for item in generated:
        if not isinstance(item, dict):
            continue
        question_text = str(item.get("question", "")).strip()
        matched = next(
            (entry for entry in evidence
             if entry["evidence_id"] == item.get("evidence_id")),
            None
        )
        if not question_text or matched is None:
            continue
        document = documents.get(matched["file_path"])
        question = {
            "id": _new_id(),
            "question": question_text,
            "topic": str(item.get("topic") or topic).strip(),
            "question_type": str(item.get("question_type") or "fundamentals"),
            "difficulty": str(item.get("difficulty") or "medium").lower(),
            "provenance": "pdf",
            "expected_concepts": item.get("expected_concepts", []),
            "hint": str(item.get("hint") or ""),
            "sample_answer": str(item.get("sample_answer") or ""),
            "document_id": document["id"] if document else None,
            "page_number": int(matched["page"])
        }
        save_study_question(question)
        saved.append(question)
    if not saved:
        raise RuntimeError(
            "The model did not return questions tied to valid PDF evidence. "
            "Try generating again with a narrower topic."
        )
    return saved


def generate_web_questions(source_id, count=5):
    source = get_study_source(source_id)
    if source is None or source["source_type"] != "web":
        raise ValueError("Choose an imported web source.")
    excerpt = source["metadata"].get("excerpt", "")
    if not excerpt:
        raise ValueError("This saved source has no readable excerpt.")
    prompt = f"""
Create {max(1, min(int(count), 8))} distinct interview questions grounded in
the saved article excerpt. Do not claim unsupported facts. Return only a JSON
array with question, topic, question_type, difficulty (easy/medium/hard),
expected_concepts (array), hint, and sample_answer.

Article: {source['title']}
Excerpt:
{excerpt}
""".strip()
    generated = _local_json_response(prompt, list)
    saved = []
    for item in generated:
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            continue
        question = {
            "id": _new_id(),
            "question": str(item["question"]).strip(),
            "topic": str(item.get("topic") or source["title"]).strip(),
            "question_type": str(item.get("question_type") or "fundamentals"),
            "difficulty": str(item.get("difficulty") or "medium").lower(),
            "provenance": "web",
            "expected_concepts": item.get("expected_concepts", []),
            "hint": str(item.get("hint") or ""),
            "sample_answer": str(item.get("sample_answer") or ""),
            "source_id": source_id
        }
        save_study_question(question)
        saved.append(question)
    if not saved:
        raise RuntimeError("The local model did not return valid web questions.")
    return saved


def generate_pdf_notes(topic, file_paths=None):
    passages = retrieve_study_passages(topic, top_k=8, file_paths=file_paths)
    evidence = _format_evidence(passages)
    if not evidence:
        raise ValueError("No matching PDF passages were found for this topic.")
    prompt = f"""
Write concise interview revision notes about {topic!r} using only this PDF
evidence. Organize as a short definition, key points, and one practical
interview reminder. Add inline citations like [E1] matching the evidence IDs.
If the evidence is insufficient for part of the topic, say so rather than
filling gaps from outside knowledge.

PDF evidence:
{json.dumps(evidence, ensure_ascii=False)}
""".strip()
    response = ask_ollama(prompt)
    if is_ollama_error(response):
        raise RuntimeError(response)
    if response.strip() == "No response received from Ollama.":
        raise RuntimeError("The local model returned no response.")
    note = {
        "id": _new_id(),
        "topic": topic,
        "note_text": response.strip(),
        "source_evidence": [
            {
                "citation": item["citation"],
                "file_path": item["file_path"],
                "page": item["page"]
            }
            for item in evidence
        ]
    }
    save_study_note(note)
    return note


def evaluate_study_answer(question, answer):
    if not answer or not answer.strip():
        raise ValueError("Write or transcribe an answer before submitting.")

    evidence = []
    if question.get("provenance") == "pdf" and question.get("document_id"):
        docs = {
            item["id"]: item for item in get_study_documents()
        }
        document = docs.get(question["document_id"])
        if document:
            evidence = retrieve_study_passages(
                question["question"],
                top_k=5,
                file_paths=[document["file_path"]]
            )
    elif question.get("source_id"):
        source = get_study_source(question["source_id"])
        if source:
            excerpt = source["metadata"].get("excerpt", "")
            evidence = [{
                "text": excerpt,
                "filename": (
                    f"Manual reference — {source['title']}"
                    if source["source_type"] == "manual"
                    else source["title"]
                ),
                "file_path": source["url"],
                "page": (
                    "manual" if source["source_type"] == "manual" else "web"
                )
            }] if excerpt else []
    if not evidence and question.get("sample_answer"):
        evidence = [{
            "text": question["sample_answer"],
            "filename": question.get("source_title") or question.get("filename") or "Saved answer",
            "file_path": question.get("source_url", ""),
            "page": question.get("page_number") or "?"
        }]
    if not evidence:
        raise RuntimeError(
            "No saved source evidence is available for this question, so it "
            "cannot be evaluated as source-grounded."
        )

    prompt = f"""
Evaluate the candidate answer only against the question and supplied source
evidence. Be fair: accept equivalent terminology, identify unsupported claims,
and do not require wording copied from the source. Return only JSON with:
score (integer 0-100), strengths (array), missing_concepts (array),
unsupported_claims (array), feedback (string), next_step (string).

Question: {question['question']}
Expected concepts: {json.dumps(question.get('expected_concepts', []))}
Candidate answer: {answer}
Source evidence: {json.dumps(evidence, ensure_ascii=False)}
""".strip()
    result = _local_json_response(prompt, dict)
    score = result.get("score")
    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
        raise RuntimeError("The local model returned an invalid answer score.")
    result.setdefault("strengths", [])
    result.setdefault("missing_concepts", [])
    result.setdefault("unsupported_claims", [])
    result.setdefault("feedback", "")
    result.setdefault("next_step", "")

    schedule = get_study_questions()
    previous = next(
        (item for item in schedule if item["id"] == question["id"]),
        None
    )
    attempt_count = (previous["attempt_count"] if previous else 0) + 1
    intervals = (
        [1, 3, 7, 14, 30] if score >= 90 else
        [1, 3, 7, 14] if score >= 75 else
        [1, 2, 4] if score >= 60 else
        [1]
    )
    days = intervals[min(attempt_count - 1, len(intervals) - 1)]
    next_review_at = (
        datetime.now(timezone.utc) + timedelta(days=days)
    ).strftime("%Y-%m-%d %H:%M:%S")
    evidence_citations = [
        {
            "citation": (
                f"{item.get('filename', 'unknown')} "
                f"({item.get('file_path', '')})"
                if item.get("page") == "web"
                else f"{item.get('filename', 'unknown')} "
                     "(saved offline reference)"
                if item.get("page") == "manual"
                else f"{item.get('filename', 'unknown')}, "
                     f"page {item.get('page', '?')}"
            ),
            "file_path": item.get("file_path", "")
        }
        for item in evidence
    ]
    save_practice_attempt({
        "id": _new_id(),
        "question_id": question["id"],
        "session_id": _new_id(),
        "answer": answer.strip(),
        "score": score,
        "score_breakdown": {
            "strengths": result["strengths"],
            "missing_concepts": result["missing_concepts"],
            "unsupported_claims": result["unsupported_claims"]
        },
        "feedback": result,
        "source_evidence": evidence_citations,
        "next_review_at": next_review_at,
        "reason": (
            "Review fundamentals and missing concepts"
            if score < 75 else "Spaced review"
        ),
        "extra_revision": score < 60
    })
    return {
        **result,
        "source_evidence": evidence_citations,
        "next_review_at": next_review_at
    }


def get_saved_web_sources():
    return get_study_sources("web")


def get_question_bank(due_only=False):
    return get_study_questions(due_only=due_only)
