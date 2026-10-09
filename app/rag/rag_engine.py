import hashlib
from pathlib import Path

import chromadb
import requests

from app.config import get_settings
from app.rag.text_splitter import split_text
from app.utils.file_reader import extract_pdf_pages


SETTINGS = get_settings()
CHROMA_PATH = SETTINGS.chroma_path
COLLECTION_NAME = "interview_context"
STUDY_COLLECTION_NAME = "study_materials"
PDF_DIRECTORY = SETTINGS.pdf_directory


def get_embedding(text):
    if not text or not str(text).strip():
        raise RuntimeError("Cannot create an embedding for empty text.")

    settings = get_settings()
    def _parse_embedding(data):
        embedding = data.get("embedding")
        if embedding:
            return embedding

        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            first_embedding = embeddings[0]
            if isinstance(first_embedding, list):
                return first_embedding

        return []

    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/embed",
            json={"model": settings.embedding_model, "input": text},
            timeout=settings.embedding_timeout_seconds
        )
        response.raise_for_status()
        data = response.json()
        embedding = _parse_embedding(data)

        if embedding:
            return embedding

        raise RuntimeError("No embedding returned from Ollama.")
    except requests.exceptions.ConnectionError as error:
        raise RuntimeError(
            f"Ollama is not running. Could not connect to "
            f"{settings.ollama_base_url}.\n"
            f"Please start Ollama and ensure model '{settings.embedding_model}' "
            f"is installed ('ollama pull {settings.embedding_model}')."
        ) from error
    except requests.exceptions.HTTPError as error:
        raise RuntimeError(
            f"Embedding request failed with status {response.status_code}: {response.text}"
        ) from error
    except Exception as error:
        raise RuntimeError(f"Embedding error: {str(error)}")


def get_chroma_client():
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_study_collection():
    settings = get_settings()
    model_key = hashlib.sha256(
        settings.embedding_model.encode("utf-8")
    ).hexdigest()[:12]
    collection_name = f"{STUDY_COLLECTION_NAME}_{model_key}"
    collection = get_chroma_client().get_or_create_collection(
        name=collection_name,
        metadata={"embedding_model": settings.embedding_model}
    )
    if (collection.metadata or {}).get("embedding_model") != settings.embedding_model:
        raise RuntimeError(
            f"Embedding-model metadata mismatch in Chroma collection "
            f"'{collection_name}'. Use a different collection before indexing."
        )
    return collection


def reset_collection():
    client = get_chroma_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    return client.get_or_create_collection(name=COLLECTION_NAME)


def build_rag_index(resume_text, job_description_text):
    collection = reset_collection()

    documents = []
    metadatas = []
    ids = []
    embeddings = []

    resume_chunks = split_text(resume_text)
    jd_chunks = split_text(job_description_text)

    counter = 1

    for chunk in resume_chunks:
        documents.append(chunk)
        metadatas.append({"source": "resume"})
        ids.append(f"resume_{counter}")
        embeddings.append(get_embedding(chunk))
        counter += 1

    counter = 1

    for chunk in jd_chunks:
        documents.append(chunk)
        metadatas.append({"source": "job_description"})
        ids.append(f"jd_{counter}")
        embeddings.append(get_embedding(chunk))
        counter += 1

    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )

    return len(documents)


def retrieve_context(query, top_k=5):
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    try:
        query_embedding = get_embedding(query)
    except Exception:
        return ""

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context_parts = []

    for doc, meta in zip(documents, metadatas):
        source = meta.get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{doc}")

    return "\n\n".join(context_parts)


def index_study_pdfs(directory=PDF_DIRECTORY):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise RuntimeError(f"PDF study folder does not exist: {directory}")

    collection = get_study_collection()
    pdf_paths = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.name.lower()
    )
    present_paths = {str(path.resolve()) for path in pdf_paths}
    indexed = 0
    unchanged = 0
    errors = []
    document_details = []

    for pdf_path in pdf_paths:
        path_key = str(pdf_path.resolve())
        file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        existing = collection.get(
            where={"file_path": path_key},
            include=["metadatas"]
        )
        existing_ids = existing.get("ids", [])
        existing_metadatas = existing.get("metadatas", [])

        if existing_metadatas and all(
            metadata.get("file_hash") == file_hash
            for metadata in existing_metadatas
        ):
            unchanged += 1
            try:
                page_count = len(extract_pdf_pages(pdf_path))
            except RuntimeError as error:
                errors.append(str(error))
                document_details.append({
                    "file_path": path_key,
                    "filename": pdf_path.name,
                    "sha256": file_hash,
                    "page_count": 0,
                    "status": "error",
                    "error": str(error)
                })
                continue
            document_details.append({
                "file_path": path_key,
                "filename": pdf_path.name,
                "sha256": file_hash,
                "page_count": page_count,
                "status": "indexed",
                "error": ""
            })
            continue

        try:
            pages = extract_pdf_pages(pdf_path)
        except RuntimeError as error:
            if existing_ids:
                collection.delete(ids=existing_ids)
            errors.append(str(error))
            document_details.append({
                "file_path": path_key,
                "filename": pdf_path.name,
                "sha256": file_hash,
                "page_count": 0,
                "status": "error",
                "error": str(error)
            })
            continue

        documents = []
        metadatas = []
        ids = []
        embeddings = []

        for page_number, page_text in enumerate(pages, start=1):
            for chunk_number, chunk in enumerate(split_text(page_text), start=1):
                chunk_id = hashlib.sha256(
                    f"{path_key}:{file_hash}:{page_number}:{chunk_number}".encode("utf-8")
                ).hexdigest()
                documents.append(chunk)
                metadatas.append({
                    "source": "study_pdf",
                    "file_path": path_key,
                    "directory": str(directory),
                    "filename": pdf_path.name,
                    "page": page_number,
                    "file_hash": file_hash
                })
                ids.append(chunk_id)
                try:
                    embeddings.append(get_embedding(chunk))
                except RuntimeError as error:
                    raise RuntimeError(
                        f"Could not index '{pdf_path.name}': {error}"
                    ) from error

        if not documents:
            if existing_ids:
                collection.delete(ids=existing_ids)
            error = (
                f"No extractable text found in '{pdf_path.name}'. "
                "It may be scanned; OCR support is not available yet."
            )
            errors.append(error)
            document_details.append({
                "file_path": path_key,
                "filename": pdf_path.name,
                "sha256": file_hash,
                "page_count": len(pages),
                "status": "error",
                "error": error
            })
            continue

        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        obsolete_ids = [chunk_id for chunk_id in existing_ids if chunk_id not in ids]
        if obsolete_ids:
            collection.delete(ids=obsolete_ids)
        indexed += 1
        document_details.append({
            "file_path": path_key,
            "filename": pdf_path.name,
            "sha256": file_hash,
            "page_count": len(pages),
            "status": "indexed",
            "error": ""
        })

    stored = collection.get(
        where={"source": "study_pdf"},
        include=["metadatas"]
    )
    stale_ids = [
        chunk_id
        for chunk_id, metadata in zip(stored.get("ids", []), stored.get("metadatas", []))
        if metadata.get("directory") == str(directory)
        and metadata.get("file_path") not in present_paths
    ]
    if stale_ids:
        collection.delete(ids=stale_ids)

    return {
        "indexed": indexed,
        "unchanged": unchanged,
        "removed": len(stale_ids),
        "pdf_count": len(pdf_paths),
        "errors": errors,
        "documents": document_details
    }


def get_study_pdf_library():
    collection = get_study_collection()
    stored = collection.get(
        where={"source": "study_pdf"},
        include=["metadatas"]
    )

    library = {}
    for metadata in stored.get("metadatas", []):
        path = metadata.get("file_path")
        if path:
            library[path] = {
                "filename": metadata.get("filename", Path(path).name),
                "file_path": path
            }

    return sorted(library.values(), key=lambda item: item["filename"].lower())


def retrieve_study_context(query, top_k=5, file_paths=None):
    passages = retrieve_study_passages(query, top_k=top_k, file_paths=file_paths)
    return "\n\n".join(
        f"[Source: {passage['filename']}, page {passage['page']}]\n"
        f"{passage['text']}"
        for passage in passages
    )


def retrieve_study_passages(query, top_k=5, file_paths=None):
    if not query or not str(query).strip():
        return []

    collection = get_study_collection()
    if collection.count() == 0:
        return []

    where = {"source": "study_pdf"}
    if file_paths:
        where = {
            "$and": [
                {"source": "study_pdf"},
                {"file_path": {"$in": list(file_paths)}}
            ]
        }

    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    return [
        {
            "text": document,
            "file_path": metadata.get("file_path", ""),
            "filename": metadata.get("filename", "unknown"),
            "page": metadata.get("page", "?")
        }
        for document, metadata in zip(documents, metadatas)
    ]