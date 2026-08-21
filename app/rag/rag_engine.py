from pathlib import Path

import chromadb
import requests

from app.rag.text_splitter import split_text


CHROMA_PATH = Path("app/rag/chroma_db")
COLLECTION_NAME = "interview_context"

OLLAMA_EMBED_API_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL_NAME = "nomic-embed-text"


def get_embedding(text):
    if not text or not str(text).strip():
        raise RuntimeError("Cannot create an embedding for empty text.")

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
            OLLAMA_EMBED_API_URL,
            json={"model": EMBEDDING_MODEL_NAME, "input": text},
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        embedding = _parse_embedding(data)

        if embedding:
            return embedding

        raise RuntimeError("No embedding returned from Ollama.")
    except requests.exceptions.ConnectionError as error:
        raise RuntimeError(
            f"Ollama is not running. Could not connect to {OLLAMA_EMBED_API_URL}.\n"
            f"Please start Ollama by running 'ollama serve' in your terminal and ensure model '{EMBEDDING_MODEL_NAME}' is pulled ('ollama pull {EMBEDDING_MODEL_NAME}')."
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