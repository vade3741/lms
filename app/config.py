import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    chat_model: str
    embedding_model: str
    ollama_timeout_seconds: int
    embedding_timeout_seconds: int
    sqlite_db_path: Path
    chroma_path: Path
    pdf_directory: Path


def get_settings(environ=None):
    environ = os.environ if environ is None else environ

    return Settings(
        ollama_base_url=environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        ).rstrip("/"),
        chat_model=environ.get("OLLAMA_CHAT_MODEL", "qwen2.5-coder:7b"),
        embedding_model=environ.get(
            "OLLAMA_EMBEDDING_MODEL", "embeddinggemma:latest"
        ),
        ollama_timeout_seconds=int(environ.get("OLLAMA_TIMEOUT_SECONDS", "180")),
        embedding_timeout_seconds=int(
            environ.get("OLLAMA_EMBEDDING_TIMEOUT_SECONDS", "120")
        ),
        sqlite_db_path=Path(
            environ.get(
                "AI_INTERVIEW_DB_PATH",
                PROJECT_ROOT / "app" / "database" / "interviews.db"
            )
        ).expanduser(),
        chroma_path=Path(
            environ.get(
                "AI_INTERVIEW_CHROMA_PATH",
                PROJECT_ROOT / "app" / "rag" / "chroma_db"
            )
        ).expanduser(),
        pdf_directory=Path(
            environ.get("AI_INTERVIEW_PDF_DIRECTORY", PROJECT_ROOT / "pdf")
        ).expanduser()
    )
