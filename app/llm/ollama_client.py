import requests

from app.config import get_settings


def check_ollama_status(timeout=5):
    settings = get_settings()
    status = {
        "service_available": False,
        "chat_model": settings.chat_model,
        "chat_model_available": False,
        "embedding_model": settings.embedding_model,
        "embedding_model_available": False,
        "error": ""
    }

    try:
        response = requests.get(
            f"{settings.ollama_base_url}/api/tags",
            timeout=timeout
        )
        response.raise_for_status()
        models = response.json().get("models", [])
    except requests.exceptions.ConnectionError:
        status["error"] = (
            "Ollama is not running. Start the Ollama app or run `ollama serve`."
        )
        return status
    except requests.exceptions.Timeout:
        status["error"] = "Ollama did not respond before the status-check timeout."
        return status
    except requests.exceptions.RequestException as error:
        status["error"] = f"Could not check Ollama: {error}"
        return status
    except ValueError:
        status["error"] = "Ollama returned an invalid model-list response."
        return status

    installed_models = {
        model_name
        for model in models
        if (model_name := model.get("name") or model.get("model"))
    }
    status["service_available"] = True
    status["chat_model_available"] = settings.chat_model in installed_models
    status["embedding_model_available"] = (
        settings.embedding_model in installed_models
    )
    return status


def ask_ollama(prompt):
    settings = get_settings()
    payload = {
        "model": settings.chat_model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "10m",
        "options": {
            "temperature": 0.1,
            "num_predict": 1024,
            "num_ctx": 4096
        }
    }

    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/generate",
            json=payload,
            timeout=settings.ollama_timeout_seconds
        )
        response.raise_for_status()
        data = response.json()

        return data.get("response", "No response received from Ollama.")

    except requests.exceptions.ConnectionError:
        return (
            f"Error: Ollama is not running. Start it and ensure "
            f"'{settings.chat_model}' is installed."
        )

    except requests.exceptions.Timeout:
        return (
            f"Error: Ollama timed out while using {settings.chat_model}. "
            "Try a shorter prompt or wait for the model to warm up."
        )

    except Exception as e:
        return f"Error: {str(e)}"


def is_ollama_error(text):
    if not text:
        return False

    normalized = str(text).strip().lower()
    return normalized.startswith("error:") or "ollama timed out" in normalized or "ollama is not running" in normalized