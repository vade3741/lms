import requests


OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "gemma2:2b"
MODEL_GUIDANCE = "Try gemma2:2b for speed, or qwen2.5-coder:3b if you need stronger coding output."


def ask_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
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
            OLLAMA_GENERATE_URL,
            json=payload,
            timeout=180
        )
        response.raise_for_status()
        data = response.json()

        return data.get("response", "No response received from Ollama.")

    except requests.exceptions.ConnectionError:
        return f"Error: Ollama is not running. Run: ollama run {MODEL_NAME}. {MODEL_GUIDANCE}"

    except requests.exceptions.Timeout:
        return (
            f"Error: Ollama timed out while using {MODEL_NAME}. "
            f"Try a shorter prompt, wait for the model to warm up, or switch models. {MODEL_GUIDANCE}"
        )

    except Exception as e:
        return f"Error: {str(e)}"


def is_ollama_error(text):
    if not text:
        return False

    normalized = str(text).strip().lower()
    return normalized.startswith("error:") or "ollama timed out" in normalized or "ollama is not running" in normalized