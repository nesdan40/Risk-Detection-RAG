"""
Ollama HTTP client for the Streamlit demo UI.

Provides:
  - generate(prompt)          — call /api/generate and return the response text.
  - check_model_available()   — verify the LLM_MODEL is pulled on the Ollama server.

All configuration is read exclusively from src/config/settings.py.
No model names or endpoint URLs are hardcoded here.
"""

import requests

from config.settings import LLM_MODEL, OLLAMA_BASE_URL


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate(prompt: str) -> str:
    """
    Call the Ollama /api/generate endpoint with streaming and return the full response text.

    Uses streaming (stream=True) so the connection stays alive while the model
    generates, preventing timeout on longer responses.

    Args:
        prompt: The prompt string to send. Must be a non-empty, non-whitespace str.

    Returns:
        The complete generated response as a single string.

    Raises:
        ValueError:    If prompt is not a str, is empty, or is whitespace-only.
        TimeoutError:  If the HTTP connection times out.
        RuntimeError:  If the HTTP status is not 2xx or the response is malformed.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            "prompt must be a non-empty, non-whitespace string; "
            f"got {type(prompt).__name__!r}: {prompt!r}"
        )

    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": True,   # stream to avoid timeout on long responses
    }

    try:
        response = requests.post(url, json=payload, stream=True, timeout=120)
    except requests.Timeout:
        raise TimeoutError(f"Ollama request to {url} timed out.")
    except requests.ConnectionError as exc:
        raise RuntimeError(f"Failed to connect to Ollama at {url}: {exc}") from exc

    if not response.ok:
        raise RuntimeError(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")

    # Accumulate streamed token chunks
    import json as _json
    parts: list[str] = []
    try:
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = _json.loads(line)
            except ValueError:
                continue
            token = chunk.get("response", "")
            if token:
                parts.append(token)
            if chunk.get("done", False):
                break
    except requests.ChunkedEncodingError as exc:
        raise RuntimeError(f"Ollama stream was interrupted: {exc}") from exc

    result = "".join(parts)
    if not result:
        raise RuntimeError("Ollama returned an empty response.")
    return result


# ---------------------------------------------------------------------------
# Startup-check helper (not part of the public API)
# ---------------------------------------------------------------------------


def check_model_available() -> tuple[bool, str]:
    """
    Verify that the Ollama server is running and that LLM_MODEL is pulled.

    Sends a GET to ``{OLLAMA_BASE_URL}/api/tags`` with a 5-second timeout and
    checks whether LLM_MODEL appears in the returned models list.

    Returns:
        ``(True, "")``                              — server up, model present.
        ``(False, "Ollama not running")``           — server unreachable.
        ``(False, "Model not pulled. Run: ollama pull {LLM_MODEL}")``
                                                    — server up, model absent.
    """
    url = f"{OLLAMA_BASE_URL}/api/tags"

    try:
        response = requests.get(url, timeout=5)
    except (requests.ConnectionError, requests.Timeout):
        return (False, "Ollama not running")

    if not response.ok:
        return (False, "Ollama not running")

    try:
        data = response.json()
    except ValueError:
        return (False, "Ollama not running")

    models = data.get("models", [])
    model_names = [m.get("name", "") for m in models if isinstance(m, dict)]

    if LLM_MODEL in model_names:
        return (True, "")

    return (False, f"Model not pulled. Run: ollama pull {LLM_MODEL}")
