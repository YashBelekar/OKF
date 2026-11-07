import re

import requests

from config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_CONTEXT_TOKENS,
    OLLAMA_MAX_TOKENS,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_SECONDS,
)


class LLMError(Exception):
    """Raised with a user-friendly message when the local LLM cannot answer."""


def strip_thinking(text):
    """Remove <think>...</think> blocks emitted by reasoning models such as Qwen3."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)  # unbalanced closer
    return text.strip()


class LLMClient:
    def __init__(self, base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL,
                 temperature=OLLAMA_TEMPERATURE, max_tokens=OLLAMA_MAX_TOKENS,
                 context_tokens=OLLAMA_CONTEXT_TOKENS, timeout=OLLAMA_TIMEOUT_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_tokens = context_tokens
        self.timeout = timeout

    def is_available(self):
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=3).raise_for_status()
            return True
        except requests.RequestException:
            return False

    def list_models(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except (requests.RequestException, ValueError, KeyError):
            return []

    def generate(self, system_prompt, user_prompt, json_mode=False):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": self.context_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
            )
        except requests.ConnectionError as error:
            raise LLMError(
                "Cannot reach Ollama at "
                f"{self.base_url}. Start it (open the Ollama app or run 'ollama serve')."
            ) from error
        except requests.Timeout as error:
            raise LLMError("The local model timed out. Try a smaller model or fewer sources.") from error

        if response.status_code == 404:
            raise LLMError(
                f"Model '{self.model}' is not installed. Run: ollama pull {self.model}"
            )
        try:
            response.raise_for_status()
            content = response.json()["message"]["content"]
        except (requests.RequestException, KeyError, ValueError) as error:
            raise LLMError(f"Unexpected response from Ollama: {error}") from error
        return strip_thinking(content)
