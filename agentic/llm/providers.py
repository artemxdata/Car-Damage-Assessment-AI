from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol

import httpx


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class OpenAICompatibleProvider:
    """
    Minimal OpenAI-compatible chat/completions client.
    Works with:
      - OpenAI
      - any OpenAI-compatible gateway (DeepSeek, Groq, etc)
    via env:
      LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
    """
    base_url: str
    api_key: str
    model: str = "gpt-4o-mini"
    timeout_s: float = 30.0

    def generate(self, prompt: str) -> str:
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You are a senior vehicle damage assessor. Be concise and practical."},
                {"role": "user", "content": prompt},
            ],
        }

        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()

        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return ""


@dataclass
class NoopProvider:
    """Fallback when no LLM configured."""
    def generate(self, prompt: str) -> str:
        return ""


def build_provider_from_env() -> LLMProvider:
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()

    if base_url and api_key:
        return OpenAICompatibleProvider(base_url=base_url, api_key=api_key, model=model)

    return NoopProvider()
