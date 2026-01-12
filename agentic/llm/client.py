from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import os
import json
import requests

try:
    # Optional: load .env in local dev
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool
    provider: str
    base_url: str
    api_key: str
    model: str
    timeout_s: int = 30


def _env_bool(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def load_llm_config() -> Optional[LLMConfig]:
    """
    Reads env vars (optionally from .env via python-dotenv).
    Expected vars:
      LLM_ENABLED=0/1
      LLM_PROVIDER=openai
      LLM_BASE_URL=https://api.proxyapi.ru/openai/v1   (OpenAI-compatible)
      LLM_API_KEY=...
      LLM_MODEL=gpt-4o-mini
      LLM_TIMEOUT_S=30
    """
    enabled = _env_bool("LLM_ENABLED", "0")
    if not enabled:
        return None

    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    base_url = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    timeout_s = int(os.getenv("LLM_TIMEOUT_S", "30").strip() or "30")

    if not base_url or not api_key:
        # misconfigured -> disable
        return None

    return LLMConfig(
        enabled=True,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_s=timeout_s,
    )


class OpenAICompatClient:
    """
    Minimal OpenAI-compatible client using requests.
    Works with ProxyAPI that mirrors OpenAI endpoints:
      POST {base_url}/chat/completions
    """
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def chat(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        url = f"{self.cfg.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.cfg.timeout_s)
        r.raise_for_status()
        data = r.json()

        # OpenAI-like: choices[0].message.content
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            # Fallback: return raw json for debugging
            return json.dumps(data, ensure_ascii=False, indent=2)


def get_llm_client(cfg: LLMConfig) -> OpenAICompatClient:
    # For future: switch by provider
    return OpenAICompatClient(cfg)
