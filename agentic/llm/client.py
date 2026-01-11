from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError as e:
    raise RuntimeError(
        "python-dotenv package is required to load .env. "
        "Install with: pip install python-dotenv"
    ) from e

try:
    from openai import OpenAI
except ImportError as e:
    raise RuntimeError(
        "openai package is required for LLM support. "
        "Install with: pip install openai"
    ) from e


# Load .env from current working directory (repo root when running app.py / streamlit)
load_dotenv()


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str


def load_llm_config() -> Optional[LLMConfig]:
    """
    Load LLM configuration from environment variables.

    Required:
      - LLM_BASE_URL  (e.g. https://proxy.example.com/v1 or https://proxy.example.com)
      - LLM_API_KEY
      - LLM_MODEL     (e.g. gpt-4o-mini)

    If any value is missing, returns None (LLM disabled gracefully).
    """
    base_url = (os.getenv("LLM_BASE_URL") or "").strip()
    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    model = (os.getenv("LLM_MODEL") or "").strip()

    if not base_url or not api_key or not model:
        return None

    # Normalize base_url for OpenAI-compatible proxies
    if base_url.endswith("/"):
        base_url = base_url[:-1]
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    return LLMConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
    )


def get_llm_client(cfg: LLMConfig) -> OpenAI:
    """
    Create OpenAI-compatible client.
    Works with OpenAI, DeepSeek, ProxyAPI, etc.
    """
    return OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
    )
