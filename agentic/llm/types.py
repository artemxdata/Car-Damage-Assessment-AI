from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from agentic.rag.simple_retriever import RetrievedChunk


@dataclass
class ExpertInput:
    decision_action: str
    decision_reason: str
    sop_evidence: Optional[str]
    signal: dict
    primary_detection: Optional[dict] = None
    kb_chunks: list[RetrievedChunk] | None = None
