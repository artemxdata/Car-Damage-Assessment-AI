from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import re


@dataclass
class RetrievedChunk:
    source: str
    score: float
    text: str


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9_\-\s]", " ", text)
    return [t for t in text.split() if len(t) >= 2]


def _score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_set = set(doc_tokens)
    hits = sum(1 for t in query_tokens if t in doc_set)
    return hits / max(1, len(set(query_tokens)))


def _chunk_markdown(md: str, max_chars: int = 900) -> List[str]:
    """
    Super simple chunker:
    - split by headings
    - keep chunks <= max_chars
    """
    parts = re.split(r"\n(?=#+\s)", md)
    chunks: List[str] = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


class SimpleRetriever:
    def __init__(self, knowledge_dir: str | Path = "knowledge"):
        self.knowledge_dir = Path(knowledge_dir)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        q_tokens = _tokenize(query)
        if not self.knowledge_dir.exists():
            return []

        results: List[RetrievedChunk] = []
        for fp in sorted(self.knowledge_dir.glob("*.md")):
            try:
                md = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for chunk in _chunk_markdown(md):
                score = _score(q_tokens, _tokenize(chunk))
                if score > 0:
                    results.append(
                        RetrievedChunk(
                            source=fp.name,
                            score=score,
                            text=chunk[:1200],
                        )
                    )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
