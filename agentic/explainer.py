from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional
import json

from agentic.schemas import Decision
from agentic.llm.client import load_llm_config, get_llm_client
from agentic.rag.simple_retriever import SimpleRetriever


SYSTEM_PROMPT = """You are a senior vehicle damage assessor assistant.
Your job: explain the decision and give practical next actions.
IMPORTANT:
- Do NOT change or override the agent decision.
- Be concise, actionable, and safe.
- If info is insufficient, request specific additional photos/angles.
Return ONLY valid JSON (no markdown)."""


def _safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    # Try strict JSON first; then attempt to extract a JSON object from text.
    try:
        return json.loads(text)
    except Exception:
        pass

    # crude extraction: first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def build_expert_insight(
    *,
    decision: Decision,
    primary_detection: Dict[str, Any],
    all_detections: List[Dict[str, Any]],
    sop_evidence: Optional[str] = None,
    knowledge_dir: str = "knowledge",
) -> Dict[str, Any]:
    """
    Returns a dict for UI rendering:
      - enabled: bool
      - error: str|None
      - insight: dict|None (structured)
      - kb_hits: list (debug)
    """
    cfg = load_llm_config()
    if not cfg:
        return {"enabled": False, "error": "LLM disabled (set LLM_ENABLED=1 and LLM_* env vars).", "insight": None, "kb_hits": []}

    # Retrieve KB context (optional)
    retriever = SimpleRetriever(knowledge_dir=knowledge_dir)
    query = f"{primary_detection.get('type','')} {primary_detection.get('severity','')} decision {decision.action}"
    kb_hits = retriever.retrieve(query, top_k=2)

    kb_text = ""
    for h in kb_hits:
        kb_text += f"\n\nSOURCE: {h.source} (score={h.score:.2f})\n{h.text}"

    user_payload = {
        "decision": asdict(decision),
        "primary_detection": primary_detection,
        "all_detections": all_detections,
        "sop_evidence": sop_evidence or "",
        "knowledge": kb_text.strip(),
        "output_schema": {
            "customer_message": "string (short, friendly, no jargon)",
            "operator_playbook": ["bullet strings (checklist)"],
            "extra_photos_needed": ["bullet strings (specific angles)"],
            "risks": ["bullet strings (what could be hidden/unsafe)"],
        },
    }

    client = get_llm_client(cfg)
    raw = client.chat(
        system=SYSTEM_PROMPT,
        user=json.dumps(user_payload, ensure_ascii=False, indent=2),
        temperature=0.2,
    )

    parsed = _safe_json_parse(raw)
    if not parsed:
        return {
            "enabled": True,
            "error": "LLM returned non-JSON. (Enable developer mode to see raw.)",
            "insight": {"raw": raw},
            "kb_hits": [asdict(h) for h in kb_hits],
        }

    # minimal normalization
    parsed.setdefault("customer_message", "")
    parsed.setdefault("operator_playbook", [])
    parsed.setdefault("extra_photos_needed", [])
    parsed.setdefault("risks", [])
    return {
        "enabled": True,
        "error": None,
        "insight": parsed,
        "kb_hits": [asdict(h) for h in kb_hits],
        "raw": raw,
    }
