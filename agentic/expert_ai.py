from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agentic.llm.providers import build_provider_from_env
from agentic.llm.prompts import build_expert_prompt
from agentic.llm.types import ExpertInput
from agentic.rag.simple_retriever import SimpleRetriever, RetrievedChunk


@dataclass
class ExpertResult:
    text: str
    used_llm: bool
    kb_hits: int


def build_default_commentary(signal: dict, decision_action: str, primary: Optional[dict]) -> str:
    """
    Deterministic fallback commentary if LLM not configured.
    Still looks like "expert", but it's template-based.
    """
    dtype = signal.get("damage_type", "damage")
    sev = signal.get("severity", "unknown")
    conf = float(signal.get("confidence", 0.0) or 0.0)

    conf_note = "high" if conf >= 0.85 else ("medium" if conf >= 0.55 else "low")

    det = primary or {}
    est = det.get("estimated_cost")

    return f"""### Expert insight
- Detected **{dtype}** with **{sev}** severity (confidence: **{conf_note}**, {conf:.2f}).
- The system decision is **{decision_action}** based on internal handling policy.
- Estimated repair cost shown is **{est}** (demo estimate).

### What to confirm next
- Capture **close-up** and **wide** shots of the affected panel.
- Check if damage is near **edges, seams, or structural points**.
- Confirm panel alignment and paint cracking around the area.
- Verify vehicle ID/VIN context if photos are from multiple vehicles.

### Risk notes
- Hidden damage may exist behind the panel (verify on inspection).
- If angles/lighting are poor, confidence and severity may shift after new photos.
""".strip()


def generate_expert_commentary(
    *,
    decision_action: str,
    decision_reason: str,
    sop_evidence: Optional[str],
    signal: dict,
    primary_detection: Optional[dict],
    knowledge_dir,
    enable_llm: bool = True,
    top_k: int = 3,
) -> ExpertResult:
    """
    Non-decision LLM layer:
    - Retrieves KB chunks (simple RAG)
    - Uses LLM if configured and enable_llm=True
    - Otherwise returns deterministic template fallback
    """
    retriever = SimpleRetriever(knowledge_dir=knowledge_dir)
    query = f"{signal.get('damage_type','damage')} {signal.get('severity','')} verification next steps"
    kb_chunks: list[RetrievedChunk] = retriever.retrieve(query=query, top_k=top_k)

    if not enable_llm:
        return ExpertResult(
            text=build_default_commentary(signal, decision_action, primary_detection),
            used_llm=False,
            kb_hits=len(kb_chunks),
        )

    provider = build_provider_from_env()
    inp = ExpertInput(
        decision_action=decision_action,
        decision_reason=decision_reason,
        sop_evidence=sop_evidence,
        signal=signal,
        primary_detection=primary_detection,
        kb_chunks=kb_chunks,
    )
    prompt = build_expert_prompt(inp)
    out = (provider.generate(prompt) or "").strip()

    if out:
        return ExpertResult(text=out, used_llm=True, kb_hits=len(kb_chunks))

    # Fallback if provider is noop or failed silently
    return ExpertResult(
        text=build_default_commentary(signal, decision_action, primary_detection),
        used_llm=False,
        kb_hits=len(kb_chunks),
    )
