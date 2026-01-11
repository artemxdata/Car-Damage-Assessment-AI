from __future__ import annotations

from typing import Any, Iterable

from agentic.llm.types import ExpertInput


def build_expert_prompt(inp: ExpertInput) -> str:
    """
    Expert commentary prompt:
    - NOT making final approval decision (policy already did)
    - Adds practical guidance, what to check next, risks, and confidence caveats
    """
    kb_text = ""
    if inp.kb_chunks:
        kb_text = "\n\n".join(
            [
                f"[{c.source} | score={c.score:.2f}]\n{c.text}".strip()
                for c in inp.kb_chunks
            ]
        )

    sop_text = (inp.sop_evidence or "").strip()

    signal = inp.signal
    det = inp.primary_detection or {}

    return f"""
You are a senior vehicle damage assessor writing a short "expert commentary" for a customer-facing demo.

IMPORTANT RULES:
- You must NOT change or override the system decision.
- You must NOT say "I am an AI" or mention policies/rules.yaml.
- No legal/medical/safety claims. If unsure, recommend inspection.
- Keep it concise and actionable. Avoid jargon.

OUTPUT FORMAT (Markdown):
### Expert insight
- 2-4 bullet points of practical assessment

### What to confirm next
- 3-5 bullet points checklist

### Risk notes
- 2-3 bullets about hidden damage / uncertainty

INPUT:
Decision: {inp.decision_action}
Reason (internal): {inp.decision_reason}

Primary detection (raw):
- type: {det.get("type")}
- severity: {det.get("severity")}
- confidence: {det.get("confidence")}
- bbox: {det.get("bbox")}
- area_percentage: {det.get("area_percentage")}
- estimated_cost: {det.get("estimated_cost")}

Normalized signal:
- damage_type: {signal.get("damage_type")}
- severity: {signal.get("severity")}
- confidence: {signal.get("confidence")}

SOP excerpt (why decision):
{sop_text if sop_text else "(none)"}

Knowledge base (practical notes):
{kb_text if kb_text else "(none)"}
""".strip()
