from __future__ import annotations

from pathlib import Path

from agentic.policy_loader import Policy, load_policy
from agentic.schemas import Decision, DamageSignal
from agentic.sop_retriever import load_sop_section
from agentic.rag.simple_retriever import SimpleRetriever


def _sig_get(signal, key: str, default=None):
    if isinstance(signal, dict):
        return signal.get(key, default)
    return getattr(signal, key, default)


def _match_rule(signal, cond: dict) -> bool:
    damage_type = _sig_get(signal, "damage_type")
    severity = _sig_get(signal, "severity")
    confidence = float(_sig_get(signal, "confidence", 0.0))

    if "damage_type_in" in cond:
        if damage_type not in cond["damage_type_in"]:
            return False

    if "severity_in" in cond:
        if severity not in cond["severity_in"]:
            return False

    if "confidence_gte" in cond:
        if confidence < float(cond["confidence_gte"]):
            return False

    if "confidence_lt" in cond:
        if confidence >= float(cond["confidence_lt"]):
            return False

    return True


def _extract_next_steps(sop_text: str) -> list[str]:
    """
    Extract next steps from SOP markdown.
    Supports:
      **Next steps:** some text
      **Next steps:**
        - item
        - item
    """
    if not sop_text:
        return []

    lines = sop_text.splitlines()
    next_steps: list[str] = []

    for i, line in enumerate(lines):
        if line.strip().lower().startswith("**next steps:**"):
            after = line.split(":", 1)[-1].strip()
            if after:
                next_steps.append(after)
                return next_steps

            j = i + 1
            while j < len(lines):
                s = lines[j].strip()
                if not s:
                    j += 1
                    continue
                if s.startswith("- "):
                    next_steps.append(s[2:].strip())
                    j += 1
                    continue
                break
            return next_steps

    return next_steps


class DecisionAgent:
    """
    Policy-first agent:
    - rules.yaml decides action
    - SOP provides evidence/next-steps
    - Retriever exists for optional UI "expert insights" (not part of the decision)
    """
    def __init__(self, policies_dir: str | Path = "policies"):
        self.policies_dir = Path(policies_dir)
        self.policy: Policy = load_policy(self.policies_dir)

        # Simple Text-RAG retriever over ./knowledge/*.md
        # policies_dir is usually ./policies -> parent is repo root
        self.retriever = SimpleRetriever(knowledge_dir=self.policies_dir.parent / "knowledge")

    def decide(self, signal: DamageSignal) -> Decision:
        for rule in self.policy.rules:
            if _match_rule(signal, rule.cond):
                policy_ref, sop_text = load_sop_section(self.policies_dir, rule.sop_ref)
                next_steps = _extract_next_steps(sop_text)

                return Decision(
                    action=rule.action,  # type: ignore
                    reason=rule.reason,
                    policy_refs=[policy_ref],
                    next_steps=next_steps,
                    evidence=sop_text[:1200] if sop_text else None,
                )

        # Default fallback (safe) -> keep it consistent via SOP section
        fallback_ref = "damage_triage.md#low-confidence"
        policy_ref, sop_text = load_sop_section(self.policies_dir, fallback_ref)
        next_steps = _extract_next_steps(sop_text)

        return Decision(
            action="HUMAN_REVIEW",
            reason="No policy rule matched. Defaulting to human review for safety.",
            policy_refs=[policy_ref],
            next_steps=next_steps or ["Request additional images and verify context."],
            evidence=sop_text[:1200] if sop_text else None,
        )
