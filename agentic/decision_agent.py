from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from agentic.policy_loader import Policy, load_policy
from agentic.schemas import Decision, DamageSignal
from agentic.sop_retriever import load_sop_section


def _match_rule(signal: DamageSignal, cond: dict) -> bool:
    # Supported conditions (minimal for spike):
    # - damage_type_in: list[str]
    # - severity_in: list[str]
    # - confidence_gte: float
    # - confidence_lt: float

    if "damage_type_in" in cond:
        if signal.damage_type not in cond["damage_type_in"]:
            return False

    if "severity_in" in cond:
        if signal.severity not in cond["severity_in"]:
            return False

    if "confidence_gte" in cond:
        if signal.confidence < float(cond["confidence_gte"]):
            return False

    if "confidence_lt" in cond:
        if signal.confidence >= float(cond["confidence_lt"]):
            return False

    return True


class DecisionAgent:
    def __init__(self, policies_dir: str | Path = "policies"):
        self.policies_dir = Path(policies_dir)
        self.policy: Policy = load_policy(self.policies_dir)

    def decide(self, signal: DamageSignal) -> Decision:
        # Evaluate rules in order
        for rule in self.policy.rules:
            if _match_rule(signal, rule.cond):
                policy_ref, sop_text = load_sop_section(self.policies_dir, rule.sop_ref)

                next_steps = []
                # Very small parsing: extract "Next steps" lines if present
                for line in sop_text.splitlines():
                    if line.strip().lower().startswith("**next steps:**"):
                        next_steps.append(line.split(":", 1)[-1].strip())

                return Decision(
                    action=rule.action,  # type: ignore
                    reason=rule.reason,
                    policy_refs=[policy_ref],
                    next_steps=next_steps,
                )

        # Default fallback (safe)
        return Decision(
            action="HUMAN_REVIEW",
            reason="No policy rule matched. Defaulting to human review for safety.",
            policy_refs=["SOP:damage_triage.md#low-confidence"],
            next_steps=["Request additional images and verify context."],
        )
