from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from agentic.policy_loader import Policy, load_policy
from agentic.schemas import Decision, DamageSignal
from agentic.sop_retriever import load_sop_section

def _sig_get(signal, key: str, default=None):
    # supports both dict and objects/dataclasses
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
		    evidence=sop_text[:600] if sop_text else None,
                )

        # Default fallback (safe)
        return Decision(
            action="HUMAN_REVIEW",
            reason="No policy rule matched. Defaulting to human review for safety.",
            policy_refs=["SOP:damage_triage.md#low-confidence"],
            next_steps=["Request additional images and verify context."],
   	    evidence="Fallback safety policy: route to human review when rules do not match.",
        )
