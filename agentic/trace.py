from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TraceStep:
    title: str
    details: List[str]


def _fmt_primary(primary: Optional[Dict[str, Any]], signal: Any) -> str:
    if primary:
        t = primary.get("type", "Unknown")
        sev = primary.get("severity", "unknown")
        conf = float(primary.get("confidence", 0.0) or 0.0)
        return f"{t} · {sev} · conf {conf:.2f}"
    if signal is not None:
        t = getattr(signal, "damage_type", None) or getattr(signal, "type", None) or "Unknown"
        sev = getattr(signal, "severity", None) or "unknown"
        conf = float(getattr(signal, "confidence", 0.0) or 0.0)
        return f"{t} · {sev} · conf {conf:.2f}"
    return "(no primary)"


def build_decision_trace(
    *,
    primary_detection: Optional[Dict[str, Any]] = None,
    signal: Any = None,
    decision: Any = None,
) -> Dict[str, Any]:
    """
    Returns a UI-friendly decision trace dict.
    Designed to be stable and safe even if some fields are missing.
    """
    steps: List[TraceStep] = []

    # 1) Primary selection
    steps.append(
        TraceStep(
            title="Primary detection selected",
            details=[_fmt_primary(primary_detection, signal)],
        )
    )

    # 2) Policy matched
    policy_refs = []
    if decision is not None:
        policy_refs = list(getattr(decision, "policy_refs", []) or [])
    if policy_refs:
        steps.append(
            TraceStep(
                title="Policy matched",
                details=policy_refs,
            )
        )
    else:
        steps.append(
            TraceStep(
                title="Policy matched",
                details=["(no explicit policy reference)"],
            )
        )

    # 3) SOP / evidence reference
    evidence = None
    if decision is not None:
        evidence = getattr(decision, "evidence", None)
    if evidence:
        steps.append(
            TraceStep(
                title="SOP / evidence",
                details=["SOP excerpt attached (see debug / evidence block)"],
            )
        )
    else:
        steps.append(
            TraceStep(
                title="SOP / evidence",
                details=["(no SOP excerpt)"],
            )
        )

    # 4) Action derived
    action = getattr(decision, "action", None) if decision is not None else None
    reason = getattr(decision, "reason", None) if decision is not None else None
    if action:
        lines = [f"Action: {action}"]
        if reason:
            lines.append(f"Reason: {reason}")
        steps.append(TraceStep(title="Action derived", details=lines))
    else:
        steps.append(TraceStep(title="Action derived", details=["(no decision)"]))

    return {
        "title": "Decision Trace",
        "steps": [{"title": s.title, "details": s.details} for s in steps],
        "policy_refs": policy_refs,
        "has_evidence": bool(evidence),
    }
