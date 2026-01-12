from __future__ import annotations

from typing import Any, Dict, List, Optional


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _pick(d: Any, key: str, default=None):
    if d is None:
        return default
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _format_primary(primary: Optional[Dict[str, Any]] = None, signal: Any = None) -> str:
    """
    Supports both:
      - CV primary detection dict: {type, severity, confidence, ...}
      - agentic DamageSignal-like: {damage_type, severity, confidence, ...}
    """
    if primary:
        t = _pick(primary, "type", "Unknown")
        sev = _pick(primary, "severity", "unknown")
        conf = _as_float(_pick(primary, "confidence", 0.0))
        return f"Detected: {t} · Severity: {sev} · Confidence: {conf:.2f}"

    if signal is not None:
        t = _pick(signal, "damage_type", _pick(signal, "type", "Unknown"))
        sev = _pick(signal, "severity", "unknown")
        conf = _as_float(_pick(signal, "confidence", 0.0))
        return f"Detected: {t} · Severity: {sev} · Confidence: {conf:.2f}"

    return "Detected: (no primary signal)"


def build_customer_explanation(
    action: Optional[str] = None,
    reason: Optional[str] = None,
    primary_detection: Optional[Dict[str, Any]] = None,
    next_steps: Optional[List[str]] = None,
    policy_refs: Optional[List[str]] = None,
    evidence: Optional[str] = None,
    sop_text: Optional[str] = None,
    signal: Any = None,
    # ✅ Aliases used in your app.py right now:
    decision_action: Optional[str] = None,
    decision_reason: Optional[str] = None,
    **_kwargs,
) -> Dict[str, Any]:
    """
    Product-ready, backwards-compatible explanation builder.

    IMPORTANT: Your app calls:
      build_customer_explanation(decision_action=..., decision_reason=..., ...)

    So we accept both:
      - action/reason (canonical)
      - decision_action/decision_reason (aliases)

    Also returns keys expected by app.py:
      - summary
      - why_bullets
      - next_steps
      - policy_refs
    """
    # Resolve aliases
    action = action or decision_action or "HUMAN_REVIEW"
    reason = reason or decision_reason or "Decision reason not provided."

    # evidence alias
    if evidence is None and sop_text:
        evidence = sop_text

    steps = list(next_steps or [])
    refs = list(policy_refs or [])

    if action == "AUTO_APPROVE":
        title = "Auto Approved"
        badge = "success"
        summary = "This case matches our auto-approval policy."
    elif action == "ESCALATE":
        title = "Escalated"
        badge = "error"
        summary = "This case should be escalated to a specialist assessor."
    else:
        title = "Needs Human Review"
        badge = "warning"
        summary = "A human operator should confirm the damage details before approval."

    why_bullets: List[str] = [
        _format_primary(primary_detection, signal),
        reason,
    ]

    # If you want refs visible to user you can add; otherwise keep for dev/debug only
    if refs:
        why_bullets.append("Policy refs: " + ", ".join(refs))

    return {
        # UI-friendly
        "title": title,
        "badge": badge,
        "summary": summary,

        # app.py expects these
        "why_bullets": why_bullets,
        "next_steps": steps,
        "policy_refs": refs,

        # optional debug/evidence
        "evidence": evidence,
    }


def format_kb_insights(chunks: List[Any], max_items: int = 4) -> List[str]:
    """
    app.py expects iterable list of strings:
      for it in insights: st.write(f"- {it}")

    So return List[str] (not dict).
    """
    out: List[str] = []
    for ch in (chunks or [])[:max_items]:
        src = _pick(ch, "source", "unknown")
        score = _as_float(_pick(ch, "score", 0.0))
        text = (_pick(ch, "text", "") or "").strip()
        first_line = text.splitlines()[0].strip() if text else ""
        if first_line:
            out.append(f"{src} (score={score:.2f}) — {first_line}")
        else:
            out.append(f"{src} (score={score:.2f})")
    return out


def build_expert_insight(**kwargs) -> Dict[str, Any]:
    """
    Your app uses build_expert_insight(...) as LLM-copilot entrypoint.
    We proxy to agentic.expert_ai.build_expert_insight if it exists.
    """
    try:
        from agentic.expert_ai import build_expert_insight as _impl  # type: ignore
        return _impl(**kwargs)
    except Exception as e:
        return {
            "enabled": False,
            "error": f"Expert insight disabled: {e}",
            "insight": None,
            "raw": None,
            "kb_hits": None,
        }
