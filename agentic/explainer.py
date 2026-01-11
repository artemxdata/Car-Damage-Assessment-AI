from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def _sig_get(signal: Any, key: str, default=None):
    if isinstance(signal, dict):
        return signal.get(key, default)
    return getattr(signal, key, default)


def _strip_markdown(md: str) -> str:
    """
    Lightweight Markdown cleaner: removes headings, bold markers, code fences.
    Keeps content readable for end-users.
    """
    if not md:
        return ""

    text = md

    # remove code fences
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # remove headings markers
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)

    # remove bold/italic markers
    text = text.replace("**", "").replace("__", "").replace("*", "")

    # collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text


def _extract_inline_field(md: str, field_name: str) -> Optional[str]:
    """
    Extract inline fields like:
      Action: HUMAN_REVIEW
      Next steps: ...
    from SOP section text (markdown).
    """
    if not md:
        return None

    # match variations: "Action:", "**Action:**", "action:"
    pattern = re.compile(rf"^\s*(\*\*)?{re.escape(field_name)}(\*\*)?\s*:\s*(.+)\s*$", re.IGNORECASE | re.MULTILINE)
    m = pattern.search(md)
    if not m:
        return None
    return m.group(3).strip()


def _extract_bullets_after(md: str, anchor: str, max_items: int = 6) -> List[str]:
    """
    After a line that starts with anchor (like "Next steps"),
    parse subsequent bullet list "- item".
    """
    if not md:
        return []
    lines = md.splitlines()
    out: List[str] = []
    anchor_re = re.compile(rf"^\s*(\*\*)?{re.escape(anchor)}(\*\*)?\s*:\s*$", re.IGNORECASE)

    start_idx = None
    for i, line in enumerate(lines):
        if anchor_re.match(line.strip()):
            start_idx = i + 1
            break
        # also handle inline "Next steps: text"
        if line.strip().lower().startswith(anchor.lower()) and ":" in line:
            # inline already handled elsewhere, return empty here
            return []

    if start_idx is None:
        return []

    for j in range(start_idx, len(lines)):
        s = lines[j].strip()
        if not s:
            continue
        if s.startswith("- "):
            out.append(s[2:].strip())
            if len(out) >= max_items:
                break
            continue
        # stop if section ends
        if re.match(r"^\s*#{1,6}\s+", s):
            break
        # stop on "Action:" etc.
        if re.match(r"^\s*(\*\*)?\w+(\*\*)?\s*:\s*", s):
            break

    return out


def build_customer_explanation(
    decision_action: str,
    decision_reason: str,
    policy_refs: List[str],
    next_steps: List[str],
    sop_text: Optional[str],
    signal: Any,
) -> Dict[str, Any]:
    """
    Returns product-ready text blocks for UI (no raw markdown).
    """
    dmg_type = _sig_get(signal, "damage_type", "unknown")
    severity = _sig_get(signal, "severity", None)
    conf = _sig_get(signal, "confidence", None)

    # Make a tight one-liner summary
    parts = []
    if severity:
        parts.append(str(severity))
    if dmg_type:
        parts.append(str(dmg_type))
    what = " ".join(parts).strip() or "damage"

    conf_txt = ""
    try:
        if conf is not None:
            conf_txt = f" (confidence {float(conf):.2f})"
    except Exception:
        conf_txt = ""

    title = {
        "AUTO_APPROVE": "Eligible for auto-approval",
        "HUMAN_REVIEW": "Requires human review",
        "ESCALATE": "Escalate to specialist",
    }.get(decision_action, decision_action)

    summary = f"{title}: {what}{conf_txt}."

    # SOP-derived “why” bullets (cleaned)
    sop_clean = _strip_markdown(sop_text or "")
    why_bullets: List[str] = []

    # If SOP has a rationale line, use it
    # Otherwise derive from decision_reason and SOP next steps
    if decision_reason:
        why_bullets.append(decision_reason)

    # pull some lines from SOP (first 2–4 meaningful lines)
    if sop_clean:
        lines = [ln.strip() for ln in sop_clean.splitlines() if ln.strip()]
        # skip lines that look like "version:" / "thresholds" etc
        filtered = []
        for ln in lines:
            if ln.lower().startswith("version:"):
                continue
            if ln.lower().startswith("threshold"):
                continue
            if ln.lower().startswith("rules:"):
                continue
            filtered.append(ln)
        # take top 3 lines max (besides reason)
        for ln in filtered[:3]:
            if ln not in why_bullets and len(why_bullets) < 4:
                why_bullets.append(ln)

    # Next steps: prefer explicit next_steps, fallback from SOP bullets
    steps = list(next_steps or [])
    if not steps and sop_text:
        inline = _extract_inline_field(sop_text, "Next steps")
        if inline:
            steps = [inline]
        else:
            steps = _extract_bullets_after(sop_text, "Next steps", max_items=6)

    # Policy refs: keep as short strings
    refs = policy_refs or []
    ref_primary = refs[0] if refs else None

    return {
        "title": title,
        "summary": summary,
        "why_bullets": why_bullets[:5],
        "next_steps": steps[:8],
        "policy_ref_primary": ref_primary,
        "policy_refs": refs[:6],
    }


def format_kb_insights(chunks: List[Any], max_items: int = 4) -> List[str]:
    """
    Convert RetrievedChunk list into customer-friendly bullets.
    No scores, no 'retrieved', no debug.
    """
    if not chunks:
        return []

    insights: List[str] = []
    for ch in chunks[:max_items]:
        txt = getattr(ch, "text", "") or ""
        txt = _strip_markdown(txt)

        # Take 1–2 sentences max
        sentences = re.split(r"(?<=[.!?])\s+", txt)
        one = " ".join(sentences[:2]).strip()
        if not one:
            continue

        # Make it bullet-ish
        one = re.sub(r"\s+", " ", one).strip()
        # Avoid too long bullets
        if len(one) > 240:
            one = one[:237].rstrip() + "..."

        insights.append(one)

    # de-dup
    out: List[str] = []
    for i in insights:
        if i not in out:
            out.append(i)
    return out[:max_items]
