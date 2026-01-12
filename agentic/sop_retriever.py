from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def extract_section(markdown_text: str, anchor: str) -> str:
    """
    Very lightweight SOP retrieval:
    - anchor is like "#minor-scratch"
    - we find a heading "## minor-scratch" and return until next "## "
    """
    anchor = anchor.lstrip("#").strip()
    lines = markdown_text.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() == f"## {anchor}".lower():
            start_idx = i
            break

    if start_idx is None:
        return ""

    out: List[str] = []
    for j in range(start_idx, len(lines)):
        line = lines[j]
        if j != start_idx and line.strip().startswith("## "):
            break
        out.append(line)

    return "\n".join(out).strip()


def load_sop_section(policies_dir: str | Path, sop_ref: str) -> Tuple[str, str]:
    """
    sop_ref: 'damage_triage.md#minor-scratch'
    returns: (policy_ref, extracted_text)
    """
    policies_dir = Path(policies_dir)
    if "#" in sop_ref:
        fname, anchor = sop_ref.split("#", 1)
        anchor = f"#{anchor}"
    else:
        fname, anchor = sop_ref, ""

    md_path = policies_dir / fname
    text = md_path.read_text(encoding="utf-8")

    section = extract_section(text, anchor) if anchor else text
    return f"SOP:{fname}{anchor}", section
