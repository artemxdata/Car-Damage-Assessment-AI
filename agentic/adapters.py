from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


def pick_primary_detection(detections: list[dict]) -> Optional[dict]:
    """
    Pick a single "primary" detection to drive the agent decision.
    Heuristic: highest severity, then highest confidence.
    """
    if not detections:
        return None

    severity_rank = {"Severe": 3, "Moderate": 2, "Light": 1}
    return sorted(
        detections,
        key=lambda d: (severity_rank.get(d.get("severity"), 0), d.get("confidence", 0.0)),
        reverse=True,
    )[0]


def _normalize_damage_type(raw: str) -> str:
    s = (raw or "").strip().lower()
    # normalize common variants
    if "scratch" in s:
        return "scratch"
    if "dent" in s:
        return "dent"
    if "paint" in s:
        return "paint_damage"
    if "broken" in s:
        return "broken_part"
    # fallback: snake-ish
    return s.replace(" ", "_")


def _normalize_severity(raw: str) -> str:
    s = (raw or "").strip().lower()
    mapping = {
        "light": "minor",
        "minor": "minor",
        "moderate": "moderate",
        "medium": "moderate",
        "severe": "severe",
        "high": "severe",
    }
    return mapping.get(s, s.replace(" ", "_"))


def detection_to_damage_signal(detection: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a detection dict from the CV layer into an agent-ready 'signal'
    that matches policy schema (rules.yaml).
    """
    raw_type = detection.get("type", "")
    raw_sev = detection.get("severity", "")
    conf = float(detection.get("confidence", 0.0))

    signal = {
        "damage_type": _normalize_damage_type(raw_type),   # e.g. 'scratch'
        "severity": _normalize_severity(raw_sev),          # e.g. 'minor'
        "confidence": conf,
        "bbox": detection.get("bbox"),
        "area_percentage": float(detection.get("area_percentage", 0.0)),
        "estimated_cost": float(detection.get("estimated_cost", 0.0)),
        "raw": detection,  # keep original for debugging
    }
    return signal
