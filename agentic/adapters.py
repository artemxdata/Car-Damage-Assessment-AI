from __future__ import annotations

from typing import Any, Optional

from agentic.schemas import DamageSignal


_SEVERITY_MAP = {
    "Light": "minor",
    "Moderate": "moderate",
    "Severe": "severe",
}

_TYPE_MAP = {
    "Scratch": "scratch",
    "Dent": "dent",
    "Paint Damage": "paint_damage",
    "Broken Part": "broken_part",
}


def pick_primary_detection(detections: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """
    Pick a single "primary" detection to drive the workflow.
    Strategy:
      1) Highest severity (Severe > Moderate > Light)
      2) If tie, highest confidence
    """
    if not detections:
        return None

    sev_rank = {"Light": 1, "Moderate": 2, "Severe": 3}

    def key(d: dict[str, Any]) -> tuple[int, float]:
        return (sev_rank.get(str(d.get("severity")), 0), float(d.get("confidence", 0.0)))

    return sorted(detections, key=key, reverse=True)[0]


def detection_to_damage_signal(detection: dict[str, Any]) -> DamageSignal:
    raw_type = str(detection.get("type", "unknown"))
    raw_sev = str(detection.get("severity", ""))

    damage_type = _TYPE_MAP.get(raw_type, raw_type.lower().replace(" ", "_"))
    severity = _SEVERITY_MAP.get(raw_sev, None)
    confidence = float(detection.get("confidence", 0.0))

    return DamageSignal(
        damage_type=damage_type,
        confidence=confidence,
        severity=severity,  # minor/moderate/severe or None
        regions=[],
        notes=None,
    )
