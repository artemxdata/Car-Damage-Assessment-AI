from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Data model
# ----------------------------

@dataclass
class RepairStrategy:
    name: str
    summary: str
    steps: List[str]
    eta_days: Tuple[int, int]          # (min, max)
    cost_usd: Tuple[int, int]          # (min, max)
    risk_level: str                    # "low" | "medium" | "high"


# ----------------------------
# Helpers
# ----------------------------

def _pick(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _norm_severity(sev: Any) -> str:
    s = str(sev or "").strip().lower()
    # allow CV demo "Light/Moderate/Severe"
    if s in ("light", "minor"):
        return "minor"
    if s in ("moderate", "medium"):
        return "moderate"
    if s in ("severe", "critical", "high"):
        return "severe"
    return "unknown"


def _norm_damage_type(t: Any) -> str:
    s = str(t or "").strip().lower()
    # normalize common variants
    if "scratch" in s:
        return "scratch"
    if "dent" in s:
        return "dent"
    if "paint" in s:
        return "paint_damage"
    if "broken" in s or "part" in s:
        return "broken_part"
    return s or "unknown"


def _normalize_one_detection(d: Any) -> Dict[str, Any]:
    """
    Accepts:
      - CV detection dict: {type, severity, confidence, area_percentage, estimated_cost, ...}
      - agentic DamageSignal-like: {damage_type, severity, confidence, ...}
      - arbitrary object with attributes
    Returns unified dict: {type, severity, confidence, ...}
    """
    dtype = _pick(d, "type", None)
    if dtype is None:
        dtype = _pick(d, "damage_type", None)

    sev = _pick(d, "severity", None)
    conf = _as_float(_pick(d, "confidence", 0.0))

    area = _as_float(_pick(d, "area_percentage", 0.0))
    est_cost = _as_float(_pick(d, "estimated_cost", 0.0))

    return {
        "type": _norm_damage_type(dtype),
        "severity": _norm_severity(sev),
        "confidence": conf,
        "area_percentage": area,
        "estimated_cost": est_cost,
        "raw": d,
    }


def _normalize_detections(primary_or_dets: Any) -> List[Dict[str, Any]]:
    """
    Robust input normalization:
      - if dict -> [dict]
      - if list/tuple -> list
      - if object -> [object]
      - if None -> []
    """
    if primary_or_dets is None:
        return []

    # dict => treat as a single detection (IMPORTANT FIX)
    if isinstance(primary_or_dets, dict):
        return [_normalize_one_detection(primary_or_dets)]

    # list/tuple => normalize each element
    if isinstance(primary_or_dets, (list, tuple)):
        out: List[Dict[str, Any]] = []
        for x in primary_or_dets:
            out.append(_normalize_one_detection(x))
        return out

    # any other object => single
    return [_normalize_one_detection(primary_or_dets)]


def _pick_primary(dets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not dets:
        return None

    sev_rank = {"unknown": 0, "minor": 1, "moderate": 2, "severe": 3}

    # Choose primary by severity then confidence then area
    dets_sorted = sorted(
        dets,
        key=lambda d: (
            sev_rank.get(str(d.get("severity", "unknown")), 0),
            float(d.get("confidence", 0.0) or 0.0),
            float(d.get("area_percentage", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return dets_sorted[0]


# ----------------------------
# Public API
# ----------------------------

def build_damage_story(primary_or_dets: Any) -> Dict[str, Any]:
    """
    Returns:
      {
        damage_type, severity, confidence,
        consequences: [...],
        safety_note,
        resale_impact
      }
    """
    dets = _normalize_detections(primary_or_dets)
    primary = _pick_primary(dets)
    if not primary:
        return {
            "damage_type": "unknown",
            "severity": "unknown",
            "confidence": 0.0,
            "consequences": ["No reliable detection available."],
            "safety_note": "Upload clearer photos for a confident assessment.",
            "resale_impact": "unknown",
        }

    dtype = primary["type"]
    sev = primary["severity"]
    conf = float(primary["confidence"] or 0.0)

    consequences: List[str] = []
    safety_note = "No immediate safety concerns detected, but verify in-person if unsure."
    resale_impact = "low"

    if dtype == "scratch":
        consequences = [
            "Paint layer may degrade over time if left untreated (especially if deep).",
            "If metal is exposed, oxidation/rust risk increases in wet/salty climates.",
            "Cosmetic damage can reduce perceived vehicle condition at resale.",
        ]
        safety_note = "Safety risk is usually low unless scratch is near sensors/lights."
        resale_impact = "low" if sev == "minor" else "medium"

    elif dtype == "dent":
        consequences = [
            "Panel deformation may worsen if impacts continue or stress accumulates.",
            "Hidden damage behind the panel is possible (mounts/inner structure).",
            "Paint cracking around the dent can lead to corrosion later.",
        ]
        safety_note = "Check alignment of doors/hood/trunk and any nearby structural points."
        resale_impact = "medium" if sev in ("minor", "moderate") else "high"

    elif dtype == "paint_damage":
        consequences = [
            "Clearcoat failure can spread (peeling/fading) if not sealed.",
            "Exposed layers can absorb moisture → corrosion risk over time.",
            "Color mismatch after repair is a common quality issue.",
        ]
        safety_note = "Low immediate risk, but treat early to prevent spreading and rust."
        resale_impact = "medium" if sev != "minor" else "low"

    elif dtype == "broken_part":
        consequences = [
            "Broken components can compromise safety and road legality.",
            "Water ingress can damage electronics and wiring behind the area.",
            "Driving with broken parts can cause secondary damage.",
        ]
        safety_note = "Potential safety risk. Avoid high-speed driving until verified."
        resale_impact = "high"

    else:
        consequences = [
            "Further inspection recommended to confirm the damage type and extent.",
            "Request additional angles and close-up shots to reduce uncertainty.",
        ]
        safety_note = "If anything looks structural, escalate to in-person inspection."
        resale_impact = "unknown"

    # Severity tuning
    if sev == "severe":
        consequences.insert(0, "Severity is high: risk of hidden damage and higher repair cost is elevated.")
        safety_note = "High severity: recommend professional inspection ASAP."
        resale_impact = "high"
    elif sev == "moderate":
        consequences.insert(0, "Moderate severity: repair is recommended within a reasonable timeframe.")
        resale_impact = "medium" if resale_impact != "high" else "high"
    elif sev == "minor":
        consequences.insert(0, "Minor severity: likely cosmetic, but confirm depth/extent in better lighting.")

    return {
        "damage_type": dtype,
        "severity": sev,
        "confidence": conf,
        "consequences": consequences,
        "safety_note": safety_note,
        "resale_impact": resale_impact,
    }


def build_repair_strategies(primary_or_dets: Any) -> List[RepairStrategy]:
    """
    Returns a shortlist of repair strategies (heuristics).
    Works with dict / list[dict] / DamageSignal-like objects.
    """
    dets = _normalize_detections(primary_or_dets)
    primary = _pick_primary(dets)
    if not primary:
        return [
            RepairStrategy(
                name="Request better photos",
                summary="No reliable primary damage detected. Ask for clearer angles and lighting.",
                steps=["Close-up of the damage", "Wide shot including the whole panel", "Side lighting to show depth"],
                eta_days=(0, 1),
                cost_usd=(0, 0),
                risk_level="low",
            )
        ]

    dtype = primary["type"]
    sev = primary["severity"]

    strategies: List[RepairStrategy] = []

    if dtype == "scratch":
        strategies.append(
            RepairStrategy(
                name="Polish / Buff",
                summary="Best for minor clearcoat scratches; improves appearance quickly.",
                steps=["Wash panel", "Clay bar if needed", "Polish compound", "Protect with wax/ceramic sealant"],
                eta_days=(0, 1),
                cost_usd=(40, 120),
                risk_level="low",
            )
        )
        strategies.append(
            RepairStrategy(
                name="Spot repaint (localized)",
                summary="For deeper scratches that reach basecoat/primer; restores finish.",
                steps=["Sand and feather edges", "Primer (if exposed)", "Color match", "Clearcoat + blend"],
                eta_days=(1, 3),
                cost_usd=(150, 450),
                risk_level="medium",
            )
        )
        if sev in ("moderate", "severe"):
            strategies.append(
                RepairStrategy(
                    name="Full panel repaint",
                    summary="For widespread scratches or mismatched paint; best cosmetic consistency.",
                    steps=["Panel prep", "Full repaint", "Clearcoat", "Cure and polish"],
                    eta_days=(2, 5),
                    cost_usd=(350, 900),
                    risk_level="medium",
                )
            )

    elif dtype == "dent":
        strategies.append(
            RepairStrategy(
                name="Paintless Dent Repair (PDR)",
                summary="Works if paint is intact and dent is accessible; preserves OEM paint.",
                steps=["Inspect paint cracks", "Access backside", "PDR pull/push", "Final light polish"],
                eta_days=(0, 2),
                cost_usd=(120, 450),
                risk_level="low",
            )
        )
        strategies.append(
            RepairStrategy(
                name="Bodywork + repaint",
                summary="For dents with paint damage or sharp creases; strongest finish but higher cost.",
                steps=["Disassemble trim", "Pull dent", "Filler if needed", "Sand", "Prime", "Paint + blend"],
                eta_days=(2, 7),
                cost_usd=(400, 1400),
                risk_level="medium",
            )
        )
        if sev == "severe":
            strategies.append(
                RepairStrategy(
                    name="Panel replacement",
                    summary="If deformation impacts structure/alignment; fastest route to restore geometry.",
                    steps=["Confirm structural impact", "Order parts", "Replace panel", "Paint match", "Calibration if sensors"],
                    eta_days=(5, 14),
                    cost_usd=(900, 3000),
                    risk_level="high",
                )
            )

    elif dtype == "paint_damage":
        strategies.append(
            RepairStrategy(
                name="Seal / protect (temporary)",
                summary="Short-term option to stop spreading; not a true repair.",
                steps=["Clean", "Feather peeling edges", "Apply sealant/clear touch-up", "Monitor for spread"],
                eta_days=(0, 1),
                cost_usd=(20, 80),
                risk_level="medium",
            )
        )
        strategies.append(
            RepairStrategy(
                name="Repaint affected area",
                summary="Restores finish and prevents corrosion risk.",
                steps=["Prep surface", "Prime if needed", "Color match", "Clearcoat", "Blend adjacent areas"],
                eta_days=(2, 6),
                cost_usd=(300, 1200),
                risk_level="medium",
            )
        )

    elif dtype == "broken_part":
        strategies.append(
            RepairStrategy(
                name="Replace broken component",
                summary="Safety-first: replace damaged part and verify alignment.",
                steps=["Identify broken parts", "Order OEM/aftermarket", "Replace", "Inspect mounts", "Road test"],
                eta_days=(2, 10),
                cost_usd=(250, 2500),
                risk_level="high",
            )
        )

    else:
        strategies.append(
            RepairStrategy(
                name="Technician inspection",
                summary="Damage type unclear. Recommend in-person inspection and additional photos.",
                steps=["Close-up photos", "Wide shot including panel", "Side lighting", "Technician review"],
                eta_days=(0, 3),
                cost_usd=(0, 150),
                risk_level="medium",
            )
        )

    # Severity tweak: push urgency
    if sev == "severe":
        for s in strategies:
            if s.risk_level == "low":
                s.risk_level = "medium"

    return strategies
