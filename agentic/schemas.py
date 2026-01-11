from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


DamageType = str
Severity = Literal["minor", "moderate", "severe"]
Action = Literal["AUTO_APPROVE", "HUMAN_REVIEW", "ESCALATE"]


@dataclass
class DamageRegion:
    x: float
    y: float
    w: float
    h: float
    score: float


@dataclass
class DamageSignal:
    damage_type: DamageType
    confidence: float
    severity: Optional[Severity] = None
    regions: List[DamageRegion] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class Decision:
    action: Action
    reason: str
    policy_refs: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    evidence: Optional[str] = None
    # NEW: debug-only RAG context (should be hidden for customers by default)
    kb_evidence: Optional[str] = None
