from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Rule:
    name: str
    cond: Dict[str, Any]
    action: str
    sop_ref: str
    reason: str


@dataclass
class Policy:
    thresholds: Dict[str, float]
    rules: List[Rule]


def load_policy(policies_dir: str | Path) -> Policy:
    policies_dir = Path(policies_dir)
    rules_path = policies_dir / "rules.yaml"

    data = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    thresholds = data.get("thresholds", {}) or {}

    rules: List[Rule] = []
    for item in data.get("rules", []):
        rules.append(
            Rule(
                name=item["name"],
                cond=item.get("if", {}) or {},
                action=item["then"]["action"],
                sop_ref=item["then"]["sop_ref"],
                reason=item["then"]["reason"],
            )
        )

    return Policy(thresholds=thresholds, rules=rules)
