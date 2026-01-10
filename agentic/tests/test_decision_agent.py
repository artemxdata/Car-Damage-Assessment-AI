from agentic.decision_agent import DecisionAgent
from agentic.schemas import DamageSignal


def test_auto_approve_minor_scratch():
    agent = DecisionAgent(policies_dir="policies")
    signal = DamageSignal(damage_type="scratch", confidence=0.92, severity="minor")
    decision = agent.decide(signal)
    assert decision.action == "AUTO_APPROVE"


def test_low_confidence_human_review():
    agent = DecisionAgent(policies_dir="policies")
    signal = DamageSignal(damage_type="dent", confidence=0.40, severity="moderate")
    decision = agent.decide(signal)
    assert decision.action == "HUMAN_REVIEW"
