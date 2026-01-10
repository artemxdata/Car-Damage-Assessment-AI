# Damage Triage SOP (Demo)

## low-confidence
If the model confidence is low, route the case to human review.  
Rationale: avoid automation on uncertain predictions.

**Action:** HUMAN_REVIEW  
**Next steps:** request better photos / additional angles; verify vehicle ID and context.

---

## minor-scratch
Minor scratch with high confidence can be auto-approved for standard processing.

**Action:** AUTO_APPROVE  
**Next steps:** create repair ticket; estimate based on standard scratch policy; notify customer.

---

## moderate-damage
Moderate damage should be reviewed by an operator before approval.

**Action:** HUMAN_REVIEW  
**Next steps:** verify severity; check affected area; confirm if replacement is required.

---

## severe-damage
Severe damage must be escalated for specialist assessment.

**Action:** ESCALATE  
**Next steps:** assign to senior assessor; request additional documentation; safety check if needed.
