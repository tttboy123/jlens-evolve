"""v3 Evidence-Centric Evolution Platform.

The public interfaces are intentionally small: strategies create neutral execution
plans, the runtime emits immutable receipts, observers emit evidence, and only
governance may promote inactive candidates.
"""

from .contracts import (
    Authorization,
    Claim,
    ClaimClassification,
    ClaimGrade,
    Cohort,
    ContractViolation,
    EvidenceEnvelope,
    ExecutionLimits,
    ExecutionPlan,
    ModelIdentity,
    Receipt,
    TaskRevision,
)

__all__ = [
    "Authorization",
    "Claim",
    "ClaimClassification",
    "ClaimGrade",
    "Cohort",
    "ContractViolation",
    "EvidenceEnvelope",
    "ExecutionLimits",
    "ExecutionPlan",
    "ModelIdentity",
    "Receipt",
    "TaskRevision",
]
