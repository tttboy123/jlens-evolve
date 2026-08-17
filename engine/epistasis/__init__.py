"""Epistasis / diversity-collapse experiment suite for the RSI operator set.

Experiments A-E test whether the RSI plateau is caused by operator-set coverage,
composition (emergence), lineage composition, epistatic interaction, or a
cross-task validation threshold.  The harness is model-agnostic: the default
``deterministic`` mode isolates operator/landscape effects without any model
call; an OpenAI-compatible ``llm`` mode lets any model vendor plug in their own
endpoint and reuse the same evidence and correlation machinery.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .operators import (
    OPERATOR_IDS,
    OpResult,
    apply_operators,
)
from .schema import TaskSchema
from .tasks import TaskSpec, generate_synthetic_task, load_task

__all__ = [
    "OPERATOR_IDS",
    "OpResult",
    "TaskSchema",
    "TaskSpec",
    "apply_operators",
    "generate_synthetic_task",
    "load_task",
]
