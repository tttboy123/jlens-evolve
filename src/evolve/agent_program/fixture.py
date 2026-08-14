"""Deterministic, zero-network executor for complete AgentProgram fixtures."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

from evolve.contracts import Cohort, ExecutionPlan, canonical_json

from .revision import AgentProgramRevision, AgentProgramViolation


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class DeterministicFixtureAgentProgramTransport:
    """Execute an allowlisted local fixture without a model or network call."""

    remote = False

    def __init__(self, revision_roots: Mapping[str, str | Path]) -> None:
        if not revision_roots:
            raise AgentProgramViolation("fixture transport requires program revisions")
        self._revision_roots = {
            revision_id: Path(root).resolve()
            for revision_id, root in revision_roots.items()
        }
        if not all(
            isinstance(revision_id, str) and revision_id.strip()
            for revision_id in self._revision_roots
        ):
            raise AgentProgramViolation("fixture revision identities are invalid")

    def infer(
        self, plan: ExecutionPlan, workspace: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if plan.task.cohort is not Cohort.FEEDBACK:
            raise AgentProgramViolation("fixture AgentProgram is feedback-only")
        if plan.metadata.get("execution_profile") != "fixture":
            raise AgentProgramViolation("fixture execution profile is missing")
        root = self._revision_roots.get(plan.candidate_revision_id)
        if root is None:
            raise AgentProgramViolation("planned AgentProgram revision is unavailable")
        revision = AgentProgramRevision.load(root)
        if revision.revision_id != plan.candidate_revision_id:
            raise AgentProgramViolation("planned AgentProgram revision identity drift")
        expected_metadata = {
            "program_bundle_sha256": revision.bundle_sha256,
            "program_prompt_sha256": revision.artifact_hash("PROGRAM-PROMPT.txt"),
            "program_context_sha256": revision.artifact_hash("CONTEXT.json"),
            "program_tool_policy_sha256": revision.artifact_hash("TOOL-POLICY.json"),
            "program_capabilities_sha256": revision.artifact_hash(
                "CAPABILITIES.json"
            ),
        }
        for name, expected in expected_metadata.items():
            if plan.metadata.get(name) != expected:
                raise AgentProgramViolation(f"planned {name} identity drift")
        if workspace.get("task_revision_id") != plan.task.revision_id:
            raise AgentProgramViolation("fixture workspace task identity drift")
        if workspace.get("task_source_sha256") != plan.task.source_sha256:
            raise AgentProgramViolation("fixture workspace source identity drift")
        score = revision.context.get("fixture_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise AgentProgramViolation("fixture_score must be finite numeric data")
        projection = {
            "program_id": revision.program_id,
            "revision_id": revision.revision_id,
            "parent_revision_id": revision.parent_revision_id,
            "program_prompt": revision.program_prompt,
            "program_context": dict(revision.context),
            "program_tool_policy": list(revision.tool_policy),
            "program_capability_revision_ids": list(
                revision.capability_revision_ids
            ),
            "task_revision_id": plan.task.revision_id,
            "task_source_sha256": plan.task.source_sha256,
        }
        projection_sha256 = _sha256_text(canonical_json(projection))
        patch = (
            "diff --git a/FIXTURE-AGENT-PROGRAM.txt "
            "b/FIXTURE-AGENT-PROGRAM.txt\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/FIXTURE-AGENT-PROGRAM.txt\n"
            f"@@ -0,0 +1 @@\n+{projection_sha256}\n"
        )
        patch_sha256 = _sha256_text(patch)
        return {
            **projection,
            "execution_scope": "fixture",
            "program_bundle_sha256": revision.bundle_sha256,
            "program_prompt_sha256": expected_metadata["program_prompt_sha256"],
            "program_context_sha256": expected_metadata["program_context_sha256"],
            "program_tool_policy_sha256": expected_metadata[
                "program_tool_policy_sha256"
            ],
            "program_capabilities_sha256": expected_metadata[
                "program_capabilities_sha256"
            ],
            "program_projection_sha256": projection_sha256,
            "fixture_score": float(score),
            "patch": patch,
            "patch_sha256": patch_sha256,
            "prediction_sha256": patch_sha256,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_cny": 0,
            "network_calls": 0,
        }


__all__ = ["DeterministicFixtureAgentProgramTransport"]
