"""Hash-verified adapter from complete AgentProgram revisions to model execution."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Protocol

from evolve.agent_program import AgentProgramRevision, AgentProgramViolation
from evolve.contracts import ContractViolation, ExecutionPlan


class AgentProgramExecutor(Protocol):
    """Boundary implemented by a real local or remote AgentProgram executor."""

    remote: bool

    def infer_program(
        self,
        revision: AgentProgramRevision,
        plan: ExecutionPlan,
        workspace: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class HashVerifiedAgentProgramTransport:
    """Load and verify a full revision before delegating one model execution."""

    def __init__(
        self,
        revision_roots: Mapping[str, str | Path],
        *,
        executor: AgentProgramExecutor,
    ) -> None:
        if not revision_roots:
            raise ContractViolation("AgentProgram transport requires revision roots")
        self._revision_roots = {
            revision_id: Path(root).resolve()
            for revision_id, root in revision_roots.items()
        }
        if any(
            not isinstance(revision_id, str) or not revision_id.strip()
            for revision_id in self._revision_roots
        ):
            raise ContractViolation("AgentProgram revision root identity is invalid")
        if not isinstance(getattr(executor, "remote", None), bool):
            raise ContractViolation("AgentProgram executor locality is invalid")
        self._executor = executor
        self.remote = executor.remote

    def infer(
        self, plan: ExecutionPlan, workspace: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if plan.metadata.get("execution_profile") != "live":
            raise ContractViolation("live AgentProgram execution profile is missing")
        root = self._revision_roots.get(plan.candidate_revision_id)
        if root is None:
            raise ContractViolation("planned AgentProgram revision is unavailable")
        try:
            revision = AgentProgramRevision.load(root)
        except AgentProgramViolation as error:
            raise ContractViolation("AgentProgram revision failed hash verification") from error
        if revision.revision_id != plan.candidate_revision_id:
            raise ContractViolation("AgentProgram revision root identity drift")
        expected = {
            "program_bundle_sha256": revision.bundle_sha256,
            "program_prompt_sha256": revision.artifact_hash("PROGRAM-PROMPT.txt"),
            "program_context_sha256": revision.artifact_hash("CONTEXT.json"),
            "program_tool_policy_sha256": revision.artifact_hash("TOOL-POLICY.json"),
            "program_capabilities_sha256": revision.artifact_hash(
                "CAPABILITIES.json"
            ),
        }
        for name, digest in expected.items():
            if plan.metadata.get(name) != digest:
                raise ContractViolation(f"planned {name} identity drift")
        if plan.arm == "candidate" and plan.metadata.get(
            "parent_revision_id"
        ) != revision.parent_revision_id:
            raise ContractViolation("planned AgentProgram parent lineage drift")
        if plan.arm == "search-parent" and plan.metadata.get(
            "parent_revision_id"
        ) is not None:
            raise ContractViolation("search-parent tournament lineage is invalid")
        if workspace.get("task_revision_id") != plan.task.revision_id:
            raise ContractViolation("AgentProgram workspace task identity drift")
        if workspace.get("task_source_sha256") != plan.task.source_sha256:
            raise ContractViolation("AgentProgram workspace source identity drift")
        output = self._executor.infer_program(revision, plan, workspace)
        if not isinstance(output, Mapping):
            raise ContractViolation("AgentProgram executor output must be an object")
        result = dict(output)
        protected: dict[str, Any] = {
            "execution_scope": "live",
            "program_id": revision.program_id,
            "revision_id": revision.revision_id,
            "parent_revision_id": revision.parent_revision_id,
            **expected,
        }
        for name, value in protected.items():
            if name in result and result[name] != value:
                raise ContractViolation(f"AgentProgram executor {name} drift")
            result[name] = value
        patch = result.get("patch")
        patch_sha256 = result.get("patch_sha256")
        if (
            not isinstance(patch, str)
            or not isinstance(patch_sha256, str)
            or hashlib.sha256(patch.encode("utf-8")).hexdigest() != patch_sha256
            or result.get("prediction_sha256") != patch_sha256
        ):
            raise ContractViolation("AgentProgram executor patch identity drift")
        return result


__all__ = ["AgentProgramExecutor", "HashVerifiedAgentProgramTransport"]
