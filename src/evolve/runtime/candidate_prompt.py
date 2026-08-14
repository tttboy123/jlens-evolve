"""Model transport that consumes a compiled candidate only on the taught arm."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from evolve.contracts import Cohort, ContractViolation, ExecutionPlan, canonical_json
from evolve.proposals import CompiledRevision


class CandidatePromptBackend(Protocol):
    remote: bool

    def infer(
        self,
        plan: ExecutionPlan,
        workspace: Mapping[str, Any],
        prompt: str,
    ) -> Mapping[str, Any]: ...


BasePromptBuilder = Callable[[ExecutionPlan, Mapping[str, Any]], str]


def compiled_candidate_prompt(compiled: CompiledRevision, task_id: str) -> str:
    """Render the single canonical runtime projection of a compiled candidate."""

    routes = dict(compiled.router.routes)
    operator_id = routes.get(task_id)
    if operator_id is None:
        raise ContractViolation("compiled Router has no route for taught task")
    if operator_id != compiled.operator.operator_id:
        raise ContractViolation("compiled Router selected another Operator")
    return canonical_json(
        {
            "candidate_id": compiled.change_set.candidate_id,
            "revision_id": compiled.change_set.revision_id,
            "protocol": compiled.skill.protocol,
            "prompt_template": compiled.skill.prompt_template,
            "skill_text": compiled.skill.skill_text,
            "operator": {
                "id": compiled.operator.operator_id,
                "kind": compiled.operator.kind,
                "arguments": list(compiled.operator.arguments),
                "instruction": compiled.operator.instruction,
            },
            "route": {"task_id": task_id, "operator_id": operator_id},
        }
    )


class CandidatePromptTransport:
    """Build a causal prompt without exposing candidate data to baseline."""

    def __init__(
        self,
        *,
        backend: CandidatePromptBackend,
        compiled: CompiledRevision | None,
        base_prompt_builder: BasePromptBuilder,
    ) -> None:
        self._backend = backend
        self._compiled = compiled
        self._base_prompt_builder = base_prompt_builder
        self.remote = backend.remote

    def infer(
        self, plan: ExecutionPlan, workspace: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if plan.task.cohort is not Cohort.FEEDBACK:
            raise ContractViolation("candidate prompt transport is feedback-only")
        base_prompt = self._base_prompt_builder(plan, workspace)
        if not isinstance(base_prompt, str) or not base_prompt.strip():
            raise ContractViolation("base prompt builder returned empty text")

        if plan.arm == "baseline":
            output = dict(self._backend.infer(plan, workspace, base_prompt))
            output.update(
                {
                    "candidate_consumed": False,
                    "candidate_bundle_sha256": None,
                    "candidate_revision_id": None,
                    "prompt_sha256": hashlib.sha256(
                        base_prompt.encode("utf-8")
                    ).hexdigest(),
                }
            )
            return output
        if plan.arm != "taught":
            raise ContractViolation("candidate prompt transport requires paired arms")
        if self._compiled is None:
            raise ContractViolation(
                "taught execution requires a compiled candidate; fallback is forbidden"
            )

        # Re-read and hash every artifact at the moment of taught execution.  The
        # baseline branch intentionally never evaluates this candidate state.
        compiled = CompiledRevision.load(self._compiled.root)
        if compiled != self._compiled:
            raise ContractViolation("compiled candidate identity changed")
        if plan.candidate_revision_id != compiled.change_set.revision_id:
            raise ContractViolation("taught plan candidate revision mismatch")
        candidate_prompt = compiled_candidate_prompt(compiled, plan.task.task_id)
        prompt = base_prompt + "\n\nCOMPILED-CANDIDATE:\n" + candidate_prompt
        output = dict(self._backend.infer(plan, workspace, prompt))
        output.update(
            {
                "candidate_consumed": True,
                "candidate_bundle_sha256": compiled.bundle_sha256,
                "candidate_revision_id": compiled.change_set.revision_id,
                "compiled_artifact_sha256": dict(compiled.artifact_sha256),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
        return output


__all__ = [
    "CandidatePromptBackend",
    "CandidatePromptTransport",
    "compiled_candidate_prompt",
]
