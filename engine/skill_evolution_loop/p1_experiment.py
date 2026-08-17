"""Real local-Qwen P1 condition wiring; no parent or network transport."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import ContractError, LoopRevision
from .eval_manifest import EvaluationTaskSet
from .experiment import ExperimentCondition, PairedExperimentRunner
from .mlx_student import MlxHunkGenerator, MlxStructuredGenerator
from .p1_local_feedback import load_frozen_p1_local_feedback_revision
from .p1_parent import load_frozen_p1_parent_revision
from .student_adapter import HunkStudentAdapter, StudentAdapter
from .target_selection import TargetSelectionManifest


def build_p1_conditions(
    skill_text: str,
    generation_configs: dict[str, dict[str, Any]] | None = None,
    *,
    structured_taught_revision: LoopRevision | None = None,
) -> list[ExperimentCondition]:
    """Keep task prompts fixed within each mechanism; vary only teaching text."""
    baseline_skill = (
        "No additional domain teaching is provided. Follow the output protocol and "
        "use only the supplied repository evidence."
    )
    definitions = [
        (
            "structured-baseline",
            "structured",
            "baseline",
            "structured-search-replace-v1",
            baseline_skill,
            "Return exactly one JSON object with file, search, replace, diagnostic.",
        ),
        (
            "structured-taught",
            "structured",
            "taught",
            "structured-search-replace-v1",
            skill_text,
            "Return exactly one JSON object with file, search, replace, diagnostic.",
        ),
        (
            "hunk-baseline",
            "hunk",
            "baseline",
            "unified-diff-hunk-v1",
            baseline_skill,
            "Return exactly one small unified diff for one allowed target.",
        ),
        (
            "hunk-taught",
            "hunk",
            "taught",
            "unified-diff-hunk-v1",
            skill_text,
            "Return exactly one small unified diff for one allowed target.",
        ),
    ]
    if structured_taught_revision is not None:
        structured_taught_revision.validate()
        if (
            structured_taught_revision.protocol != "structured-search-replace-v1"
            or structured_taught_revision.prompt_template
            != "Return exactly one JSON object with file, search, replace, diagnostic."
            or structured_taught_revision.skill_text != skill_text
        ):
            raise ContractError("P1 parent revision changed the paired prompt contract")
    conditions: list[ExperimentCondition] = []
    for (
        condition_id,
        mechanism,
        teaching,
        protocol,
        condition_skill,
        prompt_template,
    ) in definitions:
        revision = (
            structured_taught_revision
            if condition_id == "structured-taught"
            and structured_taught_revision is not None
            else LoopRevision.create(
                skill_id="p1-local-qwen-skill",
                revision_id=f"p1-{condition_id}-v1",
                parent_revision_id=None,
                source_round=0,
                protocol=protocol,
                skill_text=condition_skill,
                prompt_template=prompt_template,
                eval_note="P1 real local-Qwen paired mechanism comparison.",
            )
        )
        conditions.append(
            ExperimentCondition.create(
                condition_id=condition_id,
                mechanism=mechanism,
                teaching=teaching,
                revision=revision,
                generation_config=(generation_configs or {}).get(mechanism),
            )
        )
    return conditions


def run_local_qwen_p1(
    *,
    manifest_path: Path,
    skill_path: Path | None,
    skill_revision_path: Path | None = None,
    evidence_root: Path,
    workspace_root: Path,
    model_path: Path,
    max_cells: int | None,
    target_selection_path: Path | None = None,
    structured_max_tokens: int = 768,
    structured_context_chars: int = 80_000,
    hunk_max_tokens: int = 512,
    task_ids: set[str] | None = None,
    condition_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Run or resume P1 cells using only local MLX inference."""
    try:
        taskset = EvaluationTaskSet.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("P1 manifest is unreadable") from exc
    if (skill_path is None) == (skill_revision_path is None):
        raise ContractError("P1 requires exactly one Skill or parent revision")
    taught_revision = None
    try:
        if skill_revision_path is not None:
            try:
                wrapper = json.loads(skill_revision_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ContractError("P1 teaching revision is unreadable") from exc
            if wrapper.get("candidate_source") == "local-feedback-compiler-v1":
                taught_revision = load_frozen_p1_local_feedback_revision(
                    skill_revision_path
                )
            else:
                taught_revision = load_frozen_p1_parent_revision(skill_revision_path)
            skill_text = taught_revision.skill_text
        else:
            assert skill_path is not None
            skill_text = skill_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError("P1 teaching Skill is unreadable") from exc
    if not model_path.is_dir():
        raise ContractError(f"local Qwen model is missing: {model_path}")
    qualification_fingerprint = None
    if target_selection_path is not None:
        try:
            target_selection = TargetSelectionManifest.from_dict(
                json.loads(target_selection_path.read_text(encoding="utf-8")),
                taskset=taskset,
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError("P1 target selection manifest is unreadable") from exc
        qualification_fingerprint = target_selection.fingerprint
    structured = MlxStructuredGenerator(
        model_path=str(model_path.resolve()),
        max_tokens=structured_max_tokens,
        max_context_chars=structured_context_chars,
        max_structural_repairs=2,
        use_grounding_plan=True,
        use_semantic_critic=True,
    )
    hunk = MlxHunkGenerator(
        model_path=str(model_path.resolve()),
        max_tokens=hunk_max_tokens,
    )
    adapters = {
        "structured": StudentAdapter(generator=structured),
        "hunk": HunkStudentAdapter(generator=hunk),
    }
    generation_configs = {
        mechanism: adapter.experiment_config()
        for mechanism, adapter in adapters.items()
    }
    conditions = build_p1_conditions(
        skill_text,
        generation_configs,
        structured_taught_revision=taught_revision,
    )
    if condition_ids is not None:
        known = {condition.condition_id for condition in conditions}
        if not condition_ids <= known:
            raise ContractError("unknown experiment condition filter")
        conditions = [
            condition
            for condition in conditions
            if condition.condition_id in condition_ids
        ]
    runner = PairedExperimentRunner(
        taskset=taskset,
        adapters=adapters,
        conditions=conditions,
        evidence_root=evidence_root,
        workspace_root=workspace_root,
        qualification_fingerprint=qualification_fingerprint,
    )
    return runner.run(
        max_cells=max_cells,
        task_ids=task_ids,
        condition_ids=condition_ids,
    )
