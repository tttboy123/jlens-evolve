"""Freeze and run the r6 typed-operator local-Qwen experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import ContractError, LoopRevision, canonical_json, sha256_json
from .eval_manifest import EvaluationTaskSet
from .experiment import PairedExperimentRunner
from .operator_student import (
    MlxOperatorPlanGenerator,
    OperatorPlanAdapter,
    build_operator_conditions,
)
from .p1_local_feedback import load_frozen_p1_local_feedback_revision
from .target_selection import TargetSelectionManifest


def freeze_operator_skill_revision(
    *,
    strategy_response_path: Path,
    output_path: Path,
    parent_revision_id: str,
    source_round: int,
    pattern_revision_path: Path | None = None,
) -> dict[str, Any]:
    """Compile only DeepSeek's compact requirements into an inactive revision."""
    try:
        raw = strategy_response_path.read_bytes()
        strategy_wrapper = json.loads(raw)
        strategy = strategy_wrapper["response"]["strategy"]
        requirements = strategy["compiled_skill_requirements"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ContractError("operator strategy response is unreadable") from exc
    if (
        strategy_wrapper.get("holdout_task_ids_included") is not False
        or not isinstance(requirements, list)
        or not 1 <= len(requirements) <= 10
        or any(not isinstance(row, str) or not row.strip() for row in requirements)
    ):
        raise ContractError("operator strategy compact requirements are invalid")
    if pattern_revision_path is None:
        rules = "\n".join(
            f"{index}. {requirement.strip()}"
            for index, requirement in enumerate(requirements, 1)
        )
        domain_section = ""
        candidate_source = "deepseek-realization-strategy-compiler-v1"
        pattern_evidence: dict[str, Any] = {}
    else:
        pattern_revision = load_frozen_p1_local_feedback_revision(pattern_revision_path)
        marker = "## Pattern cards\n"
        end_marker = "\n\n## Commit gate"
        if marker not in pattern_revision.skill_text:
            raise ContractError("pattern revision has no Pattern cards section")
        pattern_body = pattern_revision.skill_text.split(marker, 1)[1]
        pattern_body = pattern_body.split(end_marker, 1)[0].strip()
        if not pattern_body:
            raise ContractError("pattern revision Pattern cards are empty")
        domain_section = f"\n\n## Pattern cards\n{pattern_body}"
        rules = (
            "1. Emit a concrete Defect, Trigger, and DesiredBoundary.\n"
            "2. Select the smallest system-catalog operator matching that boundary.\n"
            "3. Copy AST-equivalent selectors from the supplied symbol; use the "
            "correct expression/statement kind and occurrence.\n"
            "4. Reject no-op, copied, unrelated, or opposite-direction operations."
        )
        candidate_source = "pattern-card-operator-composite-v1"
        pattern_raw = pattern_revision_path.read_bytes()
        pattern_evidence = {
            "source_pattern_revision_file_sha256": hashlib.sha256(
                pattern_raw
            ).hexdigest(),
            "source_pattern_revision_fingerprint": pattern_revision.fingerprint,
        }
    skill_text = (
        "---\n"
        "name: typed-operator-realization\n"
        "project_local_only: true\n"
        "auto_install: false\n"
        "active: false\n"
        "---\n\n"
        "# Typed operator realization\n\n"
        "Apply these requirements after diagnosing the supplied source. The system "
        "operator catalog and JSON schema are authoritative. Do not emit a diff or "
        "complete definition.\n\n"
        f"{rules}"
        f"{domain_section}\n\n"
        "Use exact AST-equivalent selectors copied from the supplied symbol and the "
        "smallest operation that implements DesiredBoundary. Preserve unrelated "
        "code. This candidate is experimental, inactive, and unpromoted."
    )
    if len(skill_text) > 2_500:
        raise ContractError("compiled operator Skill exceeds 2500 characters")
    revision = LoopRevision.create(
        skill_id="p1-local-qwen-operator-skill",
        revision_id=f"p1-local-qwen-operator-skill-r{source_round:03d}",
        parent_revision_id=parent_revision_id,
        source_round=source_round,
        protocol="python-typed-operator-plan-v1",
        skill_text=skill_text,
        prompt_template="Return exactly one typed operator plan JSON object.",
        eval_note=(
            "Compiled from the authorized r6 realization strategy; inactive, "
            "gold-free, feedback-only, and unpromoted."
        ),
    )
    content = {
        "schema_version": 1,
        "candidate_source": candidate_source,
        "source_strategy_file_sha256": hashlib.sha256(raw).hexdigest(),
        "source_strategy_evidence_sha256": strategy_wrapper.get("evidence_sha256"),
        "requirement_count": len(requirements),
        "holdout_task_ids_included": False,
        "next_revision": revision.to_dict(),
        "candidate_status": "inactive",
        "auto_activate": False,
        "network_calls_performed": False,
        **pattern_evidence,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    output = output_path.resolve()
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError("operator Skill evidence is unreadable") from exc
        if existing != report:
            raise ContractError("frozen operator Skill evidence does not match")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report


def load_frozen_operator_skill_revision(path: Path) -> LoopRevision:
    """Load a verified inactive operator Skill revision."""
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("operator Skill evidence is unreadable") from exc
    if not isinstance(wrapper, dict):
        raise ContractError("operator Skill evidence must be an object")
    content = {key: value for key, value in wrapper.items() if key != "evidence_sha256"}
    if wrapper.get("evidence_sha256") != sha256_json(content):
        raise ContractError("operator Skill evidence sha256 mismatch")
    if (
        wrapper.get("candidate_source")
        not in {
            "deepseek-realization-strategy-compiler-v1",
            "pattern-card-operator-composite-v1",
            "round1-feedback-r010-compiler-v1",
            "feedback-pattern-card-trigger-refinement-v1",
        }
        or wrapper.get("holdout_task_ids_included") is not False
        or wrapper.get("candidate_status") != "inactive"
        or wrapper.get("auto_activate") is not False
        or wrapper.get("network_calls_performed") is not False
    ):
        raise ContractError("operator Skill evidence boundary is invalid")
    revision = LoopRevision.from_dict(wrapper.get("next_revision"))
    if (
        revision.protocol != "python-typed-operator-plan-v1"
        or "active: false" not in revision.skill_text
        or "auto_install: false" not in revision.skill_text
    ):
        raise ContractError("operator Skill revision is not explicitly inactive")
    return revision


def run_local_qwen_operator_p1(
    *,
    manifest_path: Path,
    skill_revision_path: Path,
    target_selection_path: Path,
    evidence_root: Path,
    workspace_root: Path,
    model_path: Path,
    max_cells: int | None,
    max_tokens: int = 1536,
    context_chars: int = 8_000,
    enable_thinking: bool = False,
    task_ids: set[str] | None = None,
    condition_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Run/resume the strict typed-operator baseline/taught comparison."""
    try:
        taskset = EvaluationTaskSet.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        target_selection = TargetSelectionManifest.from_dict(
            json.loads(target_selection_path.read_text(encoding="utf-8")),
            taskset=taskset,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("operator P1 inputs are unreadable") from exc
    taught_revision = load_frozen_operator_skill_revision(skill_revision_path)
    if not model_path.is_dir():
        raise ContractError(f"local Qwen model is missing: {model_path}")

    generator = MlxOperatorPlanGenerator(
        model_path=str(model_path.resolve()),
        max_tokens=max_tokens,
        max_context_chars=context_chars,
        enable_thinking=enable_thinking,
    )
    adapter = OperatorPlanAdapter(generator=generator)
    generation_config = adapter.experiment_config()
    conditions = build_operator_conditions(
        taught_skill=taught_revision.skill_text,
        parent_revision_id=taught_revision.revision_id,
        source_round=taught_revision.source_round,
        generation_config=generation_config,
    )
    if condition_ids is not None:
        known = {condition.condition_id for condition in conditions}
        if not condition_ids <= known:
            raise ContractError("unknown operator experiment condition filter")
        conditions = [
            condition
            for condition in conditions
            if condition.condition_id in condition_ids
        ]
    runner = PairedExperimentRunner(
        taskset=taskset,
        adapters={"operator": adapter},
        conditions=conditions,
        evidence_root=evidence_root,
        workspace_root=workspace_root,
        qualification_fingerprint=target_selection.fingerprint,
    )
    return runner.run(
        max_cells=max_cells,
        task_ids=task_ids,
        condition_ids=condition_ids,
    )
