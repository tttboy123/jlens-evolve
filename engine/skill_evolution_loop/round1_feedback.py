"""Freeze complete, holdout-free Round 1 evidence for the next teacher round."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from teacher_api import TeacherClient, TeacherProvider, TeacherSample

from .composition import load_frozen_attempt
from .contracts import ContractError, LoopRevision, canonical_json, sha256_json
from .evolution_catalog import EvolutionCatalog
from .ledger import ParentCallLedger
from .p1_native import _read_native_cell
from .p1_parent import P1ParentCallAuthorization
from .round1_run import _load_routes, _load_taskset

_CAMPAIGN_LIMIT = 3_000_000
_STRATEGY_FIELDS = frozenset(
    {
        "failure_diagnosis",
        "mechanism_revision",
        "operator_skill_requirements",
        "span_skill_requirements",
        "verification_changes",
        "next_experiment",
    }
)
_OPERATOR_NAMES = frozenset(
    {
        "replace_condition",
        "replace_expression",
        "replace_statement",
        "insert_method",
        "replace_method_body",
        "insert_assignment_before",
        "replace_constant",
        "align_trailing_defaults",
        "normalize_inline_wrapper_boundaries",
    }
)


def augment_feedback_request_with_catalog_context(
    wrapper: dict[str, Any],
    *,
    catalog: EvolutionCatalog,
    capability_tags: tuple[str, ...] = (),
    task_tags: tuple[str, ...] = (),
    failure_mode_tags: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Bind bounded dedup context into the request before authorization."""

    request = wrapper.get("request")
    if not isinstance(request, dict):
        raise ContractError("feedback request is invalid")
    if "evolution_catalog_context" in request:
        raise ContractError("feedback request already has catalog context")
    context = catalog.proposal_context(
        capability_tags=capability_tags,
        task_tags=task_tags,
        failure_mode_tags=failure_mode_tags,
    )
    augmented_request = {**request, "evolution_catalog_context": context}
    augmented = {
        **wrapper,
        "request": augmented_request,
        "request_sha256": sha256_json(augmented_request),
        "evolution_catalog_query_fingerprint": context["query_fingerprint"],
    }
    augmented.pop("evidence_sha256", None)
    return {**augmented, "evidence_sha256": sha256_json(augmented)}


_OPERATOR_NAME_TOKEN = re.compile(
    r"\b(?:(?:replace|insert|align|normalize)_[a-z0-9_]+|delete)\b"
)
_CATALOG_ONLY_REQUIREMENT = (
    "Use only the framework-supplied operator catalog; never invent operator names."
)
_SPAN_SCHEMA_MARKERS = (
    "confidence",
    "edit_semantics",
    "never include plans",
    "predicate_path",
    "source_span_confidence",
    "target_files",
    "target_symbol",
)
_SPAN_SCHEMA_REQUIREMENT = (
    "Use only the framework-supplied span bundle schema; never invent or remove fields."
)
_SHARED_EDIT_SEMANTIC_MARKERS = (
    "boundary",
    "conditional",
    "preserve",
    "state",
    "truth table",
)


def compile_round1_feedback_skills(
    *,
    strategy_response_path: Path,
    parent_operator_skill_path: Path,
    operator_output_path: Path,
    span_output_path: Path,
    source_round: int,
) -> dict[str, Any]:
    """Compile one verified teacher response into two inactive r010 Skills."""

    if type(source_round) is not int or source_round < 1:
        raise ContractError("Round 1 feedback source_round must be positive")
    response = _load_evidence(
        strategy_response_path.resolve(), "Round 1 feedback strategy response"
    )
    if (
        response.get("candidate_status") != "advisory_inactive"
        or response.get("auto_apply") is not False
        or response.get("holdout_cells_included") is not False
    ):
        raise ContractError("Round 1 feedback strategy boundary is invalid")
    strategy = response.get("strategy")
    if not isinstance(strategy, dict) or set(strategy) != _STRATEGY_FIELDS:
        raise ContractError("Round 1 feedback strategy fields are invalid")
    from .p1_operator import load_frozen_operator_skill_revision

    parent = load_frozen_operator_skill_revision(parent_operator_skill_path.resolve())
    operator_requirements, compiler_normalizations = _compile_operator_requirements(
        strategy.get("operator_skill_requirements")
    )
    raw_span_requirements = _compile_requirements(
        strategy.get("span_skill_requirements"), "span"
    )
    raw_operator_requirements = _compile_requirements(
        strategy.get("operator_skill_requirements"), "operator"
    )
    shared_edit_requirements = tuple(
        requirement
        for requirement in raw_operator_requirements
        if any(
            marker in requirement.casefold() for marker in _SHARED_EDIT_SEMANTIC_MARKERS
        )
        and not _OPERATOR_NAME_TOKEN.search(requirement)
    )
    combined_span_requirements = (
        *shared_edit_requirements,
        *raw_span_requirements,
    )[:10]
    span_requirements, span_compiler_normalizations = _compile_span_requirements(
        list(combined_span_requirements)
    )
    operator_skill = _render_feedback_skill(
        name="typed-operator-realization-r010",
        title="Typed operator realization r010",
        preamble=(
            "Localize to one supplied file and symbol, copy one exact source node, "
            "then classify the edit before choosing an operator."
        ),
        requirements=operator_requirements,
    )
    span_skill = _render_feedback_skill(
        name="exact-span-bundle-realization-r010",
        title="Exact-span bundle realization r010",
        preamble=(
            "Localize to supplied source, then emit one bounded unique exact-span "
            "replacement per selected file."
        ),
        requirements=span_requirements,
    )
    strategy_file_sha = hashlib.sha256(
        strategy_response_path.resolve().read_bytes()
    ).hexdigest()
    revision = LoopRevision.create(
        skill_id="p1-local-qwen-operator-skill",
        revision_id=f"p1-local-qwen-operator-skill-r{source_round:03d}",
        parent_revision_id=parent.revision_id,
        source_round=source_round,
        protocol="python-typed-operator-plan-v1",
        skill_text=operator_skill,
        prompt_template="Return exactly one typed operator plan JSON object.",
        eval_note=(
            "Compiled from the authorized Round 1 feedback strategy; inactive, "
            "gold-free, feedback-only, and unpromoted."
        ),
    )
    operator_content = {
        "schema_version": 1,
        "candidate_source": "round1-feedback-r010-compiler-v1",
        "source_strategy_file_sha256": strategy_file_sha,
        "source_strategy_evidence_sha256": response["evidence_sha256"],
        "requirement_count": len(operator_requirements),
        "compiler_normalizations": list(compiler_normalizations),
        "holdout_task_ids_included": False,
        "next_revision": revision.to_dict(),
        "candidate_status": "inactive",
        "auto_activate": False,
        "network_calls_performed": False,
    }
    operator_report = {
        **operator_content,
        "evidence_sha256": sha256_json(operator_content),
    }
    _freeze(operator_output_path.resolve(), operator_report)
    span_content = {
        "schema_version": 1,
        "source_strategy_file_sha256": strategy_file_sha,
        "source_strategy_evidence_sha256": response["evidence_sha256"],
        "source_operator_skill_file_sha256": hashlib.sha256(
            operator_output_path.resolve().read_bytes()
        ).hexdigest(),
        "source_operator_skill_fingerprint": revision.fingerprint,
        "compiler": "round1-feedback-span-r010-compiler-v1",
        "skill_text": span_skill,
        "compiler_normalizations": list(span_compiler_normalizations),
        "candidate_status": "inactive",
        "auto_activate": False,
        "new_domain_knowledge_added": False,
        "network_calls_performed": False,
    }
    span_report = {**span_content, "evidence_sha256": sha256_json(span_content)}
    _freeze(span_output_path.resolve(), span_report)
    content = {
        "schema_version": 1,
        "source_round": source_round,
        "source_strategy_evidence_sha256": response["evidence_sha256"],
        "operator_skill_evidence_sha256": operator_report["evidence_sha256"],
        "span_skill_evidence_sha256": span_report["evidence_sha256"],
        "candidate_status": "inactive",
        "auto_activate": False,
        "holdout_cells_included": False,
    }
    return {**content, "evidence_sha256": sha256_json(content)}


def _compile_requirements(value: Any, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 10
        or any(not isinstance(row, str) or not row.strip() for row in value)
    ):
        raise ContractError(f"Round 1 {label} Skill requirements are invalid")
    return tuple(row.strip() for row in value)


def _compile_operator_requirements(
    value: Any,
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    requirements = _compile_requirements(value, "operator")
    projected: list[str] = []
    normalizations: list[dict[str, Any]] = []
    for requirement in requirements:
        mentioned = set(_OPERATOR_NAME_TOKEN.findall(requirement))
        unsupported = sorted(mentioned - _OPERATOR_NAMES)
        compiled = requirement
        if unsupported:
            compiled = _CATALOG_ONLY_REQUIREMENT
            normalizations.append(
                {
                    "reason": "unsupported-operator-name",
                    "unsupported": unsupported,
                }
            )
        if compiled not in projected:
            projected.append(compiled)
    return tuple(projected), tuple(normalizations)


def _compile_span_requirements(
    value: Any,
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    requirements = _compile_requirements(value, "span")
    projected: list[str] = []
    normalizations: list[dict[str, Any]] = []
    for requirement in requirements:
        lowered = requirement.casefold()
        markers = sorted(marker for marker in _SPAN_SCHEMA_MARKERS if marker in lowered)
        compiled = requirement
        if markers:
            compiled = _SPAN_SCHEMA_REQUIREMENT
            normalizations.append(
                {"reason": "unsupported-span-schema", "markers": markers}
            )
        if compiled not in projected:
            projected.append(compiled)
    return tuple(projected), tuple(normalizations)


def _render_feedback_skill(
    *, name: str, title: str, preamble: str, requirements: tuple[str, ...]
) -> str:
    rules = "\n".join(
        f"{index}. {requirement}" for index, requirement in enumerate(requirements, 1)
    )
    skill = (
        "---\n"
        f"name: {name}\n"
        "project_local_only: true\n"
        "auto_install: false\n"
        "active: false\n"
        "---\n\n"
        f"# {title}\n\n"
        f"{preamble}\n\n"
        f"{rules}\n\n"
        "If the required source cannot be copied and matched uniquely, report "
        "unresolved instead of guessing. Preserve unrelated behavior. This "
        "candidate is experimental, inactive, and unpromoted."
    )
    if len(skill) > 2_500:
        raise ContractError("compiled Round 1 feedback Skill exceeds 2500 characters")
    return skill


def freeze_round1_feedback_request(
    *,
    taskset_path: Path,
    routes_path: Path,
    experiment_root: Path,
    native_root: Path,
    output_path: Path,
    post_holdout_projection: bool = False,
) -> dict[str, Any]:
    """Bind every feedback A/B cell and native receipt to one request.

    The holdout partition is used only to prove that the manifest is a frozen
    3+3-or-larger split. No holdout task identity, instruction, attempt, or receipt is
    serialized across the teacher boundary.
    """

    taskset = _load_taskset(taskset_path.resolve())
    routes = _load_routes(routes_path.resolve(), taskset)
    feedback = tuple(task for task in taskset.tasks if task.cohort == "feedback")
    holdout = tuple(task for task in taskset.tasks if task.cohort == "holdout")
    if len(feedback) < 3 or len(holdout) < 3:
        raise ContractError("Round 1 feedback request requires at least 3+3 tasks")

    feedback_ids = {task.task_id for task in feedback}
    source_holdout_evidence_present = _partition_evidence_present(
        experiment_root.resolve(), feedback_ids
    )
    if source_holdout_evidence_present and not post_holdout_projection:
        raise ContractError("holdout experiment evidence is prohibited")
    _reject_partition_evidence(
        native_root.resolve(), feedback_ids, "holdout native evidence"
    )

    native_summary = _load_native_summary(native_root.resolve() / "SUMMARY.json")
    planned_feedback_cells = len(feedback) * 2
    if (
        native_summary.get("status") != "complete"
        or native_summary.get("evaluation_scope") != "round1-feedback-only"
        or native_summary.get("planned_cells") != planned_feedback_cells
        or native_summary.get("generated_feedback_cells") != planned_feedback_cells
        or native_summary.get("completed_cells") != planned_feedback_cells
        or native_summary.get("holdout_cells_opened") is not False
        or native_summary.get("full_capability_gate_evaluated") is not False
    ):
        raise ContractError(
            "Round 1 request requires complete feedback native evidence"
        )
    if native_summary.get("taskset_fingerprint") != taskset.fingerprint:
        raise ContractError("Round 1 feedback native TaskSet mismatch")

    experiment = experiment_root.resolve()
    native = native_root.resolve()
    reason_counts: dict[str, Counter[str]] = {}
    current_skills: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    cell_count = 0
    for task in feedback:
        mechanism = routes[task.task_id]
        arms: dict[str, Any] = {}
        for teaching in ("baseline", "taught"):
            condition_id = f"{mechanism}-{teaching}"
            cell_dir = experiment / "cells" / task.task_id / condition_id
            attempt = load_frozen_attempt(cell_dir / "ATTEMPT.json")
            task_record = attempt.get("task", {})
            condition = attempt.get("condition", {})
            if (
                attempt.get("taskset_fingerprint") != taskset.fingerprint
                or task_record.get("task_id") != task.task_id
                or task_record.get("cohort") != "feedback"
                or condition.get("condition_id") != condition_id
                or condition.get("mechanism") != mechanism
                or condition.get("teaching") != teaching
            ):
                raise ContractError("Round 1 feedback attempt boundary mismatch")
            raw_path = cell_dir / "raw-output.txt"
            try:
                raw_output = raw_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ContractError(
                    "Round 1 feedback raw output is unreadable"
                ) from exc
            raw_sha = hashlib.sha256(raw_output.encode()).hexdigest()
            recorded_raw_sha = attempt.get("attempt", {}).get("raw_output_sha256")
            artifact_raw_sha = attempt.get("artifact_sha256", {}).get("raw-output.txt")
            if recorded_raw_sha != raw_sha or (
                artifact_raw_sha is not None and artifact_raw_sha != raw_sha
            ):
                raise ContractError("Round 1 feedback raw output sha256 mismatch")

            native_cell = _read_native_cell(
                native / "cells" / task.task_id / condition_id
            )
            if (
                native_cell.get("taskset_fingerprint") != taskset.fingerprint
                or native_cell.get("task_id") != task.task_id
                or native_cell.get("cohort") != "feedback"
                or native_cell.get("condition_id") != condition_id
                or native_cell.get("experiment_cell_sha256")
                != attempt.get("evidence_sha256")
                or native_cell.get("holdout_cells_opened") is not False
            ):
                raise ContractError("Round 1 feedback native cell boundary mismatch")

            outcome = attempt.get("attempt", {})
            reason = outcome.get("failure_reason") or (
                "native-resolved"
                if native_cell.get("outcome", {}).get("resolved") is True
                else "native-unresolved"
            )
            reason_counts.setdefault(condition_id, Counter())[str(reason)] += 1
            arms[teaching] = {
                "condition_id": condition_id,
                "experiment_cell_sha256": attempt["evidence_sha256"],
                "native_cell_sha256": native_cell["evidence_sha256"],
                "structural_valid": outcome.get("structural_valid"),
                "failure_reason": outcome.get("failure_reason"),
                "detail": outcome.get("detail"),
                "raw_output": raw_output,
                "raw_output_sha256": raw_sha,
                "patch_sha256": outcome.get("patch_sha256"),
                "native_outcome": native_cell.get("outcome"),
                "native_report_sha256": (native_cell.get("native_report") or {}).get(
                    "sha256"
                ),
            }
            if teaching == "taught":
                revision = condition.get("revision")
                if not isinstance(revision, dict):
                    raise ContractError("Round 1 taught revision is invalid")
                prior = current_skills.setdefault(mechanism, revision)
                if prior != revision:
                    raise ContractError(
                        "Round 1 mechanism uses multiple taught revisions"
                    )
            cell_count += 1
        failures.append(
            {
                "task_id": task.task_id,
                "mechanism": mechanism,
                "issue": task.instruction,
                "allowed_targets": list(task.allowed_targets),
                "baseline": arms["baseline"],
                "taught": arms["taught"],
            }
        )

    if cell_count != planned_feedback_cells:
        raise ContractError("Round 1 feedback request did not bind every feedback cell")
    constraints = [
        "Use only these feedback tasks; never request or infer holdout evidence.",
        "Do not modify model weights, tests, native harnesses, or the frozen engine.",
        "Keep candidate Skills inactive and under the Student injection limit.",
        "Separate mechanism improvements from causal Skill A/B claims.",
        "Official native resolved is the sole capability credit.",
    ]
    if source_holdout_evidence_present:
        constraints.append(
            "The prior holdout is burned and must never be reused after this feedback-only projection."
        )
    request = {
        "schema_version": 1,
        "request_type": "round1-feedback-skill-evolution-v1",
        "objective": (
            "Revise the inactive Skill/action-space strategy so the frozen local 4B "
            "Student converts diagnosis into native-resolving patches."
        ),
        "feedback_gain_count": native_summary.get("feedback_gain_count"),
        "feedback_gain_gate_passed": native_summary.get("feedback_gain_gate_passed"),
        "condition_failure_counts": {
            condition: dict(sorted(counts.items()))
            for condition, counts in sorted(reason_counts.items())
        },
        "current_inactive_skills": current_skills,
        "failures": failures,
        "constraints": constraints,
    }
    request_sha = sha256_json(request)
    content = {
        "schema_version": 1,
        "request": request,
        "request_sha256": request_sha,
        "taskset_fingerprint": taskset.fingerprint,
        "native_summary_sha256": native_summary["summary_sha256"],
        "native_cell_evidence_fingerprint": native_summary["cell_evidence_fingerprint"],
        "feedback_task_count": len(failures),
        "feedback_cell_count": cell_count,
        "holdout_cells_included": False,
        "source_holdout_evidence_present": source_holdout_evidence_present,
        "current_holdout_reuse_prohibited": source_holdout_evidence_present,
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    _freeze(output_path.resolve(), report)
    return report


def freeze_round1_realization_feedback_request(
    *,
    taskset_path: Path,
    routes_path: Path,
    experiment_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze completed feedback-only structural failures before native eligibility."""

    taskset = _load_taskset(taskset_path.resolve())
    routes = _load_routes(routes_path.resolve(), taskset)
    feedback = tuple(task for task in taskset.tasks if task.cohort == "feedback")
    feedback_ids = {task.task_id for task in feedback}
    experiment = experiment_root.resolve()
    _reject_partition_evidence(experiment, feedback_ids, "holdout experiment evidence")
    failures: list[dict[str, Any]] = []
    current_skills: dict[str, dict[str, Any]] = {}
    for task in feedback:
        mechanism = routes[task.task_id]
        arms: dict[str, Any] = {}
        complete = True
        for teaching in ("baseline", "taught"):
            condition_id = f"{mechanism}-{teaching}"
            cell_dir = experiment / "cells" / task.task_id / condition_id
            attempt_path = cell_dir / "ATTEMPT.json"
            if not attempt_path.exists():
                complete = False
                break
            attempt = load_frozen_attempt(attempt_path)
            task_record = attempt.get("task", {})
            condition = attempt.get("condition", {})
            if (
                attempt.get("taskset_fingerprint") != taskset.fingerprint
                or task_record.get("task_id") != task.task_id
                or task_record.get("cohort") != "feedback"
                or condition.get("condition_id") != condition_id
                or condition.get("teaching") != teaching
            ):
                raise ContractError("Round 1 realization feedback boundary mismatch")
            outcome = attempt.get("attempt", {})
            traces: list[dict[str, Any]] = []
            for trace in attempt.get("generation_trace", []):
                if not isinstance(trace, dict) or not isinstance(
                    trace.get("path"), str
                ):
                    raise ContractError("Round 1 realization trace is invalid")
                raw_path = cell_dir / trace["path"]
                raw = raw_path.read_text(encoding="utf-8")
                if hashlib.sha256(raw.encode()).hexdigest() != trace.get("sha256"):
                    raise ContractError("Round 1 realization trace sha256 mismatch")
                trace_row = {
                    "candidate_id": trace.get("kind"),
                    "raw_output": raw,
                    "raw_output_sha256": trace["sha256"],
                    "stage_result": trace.get("stage_result"),
                }
                traces.append(trace_row)
                prompt_path_value = trace.get("prompt_path")
                if prompt_path_value is not None:
                    if not isinstance(prompt_path_value, str):
                        raise ContractError(
                            "Round 1 realization prompt path is invalid"
                        )
                    prompt_path = cell_dir / prompt_path_value
                    prompt_bytes = prompt_path.read_bytes()
                    try:
                        prompt = prompt_bytes.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise ContractError(
                            "Round 1 realization prompt is not UTF-8"
                        ) from exc
                    prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
                    if prompt_sha256 != trace.get("prompt_sha256"):
                        raise ContractError(
                            "Round 1 realization prompt sha256 mismatch"
                        )
                    trace_row.update(
                        {
                            "generation_prompt": prompt,
                            "generation_prompt_sha256": prompt_sha256,
                        }
                    )
            arms[teaching] = {
                "condition_id": condition_id,
                "experiment_cell_sha256": attempt["evidence_sha256"],
                "structural_valid": outcome.get("structural_valid"),
                "failure_reason": outcome.get("failure_reason"),
                "detail": outcome.get("detail"),
                "patch_sha256": outcome.get("patch_sha256"),
                "generation_candidates": traces,
                "shared_context": attempt.get("shared_context"),
            }
            if teaching == "taught":
                revision = condition.get("revision")
                if not isinstance(revision, dict):
                    raise ContractError("Round 1 taught revision is invalid")
                prior = current_skills.setdefault(mechanism, revision)
                if prior != revision:
                    raise ContractError(
                        "Round 1 mechanism uses multiple taught revisions"
                    )
        if complete:
            failures.append(
                {
                    "task_id": task.task_id,
                    "mechanism": mechanism,
                    "issue": task.instruction,
                    "allowed_targets": list(task.allowed_targets),
                    "baseline": arms["baseline"],
                    "taught": arms["taught"],
                }
            )
    if not failures:
        raise ContractError("Round 1 realization feedback has no complete pair")
    request = {
        "schema_version": 1,
        "request_type": "round1-realization-feedback-v1",
        "objective": (
            "Revise the inactive Skill and typed edit interface so a frozen local "
            "4B Student converts the shared diagnosis/localization into a "
            "structurally valid, native-eligible repair."
        ),
        "current_inactive_skills": current_skills,
        "failures": failures,
        "constraints": [
            "Feedback-only; never request or infer holdout or gold evidence.",
            "Treat frozen diagnosis, target files, and qualified symbols as framework state.",
            "Separate JSON/selector/operator failures from semantic boundary failures.",
            "Keep candidate Skills inactive and do not modify tests, native judges, or the frozen engine.",
            "Official native resolved remains the sole capability credit.",
        ],
    }
    request_sha = sha256_json(request)
    content = {
        "schema_version": 1,
        "request": request,
        "request_sha256": request_sha,
        "taskset_fingerprint": taskset.fingerprint,
        "feedback_task_count": len(failures),
        "feedback_cell_count": len(failures) * 2,
        "native_receipts_included": False,
        "holdout_cells_included": False,
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    _freeze(output_path.resolve(), report)
    return report


def freeze_round1_targeted_native_feedback_request(
    *,
    taskset_path: Path,
    routes_path: Path,
    experiment_root: Path,
    native_root: Path,
    task_ids: tuple[str, ...],
    output_path: Path,
    prior_request_paths: tuple[Path, ...] = (),
    disproven_mechanism_capabilities: tuple[dict[str, str], ...] = (),
) -> dict[str, Any]:
    """Freeze explicit completed feedback A/B native pairs without opening holdout."""

    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ContractError(
            "targeted native feedback tasks must be non-empty and unique"
        )
    capability_fields = {
        "capability_id",
        "implementation_sha256",
        "test_sha256",
        "native_conclusion",
    }
    for capability in disproven_mechanism_capabilities:
        if (
            not isinstance(capability, dict)
            or set(capability) != capability_fields
            or not isinstance(capability["capability_id"], str)
            or not capability["capability_id"].strip()
            or not isinstance(capability["native_conclusion"], str)
            or not capability["native_conclusion"].strip()
            or any(
                not isinstance(capability[field], str)
                or re.fullmatch(r"[0-9a-f]{64}", capability[field]) is None
                for field in ("implementation_sha256", "test_sha256")
            )
        ):
            raise ContractError("disproven mechanism capability evidence is invalid")
    taskset = _load_taskset(taskset_path.resolve())
    routes = _load_routes(routes_path.resolve(), taskset)
    feedback_by_id = {
        task.task_id: task for task in taskset.tasks if task.cohort == "feedback"
    }
    outside = sorted(set(task_ids) - set(feedback_by_id))
    if outside:
        raise ContractError(
            "targeted native feedback task is outside feedback cohort: "
            + ", ".join(outside)
        )
    feedback_ids = set(feedback_by_id)
    experiment = experiment_root.resolve()
    native = native_root.resolve()
    _reject_partition_evidence(experiment, feedback_ids, "holdout experiment evidence")
    _reject_partition_evidence(native, feedback_ids, "holdout native evidence")
    native_summary = _load_native_summary(native / "PROGRESS.json")
    if (
        native_summary.get("evaluation_scope") != "round1-feedback-only"
        or native_summary.get("taskset_fingerprint") != taskset.fingerprint
        or native_summary.get("holdout_cells_opened") is not False
        or native_summary.get("full_capability_gate_evaluated") is not False
        or type(native_summary.get("completed_cells")) is not int
        or native_summary.get("completed_cells", 0) < len(task_ids) * 2
    ):
        raise ContractError("targeted native feedback summary boundary is invalid")

    failures: list[dict[str, Any]] = []
    current_skills: dict[str, dict[str, Any]] = {}
    reason_counts: dict[str, Counter[str]] = {}
    for task_id in task_ids:
        task = feedback_by_id[task_id]
        mechanism = routes[task_id]
        arms: dict[str, Any] = {}
        for teaching in ("baseline", "taught"):
            condition_id = f"{mechanism}-{teaching}"
            cell_dir = experiment / "cells" / task_id / condition_id
            attempt = load_frozen_attempt(cell_dir / "ATTEMPT.json")
            task_record = attempt.get("task", {})
            condition = attempt.get("condition", {})
            if (
                attempt.get("taskset_fingerprint") != taskset.fingerprint
                or task_record.get("task_id") != task_id
                or task_record.get("cohort") != "feedback"
                or condition.get("condition_id") != condition_id
                or condition.get("mechanism") != mechanism
                or condition.get("teaching") != teaching
            ):
                raise ContractError(
                    "targeted native feedback attempt boundary mismatch"
                )
            raw_path = cell_dir / "raw-output.txt"
            try:
                raw_output = raw_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ContractError(
                    "targeted native feedback raw output is unreadable"
                ) from exc
            raw_sha = hashlib.sha256(raw_output.encode()).hexdigest()
            if attempt.get("attempt", {}).get("raw_output_sha256") != raw_sha:
                raise ContractError(
                    "targeted native feedback raw output sha256 mismatch"
                )
            traces: list[dict[str, Any]] = []
            for trace in attempt.get("generation_trace", []):
                if not isinstance(trace, dict) or not isinstance(
                    trace.get("path"), str
                ):
                    raise ContractError("targeted native generation trace is invalid")
                generation_path = cell_dir / trace["path"]
                generation_bytes = generation_path.read_bytes()
                generation_sha = hashlib.sha256(generation_bytes).hexdigest()
                if generation_sha != trace.get("sha256"):
                    raise ContractError("targeted native generation sha256 mismatch")
                try:
                    generation_output = generation_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ContractError(
                        "targeted native generation output is not UTF-8"
                    ) from exc
                trace_row = {
                    "candidate_id": trace.get("kind"),
                    "raw_output": generation_output,
                    "raw_output_sha256": generation_sha,
                    "stage_result": trace.get("stage_result"),
                }
                prompt_value = trace.get("prompt_path")
                if prompt_value is not None:
                    if not isinstance(prompt_value, str):
                        raise ContractError("targeted native prompt path is invalid")
                    prompt_path = cell_dir / prompt_value
                    prompt_bytes = prompt_path.read_bytes()
                    prompt_sha = hashlib.sha256(prompt_bytes).hexdigest()
                    if prompt_sha != trace.get("prompt_sha256"):
                        raise ContractError("targeted native prompt sha256 mismatch")
                    try:
                        prompt = prompt_bytes.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise ContractError(
                            "targeted native generation prompt is not UTF-8"
                        ) from exc
                    trace_row.update(
                        {
                            "generation_prompt": prompt,
                            "generation_prompt_sha256": prompt_sha,
                        }
                    )
                traces.append(trace_row)
            native_cell = _read_native_cell(native / "cells" / task_id / condition_id)
            if (
                native_cell.get("taskset_fingerprint") != taskset.fingerprint
                or native_cell.get("task_id") != task_id
                or native_cell.get("cohort") != "feedback"
                or native_cell.get("condition_id") != condition_id
                or native_cell.get("experiment_cell_sha256")
                != attempt.get("evidence_sha256")
                or native_cell.get("holdout_cells_opened") is not False
            ):
                raise ContractError("targeted native feedback cell boundary mismatch")
            outcome = attempt.get("attempt", {})
            reason = outcome.get("failure_reason") or (
                "native-resolved"
                if native_cell.get("outcome", {}).get("resolved") is True
                else "native-unresolved"
            )
            reason_counts.setdefault(condition_id, Counter())[str(reason)] += 1
            arms[teaching] = {
                "condition_id": condition_id,
                "experiment_cell_sha256": attempt["evidence_sha256"],
                "native_cell_sha256": native_cell["evidence_sha256"],
                "structural_valid": outcome.get("structural_valid"),
                "failure_reason": outcome.get("failure_reason"),
                "detail": outcome.get("detail"),
                "raw_output": raw_output,
                "raw_output_sha256": raw_sha,
                "generation_candidates": traces,
                "patch_sha256": outcome.get("patch_sha256"),
                "native_outcome": native_cell.get("outcome"),
                "native_report_sha256": (native_cell.get("native_report") or {}).get(
                    "sha256"
                ),
            }
            if teaching == "taught":
                revision = condition.get("revision")
                if not isinstance(revision, dict):
                    raise ContractError("targeted taught revision is invalid")
                prior = current_skills.setdefault(mechanism, revision)
                if prior != revision:
                    raise ContractError("targeted mechanism uses multiple revisions")
        failures.append(
            {
                "task_id": task_id,
                "mechanism": mechanism,
                "issue": task.instruction,
                "allowed_targets": list(task.allowed_targets),
                "baseline": arms["baseline"],
                "taught": arms["taught"],
            }
        )

    prior_feedback_iterations: list[dict[str, Any]] = []
    for prior_path in prior_request_paths:
        prior = _load_request(prior_path.resolve())
        prior_request = prior["request"]
        if (
            prior.get("taskset_fingerprint") != taskset.fingerprint
            or prior_request.get("request_type") != "round1-targeted-native-feedback-v1"
            or prior.get("native_receipts_included") is not True
            or prior.get("holdout_cells_included") is not False
            or not isinstance(prior_request.get("failures"), list)
        ):
            raise ContractError("prior targeted native feedback boundary is invalid")
        prior_feedback_iterations.append(
            {
                "request_sha256": prior["request_sha256"],
                "condition_failure_counts": prior_request.get(
                    "condition_failure_counts", {}
                ),
                "failures": prior_request["failures"],
            }
        )

    request = {
        "schema_version": 1,
        "request_type": "round1-targeted-native-feedback-v1",
        "objective": (
            "Revise the inactive Skill so the frozen 4B selects a native-resolving "
            "typed action without the recorded feedback regressions."
        ),
        "condition_failure_counts": {
            condition: dict(sorted(counts.items()))
            for condition, counts in sorted(reason_counts.items())
        },
        "current_inactive_skills": current_skills,
        "disproven_mechanism_capabilities": list(disproven_mechanism_capabilities),
        "prior_feedback_iterations": prior_feedback_iterations,
        "failures": failures,
        "constraints": [
            "Use only selected feedback native evidence; never request holdout or gold.",
            (
                "Keep frozen localization unchanged. Prefer an action-selection-only "
                "revision, but when the recorded native failures show that the current "
                "typed catalog cannot express a non-regressing repair, propose a general "
                "source-derived catalog extension with a deterministic materializer and "
                "verifier."
            ),
            "Compile a general action-selection rule, not a task-specific patch.",
            (
                "Review every prior feedback iteration and do not recommend an "
                "action already disproved by its native evidence."
            ),
            (
                "Mechanisms listed in disproven_mechanism_capabilities already exist "
                "and must not repeat as the next mechanism revision."
            ),
            "Keep candidate Skills inactive and do not modify tests or the native judge.",
            "Native resolved without regression remains the sole capability credit.",
        ],
    }
    content = {
        "schema_version": 1,
        "request": request,
        "request_sha256": sha256_json(request),
        "taskset_fingerprint": taskset.fingerprint,
        "native_summary_sha256": native_summary["summary_sha256"],
        "feedback_task_count": len(task_ids),
        "feedback_cell_count": len(task_ids) * 2,
        "feedback_iteration_count": len(prior_feedback_iterations) + 1,
        "native_receipts_included": True,
        "holdout_cells_included": False,
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    _freeze(output_path.resolve(), report)
    return report


def create_round1_feedback_authorization(
    *,
    request_path: Path,
    campaign_checkpoint_path: Path,
    output_path: Path,
    expires_at: datetime,
    maximum_output_tokens: int = 256_000,
) -> dict[str, Any]:
    """Bind the user's cumulative 3M grant to one exact feedback request."""

    request = _load_request(request_path.resolve())
    checkpoint = _load_campaign_checkpoint(campaign_checkpoint_path.resolve())
    approval = P1ParentCallAuthorization.create(
        request_sha256=request["request_sha256"],
        model="deepseek-v4-flash",
        maximum_output_tokens=maximum_output_tokens,
        authorization_id="round1-feedback-v4-flash-3m-user-grant",
        approved_by="user:deepseek-v4-flash-3000000-total-tokens",
        expires_at=expires_at,
    )
    content = {
        "schema_version": 1,
        "campaign_total_token_limit": _CAMPAIGN_LIMIT,
        "campaign_tokens_before": checkpoint["campaign_tokens_after"],
        "campaign_checkpoint_evidence_sha256": checkpoint["evidence_sha256"],
        "single_call_authorization": approval.to_dict(),
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    _freeze(output_path.resolve(), report)
    return report


def dispatch_round1_feedback_strategy(
    *,
    request_path: Path,
    authorization_path: Path,
    campaign_checkpoint_path: Path,
    ledger_path: Path,
    output_path: Path,
    client: TeacherClient | None = None,
) -> dict[str, Any]:
    """Run one replay-safe DeepSeek strategy call against feedback evidence."""

    request = _load_request(request_path.resolve())
    authorization = _load_evidence(
        authorization_path.resolve(), "Round 1 feedback authorization"
    )
    checkpoint = _load_campaign_checkpoint(campaign_checkpoint_path.resolve())
    if (
        authorization.get("campaign_total_token_limit") != _CAMPAIGN_LIMIT
        or authorization.get("campaign_tokens_before")
        != checkpoint["campaign_tokens_after"]
        or authorization.get("campaign_checkpoint_evidence_sha256")
        != checkpoint["evidence_sha256"]
    ):
        raise ContractError("Round 1 feedback campaign checkpoint mismatch")
    approval = P1ParentCallAuthorization.from_dict(
        authorization.get("single_call_authorization")
    )
    configured = client or TeacherClient.from_env(TeacherProvider.DEEPSEEK)
    approval.validate(client=configured)
    if approval.request_sha256 != request["request_sha256"]:
        raise ContractError("Round 1 feedback request authorization mismatch")
    output = output_path.resolve()
    if output.exists():
        return _load_evidence(output, "Round 1 feedback strategy response")

    reservation = (
        len(canonical_json(request["request"]).encode("utf-8"))
        + 20_000
        + approval.maximum_output_tokens
    )
    tokens_before = int(checkpoint["campaign_tokens_after"])
    if tokens_before + reservation > _CAMPAIGN_LIMIT:
        raise ContractError("DeepSeek campaign token reservation exceeds 3000000")

    ledger = ParentCallLedger(ledger_path, approval.loop_authorization)
    call_id = "round1-feedback-strategy-001"
    existing = ledger.get(call_id)
    if existing is not None and existing.request_sha256 != request["request_sha256"]:
        raise ContractError("Round 1 feedback call ID belongs to another request")
    recovered_raw_sha256: str | None = None
    if existing is not None and existing.status == "aborted":
        raw = _load_evidence(
            output.with_name(f"{output.stem}.raw.json"),
            "Round 1 feedback raw response",
        )
        if (
            raw.get("event_type") != "parent-strategy-raw-response"
            or raw.get("call_id") != call_id
            or raw.get("request_sha256") != request["request_sha256"]
            or raw.get("network_calls_performed") is not True
            or not isinstance(raw.get("response_text"), str)
            or not isinstance(raw.get("usage"), dict)
            or not isinstance(raw.get("provider"), str)
            or not isinstance(raw.get("model"), str)
        ):
            raise ContractError("Round 1 feedback raw response boundary is invalid")
        strategy = _parse_strategy(raw["response_text"])
        response_record = {
            "schema_version": 1,
            "strategy": strategy,
            "usage": {
                **raw["usage"],
                "provider": raw["provider"],
                "model": raw["model"],
                "maximum_output_tokens": approval.maximum_output_tokens,
            },
        }
        recovered_raw_sha256 = str(raw["evidence_sha256"])
    elif existing is not None and existing.status != "completed":
        raise ContractError("Round 1 feedback call is already terminal or in-flight")
    elif existing is None:
        ledger.reserve(call_id=call_id, request_sha256=request["request_sha256"])
        try:
            response = configured.complete(
                TeacherSample(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are revising an inactive, leakage-free Skill and "
                                "patch-realization mechanism for a frozen local 4B software "
                                "engineering Student. Analyze all supplied feedback A/B and "
                                "native receipts. Separate localization, selector, syntax, "
                                "semantic, verifier, and model-capacity causes. Return exactly "
                                "one JSON object with failure_diagnosis and mechanism_revision "
                                "as non-empty strings, plus non-empty arrays named "
                                "operator_skill_requirements, span_skill_requirements, "
                                "verification_changes, and next_experiment. Requirements must "
                                "be compact enough to compile into inactive Skills under 2500 "
                                "characters. Never request holdout/gold evidence, modify tests, "
                                "change the native judge, activate a Skill, or claim capability "
                                "from structural validity alone. Prefer a new falsifiable "
                                "mechanism over more retries of an unchanged prompt."
                            ),
                        },
                        {"role": "user", "content": canonical_json(request["request"])},
                    ],
                    metadata={"request_sha256": request["request_sha256"]},
                    max_output_tokens=approval.maximum_output_tokens,
                    response_format={"type": "json_object"},
                    thinking=True,
                    reasoning_effort="high",
                )
            )
            raw_content = {
                "schema_version": 1,
                "event_type": "parent-strategy-raw-response",
                "call_id": call_id,
                "request_sha256": request["request_sha256"],
                "response_text": response.text,
                "usage": response.usage,
                "provider": response.provider.value,
                "model": response.model,
                "network_calls_performed": True,
            }
            _freeze(
                output.with_name(f"{output.stem}.raw.json"),
                {**raw_content, "evidence_sha256": sha256_json(raw_content)},
            )
            strategy = _parse_strategy(response.text)
            response_record = {
                "schema_version": 1,
                "strategy": strategy,
                "usage": {
                    **response.usage,
                    "provider": response.provider.value,
                    "model": response.model,
                    "maximum_output_tokens": approval.maximum_output_tokens,
                },
            }
            ledger.complete(
                call_id=call_id,
                response_sha256=sha256_json(response_record),
                response=response_record,
                usage=response_record["usage"],
            )
        except Exception as exc:
            ledger.abort(call_id=call_id, reason=f"{type(exc).__name__}: {exc}")
            raise
    else:
        response_record = existing.response
        if not isinstance(response_record, dict):
            raise ContractError("Round 1 feedback ledger response is invalid")
        strategy = response_record.get("strategy")
        if not isinstance(strategy, dict):
            raise ContractError("Round 1 feedback ledger strategy is invalid")

    usage = response_record.get("usage")
    if not isinstance(usage, dict):
        raise ContractError("Round 1 feedback strategy usage is invalid")
    tokens_charged = _usage_total(usage)
    tokens_after = tokens_before + tokens_charged
    if tokens_after > _CAMPAIGN_LIMIT:
        raise ContractError("DeepSeek campaign token budget exceeded")
    content = {
        "schema_version": 1,
        "event_type": "parent-strategy-response",
        "call_id": call_id,
        "request_sha256": request["request_sha256"],
        "authorization_fingerprint": approval.fingerprint,
        "strategy": strategy,
        "usage": usage,
        "tokens_charged": tokens_charged,
        "campaign_tokens_before": tokens_before,
        "campaign_tokens_after": tokens_after,
        "campaign_total_token_limit": _CAMPAIGN_LIMIT,
        "candidate_status": "advisory_inactive",
        "auto_apply": False,
        "holdout_cells_included": False,
        "network_calls_performed": True,
    }
    if recovered_raw_sha256 is not None:
        content["recovered_from_raw_evidence_sha256"] = recovered_raw_sha256
        content["response_normalization"] = "single-next-experiment-to-list-v1"
    report = {**content, "evidence_sha256": sha256_json(content)}
    _freeze(output, report)
    return report


def _reject_partition_evidence(root: Path, feedback_ids: set[str], label: str) -> None:
    if _partition_evidence_present(root, feedback_ids):
        raise ContractError(f"{label} is prohibited")


def _partition_evidence_present(root: Path, feedback_ids: set[str]) -> bool:
    cells = root / "cells"
    if not cells.exists():
        return False
    return any(
        path.is_dir() and path.name not in feedback_ids for path in cells.iterdir()
    )


def _load_native_summary(path: Path) -> dict[str, Any]:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("Round 1 feedback native summary is unreadable") from exc
    if not isinstance(summary, dict):
        raise ContractError("Round 1 feedback native summary must be an object")
    content = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if summary.get("summary_sha256") != sha256_json(content):
        raise ContractError("Round 1 feedback native summary sha256 mismatch")
    return summary


def _load_request(path: Path) -> dict[str, Any]:
    wrapper = _load_evidence(path, "Round 1 feedback request")
    request = wrapper.get("request")
    request_type = request.get("request_type") if isinstance(request, dict) else None
    counts_valid = (
        type(wrapper.get("feedback_task_count")) is int
        and wrapper["feedback_task_count"] >= 3
        and wrapper.get("feedback_cell_count") == wrapper["feedback_task_count"] * 2
        if request_type == "round1-feedback-skill-evolution-v1"
        else request_type
        in {
            "round1-realization-feedback-v1",
            "round1-targeted-native-feedback-v1",
        }
        and type(wrapper.get("feedback_task_count")) is int
        and 1 <= wrapper["feedback_task_count"] <= 30
        and wrapper.get("feedback_cell_count") == wrapper["feedback_task_count"] * 2
        and wrapper.get("native_receipts_included")
        == (request_type == "round1-targeted-native-feedback-v1")
    )
    if (
        not isinstance(request, dict)
        or wrapper.get("request_sha256") != sha256_json(request)
        or not counts_valid
        or wrapper.get("holdout_cells_included") is not False
        or wrapper.get("network_calls_performed") is not False
        or (
            wrapper.get("source_holdout_evidence_present") is True
            and wrapper.get("current_holdout_reuse_prohibited") is not True
        )
    ):
        raise ContractError("Round 1 feedback request boundary is invalid")
    return wrapper


def _load_campaign_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = _load_evidence(path, "DeepSeek campaign checkpoint")
    if (
        checkpoint.get("campaign_total_token_limit") != _CAMPAIGN_LIMIT
        or type(checkpoint.get("campaign_tokens_after")) is not int
        or not 0 <= checkpoint["campaign_tokens_after"] <= _CAMPAIGN_LIMIT
        or checkpoint.get("candidate_status") != "advisory_inactive"
        or checkpoint.get("auto_apply") is not False
        or checkpoint.get("network_calls_performed") is not True
    ):
        raise ContractError("DeepSeek campaign checkpoint boundary is invalid")
    return checkpoint


def _load_evidence(path: Path, label: str) -> dict[str, Any]:
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is unreadable") from exc
    if not isinstance(wrapper, dict):
        raise ContractError(f"{label} must be an object")
    content = {key: value for key, value in wrapper.items() if key != "evidence_sha256"}
    if wrapper.get("evidence_sha256") != sha256_json(content):
        raise ContractError(f"{label} evidence sha256 mismatch")
    return wrapper


def _parse_strategy(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    try:
        strategy = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ContractError("DeepSeek Round 1 feedback strategy is not JSON") from exc
    if not isinstance(strategy, dict) or set(strategy) != _STRATEGY_FIELDS:
        raise ContractError("DeepSeek Round 1 feedback strategy fields are invalid")
    for field in ("failure_diagnosis", "mechanism_revision"):
        if not isinstance(strategy[field], str) or not strategy[field].strip():
            raise ContractError("DeepSeek Round 1 feedback strategy text is invalid")
    for field in (
        "operator_skill_requirements",
        "span_skill_requirements",
        "verification_changes",
        "next_experiment",
    ):
        values = strategy[field]
        if field == "next_experiment" and isinstance(values, str):
            values = [values]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(row, str) or not row.strip() for row in values)
        ):
            raise ContractError("DeepSeek Round 1 feedback strategy list is invalid")
        strategy[field] = [row.strip() for row in values]
    return strategy


def _usage_total(usage: dict[str, Any]) -> int:
    total = usage.get("total_tokens")
    if type(total) is int and total >= 0:
        return total
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    if (
        type(prompt) is not int
        or prompt < 0
        or type(completion) is not int
        or completion < 0
    ):
        raise ContractError("DeepSeek Round 1 feedback usage is invalid")
    return prompt + completion


def _freeze(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError("Round 1 feedback request is unreadable") from exc
        if existing != payload:
            raise ContractError("frozen Round 1 feedback request changed")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
