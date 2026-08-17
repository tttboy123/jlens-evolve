"""Build and run the leakage-gated routed Round 1 Student experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .contracts import ContractError, canonical_json, sha256_json
from .eval_manifest import EvaluationTaskSet
from .experiment import PairedExperimentRunner
from .model_transport import FileCachedModelTransport, OpenAICompatibleTransport
from .operator_student import (
    MlxOperatorPlanGenerator,
    OperatorPlanAdapter,
    build_operator_conditions,
)
from .p1_operator import load_frozen_operator_skill_revision
from .span_student import MlxSpanPlanGenerator, SpanPlanAdapter, build_span_conditions


class GenerationCallBudget:
    """Fail closed before starting a model generation beyond one process budget."""

    def __init__(self, *, delegate: Callable[..., str], maximum_calls: int) -> None:
        if type(maximum_calls) is not int or maximum_calls < 1:
            raise ContractError("generation call budget must be positive")
        self.delegate = delegate
        self.maximum_calls = maximum_calls
        self.started_calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        if self.started_calls >= self.maximum_calls:
            raise ContractError("generation call budget exhausted")
        self.started_calls += 1
        return self.delegate(*args, **kwargs)


class BudgetedModelTransport:
    """Apply the same process-wide admission counter to any model transport."""

    def __init__(self, *, delegate: Any, maximum_calls: int) -> None:
        self.delegate = delegate
        self.maximum_calls = maximum_calls
        self.started_calls = 0

    def _admit(self) -> None:
        if self.started_calls >= self.maximum_calls:
            raise ContractError("generation call budget exhausted")
        self.started_calls += 1

    def generate(self, request: Any) -> Any:
        self._admit()
        return self.delegate.generate(request)

    def generate_prompt(self, request: Any) -> Any:
        self._admit()
        return self.delegate.generate_prompt(request)

    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "generation-call-budget-model-transport",
            "maximum_calls": self.maximum_calls,
            "delegate": self.delegate.identity(),
        }


def compile_round1_span_skill(
    *, operator_skill_path: Path, output_path: Path
) -> dict[str, Any]:
    """Compile the proven inactive r8 teaching into the span action contract."""

    revision = load_frozen_operator_skill_revision(operator_skill_path)
    marker = "## Pattern cards\n"
    if marker not in revision.skill_text:
        raise ContractError("Round 1 source Skill has no Pattern cards")
    pattern_cards = revision.skill_text.split(marker, 1)[1]
    pattern_cards = pattern_cards.split("\n\nUse exact AST-equivalent", 1)[0].strip()
    skill_text = (
        "---\n"
        "name: exact-span-bundle-realization\n"
        "project_local_only: true\n"
        "auto_install: false\n"
        "active: false\n"
        "---\n\n"
        "# Exact-span bundle realization\n\n"
        "1. Emit a concrete Defect, Trigger, and DesiredBoundary.\n"
        "2. Choose only supplied candidate files; use two files only when one invariant requires both.\n"
        "3. Copy the smallest unique before span exactly and provide its complete changed after span.\n"
        "4. Reject no-op, copied, unrelated, or opposite-direction operations.\n\n"
        f"## Pattern cards\n{pattern_cards}\n\n"
        "Preserve unrelated behavior. This compiled candidate is experimental, inactive, and unpromoted."
    )
    if len(skill_text) > 2_500:
        raise ContractError("Round 1 compiled span Skill exceeds 2500 characters")
    content = {
        "schema_version": 1,
        "source_operator_skill_file_sha256": hashlib.sha256(
            operator_skill_path.read_bytes()
        ).hexdigest(),
        "source_operator_skill_fingerprint": revision.fingerprint,
        "compiler": "operator-to-span-bundle-action-contract-v1",
        "skill_text": skill_text,
        "candidate_status": "inactive",
        "auto_activate": False,
        "new_domain_knowledge_added": False,
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    output = output_path.resolve()
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError("Round 1 span Skill is unreadable") from exc
        if existing != report:
            raise ContractError("frozen Round 1 span Skill does not match replay")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report


def run_round1_feedback(
    *,
    taskset_path: Path,
    routes_path: Path,
    target_audit_path: Path,
    operator_skill_path: Path,
    span_skill_path: Path,
    model_path: Path,
    evidence_root: Path,
    workspace_root: Path,
    max_cells: int | None = None,
    realization_candidates: int = 1,
    max_plan_repairs: int | None = None,
    maximum_generation_calls: int | None = None,
    shared_diagnosis_localization: bool = False,
    task_ids: tuple[str, ...] | None = None,
    transport_base_url: str | None = None,
    transport_model: str | None = None,
    transport_api_key_env: str | None = None,
    generation_cache_root: Path | None = None,
    futility_min_cells_per_mechanism: int | None = None,
    futility_min_structural_rate: float = 0.0,
    shared_context_source_root: Path | None = None,
) -> dict[str, Any]:
    """Run/resume feedback cells only; holdout cannot be selected here."""

    return _run_round1_cohort(
        cohort="feedback",
        taskset_path=taskset_path,
        routes_path=routes_path,
        target_audit_path=target_audit_path,
        operator_skill_path=operator_skill_path,
        span_skill_path=span_skill_path,
        model_path=model_path,
        evidence_root=evidence_root,
        workspace_root=workspace_root,
        max_cells=max_cells,
        feedback_gain_path=None,
        realization_candidates=realization_candidates,
        max_plan_repairs=max_plan_repairs,
        maximum_generation_calls=maximum_generation_calls,
        shared_diagnosis_localization=shared_diagnosis_localization,
        requested_task_ids=task_ids,
        transport_base_url=transport_base_url,
        transport_model=transport_model,
        transport_api_key_env=transport_api_key_env,
        generation_cache_root=generation_cache_root,
        futility_min_cells_per_mechanism=futility_min_cells_per_mechanism,
        futility_min_structural_rate=futility_min_structural_rate,
        shared_context_source_root=shared_context_source_root,
    )


def run_round1_holdout(
    *,
    taskset_path: Path,
    routes_path: Path,
    target_audit_path: Path,
    operator_skill_path: Path,
    span_skill_path: Path,
    feedback_gain_path: Path,
    model_path: Path,
    evidence_root: Path,
    workspace_root: Path,
    max_cells: int | None = None,
    realization_candidates: int = 1,
    max_plan_repairs: int | None = None,
    maximum_generation_calls: int | None = None,
    shared_diagnosis_localization: bool = False,
    task_ids: tuple[str, ...] | None = None,
    transport_base_url: str | None = None,
    transport_model: str | None = None,
    transport_api_key_env: str | None = None,
    generation_cache_root: Path | None = None,
    futility_min_cells_per_mechanism: int | None = None,
    futility_min_structural_rate: float = 0.0,
    shared_context_source_root: Path | None = None,
) -> dict[str, Any]:
    """Run/resume the 30 holdout tasks only after a frozen feedback gain."""

    return _run_round1_cohort(
        cohort="holdout",
        taskset_path=taskset_path,
        routes_path=routes_path,
        target_audit_path=target_audit_path,
        operator_skill_path=operator_skill_path,
        span_skill_path=span_skill_path,
        model_path=model_path,
        evidence_root=evidence_root,
        workspace_root=workspace_root,
        max_cells=max_cells,
        feedback_gain_path=feedback_gain_path,
        realization_candidates=realization_candidates,
        max_plan_repairs=max_plan_repairs,
        maximum_generation_calls=maximum_generation_calls,
        shared_diagnosis_localization=shared_diagnosis_localization,
        requested_task_ids=task_ids,
        transport_base_url=transport_base_url,
        transport_model=transport_model,
        transport_api_key_env=transport_api_key_env,
        generation_cache_root=generation_cache_root,
        futility_min_cells_per_mechanism=futility_min_cells_per_mechanism,
        futility_min_structural_rate=futility_min_structural_rate,
        shared_context_source_root=shared_context_source_root,
    )


def _run_round1_cohort(
    *,
    cohort: str,
    taskset_path: Path,
    routes_path: Path,
    target_audit_path: Path,
    operator_skill_path: Path,
    span_skill_path: Path,
    model_path: Path,
    evidence_root: Path,
    workspace_root: Path,
    max_cells: int | None,
    feedback_gain_path: Path | None,
    realization_candidates: int,
    max_plan_repairs: int | None,
    maximum_generation_calls: int | None,
    shared_diagnosis_localization: bool,
    requested_task_ids: tuple[str, ...] | None,
    transport_base_url: str | None,
    transport_model: str | None,
    transport_api_key_env: str | None,
    generation_cache_root: Path | None,
    futility_min_cells_per_mechanism: int | None,
    futility_min_structural_rate: float,
    shared_context_source_root: Path | None,
) -> dict[str, Any]:
    """Shared runner while preserving separate feedback/holdout entry points."""

    if type(realization_candidates) is not int or not 1 <= realization_candidates <= 8:
        raise ContractError("Round 1 realization_candidates must be between 1 and 8")
    if max_plan_repairs is not None and (
        type(max_plan_repairs) is not int or not 0 <= max_plan_repairs <= 3
    ):
        raise ContractError("Round 1 max_plan_repairs must be between 0 and 3")
    if maximum_generation_calls is not None and (
        type(maximum_generation_calls) is not int or maximum_generation_calls < 1
    ):
        raise ContractError("Round 1 maximum_generation_calls must be positive")
    if type(shared_diagnosis_localization) is not bool:
        raise ContractError("Round 1 shared diagnosis/localization flag is invalid")

    taskset = _load_taskset(taskset_path)
    gain_gate = None
    if cohort == "holdout":
        if feedback_gain_path is None:
            raise ContractError("Round 1 holdout requires a feedback gain gate")
        gain_gate = load_round1_feedback_gain_gate(feedback_gain_path, taskset)
    elif cohort != "feedback" or feedback_gain_path is not None:
        raise ContractError("Round 1 cohort boundary is invalid")
    routes = _load_routes(routes_path, taskset)
    qualification_fingerprint = _load_target_audit(target_audit_path, taskset)
    operator_revision = load_frozen_operator_skill_revision(operator_skill_path)
    span_wrapper = _load_frozen(span_skill_path, "Round 1 span Skill")
    if (
        span_wrapper.get("candidate_status") != "inactive"
        or span_wrapper.get("auto_activate") is not False
        or span_wrapper.get("new_domain_knowledge_added") is not False
    ):
        raise ContractError("Round 1 span Skill boundary is invalid")

    transport = None
    generator_runtime: dict[str, Any]
    if transport_base_url is not None or transport_model is not None:
        if not transport_base_url or not transport_model:
            raise ContractError("remote transport requires base URL and model")
        transport = OpenAICompatibleTransport(
            base_url=transport_base_url,
            model=transport_model,
            api_key_env=transport_api_key_env,
        )
        if maximum_generation_calls is not None:
            transport = BudgetedModelTransport(
                delegate=transport, maximum_calls=maximum_generation_calls
            )
        if generation_cache_root is not None:
            transport = FileCachedModelTransport(
                delegate=transport,
                cache_root=generation_cache_root,
            )
        generator_runtime = {"model_transport": transport}
    else:
        from mlx_lm import generate, load

        cached: list[tuple[Any, Any]] = []

        def shared_loader(path: str) -> tuple[Any, Any]:
            if not cached:
                cached.append(load(path))
            return cached[0]

        local_generate = (
            GenerationCallBudget(
                delegate=generate, maximum_calls=maximum_generation_calls
            )
            if maximum_generation_calls is not None
            else generate
        )
        generator_runtime = {
            "loader": shared_loader,
            "text_generator": local_generate,
        }

    effective_max_plan_repairs = (
        max_plan_repairs
        if max_plan_repairs is not None
        else (0 if realization_candidates > 1 else 1)
    )
    operator_generator = MlxOperatorPlanGenerator(
        model_path=model_path,
        **generator_runtime,
        max_plan_repairs=effective_max_plan_repairs,
    )
    span_generator = MlxSpanPlanGenerator(
        model_path=model_path,
        **generator_runtime,
        max_plan_repairs=effective_max_plan_repairs,
    )
    base_adapters = {
        "operator": OperatorPlanAdapter(generator=operator_generator),
        "span": SpanPlanAdapter(generator=span_generator),
    }
    if shared_diagnosis_localization:
        from .round1_realization import build_round1_shared_realization_adapter

        adapters = {
            mechanism: build_round1_shared_realization_adapter(
                mechanism=mechanism,
                base_adapter=adapter,
                maximum_candidates=realization_candidates,
            )
            for mechanism, adapter in base_adapters.items()
        }
    elif realization_candidates == 1:
        adapters = base_adapters
    else:
        from .round1_realization import build_round1_realization_adapter

        adapters = {
            mechanism: build_round1_realization_adapter(
                mechanism=mechanism,
                base_adapter=adapter,
                maximum_candidates=realization_candidates,
            )
            for mechanism, adapter in base_adapters.items()
        }
    conditions = [
        *build_operator_conditions(
            taught_skill=operator_revision.skill_text,
            parent_revision_id=operator_revision.revision_id,
            source_round=operator_revision.source_round,
            generation_config=adapters["operator"].experiment_config(),
        ),
        *build_span_conditions(
            taught_skill=span_wrapper["skill_text"],
            parent_revision_id=operator_revision.revision_id,
            source_round=operator_revision.source_round,
            generation_config=adapters["span"].experiment_config(),
        ),
    ]
    runner = PairedExperimentRunner(
        taskset=taskset,
        adapters=adapters,
        conditions=conditions,
        evidence_root=evidence_root,
        workspace_root=workspace_root,
        qualification_fingerprint=qualification_fingerprint,
        mechanism_routes=routes,
        shared_context_source_root=shared_context_source_root,
    )
    cohort_ids = {task.task_id for task in taskset.tasks if task.cohort == cohort}
    if len(cohort_ids) < 3:
        raise ContractError(f"Round 1 {cohort} cohort must contain at least 3 tasks")
    selected_ids = _select_round1_task_ids(
        taskset, cohort=cohort, requested=requested_task_ids
    )
    projection = runner.run(
        task_ids=selected_ids,
        max_cells=max_cells,
        futility_min_cells_per_mechanism=futility_min_cells_per_mechanism,
        futility_min_structural_rate=futility_min_structural_rate,
    )
    transport_metrics = (
        transport.aggregate_metrics()
        if isinstance(transport, FileCachedModelTransport)
        else {
            "cache_entries": 0,
            "remote_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "current_process_cache_hits": 0,
            "current_process_cache_misses": 0,
            "current_process_cache_writes": 0,
        }
    )
    if transport is not None:
        _freeze_model_transport_metrics(
            evidence_root=evidence_root,
            experiment_projection_sha256=projection["summary_sha256"],
            transport_identity=transport.identity(),
            metrics=transport_metrics,
        )
    if cohort == "feedback":
        return projection

    completed = sum(
        (
            evidence_root.resolve()
            / "cells"
            / task.task_id
            / f"{routes[task.task_id]}-{teaching}"
        ).is_dir()
        for task in taskset.tasks
        if task.cohort == "holdout"
        for teaching in ("baseline", "taught")
    )
    planned_cells = len(cohort_ids) * 2
    content = {
        "schema_version": 1,
        "evaluation_scope": "round1-holdout-only",
        "status": "complete" if completed == planned_cells else "partial",
        "taskset_fingerprint": taskset.fingerprint,
        "planned_cells": planned_cells,
        "completed_cells": completed,
        "feedback_gain_summary_sha256": gain_gate["summary_sha256"],
        "feedback_gain_count": gain_gate["feedback_gain_count"],
        "experiment_projection_sha256": projection["summary_sha256"],
        "holdout_cells_opened": True,
        "model_transport_metrics": transport_metrics,
        "network_calls_performed": False,
    }
    report = {**content, "summary_sha256": sha256_json(content)}
    _write_holdout_summary(evidence_root.resolve(), report)
    return report


def _freeze_model_transport_metrics(
    *,
    evidence_root: Path,
    experiment_projection_sha256: str,
    transport_identity: dict[str, Any],
    metrics: dict[str, int],
) -> dict[str, Any]:
    """Append an identity-bound receipt without changing experiment semantics."""

    if not isinstance(transport_identity, dict) or not transport_identity:
        raise ContractError("model transport identity is invalid")
    if (
        not isinstance(experiment_projection_sha256, str)
        or len(experiment_projection_sha256) != 64
    ):
        raise ContractError("experiment projection sha256 is invalid")
    expected_metrics = {
        "cache_entries",
        "remote_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "current_process_cache_hits",
        "current_process_cache_misses",
        "current_process_cache_writes",
    }
    if set(metrics) != expected_metrics or any(
        type(value) is not int or value < 0 for value in metrics.values()
    ):
        raise ContractError("model transport metrics are invalid")
    content = {
        "schema_version": 1,
        "contract": "model-transport-execution-metrics-v1",
        "experiment_projection_sha256": experiment_projection_sha256,
        "transport_identity": transport_identity,
        "transport_identity_sha256": sha256_json(transport_identity),
        "model_transport_metrics": metrics,
        # ``remote_calls`` is reconstructed historical cache provenance. Only
        # current misses can cross the endpoint boundary in this execution.
        "network_calls_performed": metrics["current_process_cache_misses"] > 0,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    target = (
        evidence_root.resolve()
        / "model-transport-metrics"
        / f"{report['evidence_sha256']}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError(
                "model transport metrics receipt is unreadable"
            ) from exc
        if existing != report:
            raise ContractError("model transport metrics receipt changed")
    else:
        target.write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report


def _select_round1_task_ids(
    taskset: EvaluationTaskSet,
    *,
    cohort: str,
    requested: tuple[str, ...] | None,
) -> set[str]:
    cohort_ids = {task.task_id for task in taskset.tasks if task.cohort == cohort}
    if requested is None:
        return cohort_ids
    selected = set(requested)
    if not selected or len(selected) != len(requested):
        raise ContractError(f"explicit Round 1 {cohort} tasks must be unique")
    outside = sorted(selected - cohort_ids)
    if outside:
        raise ContractError(
            f"explicit task is outside {cohort} cohort: " + ", ".join(outside)
        )
    return selected


def load_round1_feedback_gain_gate(
    path: Path, taskset: EvaluationTaskSet
) -> dict[str, Any]:
    """Validate the sole artifact allowed to unlock Round 1 holdout."""

    try:
        summary = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("Round 1 feedback gain summary is unreadable") from exc
    if not isinstance(summary, dict):
        raise ContractError("Round 1 feedback gain summary must be an object")
    content = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if summary.get("summary_sha256") != sha256_json(content):
        raise ContractError("Round 1 feedback gain summary sha256 mismatch")
    planned_feedback_cells = (
        sum(task.cohort == "feedback" for task in taskset.tasks) * 2
    )
    if (
        summary.get("evaluation_scope") != "round1-feedback-only"
        or summary.get("status") != "complete"
        or summary.get("taskset_fingerprint") != taskset.fingerprint
        or summary.get("planned_cells") != planned_feedback_cells
        or summary.get("generated_feedback_cells") != planned_feedback_cells
        or summary.get("completed_cells") != planned_feedback_cells
        or summary.get("full_capability_gate_evaluated") is not False
        or summary.get("holdout_cells_opened") is not False
        or summary.get("network_calls_performed") is not False
        or summary.get("native_evaluator_failure_count", 0) != 0
    ):
        raise ContractError("Round 1 feedback gain summary boundary is invalid")
    if (
        type(summary.get("feedback_gain_count")) is not int
        or summary["feedback_gain_count"] < 1
        or summary.get("feedback_gain_gate_passed") is not True
    ):
        raise ContractError("feedback gain has not unlocked holdout")
    return summary


def _write_holdout_summary(root: Path, report: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    target = root / (
        "HOLDOUT-SUMMARY.json"
        if report["status"] == "complete"
        else "HOLDOUT-PROGRESS.json"
    )
    if target.exists() and report["status"] == "complete":
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError("Round 1 holdout summary is unreadable") from exc
        if existing != report:
            raise ContractError("frozen Round 1 holdout summary changed")
        return
    target.write_text(canonical_json(report) + "\n", encoding="utf-8")


def _load_taskset(path: Path) -> EvaluationTaskSet:
    try:
        taskset = EvaluationTaskSet.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ContractError) as exc:
        raise ContractError("Round 1 executable TaskSet is unreadable") from exc
    taskset.validate()
    return taskset


def _load_routes(path: Path, taskset: EvaluationTaskSet) -> dict[str, str]:
    wrapper = _load_frozen(path, "Round 1 mechanism routes")
    routes = wrapper.get("routes")
    if (
        wrapper.get("taskset_fingerprint") != taskset.fingerprint
        or not isinstance(routes, dict)
        or set(routes) != {task.task_id for task in taskset.tasks}
    ):
        raise ContractError("Round 1 mechanism routes do not match TaskSet")
    return {str(key): str(value) for key, value in routes.items()}


def _load_target_audit(path: Path, taskset: EvaluationTaskSet) -> str:
    wrapper = _load_frozen(path, "Round 1 target audit")
    audit = wrapper.get("audit")
    if (
        not isinstance(audit, dict)
        or audit.get("taskset_fingerprint") != taskset.fingerprint
        or audit.get("ready") is not True
        or audit.get("ready_tasks") != len(taskset.tasks)
        or wrapper.get("student_visible") is not False
    ):
        raise ContractError("Round 1 target audit has not unlocked execution")
    return str(wrapper["evidence_sha256"])


def _load_frozen(path: Path, label: str) -> dict[str, Any]:
    try:
        wrapper = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is unreadable") from exc
    content = {key: value for key, value in wrapper.items() if key != "evidence_sha256"}
    if wrapper.get("evidence_sha256") != sha256_json(content):
        raise ContractError(f"{label} evidence was tampered")
    return wrapper
