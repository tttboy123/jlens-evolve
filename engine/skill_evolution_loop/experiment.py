"""Resumable paired Student experiments with append-only per-cell evidence."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ContractError, LoopRevision, canonical_json, sha256_json
from .eval_manifest import (
    EvaluationTask,
    EvaluationTaskSet,
    materialize_evaluation_task,
)
from .student_adapter import StudentAdapter, StudentAttempt, StudentTask


class ExperimentRootLease:
    """Cross-process, fail-closed single-writer lease for an experiment root."""

    _active_roots: set[Path] = set()
    _guard = threading.Lock()

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._handle: Any | None = None

    def __enter__(self) -> ExperimentRootLease:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._guard:
            if self.root in self._active_roots:
                raise ContractError("experiment root already has an active writer")
            handle = (self.root / ".writer.lock").open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise ContractError(
                    "experiment root already has an active writer"
                ) from exc
            self._active_roots.add(self.root)
            self._handle = handle
        handle.seek(0)
        handle.truncate()
        handle.write(
            canonical_json(
                {
                    "schema_version": 1,
                    "pid": os.getpid(),
                    "acquired_unix_ns": time.time_ns(),
                }
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        return self

    def __exit__(self, *_args: object) -> None:
        handle = self._handle
        if handle is None:
            return
        with self._guard:
            self._active_roots.discard(self.root)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            self._handle = None


@dataclass(frozen=True)
class ExperimentCondition:
    """One mechanism × teaching cell applied to every task."""

    schema_version: int
    condition_id: str
    mechanism: str
    teaching: str
    revision: LoopRevision
    generation_config: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        condition_id: str,
        mechanism: str,
        teaching: str,
        revision: LoopRevision,
        generation_config: dict[str, Any] | None = None,
    ) -> ExperimentCondition:
        condition = cls(
            schema_version=1,
            condition_id=condition_id,
            mechanism=mechanism,
            teaching=teaching,
            revision=revision,
            generation_config=dict(generation_config or {}),
        )
        condition.validate()
        return condition

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported experiment condition schema")
        if not self.condition_id or any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
            for char in self.condition_id
        ):
            raise ContractError("invalid experiment condition id")
        if not self.mechanism.strip():
            raise ContractError("experiment mechanism must be non-empty")
        if self.teaching not in {"baseline", "taught"}:
            raise ContractError("experiment teaching must be baseline or taught")
        self.revision.validate()
        if not isinstance(self.generation_config, dict):
            raise ContractError("experiment generation config must be an object")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "condition_id": self.condition_id,
            "mechanism": self.mechanism,
            "teaching": self.teaching,
            "revision": self.revision.to_dict(),
            "generation_config": self.generation_config,
            "fingerprint": self.fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(
            {
                "schema_version": self.schema_version,
                "condition_id": self.condition_id,
                "mechanism": self.mechanism,
                "teaching": self.teaching,
                "revision": self.revision.to_dict(),
                "generation_config": self.generation_config,
            }
        )


class PairedExperimentRunner:
    """Run and resume all task × condition cells without overwriting evidence."""

    def __init__(
        self,
        *,
        taskset: EvaluationTaskSet,
        adapters: dict[str, StudentAdapter],
        conditions: list[ExperimentCondition],
        evidence_root: Path,
        workspace_root: Path,
        qualification_fingerprint: str | None = None,
        mechanism_routes: dict[str, str] | None = None,
        shared_context_source_root: Path | None = None,
    ) -> None:
        taskset.validate()
        if not conditions:
            raise ContractError("paired experiment requires conditions")
        if len({row.condition_id for row in conditions}) != len(conditions):
            raise ContractError("experiment condition ids must be unique")
        for condition in conditions:
            condition.validate()
            if condition.mechanism not in adapters:
                raise ContractError(
                    f"missing adapter for mechanism: {condition.mechanism}"
                )
            configured = getattr(
                adapters[condition.mechanism], "experiment_config", None
            )
            if condition.generation_config:
                if configured is None or configured() != condition.generation_config:
                    raise ContractError(
                        f"generation config mismatch: {condition.condition_id}"
                    )
        self.taskset = taskset
        self.adapters = dict(adapters)
        self.conditions = tuple(conditions)
        if mechanism_routes is not None:
            known_tasks = {task.task_id for task in taskset.tasks}
            if set(mechanism_routes) != known_tasks:
                raise ContractError(
                    "mechanism routes require exactly one route per task"
                )
            condition_mechanisms = {condition.mechanism for condition in conditions}
            for task_id, mechanism in mechanism_routes.items():
                if mechanism not in adapters or mechanism not in condition_mechanisms:
                    raise ContractError(
                        f"invalid mechanism route: {task_id}/{mechanism}"
                    )
                teachings = {
                    condition.teaching
                    for condition in conditions
                    if condition.mechanism == mechanism
                }
                if teachings != {"baseline", "taught"}:
                    raise ContractError(f"routed mechanism is not paired: {mechanism}")
        self.mechanism_routes = (
            dict(sorted(mechanism_routes.items()))
            if mechanism_routes is not None
            else None
        )
        self.evidence_root = evidence_root.resolve()
        self.workspace_root = workspace_root.resolve()
        self.shared_context_source_root = (
            shared_context_source_root.resolve()
            if shared_context_source_root is not None
            else None
        )
        if (
            qualification_fingerprint is not None
            and re.fullmatch(r"[0-9a-f]{64}", qualification_fingerprint) is None
        ):
            raise ContractError("invalid experiment qualification fingerprint")
        self.qualification_fingerprint = qualification_fingerprint
        self._stopped_mechanisms: dict[str, dict[str, Any]] = {}

    def run(
        self,
        *,
        max_cells: int | None = None,
        task_ids: set[str] | None = None,
        condition_ids: set[str] | None = None,
        futility_min_cells_per_mechanism: int | None = None,
        futility_min_structural_rate: float = 0.0,
    ) -> dict[str, Any]:
        with ExperimentRootLease(self.evidence_root):
            return self._run_with_lease(
                max_cells=max_cells,
                task_ids=task_ids,
                condition_ids=condition_ids,
                futility_min_cells_per_mechanism=futility_min_cells_per_mechanism,
                futility_min_structural_rate=futility_min_structural_rate,
            )

    def _run_with_lease(
        self,
        *,
        max_cells: int | None,
        task_ids: set[str] | None,
        condition_ids: set[str] | None,
        futility_min_cells_per_mechanism: int | None,
        futility_min_structural_rate: float,
    ) -> dict[str, Any]:
        if max_cells is not None and (type(max_cells) is not int or max_cells < 1):
            raise ContractError("max_cells must be a positive integer")
        known_tasks = {row.task_id for row in self.taskset.tasks}
        known_conditions = {row.condition_id for row in self.conditions}
        if task_ids is not None and not task_ids <= known_tasks:
            raise ContractError("unknown experiment task filter")
        if condition_ids is not None and not condition_ids <= known_conditions:
            raise ContractError("unknown experiment condition filter")
        if futility_min_cells_per_mechanism is not None and (
            type(futility_min_cells_per_mechanism) is not int
            or futility_min_cells_per_mechanism < 2
        ):
            raise ContractError("futility smoke window must contain at least two cells")
        if not 0.0 <= futility_min_structural_rate <= 1.0:
            raise ContractError("futility structural rate must be between zero and one")
        preflight = self.taskset.preflight()
        if not preflight.ready:
            raise ContractError(
                "taskset preflight failed: " + "; ".join(preflight.errors)
            )
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self._stopped_mechanisms = {}
        eligible_task_ids = task_ids or known_tasks
        eligible_condition_ids = condition_ids or known_conditions
        if futility_min_cells_per_mechanism is not None:
            for mechanism in sorted({row.mechanism for row in self.conditions}):
                self._apply_futility_gate(
                    mechanism=mechanism,
                    minimum_cells=futility_min_cells_per_mechanism,
                    minimum_structural_rate=futility_min_structural_rate,
                    eligible_task_ids=eligible_task_ids,
                    eligible_condition_ids=eligible_condition_ids,
                )
        executed = 0
        for task in self.taskset.tasks:
            if task_ids is not None and task.task_id not in task_ids:
                continue
            for condition in self._conditions_for_task(task):
                if condition.mechanism in self._stopped_mechanisms:
                    continue
                if (
                    condition_ids is not None
                    and condition.condition_id not in condition_ids
                ):
                    continue
                target = self._cell_path(task, condition)
                if target.exists():
                    self._load_cell(task, condition)
                    continue
                if max_cells is not None and executed >= max_cells:
                    continue
                self._run_cell(task, condition)
                executed += 1
                if futility_min_cells_per_mechanism is not None:
                    self._apply_futility_gate(
                        mechanism=condition.mechanism,
                        minimum_cells=futility_min_cells_per_mechanism,
                        minimum_structural_rate=futility_min_structural_rate,
                        eligible_task_ids=eligible_task_ids,
                        eligible_condition_ids=eligible_condition_ids,
                    )
                # Cells are the append-only source of truth. Refresh the mutable
                # progress projection after every atomic freeze so an interrupted
                # multi-hour run reports the exact resumable frontier.
                self._write_summary(self.summarize())
        summary = self.summarize()
        self._write_summary(summary)
        return summary

    def _apply_futility_gate(
        self,
        *,
        mechanism: str,
        minimum_cells: int,
        minimum_structural_rate: float,
        eligible_task_ids: set[str],
        eligible_condition_ids: set[str],
    ) -> None:
        reports = [
            self._load_cell(task, condition)
            for task in self.taskset.tasks
            for condition in self._conditions_for_task(task)
            if condition.mechanism == mechanism
            and task.task_id in eligible_task_ids
            and condition.condition_id in eligible_condition_ids
            and self._cell_path(task, condition).exists()
        ]
        if len(reports) < minimum_cells:
            return
        structural = sum(row["attempt"]["structural_valid"] for row in reports)
        rate = structural / len(reports)
        if rate < minimum_structural_rate:
            self._stopped_mechanisms[mechanism] = {
                "completed_cells": len(reports),
                "structural_valid": structural,
                "structural_rate": rate,
                "reason": "structural-rate-below-floor",
            }

    def summarize(self) -> dict[str, Any]:
        reports: list[dict[str, Any]] = []
        for task in self.taskset.tasks:
            for condition in self._conditions_for_task(task):
                target = self._cell_path(task, condition)
                if target.exists():
                    reports.append(self._load_cell(task, condition))
        planned = sum(
            len(self._conditions_for_task(task)) for task in self.taskset.tasks
        )
        # Aggregate rates alone are not an evidence identity: two runs with the
        # same pass/failure counts can contain different model outputs and
        # patches. Bind the mutable projection to the append-only cell records
        # so its SHA is replay-specific without exposing evaluator labels.
        cell_evidence_fingerprint = sha256_json(
            sorted(
                (
                    row["task"]["task_id"],
                    row["condition"]["condition_id"],
                    row["evidence_sha256"],
                )
                for row in reports
            )
        )
        by_condition: dict[str, Any] = {}
        for condition in self.conditions:
            rows = [
                row
                for row in reports
                if row["condition"]["condition_id"] == condition.condition_id
            ]
            reasons = Counter(
                row["attempt"]["failure_reason"]
                for row in rows
                if row["attempt"]["failure_reason"] is not None
            )
            structural = sum(row["attempt"]["structural_valid"] for row in rows)
            by_condition[condition.condition_id] = {
                "completed": len(rows),
                "structural_valid": structural,
                "structural_rate": structural / len(rows) if rows else 0.0,
                "reason_counts": dict(sorted(reasons.items())),
            }
        attempt_counts = [
            sum(
                "attempt-" in str(trace.get("kind", ""))
                for trace in row.get("generation_trace", ())
            )
            for row in reports
        ]
        generation_attempts = sum(attempt_counts)
        retry_attempts = sum(max(0, count - 1) for count in attempt_counts)
        rescued_retries = sum(
            bool(row["attempt"]["structural_valid"]) and count > 1
            for row, count in zip(reports, attempt_counts, strict=True)
        )
        elapsed_total = round(sum(row["elapsed_seconds"] for row in reports), 6)
        structural_total = sum(row["attempt"]["structural_valid"] for row in reports)
        efficiency_metrics = {
            "generation_attempts": generation_attempts,
            "retry_attempts": retry_attempts,
            "rescued_retries": rescued_retries,
            "retry_rescue_rate": (
                rescued_retries / retry_attempts if retry_attempts else 0.0
            ),
            "elapsed_seconds_total": elapsed_total,
            "elapsed_seconds_per_structural_valid": (
                elapsed_total / structural_total if structural_total else None
            ),
        }
        content = {
            "schema_version": 1,
            "status": "complete" if len(reports) == planned else "partial",
            "taskset_id": self.taskset.taskset_id,
            "taskset_fingerprint": self.taskset.fingerprint,
            "cohort_counts": self.taskset.cohort_counts,
            "planned_cells": planned,
            "completed_cells": len(reports),
            "cell_evidence_fingerprint": cell_evidence_fingerprint,
            "condition_metrics": by_condition,
            "efficiency_metrics": efficiency_metrics,
            "stopped_mechanisms": dict(sorted(self._stopped_mechanisms.items())),
            "network_calls_performed": False,
        }
        if self.qualification_fingerprint is not None:
            content["qualification_fingerprint"] = self.qualification_fingerprint
        if self.mechanism_routes is not None:
            content["mechanism_routes"] = self.mechanism_routes
        return {**content, "summary_sha256": sha256_json(content)}

    def _conditions_for_task(
        self, task: EvaluationTask
    ) -> tuple[ExperimentCondition, ...]:
        if self.mechanism_routes is None:
            return self.conditions
        mechanism = self.mechanism_routes[task.task_id]
        return tuple(
            condition
            for condition in self.conditions
            if condition.mechanism == mechanism
        )

    def _run_cell(
        self, task: EvaluationTask, condition: ExperimentCondition
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(
            prefix=f"{task.task_id}-{condition.condition_id}-",
            dir=self.workspace_root,
        ) as temporary:
            checkout = materialize_evaluation_task(task, Path(temporary) / "checkout")
            student_task = StudentTask.create(
                task_id=task.task_id,
                checkout=checkout,
                instruction=task.instruction,
                allowed_targets=list(task.allowed_targets),
                cohort=task.cohort,
            )
            started = time.monotonic()
            adapter = self.adapters[condition.mechanism]
            shared_context = self._bind_shared_context(
                task, student_task, condition, adapter
            )
            attempt = adapter.run(student_task, condition.revision)
            generation_trace = self._generation_trace(adapter)
            realization_evidence = self._realization_evidence(adapter)
            elapsed = time.monotonic() - started
        return self._freeze_cell(
            task,
            condition,
            attempt,
            elapsed,
            generation_trace,
            realization_evidence,
            shared_context,
        )

    def _freeze_cell(
        self,
        task: EvaluationTask,
        condition: ExperimentCondition,
        attempt: StudentAttempt,
        elapsed_seconds: float,
        generation_trace: tuple[
            tuple[str, str, str | None, dict[str, Any] | None], ...
        ],
        realization_evidence: dict[str, Any] | None,
        shared_context: dict[str, str] | None,
    ) -> dict[str, Any]:
        target = self._cell_path(task, condition)
        if target.exists():
            raise ContractError("experiment cell evidence already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{condition.condition_id}-", dir=target.parent)
        )
        raw = temporary / "raw-output.txt"
        patch = temporary / "patch.diff"
        raw.write_text(attempt.raw_output, encoding="utf-8")
        patch.write_text(attempt.patch, encoding="utf-8")
        artifact_hashes = {
            raw.name: self._file_sha256(raw),
            patch.name: self._file_sha256(patch),
        }
        trace_evidence = []
        for index, (kind, output, prompt, stage_result) in enumerate(generation_trace):
            trace_path = temporary / f"generation-output-{index:03d}.txt"
            trace_path.write_text(output, encoding="utf-8")
            digest = self._file_sha256(trace_path)
            artifact_hashes[trace_path.name] = digest
            trace_row = {
                "attempt_index": index,
                "kind": kind,
                "path": trace_path.name,
                "sha256": digest,
            }
            if prompt is not None:
                prompt_path = temporary / f"generation-prompt-{index:03d}.txt"
                prompt_path.write_text(prompt, encoding="utf-8")
                prompt_digest = self._file_sha256(prompt_path)
                artifact_hashes[prompt_path.name] = prompt_digest
                trace_row.update(
                    {
                        "prompt_path": prompt_path.name,
                        "prompt_sha256": prompt_digest,
                    }
                )
            if stage_result is not None:
                trace_row["stage_result"] = stage_result
            trace_evidence.append(trace_row)
        realization_selection = None
        if realization_evidence is not None:
            realization_path = temporary / "realization-selection.json"
            realization_path.write_text(
                canonical_json(realization_evidence) + "\n", encoding="utf-8"
            )
            realization_digest = self._file_sha256(realization_path)
            artifact_hashes[realization_path.name] = realization_digest
            realization_selection = {
                "path": realization_path.name,
                "sha256": realization_digest,
                "evidence_sha256": realization_evidence["evidence_sha256"],
            }
        content = {
            "schema_version": 1,
            "taskset_fingerprint": self.taskset.fingerprint,
            "task": task.to_dict(),
            "condition": condition.to_dict(),
            "attempt": self._attempt_dict(attempt),
            "elapsed_seconds": round(elapsed_seconds, 6),
            "artifact_sha256": artifact_hashes,
            "generation_trace": trace_evidence,
            "network_calls_performed": False,
        }
        if realization_selection is not None:
            content["realization_selection"] = realization_selection
        if shared_context is not None:
            content["shared_context"] = shared_context
        if self.qualification_fingerprint is not None:
            content["qualification_fingerprint"] = self.qualification_fingerprint
        report = {**content, "evidence_sha256": sha256_json(content)}
        (temporary / "ATTEMPT.json").write_text(
            canonical_json(report) + "\n", encoding="utf-8"
        )
        temporary.replace(target)
        return report

    def _bind_shared_context(
        self,
        task: EvaluationTask,
        student_task: StudentTask,
        condition: ExperimentCondition,
        adapter: StudentAdapter,
    ) -> dict[str, str] | None:
        prepare = getattr(adapter, "prepare_shared_context", None)
        bind = getattr(adapter, "bind_shared_context", None)
        if prepare is None and bind is None:
            return None
        if prepare is None or bind is None:
            raise ContractError(
                "shared context adapter requires prepare and bind methods"
            )
        target = (
            self.evidence_root
            / "shared-contexts"
            / task.task_id
            / f"{condition.mechanism}.json"
        )
        if target.exists():
            evidence = self._load_shared_context(
                target, task=task, mechanism=condition.mechanism
            )
        else:
            source = (
                self.shared_context_source_root
                / "shared-contexts"
                / task.task_id
                / f"{condition.mechanism}.json"
                if self.shared_context_source_root is not None
                else None
            )
            if source is not None and source.exists():
                evidence = self._load_shared_context(
                    source, task=task, mechanism=condition.mechanism
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{condition.mechanism}-",
                    suffix=".json",
                    dir=target.parent,
                    delete=False,
                ) as handle:
                    handle.write(source.read_bytes())
                    temporary = Path(handle.name)
                temporary.replace(target)
            else:
                if condition.teaching != "baseline":
                    raise ContractError(
                        "shared context must be prepared by the baseline arm"
                    )
                evidence = prepare(student_task, condition.revision)
                self._validate_shared_context(
                    evidence, task=task, mechanism=condition.mechanism
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    prefix=f".{condition.mechanism}-",
                    suffix=".json",
                    dir=target.parent,
                    delete=False,
                ) as handle:
                    handle.write(canonical_json(evidence) + "\n")
                    temporary = Path(handle.name)
                temporary.replace(target)
        bind(dict(evidence))
        return {
            "path": str(target.relative_to(self.evidence_root)),
            "sha256": self._file_sha256(target),
            "evidence_sha256": evidence["evidence_sha256"],
        }

    def _load_shared_context(
        self, target: Path, *, task: EvaluationTask, mechanism: str
    ) -> dict[str, Any]:
        try:
            evidence = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError("shared context evidence is unreadable") from exc
        self._validate_shared_context(evidence, task=task, mechanism=mechanism)
        return evidence

    @staticmethod
    def _validate_shared_context(
        evidence: Any, *, task: EvaluationTask, mechanism: str
    ) -> None:
        if not isinstance(evidence, dict) or not evidence:
            raise ContractError("shared context evidence must be an object")
        content = {
            key: value for key, value in evidence.items() if key != "evidence_sha256"
        }
        if evidence.get("evidence_sha256") != sha256_json(content):
            raise ContractError("shared context evidence sha256 mismatch")
        if (
            evidence.get("task_id") != task.task_id
            or evidence.get("mechanism") != mechanism
        ):
            raise ContractError("shared context task or mechanism mismatch")
        if (
            evidence.get("native_labels_visible") is not False
            or evidence.get("reference_patch_visible") is not False
        ):
            raise ContractError("shared context leaks evaluator-only evidence")

    @staticmethod
    def _generation_trace(
        adapter: StudentAdapter,
    ) -> tuple[tuple[str, str, str | None, dict[str, Any] | None], ...]:
        trace_reader = getattr(adapter.generator, "generation_trace", None)
        if trace_reader is None:
            return ()
        outputs = trace_reader()
        if not isinstance(outputs, tuple) or any(
            not isinstance(output, str) for output in outputs
        ):
            raise ContractError("student generator trace is invalid")
        kinds_reader = getattr(adapter.generator, "generation_trace_kinds", None)
        kinds = (
            kinds_reader()
            if kinds_reader is not None
            else tuple("generation-attempt" for _output in outputs)
        )
        if (
            not isinstance(kinds, tuple)
            or len(kinds) != len(outputs)
            or any(not isinstance(kind, str) or not kind for kind in kinds)
        ):
            raise ContractError("student generator trace kinds are invalid")
        prompts_reader = getattr(adapter.generator, "generation_prompt_trace", None)
        prompts = (
            prompts_reader()
            if prompts_reader is not None
            else tuple(None for _output in outputs)
        )
        if (
            not isinstance(prompts, tuple)
            or len(prompts) != len(outputs)
            or any(
                prompt is not None and not isinstance(prompt, str) for prompt in prompts
            )
        ):
            raise ContractError("student generator prompt trace is invalid")
        results_reader = getattr(adapter.generator, "generation_trace_results", None)
        results = (
            results_reader()
            if results_reader is not None
            else tuple(None for _output in outputs)
        )
        if (
            not isinstance(results, tuple)
            or len(results) != len(outputs)
            or any(
                result is not None
                and (
                    not isinstance(result, dict)
                    or not isinstance(result.get("status"), str)
                )
                for result in results
            )
        ):
            raise ContractError("student generator stage results are invalid")
        return tuple(zip(kinds, outputs, prompts, results, strict=True))

    @staticmethod
    def _realization_evidence(adapter: StudentAdapter) -> dict[str, Any] | None:
        """Read an optional self-hashed candidate decision after one cell run."""

        reader = getattr(adapter, "realization_evidence", None)
        if reader is None:
            return None
        evidence = reader()
        if evidence is None:
            return None
        if not isinstance(evidence, dict) or not evidence:
            raise ContractError("realization selection evidence must be an object")
        content = {
            key: value for key, value in evidence.items() if key != "evidence_sha256"
        }
        if evidence.get("evidence_sha256") != sha256_json(content):
            raise ContractError("realization selection evidence sha256 mismatch")
        return dict(evidence)

    def _load_cell(
        self, task: EvaluationTask, condition: ExperimentCondition
    ) -> dict[str, Any]:
        target = self._cell_path(task, condition)
        try:
            report = json.loads((target / "ATTEMPT.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError("experiment cell evidence is unreadable") from exc
        if not isinstance(report, dict) or "evidence_sha256" not in report:
            raise ContractError("experiment cell evidence fields are invalid")
        content = {
            key: value for key, value in report.items() if key != "evidence_sha256"
        }
        if report["evidence_sha256"] != sha256_json(content):
            raise ContractError("experiment cell evidence sha256 mismatch")
        if report.get("taskset_fingerprint") != self.taskset.fingerprint:
            raise ContractError("experiment taskset fingerprint mismatch")
        if report.get("task", {}).get("fingerprint") != task.fingerprint:
            raise ContractError("experiment task fingerprint mismatch")
        if report.get("condition", {}).get("fingerprint") != condition.fingerprint:
            raise ContractError("experiment condition fingerprint mismatch")
        if report.get("qualification_fingerprint") != self.qualification_fingerprint:
            raise ContractError("experiment qualification fingerprint mismatch")
        for name, digest in report.get("artifact_sha256", {}).items():
            artifact = target / name
            if not artifact.is_file() or self._file_sha256(artifact) != digest:
                raise ContractError("experiment cell artifact sha256 mismatch")
        shared = report.get("shared_context")
        if shared is not None:
            if not isinstance(shared, dict) or set(shared) != {
                "path",
                "sha256",
                "evidence_sha256",
            }:
                raise ContractError("shared context reference is invalid")
            relative = Path(str(shared["path"]))
            shared_path = (self.evidence_root / relative).resolve()
            try:
                shared_path.relative_to(self.evidence_root)
            except ValueError as exc:
                raise ContractError(
                    "shared context path escapes evidence root"
                ) from exc
            if (
                not shared_path.is_file()
                or self._file_sha256(shared_path) != shared["sha256"]
            ):
                raise ContractError("shared context artifact sha256 mismatch")
            frozen = self._load_shared_context(
                shared_path, task=task, mechanism=condition.mechanism
            )
            if frozen["evidence_sha256"] != shared["evidence_sha256"]:
                raise ContractError("shared context evidence reference mismatch")
        return report

    def _write_summary(self, summary: dict[str, Any]) -> None:
        name = "SUMMARY.json" if summary["status"] == "complete" else "PROGRESS.json"
        target = self.evidence_root / name
        if target.exists() and name == "SUMMARY.json":
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ContractError("experiment summary is unreadable") from exc
            if existing != summary:
                raise ContractError("frozen experiment summary does not match evidence")
            return
        target.write_text(canonical_json(summary) + "\n", encoding="utf-8")

    def _cell_path(self, task: EvaluationTask, condition: ExperimentCondition) -> Path:
        return self.evidence_root / "cells" / task.task_id / condition.condition_id

    @staticmethod
    def _attempt_dict(attempt: StudentAttempt) -> dict[str, Any]:
        return {
            "task_id": attempt.task.task_id,
            "revision_id": attempt.revision_id,
            "raw_output_sha256": attempt.raw_output_sha256,
            "patch_sha256": attempt.patch_sha256,
            "target_file": attempt.target_file,
            "before_sha256": attempt.before_sha256,
            "after_sha256": attempt.after_sha256,
            "implementation_fingerprint": attempt.implementation_fingerprint,
            "structural_valid": attempt.structural_valid,
            "failure_reason": attempt.failure_reason,
            "detail": attempt.detail,
        }

    @staticmethod
    def _file_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
