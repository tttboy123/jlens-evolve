"""Eval-driven tests for resumable paired Student experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from skill_evolution_loop import (
    ContractError,
    EvaluationTask,
    EvaluationTaskSet,
    ExperimentCondition,
    ExperimentEvidenceSource,
    LoopRevision,
    PairedExperimentRunner,
    StudentAdapter,
    StudentAttempt,
    TargetSelectionManifest,
    TargetSelectionRecord,
    compose_experiment_evidence,
    freeze_p1_parent_request,
)
from skill_evolution_loop.contracts import sha256_json


def _source_repo(path: Path) -> tuple[Path, str]:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    source = path / "src/example.py"
    source.parent.mkdir()
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Experiment Test",
            "-c",
            "user.email=experiment@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=path,
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()
    return path, revision


def _taskset(source: Path, revision: str) -> EvaluationTaskSet:
    return EvaluationTaskSet.create(
        taskset_id="experiment-fixture",
        tasks=[
            EvaluationTask.create(
                task_id=f"eval-{number:03d}",
                instance_id=f"project__repo-{number}",
                benchmark_id="fixture",
                benchmark_base_commit=revision,
                repo="project/repo",
                source_repository=source,
                source_revision=revision,
                instruction=f"Change value for task {number}.",
                allowed_targets=["src/example.py"],
                cohort="feedback" if number <= 3 else "holdout",
            )
            for number in range(1, 7)
        ],
    )


def _revision(revision_id: str, skill: str) -> LoopRevision:
    return LoopRevision.create(
        skill_id="experiment-skill",
        revision_id=revision_id,
        parent_revision_id=None,
        source_round=0,
        protocol="structured-search-replace-v1",
        skill_text=skill,
        prompt_template="Return a bounded edit.",
        eval_note="experiment fixture",
    )


def test_paired_runner_freezes_24_cells_and_resumes_without_regeneration(
    tmp_path: Path,
) -> None:
    source, revision = _source_repo(tmp_path / "source")
    taskset = _taskset(source, revision)
    calls: list[tuple[str, str]] = []

    def generator(task, skill_revision):
        calls.append((task.task_id, skill_revision.revision_id))
        return json.dumps(
            {
                "file": "src/example.py",
                "search": "value = 1",
                "replace": "value = 2",
                "diagnostic": "fixture edit",
            }
        )

    revisions = {
        "baseline": _revision("baseline-001", "No teaching."),
        "taught": _revision("taught-001", "Use one exact span."),
    }
    conditions = [
        ExperimentCondition.create(
            condition_id=f"{mechanism}-{teaching}",
            mechanism=mechanism,
            teaching=teaching,
            revision=revisions[teaching],
        )
        for mechanism in ("structured", "hunk")
        for teaching in ("baseline", "taught")
    ]
    runner = PairedExperimentRunner(
        taskset=taskset,
        adapters={
            "structured": StudentAdapter(generator=generator),
            "hunk": StudentAdapter(generator=generator),
        },
        conditions=conditions,
        evidence_root=tmp_path / "evidence",
        workspace_root=tmp_path / "workspaces",
        qualification_fingerprint="a" * 64,
    )

    partial = runner.run(max_cells=2)
    completed = runner.run()
    resumed = runner.run()

    assert partial["status"] == "partial"
    assert partial["completed_cells"] == 2
    assert completed["status"] == "complete"
    assert completed["completed_cells"] == 24
    assert completed["planned_cells"] == 24
    assert resumed == completed
    assert completed["qualification_fingerprint"] == "a" * 64
    assert len(calls) == 24
    cell = tmp_path / "evidence/cells/eval-001/structured-baseline"
    assert (cell / "ATTEMPT.json").is_file()
    assert (cell / "raw-output.txt").is_file()
    assert (cell / "patch.diff").is_file()
    report = json.loads((cell / "ATTEMPT.json").read_text(encoding="utf-8"))
    assert report["qualification_fingerprint"] == "a" * 64

    mismatched = PairedExperimentRunner(
        taskset=taskset,
        adapters={
            "structured": StudentAdapter(generator=generator),
            "hunk": StudentAdapter(generator=generator),
        },
        conditions=conditions,
        evidence_root=tmp_path / "evidence",
        workspace_root=tmp_path / "workspaces",
        qualification_fingerprint="b" * 64,
    )
    with pytest.raises(ContractError, match="qualification fingerprint"):
        mismatched.run(max_cells=1)


def test_paired_runner_refreshes_progress_after_each_frozen_cell(
    tmp_path: Path,
) -> None:
    source, revision = _source_repo(tmp_path / "source")
    taskset = _taskset(source, revision)

    def generator(task, skill_revision):
        return json.dumps(
            {
                "file": "src/example.py",
                "search": "value = 1",
                "replace": "value = 2",
                "diagnostic": f"{task.task_id}/{skill_revision.revision_id}",
            }
        )

    delegate = StudentAdapter(generator=generator)

    class InterruptingAdapter:
        generator = object()

        def __init__(self) -> None:
            self.calls = 0

        def run(self, task, skill_revision):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated interruption")
            return delegate.run(task, skill_revision)

    conditions = [
        ExperimentCondition.create(
            condition_id=f"structured-{teaching}",
            mechanism="structured",
            teaching=teaching,
            revision=_revision(f"{teaching}-progress", teaching),
        )
        for teaching in ("baseline", "taught")
    ]
    runner = PairedExperimentRunner(
        taskset=taskset,
        adapters={"structured": InterruptingAdapter()},
        conditions=conditions,
        evidence_root=tmp_path / "evidence",
        workspace_root=tmp_path / "workspaces",
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.run()

    progress = json.loads(
        (tmp_path / "evidence/PROGRESS.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "partial"
    assert progress["completed_cells"] == 1
    assert progress["planned_cells"] == 12
    assert len(progress["cell_evidence_fingerprint"]) == 64

    attempt = json.loads(
        next((tmp_path / "evidence/cells").glob("*/*/ATTEMPT.json")).read_text(
            encoding="utf-8"
        )
    )
    assert progress["cell_evidence_fingerprint"] == sha256_json(
        [
            [
                attempt["task"]["task_id"],
                attempt["condition"]["condition_id"],
                attempt["evidence_sha256"],
            ]
        ]
    )


def test_paired_runner_stops_futile_mechanism_after_smoke_window(
    tmp_path: Path,
) -> None:
    source, revision = _source_repo(tmp_path / "source")
    taskset = _taskset(source, revision)
    calls = 0

    def invalid_generator(_task, _skill_revision):
        nonlocal calls
        calls += 1
        return "UNRESOLVED: no bounded edit"

    conditions = [
        ExperimentCondition.create(
            condition_id=f"structured-{teaching}",
            mechanism="structured",
            teaching=teaching,
            revision=_revision(f"{teaching}-futility", teaching),
        )
        for teaching in ("baseline", "taught")
    ]
    runner = PairedExperimentRunner(
        taskset=taskset,
        adapters={"structured": StudentAdapter(generator=invalid_generator)},
        conditions=conditions,
        evidence_root=tmp_path / "evidence",
        workspace_root=tmp_path / "workspaces",
    )

    summary = runner.run(
        futility_min_cells_per_mechanism=4,
        futility_min_structural_rate=0.01,
    )

    assert calls == 4
    assert summary["status"] == "partial"
    assert summary["stopped_mechanisms"] == {
        "structured": {
            "completed_cells": 4,
            "structural_valid": 0,
            "structural_rate": 0.0,
            "reason": "structural-rate-below-floor",
        }
    }
    assert summary["efficiency_metrics"]["generation_attempts"] == 0
    assert summary["efficiency_metrics"]["elapsed_seconds_total"] >= 0


def test_paired_runner_routes_each_task_to_one_frozen_mechanism(
    tmp_path: Path,
) -> None:
    source, revision = _source_repo(tmp_path / "source")
    taskset = _taskset(source, revision)

    def generator(task, skill_revision):
        return json.dumps(
            {
                "file": "src/example.py",
                "search": "value = 1",
                "replace": "value = 2",
                "diagnostic": f"{task.task_id}/{skill_revision.revision_id}",
            }
        )

    revisions = {
        name: _revision(f"{name}-routed", name) for name in ("baseline", "taught")
    }
    conditions = [
        ExperimentCondition.create(
            condition_id=f"{mechanism}-{teaching}",
            mechanism=mechanism,
            teaching=teaching,
            revision=revisions[teaching],
        )
        for mechanism in ("structured", "hunk")
        for teaching in ("baseline", "taught")
    ]
    routes = {
        task.task_id: "structured" if index < 3 else "hunk"
        for index, task in enumerate(taskset.tasks)
    }
    runner = PairedExperimentRunner(
        taskset=taskset,
        adapters={
            "structured": StudentAdapter(generator=generator),
            "hunk": StudentAdapter(generator=generator),
        },
        conditions=conditions,
        evidence_root=tmp_path / "evidence",
        workspace_root=tmp_path / "workspaces",
        mechanism_routes=routes,
    )

    summary = runner.run()

    assert summary["planned_cells"] == summary["completed_cells"] == 12
    assert summary["mechanism_routes"] == routes
    assert not (tmp_path / "evidence/cells/eval-001/hunk-baseline").exists()
    assert not (tmp_path / "evidence/cells/eval-006/structured-baseline").exists()


def test_paired_runner_freezes_every_local_generation_repair_output(
    tmp_path: Path,
) -> None:
    source, revision = _source_repo(tmp_path / "source")
    taskset = _taskset(source, revision)

    class TracedGenerator:
        def __init__(self) -> None:
            self.trace: tuple[str, ...] = ()

        def __call__(self, task, skill_revision):
            rejected = '{"search":"same","replace":"same"}'
            accepted = json.dumps(
                {
                    "file": "src/example.py",
                    "search": "value = 1",
                    "replace": "value = 2",
                    "diagnostic": "repaired fixture",
                }
            )
            self.trace = (rejected, accepted)
            return accepted

        def generation_trace(self) -> tuple[str, ...]:
            return self.trace

        def generation_prompt_trace(self) -> tuple[str, ...]:
            return ("fixture prompt 0", "fixture prompt 1")

        def generation_trace_results(self) -> tuple[dict[str, str], ...]:
            return (
                {"status": "structural-rejected"},
                {"status": "structural-valid"},
            )

    generator = TracedGenerator()
    condition = ExperimentCondition.create(
        condition_id="structured-taught",
        mechanism="structured",
        teaching="taught",
        revision=_revision("taught-traced", "Use repair evidence."),
    )
    runner = PairedExperimentRunner(
        taskset=taskset,
        adapters={"structured": StudentAdapter(generator=generator)},
        conditions=[condition],
        evidence_root=tmp_path / "evidence",
        workspace_root=tmp_path / "workspaces",
    )

    runner.run(max_cells=1)

    cell = tmp_path / "evidence/cells/eval-001/structured-taught"
    report = json.loads((cell / "ATTEMPT.json").read_text(encoding="utf-8"))
    assert [row["attempt_index"] for row in report["generation_trace"]] == [0, 1]
    assert [row["kind"] for row in report["generation_trace"]] == [
        "generation-attempt",
        "generation-attempt",
    ]
    assert (cell / "generation-output-000.txt").read_text() == generator.trace[0]
    assert (cell / "generation-output-001.txt").read_text() == generator.trace[1]
    assert (cell / "generation-prompt-000.txt").read_text() == "fixture prompt 0"
    assert (cell / "generation-prompt-001.txt").read_text() == "fixture prompt 1"
    assert (
        report["generation_trace"][1]["prompt_sha256"]
        == report["artifact_sha256"]["generation-prompt-001.txt"]
    )
    assert report["generation_trace"][0]["stage_result"] == {
        "status": "structural-rejected"
    }
    assert report["generation_trace"][1]["stage_result"] == {
        "status": "structural-valid"
    }


def test_paired_runner_freezes_optional_realization_selection_evidence(
    tmp_path: Path,
) -> None:
    source, revision = _source_repo(tmp_path / "source")
    taskset = _taskset(source, revision)

    def generator(task, skill_revision):
        return json.dumps(
            {
                "file": "src/example.py",
                "search": "value = 1",
                "replace": "value = 2",
                "diagnostic": f"{task.task_id}/{skill_revision.revision_id}",
            }
        )

    delegate = StudentAdapter(generator=generator)

    class RealizationAdapter:
        generator = object()

        def run(self, task, skill_revision):
            self._evidence = {
                "schema_version": 1,
                "diagnosis_sha256": "d" * 64,
                "selected_candidate_id": "candidate-001",
            }
            from skill_evolution_loop.contracts import sha256_json

            self._evidence["evidence_sha256"] = sha256_json(self._evidence)
            return delegate.run(task, skill_revision)

        def realization_evidence(self):
            return self._evidence

    condition = ExperimentCondition.create(
        condition_id="structured-taught",
        mechanism="structured",
        teaching="taught",
        revision=_revision("taught-realization", "Use bounded candidates."),
    )
    runner = PairedExperimentRunner(
        taskset=taskset,
        adapters={"structured": RealizationAdapter()},
        conditions=[condition],
        evidence_root=tmp_path / "evidence",
        workspace_root=tmp_path / "workspaces",
    )

    runner.run(max_cells=1)

    cell = tmp_path / "evidence/cells/eval-001/structured-taught"
    report = json.loads((cell / "ATTEMPT.json").read_text(encoding="utf-8"))
    frozen = json.loads(
        (cell / "realization-selection.json").read_text(encoding="utf-8")
    )
    assert frozen["selected_candidate_id"] == "candidate-001"
    assert report["realization_selection"] == {
        "path": "realization-selection.json",
        "sha256": report["artifact_sha256"]["realization-selection.json"],
        "evidence_sha256": frozen["evidence_sha256"],
    }


def test_paired_runner_freezes_and_reuses_one_shared_context_per_task(
    tmp_path: Path,
) -> None:
    source, revision = _source_repo(tmp_path / "source")
    taskset = _taskset(source, revision)

    def generator(task, skill_revision):
        return json.dumps(
            {
                "file": "src/example.py",
                "search": "value = 1",
                "replace": "value = 2",
                "diagnostic": f"{task.task_id}/{skill_revision.revision_id}",
            }
        )

    delegate = StudentAdapter(generator=generator)

    class SharedContextAdapter:
        generator = object()

        def __init__(self) -> None:
            self.prepare_calls = 0
            self.bound: list[str] = []

        def prepare_shared_context(self, task, _revision):
            self.prepare_calls += 1
            content = {
                "schema_version": 1,
                "contract": "fixture-shared-context-v1",
                "task_id": task.task_id,
                "mechanism": "structured",
                "native_labels_visible": False,
                "reference_patch_visible": False,
                "diagnosis": "value is stale",
                "localization": ["src/example.py"],
            }
            from skill_evolution_loop.contracts import sha256_json

            return {**content, "evidence_sha256": sha256_json(content)}

        def bind_shared_context(self, evidence):
            self.bound.append(evidence["evidence_sha256"])

        def run(self, task, skill_revision):
            assert self.bound
            return delegate.run(task, skill_revision)

    adapter = SharedContextAdapter()
    conditions = [
        ExperimentCondition.create(
            condition_id=f"structured-{teaching}",
            mechanism="structured",
            teaching=teaching,
            revision=_revision(f"{teaching}-shared", teaching),
        )
        for teaching in ("baseline", "taught")
    ]
    evidence_root = tmp_path / "evidence"
    runner = PairedExperimentRunner(
        taskset=taskset,
        adapters={"structured": adapter},
        conditions=conditions,
        evidence_root=evidence_root,
        workspace_root=tmp_path / "workspaces",
    )

    runner.run(task_ids={"eval-001"})
    runner.run(task_ids={"eval-001"})

    shared_path = evidence_root / "shared-contexts/eval-001/structured.json"
    shared = json.loads(shared_path.read_text(encoding="utf-8"))
    baseline = json.loads(
        (evidence_root / "cells/eval-001/structured-baseline/ATTEMPT.json").read_text()
    )
    taught = json.loads(
        (evidence_root / "cells/eval-001/structured-taught/ATTEMPT.json").read_text()
    )
    assert adapter.prepare_calls == 1
    assert len(set(adapter.bound)) == 1
    assert baseline["shared_context"]["evidence_sha256"] == shared["evidence_sha256"]
    assert taught["shared_context"] == baseline["shared_context"]
    assert shared["native_labels_visible"] is False
    assert shared["reference_patch_visible"] is False

    shared["diagnosis"] = "tampered"
    shared_path.write_text(json.dumps(shared), encoding="utf-8")
    with pytest.raises(ContractError, match="shared context artifact sha256"):
        runner.run(task_ids={"eval-001"})


def test_paired_runner_imports_frozen_shared_context_without_repreparing(
    tmp_path: Path,
) -> None:
    source, revision = _source_repo(tmp_path / "source")
    taskset = _taskset(source, revision)
    source_root = tmp_path / "source-evidence"
    imported = source_root / "shared-contexts/eval-001/structured.json"
    imported.parent.mkdir(parents=True)
    content = {
        "schema_version": 1,
        "contract": "fixture-shared-context-v1",
        "task_id": "eval-001",
        "mechanism": "structured",
        "native_labels_visible": False,
        "reference_patch_visible": False,
        "diagnosis": "frozen",
    }
    imported.write_text(
        json.dumps({**content, "evidence_sha256": sha256_json(content)})
    )

    class Adapter:
        generator = object()

        def __init__(self) -> None:
            self.bound = []

        def prepare_shared_context(self, *_args):
            raise AssertionError("imported context must not be regenerated")

        def bind_shared_context(self, evidence):
            self.bound.append(evidence)

        def run(self, task, revision):
            return StudentAttempt(
                task=task,
                revision_id=revision.revision_id,
                raw_output="unresolved",
                raw_output_sha256=hashlib.sha256(b"unresolved").hexdigest(),
                edit=None,
                patch="",
                patch_sha256=None,
                target_file=None,
                before_sha256=None,
                after_sha256=None,
                implementation_fingerprint=None,
                structural_valid=False,
                failure_reason="unresolved",
                detail="fixture",
            )

    adapter = Adapter()
    conditions = [
        ExperimentCondition.create(
            condition_id="structured-baseline",
            mechanism="structured",
            teaching="baseline",
            revision=_revision("baseline-import", "baseline"),
        )
    ]
    evidence_root = tmp_path / "evidence"
    runner = PairedExperimentRunner(
        taskset=taskset,
        adapters={"structured": adapter},
        conditions=conditions,
        evidence_root=evidence_root,
        workspace_root=tmp_path / "workspaces",
        shared_context_source_root=source_root,
    )

    runner.run(task_ids={"eval-001"}, max_cells=1)

    target = evidence_root / "shared-contexts/eval-001/structured.json"
    assert target.read_bytes() == imported.read_bytes()
    assert adapter.bound[0]["diagnosis"] == "frozen"


def test_composition_reuses_qualified_cohorts_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    source, revision = _source_repo(tmp_path / "source")
    taskset = _taskset(source, revision)
    selection = TargetSelectionManifest.create(
        taskset=taskset,
        records=[
            TargetSelectionRecord.create(
                task=task,
                selector_id="fixture-selector-v1",
                evidence=["Issue instruction names src/example.py."],
            )
            for task in taskset.tasks
        ],
    )
    taskset_path = tmp_path / "TASKSET.json"
    selection_path = tmp_path / "TARGET-SELECTION.json"
    taskset_path.write_text(
        json.dumps(taskset.to_dict(), sort_keys=True), encoding="utf-8"
    )
    selection_path.write_text(
        json.dumps(selection.to_dict(), sort_keys=True), encoding="utf-8"
    )

    def generator(task, skill_revision):
        return json.dumps(
            {
                "file": "src/example.py",
                "search": "value = 1",
                "replace": "value = 2",
                "diagnostic": skill_revision.revision_id,
            }
        )

    conditions = [
        ExperimentCondition.create(
            condition_id=teaching,
            mechanism="structured",
            teaching=teaching,
            revision=_revision(teaching, f"{teaching} teaching"),
        )
        for teaching in ("baseline", "taught")
    ]
    roots = {
        "feedback": tmp_path / "feedback-evidence",
        "holdout": tmp_path / "holdout-evidence",
    }
    for cohort, evidence_root in roots.items():
        runner = PairedExperimentRunner(
            taskset=taskset,
            adapters={"structured": StudentAdapter(generator=generator)},
            conditions=conditions,
            evidence_root=evidence_root,
            workspace_root=tmp_path / f"{cohort}-workspaces",
            qualification_fingerprint=selection.fingerprint,
        )
        runner.run(
            task_ids={task.task_id for task in taskset.tasks if task.cohort == cohort}
        )

    sources = [
        ExperimentEvidenceSource(
            cohort=cohort,
            experiment_root=root,
            taskset_path=taskset_path,
            target_selection_path=selection_path,
        )
        for cohort, root in roots.items()
    ]
    output = tmp_path / "COMPOSITION.json"
    report = compose_experiment_evidence(
        taskset_path=taskset_path,
        target_selection_path=selection_path,
        sources=sources,
        output_path=output,
    )

    assert report["status"] == "complete"
    assert report["completed_cells"] == 12
    assert report["planned_cells"] == 12
    assert (
        compose_experiment_evidence(
            taskset_path=taskset_path,
            target_selection_path=selection_path,
            sources=sources,
            output_path=output,
        )
        == report
    )

    semantic_path = tmp_path / "SEMANTIC-REVIEW.json"
    semantic_path.write_text(
        json.dumps(
            {
                "taskset_fingerprint": taskset.fingerprint,
                "composition_sha256": report["composition_sha256"],
                "rows": [
                    {
                        "task_id": task.task_id,
                        "cohort": task.cohort,
                        "evidence": f"{task.task_id} remained unresolved.",
                    }
                    for task in taskset.tasks
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    request_report = freeze_p1_parent_request(
        composition_path=output,
        semantic_review_path=semantic_path,
        output_path=tmp_path / "PARENT-REQUEST.json",
        condition_id="taught",
    )
    request_tasks = {
        row["task_id"]
        for row in request_report["parent_request"]["feedback"]["arm_evidence"]
    }
    assert request_report["feedback_task_count"] == 3
    assert request_report["holdout_task_ids_included"] is False
    assert request_tasks == {"eval-001", "eval-002", "eval-003"}

    patch = roots["holdout"] / "cells/eval-004/baseline/patch.diff"
    patch.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ContractError, match="artifact sha256 mismatch"):
        compose_experiment_evidence(
            taskset_path=taskset_path,
            target_selection_path=selection_path,
            sources=sources,
            output_path=tmp_path / "TAMPERED.json",
        )
