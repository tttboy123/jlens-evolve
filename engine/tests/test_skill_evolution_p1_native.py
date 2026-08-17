import hashlib
import json
from pathlib import Path

import pytest

from official_patch_evaluator import OfficialEvaluatorError, OfficialEvaluatorTimeout
from skill_evolution_loop.contracts import (
    ContractError,
    canonical_json,
    sha256_json,
)
from skill_evolution_loop.eval_manifest import EvaluationTask, EvaluationTaskSet
from skill_evolution_loop.p1_native import (
    evaluate_p1_feedback_cell_native,
    evaluate_p1_holdout_cell_native,
    multi_materialized_identity,
    normalize_native_report,
    summarize_native_cells,
    validate_paired_native_admission,
    validate_paired_treatment_prompt_identity,
)
from skill_evolution_loop.round1_native import (
    project_round1_holdout_native_summary,
    retry_round1_holdout_native_failures,
    run_round1_feedback_native,
    run_round1_holdout_native,
)


def test_normalizes_swe_summary_and_rejects_ambiguous_outcome() -> None:
    report = {
        "schema_version": 2,
        "submitted_ids": ["repo__repo-1"],
        "resolved_ids": ["repo__repo-1"],
        "unresolved_ids": [],
        "error_ids": [],
        "empty_patch_ids": [],
    }

    outcome = normalize_native_report(
        report, benchmark_id="swe-bench-verified", instance_id="repo__repo-1"
    )

    assert outcome.resolved is True
    assert outcome.native_valid is True
    report["unresolved_ids"] = ["repo__repo-1"]
    with pytest.raises(ContractError, match="ambiguous"):
        normalize_native_report(
            report, benchmark_id="swe-bench-verified", instance_id="repo__repo-1"
        )


def test_normalizes_legacy_swe_regressions() -> None:
    report = {
        "repo__repo-1": {
            "resolved": False,
            "patch_successfully_applied": True,
            "tests_status": {
                "PASS_TO_PASS": {"failure": ["test_b", "test_a"]},
                "PASS_TO_FAIL": {"failure": ["test_a"]},
            },
        }
    }

    outcome = normalize_native_report(
        report, benchmark_id="swe-bench-verified", instance_id="repo__repo-1"
    )

    assert outcome.resolved is False
    assert outcome.native_valid is True
    assert outcome.regression_test_names == ("test_a", "test_b")


def test_normalizes_empty_multi_error_as_no_evaluator_error() -> None:
    report = {
        "org": "clap-rs",
        "repo": "clap",
        "number": 5228,
        "valid": True,
        "error_msg": "",
        "test_patch_result": {"passed_tests": ["stable"], "failed_tests": []},
        "fix_patch_result": {"passed_tests": ["stable"], "failed_tests": []},
    }

    outcome = normalize_native_report(
        report,
        benchmark_id="multi-swe-bench-flash",
        instance_id="clap-rs__clap-5228",
    )

    assert outcome.resolved is True
    assert outcome.native_valid is True
    assert outcome.native_error is None


def test_multi_identity_uses_canonical_frozen_row(tmp_path: Path) -> None:
    row = {"instance_id": "org__repo-7", "org": "org", "repo": "repo", "number": 7}
    dataset = tmp_path / "inputs/multi-swe-bench-flash/multi_swe_bench_flash.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")

    materialized = multi_materialized_identity(tmp_path, "org__repo-7")

    assert materialized == {
        "source_content_sha256": hashlib.sha256(
            canonical_json(row).encode()
        ).hexdigest()
    }


def test_paired_summary_requires_feedback_gain_and_no_holdout_regression() -> None:
    def cell(
        task_id: str,
        cohort: str,
        teaching: str,
        resolved: bool,
        native_error: str | None = None,
        regression_test_names: list[str] | None = None,
    ) -> dict[str, object]:
        return {
            "task_id": task_id,
            "cohort": cohort,
            "mechanism": "structured",
            "teaching": teaching,
            "outcome": {
                "resolved": resolved,
                "native_error": native_error,
                "native_valid": native_error is None,
                "regression_test_names": regression_test_names or [],
            },
        }

    cells = [
        cell("feedback-1", "feedback", "baseline", False),
        cell("feedback-1", "feedback", "taught", True),
        cell("holdout-1", "holdout", "baseline", True),
        cell("holdout-1", "holdout", "taught", True),
    ]

    summary = summarize_native_cells(cells)

    assert summary["feedback_gain_count"] == 1
    assert summary["holdout_regression_count"] == 0
    assert summary["capability_gate_passed"] is True

    cells[-1] = cell("holdout-1", "holdout", "taught", False)
    assert summarize_native_cells(cells)["capability_gate_passed"] is False

    cells = [
        cell("feedback-2", "feedback", "baseline", False, "evaluator_error"),
        cell("feedback-2", "feedback", "taught", True),
    ]
    infrastructure = summarize_native_cells(cells)
    assert infrastructure["feedback_gain_count"] == 0
    assert infrastructure["pairs"][0]["evaluator_valid"] is False

    cells = [
        cell("holdout-2", "holdout", "baseline", True),
        cell(
            "holdout-2",
            "holdout",
            "taught",
            False,
            regression_test_names=["test_existing_behavior"],
        ),
    ]
    safety = summarize_native_cells(cells)
    assert safety["holdout_evaluable_pair_count"] == 1
    assert safety["holdout_safety_failure_count"] == 1
    assert safety["holdout_safety_failures"][0]["regression_test_names"] == [
        "test_existing_behavior"
    ]

    cells = [
        cell(
            "holdout-shared-failure",
            "holdout",
            "baseline",
            False,
            regression_test_names=["test_harness_parser_mismatch"],
        ),
        cell(
            "holdout-shared-failure",
            "holdout",
            "taught",
            False,
            regression_test_names=["test_harness_parser_mismatch"],
        ),
    ]
    shared_failure = summarize_native_cells(cells)
    assert shared_failure["holdout_evaluable_pair_count"] == 1
    assert shared_failure["holdout_safety_failure_count"] == 0

    empty_sha256 = hashlib.sha256(b"").hexdigest()
    no_op_cells = [
        {
            **cell("holdout-no-op", "holdout", teaching, False),
            "patch_sha256": empty_sha256,
            "native_report": None,
            "native_evaluator_failure": None,
            "outcome": {
                "resolved": False,
                "native_error": "structural_invalid",
                "native_valid": False,
                "regression_test_names": [],
            },
        }
        for teaching in ("baseline", "taught")
    ]
    no_op = summarize_native_cells(no_op_cells)
    assert no_op["holdout_evaluable_pair_count"] == 0
    assert no_op["holdout_safety_qualified_pair_count"] == 1
    assert no_op["pairs"][0]["no_op_equivalent"] is True


def _feedback_gate_taskset(
    tmp_path: Path, *, total: int = 6, feedback: int = 3
) -> EvaluationTaskSet:
    return EvaluationTaskSet.create(
        taskset_id="feedback-native-gate",
        tasks=[
            EvaluationTask.create(
                task_id=f"eval-{number}",
                instance_id=f"repo__repo-{number}",
                benchmark_id="swe-bench-verified",
                benchmark_base_commit="a" * 40,
                repo="repo/repo",
                source_repository=tmp_path,
                source_revision="a" * 40,
                instruction=f"Fix task {number}.",
                allowed_targets=["src/example.py"],
                cohort="feedback" if number <= feedback else "holdout",
            )
            for number in range(1, total + 1)
        ],
    )


def _write_attempt(
    path: Path,
    *,
    condition_id: str,
    teaching: str = "taught",
    structural_valid: bool = True,
    implementation_fingerprint: str | None = None,
) -> None:
    path.mkdir(parents=True)
    patch = "--- a/src/example.py\n+++ b/src/example.py\n"
    prompt = f"Teaching Skill:\n{teaching.upper()} treatment\n"
    (path / "generation-prompt-000.txt").write_text(prompt, encoding="utf-8")
    (path / "patch.diff").write_text(patch, encoding="utf-8")
    content = {
        "artifact_sha256": {
            "patch.diff": hashlib.sha256(patch.encode()).hexdigest(),
            "generation-prompt-000.txt": hashlib.sha256(prompt.encode()).hexdigest(),
        },
        "attempt": {
            "structural_valid": structural_valid,
            "implementation_fingerprint": implementation_fingerprint
            or (("c" if teaching == "baseline" else "d") * 64),
        },
        "condition": {
            "condition_id": condition_id,
            "mechanism": "operator",
            "teaching": teaching,
            "revision": {
                "fingerprint": "b" * 64,
                "skill_text": f"{teaching.upper()} treatment",
            },
        },
        "generation_trace": [
            {
                "kind": "operator-plan-attempt-0",
                "path": "generation-output-000.txt",
                "prompt_path": "generation-prompt-000.txt",
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        ],
    }
    (path / "ATTEMPT.json").write_text(
        canonical_json({**content, "evidence_sha256": sha256_json(content)}) + "\n",
        encoding="utf-8",
    )


def test_paired_treatment_prompt_identity_rejects_identical_ab_prompts(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment"
    for teaching in ("baseline", "taught"):
        _write_attempt(
            experiment / "cells/eval-1" / f"operator-{teaching}",
            condition_id=f"operator-{teaching}",
            teaching=teaching,
        )
    taught = experiment / "cells/eval-1/operator-taught"
    baseline_prompt = (
        experiment / "cells/eval-1/operator-baseline/generation-prompt-000.txt"
    ).read_text(encoding="utf-8")
    (taught / "generation-prompt-000.txt").write_text(baseline_prompt, encoding="utf-8")
    attempt_path = taught / "ATTEMPT.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(baseline_prompt.encode()).hexdigest()
    content = {key: value for key, value in attempt.items() if key != "evidence_sha256"}
    content["artifact_sha256"]["generation-prompt-000.txt"] = digest
    content["generation_trace"][0]["prompt_sha256"] = digest
    attempt_path.write_text(
        canonical_json({**content, "evidence_sha256": sha256_json(content)}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="treatment prompts are identical"):
        validate_paired_treatment_prompt_identity(
            experiment_root=experiment,
            task_id="eval-1",
            mechanism="operator",
        )


def test_paired_treatment_prompt_identity_binds_distinct_prompt_evidence(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment"
    for teaching in ("baseline", "taught"):
        _write_attempt(
            experiment / "cells/eval-1" / f"operator-{teaching}",
            condition_id=f"operator-{teaching}",
            teaching=teaching,
        )

    receipt = validate_paired_treatment_prompt_identity(
        experiment_root=experiment,
        task_id="eval-1",
        mechanism="operator",
    )

    assert receipt["causal_pair_valid"] is True
    assert receipt["treatment_prompt_sequences_differ"] is True
    assert (
        receipt["baseline_prompt_sequence_sha256"]
        != (receipt["taught_prompt_sequence_sha256"])
    )
    assert (
        receipt["baseline_skill_text_sha256"] != (receipt["taught_skill_text_sha256"])
    )


def test_native_admission_fails_closed_before_evaluator_for_identical_prompts(
    tmp_path: Path,
) -> None:
    taskset = _feedback_gate_taskset(tmp_path)
    manifest = tmp_path / "TASKSET.json"
    manifest.write_text(canonical_json(taskset.to_dict()) + "\n", encoding="utf-8")
    experiment = tmp_path / "experiment"
    for teaching in ("baseline", "taught"):
        _write_attempt(
            experiment / "cells/eval-1" / f"operator-{teaching}",
            condition_id=f"operator-{teaching}",
            teaching=teaching,
        )
    baseline = experiment / "cells/eval-1/operator-baseline"
    taught = experiment / "cells/eval-1/operator-taught"
    prompt = (baseline / "generation-prompt-000.txt").read_text(encoding="utf-8")
    (taught / "generation-prompt-000.txt").write_text(prompt, encoding="utf-8")
    attempt_path = taught / "ATTEMPT.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    content = {key: value for key, value in attempt.items() if key != "evidence_sha256"}
    content["artifact_sha256"]["generation-prompt-000.txt"] = digest
    content["generation_trace"][0]["prompt_sha256"] = digest
    attempt_path.write_text(
        canonical_json({**content, "evidence_sha256": sha256_json(content)}) + "\n",
        encoding="utf-8",
    )
    calls = 0

    def evaluator(*_args):
        nonlocal calls
        calls += 1
        raise AssertionError("must not run")

    with pytest.raises(ContractError, match="treatment prompts are identical"):
        evaluate_p1_feedback_cell_native(
            manifest_path=manifest,
            experiment_root=experiment,
            evidence_root=tmp_path / "native",
            pool_root=tmp_path,
            evaluator=evaluator,
            task_id="eval-1",
            condition_id="operator-taught",
        )
    assert calls == 0


def test_paired_native_admission_requires_both_structural_arms(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment"
    _write_attempt(
        experiment / "cells/eval-1/operator-baseline",
        condition_id="operator-baseline",
        teaching="baseline",
        structural_valid=True,
    )
    _write_attempt(
        experiment / "cells/eval-1/operator-taught",
        condition_id="operator-taught",
        teaching="taught",
        structural_valid=False,
    )

    receipt = validate_paired_native_admission(
        experiment_root=experiment,
        task_id="eval-1",
        mechanism="operator",
    )

    assert receipt["native_admitted"] is False
    assert receipt["reason"] == "paired-structural-invalid"
    assert receipt["structural_valid"] == {"baseline": True, "taught": False}
    assert receipt["paired_treatment_prompt_identity"] is None


def test_paired_native_admission_rejects_identical_implementation(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment"
    for teaching in ("baseline", "taught"):
        _write_attempt(
            experiment / "cells/eval-1" / f"operator-{teaching}",
            condition_id=f"operator-{teaching}",
            teaching=teaching,
            implementation_fingerprint="e" * 64,
        )

    receipt = validate_paired_native_admission(
        experiment_root=experiment,
        task_id="eval-1",
        mechanism="operator",
    )

    assert receipt["native_admitted"] is False
    assert receipt["reason"] == "paired-implementation-identical"
    assert receipt["implementation_fingerprints_differ"] is False


def test_native_pair_gate_skips_evaluator_when_other_arm_is_invalid(
    tmp_path: Path,
) -> None:
    taskset = _feedback_gate_taskset(tmp_path)
    manifest = tmp_path / "TASKSET.json"
    manifest.write_text(canonical_json(taskset.to_dict()) + "\n", encoding="utf-8")
    experiment = tmp_path / "experiment"
    _write_attempt(
        experiment / "cells/eval-1/operator-baseline",
        condition_id="operator-baseline",
        teaching="baseline",
        structural_valid=True,
    )
    _write_attempt(
        experiment / "cells/eval-1/operator-taught",
        condition_id="operator-taught",
        teaching="taught",
        structural_valid=False,
    )
    calls = 0

    def evaluator(*_args):
        nonlocal calls
        calls += 1
        raise AssertionError("must not run")

    baseline = evaluate_p1_feedback_cell_native(
        manifest_path=manifest,
        experiment_root=experiment,
        evidence_root=tmp_path / "native",
        pool_root=tmp_path,
        evaluator=evaluator,
        task_id="eval-1",
        condition_id="operator-baseline",
    )
    taught = evaluate_p1_feedback_cell_native(
        manifest_path=manifest,
        experiment_root=experiment,
        evidence_root=tmp_path / "native",
        pool_root=tmp_path,
        evaluator=evaluator,
        task_id="eval-1",
        condition_id="operator-taught",
    )

    assert calls == 0
    assert baseline["outcome"]["native_error"] == "paired_structural_invalid"
    assert taught["outcome"]["native_error"] == "structural_invalid"
    assert baseline["paired_native_admission"]["native_admitted"] is False
    assert taught["paired_native_admission"] == baseline["paired_native_admission"]
    summary = summarize_native_cells([baseline, taught])
    assert summary["native_admitted_pair_count"] == 0
    assert summary["paired_structural_invalid_pair_count"] == 1
    assert summary["native_invocations_avoided_by_pair_gate"] == 1


def test_feedback_native_cell_runs_before_complete_experiment_and_blocks_holdout(
    tmp_path: Path,
) -> None:
    taskset = _feedback_gate_taskset(tmp_path)
    manifest = tmp_path / "TASKSET.json"
    manifest.write_text(canonical_json(taskset.to_dict()) + "\n", encoding="utf-8")
    experiment = tmp_path / "experiment"
    _write_attempt(
        experiment / "cells/eval-1/operator-baseline",
        condition_id="operator-baseline",
        teaching="baseline",
    )
    feedback_cell = experiment / "cells/eval-1/operator-taught"
    _write_attempt(feedback_cell, condition_id="operator-taught")
    native_report = tmp_path / "native-report.json"
    native_report.write_text(
        canonical_json(
            {
                "schema_version": 2,
                "submitted_ids": ["repo__repo-1"],
                "resolved_ids": ["repo__repo-1"],
                "unresolved_ids": [],
                "error_ids": [],
                "empty_patch_ids": [],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def evaluator(invocation, materialized, receipt):
        calls.append((invocation, materialized, receipt))
        return native_report

    result = evaluate_p1_feedback_cell_native(
        manifest_path=manifest,
        experiment_root=experiment,
        evidence_root=tmp_path / "evidence",
        pool_root=tmp_path,
        evaluator=evaluator,
        task_id="eval-1",
        condition_id="operator-taught",
    )

    assert result["outcome"]["resolved"] is True
    assert result["holdout_cells_opened"] is False
    assert len(calls) == 1
    assert (
        evaluate_p1_feedback_cell_native(
            manifest_path=manifest,
            experiment_root=experiment,
            evidence_root=tmp_path / "evidence",
            pool_root=tmp_path,
            evaluator=evaluator,
            task_id="eval-1",
            condition_id="operator-taught",
        )
        == result
    )
    assert len(calls) == 1

    _write_attempt(
        experiment / "cells/eval-4/operator-baseline",
        condition_id="operator-baseline",
        teaching="baseline",
    )
    holdout_cell = experiment / "cells/eval-4/operator-taught"
    _write_attempt(holdout_cell, condition_id="operator-taught")
    with pytest.raises(ContractError, match="cannot open holdout"):
        evaluate_p1_feedback_cell_native(
            manifest_path=manifest,
            experiment_root=experiment,
            evidence_root=tmp_path / "evidence",
            pool_root=tmp_path,
            evaluator=evaluator,
            task_id="eval-4",
            condition_id="operator-taught",
        )

    holdout_report = tmp_path / "holdout-native-report.json"
    holdout_report.write_text(
        canonical_json(
            {
                "schema_version": 2,
                "submitted_ids": ["repo__repo-4"],
                "resolved_ids": ["repo__repo-4"],
                "unresolved_ids": [],
                "error_ids": [],
                "empty_patch_ids": [],
            }
        ),
        encoding="utf-8",
    )

    holdout = evaluate_p1_holdout_cell_native(
        manifest_path=manifest,
        experiment_root=experiment,
        evidence_root=tmp_path / "holdout-evidence",
        pool_root=tmp_path,
        evaluator=lambda _invocation, _materialized, _receipt: holdout_report,
        task_id="eval-4",
        condition_id="operator-taught",
        feedback_gain_summary_sha256="c" * 64,
    )

    assert holdout["cohort"] == "holdout"
    assert holdout["holdout_cells_opened"] is True
    assert holdout["feedback_gain_summary_sha256"] == "c" * 64


def test_feedback_native_cell_freezes_recoverable_evaluator_timeout(
    tmp_path: Path,
) -> None:
    taskset = _feedback_gate_taskset(tmp_path)
    manifest = tmp_path / "TASKSET.json"
    manifest.write_text(canonical_json(taskset.to_dict()) + "\n", encoding="utf-8")
    experiment = tmp_path / "experiment"
    _write_attempt(
        experiment / "cells/eval-1/operator-baseline",
        condition_id="operator-baseline",
        teaching="baseline",
    )
    _write_attempt(
        experiment / "cells/eval-1/operator-taught",
        condition_id="operator-taught",
        teaching="taught",
    )
    failure = tmp_path / "official/NATIVE-EVALUATOR-FAILURE.json"
    failure.parent.mkdir()
    failure.write_text('{"error_code":"evaluator_timeout"}\n', encoding="utf-8")
    calls = 0

    def evaluator(_invocation, _materialized, _receipt):
        nonlocal calls
        calls += 1
        raise OfficialEvaluatorTimeout(
            "official evaluator timed out after 17 seconds",
            evidence_path=failure,
        )

    result = evaluate_p1_feedback_cell_native(
        manifest_path=manifest,
        experiment_root=experiment,
        evidence_root=tmp_path / "evidence",
        pool_root=tmp_path,
        evaluator=evaluator,
        task_id="eval-1",
        condition_id="operator-baseline",
    )

    assert result["outcome"] == {
        "resolved": False,
        "native_valid": False,
        "native_error": "evaluator_timeout",
        "regression_test_names": [],
    }
    assert result["native_report"] is None
    assert result["native_evaluator_failure"]["path"] == str(failure.resolve())
    assert (
        result["native_evaluator_failure"]["sha256"]
        == hashlib.sha256(failure.read_bytes()).hexdigest()
    )
    assert (
        evaluate_p1_feedback_cell_native(
            manifest_path=manifest,
            experiment_root=experiment,
            evidence_root=tmp_path / "evidence",
            pool_root=tmp_path,
            evaluator=evaluator,
            task_id="eval-1",
            condition_id="operator-baseline",
        )
        == result
    )
    assert calls == 1


def test_feedback_native_cell_freezes_preflight_failure_without_evaluator_receipt(
    tmp_path: Path,
) -> None:
    taskset = _feedback_gate_taskset(tmp_path)
    manifest = tmp_path / "TASKSET.json"
    manifest.write_text(canonical_json(taskset.to_dict()) + "\n", encoding="utf-8")
    experiment = tmp_path / "experiment"
    _write_attempt(
        experiment / "cells/eval-1/operator-baseline",
        condition_id="operator-baseline",
        teaching="baseline",
    )
    _write_attempt(
        experiment / "cells/eval-1/operator-taught",
        condition_id="operator-taught",
        teaching="taught",
    )

    result = evaluate_p1_feedback_cell_native(
        manifest_path=manifest,
        experiment_root=experiment,
        evidence_root=tmp_path / "evidence",
        pool_root=tmp_path,
        evaluator=lambda *_args: (_ for _ in ()).throw(
            OfficialEvaluatorError("repository prefetch failed")
        ),
        task_id="eval-1",
        condition_id="operator-baseline",
    )

    failure = result["native_evaluator_failure"]
    assert result["outcome"]["native_error"] == "evaluator_error"
    assert failure is not None
    frozen = json.loads(Path(failure["path"]).read_text(encoding="utf-8"))
    assert frozen == {
        "error_code": "evaluator_error",
        "reason": "repository prefetch failed",
        "schema_version": 1,
        "status": "failed",
    }
    assert (
        failure["sha256"]
        == hashlib.sha256(Path(failure["path"]).read_bytes()).hexdigest()
    )


def test_round1_feedback_native_batches_only_feedback_and_resumes(
    tmp_path: Path,
) -> None:
    taskset = _feedback_gate_taskset(tmp_path, total=60, feedback=30)
    manifest = tmp_path / "TASKSET.json"
    manifest.write_text(canonical_json(taskset.to_dict()) + "\n", encoding="utf-8")
    routes_content = {
        "schema_version": 1,
        "taskset_fingerprint": taskset.fingerprint,
        "routes": {task.task_id: "operator" for task in taskset.tasks},
    }
    routes = tmp_path / "ROUTES.json"
    routes.write_text(
        canonical_json(
            {
                **routes_content,
                "evidence_sha256": sha256_json(routes_content),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment"
    for teaching in ("baseline", "taught"):
        condition = f"operator-{teaching}"
        _write_attempt(
            experiment / "cells/eval-1" / condition,
            condition_id=condition,
            teaching=teaching,
        )
    _write_attempt(
        experiment / "cells/eval-31/operator-taught",
        condition_id="operator-taught",
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    calls = []

    def evaluator(invocation, materialized, receipt):
        calls.append(invocation)
        resolved = invocation.arm.endswith("taught")
        report = reports / f"{invocation.arm}.json"
        report.write_text(
            canonical_json(
                {
                    "schema_version": 2,
                    "submitted_ids": ["repo__repo-1"],
                    "resolved_ids": ["repo__repo-1"] if resolved else [],
                    "unresolved_ids": [] if resolved else ["repo__repo-1"],
                    "error_ids": [],
                    "empty_patch_ids": [],
                }
            ),
            encoding="utf-8",
        )
        return report

    partial = run_round1_feedback_native(
        taskset_path=manifest,
        routes_path=routes,
        experiment_root=experiment,
        evidence_root=tmp_path / "native",
        pool_root=tmp_path,
        evaluator=evaluator,
        max_cells=1,
    )
    completed_generated = run_round1_feedback_native(
        taskset_path=manifest,
        routes_path=routes,
        experiment_root=experiment,
        evidence_root=tmp_path / "native",
        pool_root=tmp_path,
        evaluator=evaluator,
    )

    assert partial["completed_cells"] == 1
    assert completed_generated["status"] == "partial"
    assert completed_generated["generated_feedback_cells"] == 2
    assert completed_generated["completed_cells"] == 2
    assert len(completed_generated["cell_evidence_fingerprint"]) == 64
    assert (
        partial["cell_evidence_fingerprint"]
        != (completed_generated["cell_evidence_fingerprint"])
    )
    assert completed_generated["feedback_gain_count"] == 1
    assert completed_generated["native_evaluator_failure_count"] == 0
    assert completed_generated["feedback_gain_gate_passed"] is True
    assert completed_generated["full_capability_gate_evaluated"] is False
    assert completed_generated["holdout_cells_opened"] is False
    assert [(call.round_id, call.arm) for call in calls] == [
        ("eval-1", "operator-baseline"),
        ("eval-1", "operator-taught"),
    ]
    assert not (tmp_path / "native/cells/eval-31").exists()


def test_round1_holdout_native_requires_gain_and_closes_full_capability_gate(
    tmp_path: Path,
) -> None:
    taskset = _feedback_gate_taskset(tmp_path, total=60, feedback=30)
    manifest = tmp_path / "TASKSET.json"
    manifest.write_text(canonical_json(taskset.to_dict()) + "\n", encoding="utf-8")
    routes_content = {
        "schema_version": 1,
        "taskset_fingerprint": taskset.fingerprint,
        "routes": {task.task_id: "operator" for task in taskset.tasks},
    }
    routes = tmp_path / "ROUTES.json"
    routes.write_text(
        canonical_json(
            {**routes_content, "evidence_sha256": sha256_json(routes_content)}
        )
        + "\n",
        encoding="utf-8",
    )
    feedback_content = {
        "schema_version": 1,
        "evaluation_scope": "round1-feedback-only",
        "status": "complete",
        "taskset_fingerprint": taskset.fingerprint,
        "planned_cells": 60,
        "generated_feedback_cells": 60,
        "completed_cells": 60,
        "cell_evidence_fingerprint": "a" * 64,
        "pairs": [
            {
                "task_id": "eval-1",
                "cohort": "feedback",
                "mechanism": "operator",
                "baseline_resolved": False,
                "taught_resolved": True,
                "gained": True,
                "regressed": False,
            }
        ],
        "feedback_gain_count": 1,
        "feedback_gain_gate_passed": True,
        "full_capability_gate_evaluated": False,
        "holdout_cells_opened": False,
        "network_calls_performed": False,
    }
    feedback = {
        **feedback_content,
        "summary_sha256": sha256_json(feedback_content),
    }
    feedback_path = tmp_path / "feedback/SUMMARY.json"
    feedback_path.parent.mkdir()
    feedback_path.write_text(canonical_json(feedback) + "\n", encoding="utf-8")
    experiment = tmp_path / "holdout-experiment"
    for number in range(31, 61):
        for teaching in ("baseline", "taught"):
            condition = f"operator-{teaching}"
            _write_attempt(
                experiment / "cells" / f"eval-{number}" / condition,
                condition_id=condition,
                teaching=teaching,
                structural_valid=False,
            )
    holdout_content = {
        "schema_version": 1,
        "evaluation_scope": "round1-holdout-only",
        "status": "complete",
        "taskset_fingerprint": taskset.fingerprint,
        "planned_cells": 60,
        "completed_cells": 60,
        "feedback_gain_summary_sha256": feedback["summary_sha256"],
        "feedback_gain_count": 1,
        "experiment_projection_sha256": "b" * 64,
        "holdout_cells_opened": True,
        "network_calls_performed": False,
    }
    (experiment / "HOLDOUT-SUMMARY.json").write_text(
        canonical_json(
            {**holdout_content, "summary_sha256": sha256_json(holdout_content)}
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_round1_holdout_native(
        taskset_path=manifest,
        routes_path=routes,
        experiment_root=experiment,
        feedback_gain_path=feedback_path,
        evidence_root=tmp_path / "holdout-native",
        pool_root=tmp_path,
        evaluator=lambda *_args: pytest.fail("structural-invalid cells need no judge"),
    )

    assert summary["status"] == "complete"
    assert summary["completed_holdout_cells"] == 60
    assert summary["feedback_gain_count"] == 1
    assert summary["holdout_regression_count"] == 0
    assert summary["native_evaluator_failure_count"] == 0
    assert summary["holdout_evaluable_pair_count"] == 0
    assert summary["minimum_holdout_evaluable_pairs"] == 3
    assert all(pair["evaluator_valid"] is False for pair in summary["pairs"])
    assert summary["capability_gate_passed"] is False
    assert summary["full_capability_gate_evaluated"] is True
    assert summary["holdout_cells_opened"] is True
    assert "selected_task_ids" not in summary
    assert not (tmp_path / "holdout-native/cells/eval-1").exists()

    projection_path = tmp_path / "projection/SAFETY-QUALIFIED-SUMMARY.json"
    projection = project_round1_holdout_native_summary(
        taskset_path=manifest,
        routes_path=routes,
        experiment_root=experiment,
        feedback_gain_path=feedback_path,
        evidence_root=tmp_path / "holdout-native",
        output_path=projection_path,
    )
    replay = project_round1_holdout_native_summary(
        taskset_path=manifest,
        routes_path=routes,
        experiment_root=experiment,
        feedback_gain_path=feedback_path,
        evidence_root=tmp_path / "holdout-native",
        output_path=projection_path,
    )

    assert replay == projection
    assert (
        projection["cell_evidence_fingerprint"] == summary["cell_evidence_fingerprint"]
    )
    assert projection["aggregation_projection"] == {
        "schema_version": 1,
        "contract": "holdout-safety-qualified-v1",
        "source_native_summary_sha256": summary["summary_sha256"],
        "source_native_summary_file_sha256": hashlib.sha256(
            (tmp_path / "holdout-native/SUMMARY.json").read_bytes()
        ).hexdigest(),
        "source_native_evidence_root": str((tmp_path / "holdout-native").resolve()),
        "native_cells_reexecuted": False,
        "network_calls_performed": False,
    }


def test_round1_holdout_native_explicit_tasks_evaluate_only_a_partial_shard(
    tmp_path: Path,
) -> None:
    taskset = _feedback_gate_taskset(tmp_path, total=60, feedback=30)
    manifest = tmp_path / "TASKSET.json"
    manifest.write_text(canonical_json(taskset.to_dict()) + "\n", encoding="utf-8")
    routes_content = {
        "schema_version": 1,
        "taskset_fingerprint": taskset.fingerprint,
        "routes": {task.task_id: "operator" for task in taskset.tasks},
    }
    routes = tmp_path / "ROUTES.json"
    routes.write_text(
        canonical_json(
            {**routes_content, "evidence_sha256": sha256_json(routes_content)}
        )
        + "\n",
        encoding="utf-8",
    )
    feedback_content = {
        "schema_version": 1,
        "evaluation_scope": "round1-feedback-only",
        "status": "complete",
        "taskset_fingerprint": taskset.fingerprint,
        "planned_cells": 60,
        "generated_feedback_cells": 60,
        "completed_cells": 60,
        "cell_evidence_fingerprint": "a" * 64,
        "pairs": [],
        "feedback_gain_count": 1,
        "feedback_gain_gate_passed": True,
        "full_capability_gate_evaluated": False,
        "holdout_cells_opened": False,
        "network_calls_performed": False,
    }
    feedback = {**feedback_content, "summary_sha256": sha256_json(feedback_content)}
    feedback_path = tmp_path / "feedback/SUMMARY.json"
    feedback_path.parent.mkdir()
    feedback_path.write_text(canonical_json(feedback) + "\n", encoding="utf-8")

    experiment = tmp_path / "holdout-experiment"
    selected = ("eval-31", "eval-32")
    for task_id in selected:
        for teaching in ("baseline", "taught"):
            condition = f"operator-{teaching}"
            _write_attempt(
                experiment / "cells" / task_id / condition,
                condition_id=condition,
                teaching=teaching,
                structural_valid=False,
            )
    progress_content = {
        "schema_version": 1,
        "evaluation_scope": "round1-holdout-only",
        "status": "partial",
        "taskset_fingerprint": taskset.fingerprint,
        "planned_cells": 60,
        "completed_cells": 4,
        "feedback_gain_summary_sha256": feedback["summary_sha256"],
        "feedback_gain_count": 1,
        "experiment_projection_sha256": "b" * 64,
        "holdout_cells_opened": True,
        "network_calls_performed": False,
    }
    (experiment / "HOLDOUT-PROGRESS.json").write_text(
        canonical_json(
            {**progress_content, "summary_sha256": sha256_json(progress_content)}
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_round1_holdout_native(
        taskset_path=manifest,
        routes_path=routes,
        experiment_root=experiment,
        feedback_gain_path=feedback_path,
        evidence_root=tmp_path / "holdout-native-shard",
        pool_root=tmp_path,
        evaluator=lambda *_args: pytest.fail("structural-invalid cells need no judge"),
        task_ids=selected,
    )

    assert summary["evaluation_scope"] == "round1-holdout-shard"
    assert summary["status"] == "complete"
    assert summary["selected_task_ids"] == list(selected)
    assert summary["planned_holdout_cells"] == 4
    assert summary["completed_holdout_cells"] == 4
    assert summary["holdout_regression_count"] == 0
    assert summary["capability_gate_passed"] is False
    assert summary["full_capability_gate_evaluated"] is False
    assert not (tmp_path / "holdout-native-shard/cells/eval-33").exists()


def test_round1_holdout_native_retry_overlays_only_infrastructure_failures(
    tmp_path: Path,
) -> None:
    taskset = _feedback_gate_taskset(tmp_path)
    manifest = tmp_path / "TASKSET.json"
    manifest.write_text(canonical_json(taskset.to_dict()) + "\n", encoding="utf-8")
    routes_content = {
        "schema_version": 1,
        "taskset_fingerprint": taskset.fingerprint,
        "routes": {task.task_id: "operator" for task in taskset.tasks},
    }
    routes = tmp_path / "ROUTES.json"
    routes.write_text(
        canonical_json(
            {**routes_content, "evidence_sha256": sha256_json(routes_content)}
        )
        + "\n",
        encoding="utf-8",
    )
    feedback_content = {
        "schema_version": 1,
        "evaluation_scope": "round1-feedback-only",
        "status": "complete",
        "taskset_fingerprint": taskset.fingerprint,
        "planned_cells": 6,
        "generated_feedback_cells": 6,
        "completed_cells": 6,
        "cell_evidence_fingerprint": "a" * 64,
        "pairs": [],
        "feedback_gain_count": 1,
        "feedback_gain_gate_passed": True,
        "full_capability_gate_evaluated": False,
        "holdout_cells_opened": False,
        "network_calls_performed": False,
    }
    feedback = {**feedback_content, "summary_sha256": sha256_json(feedback_content)}
    feedback_path = tmp_path / "feedback/SUMMARY.json"
    feedback_path.parent.mkdir()
    feedback_path.write_text(canonical_json(feedback) + "\n", encoding="utf-8")
    experiment = tmp_path / "holdout-experiment"
    for number in range(4, 7):
        for teaching in ("baseline", "taught"):
            _write_attempt(
                experiment / "cells" / f"eval-{number}" / f"operator-{teaching}",
                condition_id=f"operator-{teaching}",
                teaching=teaching,
                structural_valid=number == 4,
            )
    holdout_content = {
        "schema_version": 1,
        "evaluation_scope": "round1-holdout-only",
        "status": "complete",
        "taskset_fingerprint": taskset.fingerprint,
        "planned_cells": 6,
        "completed_cells": 6,
        "feedback_gain_summary_sha256": feedback["summary_sha256"],
        "feedback_gain_count": 1,
        "experiment_projection_sha256": "b" * 64,
        "holdout_cells_opened": True,
        "network_calls_performed": False,
    }
    (experiment / "HOLDOUT-SUMMARY.json").write_text(
        canonical_json(
            {**holdout_content, "summary_sha256": sha256_json(holdout_content)}
        )
        + "\n",
        encoding="utf-8",
    )
    failure = tmp_path / "official-failed/NATIVE-EVALUATOR-FAILURE.json"
    failure.parent.mkdir()
    failure.write_text('{"error_code":"evaluator_error"}\n', encoding="utf-8")

    source = run_round1_holdout_native(
        taskset_path=manifest,
        routes_path=routes,
        experiment_root=experiment,
        feedback_gain_path=feedback_path,
        evidence_root=tmp_path / "source-native",
        pool_root=tmp_path,
        evaluator=lambda *_args: (_ for _ in ()).throw(
            OfficialEvaluatorError("cold image failed", evidence_path=failure)
        ),
    )
    assert source["native_evaluator_failure_count"] == 2
    source_cell = tmp_path / "source-native/cells/eval-4/operator-taught"
    source_bytes = (source_cell / "NATIVE-CELL.json").read_bytes()
    report = tmp_path / "retry-report.json"
    report.write_text(
        canonical_json(
            {
                "schema_version": 2,
                "submitted_ids": ["repo__repo-4"],
                "resolved_ids": ["repo__repo-4"],
                "unresolved_ids": [],
                "error_ids": [],
                "empty_patch_ids": [],
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def retry_evaluator(invocation, _materialized, _receipt):
        calls.append(invocation.arm)
        return report

    projected = retry_round1_holdout_native_failures(
        taskset_path=manifest,
        routes_path=routes,
        experiment_root=experiment,
        feedback_gain_path=feedback_path,
        source_evidence_root=tmp_path / "source-native",
        retry_evidence_root=tmp_path / "retry-native",
        pool_root=tmp_path,
        evaluator=retry_evaluator,
        output_path=tmp_path / "projection/RETRIED-SUMMARY.json",
    )

    assert calls == ["operator-baseline", "operator-taught"]
    assert (source_cell / "NATIVE-CELL.json").read_bytes() == source_bytes
    assert projected["native_evaluator_failure_count"] == 0
    assert projected["holdout_evaluable_pair_count"] == 1
    assert projected["aggregation_projection"]["contract"] == (
        "native-infrastructure-retry-overlay-v1"
    )
    assert projected["aggregation_projection"]["replaced_failure_count"] == 2
    assert projected["aggregation_projection"]["native_cells_reexecuted"] is True
