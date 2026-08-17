from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from skill_evolution_loop.contracts import ContractError, canonical_json, sha256_json
from skill_evolution_loop.eval_manifest import EvaluationTask, EvaluationTaskSet
from skill_evolution_loop.round1_feedback import (
    _parse_strategy,
    compile_round1_feedback_skills,
    create_round1_feedback_authorization,
    dispatch_round1_feedback_strategy,
    freeze_round1_feedback_request,
    freeze_round1_realization_feedback_request,
    freeze_round1_targeted_native_feedback_request,
)
from teacher_api import TeacherConfig, TeacherProvider, TeacherResponse


class _Client:
    config = TeacherConfig(
        provider=TeacherProvider.DEEPSEEK,
        api_base="https://example.invalid",
        model="deepseek-v4-flash",
        api_key_env="TEST_KEY",
    )

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, _sample):
        self.calls += 1
        response = {
            "failure_diagnosis": "Selectors are copied semantically, not exactly.",
            "mechanism_revision": "Add deterministic selector candidates.",
            "operator_skill_requirements": ["Choose one supplied exact selector."],
            "span_skill_requirements": ["Copy one unique bounded before span."],
            "verification_changes": [
                "Classify selector and semantic failures separately."
            ],
            "next_experiment": ["Compile inactive r10 skills", "rerun feedback A/B"],
        }
        return TeacherResponse(
            provider=TeacherProvider.DEEPSEEK,
            model="deepseek-v4-flash",
            text=canonical_json(response),
            usage={
                "prompt_tokens": 300,
                "completion_tokens": 700,
                "total_tokens": 1000,
            },
        )


def test_feedback_strategy_normalizes_one_next_experiment_string() -> None:
    strategy = _parse_strategy(
        canonical_json(
            {
                "failure_diagnosis": "The Skill conflicts with the renderer.",
                "mechanism_revision": "Align the typed operator contract.",
                "operator_skill_requirements": ["Allow replace_condition."],
                "span_skill_requirements": ["Copy one predicate."],
                "verification_changes": ["Parse the replacement."],
                "next_experiment": "Rerun the same frozen A/B pair.",
            }
        )
    )

    assert strategy["next_experiment"] == ["Rerun the same frozen A/B pair."]


def _freeze(path: Path, content: dict[str, object]) -> dict[str, object]:
    payload = {**content, "evidence_sha256": sha256_json(content)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    taskset = EvaluationTaskSet.create(
        taskset_id="round1-feedback-request",
        tasks=[
            EvaluationTask.create(
                task_id=f"eval-{number}",
                instance_id=f"repo__repo-{number}",
                benchmark_id="swe-bench-verified",
                benchmark_base_commit="a" * 40,
                repo="repo/repo",
                source_repository=tmp_path,
                source_revision="a" * 40,
                instruction=f"Fix feedback issue {number}.",
                allowed_targets=["src/example.py"],
                cohort="feedback" if number <= 30 else "holdout",
            )
            for number in range(1, 61)
        ],
    )
    manifest = tmp_path / "TASKSET.json"
    manifest.write_text(canonical_json(taskset.to_dict()) + "\n", encoding="utf-8")
    routes_content = {
        "schema_version": 1,
        "taskset_fingerprint": taskset.fingerprint,
        "routes": {task.task_id: "operator" for task in taskset.tasks},
    }
    routes = tmp_path / "ROUTES.json"
    _freeze(routes, routes_content)
    experiment = tmp_path / "experiment"
    native = tmp_path / "native"
    pairs = []
    for number in range(1, 31):
        task_id = f"eval-{number}"
        for teaching in ("baseline", "taught"):
            condition_id = f"operator-{teaching}"
            cell = experiment / "cells" / task_id / condition_id
            cell.mkdir(parents=True)
            raw = f"{teaching} feedback raw {number}"
            (cell / "raw-output.txt").write_text(raw, encoding="utf-8")
            attempt_content = {
                "schema_version": 1,
                "taskset_fingerprint": taskset.fingerprint,
                "task": {
                    "task_id": task_id,
                    "cohort": "feedback",
                    "instruction": f"Fix feedback issue {number}.",
                },
                "condition": {
                    "condition_id": condition_id,
                    "mechanism": "operator",
                    "teaching": teaching,
                    "revision": {
                        "revision_id": f"operator-{teaching}-r9",
                        "fingerprint": ("b" if teaching == "baseline" else "c") * 64,
                        "skill_text": (
                            "No teaching."
                            if teaching == "baseline"
                            else "active: false\nOperator teaching."
                        ),
                    },
                },
                "attempt": {
                    "structural_valid": False,
                    "failure_reason": "apply-fail",
                    "detail": f"selector mismatch {number}",
                    "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "patch_sha256": None,
                },
                "artifact_sha256": {},
                "network_calls_performed": False,
            }
            attempt = _freeze(cell / "ATTEMPT.json", attempt_content)
            native_content = {
                "schema_version": 1,
                "taskset_fingerprint": taskset.fingerprint,
                "task_id": task_id,
                "cohort": "feedback",
                "condition_id": condition_id,
                "mechanism": "operator",
                "teaching": teaching,
                "experiment_cell_sha256": attempt["evidence_sha256"],
                "holdout_cells_opened": False,
                "outcome": {
                    "resolved": False,
                    "native_valid": False,
                    "native_error": None,
                    "regression_test_names": [],
                },
                "native_report": None,
                "network_calls_performed": False,
            }
            _freeze(
                native / "cells" / task_id / condition_id / "NATIVE-CELL.json",
                native_content,
            )
        pairs.append(
            {
                "task_id": task_id,
                "cohort": "feedback",
                "mechanism": "operator",
                "baseline_resolved": False,
                "taught_resolved": False,
                "gained": False,
                "regressed": False,
            }
        )
    summary_content = {
        "schema_version": 1,
        "evaluation_scope": "round1-feedback-only",
        "status": "complete",
        "taskset_fingerprint": taskset.fingerprint,
        "planned_cells": 60,
        "generated_feedback_cells": 60,
        "completed_cells": 60,
        "cell_evidence_fingerprint": "d" * 64,
        "native_invocations": 0,
        "pairs": pairs,
        "feedback_gain_count": 0,
        "feedback_gain_gate_passed": False,
        "full_capability_gate_evaluated": False,
        "holdout_cells_opened": False,
        "network_calls_performed": False,
    }
    summary = {**summary_content, "summary_sha256": sha256_json(summary_content)}
    native.mkdir(exist_ok=True)
    (native / "SUMMARY.json").write_text(
        canonical_json(summary) + "\n", encoding="utf-8"
    )
    return manifest, routes, experiment, native


def test_round1_feedback_request_is_complete_replayable_and_holdout_free(
    tmp_path: Path,
) -> None:
    manifest, routes, experiment, native = _fixture(tmp_path)
    output = tmp_path / "REQUEST.json"

    first = freeze_round1_feedback_request(
        taskset_path=manifest,
        routes_path=routes,
        experiment_root=experiment,
        native_root=native,
        output_path=output,
    )
    second = freeze_round1_feedback_request(
        taskset_path=manifest,
        routes_path=routes,
        experiment_root=experiment,
        native_root=native,
        output_path=output,
    )

    serialized = canonical_json(first)
    assert first == second
    assert first["feedback_task_count"] == 30
    assert first["feedback_cell_count"] == 60
    assert first["holdout_cells_included"] is False
    assert first["request"]["feedback_gain_count"] == 0
    assert len(first["request"]["failures"]) == 30
    assert "taught feedback raw 1" in serialized
    assert "eval-31" not in serialized


def test_round1_feedback_request_rejects_partial_or_opened_holdout(
    tmp_path: Path,
) -> None:
    manifest, routes, experiment, native = _fixture(tmp_path)
    summary_path = native / "SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    content = {key: value for key, value in summary.items() if key != "summary_sha256"}
    content["status"] = "partial"
    content["completed_cells"] = 59
    summary_path.write_text(
        canonical_json({**content, "summary_sha256": sha256_json(content)}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="complete feedback native"):
        freeze_round1_feedback_request(
            taskset_path=manifest,
            routes_path=routes,
            experiment_root=experiment,
            native_root=native,
            output_path=tmp_path / "REQUEST.partial.json",
        )


def test_post_holdout_feedback_projection_burns_and_excludes_prior_holdout(
    tmp_path: Path,
) -> None:
    manifest, routes, experiment, native = _fixture(tmp_path)
    (experiment / "cells/eval-31/operator-taught").mkdir(parents=True)

    with pytest.raises(
        ContractError, match="holdout experiment evidence is prohibited"
    ):
        freeze_round1_feedback_request(
            taskset_path=manifest,
            routes_path=routes,
            experiment_root=experiment,
            native_root=native,
            output_path=tmp_path / "REQUEST.rejected.json",
        )

    projected = freeze_round1_feedback_request(
        taskset_path=manifest,
        routes_path=routes,
        experiment_root=experiment,
        native_root=native,
        output_path=tmp_path / "REQUEST.projected.json",
        post_holdout_projection=True,
    )

    serialized = canonical_json(projected)
    assert projected["source_holdout_evidence_present"] is True
    assert projected["current_holdout_reuse_prohibited"] is True
    assert projected["holdout_cells_included"] is False
    assert "prior holdout is burned" in " ".join(projected["request"]["constraints"])
    assert "eval-31" not in serialized


def test_round1_realization_feedback_request_accepts_partial_structural_pairs(
    tmp_path: Path,
) -> None:
    manifest, routes, experiment, _native = _fixture(tmp_path)
    for number in range(2, 31):
        shutil.rmtree(experiment / "cells" / f"eval-{number}")
    for teaching in ("baseline", "taught"):
        cell = experiment / "cells/eval-1" / f"operator-{teaching}"
        prompt = f"{teaching} READ-ONLY repository evidence and editable selector"
        (cell / "generation-prompt-000.txt").write_text(prompt, encoding="utf-8")
        attempt_path = cell / "ATTEMPT.json"
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        content = {
            key: value for key, value in attempt.items() if key != "evidence_sha256"
        }
        raw = (cell / "raw-output.txt").read_text(encoding="utf-8")
        content["generation_trace"] = [
            {
                "kind": "fixture-candidate",
                "path": "raw-output.txt",
                "sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "prompt_path": "generation-prompt-000.txt",
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        ]
        _freeze(attempt_path, content)

    request = freeze_round1_realization_feedback_request(
        taskset_path=manifest,
        routes_path=routes,
        experiment_root=experiment,
        output_path=tmp_path / "REALIZATION-REQUEST.json",
    )

    assert request["feedback_task_count"] == 1
    assert request["feedback_cell_count"] == 2
    assert request["native_receipts_included"] is False
    assert request["request"]["request_type"] == "round1-realization-feedback-v1"
    assert request["request"]["failures"][0]["task_id"] == "eval-1"
    trace = request["request"]["failures"][0]["taught"]["generation_candidates"][0]
    assert "READ-ONLY repository evidence" in trace["generation_prompt"]
    assert (
        trace["generation_prompt_sha256"]
        == hashlib.sha256(trace["generation_prompt"].encode()).hexdigest()
    )
    assert "eval-31" not in canonical_json(request)


def test_targeted_native_feedback_request_carries_prior_native_iterations(
    tmp_path: Path,
) -> None:
    manifest, routes, experiment, native = _fixture(tmp_path)
    shutil.copyfile(native / "SUMMARY.json", native / "PROGRESS.json")
    prior_path = tmp_path / "PRIOR-REQUEST.json"
    prior = freeze_round1_targeted_native_feedback_request(
        taskset_path=manifest,
        routes_path=routes,
        experiment_root=experiment,
        native_root=native,
        task_ids=("eval-1",),
        output_path=prior_path,
    )

    current = freeze_round1_targeted_native_feedback_request(
        taskset_path=manifest,
        routes_path=routes,
        experiment_root=experiment,
        native_root=native,
        task_ids=("eval-1",),
        prior_request_paths=(prior_path,),
        output_path=tmp_path / "CURRENT-REQUEST.json",
    )

    assert current["feedback_iteration_count"] == 2
    history = current["request"]["prior_feedback_iterations"]
    assert len(history) == 1
    assert history[0]["request_sha256"] == prior["request_sha256"]
    assert history[0]["failures"][0]["task_id"] == "eval-1"
    assert "do not recommend an action already disproved" in " ".join(
        current["request"]["constraints"]
    )


def test_targeted_native_feedback_request_carries_disproven_mechanism_capabilities(
    tmp_path: Path,
) -> None:
    manifest, routes, experiment, native = _fixture(tmp_path)
    shutil.copyfile(native / "SUMMARY.json", native / "PROGRESS.json")

    request = freeze_round1_targeted_native_feedback_request(
        taskset_path=manifest,
        routes_path=routes,
        experiment_root=experiment,
        native_root=native,
        task_ids=("eval-1",),
        output_path=tmp_path / "REQUEST.json",
        disproven_mechanism_capabilities=(
            {
                "capability_id": "issue-verbatim-renderer-owned-action-v1",
                "implementation_sha256": "a" * 64,
                "test_sha256": "b" * 64,
                "native_conclusion": "full feedback run did not establish transfer",
            },
        ),
    )

    capabilities = request["request"]["disproven_mechanism_capabilities"]
    assert capabilities[0]["capability_id"] == (
        "issue-verbatim-renderer-owned-action-v1"
    )
    assert "must not repeat" in " ".join(request["request"]["constraints"])

    manifest2, routes2, experiment2, native2 = _fixture(tmp_path / "third")
    (native2 / "cells/eval-31/operator-taught").mkdir(parents=True)
    with pytest.raises(ContractError, match="holdout native evidence is prohibited"):
        freeze_round1_feedback_request(
            taskset_path=manifest2,
            routes_path=routes2,
            experiment_root=experiment2,
            native_root=native2,
            output_path=tmp_path / "REQUEST.holdout.json",
        )


def test_round1_feedback_strategy_dispatch_uses_campaign_checkpoint_and_replays(
    tmp_path: Path,
) -> None:
    manifest, routes, experiment, native = _fixture(tmp_path)
    request_path = tmp_path / "REQUEST.json"
    request = freeze_round1_feedback_request(
        taskset_path=manifest,
        routes_path=routes,
        experiment_root=experiment,
        native_root=native,
        output_path=request_path,
    )
    checkpoint_content = {
        "schema_version": 1,
        "call_id": "round1-localizer-review-001",
        "request_sha256": "e" * 64,
        "campaign_tokens_after": 162_540,
        "campaign_total_token_limit": 3_000_000,
        "candidate_status": "advisory_inactive",
        "auto_apply": False,
        "network_calls_performed": True,
    }
    checkpoint = tmp_path / "CAMPAIGN.json"
    _freeze(checkpoint, checkpoint_content)
    authorization = tmp_path / "AUTHORIZATION.json"
    auth = create_round1_feedback_authorization(
        request_path=request_path,
        campaign_checkpoint_path=checkpoint,
        output_path=authorization,
        expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        maximum_output_tokens=10_000,
    )
    client = _Client()

    first = dispatch_round1_feedback_strategy(
        request_path=request_path,
        authorization_path=authorization,
        campaign_checkpoint_path=checkpoint,
        ledger_path=tmp_path / "LEDGER.json",
        output_path=tmp_path / "RESPONSE.json",
        client=client,
    )
    second = dispatch_round1_feedback_strategy(
        request_path=request_path,
        authorization_path=authorization,
        campaign_checkpoint_path=checkpoint,
        ledger_path=tmp_path / "LEDGER.json",
        output_path=tmp_path / "RESPONSE.json",
        client=client,
    )

    assert first == second
    assert client.calls == 1
    assert (
        auth["single_call_authorization"]["request_sha256"] == request["request_sha256"]
    )
    assert first["event_type"] == "parent-strategy-response"
    assert first["tokens_charged"] == 1000
    assert first["campaign_tokens_before"] == 162_540
    assert first["campaign_tokens_after"] == 163_540
    assert first["candidate_status"] == "advisory_inactive"
    assert first["auto_apply"] is False


def test_targeted_native_feedback_request_binds_only_explicit_feedback_pair(
    tmp_path: Path,
) -> None:
    manifest, routes, experiment, native = _fixture(tmp_path)
    shutil.copyfile(native / "SUMMARY.json", native / "PROGRESS.json")

    request = freeze_round1_targeted_native_feedback_request(
        taskset_path=manifest,
        routes_path=routes,
        experiment_root=experiment,
        native_root=native,
        task_ids=("eval-1",),
        output_path=tmp_path / "TARGETED-REQUEST.json",
    )

    assert request["feedback_task_count"] == 1
    assert request["feedback_cell_count"] == 2
    assert request["feedback_iteration_count"] == 1
    assert request["native_receipts_included"] is True
    assert request["holdout_cells_included"] is False
    assert request["request"]["request_type"] == ("round1-targeted-native-feedback-v1")
    assert request["request"]["failures"][0]["task_id"] == "eval-1"
    assert "native_outcome" in request["request"]["failures"][0]["taught"]
    assert request["request"]["failures"][0]["taught"]["generation_candidates"] == []
    assert "source-derived catalog extension" in " ".join(
        request["request"]["constraints"]
    )
    assert "eval-2" not in canonical_json(request)
    assert "eval-31" not in canonical_json(request)


def test_round1_feedback_compiler_freezes_inactive_r010_skills(tmp_path: Path) -> None:
    response_content = {
        "schema_version": 1,
        "event_type": "parent-strategy-response",
        "strategy": {
            "failure_diagnosis": "The action classifier is missing.",
            "mechanism_revision": "Classify before selecting an operator.",
            "operator_skill_requirements": [
                "Copy one exact AST selector from the chosen symbol.",
                "Use replace_expression only for one expression; otherwise use replace_statement.",
                "Preserve parsed state with a conditional default guard.",
            ],
            "span_skill_requirements": [
                "Use one unique exact before span per file.",
                "Reject duplicate files and no-op replacements.",
            ],
            "verification_changes": ["Classify failures before native evaluation."],
            "next_experiment": ["Rerun the same feedback A/B."],
        },
        "candidate_status": "advisory_inactive",
        "auto_apply": False,
        "holdout_cells_included": False,
        "network_calls_performed": True,
    }
    response = tmp_path / "RESPONSE.json"
    _freeze(response, response_content)
    parent = Path(
        "artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/"
        "p1-r6-operator-pattern-skill-v2/OPERATOR-PATTERN-REVISION.round-006.json"
    )

    report = compile_round1_feedback_skills(
        strategy_response_path=response,
        parent_operator_skill_path=parent,
        operator_output_path=tmp_path / "OPERATOR-SKILL.json",
        span_output_path=tmp_path / "SPAN-SKILL.json",
        source_round=10,
    )

    assert report["candidate_status"] == "inactive"
    assert report["auto_activate"] is False
    assert report["source_round"] == 10
    operator = json.loads((tmp_path / "OPERATOR-SKILL.json").read_text())
    span = json.loads((tmp_path / "SPAN-SKILL.json").read_text())
    assert operator["next_revision"]["revision_id"].endswith("r010")
    assert operator["next_revision"]["parent_revision_id"].endswith("r006")
    assert "active: false" in operator["next_revision"]["skill_text"]
    assert span["candidate_status"] == "inactive"
    assert span["new_domain_knowledge_added"] is False
    assert (
        "Preserve parsed state with a conditional default guard." in span["skill_text"]
    )


def test_round1_feedback_compiler_normalizes_unknown_operator_names(
    tmp_path: Path,
) -> None:
    response_content = {
        "schema_version": 1,
        "event_type": "parent-strategy-response",
        "strategy": {
            "failure_diagnosis": "The teacher mixed valid and invented operators.",
            "mechanism_revision": "Constrain the action space before compilation.",
            "operator_skill_requirements": [
                "Require operator names from replace_condition, insert_after, delete.",
                "Require a truth table before condition edits.",
            ],
            "span_skill_requirements": [
                "Return target_files, target_symbol, predicate_path, edit_semantics, "
                "source_span_confidence; never include plans.",
                "Trace the root predicate before choosing a span.",
            ],
            "verification_changes": ["Reject invented operators."],
            "next_experiment": ["Rerun the feedback pair."],
        },
        "candidate_status": "advisory_inactive",
        "auto_apply": False,
        "holdout_cells_included": False,
        "network_calls_performed": True,
    }
    response = tmp_path / "RESPONSE.json"
    _freeze(response, response_content)
    parent = Path(
        "artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/"
        "p1-r6-operator-pattern-skill-v2/OPERATOR-PATTERN-REVISION.round-006.json"
    )

    compile_round1_feedback_skills(
        strategy_response_path=response,
        parent_operator_skill_path=parent,
        operator_output_path=tmp_path / "OPERATOR-SKILL.json",
        span_output_path=tmp_path / "SPAN-SKILL.json",
        source_round=24,
    )

    operator = json.loads((tmp_path / "OPERATOR-SKILL.json").read_text())
    span = json.loads((tmp_path / "SPAN-SKILL.json").read_text())
    skill = operator["next_revision"]["skill_text"]
    assert "insert_after" not in skill
    assert "delete" not in skill
    assert "framework-supplied operator catalog" in skill
    assert operator["compiler_normalizations"] == [
        {
            "reason": "unsupported-operator-name",
            "unsupported": ["delete", "insert_after"],
        }
    ]
    assert "source_span_confidence" not in span["skill_text"]
    assert "never include plans" not in span["skill_text"]
    assert "target_files" not in span["skill_text"]
    assert "predicate_path" not in span["skill_text"]
    assert "framework-supplied span bundle schema" in span["skill_text"]
    assert span["compiler_normalizations"] == [
        {
            "reason": "unsupported-span-schema",
            "markers": [
                "confidence",
                "edit_semantics",
                "never include plans",
                "predicate_path",
                "source_span_confidence",
                "target_files",
                "target_symbol",
            ],
        }
    ]
