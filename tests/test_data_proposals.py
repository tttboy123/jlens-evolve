from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evolve.contracts import Cohort, ContractViolation
from evolve.data import TaskRegistry
from evolve.fresh_feedback import _freeze_release_candidate_artifacts
from evolve.proposals import (
    CandidateCompiler,
    CandidateProposer,
    CompileSpec,
    PricingCnyPerMillionTokens,
)


def test_task_registry_freezes_feedback_tasks_and_deduplicates_by_content(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"problem":"fix it"}\n', encoding="utf-8")
    registry = TaskRegistry(tmp_path / "registry")

    first = registry.import_task(
        task_id="sphinx-7757",
        revision_id="task-r1",
        project="sphinx",
        cohort=Cohort.FEEDBACK,
        source=source,
        evaluator_id="swebench@sha256:" + "a" * 64,
    )
    second = registry.import_task(
        task_id="sphinx-7757",
        revision_id="task-r1",
        project="sphinx",
        cohort=Cohort.FEEDBACK,
        source=source,
        evaluator_id="swebench@sha256:" + "a" * 64,
    )

    assert first == second
    assert first.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(registry.all()) == 1


@pytest.mark.parametrize("cohort", [Cohort.HOLDOUT, Cohort.BURNED, Cohort.FINAL_SEALED])
def test_task_registry_denies_non_feedback_without_independent_authorization(
    tmp_path: Path, cohort: Cohort
) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(ContractViolation, match="feedback"):
        TaskRegistry(tmp_path / "registry").import_task(
            task_id="sealed",
            revision_id="r1",
            project="project",
            cohort=cohort,
            source=source,
            evaluator_id="native@sha256:" + "a" * 64,
        )


def test_teacher_proposer_freezes_request_response_usage_and_inactive_candidate(
    tmp_path: Path,
) -> None:
    calls = 0

    def transport(request: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        assert "gold" not in json.dumps(request).casefold()
        return {
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "protocol": "typed-operator-plan-v1",
                                "prompt_template": "Return one operator plan.",
                                "skill_text": "Preserve unrelated behavior.",
                                "eval_note": "Validate with native matched A/B.",
                            }
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        }

    proposer = CandidateProposer(
        root=tmp_path / "teacher",
        provider="deepseek",
        model="deepseek-v4-flash",
        transport=transport,
        pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        hard_budget_cny=10.0,
    )
    result = proposer.propose(
        request_id="request-1",
        failure_package={"task_ids": ["feedback-a"], "failures": ["unresolved"]},
        max_output_tokens=1000,
    )
    replay = proposer.propose(
        request_id="request-1",
        failure_package={"task_ids": ["feedback-a"], "failures": ["unresolved"]},
        max_output_tokens=1000,
    )

    assert calls == 1
    assert replay == result
    assert result.candidate.active is False
    assert result.usage.estimated_cost_cny == 0.006
    assert result.request_path.is_file() and result.response_path.is_file()
    assert proposer.cost_ledger.snapshot().spent_cost_cny == 0.006
    restarted = CandidateProposer(
        root=tmp_path / "teacher",
        provider="deepseek",
        model="deepseek-v4-flash",
        transport=lambda _request: pytest.fail("replay must not dispatch"),
        pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        hard_budget_cny=10.0,
    )
    assert (
        restarted.propose(
            request_id="request-1",
            failure_package={
                "task_ids": ["feedback-a"],
                "failures": ["unresolved"],
            },
            max_output_tokens=1000,
        )
        == result
    )
    assert restarted.cost_ledger.snapshot().spent_cost_cny == 0.006
    assert len(restarted.cost_ledger.events()) == 3

    compiled = CandidateCompiler().compile(
        request_path=result.request_path,
        response_path=result.response_path,
        compile_spec=CompileSpec(
            candidate_id=result.candidate.candidate_id,
            revision_id="candidate-r1",
            parent_revision_id="baseline-r1",
            cohort=Cohort.FEEDBACK,
            operator_id="operator-r1",
            operator_instruction="Apply the compiled teaching.",
            routes=(("feedback-a", "operator-r1"),),
        ),
        output_root=tmp_path / "compiled",
    )
    assert compiled.skill.skill_text == result.candidate.skill_text
    assert compiled.change_set.source_candidate_sha256
    release_root = tmp_path / "release"
    release_root.mkdir()
    _freeze_release_candidate_artifacts(compiled, release_root)
    assert (release_root / "TEACHER-REQUEST.json").read_bytes() == (
        result.request_path.read_bytes()
    )
    assert (release_root / "COMPILED-REVISION.json").read_bytes() == (
        compiled.manifest_path.read_bytes()
    )
    _freeze_release_candidate_artifacts(compiled, release_root)


def test_teacher_proposer_stops_before_dispatch_when_reservation_can_exceed_budget(
    tmp_path: Path,
) -> None:
    proposer = CandidateProposer(
        root=tmp_path,
        provider="deepseek",
        model="deepseek-v4-flash",
        transport=lambda _request: pytest.fail("must not dispatch"),
        pricing=PricingCnyPerMillionTokens(input=100.0, output=100.0),
        hard_budget_cny=0.01,
    )

    with pytest.raises(ContractViolation, match="budget"):
        proposer.propose(
            request_id="too-expensive",
            failure_package={"failures": ["x"]},
            max_output_tokens=1000,
        )


def test_teacher_proposer_never_redispatches_unreconciled_reservation(
    tmp_path: Path,
) -> None:
    calls = 0

    def failing_transport(_request: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise RuntimeError("connection outcome unknown")

    root = tmp_path / "teacher"
    first = CandidateProposer(
        root=root,
        provider="deepseek",
        model="deepseek-v4-flash",
        transport=failing_transport,
        pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        hard_budget_cny=10.0,
    )
    with pytest.raises(RuntimeError, match="unknown"):
        first.propose(
            request_id="request-crash",
            failure_package={"failures": ["unresolved"]},
            max_output_tokens=1000,
        )

    restarted = CandidateProposer(
        root=root,
        provider="deepseek",
        model="deepseek-v4-flash",
        transport=lambda _request: pytest.fail("must not redispatch"),
        pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        hard_budget_cny=10.0,
    )
    with pytest.raises(ContractViolation, match="manual reconcile"):
        restarted.propose(
            request_id="request-crash",
            failure_package={"failures": ["unresolved"]},
            max_output_tokens=1000,
        )
    assert calls == 1
