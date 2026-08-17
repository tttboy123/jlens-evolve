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


def test_teacher_leakage_gate_allows_safety_prose_but_rejects_protected_fields(
    tmp_path: Path,
) -> None:
    calls = 0

    def transport(_request: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
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
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

    proposer = CandidateProposer(
        root=tmp_path / "teacher",
        provider="deepseek",
        model="deepseek-v4-flash",
        transport=transport,
        pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        hard_budget_cny=10.0,
    )
    proposer.propose(
        request_id="safe-prose",
        failure_package={
            "goal": "Use feedback only without opening holdout data.",
            "failures": ["unresolved"],
        },
        max_output_tokens=1000,
    )
    with pytest.raises(ContractViolation, match="prohibited leakage"):
        proposer.propose(
            request_id="protected-field",
            failure_package={"reference_patch": "secret"},
            max_output_tokens=1000,
        )
    assert calls == 1


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


def test_teacher_proposer_freezes_and_charges_invalid_provider_response(
    tmp_path: Path,
) -> None:
    calls = 0

    def invalid_candidate(_request: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "protocol": "missing-required-fields",
                            }
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        }

    root = tmp_path / "teacher"
    proposer = CandidateProposer(
        root=root,
        provider="deepseek",
        model="deepseek-v4-flash",
        transport=invalid_candidate,
        pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        hard_budget_cny=10.0,
    )

    with pytest.raises(ContractViolation, match="candidate fields"):
        proposer.propose(
            request_id="invalid-candidate",
            failure_package={"failures": ["unresolved"]},
            max_output_tokens=1000,
        )

    raw_response = root / "invalid-candidate" / "TEACHER-RAW-RESPONSE.json"
    assert raw_response.is_file()
    assert not (root / "invalid-candidate" / "TEACHER-RESPONSE.json").exists()
    assert proposer.cost_ledger.snapshot().spent_cost_cny == 0.006
    assert proposer.cost_ledger.snapshot().reserved_cost_cny == 0

    restarted = CandidateProposer(
        root=root,
        provider="deepseek",
        model="deepseek-v4-flash",
        transport=lambda _request: pytest.fail("must replay raw response"),
        pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        hard_budget_cny=10.0,
    )
    with pytest.raises(ContractViolation, match="candidate fields"):
        restarted.propose(
            request_id="invalid-candidate",
            failure_package={"failures": ["unresolved"]},
            max_output_tokens=1000,
        )
    assert calls == 1
    assert restarted.cost_ledger.snapshot().spent_cost_cny == 0.006


def test_candidate_routes_normalize_teacher_derived_keys():

    from evolve.proposals.candidate_chain import _candidate_routes

    required = ("django__django-11149", "laravel__framework-52684")
    router = {
        "routes": {
            "feedback-django__django-11149@e245046bb6e8": "op-v1",
            "laravel__framework-52684": "op-v1",
        }
    }
    routes = _candidate_routes(router, "op-v1", required_task_ids=required)
    assert set(task_id for task_id, _ in routes) == set(required)


def test_candidate_routes_fail_closed_on_ambiguous_key():
    import pytest

    from evolve.contracts import ContractViolation
    from evolve.proposals.candidate_chain import _candidate_routes

    # Key contains two required ids -> ambiguous -> fail closed.
    required = ("django__django-11149", "django__django-11551")
    router = {
        "routes": {"feedback-django__django-11149_x_django__django-11551@abc": "op-v1"}
    }
    with pytest.raises(ContractViolation):
        _candidate_routes(router, "op-v1", required_task_ids=required)


def test_candidate_routes_strict_without_required_ids():
    import pytest

    from evolve.contracts import ContractViolation
    from evolve.proposals.candidate_chain import _candidate_routes

    # Without required ids the strict identifier rule applies (legacy path).
    router = {"routes": {"feedback-x@y": "op-v1"}}
    with pytest.raises(ContractViolation):
        _candidate_routes(router, "op-v1")


def test_validate_change_set_source_round1_prefixed_router_schema_v2() -> None:
    """Regression: teacher Router keys with round1- prefixes must normalize in
    the integrity re-derivation too, else synthesized projection mismatches."""
    from evolve.contracts import Cohort
    from evolve.proposals.candidate_chain import (
        CandidateChangeSet,
        CompileSpec,
        _validate_change_set_source,
        content_sha256,
    )

    required = (
        "django__django-15277",
        "phpoffice__phpspreadsheet-3463",
        "django__django-15315",
    )
    operator_id = "debug-fix-test-operator"
    teacher = {
        "candidate_schema_version": 2,
        "candidate": {
            "protocol": "execution-protocol-v1",
            "prompt_template": "You are an expert software engineer. {instruction}",
            "skill_text": "Execute the debug-fix-test protocol.",
            "eval_note": "Native evaluation procedure.",
            "operator": {
                "id": operator_id,
                "kind": "zero-arg",
                "arguments": [],
                "instruction": "Execute the debug-fix-test protocol.",
            },
            "router": {
                "routes": {
                    "round1-django__django-15277": operator_id,
                    "round1-django__django-15315": operator_id,
                    "round1-phpoffice__phpspreadsheet-3463": operator_id,
                }
            },
            "memory_policy": {"type": "none"},
            "preconditions": ["repo checked out"],
            "expected_external_effect": "patch resolves the issue",
            "expected_internal_effect": "model produces a correct patch",
            "falsification": "patch fails",
        },
    }
    compile_spec = CompileSpec(
        candidate_id="candidate-abc",
        revision_id="candidate-abc-r0002",
        parent_revision_id="empty-harness-v1",
        cohort=Cohort.FEEDBACK,
        operator_id=operator_id,
        operator_instruction="Execute the debug-fix-test protocol.",
        required_route_task_ids=required,
        routes=tuple((task_id, operator_id) for task_id in required),
    )
    change_set = CandidateChangeSet(
        candidate_id=compile_spec.candidate_id,
        revision_id=compile_spec.revision_id,
        parent_revision_id=compile_spec.parent_revision_id,
        source_candidate_sha256=content_sha256(teacher["candidate"]),
        compile_spec_sha256=compile_spec.content_sha256,
        protocol=teacher["candidate"]["protocol"],
        prompt_template=teacher["candidate"]["prompt_template"],
        skill_text=teacher["candidate"]["skill_text"],
        eval_note=teacher["candidate"]["eval_note"],
        operator_id=operator_id,
        operator_instruction=teacher["candidate"]["operator"]["instruction"],
        # Order follows the teacher Router (round1- prefixed -> normalized).
        routes=(
            ("django__django-15277", operator_id),
            ("django__django-15315", operator_id),
            ("phpoffice__phpspreadsheet-3463", operator_id),
        ),
        memory_policy={"type": "none"},
        preconditions=("repo checked out",),
        expected_external_effect="patch resolves the issue",
        expected_internal_effect="model produces a correct patch",
        falsification="patch fails",
        synthesized_task_ids=(),
    )
    # Must not raise after the round1- prefix normalization fix.
    _validate_change_set_source(change_set, teacher, compile_spec)
