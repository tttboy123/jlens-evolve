from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from evolve.contracts import Cohort, ContractViolation, canonical_json
from evolve.proposals import (
    CandidateCompiler,
    CandidateProposer,
    CompiledRevision,
    CompileSpec,
    PricingCnyPerMillionTokens,
)
from evolve.teachers import (
    DeepSeekCompatibleTeacherTransport,
    FrozenReplayTeacherTransport,
    OpenAICompatibleTeacherTransport,
    TeacherTransport,
    build_teacher_transport,
)


def _candidate(*, memory_policy: object = None) -> dict[str, object]:
    return {
        "protocol": "autonomous-skill-harness-v2",
        "prompt_template": "Inspect, plan, execute, verify.",
        "skill_text": "Preserve unrelated behavior and falsify the change.",
        "operator": {
            "id": "repair-source",
            "kind": "zero-arg",
            "arguments": [],
            "instruction": "Repair only the demonstrated failure mode.",
        },
        "router": {
            "routes": [
                {"task_id": "feedback-a", "operator_id": "repair-source"}
            ]
        },
        "memory_policy": memory_policy,
        "preconditions": ["feedback cohort", "clean evaluator boundary"],
        "expected_external_effect": "The task passes its native evaluator.",
        "expected_internal_effect": "The harness selects repair-source once.",
        "falsification": "Reject if an unrelated test changes behavior.",
        "eval_note": "Run matched baseline/taught native evaluation.",
    }


def _raw_response(candidate: dict[str, object]) -> dict[str, object]:
    return {
        "model": "deepseek-chat",
        "choices": [{"message": {"content": json.dumps(candidate)}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 200},
    }


def _propose(tmp_path: Path, candidate: dict[str, object]):
    calls: list[dict[str, object]] = []

    def transport(request: dict[str, object]) -> dict[str, object]:
        calls.append(request)
        return _raw_response(candidate)

    proposer = CandidateProposer(
        root=tmp_path / "teacher",
        provider="deepseek",
        model="deepseek-chat",
        transport=transport,
        pricing=PricingCnyPerMillionTokens(input=1.0, output=2.0),
        hard_budget_cny=10.0,
    )
    first = proposer.propose(
        request_id="round-1",
        failure_package={"failures": ["feedback-a failed"]},
        max_output_tokens=500,
    )
    second = proposer.propose(
        request_id="round-1",
        failure_package={"failures": ["feedback-a failed"]},
        max_output_tokens=500,
    )
    return first, second, calls


def test_extended_candidate_is_frozen_inactive_and_replay_does_not_dispatch(
    tmp_path: Path,
) -> None:
    full = _candidate(memory_policy={"mode": "ephemeral", "max_entries": 8})

    first, replay, calls = _propose(tmp_path, full)

    assert len(calls) == 1
    assert replay == first
    assert first.candidate.active is False
    assert first.candidate.operator == full["operator"]
    assert first.candidate.router == full["router"]
    assert first.candidate.memory_policy == full["memory_policy"]
    assert first.candidate.falsification == full["falsification"]
    receipt = json.loads(first.response_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 2
    assert receipt["candidate_status"] == "inactive"
    assert receipt["auto_activate"] is False


def test_legacy_v1_receipt_replays_without_dispatch(tmp_path: Path) -> None:
    legacy_candidate = {
        "protocol": "typed-operator-plan-v1",
        "prompt_template": "Return one operator plan.",
        "skill_text": "Preserve unrelated behavior.",
        "eval_note": "Validate with native matched A/B.",
    }
    result, _, calls = _propose(tmp_path, legacy_candidate)
    receipt = json.loads(result.response_path.read_text(encoding="utf-8"))
    receipt.pop("candidate_sha256")
    receipt.pop("receipt_sha256")
    receipt["candidate"] = {
        "candidate_id": result.candidate.candidate_id,
        **legacy_candidate,
        "active": False,
    }
    receipt["usage"] = {
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
        "estimated_cost_cny": result.usage.estimated_cost_cny,
    }
    result.response_path.write_text(
        canonical_json(receipt) + "\n", encoding="utf-8"
    )
    replay = CandidateProposer(
        root=tmp_path / "teacher",
        provider="deepseek",
        model="deepseek-chat",
        transport=lambda _request: pytest.fail("v1 replay must not dispatch"),
        pricing=PricingCnyPerMillionTokens(input=1.0, output=2.0),
        hard_budget_cny=10.0,
    ).propose(
        request_id="round-1",
        failure_package={"failures": ["feedback-a failed"]},
        max_output_tokens=500,
    )

    assert len(calls) == 1
    assert replay.candidate == result.candidate
    assert replay.usage == result.usage


def test_full_teacher_candidate_compiles_its_harness_and_binds_parent_lineage(
    tmp_path: Path,
) -> None:
    result, _, _ = _propose(
        tmp_path,
        _candidate(memory_policy={"mode": "ephemeral", "max_entries": 8}),
    )
    spec = CompileSpec(
        candidate_id=result.candidate.candidate_id,
        revision_id="candidate-r2",
        parent_revision_id="candidate-r1",
        cohort=Cohort.FEEDBACK,
    )

    compiled = CandidateCompiler().compile(
        request_path=result.request_path,
        response_path=result.response_path,
        compile_spec=spec,
        output_root=tmp_path / "compiled",
    )

    assert compiled.change_set.active is False
    assert compiled.operator.operator_id == "repair-source"
    assert compiled.operator.instruction.startswith("Repair only")
    assert compiled.router.routes == (("feedback-a", "repair-source"),)
    assert compiled.memory_policy is not None
    assert compiled.memory_policy.policy == {"max_entries": 8, "mode": "ephemeral"}
    manifest = json.loads(compiled.manifest_path.read_text(encoding="utf-8"))
    assert manifest["parent_revision_id"] == "candidate-r1"
    assert manifest["lineage"]["parent_revision_id"] == "candidate-r1"
    assert manifest["lineage"]["source_candidate_sha256"] == (
        compiled.change_set.source_candidate_sha256
    )
    assert manifest["lineage_sha256"] == hashlib.sha256(
        canonical_json(manifest["lineage"]).encode("utf-8")
    ).hexdigest()
    assert CompiledRevision.load(compiled.root) == compiled


def test_required_route_coverage_is_repair_and_survives_load_round_trip(
    tmp_path: Path,
) -> None:
    """A schema-valid under-covered Router is deterministically repaired.

    The repair is a pure function of the parsed response and the CompileSpec
    required task list, so the sealed bundle reloads to an identical revision
    without mutating the paid Teacher response.
    """

    result, _, _ = _propose(tmp_path, _candidate())
    spec = CompileSpec(
        candidate_id=result.candidate.candidate_id,
        revision_id="candidate-r2",
        parent_revision_id="candidate-r1",
        cohort=Cohort.FEEDBACK,
        required_route_task_ids=("feedback-a", "feedback-b", "feedback-c"),
    )
    compiled = CandidateCompiler().compile(
        request_path=result.request_path,
        response_path=result.response_path,
        compile_spec=spec,
        output_root=tmp_path / "compiled",
    )
    assert compiled.router.routes == (
        ("feedback-a", "repair-source"),
        ("feedback-b", "repair-source"),
        ("feedback-c", "repair-source"),
    )
    assert dict(compiled.change_set.routes) == {
        "feedback-a": "repair-source",
        "feedback-b": "repair-source",
        "feedback-c": "repair-source",
    }
    compile_spec_payload = json.loads(
        (compiled.root / "COMPILE-SPEC.json").read_text(encoding="utf-8")
    )
    assert compile_spec_payload["required_route_task_ids"] == [
        "feedback-a",
        "feedback-b",
        "feedback-c",
    ]
    manifest = json.loads(compiled.manifest_path.read_text(encoding="utf-8"))
    assert manifest["compile_spec_sha256"] == spec.content_sha256
    assert CompiledRevision.load(compiled.root) == compiled
    response_payload = json.loads(
        (compiled.root / "TEACHER-RESPONSE.json").read_text(encoding="utf-8")
    )
    assert response_payload["candidate"]["router"]["routes"] == [
        {"task_id": "feedback-a", "operator_id": "repair-source"}
    ]
    assert response_payload["receipt_sha256"]


def test_compiled_repair_is_tamper_evident_in_router_and_spec(
    tmp_path: Path,
) -> None:
    result, _, _ = _propose(tmp_path, _candidate())
    spec = CompileSpec(
        candidate_id=result.candidate.candidate_id,
        revision_id="candidate-r2",
        parent_revision_id="candidate-r1",
        cohort=Cohort.FEEDBACK,
        required_route_task_ids=("feedback-a", "feedback-b"),
    )
    compiled = CandidateCompiler().compile(
        request_path=result.request_path,
        response_path=result.response_path,
        compile_spec=spec,
        output_root=tmp_path / "compiled",
    )
    router_path = compiled.root / "COMPILED-ROUTER.json"
    router = json.loads(router_path.read_text(encoding="utf-8"))
    router["routes"] = [
        {"task_id": "feedback-a", "operator_id": "repair-source"}
    ]
    router_path.write_text(canonical_json(router) + "\n", encoding="utf-8")
    with pytest.raises(ContractViolation, match="hash mismatch"):
        CompiledRevision.load(compiled.root)

    spec_path = compiled.root / "COMPILE-SPEC.json"
    spec_payload = json.loads(spec_path.read_text(encoding="utf-8"))
    spec_payload["required_route_task_ids"] = ["feedback-a", "feedback-b", "forged"]
    spec_path.write_text(
        canonical_json(spec_payload) + "\n", encoding="utf-8"
    )
    with pytest.raises(ContractViolation, match="hash mismatch: COMPILE-SPEC.json"):
        CompiledRevision.load(compiled.root)


def test_compile_spec_required_route_task_ids_are_validated(tmp_path: Path) -> None:
    result, _, _ = _propose(tmp_path, _candidate())
    with pytest.raises(ContractViolation, match="unique"):
        CompileSpec(
            candidate_id=result.candidate.candidate_id,
            revision_id="candidate-r2",
            parent_revision_id="candidate-r1",
            cohort=Cohort.FEEDBACK,
            required_route_task_ids=("feedback-a", "feedback-a"),
        )
    with pytest.raises(ContractViolation, match="not a safe immutable identifier"):
        CompileSpec(
            candidate_id=result.candidate.candidate_id,
            revision_id="candidate-r2",
            parent_revision_id="candidate-r1",
            cohort=Cohort.FEEDBACK,
            required_route_task_ids=("not valid! id",),
        )


def test_compile_spec_omits_empty_required_routes_keeping_legacy_hash(
    tmp_path: Path,
) -> None:
    """Empty required routes stay out of the legacy content hash.

    Previously sealed bundles compiled without the repair requirement must keep
    their exact CompileSpec content hash.
    """

    result, _, _ = _propose(tmp_path, _candidate())
    legacy = CompileSpec(
        candidate_id=result.candidate.candidate_id,
        revision_id="candidate-r2",
        parent_revision_id="candidate-r1",
        cohort=Cohort.FEEDBACK,
    )
    explicit_empty = CompileSpec(
        candidate_id=result.candidate.candidate_id,
        revision_id="candidate-r2",
        parent_revision_id="candidate-r1",
        cohort=Cohort.FEEDBACK,
        required_route_task_ids=(),
    )
    assert legacy.content_sha256 == explicit_empty.content_sha256
    compiled = CandidateCompiler().compile(
        request_path=result.request_path,
        response_path=result.response_path,
        compile_spec=legacy,
        output_root=tmp_path / "compiled-legacy",
    )
    spec_payload = json.loads(
        (compiled.root / "COMPILE-SPEC.json").read_text(encoding="utf-8")
    )
    assert "required_route_task_ids" not in spec_payload
    assert CompiledRevision.load(compiled.root) == compiled


def test_compiled_lineage_and_memory_policy_are_tamper_evident(tmp_path: Path) -> None:
    result, _, _ = _propose(
        tmp_path,
        _candidate(memory_policy={"mode": "ephemeral", "max_entries": 8}),
    )
    compiled = CandidateCompiler().compile(
        request_path=result.request_path,
        response_path=result.response_path,
        compile_spec=CompileSpec(
            candidate_id=result.candidate.candidate_id,
            revision_id="candidate-r2",
            parent_revision_id="candidate-r1",
            cohort=Cohort.FEEDBACK,
        ),
        output_root=tmp_path / "compiled",
    )
    memory_path = compiled.root / "COMPILED-MEMORY-POLICY.json"
    memory_path.write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(ContractViolation, match="hash mismatch"):
        CompiledRevision.load(compiled.root)


def test_parent_lineage_tamper_is_rejected_even_when_manifest_is_valid_json(
    tmp_path: Path,
) -> None:
    result, _, _ = _propose(tmp_path, _candidate())
    compiled = CandidateCompiler().compile(
        request_path=result.request_path,
        response_path=result.response_path,
        compile_spec=CompileSpec(
            candidate_id=result.candidate.candidate_id,
            revision_id="candidate-r2",
            parent_revision_id="candidate-r1",
            cohort=Cohort.FEEDBACK,
        ),
        output_root=tmp_path / "compiled",
    )
    manifest = json.loads(compiled.manifest_path.read_text(encoding="utf-8"))
    manifest["parent_revision_id"] = "forged-parent"
    compiled.manifest_path.write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )

    with pytest.raises(ContractViolation, match="lineage"):
        CompiledRevision.load(compiled.root)


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_teacher_transports_share_protocol_and_deepseek_contract(tmp_path: Path) -> None:
    captured: list[tuple[Any, float]] = []
    raw = _raw_response(_candidate())

    def opener(request: Any, timeout: float) -> _Response:
        captured.append((request, timeout))
        return _Response(raw)

    generic = OpenAICompatibleTeacherTransport(
        endpoint="https://teacher.invalid/v1/chat/completions",
        model="teacher-model",
        api_key="secret",
        opener=opener,
    )
    deepseek = DeepSeekCompatibleTeacherTransport(
        endpoint="https://teacher.invalid/chat/completions",
        model="deepseek-chat",
        api_key="secret",
        opener=opener,
    )
    assert isinstance(generic, TeacherTransport)
    assert isinstance(deepseek, TeacherTransport)
    assert generic({"request_id": "generic", "max_output_tokens": 100}) == raw
    assert deepseek({"request_id": "deepseek", "max_output_tokens": 100}) == raw
    deepseek_body = json.loads(captured[-1][0].data)
    assert deepseek_body["thinking"] == {"type": "disabled"}
    system_contract = json.loads(
        deepseek_body["messages"][0]["content"]
    )
    assert "candidate_schema" not in system_contract
    assert set(system_contract["output_contract"]["top_level_keys"]) == set(
        _candidate()
    )
    assert system_contract["output_contract"]["return_direct_object"] is True
    assert system_contract["output_contract"]["forbidden_wrapper_keys"] == [
        "candidate",
        "candidate_schema",
        "schema",
    ]
    assert system_contract["field_contracts"]["operator"] == {
        "arguments": [],
        "id": "non-empty operator id",
        "instruction": "non-empty executable instruction",
        "kind": "zero-arg",
    }
    assert system_contract["field_contracts"]["router"] == {
        "routes": {"<selected task instance_id>": "<operator id>"}
    }
    assert system_contract["field_contracts"]["preconditions"] == [
        "non-empty execution condition"
    ]
    assert system_contract["field_contracts"]["memory_policy"] == (
        "null or a non-empty JSON object"
    )

    frozen_request = tmp_path / "request.json"
    frozen_response = tmp_path / "raw-response.json"
    request = {"request_id": "frozen", "max_output_tokens": 100}
    frozen_request.write_text(canonical_json(request) + "\n", encoding="utf-8")
    frozen_response.write_text(canonical_json(raw) + "\n", encoding="utf-8")
    frozen = FrozenReplayTeacherTransport(
        request_path=frozen_request,
        response_path=frozen_response,
    )
    assert isinstance(frozen, TeacherTransport)
    assert frozen(request) == raw
    with pytest.raises(ContractViolation, match="identity"):
        frozen({"request_id": "changed", "max_output_tokens": 100})


def test_public_transport_factory_replays_round_directory_without_api_key(
    tmp_path: Path,
) -> None:
    request = {"request_id": "goal-round-0000", "max_output_tokens": 100}
    round_root = tmp_path / "goal-round-0000"
    round_root.mkdir()
    (round_root / "TEACHER-REQUEST.json").write_text(
        canonical_json(request) + "\n", encoding="utf-8"
    )
    response = _raw_response(_candidate())
    (round_root / "TEACHER-RESPONSE.json").write_text(
        canonical_json(response) + "\n", encoding="utf-8"
    )

    transport = build_teacher_transport(
        provider="frozen-replay",
        model="deepseek-chat",
        endpoint=str(tmp_path),
        api_key_env="UNUSED_FOR_FROZEN_REPLAY",
    )

    assert transport(request) == response
    with pytest.raises(ContractViolation, match="missing"):
        transport({"request_id": "goal-round-0001", "max_output_tokens": 100})
