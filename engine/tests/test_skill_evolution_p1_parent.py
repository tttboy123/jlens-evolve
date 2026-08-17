"""Offline tests for the explicitly authorized P1 DeepSeek boundary."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from skill_evolution_loop import (
    ContractError,
    FailureEvidence,
    FeedbackPackage,
    LoopRevision,
    P1ParentCallAuthorization,
    ParentCallLedger,
    ParentModelRequest,
    audit_p1_parent_call_budget,
    dispatch_p1_parent_call,
    freeze_p1_local_feedback_revision,
    load_frozen_p1_parent_revision,
    p1_parent_preflight,
)
from skill_evolution_loop.contracts import canonical_json, sha256_json
from skill_evolution_loop.p1_experiment import build_p1_conditions
from teacher_api import TeacherClient, TeacherConfig, TeacherProvider


class _FakeResponse:
    status_code = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def json(self) -> dict[str, Any]:
        return self.payload


def _request() -> ParentModelRequest:
    revision = LoopRevision.create(
        skill_id="p1-local-qwen-skill",
        revision_id="p1-structured-taught-v1",
        parent_revision_id=None,
        source_round=0,
        protocol="structured-search-replace-v1",
        skill_text="Use one exact structured edit.",
        prompt_template="Return one JSON object.",
        eval_note="Round zero did not improve native capability.",
    )
    feedback = FeedbackPackage.create(
        current_round=0,
        arm_evidence=[
            FailureEvidence.create(
                task_id="feedback-001",
                reason_code="native-unresolved",
                diagnostic_summary="The edit was applicable but semantically wrong.",
                raw_output_sha256="a" * 64,
                extracted_edit_sha256="b" * 64,
                apply_error=None,
            )
        ],
        previous_eval_note="No native gain.",
        no_progress=False,
        rejected_fingerprints=[revision.fingerprint],
    )
    return ParentModelRequest.create(feedback=feedback, current_revision=revision)


def _freeze_request(path: Path, request: ParentModelRequest) -> None:
    content = {
        "schema_version": 1,
        "source_composition_sha256": "c" * 64,
        "condition_id": "structured-taught",
        "feedback_task_count": 1,
        "holdout_task_ids_included": False,
        "parent_request_sha256": request.sha256,
        "parent_request": request.to_dict(),
        "network_calls_performed": False,
    }
    path.write_text(
        canonical_json({**content, "evidence_sha256": sha256_json(content)}) + "\n",
        encoding="utf-8",
    )


def _approval(request: ParentModelRequest) -> P1ParentCallAuthorization:
    return P1ParentCallAuthorization.create(
        request_sha256=request.sha256,
        model="deepseek-v4-flash",
        maximum_output_tokens=32_000,
        authorization_id="p1-round-001",
        approved_by="user",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )


def _client(calls: list[dict[str, Any]]) -> TeacherClient:
    def transport(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json})
        return _FakeResponse(
            {
                "model": "deepseek-v4-flash",
                "usage": {"prompt_tokens": 100, "completion_tokens": 200},
                "choices": [
                    {
                        "message": {
                            "content": json_module.dumps(
                                {
                                    "protocol": "structured-search-replace-v1",
                                    "skill_text": (
                                        "---\nactive: false\nauto_install: false\n---\n"
                                        "Inspect behavior, choose a unique search span, "
                                        "then emit file, search, replace, and diagnostic "
                                        "fields as one exact structured edit."
                                    ),
                                    "prompt_template": (
                                        "Return exactly one JSON object with file, "
                                        "search, replace, diagnostic."
                                    ),
                                    "eval_note": "Focused teaching on semantic verification.",
                                }
                            )
                        }
                    }
                ],
            }
        )

    json_module = json
    return TeacherClient(
        TeacherConfig(
            provider=TeacherProvider.DEEPSEEK,
            api_base="https://deepseek.example.invalid/v1",
            model="deepseek-v4-flash",
            api_key_env="P1_TEST_KEY",
        ),
        api_key="test-key",
        transport=transport,
    )


def test_parent_preflight_is_blocked_without_key_or_human_authorization(
    tmp_path: Path,
) -> None:
    request = _request()
    request_path = tmp_path / "request.json"
    _freeze_request(request_path, request)
    client = TeacherClient(
        TeacherConfig(
            provider=TeacherProvider.DEEPSEEK,
            api_base="https://deepseek.example.invalid/v1",
            model="deepseek-v4-flash",
            api_key_env="P1_MISSING_KEY",
        ),
        api_key="",
    )

    report = p1_parent_preflight(
        request_evidence_path=request_path,
        client=client,
    )

    assert report["status"] == "blocked"
    assert report["checks"]["deepseek_key_configured"] is False
    assert report["checks"]["authorization_present"] is False
    assert report["network_calls_performed"] is False


def test_authorization_is_bound_to_exact_request_model_and_token_cap() -> None:
    request = _request()
    approval = _approval(request)
    calls: list[dict[str, Any]] = []
    approval.validate(request=request, client=_client(calls))

    other = ParentModelRequest.create(
        feedback=FeedbackPackage.create(
            current_round=1,
            arm_evidence=list(request.feedback.arm_evidence),
            previous_eval_note="different",
            no_progress=True,
            rejected_fingerprints=list(request.feedback.rejected_fingerprints),
        ),
        current_revision=request.current_revision,
    )
    with pytest.raises(ContractError, match="request sha256 mismatch"):
        approval.validate(request=other)
    with pytest.raises(ContractError, match="exceeds 384000"):
        P1ParentCallAuthorization.create(
            request_sha256=request.sha256,
            model="deepseek-v4-flash",
            maximum_output_tokens=384_001,
            authorization_id="p1-round-002",
            approved_by="user",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )


def test_dispatch_reserves_once_freezes_inactive_revision_and_replays(
    tmp_path: Path,
) -> None:
    request = _request()
    request_path = tmp_path / "request.json"
    approval_path = tmp_path / "authorization.json"
    _freeze_request(request_path, request)
    approval = _approval(request)
    approval_path.write_text(
        canonical_json(approval.to_dict()) + "\n", encoding="utf-8"
    )
    calls: list[dict[str, Any]] = []
    client = _client(calls)
    kwargs = {
        "request_evidence_path": request_path,
        "authorization_path": approval_path,
        "ledger_path": tmp_path / "ledger.json",
        "registry_root": tmp_path / "registry",
        "output_path": tmp_path / "response.json",
        "call_id": "p1-skill-round-001",
        "client": client,
    }

    first = dispatch_p1_parent_call(**kwargs)
    repeated = dispatch_p1_parent_call(**kwargs)

    assert repeated == first
    assert first["candidate_status"] == "inactive"
    assert first["auto_activate"] is False
    assert first["next_revision"]["source_round"] == 1
    assert len(calls) == 1
    assert calls[0]["json"]["max_tokens"] == 32_000
    assert calls[0]["json"]["response_format"] == {"type": "json_object"}
    assert calls[0]["json"]["thinking"] == {"type": "enabled"}
    assert calls[0]["json"]["reasoning_effort"] == "high"
    system_prompt = calls[0]["json"]["messages"][0]["content"]
    assert "Replace the previous skill_text entirely" in system_prompt
    assert "unified-diff" in system_prompt
    assert "exact unique search span" in system_prompt
    assert "trace where the wrong value is created" in system_prompt
    assert "active: false" in system_prompt
    assert "at most 2500 characters" in system_prompt
    assert first["goal_parent_call_index"] == 1
    assert first["goal_parent_tokens_before"] == 0
    assert first["goal_parent_tokens_used"] == 300
    assert first["goal_parent_token_limit"] == 3_000_000
    assert (
        ParentCallLedger(tmp_path / "ledger.json", approval.loop_authorization)
        .records()[0]
        .status
        == "completed"
    )
    revision = load_frozen_p1_parent_revision(tmp_path / "response.json")
    conditions = build_p1_conditions(
        revision.skill_text,
        structured_taught_revision=revision,
    )
    taught = next(
        condition
        for condition in conditions
        if condition.condition_id == "structured-taught"
    )
    baseline = next(
        condition
        for condition in conditions
        if condition.condition_id == "structured-baseline"
    )
    assert taught.revision == revision
    assert baseline.revision.source_round == 0


def test_parent_revision_cannot_change_the_paired_prompt_contract() -> None:
    revision = LoopRevision.create(
        skill_id="p1-local-qwen-skill",
        revision_id="p1-local-qwen-skill-r001-deadbeef",
        parent_revision_id="p1-structured-taught-v1",
        source_round=1,
        protocol="structured-search-replace-v1",
        skill_text="Use one exact edit.",
        prompt_template="A changed prompt would confound the pair.",
        eval_note="invalid paired prompt",
    )

    with pytest.raises(ContractError, match="paired prompt contract"):
        build_p1_conditions(
            revision.skill_text,
            structured_taught_revision=revision,
        )


def test_long_goal_parent_budget_counts_unique_frozen_responses(tmp_path: Path) -> None:
    request = _request()
    request_path = tmp_path / "request.json"
    approval_path = tmp_path / "authorization.json"
    _freeze_request(request_path, request)
    approval = _approval(request)
    approval_path.write_text(
        canonical_json(approval.to_dict()) + "\n", encoding="utf-8"
    )
    response_path = tmp_path / "response.json"
    dispatch_p1_parent_call(
        request_evidence_path=request_path,
        authorization_path=approval_path,
        ledger_path=tmp_path / "ledger.json",
        registry_root=tmp_path / "registry",
        output_path=response_path,
        call_id="p1-skill-round-001",
        client=_client([]),
    )

    calls = audit_p1_parent_call_budget(
        [response_path], next_call_id="p1-skill-round-002"
    )

    assert calls == [
        {
            "call_id": "p1-skill-round-001",
            "request_sha256": request.sha256,
            "tokens_charged": 300,
        }
    ]
    with pytest.raises(ContractError, match="already exists"):
        audit_p1_parent_call_budget([response_path], next_call_id="p1-skill-round-001")
    with pytest.raises(ContractError, match="IDs must be unique"):
        audit_p1_parent_call_budget([response_path, response_path])


def test_long_goal_budget_is_token_bounded_not_five_call_bounded(
    tmp_path: Path,
) -> None:
    paths: list[Path] = []
    for index in range(6):
        content = {
            "schema_version": 1,
            "event_type": "parent-call-terminal",
            "status": "aborted",
            "budget_consumed": True,
            "call_id": f"p1-skill-round-{index + 1:03d}",
            "request_sha256": f"{index + 1:064x}",
            "network_calls_performed": True,
            "tokens_charged": 100_000,
        }
        path = tmp_path / f"terminal-{index}.json"
        path.write_text(
            canonical_json({**content, "evidence_sha256": sha256_json(content)}) + "\n",
            encoding="utf-8",
        )
        paths.append(path)

    calls = audit_p1_parent_call_budget(
        paths,
        next_call_id="p1-skill-round-007",
        next_reserved_tokens=100_000,
    )

    assert len(calls) == 6
    with pytest.raises(ContractError, match="token budget exceeded"):
        audit_p1_parent_call_budget(
            paths,
            next_call_id="p1-skill-round-007",
            next_reserved_tokens=2_500_000,
        )


def test_local_feedback_compiler_freezes_inactive_zero_network_revision(
    tmp_path: Path,
) -> None:
    request = _request()
    request_path = tmp_path / "request.json"
    approval_path = tmp_path / "authorization.json"
    _freeze_request(request_path, request)
    approval = _approval(request)
    approval_path.write_text(
        canonical_json(approval.to_dict()) + "\n", encoding="utf-8"
    )
    response_path = tmp_path / "response.json"
    registry = tmp_path / "registry"
    parent = dispatch_p1_parent_call(
        request_evidence_path=request_path,
        authorization_path=approval_path,
        ledger_path=tmp_path / "ledger.json",
        registry_root=registry,
        output_path=response_path,
        call_id="p1-skill-round-001",
        client=_client([]),
    )
    semantic_path = tmp_path / "semantic.json"
    semantic_path.write_text(
        canonical_json(
            {
                "skill_revision_fingerprint": parent["next_revision"]["fingerprint"],
                "network_calls_performed": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = freeze_p1_local_feedback_revision(
        current_revision_path=response_path,
        semantic_review_path=semantic_path,
        pattern_cards=[
            "Symptom: a value is lost across two categories. Transformation: align "
            "one value vector before both loops. Validation: cover empty and present values."
        ],
        output_path=tmp_path / "local-revision.json",
        registry_root=registry,
    )

    assert report["candidate_source"] == "local-feedback-compiler-v1"
    assert report["network_calls_performed"] is False
    assert report["candidate_status"] == "inactive"
    assert report["next_revision"]["source_round"] == 2
    assert "active: false" in report["next_revision"]["skill_text"]
