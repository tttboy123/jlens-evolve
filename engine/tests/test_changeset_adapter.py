from __future__ import annotations

import json

from changeset_adapter import (
    ADAPTER_ID,
    CHANGESET_SCHEMA,
    adapt_changeset_response,
    build_constrained_prompt,
    validate_changeset,
)


def _valid_payload() -> dict[str, object]:
    return {
        "decision": "candidate",
        "target_surface": "Policy",
        "failure_hypothesis": "重复提案率过高，需要最小策略候选",
        "causal_status": "observational_not_causal",
        "rollback_required": True,
        "auto_apply": False,
        "verification": "matched A/B",
    }


def test_strict_validator_accepts_exact_contract_and_terminal_model_token():
    response = json.dumps(_valid_payload(), ensure_ascii=False) + "<|im_end|>"

    result = validate_changeset(response)

    assert result.valid is True
    assert result.errors == ()
    assert result.value == _valid_payload()


def test_strict_validator_accepts_predeclared_english_repetition_wording():
    payload = {
        **_valid_payload(),
        "failure_hypothesis": "repetition",
    }

    result = validate_changeset(json.dumps(payload))

    assert result.valid is True
    assert result.errors == ()


def test_strict_validator_rejects_extra_fields_enum_drift_and_causal_overclaim():
    payload = {
        **_valid_payload(),
        "decision": "investigate",
        "target_surface": "proposal pipeline",
        "causal_status": "causal",
        "auto_apply": True,
        "extra": "not allowed",
    }

    result = validate_changeset(json.dumps(payload, ensure_ascii=False))

    assert result.valid is False
    assert "exact_schema" in result.errors
    assert "decision_const" in result.errors
    assert "target_surface_enum" in result.errors
    assert "causal_status_const" in result.errors
    assert "auto_apply_const" in result.errors


def test_valid_first_pass_does_not_call_repair():
    repair_calls: list[str] = []

    result = adapt_changeset_response(
        json.dumps(_valid_payload(), ensure_ascii=False),
        task_prompt="frozen task",
        repair=lambda prompt: repair_calls.append(prompt) or "{}",
    )

    assert result.adapter_id == ADAPTER_ID
    assert result.status == "accepted"
    assert result.repairs_used == 0
    assert len(result.attempts) == 1
    assert repair_calls == []
    assert json.loads(result.final_response) == _valid_payload()


def test_invalid_first_pass_gets_exactly_one_bounded_repair():
    repair_calls: list[str] = []

    def repair(prompt: str) -> str:
        repair_calls.append(prompt)
        return json.dumps(_valid_payload(), ensure_ascii=False)

    result = adapt_changeset_response(
        '{"decision":"reject"}',
        task_prompt="JLens observed repetition without causal evidence",
        repair=repair,
    )

    assert result.status == "accepted"
    assert result.repairs_used == 1
    assert len(result.attempts) == 2
    assert len(repair_calls) == 1
    assert "exact_schema" in repair_calls[0]
    assert "JLens observed repetition" in repair_calls[0]
    assert result.attempts[0].validation.valid is False
    assert result.attempts[1].validation.valid is True


def test_second_invalid_response_is_rejected_without_a_third_call():
    repair_calls: list[str] = []

    def still_invalid(prompt: str) -> str:
        repair_calls.append(prompt)
        return '{"decision":"candidate"}'

    result = adapt_changeset_response(
        "not-json",
        task_prompt="frozen task",
        repair=still_invalid,
    )

    assert result.status == "rejected"
    assert result.repairs_used == 1
    assert len(result.attempts) == 2
    assert len(repair_calls) == 1
    assert result.value is None
    assert result.final_response == '{"decision":"candidate"}'


def test_constrained_prompt_embeds_frozen_schema_and_non_application_boundary():
    prompt = build_constrained_prompt("original frozen task")

    assert "original frozen task" in prompt
    assert ADAPTER_ID in prompt
    assert json.dumps(CHANGESET_SCHEMA, ensure_ascii=False, sort_keys=True) in prompt
    assert "auto_apply" in prompt
    assert "不要应用" in prompt
