"""Authorized DeepSeek review for the gold-free Round 1 localizer."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from teacher_api import TeacherClient, TeacherProvider, TeacherSample

from .contracts import ContractError, canonical_json, sha256_json
from .ledger import ParentCallLedger
from .p1_parent import P1ParentCallAuthorization

_FIELDS = frozenset(
    {
        "failure_diagnosis",
        "recommended_architecture",
        "localization_stages",
        "leakage_controls",
        "causal_evaluation",
        "rollout_plan",
    }
)
_CAMPAIGN_LIMIT = 3_000_000
_HISTORICAL_CHARGE = 155_227


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be an object")
    return payload


def _freeze(path: Path, payload: dict[str, Any], label: str) -> None:
    if path.exists():
        if _load_json(path, label) != payload:
            raise ContractError(f"frozen {label} does not match replay")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def freeze_round1_localization_strategy_request(
    *, audit_path: Path, research_path: Path, output_path: Path
) -> dict[str, Any]:
    """Freeze aggregate failure evidence without serializing evaluator rows."""

    audit = _load_json(audit_path.resolve(), "Round 1 target audit")
    audit_content = {
        key: value for key, value in audit.items() if key != "evidence_sha256"
    }
    if (
        audit.get("evidence_sha256") != sha256_json(audit_content)
        or audit.get("evaluator_only") is not True
        or audit.get("student_visible") is not False
        or audit.get("answer_content_serialized") is not False
    ):
        raise ContractError("Round 1 target audit boundary is invalid")
    target_audit = audit.get("audit")
    if not isinstance(target_audit, dict):
        raise ContractError("Round 1 target audit aggregate is invalid")
    research = research_path.resolve()
    research_sha = hashlib.sha256(research.read_bytes()).hexdigest()
    request = {
        "schema_version": 1,
        "request_type": "round1-gold-free-localization-review-v1",
        "objective": (
            "Design a scalable gold-free localization layer for a frozen 4B "
            "software-engineering Student before a 60-task causal Skill A/B."
        ),
        "aggregate_failure": {
            "audited_tasks": target_audit.get("audited_tasks"),
            "covered_tasks": target_audit.get("ready_tasks"),
            "uncovered_tasks": int(target_audit.get("audited_tasks", 0))
            - int(target_audit.get("ready_tasks", 0)),
            "current_retrieval": "issue lexical/identifier ranker, top-2 files",
            "current_student_context": "only the first retrieved file",
            "single_target_materializer": True,
        },
        "frontier_patterns": [
            "hierarchical file -> symbol -> fine-location localization",
            "multiple bounded patch realizations and external reranking",
            "minimal auditable search/execute interface",
            "fast-model breadth plus deep-model strategy",
            "automated evaluator plus append-only candidate archive",
        ],
        "research_evidence_sha256": research_sha,
        "constraints": [
            "Never request, infer, or encode task IDs, gold paths, gold patches, or holdout answers.",
            "Use only public issue text and pinned repository source at runtime.",
            "Freeze localization receipts identically across baseline and taught arms.",
            "Keep the frozen evolution engine, model weights, tests, and native harness unchanged.",
            "The Student may emit only typed operator or exact-span plans; deterministic code materialization remains mandatory.",
            "Official native evaluation is the sole capability judge.",
            "Design for a local 4B Student and a DeepSeek V4 Flash teacher/localizer campaign.",
        ],
    }
    request_sha = sha256_json(request)
    content = {
        "schema_version": 1,
        "request": request,
        "request_sha256": request_sha,
        "source_audit_evidence_sha256": audit["evidence_sha256"],
        "source_research_sha256": research_sha,
        "task_ids_included": False,
        "gold_paths_included": False,
        "gold_patches_included": False,
        "holdout_answers_included": False,
        "network_calls_performed": False,
    }
    payload = {**content, "evidence_sha256": sha256_json(content)}
    _freeze(output_path.resolve(), payload, "Round 1 localization strategy request")
    return payload


def create_round1_strategy_authorization(
    *,
    request_path: Path,
    output_path: Path,
    expires_at: datetime,
    maximum_output_tokens: int = 256_000,
) -> dict[str, Any]:
    """Bind the user's 3M-token campaign grant to one exact review request."""

    wrapper = _load_request(request_path)
    approval = P1ParentCallAuthorization.create(
        request_sha256=wrapper["request_sha256"],
        model="deepseek-v4-flash",
        maximum_output_tokens=maximum_output_tokens,
        authorization_id="round1-localizer-v4-flash-3m-user-grant",
        approved_by="user:deepseek-v4-flash-3000000-total-tokens",
        expires_at=expires_at,
    )
    content = {
        "schema_version": 1,
        "campaign_total_token_limit": _CAMPAIGN_LIMIT,
        "campaign_historical_tokens_charged": _HISTORICAL_CHARGE,
        "single_call_authorization": approval.to_dict(),
    }
    payload = {**content, "evidence_sha256": sha256_json(content)}
    _freeze(output_path.resolve(), payload, "Round 1 strategy authorization")
    return payload


def dispatch_round1_localization_strategy(
    *,
    request_path: Path,
    authorization_path: Path,
    ledger_path: Path,
    output_path: Path,
    client: TeacherClient | None = None,
) -> dict[str, Any]:
    """Dispatch once, reserve first, and freeze raw plus parsed evidence."""

    wrapper = _load_request(request_path)
    authorization_wrapper = _load_json(
        authorization_path.resolve(), "Round 1 strategy authorization"
    )
    authorization_content = {
        key: value
        for key, value in authorization_wrapper.items()
        if key != "evidence_sha256"
    }
    if authorization_wrapper.get("evidence_sha256") != sha256_json(
        authorization_content
    ):
        raise ContractError("Round 1 strategy authorization evidence mismatch")
    approval = P1ParentCallAuthorization.from_dict(
        authorization_wrapper["single_call_authorization"]
    )
    configured = client or TeacherClient.from_env(TeacherProvider.DEEPSEEK)
    approval.validate(client=configured)
    if approval.request_sha256 != wrapper["request_sha256"]:
        raise ContractError("Round 1 strategy request authorization mismatch")
    if output_path.exists():
        return _load_json(output_path, "Round 1 localization strategy response")

    ledger = ParentCallLedger(ledger_path, approval.loop_authorization)
    call_id = "round1-localizer-review-001"
    ledger.reserve(call_id=call_id, request_sha256=wrapper["request_sha256"])
    try:
        response = configured.complete(
            TeacherSample(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are reviewing the localization layer of a leakage-free "
                            "software-engineering Agent self-evolution experiment. Return "
                            "one JSON object with exactly: failure_diagnosis (string), "
                            "recommended_architecture (string), localization_stages (array "
                            "of concrete stages), leakage_controls (array), causal_evaluation "
                            "(string), rollout_plan (array of ordered implementation steps). "
                            "Prefer auditable, deterministic interfaces. Explicitly separate "
                            "retrieval recall from final target choice and Skill causality. "
                            "Do not request task identities, gold paths, patches, or holdout "
                            "answers. Do not propose modifying tests, native harnesses, frozen "
                            "engine state, or model weights."
                        ),
                    },
                    {"role": "user", "content": canonical_json(wrapper["request"])},
                ],
                metadata={"request_sha256": wrapper["request_sha256"]},
                max_output_tokens=approval.maximum_output_tokens,
                response_format={"type": "json_object"},
                thinking=True,
                reasoning_effort="high",
            )
        )
        raw_content = {
            "schema_version": 1,
            "call_id": call_id,
            "request_sha256": wrapper["request_sha256"],
            "response_text": response.text,
            "usage": response.usage,
            "provider": response.provider.value,
            "model": response.model,
            "network_calls_performed": True,
        }
        raw_payload = {**raw_content, "evidence_sha256": sha256_json(raw_content)}
        _freeze(
            output_path.with_name(f"{output_path.stem}.raw.json"),
            raw_payload,
            "Round 1 raw localization strategy response",
        )
        strategy = _parse_strategy(response.text)
        used = _usage_tokens(response.usage)
        response_record = {
            "schema_version": 1,
            "strategy": strategy,
            "usage": {
                **response.usage,
                "provider": response.provider.value,
                "model": response.model,
                "maximum_output_tokens": approval.maximum_output_tokens,
            },
        }
        ledger.complete(
            call_id=call_id,
            response_sha256=sha256_json(response_record),
            response=response_record,
            usage=response_record["usage"],
        )
    except Exception as exc:
        ledger.abort(call_id=call_id, reason=f"{type(exc).__name__}: {exc}")
        raise
    after = _HISTORICAL_CHARGE + used
    if after > _CAMPAIGN_LIMIT:
        raise ContractError("DeepSeek campaign token budget exceeded")
    content = {
        "schema_version": 1,
        "call_id": call_id,
        "request_sha256": wrapper["request_sha256"],
        "strategy": strategy,
        "usage": response_record["usage"],
        "campaign_tokens_before": _HISTORICAL_CHARGE,
        "campaign_tokens_charged": used,
        "campaign_tokens_after": after,
        "campaign_total_token_limit": _CAMPAIGN_LIMIT,
        "candidate_status": "advisory_inactive",
        "auto_apply": False,
        "task_ids_included": False,
        "gold_paths_included": False,
        "gold_patches_included": False,
        "holdout_answers_included": False,
        "network_calls_performed": True,
    }
    payload = {**content, "evidence_sha256": sha256_json(content)}
    _freeze(output_path.resolve(), payload, "Round 1 localization strategy response")
    return payload


def _load_request(path: Path) -> dict[str, Any]:
    wrapper = _load_json(path.resolve(), "Round 1 localization strategy request")
    content = {key: value for key, value in wrapper.items() if key != "evidence_sha256"}
    if wrapper.get("evidence_sha256") != sha256_json(content):
        raise ContractError("Round 1 localization strategy request evidence mismatch")
    request = wrapper.get("request")
    if (
        not isinstance(request, dict)
        or wrapper.get("request_sha256") != sha256_json(request)
        or wrapper.get("task_ids_included") is not False
        or wrapper.get("gold_paths_included") is not False
        or wrapper.get("gold_patches_included") is not False
        or wrapper.get("holdout_answers_included") is not False
    ):
        raise ContractError("Round 1 localization strategy request boundary is invalid")
    return wrapper


def _parse_strategy(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    try:
        strategy = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ContractError("DeepSeek Round 1 strategy is not JSON") from exc
    if not isinstance(strategy, dict) or set(strategy) != _FIELDS:
        raise ContractError("DeepSeek Round 1 strategy fields are invalid")
    for field in ("failure_diagnosis", "recommended_architecture", "causal_evaluation"):
        if not isinstance(strategy[field], str) or not strategy[field].strip():
            raise ContractError("DeepSeek Round 1 strategy text is invalid")
    for field in ("localization_stages", "leakage_controls", "rollout_plan"):
        if not isinstance(strategy[field], list) or not strategy[field]:
            raise ContractError("DeepSeek Round 1 strategy list is invalid")
        strategy[field] = [
            row.strip() if isinstance(row, str) else canonical_json(row)
            for row in strategy[field]
        ]
        if any(not row for row in strategy[field]):
            raise ContractError("DeepSeek Round 1 strategy list is invalid")
    return strategy


def _usage_tokens(usage: dict[str, Any]) -> int:
    total = usage.get("total_tokens")
    if type(total) is int and total >= 0:
        return total
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    if type(prompt) is not int or type(completion) is not int:
        raise ContractError("DeepSeek Round 1 strategy usage is invalid")
    return prompt + completion
