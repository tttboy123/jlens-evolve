"""Authorized, holdout-free DeepSeek strategy boundary for patch realization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from teacher_api import TeacherClient, TeacherProvider, TeacherSample

from .contracts import ContractError, canonical_json, sha256_json
from .ledger import ParentCallLedger
from .p1_parent import (
    audit_p1_parent_call_budget,
    load_p1_parent_authorization,
)

_STRATEGY_FIELDS = frozenset(
    {
        "failure_analysis",
        "recommended_action_space",
        "operator_catalog",
        "realization_loop",
        "verifier_policy",
        "causal_eval",
        "compiled_skill_requirements",
    }
)
_GOAL_TOKEN_LIMIT = 3_000_000


def freeze_p1_realization_strategy_request(
    *,
    checkpoint_path: Path,
    research_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze anonymized action-space failures without holdout or gold answers."""
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    try:
        sidecar_sha = sidecar.read_text(encoding="utf-8").split()[0]
        checkpoint = json.loads(checkpoint_bytes)
    except (OSError, IndexError, json.JSONDecodeError) as exc:
        raise ContractError("P1 action-space checkpoint is unreadable") from exc
    if sidecar_sha != checkpoint_sha:
        raise ContractError("P1 action-space checkpoint sha256 mismatch")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("status") != "capability_gate_not_met_sampling_stopped"
    ):
        raise ContractError("P1 action-space checkpoint is not terminal")
    experiments = checkpoint.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ContractError("P1 action-space checkpoint has no experiments")
    success_gate = checkpoint.get("success_gate")
    if not isinstance(success_gate, dict) or success_gate.get("met") is not False:
        raise ContractError("P1 action-space capability gate must be unmet")

    failure_evidence: list[dict[str, Any]] = []
    for index, row in enumerate(experiments, 1):
        if not isinstance(row, dict) or not str(row.get("finding", "")).strip():
            raise ContractError("P1 action-space experiment evidence is invalid")
        failure_evidence.append(
            {
                "feedback_alias": f"feedback-{index:02d}",
                "action_space": str(row.get("action_space", "unknown")),
                "finding": str(row["finding"]),
                "baseline_structural_valid": row.get("baseline_structural_valid"),
                "taught_structural_valid": row.get("taught_structural_valid"),
                "baseline_taught_patch_identical": row.get(
                    "baseline_taught_patch_identical"
                ),
                "baseline_raw_sha256": row.get(
                    "baseline_raw_sha256",
                    row.get("baseline", {}).get("raw_output_sha256")
                    if isinstance(row.get("baseline"), dict)
                    else None,
                ),
                "taught_raw_sha256": row.get(
                    "taught_raw_sha256",
                    row.get("taught", {}).get("raw_output_sha256")
                    if isinstance(row.get("taught"), dict)
                    else None,
                ),
            }
        )
    research_sha = hashlib.sha256(research_path.read_bytes()).hexdigest()
    fixed = checkpoint.get("fixed_inputs")
    if not isinstance(fixed, dict):
        raise ContractError("P1 action-space fixed inputs are invalid")
    request = {
        "schema_version": 1,
        "request_type": "patch-realization-strategy-v1",
        "objective": (
            "Design a transferable mechanism that makes a frozen 4B student turn "
            "a correct diagnosis into a changed, applicable, testable code patch."
        ),
        "student_model": fixed.get("student_model"),
        "taskset_fingerprint": fixed.get("taskset_fingerprint"),
        "qualification_fingerprint": fixed.get("qualification_fingerprint"),
        "current_skill_fingerprint": fixed.get("skill_revision_fingerprint"),
        "failure_evidence": failure_evidence,
        "frontier_patterns": [
            "ACI and edit-format design",
            "localize-repair-validate separation",
            "typed AST or symbol edit operators",
            "multiple realizations under one diagnosis",
            "deterministic gates plus verifier selection",
            "trajectory and failure-log distillation",
        ],
        "research_evidence_sha256": research_sha,
        "constraints": [
            "feedback-only; do not request or infer holdout or gold answers",
            "frozen engine and model weights remain unchanged",
            "strategy remains advisory and inactive",
            "official native evaluation is the final judge",
            "mechanism qualification precedes Skill causal A/B",
        ],
    }
    request_sha = sha256_json(request)
    content = {
        "schema_version": 1,
        "request_type": "patch-realization-strategy-v1",
        "source_checkpoint_sha256": checkpoint_sha,
        "source_research_sha256": research_sha,
        "holdout_task_ids_included": False,
        "gold_answers_included": False,
        "strategy_request_sha256": request_sha,
        "strategy_request": request,
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    _freeze_matching(output_path, report, "P1 realization strategy request")
    return report


def dispatch_p1_realization_strategy_call(
    *,
    request_evidence_path: Path,
    authorization_path: Path,
    ledger_path: Path,
    output_path: Path,
    call_id: str,
    prior_response_paths: list[Path] | None = None,
    client: TeacherClient | None = None,
) -> dict[str, Any]:
    """Run one strategy call and freeze it without registering or applying code."""
    wrapper = _load_frozen_request(request_evidence_path)
    request = wrapper["strategy_request"]
    request_sha = str(wrapper["strategy_request_sha256"])
    configured_client = client or TeacherClient.from_env(TeacherProvider.DEEPSEEK)
    approval = load_p1_parent_authorization(authorization_path)
    approval.validate(client=configured_client)
    if approval.request_sha256 != request_sha:
        raise ContractError("P1 strategy authorization request sha256 mismatch")
    reserved_tokens = (
        len(canonical_json(request).encode("utf-8"))
        + 20_000
        + approval.maximum_output_tokens
    )
    prior_calls = audit_p1_parent_call_budget(
        prior_response_paths or [],
        next_call_id=call_id,
        next_reserved_tokens=reserved_tokens,
    )
    prior_tokens = sum(int(row["tokens_charged"]) for row in prior_calls)
    if output_path.exists():
        existing = _load_json(output_path, "P1 realization strategy response")
        _validate_response_wrapper(existing, request_sha=request_sha)
        return existing

    ledger = ParentCallLedger(ledger_path, approval.loop_authorization)
    existing_record = ledger.get(call_id)
    if existing_record is not None:
        raise ContractError("P1 strategy call was already reserved")
    ledger.reserve(call_id=call_id, request_sha256=request_sha)
    try:
        response = configured_client.complete(
            TeacherSample(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are designing the patch-realization layer for a frozen "
                            "4B software-engineering student. The student already states "
                            "the correct diagnosis but often copies code unchanged. Design "
                            "a typed, auditable mechanism rather than another generic prompt. "
                            "Return one JSON object with exactly these fields: "
                            "failure_analysis (string), recommended_action_space (string), "
                            "operator_catalog (array of objects with name, arguments, and "
                            "postconditions), realization_loop (array of steps), "
                            "verifier_policy (array of gates), causal_eval (string), and "
                            "compiled_skill_requirements (array of short requirements). "
                            "Keep the proposal transferable. Do not include task IDs, file "
                            "paths, gold patches, or holdout answers. The strategy is advisory "
                            "and inactive; it may not change tests, harnesses, frozen engine "
                            "state, or model weights. Separate long strategy reasoning from "
                            "the compact Skill eventually shown to the student."
                        ),
                    },
                    {"role": "user", "content": canonical_json(request)},
                ],
                metadata={"strategy_request_sha256": request_sha},
                max_output_tokens=approval.maximum_output_tokens,
                response_format={"type": "json_object"},
                thinking=True,
                reasoning_effort="high",
            )
        )
        raw_content = {
            "schema_version": 1,
            "event_type": "parent-strategy-raw-response",
            "call_id": call_id,
            "request_sha256": request_sha,
            "response_text": response.text,
            "usage": response.usage,
            "provider": response.provider.value,
            "model": response.model,
            "network_calls_performed": True,
        }
        raw_report = {**raw_content, "evidence_sha256": sha256_json(raw_content)}
        raw_path = output_path.with_name(f"{output_path.stem}.raw.json")
        _freeze_matching(raw_path, raw_report, "P1 raw realization strategy response")
        strategy = _parse_strategy(response.text)
        tokens_charged = _usage_total_tokens(response.usage)
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
    if prior_tokens + tokens_charged > _GOAL_TOKEN_LIMIT:
        raise ContractError("P1 long-goal parent token budget exceeded")
    content = {
        "schema_version": 1,
        "event_type": "parent-strategy-response",
        "call_id": call_id,
        "request_sha256": request_sha,
        "authorization_fingerprint": approval.fingerprint,
        "response": response_record,
        "candidate_status": "advisory_inactive",
        "auto_apply": False,
        "holdout_task_ids_included": False,
        "gold_answers_included": False,
        "network_calls_performed": True,
        "tokens_charged": tokens_charged,
        "goal_parent_tokens_before": prior_tokens,
        "goal_parent_tokens_reserved": reserved_tokens,
        "goal_parent_token_limit": _GOAL_TOKEN_LIMIT,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    _freeze_matching(output_path, report, "P1 realization strategy response")
    return report


def _load_frozen_request(path: Path) -> dict[str, Any]:
    wrapper = _load_json(path, "P1 realization strategy request")
    evidence_sha = wrapper.get("evidence_sha256")
    content = {key: value for key, value in wrapper.items() if key != "evidence_sha256"}
    if evidence_sha != sha256_json(content):
        raise ContractError("P1 realization strategy request evidence mismatch")
    if (
        wrapper.get("request_type") != "patch-realization-strategy-v1"
        or wrapper.get("holdout_task_ids_included") is not False
        or wrapper.get("gold_answers_included") is not False
        or wrapper.get("network_calls_performed") is not False
    ):
        raise ContractError("P1 realization strategy request boundary is invalid")
    request = wrapper.get("strategy_request")
    if not isinstance(request, dict) or wrapper.get(
        "strategy_request_sha256"
    ) != sha256_json(request):
        raise ContractError("P1 realization strategy request sha256 mismatch")
    return wrapper


def _parse_strategy(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    try:
        strategy = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ContractError("DeepSeek realization strategy is not JSON") from exc
    if not isinstance(strategy, dict) or set(strategy) != _STRATEGY_FIELDS:
        raise ContractError("DeepSeek realization strategy fields are invalid")
    for field in ("failure_analysis", "recommended_action_space", "causal_eval"):
        if not isinstance(strategy[field], str) or not strategy[field].strip():
            raise ContractError("DeepSeek realization strategy text is invalid")
    catalog = strategy["operator_catalog"]
    if not isinstance(catalog, list) or not catalog:
        raise ContractError("DeepSeek realization operator catalog is invalid")
    for row in catalog:
        if (
            not isinstance(row, dict)
            or set(row) != {"name", "arguments", "postconditions"}
            or not isinstance(row["name"], str)
            or not isinstance(row["arguments"], (list, dict))
            or not isinstance(row["postconditions"], list)
        ):
            raise ContractError("DeepSeek realization operator is invalid")
        if not row["name"].strip() or not row["arguments"] or not row["postconditions"]:
            raise ContractError("DeepSeek realization operator values are invalid")
    for field in (
        "realization_loop",
        "verifier_policy",
        "compiled_skill_requirements",
    ):
        rows = strategy[field]
        if not isinstance(rows, list) or not rows:
            raise ContractError("DeepSeek realization strategy list is invalid")
        normalized = [
            row.strip() if isinstance(row, str) else canonical_json(row) for row in rows
        ]
        if any(not row for row in normalized):
            raise ContractError("DeepSeek realization strategy list is invalid")
        strategy[field] = normalized
    return strategy


def _validate_response_wrapper(wrapper: dict[str, Any], *, request_sha: str) -> None:
    evidence_sha = wrapper.get("evidence_sha256")
    content = {key: value for key, value in wrapper.items() if key != "evidence_sha256"}
    if evidence_sha != sha256_json(content):
        raise ContractError("P1 realization strategy response evidence mismatch")
    if (
        wrapper.get("event_type") != "parent-strategy-response"
        or wrapper.get("request_sha256") != request_sha
        or wrapper.get("candidate_status") != "advisory_inactive"
        or wrapper.get("auto_apply") is not False
    ):
        raise ContractError("P1 realization strategy response boundary is invalid")


def _usage_total_tokens(usage: dict[str, Any]) -> int:
    total = usage.get("total_tokens")
    if type(total) is int and total >= 0:
        return total
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    if (
        type(prompt) is not int
        or prompt < 0
        or type(completion) is not int
        or completion < 0
    ):
        raise ContractError("DeepSeek realization strategy usage is invalid")
    return prompt + completion


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    return value


def _freeze_matching(path: Path, value: dict[str, Any], label: str) -> None:
    output = path.resolve()
    if output.exists():
        if _load_json(output, label) != value:
            raise ContractError(f"frozen {label} does not match evidence")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(value) + "\n", encoding="utf-8")
