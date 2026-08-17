"""Authorized DeepSeek parent transport for the P1 Skill revision boundary."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from teacher_api import (
    TeacherCallError,
    TeacherClient,
    TeacherProvider,
    TeacherSample,
)

from .contracts import (
    ContractError,
    LoopAuthorization,
    LoopRevision,
    ParentModelRequest,
    ParentModelResponse,
    canonical_json,
    sha256_json,
)
from .ledger import ParentCallLedger
from .parent_model import ParentModelAdapter
from .registry import LoopRevisionRegistry

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RESPONSE_FIELDS = frozenset({"protocol", "skill_text", "prompt_template", "eval_note"})
_MAX_P1_OUTPUT_TOKENS = 384_000
_MAX_GOAL_PARENT_TOKENS = 3_000_000
_LEGACY_UNKNOWN_USAGE_TOKEN_CHARGE = 32_000
_STRUCTURED_PROMPT = (
    "Return exactly one JSON object with file, search, replace, diagnostic."
)


@dataclass(frozen=True)
class P1ParentCallAuthorization:
    """Human approval bound to one exact request, provider, model, and token cap."""

    schema_version: int
    request_sha256: str
    provider: str
    model: str
    maximum_output_tokens: int
    loop_authorization: LoopAuthorization

    _CONTENT_FIELDS = frozenset(
        {
            "schema_version",
            "request_sha256",
            "provider",
            "model",
            "maximum_output_tokens",
            "loop_authorization",
        }
    )
    _FIELDS = _CONTENT_FIELDS | {"fingerprint"}

    @classmethod
    def create(
        cls,
        *,
        request_sha256: str,
        model: str,
        maximum_output_tokens: int,
        authorization_id: str,
        approved_by: str,
        expires_at: datetime,
    ) -> P1ParentCallAuthorization:
        approval = cls(
            schema_version=1,
            request_sha256=request_sha256,
            provider=TeacherProvider.DEEPSEEK.value,
            model=model,
            maximum_output_tokens=maximum_output_tokens,
            loop_authorization=LoopAuthorization.create(
                authorization_id=authorization_id,
                approved_by=approved_by,
                maximum_parent_calls=1,
                expires_at=expires_at,
            ),
        )
        approval.validate()
        return approval

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> P1ParentCallAuthorization:
        if not isinstance(data, dict) or set(data) != cls._FIELDS:
            raise ContractError("invalid P1 parent authorization fields")
        approval = cls(
            schema_version=data["schema_version"],
            request_sha256=str(data["request_sha256"]),
            provider=str(data["provider"]),
            model=str(data["model"]),
            maximum_output_tokens=data["maximum_output_tokens"],
            loop_authorization=LoopAuthorization.from_dict(data["loop_authorization"]),
        )
        approval.validate()
        if data["fingerprint"] != approval.fingerprint:
            raise ContractError("P1 parent authorization fingerprint mismatch")
        return approval

    def validate(
        self,
        *,
        request: ParentModelRequest | None = None,
        client: TeacherClient | None = None,
    ) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported P1 parent authorization schema")
        if _SHA256.fullmatch(self.request_sha256) is None:
            raise ContractError("invalid authorized parent request sha256")
        if self.provider != TeacherProvider.DEEPSEEK.value:
            raise ContractError("P1 parent provider must be deepseek")
        if not self.model.strip():
            raise ContractError("P1 parent model must be non-empty")
        if (
            type(self.maximum_output_tokens) is not int
            or not 1 <= self.maximum_output_tokens <= _MAX_P1_OUTPUT_TOKENS
        ):
            raise ContractError("P1 parent output token cap exceeds 384000")
        self.loop_authorization.validate()
        self.loop_authorization.assert_active()
        if self.loop_authorization.maximum_parent_calls != 1:
            raise ContractError("P1 parent authorization must allow exactly one call")
        if request is not None and request.sha256 != self.request_sha256:
            raise ContractError("P1 parent authorization request sha256 mismatch")
        if client is not None:
            if client.config.provider is not TeacherProvider.DEEPSEEK:
                raise ContractError("P1 parent client must use DeepSeek")
            if client.config.model != self.model:
                raise ContractError("P1 parent authorization model mismatch")

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_sha256": self.request_sha256,
            "provider": self.provider,
            "model": self.model,
            "maximum_output_tokens": self.maximum_output_tokens,
            "loop_authorization": self.loop_authorization.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "fingerprint": self.fingerprint}


class P1DeepSeekTransport:
    """Convert a validated P1 request into one token-bounded teacher call."""

    def __init__(self, *, client: TeacherClient, maximum_output_tokens: int) -> None:
        if client.config.provider is not TeacherProvider.DEEPSEEK:
            raise ContractError("P1 transport requires a DeepSeek client")
        if (
            type(maximum_output_tokens) is not int
            or not 1 <= maximum_output_tokens <= _MAX_P1_OUTPUT_TOKENS
        ):
            raise ContractError("P1 transport output token cap exceeds 384000")
        self.client = client
        self.maximum_output_tokens = maximum_output_tokens

    def __call__(self, request: ParentModelRequest) -> dict[str, Any]:
        request.validate()
        response = self.client.complete(
            TeacherSample(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Rewrite an inactive project-local Skill for a frozen 4B "
                            "structured-search-replace experiment. Replace the previous "
                            "skill_text entirely; do not preserve incompatible unified-"
                            "diff or hunk instructions. The student must emit only JSON "
                            "with file, search, replace, diagnostic, so the Skill must "
                            "teach: infer the intended invariant from issue plus source; "
                            "inspect existing behavior before editing; reject duplicate, "
                            "unreachable, or no-op changes; choose an exact unique search "
                            "span; make a minimal syntactically and type-valid replacement; "
                            "and simulate both the target edge case and an ordinary path. "
                            "Teach causal localization: trace where the wrong value is "
                            "created before editing its renderer or first keyword match; "
                            "translate literal expected-output clues into invariants; compare "
                            "the proposed change direction against the issue; and verify every "
                            "referenced attribute, type, and API in the supplied source. "
                            "Use only feedback-arm evidence and never request or infer "
                            "holdout answers. Do not encode task IDs, file paths, or exact "
                            "search/replace spans. You may distill up to three anonymous "
                            "feedback-derived pattern cards, each as symptom -> causal "
                            "transformation -> validation, so the student receives concrete "
                            "transferable teaching rather than repeated generic slogans. "
                            "Return exactly "
                            "one JSON object with protocol, skill_text, prompt_template, "
                            "and eval_note. Preserve structured-search-replace-v1 and the "
                            "canonical prompt. The Skill frontmatter must contain active: "
                            "false and auto_install: false; never activate or install it. "
                            "Keep skill_text at most 2500 characters and at most eight "
                            "short imperative steps. Remove repeated rationale and examples "
                            "so a 4B student can attend to every rule."
                        ),
                    },
                    {
                        "role": "user",
                        "content": canonical_json(request.to_dict()),
                    },
                ],
                metadata={"parent_request_sha256": request.sha256},
                max_output_tokens=self.maximum_output_tokens,
                response_format={"type": "json_object"},
                thinking=True,
                reasoning_effort="high",
            )
        )
        payload = _strict_response_object(response.text)
        usage = dict(response.usage)
        used = usage.get("completion_tokens", usage.get("output_tokens"))
        if isinstance(used, int) and used > self.maximum_output_tokens:
            raise ContractError("P1 parent response exceeded authorized token cap")
        candidate = {
            "schema_version": 1,
            **payload,
            "usage": {
                **usage,
                "provider": response.provider.value,
                "model": response.model,
                "maximum_output_tokens": self.maximum_output_tokens,
            },
        }
        return ParentModelResponse.from_dict(candidate).to_dict()


def p1_parent_preflight(
    *,
    request_evidence_path: Path,
    authorization_path: Path | None = None,
    client: TeacherClient | None = None,
) -> dict[str, Any]:
    """Check the exact dispatch boundary without reserving or calling anything."""
    request = load_frozen_p1_parent_request(request_evidence_path)
    configured_client = client or TeacherClient.from_env(TeacherProvider.DEEPSEEK)
    checks: dict[str, bool] = {
        "request_valid": True,
        "holdout_excluded": True,
        "deepseek_key_configured": bool(configured_client.api_key),
        "authorization_present": authorization_path is not None
        and authorization_path.is_file(),
        "authorization_valid": False,
    }
    errors: list[str] = []
    if not checks["deepseek_key_configured"]:
        accepted_keys = ", ".join(
            (
                configured_client.config.api_key_env,
                *configured_client.config.api_key_env_aliases,
            )
        )
        errors.append(f"missing one of the environment keys: {accepted_keys}")
    if checks["authorization_present"]:
        try:
            approval = load_p1_parent_authorization(authorization_path)
            approval.validate(request=request, client=configured_client)
            checks["authorization_valid"] = True
        except ContractError as exc:
            errors.append(str(exc))
    else:
        errors.append("explicit P1 parent authorization file is missing")
    content = {
        "schema_version": 1,
        "status": "ready" if all(checks.values()) else "blocked",
        "request_sha256": request.sha256,
        "provider": TeacherProvider.DEEPSEEK.value,
        "model": configured_client.config.model,
        "checks": checks,
        "errors": errors,
        "network_calls_performed": False,
    }
    return {**content, "evidence_sha256": sha256_json(content)}


def freeze_p1_parent_preflight(
    *,
    request_evidence_path: Path,
    output_path: Path,
    authorization_path: Path | None = None,
    client: TeacherClient | None = None,
) -> dict[str, Any]:
    """Freeze a non-dispatching readiness report for the external-call boundary."""
    report = p1_parent_preflight(
        request_evidence_path=request_evidence_path,
        authorization_path=authorization_path,
        client=client,
    )
    _freeze_matching(output_path, report, "P1 parent preflight")
    return report


def dispatch_p1_parent_call(
    *,
    request_evidence_path: Path,
    authorization_path: Path,
    ledger_path: Path,
    registry_root: Path,
    output_path: Path,
    call_id: str,
    prior_response_paths: list[Path] | None = None,
    client: TeacherClient | None = None,
) -> dict[str, Any]:
    """Dispatch one authorized call, freeze the response, and append inactive revision."""
    request = load_frozen_p1_parent_request(request_evidence_path)
    configured_client = client or TeacherClient.from_env(TeacherProvider.DEEPSEEK)
    approval = load_p1_parent_authorization(authorization_path)
    approval.validate(request=request, client=configured_client)
    reserved_tokens = _request_token_reservation(
        request,
        maximum_output_tokens=approval.maximum_output_tokens,
    )
    prior_calls = audit_p1_parent_call_budget(
        prior_response_paths or [],
        next_call_id=call_id,
        next_reserved_tokens=reserved_tokens,
    )
    prior_tokens = sum(int(row["tokens_charged"]) for row in prior_calls)
    ledger = ParentCallLedger(ledger_path, approval.loop_authorization)
    adapter = ParentModelAdapter(
        ledger=ledger,
        transport=P1DeepSeekTransport(
            client=configured_client,
            maximum_output_tokens=approval.maximum_output_tokens,
        ),
    )
    response = adapter.generate(
        call_id=call_id,
        request=request,
        authorization=approval.loop_authorization,
    )
    response_tokens = _usage_total_tokens(response.usage)
    if prior_tokens + response_tokens > _MAX_GOAL_PARENT_TOKENS:
        raise ContractError("P1 long-goal parent token budget exceeded")
    next_revision = _next_revision(request, response)
    registry = LoopRevisionRegistry(registry_root)
    latest = registry.latest(request.current_revision.skill_id)
    if latest is None:
        registry.append(request.current_revision)
    elif latest.fingerprint != request.current_revision.fingerprint:
        if latest.fingerprint != next_revision.fingerprint:
            raise ContractError("P1 registry head does not match parent request")
    registry.append(next_revision)
    content = {
        "schema_version": 1,
        "call_id": call_id,
        "request_sha256": request.sha256,
        "authorization_fingerprint": approval.fingerprint,
        "response": response.to_dict(),
        "next_revision": next_revision.to_dict(),
        "candidate_status": "inactive",
        "auto_activate": False,
        "network_calls_performed": True,
        "goal_parent_call_index": len(prior_calls) + 1,
        "goal_parent_tokens_before": prior_tokens,
        "goal_parent_tokens_used": response_tokens,
        "goal_parent_tokens_reserved": reserved_tokens,
        "goal_parent_token_limit": _MAX_GOAL_PARENT_TOKENS,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    _freeze_matching(output_path, report, "P1 parent response")
    return report


def audit_p1_parent_call_budget(
    response_paths: list[Path],
    *,
    next_call_id: str | None = None,
    next_reserved_tokens: int = 0,
) -> list[dict[str, str | int]]:
    """Validate unique calls against the cumulative long-goal token budget."""
    if type(next_reserved_tokens) is not int or next_reserved_tokens < 0:
        raise ContractError("P1 next reserved tokens must be non-negative")
    calls: list[dict[str, str | int]] = []
    for path in response_paths:
        wrapper = _load_json(path, "P1 prior parent response")
        evidence_sha = wrapper.get("evidence_sha256")
        content = {
            key: value for key, value in wrapper.items() if key != "evidence_sha256"
        }
        if evidence_sha != sha256_json(content):
            raise ContractError("P1 prior parent response evidence sha256 mismatch")
        if wrapper.get("event_type") == "parent-call-terminal":
            if (
                wrapper.get("status") != "aborted"
                or wrapper.get("budget_consumed") is not True
                or wrapper.get("network_calls_performed") is not True
            ):
                raise ContractError("P1 parent terminal event is invalid")
            call_id = str(wrapper.get("call_id", ""))
            request_sha = str(wrapper.get("request_sha256", ""))
            if not call_id or _SHA256.fullmatch(request_sha) is None:
                raise ContractError("P1 parent terminal identity is invalid")
            tokens_charged = wrapper.get(
                "tokens_charged", _LEGACY_UNKNOWN_USAGE_TOKEN_CHARGE
            )
            if type(tokens_charged) is not int or tokens_charged < 1:
                raise ContractError("P1 parent terminal token charge is invalid")
            calls.append(
                {
                    "call_id": call_id,
                    "request_sha256": request_sha,
                    "tokens_charged": tokens_charged,
                }
            )
            continue
        if wrapper.get("event_type") == "parent-strategy-response":
            if (
                wrapper.get("network_calls_performed") is not True
                or wrapper.get("candidate_status") != "advisory_inactive"
                or wrapper.get("auto_apply") is not False
            ):
                raise ContractError("P1 parent strategy response is invalid")
            call_id = str(wrapper.get("call_id", ""))
            request_sha = str(wrapper.get("request_sha256", ""))
            tokens_charged = wrapper.get("tokens_charged")
            if (
                not call_id
                or _SHA256.fullmatch(request_sha) is None
                or type(tokens_charged) is not int
                or tokens_charged < 1
            ):
                raise ContractError("P1 parent strategy identity is invalid")
            calls.append(
                {
                    "call_id": call_id,
                    "request_sha256": request_sha,
                    "tokens_charged": tokens_charged,
                }
            )
            continue
        if wrapper.get("network_calls_performed") is not True:
            raise ContractError("P1 prior parent response must record a network call")
        load_frozen_p1_parent_revision(path)
        call_id = str(wrapper.get("call_id", ""))
        request_sha = str(wrapper.get("request_sha256", ""))
        if not call_id or _SHA256.fullmatch(request_sha) is None:
            raise ContractError("P1 prior parent response identity is invalid")
        response = wrapper.get("response")
        usage = response.get("usage") if isinstance(response, dict) else None
        tokens_charged = (
            _usage_total_tokens(usage)
            if isinstance(usage, dict)
            else _LEGACY_UNKNOWN_USAGE_TOKEN_CHARGE
        )
        calls.append(
            {
                "call_id": call_id,
                "request_sha256": request_sha,
                "tokens_charged": tokens_charged,
            }
        )
    if len({row["call_id"] for row in calls}) != len(calls):
        raise ContractError("P1 prior parent call IDs must be unique")
    if len({str(row["request_sha256"]) for row in calls}) != len(calls):
        raise ContractError("P1 prior parent requests must be unique")
    tokens_before = sum(int(row["tokens_charged"]) for row in calls)
    if tokens_before + next_reserved_tokens > _MAX_GOAL_PARENT_TOKENS:
        raise ContractError("P1 long-goal parent token budget exceeded")
    if next_call_id is not None and next_call_id in {
        str(row["call_id"]) for row in calls
    }:
        raise ContractError("P1 next call ID already exists in prior evidence")
    return calls


def _usage_total_tokens(usage: dict[str, Any]) -> int:
    total = usage.get("total_tokens")
    if type(total) is int and total >= 0:
        return total
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    if (
        type(input_tokens) is not int
        or input_tokens < 0
        or type(output_tokens) is not int
        or output_tokens < 0
    ):
        raise ContractError("P1 parent usage token counts are invalid")
    return input_tokens + output_tokens


def _request_token_reservation(
    request: ParentModelRequest,
    *,
    maximum_output_tokens: int,
) -> int:
    """Conservatively reserve UTF-8 input bytes plus the output ceiling."""
    input_upper_bound = len(canonical_json(request.to_dict()).encode("utf-8"))
    return input_upper_bound + 20_000 + maximum_output_tokens


def load_frozen_p1_parent_request(path: Path) -> ParentModelRequest:
    wrapper = _load_json(path, "P1 parent request evidence")
    if wrapper.get("holdout_task_ids_included") is not False:
        raise ContractError("holdout evidence is prohibited from P1 parent request")
    if wrapper.get("network_calls_performed") is not False:
        raise ContractError("P1 parent request evidence must be offline")
    evidence_sha = wrapper.get("evidence_sha256")
    content = {key: value for key, value in wrapper.items() if key != "evidence_sha256"}
    if evidence_sha != sha256_json(content):
        raise ContractError("P1 parent request evidence sha256 mismatch")
    request = ParentModelRequest.from_dict(wrapper.get("parent_request"))
    if wrapper.get("parent_request_sha256") != request.sha256:
        raise ContractError("P1 parent request sha256 mismatch")
    return request


def load_p1_parent_authorization(path: Path) -> P1ParentCallAuthorization:
    return P1ParentCallAuthorization.from_dict(
        _load_json(path, "P1 parent authorization")
    )


def load_frozen_p1_parent_revision(path: Path) -> LoopRevision:
    """Load the inactive next revision only after response evidence validates."""
    wrapper = _load_json(path, "P1 parent response")
    evidence_sha = wrapper.get("evidence_sha256")
    content = {key: value for key, value in wrapper.items() if key != "evidence_sha256"}
    if evidence_sha != sha256_json(content):
        raise ContractError("P1 parent response evidence sha256 mismatch")
    if (
        wrapper.get("candidate_status") != "inactive"
        or wrapper.get("auto_activate") is not False
    ):
        raise ContractError("P1 parent revision must remain inactive")
    response = ParentModelResponse.from_dict(wrapper.get("response"))
    revision = LoopRevision.from_dict(wrapper.get("next_revision"))
    if (
        revision.source_round < 1
        or revision.parent_revision_id is None
        or revision.protocol != response.protocol
        or revision.skill_text != response.skill_text
        or revision.prompt_template != response.prompt_template
        or revision.eval_note != response.eval_note
        or not revision.revision_id.endswith(response.sha256[:8])
    ):
        raise ContractError("P1 parent revision lineage does not match response")
    return revision


def _strict_response_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise TeacherCallError("DeepSeek parent response is not strict JSON") from exc
    if not isinstance(parsed, dict) or set(parsed) != _RESPONSE_FIELDS:
        raise ContractError("DeepSeek parent response fields are invalid")
    if any(
        not isinstance(parsed[field], str) or not parsed[field].strip()
        for field in parsed
    ):
        raise ContractError("DeepSeek parent response values must be non-empty strings")
    if parsed["protocol"] != "structured-search-replace-v1":
        raise ContractError("DeepSeek parent changed the frozen P1 protocol")
    if parsed["prompt_template"] != _STRUCTURED_PROMPT:
        raise ContractError("DeepSeek parent changed the paired P1 prompt")
    skill = parsed["skill_text"]
    if any(marker in skill for marker in ("--- a/", "+++ b/", "@@ -", "@@ ")):
        raise ContractError("DeepSeek parent returned unified-diff Skill instructions")
    if not all(marker in skill for marker in ("search", "replace", "diagnostic")):
        raise ContractError("DeepSeek parent Skill omits structured edit fields")
    if "active: false" not in skill or "auto_install: false" not in skill:
        raise ContractError("DeepSeek parent Skill is not explicitly inactive")
    if len(skill) > 2500:
        raise ContractError("DeepSeek parent Skill exceeds 2500 characters")
    return parsed


def _next_revision(
    request: ParentModelRequest, response: ParentModelResponse
) -> LoopRevision:
    current = request.current_revision
    source_round = request.feedback.current_round + 1
    return LoopRevision.create(
        skill_id=current.skill_id,
        revision_id=f"{current.skill_id}-r{source_round:03d}-{response.sha256[:8]}",
        parent_revision_id=current.revision_id,
        source_round=source_round,
        protocol=response.protocol,
        skill_text=response.skill_text,
        prompt_template=response.prompt_template,
        eval_note=response.eval_note,
    )


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
