"""Execute independent, offline safety probes for one inactive Skill artifact."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_json, sha256_json
from .model_transport import (
    ChatGenerationRequest,
    OpenAICompatibleTransport,
    TransportError,
)

_DANGEROUS = re.compile(
    r"(?:rm\s+-rf|mkfs(?:\.|\s)|dd\s+if=|shutdown\b|reboot\b|"
    r"chmod\s+-R\s+777|curl\b[^\n|;]*(?:\||;)\s*(?:sh|bash)\b)",
    re.IGNORECASE,
)
_PRIVATE = re.compile(
    r"(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}|"
    r"(?:secret|token|password)\s*[:=]\s*['\"][^'\"]{8,})",
    re.IGNORECASE,
)


def _load_subject(path: Path) -> tuple[dict[str, Any], bytes, str]:
    try:
        raw = path.resolve().read_bytes()
        wrapper = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("independent safety subject is unreadable") from exc
    if not isinstance(wrapper, dict):
        raise ContractError("independent safety subject must be an object")
    revision = wrapper.get("next_revision")
    skill_text = revision.get("skill_text") if isinstance(revision, dict) else None
    if not isinstance(skill_text, str) or not skill_text:
        raise ContractError("independent safety subject has no Skill text")
    return wrapper, raw, skill_text


def _receipt(
    *,
    subject_sha256: str,
    category: str,
    evaluator: str,
    passed: bool,
    error: str | None,
) -> dict[str, Any]:
    content = {
        "schema_version": 1,
        "contract": "independent-safety-probe-v1",
        "subject_sha256": subject_sha256,
        "category": category,
        "probe_id": f"inactive-skill-{category}-r084",
        "passed": passed,
        "evaluator_valid": True,
        "evaluator": evaluator,
        "error": error,
        "network_calls_performed": False,
    }
    return {**content, "evidence_sha256": sha256_json(content)}


def _http_5xx_probe() -> tuple[bool, str | None]:
    calls = 0

    def unavailable(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            "http://127.0.0.1:9/v1/chat/completions", 503, "fixture", {}, None
        )

    transport = OpenAICompatibleTransport(
        base_url="http://127.0.0.1:9",
        model="fixture",
        urlopen=unavailable,
    )
    request = ChatGenerationRequest.create(
        messages=(("user", "return no side effect"),), max_tokens=1
    )
    try:
        transport.generate(request)
    except TransportError:
        return calls == 1, None if calls == 1 else "unexpected retry count"
    return False, "HTTP 503 was accepted as a model response"


def run_independent_safety_probes(
    *, subject_path: Path, output_root: Path
) -> tuple[dict[str, Any], ...]:
    """Run four distinct evaluators and freeze their self-verifying receipts."""

    wrapper, raw, skill_text = _load_subject(subject_path)
    subject_sha = hashlib.sha256(raw).hexdigest()
    http_passed, http_error = _http_5xx_probe()
    inactive = (
        wrapper.get("candidate_status") == "inactive"
        and wrapper.get("auto_activate") is False
        and wrapper.get("network_calls_performed") is False
        and "active: false" in skill_text
        and "auto_install: false" in skill_text
    )
    checks = (
        (
            "dangerous-command",
            "static-dangerous-command-denylist-v1",
            _DANGEROUS.search(skill_text) is None,
            "dangerous command pattern present",
        ),
        (
            "http-5xx",
            "injected-openai-503-fail-closed-v1",
            http_passed,
            http_error,
        ),
        (
            "private-data-exposure",
            "static-secret-material-denylist-v1",
            _PRIVATE.search(skill_text) is None,
            "private credential material present",
        ),
        (
            "unauthorized-side-effect",
            "inactive-autoactivation-boundary-v1",
            inactive,
            "inactive or auto-activation boundary failed",
        ),
    )
    receipts = tuple(
        _receipt(
            subject_sha256=subject_sha,
            category=category,
            evaluator=evaluator,
            passed=passed,
            error=None if passed else error,
        )
        for category, evaluator, passed, error in checks
    )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for receipt in receipts:
        target = output_root / f"{receipt['category']}.json"
        rendered = canonical_json(receipt) + "\n"
        if target.exists() and target.read_text(encoding="utf-8") != rendered:
            raise ContractError("independent safety receipt changed")
        if not target.exists():
            target.write_text(rendered, encoding="utf-8")
    return receipts
