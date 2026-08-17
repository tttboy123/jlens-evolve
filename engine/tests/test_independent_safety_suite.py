from __future__ import annotations

import pytest

from skill_evolution_loop.contracts import ContractError
from skill_evolution_loop.independent_safety import build_independent_safety_report


def _probe(category: str, *, passed: bool = True) -> dict[str, object]:
    return {
        "category": category,
        "probe_id": f"probe-{category}",
        "passed": passed,
        "evaluator_valid": True,
        "evaluator": f"independent-{category}-v1",
        "evidence_sha256": (category[0] if category[0] in "abcdef" else "a") * 64,
    }


def _complete(*, failed: str | None = None) -> tuple[dict[str, object], ...]:
    categories = (
        "dangerous-command",
        "http-5xx",
        "private-data-exposure",
        "unauthorized-side-effect",
    )
    return tuple(_probe(category, passed=category != failed) for category in categories)


def test_independent_safety_requires_all_four_valid_probes() -> None:
    report = build_independent_safety_report(
        subject_sha256="f" * 64,
        probes=_complete(),
    )

    assert report["suite_passed"] is True
    assert report["required_categories_present"] is True
    assert report["native_admission_reused"] is False
    assert report["probe_count"] == 4


def test_independent_safety_fails_on_one_real_probe_failure() -> None:
    report = build_independent_safety_report(
        subject_sha256="f" * 64,
        probes=_complete(failed="private-data-exposure"),
    )

    assert report["suite_passed"] is False
    assert report["failed_categories"] == ["private-data-exposure"]


def test_independent_safety_rejects_missing_or_native_admission_probes() -> None:
    with pytest.raises(ContractError, match="exactly one probe"):
        build_independent_safety_report(
            subject_sha256="f" * 64,
            probes=_complete()[:-1],
        )

    probes = list(_complete())
    probes[0] = {**probes[0], "evaluator": "native.safe"}
    with pytest.raises(ContractError, match="cannot reuse native admission"):
        build_independent_safety_report(
            subject_sha256="f" * 64,
            probes=tuple(probes),
        )


def test_independent_safety_is_fail_closed_on_evaluator_error() -> None:
    probes = list(_complete())
    probes[1] = {**probes[1], "evaluator_valid": False, "error": "timeout"}
    report = build_independent_safety_report(
        subject_sha256="f" * 64,
        probes=tuple(probes),
    )

    assert report["suite_passed"] is False
    assert report["evaluator_failure_count"] == 1
    assert report["failed_categories"] == ["http-5xx"]
