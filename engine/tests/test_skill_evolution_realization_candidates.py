from __future__ import annotations

import pytest

from skill_evolution_loop.contracts import ContractError
from skill_evolution_loop.realization_candidates import (
    FrozenDiagnosis,
    RealizationCandidate,
    RealizationSelection,
    select_realization_candidate,
)


def _diagnosis(
    *, desired_boundary: str = "return the input plus two"
) -> FrozenDiagnosis:
    return FrozenDiagnosis.create(
        defect="answer adds one",
        trigger="answer is called",
        desired_boundary=desired_boundary,
    )


def _candidate(
    candidate_id: str,
    *,
    diagnosis: FrozenDiagnosis,
    patch: str,
    structural_valid: bool = True,
    failure_reason: str | None = None,
) -> RealizationCandidate:
    return RealizationCandidate.create(
        candidate_id=candidate_id,
        diagnosis_sha256=diagnosis.fingerprint,
        raw_output_sha256=(candidate_id[0] * 64),
        patch=patch,
        structural_valid=structural_valid,
        failure_reason=failure_reason,
    )


def test_realization_selection_freezes_diagnosis_and_chooses_smallest_patch() -> None:
    diagnosis = _diagnosis()
    large = _candidate(
        "candidate-large",
        diagnosis=diagnosis,
        patch=(
            "--- a/src/example.py\n+++ b/src/example.py\n"
            "@@ -1,2 +1,3 @@\n-old\n+new\n+extra\n"
        ),
    )
    small = _candidate(
        "candidate-small",
        diagnosis=diagnosis,
        patch="--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-old\n+new\n",
    )

    selection = select_realization_candidate(
        diagnosis=diagnosis,
        candidates=[large, small],
        maximum_candidates=3,
    )

    assert selection.selected_candidate_id == "candidate-small"
    assert selection.diagnosis_sha256 == diagnosis.fingerprint
    assert selection.selection_policy == "minimal-changed-lines-then-id-v1"
    assert [row["status"] for row in selection.candidate_decisions] == [
        "eligible",
        "selected",
    ]
    assert selection.to_dict()["evidence_sha256"] == selection.evidence_sha256


def test_realization_selection_rejects_diagnosis_drift_and_duplicate_patch() -> None:
    diagnosis = _diagnosis()
    drifted = _diagnosis(desired_boundary="return the input plus three")
    patch = "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-old\n+new\n"
    first = _candidate("candidate-a", diagnosis=diagnosis, patch=patch)
    duplicate = _candidate("candidate-b", diagnosis=diagnosis, patch=patch)
    mismatch = _candidate("candidate-c", diagnosis=drifted, patch=patch + "+x\n")

    selection = select_realization_candidate(
        diagnosis=diagnosis,
        candidates=[first, duplicate, mismatch],
        maximum_candidates=3,
    )

    assert selection.selected_candidate_id == "candidate-a"
    assert [row["status"] for row in selection.candidate_decisions] == [
        "selected",
        "duplicate-patch",
        "diagnosis-mismatch",
    ]


def test_realization_selection_fails_closed_when_no_candidate_is_eligible() -> None:
    diagnosis = _diagnosis()
    invalid = _candidate(
        "candidate-invalid",
        diagnosis=diagnosis,
        patch="",
        structural_valid=False,
        failure_reason="selector-no-match",
    )

    selection = select_realization_candidate(
        diagnosis=diagnosis,
        candidates=[invalid],
        maximum_candidates=2,
    )

    assert selection.selected_candidate_id is None
    assert selection.candidate_decisions == (
        {
            "candidate_id": "candidate-invalid",
            "status": "structural-invalid",
            "failure_reason": "selector-no-match",
        },
    )


@pytest.mark.parametrize(
    "failure_reason",
    [
        "inconsistent-plan",
        "identifier-drift",
        "non-executable-insertion",
        "schema-invalid",
        "selector-not-enumerated",
        "semantic-overbroad",
        "syntax-invalid",
        "unbound-name",
        "unsafe-empty-sequence",
    ],
)
def test_realization_candidate_accepts_adapter_failure_taxonomy(
    failure_reason: str,
) -> None:
    diagnosis = _diagnosis()

    candidate = _candidate(
        f"candidate-{failure_reason}",
        diagnosis=diagnosis,
        patch="",
        structural_valid=False,
        failure_reason=failure_reason,
    )

    assert candidate.failure_reason == failure_reason


def test_realization_candidate_rejects_unknown_failure_reason() -> None:
    diagnosis = _diagnosis()

    with pytest.raises(ContractError, match="invalid realization failure_reason"):
        _candidate(
            "candidate-unknown-failure",
            diagnosis=diagnosis,
            patch="",
            structural_valid=False,
            failure_reason="unknown-failure",
        )


def test_realization_selection_enforces_frozen_candidate_budget() -> None:
    diagnosis = _diagnosis()
    candidates = [
        _candidate(
            f"candidate-{index}",
            diagnosis=diagnosis,
            patch=(
                "--- a/src/example.py\n+++ b/src/example.py\n"
                f"@@ -1 +1 @@\n-old\n+new-{index}\n"
            ),
        )
        for index in range(3)
    ]

    with pytest.raises(ContractError, match="candidate budget"):
        select_realization_candidate(
            diagnosis=diagnosis,
            candidates=candidates,
            maximum_candidates=2,
        )


def test_realization_selection_round_trip_rejects_tampered_evidence() -> None:
    diagnosis = _diagnosis()
    selection = select_realization_candidate(
        diagnosis=diagnosis,
        candidates=[
            _candidate(
                "candidate-a",
                diagnosis=diagnosis,
                patch=(
                    "--- a/src/example.py\n+++ b/src/example.py\n"
                    "@@ -1 +1 @@\n-old\n+new\n"
                ),
            )
        ],
        maximum_candidates=2,
    )

    assert RealizationSelection.from_dict(selection.to_dict()) == selection

    tampered = selection.to_dict()
    tampered["maximum_candidates"] = 3
    with pytest.raises(ContractError, match="evidence sha256 mismatch"):
        RealizationSelection.from_dict(tampered)
