from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from evolve.autonomous.output import (
    export_best_harness,
    export_empty_harness,
    verify_best_harness,
)
from evolve.autonomous.verification import VerifiedCampaignClaim
from evolve.contracts import Cohort, ContractViolation, canonical_json, content_sha256
from evolve.proposals import CandidateCompiler, CompiledRevision, CompileSpec

MODEL_IDENTITY = "a" * 64


def _write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _compiled_round(tmp_path: Path) -> Path:
    request = tmp_path / "TEACHER-REQUEST.json"
    response = tmp_path / "TEACHER-RESPONSE.json"
    candidate = {
        "protocol": "skill-harness-v2",
        "prompt_template": "Use the accepted repair harness.",
        "skill_text": "Localize before editing.",
        "operator": {
            "id": "repair-source",
            "kind": "zero-arg",
            "arguments": [],
            "instruction": "Repair only the demonstrated failure.",
        },
        "router": {
            "routes": {
                "task-a": "repair-source",
                "task-b": "repair-source",
                "task-c": "repair-source",
            }
        },
        "memory_policy": None,
        "preconditions": ["feedback-only"],
        "expected_external_effect": {"native": "one gained task"},
        "expected_internal_effect": {"selection": "bounded"},
        "falsification": {"native": "regression rejects"},
        "eval_note": "Use only accepted round claims.",
    }
    _write_json(
        request,
        {
            "model": "deepseek-v4-flash",
            "failure_package": {"selected_tasks": ["task-a"]},
        },
    )
    request_sha256 = hashlib.sha256(request.read_bytes()).hexdigest()
    _write_json(
        response,
        {
            "schema_version": 2,
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "request_sha256": request_sha256,
            "raw_response_sha256": "c" * 64,
            "candidate": candidate,
            "candidate_sha256": content_sha256(candidate),
            "candidate_status": "inactive",
            "auto_activate": False,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            "network_calls": 1,
            "pricing_cny_per_million": {
                "input": 1.0,
                "output": 2.0,
            },
            "estimated_cost_cny": 0.1,
        },
    )
    compiled = CandidateCompiler().compile(
        request_path=request,
        response_path=response,
        compile_spec=CompileSpec(
            candidate_id="candidate-output",
            revision_id="candidate-output-r0000",
            parent_revision_id="empty-harness-v1",
            cohort=Cohort.FEEDBACK,
        ),
        output_root=tmp_path / "round",
    )
    return compiled.root


def _claims() -> tuple[VerifiedCampaignClaim, ...]:
    return (
        VerifiedCampaignClaim(
            task_id="task-b",
            claim_id="claim-b",
            classification="regression",
            grade="E2",
            counterfactual_pair_sha256="1" * 64,
            counterfactual_receipt_ids=tuple(f"receipt-b-{index}" for index in range(6)),
        ),
        VerifiedCampaignClaim(
            task_id="task-a",
            claim_id="claim-a",
            classification="gain",
            grade="E2",
            counterfactual_pair_sha256="2" * 64,
            counterfactual_receipt_ids=tuple(f"receipt-a-{index}" for index in range(6)),
        ),
        VerifiedCampaignClaim(
            task_id="task-c",
            claim_id="claim-c",
            classification="neutral",
            grade="E2",
            counterfactual_pair_sha256="3" * 64,
            counterfactual_receipt_ids=tuple(f"receipt-c-{index}" for index in range(6)),
        ),
    )


def _accepted_round(compiled_root: Path) -> dict[str, object]:
    compiled = CompiledRevision.load(compiled_root)
    return {
        "accepted_as_best": True,
        "candidate_id": "candidate-output",
        "candidate_revision_id": "candidate-output-r0000",
        "compiled_bundle_sha256": compiled.bundle_sha256,
        "claims": [asdict(claim) for claim in _claims()],
    }


def test_empty_best_harness_rejects_nullable_list_fields(tmp_path: Path) -> None:
    path = export_empty_harness(
        output_root=tmp_path,
        model_identity_sha256=MODEL_IDENTITY,
    )
    harness = json.loads(path.read_text())
    harness["source_claim_ids"] = None
    _write_json(path, harness)

    with pytest.raises(ContractViolation, match="empty BEST-HARNESS"):
        verify_best_harness(
            path,
            expected_model_identity_sha256=MODEL_IDENTITY,
        )


def test_empty_best_harness_accepts_empty_round_projection(tmp_path: Path) -> None:
    path = export_empty_harness(
        output_root=tmp_path,
        model_identity_sha256=MODEL_IDENTITY,
    )
    harness = json.loads(path.read_text())

    loaded = verify_best_harness(
        path,
        expected_model_identity_sha256=MODEL_IDENTITY,
        accepted_round={
            "accepted_as_best": False,
            "candidate_id": None,
            "candidate_revision_id": "empty-harness-v1",
            "compiled_bundle_sha256": harness["compiled_bundle_sha256"],
            "claims": [],
        },
    )

    assert loaded is None


def test_compiled_best_harness_must_match_accepted_round_projection(
    tmp_path: Path,
) -> None:
    compiled_root = _compiled_round(tmp_path)
    compiled = CompiledRevision.load(compiled_root)
    best_path = export_best_harness(
        output_root=tmp_path / "output",
        round_root=compiled_root,
        model_identity_sha256=MODEL_IDENTITY,
        candidate_id="candidate-output",
        candidate_revision_id="candidate-output-r0000",
        bundle_sha256=compiled.bundle_sha256,
        claims=_claims(),
    )
    harness = json.loads(best_path.read_text())
    harness["supported_task_signatures"] = ["task-a", "task-b"]
    _write_json(best_path, harness)

    with pytest.raises(ContractViolation, match="authoritative round"):
        verify_best_harness(
            best_path,
            expected_model_identity_sha256=MODEL_IDENTITY,
            accepted_round=_accepted_round(compiled_root),
        )


def test_compiled_best_harness_verifier_returns_loaded_revision_for_valid_projection(
    tmp_path: Path,
) -> None:
    compiled_root = _compiled_round(tmp_path)
    compiled = CompiledRevision.load(compiled_root)
    best_path = export_best_harness(
        output_root=tmp_path / "output",
        round_root=compiled_root,
        model_identity_sha256=MODEL_IDENTITY,
        candidate_id="candidate-output",
        candidate_revision_id="candidate-output-r0000",
        bundle_sha256=compiled.bundle_sha256,
        claims=_claims(),
    )

    loaded = verify_best_harness(
        best_path,
        expected_model_identity_sha256=MODEL_IDENTITY,
        accepted_round=_accepted_round(compiled_root),
    )

    assert loaded is not None
    assert loaded.change_set.revision_id == "candidate-output-r0000"


def test_compiled_best_harness_rejects_duplicate_authoritative_claims(
    tmp_path: Path,
) -> None:
    compiled_root = _compiled_round(tmp_path)
    compiled = CompiledRevision.load(compiled_root)
    best_path = export_best_harness(
        output_root=tmp_path / "output",
        round_root=compiled_root,
        model_identity_sha256=MODEL_IDENTITY,
        candidate_id="candidate-output",
        candidate_revision_id="candidate-output-r0000",
        bundle_sha256=compiled.bundle_sha256,
        claims=_claims(),
    )
    accepted = _accepted_round(compiled_root)
    claims = accepted["claims"]
    assert isinstance(claims, list)
    duplicate = dict(claims[0])
    duplicate["claim_id"] = "claim-duplicate"
    claims.append(duplicate)

    with pytest.raises(ContractViolation, match="non-empty and unique"):
        verify_best_harness(
            best_path,
            expected_model_identity_sha256=MODEL_IDENTITY,
            accepted_round=accepted,
        )


def test_compiled_best_harness_rejects_duplicate_authoritative_claim_ids(
    tmp_path: Path,
) -> None:
    compiled_root = _compiled_round(tmp_path)
    compiled = CompiledRevision.load(compiled_root)
    best_path = export_best_harness(
        output_root=tmp_path / "output",
        round_root=compiled_root,
        model_identity_sha256=MODEL_IDENTITY,
        candidate_id="candidate-output",
        candidate_revision_id="candidate-output-r0000",
        bundle_sha256=compiled.bundle_sha256,
        claims=_claims(),
    )
    accepted = _accepted_round(compiled_root)
    claims = accepted["claims"]
    assert isinstance(claims, list)
    claims[1]["claim_id"] = claims[0]["claim_id"]

    with pytest.raises(ContractViolation, match="non-empty and unique"):
        verify_best_harness(
            best_path,
            expected_model_identity_sha256=MODEL_IDENTITY,
            accepted_round=accepted,
        )
