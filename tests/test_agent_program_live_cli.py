from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from evolve.agent_program import AgentProgramRevision
from evolve.contracts import (
    Claim,
    ClaimClassification,
    ClaimGrade,
    EvidenceEnvelope,
    Receipt,
    canonical_json,
    content_sha256,
)
from evolve.evidence import EvidenceGraph, ReceiptStore
from evolve.reporting import AuditVerifier


def _revision(
    root: Path,
    *,
    revision_id: str,
    parent_revision_id: str | None,
    patch: str,
) -> AgentProgramRevision:
    return AgentProgramRevision.freeze(
        root,
        program_id="public-live-repair-agent",
        revision_id=revision_id,
        parent_revision_id=parent_revision_id,
        program_prompt="Produce the locally declared feedback patch.",
        context={"patch": patch},
        tool_policy=("emit_patch",),
        capability_revision_ids=("local-declarative-patch-v1",),
    )


def _run_cli(config_path: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (
            sys.executable,
            "-m",
            "evolve",
            "campaign",
            "run",
            "--strategy",
            "agent-program",
            "--config",
            str(config_path),
            "--output",
            str(output),
        ),
        check=False,
        capture_output=True,
        text=True,
    )


def _evaluator_id(expected_patch_sha256: str) -> str:
    return (
        "local-patch-match-native-v1@sha256:"
        + content_sha256({"expected_patch_sha256": expected_patch_sha256})
    )


def _valid_live_input(
    tmp_path: Path,
) -> tuple[Path, Path, Path, AgentProgramRevision, AgentProgramRevision]:
    parent_patch = "diff --git a/result.txt b/result.txt\n+parent\n"
    candidate_patch = "diff --git a/result.txt b/result.txt\n+candidate\n"
    parent = _revision(
        tmp_path / "parent",
        revision_id="program-r1",
        parent_revision_id=None,
        patch=parent_patch,
    )
    candidate = _revision(
        tmp_path / "candidate",
        revision_id="program-r2",
        parent_revision_id=parent.revision_id,
        patch=candidate_patch,
    )
    source = tmp_path / "feedback-source.txt"
    source.write_text("public live AgentProgram input\n", encoding="utf-8")
    expected_patch_sha256 = hashlib.sha256(
        candidate_patch.encode("utf-8")
    ).hexdigest()
    config = {
        "schema_version": 1,
        "campaign_id": "public-live-agent-program",
        "tournament_id": "public-live-tournament-1",
        "execution_profile": "live",
        "program_id": parent.program_id,
        "product_adapter_id": "local-declarative-agent-program-v1",
        "parent_revision_root": str(parent.root),
        "candidate_revision_roots": [str(candidate.root)],
        "generation_config": {"temperature": 0},
        "task": {
            "task_id": "public-feedback-task",
            "revision_id": "public-feedback-task-r1",
            "project": "public-local-project",
            "cohort": "feedback",
            "source_uri": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "evaluator_id": _evaluator_id(expected_patch_sha256),
            "expected_patch_sha256": expected_patch_sha256,
        },
    }
    config_path = tmp_path / "agent-program-live.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, tmp_path / "run", source, parent, candidate


def _reseal(output: Path) -> None:
    manifest = output / "EVIDENCE-MANIFEST.json"
    entries = [
        {
            "path": str(path.relative_to(output)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path != manifest
        and not path.name.endswith(".lock")
    ]
    manifest.write_text(
        canonical_json({"schema_version": 1, "entries": entries}) + "\n",
        encoding="utf-8",
    )


def test_public_cli_runs_live_complete_agent_programs_through_runtime_authority(
    tmp_path: Path,
) -> None:
    config_path, output, _, parent, candidate = _valid_live_input(tmp_path)

    completed = _run_cli(config_path, output)

    assert completed.returncode == 0, completed.stderr
    result = json.loads((output / "CAMPAIGN-RESULT.json").read_text())
    assert result["execution_scope"] == "live"
    assert result["product_adapter_id"] == "local-declarative-agent-program-v1"
    assert result["selected_parent_revision_id"] == candidate.revision_id
    assert result["search_parent_advanced"] is True
    assert result["advisory_action"] == "advance-search-parent"
    assert result["initial_execution_replayed"] == [False, False]
    assert result["authority_execution_replayed"] == [True, True]
    assert result["native_gain_claimed"] is True
    assert result["promotion_eligible"] is False
    assert result["capability_active"] is False
    assert result["holdout_opened"] is False
    assert {row["grade"] for row in result["claims"]} == {"E1"}
    assert {row["candidate_id"] for row in result["claims"]} == {
        parent.revision_id,
        candidate.revision_id,
    }

    receipts = ReceiptStore(output / "receipt-store").list_receipts()
    model_receipts = tuple(row for row in receipts if row.kind == "model")
    native_receipts = tuple(row for row in receipts if row.kind == "native_evaluation")
    assert len(model_receipts) == len(native_receipts) == 2
    assert all(row.payload["execution_scope"] == "live" for row in model_receipts)
    assert {row.payload["revision_id"] for row in model_receipts} == {
        parent.revision_id,
        candidate.revision_id,
    }
    assert all("fixture_score" not in row.payload for row in model_receipts)
    assert len(EvidenceGraph.rebuild(output / "evidence-graph", ReceiptStore(output / "receipt-store")).latest_claims()) == 2

    registry = tuple(
        json.loads(line)
        for line in (output / "registries/agent-programs.jsonl")
        .read_text()
        .splitlines()
    )
    assert [row["revision_id"] for row in registry] == ["program-r1", "program-r2"]
    assert all(row["active"] is False for row in registry)
    parent_events = tuple(
        json.loads(line)
        for line in (output / "SEARCH-PARENT.jsonl").read_text().splitlines()
    )
    assert len(parent_events) == 1
    assert parent_events[0]["execution_scope"] == "live"
    assert parent_events[0]["selected_revision_id"] == candidate.revision_id
    assert len(parent_events[0]["decision_sha256"]) == 64
    assert AuditVerifier().verify_manifest(
        output / "EVIDENCE-MANIFEST.json", root=output
    ) > 0

    sealed_before = {
        name: (output / name).read_bytes()
        for name in (
            "CAMPAIGN-RESULT.json",
            "EVIDENCE-MANIFEST.json",
            "SEARCH-PARENT.jsonl",
            "receipt-store/receipts.jsonl",
            "registries/agent-programs.jsonl",
        )
    }
    replayed = _run_cli(config_path, output)
    assert replayed.returncode == 0, replayed.stderr
    assert {
        name: (output / name).read_bytes() for name in sealed_before
    } == sealed_before


@pytest.mark.parametrize(
    ("adapter_id", "cohort", "message"),
    (
        ("os.system:sh", "feedback", "allowlisted"),
        ("local-declarative-agent-program-v1", "holdout", "feedback-only"),
    ),
)
def test_public_live_cli_rejects_unsafe_adapter_or_holdout_scope(
    tmp_path: Path, adapter_id: str, cohort: str, message: str
) -> None:
    parent = _revision(
        tmp_path / "parent",
        revision_id="program-r1",
        parent_revision_id=None,
        patch="parent patch",
    )
    candidate = _revision(
        tmp_path / "candidate",
        revision_id="program-r2",
        parent_revision_id=parent.revision_id,
        patch="candidate patch",
    )
    source = tmp_path / "source.txt"
    source.write_text("source\n", encoding="utf-8")
    expected_patch_sha256 = "a" * 64
    config = {
        "schema_version": 1,
        "campaign_id": "rejected-live-adapter",
        "tournament_id": "rejected-live-adapter-tournament",
        "execution_profile": "live",
        "program_id": parent.program_id,
        "product_adapter_id": adapter_id,
        "parent_revision_root": str(parent.root),
        "candidate_revision_roots": [str(candidate.root)],
        "generation_config": {},
        "task": {
            "task_id": "task",
            "revision_id": "task-r1",
            "project": "project",
            "cohort": cohort,
            "source_uri": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "evaluator_id": _evaluator_id(expected_patch_sha256),
            "expected_patch_sha256": expected_patch_sha256,
        },
    }
    config_path = tmp_path / "rejected.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "rejected-run"

    completed = _run_cli(config_path, output)

    assert completed.returncode != 0
    assert message in completed.stderr
    assert not (output / "receipt-store/receipts.jsonl").exists()


def test_public_live_cli_rejects_resealed_result_projection_forgery(
    tmp_path: Path,
) -> None:
    config_path, output, _, parent, _ = _valid_live_input(tmp_path)
    assert _run_cli(config_path, output).returncode == 0
    result_path = output / "CAMPAIGN-RESULT.json"
    forged = json.loads(result_path.read_text())
    forged["selected_parent_revision_id"] = parent.revision_id
    forged["search_parent_advanced"] = False
    forged["native_gain_claimed"] = False
    forged["receipt_ids"] = []
    result_path.write_text(canonical_json(forged) + "\n", encoding="utf-8")
    _reseal(output)

    replayed = _run_cli(config_path, output)

    assert replayed.returncode != 0
    assert "result authority drift" in replayed.stderr


def test_public_live_cli_rejects_external_source_drift_on_sealed_replay(
    tmp_path: Path,
) -> None:
    config_path, output, source, _, _ = _valid_live_input(tmp_path)
    assert _run_cli(config_path, output).returncode == 0
    source.write_text("mutated after seal\n", encoding="utf-8")

    replayed = _run_cli(config_path, output)

    assert replayed.returncode != 0
    assert "source identity drift" in replayed.stderr


def test_public_live_cli_rederives_claim_classification_from_native_receipt(
    tmp_path: Path,
) -> None:
    config_path, output, _, _, _ = _valid_live_input(tmp_path)
    assert _run_cli(config_path, output).returncode == 0
    claims_path = output / "evidence-graph/claims.jsonl"
    rows = [json.loads(line) for line in claims_path.read_text().splitlines()]
    rows[0]["value"]["classification"] = "gain"
    value = rows[0]["value"]
    rows[0]["content_sha256"] = Claim(
        claim_id=value["claim_id"],
        candidate_id=value["candidate_id"],
        grade=ClaimGrade(value["grade"]),
        classification=ClaimClassification(value["classification"]),
        evidence_ids=tuple(value["evidence_ids"]),
        rationale=value["rationale"],
        supersedes_claim_id=value["supersedes_claim_id"],
        counterfactual_pair_sha256=value.get("counterfactual_pair_sha256"),
        counterfactual_receipt_ids=tuple(
            value.get("counterfactual_receipt_ids", ())
        ),
    ).content_sha256
    claims_path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    _reseal(output)

    replayed = _run_cli(config_path, output)

    assert replayed.returncode != 0
    assert "Claim authority drift" in replayed.stderr


def test_public_live_cli_recomputes_native_outcome_on_coordinated_reseal(
    tmp_path: Path,
) -> None:
    config_path, output, _, _, _ = _valid_live_input(tmp_path)
    assert _run_cli(config_path, output).returncode == 0
    receipts_path = output / "receipt-store/receipts.jsonl"
    receipt_rows = [json.loads(line) for line in receipts_path.read_text().splitlines()]
    native_row = next(
        row
        for row in receipt_rows
        if row["receipt"]["kind"] == "native_evaluation"
        and row["receipt"]["payload"]["resolved"] is False
    )
    native_row["receipt"]["payload"]["resolved"] = True
    artifact = canonical_json(native_row["receipt"]["payload"]).encode("utf-8")
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    native_row["receipt"]["artifact_sha256"] = artifact_sha256
    (output / "receipt-store/artifacts/sha256" / artifact_sha256).write_bytes(
        artifact
    )
    forged_receipt = Receipt(**native_row["receipt"])
    native_row["receipt_sha256"] = forged_receipt.content_sha256
    receipts_path.write_text(
        "".join(canonical_json(row) + "\n" for row in receipt_rows),
        encoding="utf-8",
    )

    evidence_path = output / "evidence-graph/evidence.jsonl"
    evidence_rows = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    evidence_row = next(
        row
        for row in evidence_rows
        if row["value"]["observer_id"] == "native-v1"
        and forged_receipt.receipt_id in row["value"]["receipt_ids"]
    )
    evidence_row["value"]["payload"] = {
        "campaign_id": forged_receipt.campaign_id,
        "plan_id": forged_receipt.plan_id,
        "receipt_kind": forged_receipt.kind,
        **forged_receipt.payload,
    }
    evidence_row["value"]["artifact_sha256"] = artifact_sha256
    forged_evidence = EvidenceEnvelope(
        evidence_id=evidence_row["value"]["evidence_id"],
        receipt_ids=tuple(evidence_row["value"]["receipt_ids"]),
        observer_id=evidence_row["value"]["observer_id"],
        grade=ClaimGrade(evidence_row["value"]["grade"]),
        payload=evidence_row["value"]["payload"],
        artifact_sha256=artifact_sha256,
    )
    evidence_row["content_sha256"] = forged_evidence.content_sha256
    evidence_path.write_text(
        "".join(canonical_json(row) + "\n" for row in evidence_rows),
        encoding="utf-8",
    )
    _reseal(output)

    replayed = _run_cli(config_path, output)

    assert replayed.returncode != 0
    assert "native receipt drift" in replayed.stderr
