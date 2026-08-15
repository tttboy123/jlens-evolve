"""Authoritative projections for live AgentProgram tournament campaigns."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol

from evolve.contracts import (
    Claim,
    ClaimClassification,
    ClaimGrade,
    ContractViolation,
    canonical_json,
)
from evolve.evidence import EvidenceGraph, ReceiptStore
from evolve.observers import NativeOutcomeObserver
from evolve.reporting import AuditVerifier
from evolve.runtime import native_execution_identity, runtime_receipt_identity


class LiveModelTransport(Protocol):
    def infer(self, plan, workspace: Mapping[str, Any]) -> Mapping[str, Any]: ...


class LiveNativeEvaluator(Protocol):
    def evaluate(
        self, plan, workspace: Mapping[str, Any], model_output: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def verify_live_runtime_projection(
    *,
    plans,
    store: ReceiptStore,
    transport: LiveModelTransport,
    evaluator: LiveNativeEvaluator,
) -> None:
    """Recompute workspace, model, and native receipts from frozen inputs."""

    receipts = store.list_receipts()
    for plan in plans:
        plan_receipts = tuple(
            sorted(
                (receipt for receipt in receipts if receipt.plan_id == plan.plan_id),
                key=lambda receipt: receipt.sequence,
            )
        )
        if tuple(receipt.sequence for receipt in plan_receipts) != tuple(
            range(1, 7)
        ) or tuple(receipt.kind for receipt in plan_receipts) != (
            "workspace",
            "model",
            "external_trace",
            "cost",
            "native_evaluation",
            "execution_terminal",
        ):
            raise ContractViolation("live AgentProgram receipt order drift")
        by_kind = {
            kind: tuple(
                receipt
                for receipt in receipts
                if receipt.plan_id == plan.plan_id and receipt.kind == kind
            )
            for kind in ("workspace", "model", "native_evaluation")
        }
        if any(len(selected) != 1 for selected in by_kind.values()):
            raise ContractViolation(
                "live AgentProgram runtime projection is incomplete"
            )
        workspace_receipt = by_kind["workspace"][0]
        model_receipt = by_kind["model"][0]
        native_receipt = by_kind["native_evaluation"][0]
        source = Path(str(plan.task.source_uri)).resolve()
        expected_workspace = {
            "workspace_id": f"local-file-{plan.plan_id}",
            "task_revision_id": plan.task.revision_id,
            "task_source_sha256": plan.task.source_sha256,
            "source_uri": str(source),
            "execution_scope": "live",
            "plan_sha256": plan.content_sha256,
        }
        if workspace_receipt.payload != expected_workspace:
            raise ContractViolation("live AgentProgram workspace receipt drift")
        _verify_runtime_receipt(
            receipt=workspace_receipt,
            expected_payload=expected_workspace,
            store=store,
        )
        expected_model = dict(transport.infer(plan, expected_workspace))
        expected_model.update(
            {
                "provider": plan.model.provider,
                "model": plan.model.model,
                "revision": plan.model.revision,
            }
        )
        if model_receipt.payload != expected_model:
            raise ContractViolation("live AgentProgram model receipt drift")
        _verify_runtime_receipt(
            receipt=model_receipt,
            expected_payload=expected_model,
            store=store,
        )
        expected_native = dict(
            evaluator.evaluate(plan, expected_workspace, expected_model)
        )
        expected_native.update(native_execution_identity(plan))
        expected_native.update(
            {
                "model_receipt_id": model_receipt.receipt_id,
                "model_artifact_sha256": model_receipt.artifact_sha256,
                "evaluator_error": None,
            }
        )
        if native_receipt.payload != expected_native:
            raise ContractViolation("live AgentProgram native receipt drift")
        _verify_runtime_receipt(
            receipt=native_receipt,
            expected_payload=expected_native,
            store=store,
        )


def _verify_runtime_receipt(
    *, receipt, expected_payload: Mapping[str, Any], store: ReceiptStore
) -> None:
    artifact, artifact_sha256, receipt_id = runtime_receipt_identity(
        campaign_id=receipt.campaign_id,
        plan_id=receipt.plan_id,
        sequence=receipt.sequence,
        kind=receipt.kind,
        payload=expected_payload,
    )
    if (
        receipt.payload != expected_payload
        or receipt.artifact_sha256 != artifact_sha256
        or receipt.receipt_id != receipt_id
        or store.read_artifact(artifact_sha256) != artifact
    ):
        raise ContractViolation("live AgentProgram receipt artifact identity drift")


def project_live_claims(
    *,
    plans,
    graph: EvidenceGraph,
    store: ReceiptStore,
    append: bool = True,
) -> tuple[Claim, ...]:
    """Project the sole E1 Claim identity from native receipts and evidence."""

    evidence = graph.list_evidence()
    receipts = store.list_receipts()
    claims = []
    for plan in plans:
        native_receipts = tuple(
            receipt
            for receipt in receipts
            if receipt.plan_id == plan.plan_id
            and receipt.kind == "native_evaluation"
        )
        if len(native_receipts) != 1:
            raise ContractViolation(
                "live AgentProgram authority requires one native receipt"
            )
        native_receipt = native_receipts[0]
        expected_envelope = NativeOutcomeObserver().observe(native_receipt)
        if expected_envelope is None:
            raise ContractViolation("live AgentProgram native observation is missing")
        selected = tuple(
            row
            for row in evidence
            if row.observer_id == "native-v1"
            and row.payload.get("plan_id") == plan.plan_id
        )
        if len(selected) != 1 or selected[0] != expected_envelope:
            raise ContractViolation(
                "live AgentProgram native evidence projection drift"
            )
        envelope = selected[0]
        if native_receipt.payload.get("evaluator_error") not in (None, ""):
            classification = ClaimClassification.INFRA_FAILURE
        elif native_receipt.payload.get("resolved") is True:
            classification = ClaimClassification.GAIN
        else:
            classification = ClaimClassification.NEUTRAL
        identity = {
            "tournament_participant_revision_id": plan.candidate_revision_id,
            "plan_id": plan.plan_id,
            "evidence_id": envelope.evidence_id,
            "evidence_sha256": envelope.content_sha256,
            "classification": str(classification),
        }
        claim = Claim(
            claim_id=(
                f"claim-{sha256_bytes(canonical_json(identity).encode())[:24]}"
            ),
            candidate_id=plan.candidate_revision_id,
            grade=ClaimGrade.E1,
            classification=classification,
            evidence_ids=(envelope.evidence_id,),
            rationale="allowlisted local native evaluator participant outcome",
            supersedes_claim_id=None,
        )
        if append:
            graph.append_claim(claim)
        claims.append(claim)
    return tuple(claims)


def project_live_decision(
    decision, *, participants: tuple[str, ...]
) -> tuple[str, str]:
    fields = {}
    for item in decision.reason.split(";"):
        if "=" in item:
            name, value = item.split("=", 1)
            fields[name] = value
    decision_sha256 = fields.get("decision")
    winner = fields.get("winner")
    if (
        not isinstance(decision_sha256, str)
        or len(decision_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in decision_sha256
        )
        or winner not in participants
    ):
        raise ContractViolation("live AgentProgram decision projection is invalid")
    expected_action = (
        "advance-search-parent" if winner != participants[0] else "reject-candidates"
    )
    if decision.action != expected_action:
        raise ContractViolation("live AgentProgram decision action drift")
    return decision_sha256, str(winner)


def append_live_search_parent(
    *,
    path: Path,
    program_id: str,
    tournament_id: str,
    parent_revision_id: str,
    selected_revision_id: str,
    decision_sha256: str,
    claim_ids: tuple[str, ...],
) -> None:
    existing: list[dict[str, Any]] = []
    if path.exists():
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ContractViolation(
                    f"live search-parent event is invalid at line {line_number}"
                ) from error
            identity = {
                name: value for name, value in row.items() if name != "event_sha256"
            }
            if (
                row.get("event_sha256")
                != sha256_bytes(canonical_json(identity).encode("utf-8"))
                or row.get("sequence") != line_number
                or row.get("execution_scope") != "live"
                or row.get("program_id") != program_id
            ):
                raise ContractViolation("live search-parent hash chain is invalid")
            expected_previous = existing[-1]["event_sha256"] if existing else None
            if row.get("previous_event_sha256") != expected_previous:
                raise ContractViolation("live search-parent hash chain fork")
            existing.append(row)
    matching = [row for row in existing if row["tournament_id"] == tournament_id]
    if matching:
        if (
            len(matching) == 1
            and matching[0]["decision_sha256"] == decision_sha256
            and matching[0]["selected_revision_id"] == selected_revision_id
        ):
            return
        raise ContractViolation("conflicting live tournament decision replay")
    current = existing[-1]["selected_revision_id"] if existing else parent_revision_id
    if current != parent_revision_id:
        raise ContractViolation("live search-parent fork detected")
    identity = {
        "sequence": len(existing) + 1,
        "program_id": program_id,
        "tournament_id": tournament_id,
        "execution_scope": "live",
        "previous_parent_revision_id": parent_revision_id,
        "selected_revision_id": selected_revision_id,
        "decision_sha256": decision_sha256,
        "claim_ids": list(claim_ids),
        "previous_event_sha256": (
            existing[-1]["event_sha256"] if existing else None
        ),
    }
    row = {
        **identity,
        "event_sha256": sha256_bytes(canonical_json(identity).encode("utf-8")),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, (canonical_json(row) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_live_search_parent(
    *,
    path: Path,
    program_id: str,
    tournament_id: str,
    parent_revision_id: str,
    selected_revision_id: str,
    decision_sha256: str,
    claim_ids: tuple[str, ...],
) -> None:
    if not path.is_file():
        raise ContractViolation("live search-parent authority is missing")
    try:
        rows = tuple(json.loads(line) for line in path.read_text().splitlines())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractViolation("live search-parent authority is unreadable") from error
    if len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ContractViolation("live search-parent authority is ambiguous")
    row = rows[0]
    identity = {name: value for name, value in row.items() if name != "event_sha256"}
    expected = {
        "sequence": 1,
        "program_id": program_id,
        "tournament_id": tournament_id,
        "execution_scope": "live",
        "previous_parent_revision_id": parent_revision_id,
        "selected_revision_id": selected_revision_id,
        "decision_sha256": decision_sha256,
        "claim_ids": list(claim_ids),
        "previous_event_sha256": None,
    }
    if identity != expected or row.get("event_sha256") != sha256_bytes(
        canonical_json(expected).encode("utf-8")
    ):
        raise ContractViolation("live search-parent authority drift")


def build_live_report(
    *,
    campaign_id: str,
    status: object,
    strategy_id: str,
    product_adapter_id: str,
    participants: tuple[str, ...],
    parent_revision_id: str,
    selected_revision_id: str,
    decision_sha256: str,
    decision_action: str,
    receipt_ids: tuple[str, ...],
    claims: tuple[Claim, ...],
    initial_execution_replayed: tuple[bool, ...],
    authority_execution_replayed: tuple[bool, ...],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "status": str(status),
        "strategy_id": strategy_id,
        "execution_scope": "live",
        "product_adapter_id": product_adapter_id,
        "tournament_decision_sha256": decision_sha256,
        "participant_revision_ids": list(participants),
        "selected_parent_revision_id": selected_revision_id,
        "search_parent_advanced": selected_revision_id != parent_revision_id,
        "advisory_action": decision_action,
        "initial_execution_replayed": list(initial_execution_replayed),
        "authority_execution_replayed": list(authority_execution_replayed),
        "receipt_ids": list(receipt_ids),
        "claims": [
            {
                "claim_id": claim.claim_id,
                "candidate_id": claim.candidate_id,
                "grade": str(claim.grade),
                "classification": str(claim.classification),
                "evidence_ids": list(claim.evidence_ids),
            }
            for claim in claims
        ],
        "native_gain_claimed": any(
            claim.classification is ClaimClassification.GAIN for claim in claims
        ),
        "promotion_eligible": False,
        "capability_active": False,
        "holdout_opened": False,
    }


def seal_agent_program_run(output_root: Path) -> None:
    manifest_path = output_root / "EVIDENCE-MANIFEST.json"
    entries = [
        {
            "path": str(path.relative_to(output_root)),
            "sha256": sha256_bytes(path.read_bytes()),
        }
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
        and path != manifest_path
        and not path.name.endswith(".lock")
    ]
    atomic_write(
        manifest_path,
        (canonical_json({"schema_version": 1, "entries": entries}) + "\n").encode(),
    )
    AuditVerifier().verify_manifest(manifest_path, root=output_root)


__all__ = [
    "append_live_search_parent",
    "atomic_write",
    "build_live_report",
    "project_live_claims",
    "project_live_decision",
    "seal_agent_program_run",
    "sha256_bytes",
    "verify_live_runtime_projection",
    "verify_live_search_parent",
]
