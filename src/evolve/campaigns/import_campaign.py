"""Read-only legacy evidence import through the unified Campaign authority."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from evolve.contracts import (
    Authorization,
    Cohort,
    ContractViolation,
    ExecutionLimits,
    ExecutionPlan,
    ModelIdentity,
    Receipt,
    TaskRevision,
    canonical_json,
)
from evolve.evidence import ReceiptStore
from evolve.kernel import CampaignController, CheckpointManager
from evolve.reporting import AuditVerifier
from evolve.runtime import ExecutionResult
from evolve.strategies import LegacyImportStrategy, StrategyContext

from .runner import CampaignRunner, CampaignSpec


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
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


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractViolation(f"{name} must be an object")
    return value


class LegacyImportRuntime:
    """RuntimeEntry that preserves one immutable artifact without interpreting it."""

    remote = False
    reserved_model_calls = 0

    def __init__(self, *, artifact: bytes, store: ReceiptStore) -> None:
        self._artifact = artifact
        self._store = store

    def execute(
        self,
        plan: ExecutionPlan,
        authorization: Authorization,
        *,
        mechanism_prediction: object | None = None,
    ) -> ExecutionResult:
        if mechanism_prediction is not None:
            raise ContractViolation("legacy import cannot consume mechanism predictions")
        authorization.assert_allows(
            cohort=plan.task.cohort,
            reserved_cost_cny=0,
            reserved_model_calls=0,
            remote=False,
        )
        if plan.arm != "legacy-replay" or plan.task.cohort is not Cohort.FEEDBACK:
            raise ContractViolation("legacy import is feedback-only read-only replay")
        expected = str(plan.metadata["legacy_artifact_sha256"])
        if _sha256(self._artifact) != expected:
            raise ContractViolation("legacy artifact SHA-256 mismatch")
        existing = self._store.receipts_for(plan.plan_id)
        if existing:
            if len(existing) != 1 or existing[0].kind != "legacy_import":
                raise ContractViolation("legacy import resume receipt set is invalid")
            if existing[0].artifact_sha256 != expected:
                raise ContractViolation("legacy import resume artifact drift")
            return ExecutionResult(
                status="completed", receipts=existing, evidence=(), replayed=True
            )
        identity = canonical_json(
            {
                "plan_sha256": plan.content_sha256,
                "artifact_sha256": expected,
                "provenance_uri": plan.metadata["provenance_uri"],
            }
        ).encode("utf-8")
        receipt = Receipt(
            receipt_id=f"legacy-{_sha256(identity)[:32]}",
            campaign_id=plan.campaign_id,
            plan_id=plan.plan_id,
            sequence=1,
            kind="legacy_import",
            created_at=datetime.now(UTC).isoformat(),
            payload={
                "imported_revision_id": plan.candidate_revision_id,
                "legacy_artifact_sha256": expected,
                "provenance_uri": plan.metadata["provenance_uri"],
                "compatibility_mode": "read-only",
                "claims_created": 0,
                "candidates_created": 0,
            },
            artifact_sha256=expected,
        )
        stored = self._store.append(receipt, self._artifact)
        return ExecutionResult(
            status="completed", receipts=(stored,), evidence=(), replayed=False
        )


def run_legacy_import_campaign(
    *, config_path: Path, output_root: Path
) -> Mapping[str, Any]:
    """Run a strict legacy import and project only authority-backed facts."""

    config_path = config_path.resolve()
    output_root = output_root.resolve()
    try:
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractViolation("legacy import config is unreadable") from error
    required = {
        "schema_version",
        "campaign_id",
        "imported_revision_id",
        "artifact_path",
        "artifact_sha256",
        "provenance_uri",
        "task",
        "model",
    }
    if not isinstance(config, Mapping) or set(config) != required:
        raise ContractViolation("legacy import campaign config fields are invalid")
    if config["schema_version"] != 1:
        raise ContractViolation("unsupported legacy import config schema")
    task_data = _object(config["task"], "task")
    if set(task_data) != {
        "task_id",
        "revision_id",
        "project",
        "cohort",
        "source_sha256",
        "evaluator_id",
    }:
        raise ContractViolation("legacy import task fields are invalid")
    model_data = _object(config["model"], "model")
    if set(model_data) != {"provider", "model", "revision"}:
        raise ContractViolation("legacy import model fields are invalid")
    artifact_path = Path(str(config["artifact_path"])).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = config_path.parent / artifact_path
    artifact = artifact_path.resolve().read_bytes()
    artifact_sha256 = str(config["artifact_sha256"])
    if _sha256(artifact) != artifact_sha256:
        raise ContractViolation("legacy artifact SHA-256 mismatch")

    output_root.mkdir(parents=True, exist_ok=True)
    frozen_config = output_root / "IMPORT-CONFIG.json"
    if frozen_config.exists() and frozen_config.read_bytes() != config_bytes:
        raise ContractViolation("legacy import replay config drift")
    if not frozen_config.exists():
        _atomic_write(frozen_config, config_bytes)

    campaign_id = str(config["campaign_id"])
    task = TaskRevision(
        task_id=str(task_data["task_id"]),
        revision_id=str(task_data["revision_id"]),
        project=str(task_data["project"]),
        cohort=Cohort(str(task_data["cohort"])),
        source_sha256=str(task_data["source_sha256"]),
        evaluator_id=str(task_data["evaluator_id"]),
    )
    model = ModelIdentity(
        provider=str(model_data["provider"]),
        model=str(model_data["model"]),
        revision=str(model_data["revision"]),
    )
    authorization = Authorization(
        authorization_id=f"auth-{_sha256(config_bytes)[:24]}",
        campaign_id=campaign_id,
        allowed_cohorts=(Cohort.FEEDBACK,),
        max_cost_cny=0,
        max_model_calls=0,
        expires_at=datetime(2100, 1, 1, tzinfo=UTC),
        remote_calls_allowed=False,
    )
    checkpoints = CheckpointManager(output_root / "checkpoints")
    checkpoint_path = checkpoints.path_for(campaign_id)
    controller = (
        CampaignController.from_checkpoint(
            campaign_id=campaign_id,
            authorization=authorization,
            checkpoint_manager=checkpoints,
            now=datetime.now(UTC),
        )
        if checkpoint_path.exists()
        else CampaignController.create(
            campaign_id=campaign_id,
            authorization=authorization,
            checkpoint_manager=checkpoints,
            now=datetime.now(UTC),
        )
    )
    context = StrategyContext(
        campaign_id=campaign_id,
        task=task,
        model=model,
        context_policy_id="legacy-read-only-v1",
        tool_policy_id="no-tools-v1",
        observer_policy_ids=("receipt-integrity-v1",),
        limits=ExecutionLimits(max_tokens=0, max_seconds=60, max_cost_cny=0),
        inputs={
            "imported_revision_id": str(config["imported_revision_id"]),
            "legacy_artifact_sha256": artifact_sha256,
            "provenance_uri": str(config["provenance_uri"]),
        },
    )
    store = ReceiptStore(output_root / "receipt-store")
    result = CampaignRunner(
        runtime=LegacyImportRuntime(artifact=artifact, store=store),
        controller=controller,
    ).run(
        CampaignSpec(
            campaign_id=campaign_id,
            contexts=(context,),
            authorization=authorization,
        ),
        LegacyImportStrategy(),
    )
    report = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "status": str(result.status),
        "strategy_id": LegacyImportStrategy.strategy_id,
        "plan_ids": [plan.plan_id for plan in result.plans],
        "receipt_ids": [receipt.receipt_id for receipt in result.receipts],
        "claim_ids": [],
        "candidate_revision_ids": [],
        "capability_ids": [],
        "snapshot": asdict(result.snapshot) if result.snapshot is not None else None,
        "compatibility_mode": "read-only",
    }
    _atomic_write(
        output_root / "CAMPAIGN-RESULT.json",
        (canonical_json(report) + "\n").encode("utf-8"),
    )
    manifest_path = output_root / "EVIDENCE-MANIFEST.json"
    entries = [
        {
            "path": str(path.relative_to(output_root)),
            "sha256": _sha256(path.read_bytes()),
        }
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
        and path != manifest_path
        and not path.name.endswith(".lock")
    ]
    _atomic_write(
        manifest_path,
        (canonical_json({"schema_version": 1, "entries": entries}) + "\n").encode(),
    )
    AuditVerifier().verify_manifest(manifest_path, root=output_root)
    return report


__all__ = ["LegacyImportRuntime", "run_legacy_import_campaign"]
