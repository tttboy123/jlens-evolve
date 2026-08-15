"""AgentProgram fixture entry point and explicit profile dispatch."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from evolve.agent_program import (
    AgentProgramRevision,
    DeterministicFixtureAgentProgramTransport,
    SearchParentLog,
    TournamentAuthority,
)
from evolve.contracts import (
    Authorization,
    Cohort,
    ContractViolation,
    ExecutionLimits,
    ModelIdentity,
    TaskRevision,
    canonical_json,
)
from evolve.evidence import ReceiptStore
from evolve.kernel import CampaignController, CheckpointManager
from evolve.observers import (
    CostObserver,
    ExternalTraceObserver,
    NativeOutcomeObserver,
    ObserverHub,
)
from evolve.registry import AgentProgramRecord, AgentProgramRegistry
from evolve.reporting import AuditVerifier
from evolve.runtime import ExecutionRuntime
from evolve.strategies import AgentProgramSearchStrategy, StrategyContext

from .agent_program_live import run_agent_program_live_campaign
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


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractViolation(f"{name} must be an object")
    return value


class _FixtureWorkspaceManager:
    def materialize(self, plan):
        return {
            "workspace_id": f"fixture-{plan.plan_id}",
            "task_revision_id": plan.task.revision_id,
            "task_source_sha256": plan.task.source_sha256,
            "execution_scope": "fixture",
        }


class _FixtureNativeEvaluator:
    evaluator_id = "fixture-agent-program-native-v1"

    def evaluate(self, plan, workspace, model_output):
        if workspace.get("execution_scope") != "fixture":
            raise ContractViolation("AgentProgram fixture workspace scope drift")
        return {
            "resolved": False,
            "native_valid": True,
            "execution_scope": "fixture",
            "prediction_sha256": model_output["prediction_sha256"],
            "native_gain_claimed": False,
        }


def run_agent_program_fixture_campaign(
    *, config_path: Path, output_root: Path
) -> Mapping[str, Any]:
    """Execute a complete local tournament without claiming native improvement."""

    config_path = config_path.resolve()
    output_root = output_root.resolve()
    try:
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractViolation("AgentProgram campaign config is unreadable") from error
    fields = {
        "schema_version",
        "campaign_id",
        "tournament_id",
        "execution_profile",
        "program_id",
        "parent_revision_root",
        "candidate_revision_roots",
        "generation_config",
        "task",
    }
    if not isinstance(config, Mapping) or set(config) != fields:
        raise ContractViolation("AgentProgram campaign config fields are invalid")
    if config["schema_version"] != 1 or config["execution_profile"] != "fixture":
        raise ContractViolation("only AgentProgram fixture campaigns are supported")
    candidates_value = config["candidate_revision_roots"]
    if not isinstance(candidates_value, list) or not candidates_value:
        raise ContractViolation("AgentProgram candidates must be a non-empty list")
    roots = [
        Path(str(config["parent_revision_root"])).expanduser().resolve(),
        *(Path(str(value)).expanduser().resolve() for value in candidates_value),
    ]
    revisions = tuple(AgentProgramRevision.load(root) for root in roots)
    if len({revision.revision_id for revision in revisions}) != len(revisions):
        raise ContractViolation("AgentProgram tournament revisions are duplicated")
    if {revision.program_id for revision in revisions} != {str(config["program_id"])}:
        raise ContractViolation("AgentProgram campaign program identity mismatch")
    parent = revisions[0]
    if any(
        revision.parent_revision_id != parent.revision_id
        for revision in revisions[1:]
    ):
        raise ContractViolation("AgentProgram candidate parent lineage mismatch")

    task_data = _mapping(config["task"], "task")
    if set(task_data) != {
        "task_id",
        "revision_id",
        "project",
        "cohort",
        "source_sha256",
        "evaluator_id",
    }:
        raise ContractViolation("AgentProgram task fields are invalid")
    task = TaskRevision(
        task_id=str(task_data["task_id"]),
        revision_id=str(task_data["revision_id"]),
        project=str(task_data["project"]),
        cohort=Cohort(str(task_data["cohort"])),
        source_sha256=str(task_data["source_sha256"]),
        evaluator_id=str(task_data["evaluator_id"]),
    )
    if (
        task.cohort is not Cohort.FEEDBACK
        or task.evaluator_id != _FixtureNativeEvaluator.evaluator_id
    ):
        raise ContractViolation("AgentProgram fixture campaign is feedback-only")
    generation = _mapping(config["generation_config"], "generation_config")
    output_root.mkdir(parents=True, exist_ok=True)
    frozen_config = output_root / "AGENT-PROGRAM-CONFIG.json"
    if frozen_config.exists() and frozen_config.read_bytes() != config_bytes:
        raise ContractViolation("AgentProgram campaign replay config drift")
    if not frozen_config.exists():
        _atomic_write(frozen_config, config_bytes)
    program_registry = AgentProgramRegistry(
        output_root / "registries/agent-programs.jsonl"
    )
    for revision in revisions:
        program_registry.append(
            AgentProgramRecord(
                program_id=revision.program_id,
                revision_id=revision.revision_id,
                parent_revision_id=revision.parent_revision_id,
                capability_revision_ids=revision.capability_revision_ids,
                artifact_sha256=revision.bundle_sha256,
                active=False,
            )
        )

    campaign_id = str(config["campaign_id"])
    authorization = Authorization(
        authorization_id=f"auth-{_sha256(config_bytes)[:24]}",
        campaign_id=campaign_id,
        allowed_cohorts=(Cohort.FEEDBACK,),
        max_cost_cny=0,
        max_model_calls=len(revisions),
        expires_at=datetime(2100, 1, 1, tzinfo=UTC),
        remote_calls_allowed=False,
    )
    checkpoints = CheckpointManager(output_root / "checkpoints")
    controller = (
        CampaignController.from_checkpoint(
            campaign_id=campaign_id,
            authorization=authorization,
            checkpoint_manager=checkpoints,
            now=datetime.now(UTC),
        )
        if checkpoints.path_for(campaign_id).is_file()
        else CampaignController.create(
            campaign_id=campaign_id,
            authorization=authorization,
            checkpoint_manager=checkpoints,
            now=datetime.now(UTC),
        )
    )
    revision_roots = {
        revision.revision_id: str(revision.root) for revision in revisions
    }
    context = StrategyContext(
        campaign_id=campaign_id,
        task=task,
        model=ModelIdentity(
            provider="fixture",
            model="deterministic-agent-program",
            revision="v1",
        ),
        context_policy_id="fixture-context-v1",
        tool_policy_id="fixture-tools-v1",
        observer_policy_ids=("external-trace-v1", "native-v1", "cost-v1"),
        limits=ExecutionLimits(max_tokens=0, max_seconds=60, max_cost_cny=0),
        inputs={
            "parent_revision_id": parent.revision_id,
            "candidate_revision_ids": tuple(
                revision.revision_id for revision in revisions[1:]
            ),
            "tournament_id": str(config["tournament_id"]),
            "generation_config": dict(generation),
            "execution_profile": "fixture",
            "revision_roots": revision_roots,
        },
    )
    strategy = AgentProgramSearchStrategy(execution_profile="fixture")
    store = ReceiptStore(output_root / "receipt-store")
    runtime = ExecutionRuntime(
        model_transport=DeterministicFixtureAgentProgramTransport(revision_roots),
        workspace_manager=_FixtureWorkspaceManager(),
        native_evaluator=_FixtureNativeEvaluator(),
        observer_hub=ObserverHub(
            (ExternalTraceObserver(), NativeOutcomeObserver(), CostObserver())
        ),
        receipt_sink=store,
    )
    campaign = CampaignRunner(runtime=runtime, controller=controller).run(
        CampaignSpec(
            campaign_id=campaign_id,
            contexts=(context,),
            authorization=authorization,
        ),
        strategy,
    )
    decision = TournamentAuthority().decide(
        plans=campaign.plans, executions=campaign.executions
    )
    parent_log = SearchParentLog(
        output_root / "SEARCH-PARENT.jsonl", program_id=str(config["program_id"])
    )
    parent_log.append(decision)
    action = strategy.next_action(context, (), decision=decision)
    report = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "status": str(campaign.status),
        "strategy_id": strategy.strategy_id,
        "execution_scope": "fixture",
        "tournament_decision_id": decision.decision_id,
        "tournament_decision_sha256": decision.decision_sha256,
        "participant_revision_ids": list(decision.participant_revision_ids),
        "selected_parent_revision_id": parent_log.current_parent_revision_id(),
        "search_parent_advanced": decision.advanced,
        "advisory_action": action.action,
        "receipt_ids": [receipt.receipt_id for receipt in campaign.receipts],
        "claims": [],
        "native_gain_claimed": False,
        "promotion_eligible": False,
        "capability_active": False,
        "holdout_opened": False,
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


def run_agent_program_campaign(
    *, config_path: Path, output_root: Path
) -> Mapping[str, Any]:
    """Dispatch only the explicitly selected AgentProgram execution profile."""

    try:
        raw = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractViolation("AgentProgram campaign config is unreadable") from error
    if not isinstance(raw, Mapping):
        raise ContractViolation("AgentProgram campaign config must be an object")
    profile = raw.get("execution_profile")
    if profile == "fixture":
        return run_agent_program_fixture_campaign(
            config_path=config_path,
            output_root=output_root,
        )
    if profile == "live":
        return run_agent_program_live_campaign(
            config_path=config_path,
            output_root=output_root,
        )
    raise ContractViolation(
        "AgentProgram campaign config fields are invalid: execution_profile unsupported"
    )


__all__ = [
    "run_agent_program_campaign",
    "run_agent_program_fixture_campaign",
    "run_agent_program_live_campaign",
]
