"""Live AgentProgram adapters and CampaignRunner orchestration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from evolve.agent_program import AgentProgramRevision
from evolve.contracts import (
    Authorization,
    Cohort,
    ContractViolation,
    ExecutionLimits,
    ModelIdentity,
    TaskRevision,
    canonical_json,
    content_sha256,
)
from evolve.evidence import EvidenceGraph, ReceiptStore
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

from .agent_program_authority import (
    append_live_search_parent,
    atomic_write,
    build_live_report,
    project_live_claims,
    project_live_decision,
    seal_agent_program_run,
    sha256_bytes,
    verify_live_runtime_projection,
    verify_live_search_parent,
)
from .agent_program_runtime import HashVerifiedAgentProgramTransport
from .runner import CampaignRunner, CampaignSpec

LIVE_ADAPTER_ID = "local-declarative-agent-program-v1"
LIVE_EVALUATOR_ID = "local-patch-match-native-v1"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractViolation(f"{name} must be an object")
    return value


class LocalFileWorkspaceManager:
    def materialize(self, plan):
        source_uri = plan.task.source_uri
        if not isinstance(source_uri, str) or not source_uri:
            raise ContractViolation("live AgentProgram source_uri is missing")
        source = Path(source_uri).resolve()
        if (
            not source.is_file()
            or sha256_bytes(source.read_bytes()) != plan.task.source_sha256
        ):
            raise ContractViolation("live AgentProgram source identity drift")
        return {
            "workspace_id": f"local-file-{plan.plan_id}",
            "task_revision_id": plan.task.revision_id,
            "task_source_sha256": plan.task.source_sha256,
            "source_uri": str(source),
            "execution_scope": "live",
        }


class LocalDeclarativeAgentProgramExecutor:
    """Allowlisted local product adapter for a frozen declarative patch program."""

    remote = False

    def infer_program(self, revision, plan, workspace):
        if set(revision.context) != {"patch"}:
            raise ContractViolation(
                "local declarative AgentProgram context must contain only patch"
            )
        patch = revision.context.get("patch")
        if not isinstance(patch, str) or not patch:
            raise ContractViolation("local declarative AgentProgram patch is invalid")
        if revision.tool_policy != ("emit_patch",):
            raise ContractViolation(
                "local declarative AgentProgram requires emit_patch tool policy"
            )
        if "local-declarative-patch-v1" not in revision.capability_revision_ids:
            raise ContractViolation(
                "local declarative AgentProgram capability is not allowlisted"
            )
        execution_identity = {
            "adapter_id": LIVE_ADAPTER_ID,
            "program_prompt": revision.program_prompt,
            "program_context": dict(revision.context),
            "program_tool_policy": list(revision.tool_policy),
            "program_capability_revision_ids": list(
                revision.capability_revision_ids
            ),
            "task_revision_id": workspace.get("task_revision_id"),
            "task_source_sha256": workspace.get("task_source_sha256"),
        }
        patch_sha256 = sha256_bytes(patch.encode("utf-8"))
        return {
            "product_adapter_id": LIVE_ADAPTER_ID,
            "program_execution_sha256": sha256_bytes(
                canonical_json(execution_identity).encode("utf-8")
            ),
            "patch": patch,
            "patch_sha256": patch_sha256,
            "prediction_sha256": patch_sha256,
            "structural_valid": True,
            "failure_reason": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_cny": 0,
        }


class LocalPatchMatchNativeEvaluator:
    def __init__(self, *, expected_patch_sha256: str) -> None:
        if len(expected_patch_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in expected_patch_sha256
        ):
            raise ContractViolation("expected_patch_sha256 must be literal SHA-256")
        self.expected_patch_sha256 = expected_patch_sha256
        self.evaluator_id = (
            f"{LIVE_EVALUATOR_ID}@sha256:"
            + content_sha256(
                {"expected_patch_sha256": self.expected_patch_sha256}
            )
        )

    def evaluate(self, plan, workspace, model_output):
        if workspace.get("execution_scope") != "live":
            raise ContractViolation("live AgentProgram workspace scope drift")
        patch_sha256 = model_output.get("patch_sha256")
        return {
            "resolved": patch_sha256 == self.expected_patch_sha256,
            "native_valid": True,
            "native_error": None,
            "regressions": [],
            "prediction_sha256": patch_sha256,
            "product_adapter_id": LIVE_ADAPTER_ID,
        }


def _agent_program_revisions(
    config: Mapping[str, Any],
) -> tuple[AgentProgramRevision, ...]:
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
    return revisions


def _register_agent_programs(
    revisions: tuple[AgentProgramRevision, ...], output_root: Path
) -> None:
    registry = AgentProgramRegistry(output_root / "registries/agent-programs.jsonl")
    for revision in revisions:
        registry.append(
            AgentProgramRecord(
                program_id=revision.program_id,
                revision_id=revision.revision_id,
                parent_revision_id=revision.parent_revision_id,
                capability_revision_ids=revision.capability_revision_ids,
                artifact_sha256=revision.bundle_sha256,
                active=False,
            )
        )


def _verify_agent_program_registry(
    revisions: tuple[AgentProgramRevision, ...], output_root: Path
) -> None:
    registry = AgentProgramRegistry(output_root / "registries/agent-programs.jsonl")
    expected = tuple(
        AgentProgramRecord(
            program_id=revision.program_id,
            revision_id=revision.revision_id,
            parent_revision_id=revision.parent_revision_id,
            capability_revision_ids=revision.capability_revision_ids,
            artifact_sha256=revision.bundle_sha256,
            active=False,
        )
        for revision in revisions
    )
    if registry.all() != expected:
        raise ContractViolation("live AgentProgram registry projection drift")


def _validate_live_source(task: TaskRevision) -> None:
    source_uri = task.source_uri
    if not isinstance(source_uri, str) or not source_uri:
        raise ContractViolation("live AgentProgram source_uri is missing")
    source = Path(source_uri).resolve()
    if not source.is_file() or sha256_bytes(source.read_bytes()) != task.source_sha256:
        raise ContractViolation("live AgentProgram source identity drift")


def run_agent_program_live_campaign(
    *, config_path: Path, output_root: Path
) -> Mapping[str, Any]:
    """Run an allowlisted local AgentProgram product adapter with E1 authority."""

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
        "product_adapter_id",
        "parent_revision_root",
        "candidate_revision_roots",
        "generation_config",
        "task",
    }
    if not isinstance(config, Mapping) or set(config) != fields:
        raise ContractViolation("live AgentProgram campaign config fields are invalid")
    if config["schema_version"] != 1 or config["execution_profile"] != "live":
        raise ContractViolation("live AgentProgram campaign profile is invalid")
    if config["product_adapter_id"] != LIVE_ADAPTER_ID:
        raise ContractViolation("AgentProgram product adapter is not allowlisted")
    revisions = _agent_program_revisions(config)
    task_data = _mapping(config["task"], "task")
    if set(task_data) != {
        "task_id",
        "revision_id",
        "project",
        "cohort",
        "source_uri",
        "source_sha256",
        "evaluator_id",
        "expected_patch_sha256",
    }:
        raise ContractViolation("live AgentProgram task fields are invalid")
    evaluator = LocalPatchMatchNativeEvaluator(
        expected_patch_sha256=str(task_data["expected_patch_sha256"])
    )
    task = TaskRevision(
        task_id=str(task_data["task_id"]),
        revision_id=str(task_data["revision_id"]),
        project=str(task_data["project"]),
        cohort=Cohort(str(task_data["cohort"])),
        source_sha256=str(task_data["source_sha256"]),
        evaluator_id=str(task_data["evaluator_id"]),
        source_uri=str(task_data["source_uri"]),
    )
    if (
        task.cohort is not Cohort.FEEDBACK
        or task.evaluator_id != evaluator.evaluator_id
    ):
        raise ContractViolation("live AgentProgram campaign is feedback-only")
    _validate_live_source(task)
    generation = _mapping(config["generation_config"], "generation_config")
    output_root.mkdir(parents=True, exist_ok=True)
    frozen_config = output_root / "AGENT-PROGRAM-CONFIG.json"
    result_path = output_root / "CAMPAIGN-RESULT.json"
    manifest_path = output_root / "EVIDENCE-MANIFEST.json"
    if frozen_config.exists() and frozen_config.read_bytes() != config_bytes:
        raise ContractViolation("AgentProgram campaign replay config drift")
    sealed_replay = result_path.exists()
    if sealed_replay:
        if not frozen_config.is_file() or not manifest_path.is_file():
            raise ContractViolation("live AgentProgram sealed replay is incomplete")
        AuditVerifier().verify_manifest(manifest_path, root=output_root)
    if not frozen_config.exists():
        atomic_write(frozen_config, config_bytes)
    if sealed_replay:
        _verify_agent_program_registry(revisions, output_root)
    else:
        _register_agent_programs(revisions, output_root)

    campaign_id = str(config["campaign_id"])
    authorization = Authorization(
        authorization_id=f"auth-{sha256_bytes(config_bytes)[:24]}",
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
    store = ReceiptStore(output_root / "receipt-store")
    graph = EvidenceGraph(output_root / "evidence-graph")
    context = StrategyContext(
        campaign_id=campaign_id,
        task=task,
        model=ModelIdentity(
            provider="local-product",
            model="declarative-agent-program",
            revision="v1",
        ),
        context_policy_id="local-file-feedback-v1",
        tool_policy_id="allowlisted-declarative-tools-v1",
        observer_policy_ids=("external-trace-v1", "native-v1", "cost-v1"),
        limits=ExecutionLimits(max_tokens=0, max_seconds=60, max_cost_cny=0),
        inputs={
            "parent_revision_id": revisions[0].revision_id,
            "candidate_revision_ids": tuple(
                revision.revision_id for revision in revisions[1:]
            ),
            "tournament_id": str(config["tournament_id"]),
            "generation_config": dict(generation),
            "execution_profile": "live",
            "revision_roots": revision_roots,
            "claim_evidence_graph_root": str(graph.root),
            "claim_receipt_store_root": str(store.root),
        },
    )
    strategy = AgentProgramSearchStrategy(execution_profile="live")
    transport = HashVerifiedAgentProgramTransport(
        revision_roots,
        executor=LocalDeclarativeAgentProgramExecutor(),
    )
    runtime = ExecutionRuntime(
        model_transport=transport,
        workspace_manager=LocalFileWorkspaceManager(),
        native_evaluator=evaluator,
        observer_hub=ObserverHub(
            (ExternalTraceObserver(), NativeOutcomeObserver(), CostObserver()),
            graph=graph,
        ),
        receipt_sink=store,
    )
    runner = CampaignRunner(runtime=runtime, controller=controller)
    if sealed_replay:
        graph = EvidenceGraph.rebuild(graph.root, store)
        replay_plans = tuple(strategy.plan(context))
        verify_live_runtime_projection(
            plans=replay_plans,
            store=store,
            transport=transport,
            evaluator=evaluator,
        )
        claims = project_live_claims(
            plans=replay_plans,
            graph=graph,
            store=store,
            append=False,
        )
        if graph.latest_claims() != claims:
            raise ContractViolation("live AgentProgram Claim authority drift")
        authority = runner.run(
            CampaignSpec(
                campaign_id=campaign_id,
                contexts=(context,),
                authorization=authorization,
                claims=claims,
            ),
            strategy,
        )
        if len(authority.decisions) != 1 or any(
            not execution.replayed for execution in authority.executions
        ):
            raise ContractViolation("live AgentProgram sealed replay was not idempotent")
        participants = tuple(plan.candidate_revision_id for plan in authority.plans)
        decision = authority.decisions[0]
        decision_sha256, selected_revision_id = project_live_decision(
            decision,
            participants=participants,
        )
        verify_live_search_parent(
            path=output_root / "SEARCH-PARENT.jsonl",
            program_id=str(config["program_id"]),
            tournament_id=str(config["tournament_id"]),
            parent_revision_id=revisions[0].revision_id,
            selected_revision_id=selected_revision_id,
            decision_sha256=decision_sha256,
            claim_ids=decision.claim_ids,
        )
        projected = build_live_report(
            campaign_id=campaign_id,
            status=authority.status,
            strategy_id=strategy.strategy_id,
            product_adapter_id=LIVE_ADAPTER_ID,
            participants=participants,
            parent_revision_id=revisions[0].revision_id,
            selected_revision_id=selected_revision_id,
            decision_sha256=decision_sha256,
            decision_action=decision.action,
            receipt_ids=tuple(receipt.receipt_id for receipt in authority.receipts),
            claims=claims,
            initial_execution_replayed=tuple(False for _ in participants),
            authority_execution_replayed=tuple(
                execution.replayed for execution in authority.executions
            ),
        )
        try:
            replay = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ContractViolation(
                "live AgentProgram sealed result is unreadable"
            ) from error
        if replay != projected:
            raise ContractViolation("live AgentProgram sealed result authority drift")
        return projected

    first = runner.run(
        CampaignSpec(
            campaign_id=campaign_id,
            contexts=(context,),
            authorization=authorization,
        ),
        strategy,
    )
    if (
        len(first.decisions) != 1
        or first.decisions[0].action != "await-tournament-authority"
    ):
        raise ContractViolation("live AgentProgram first pass bypassed Claim authority")
    claims = project_live_claims(plans=first.plans, graph=graph, store=store)
    authority = runner.run(
        CampaignSpec(
            campaign_id=campaign_id,
            contexts=(context,),
            authorization=authorization,
            claims=claims,
        ),
        strategy,
    )
    if len(authority.decisions) != 1:
        raise ContractViolation("live AgentProgram authority decision is missing")
    participants = tuple(plan.candidate_revision_id for plan in authority.plans)
    decision = authority.decisions[0]
    decision_sha256, selected_revision_id = project_live_decision(
        decision,
        participants=participants,
    )
    append_live_search_parent(
        path=output_root / "SEARCH-PARENT.jsonl",
        program_id=str(config["program_id"]),
        tournament_id=str(config["tournament_id"]),
        parent_revision_id=revisions[0].revision_id,
        selected_revision_id=selected_revision_id,
        decision_sha256=decision_sha256,
        claim_ids=decision.claim_ids,
    )
    report = build_live_report(
        campaign_id=campaign_id,
        status=authority.status,
        strategy_id=strategy.strategy_id,
        product_adapter_id=LIVE_ADAPTER_ID,
        participants=participants,
        parent_revision_id=revisions[0].revision_id,
        selected_revision_id=selected_revision_id,
        decision_sha256=decision_sha256,
        decision_action=decision.action,
        receipt_ids=tuple(receipt.receipt_id for receipt in authority.receipts),
        claims=claims,
        initial_execution_replayed=tuple(row.replayed for row in first.executions),
        authority_execution_replayed=tuple(
            row.replayed for row in authority.executions
        ),
    )
    atomic_write(
        result_path,
        (canonical_json(report) + "\n").encode("utf-8"),
    )
    seal_agent_program_run(output_root)
    return report


__all__ = [
    "LIVE_ADAPTER_ID",
    "LIVE_EVALUATOR_ID",
    "LocalDeclarativeAgentProgramExecutor",
    "LocalFileWorkspaceManager",
    "LocalPatchMatchNativeEvaluator",
    "run_agent_program_live_campaign",
]
