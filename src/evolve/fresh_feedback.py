"""One product entry for a fresh, feedback-only local-Qwen/native campaign."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from evolve.contracts import (
    Authorization,
    Cohort,
    ContractViolation,
    ExecutionLimits,
    ModelIdentity,
    TaskRevision,
    canonical_json,
)
from evolve.evidence import (
    ClaimEngine,
    EvidenceGradeMachine,
    EvidenceGraph,
    ReceiptStore,
)
from evolve.governance import GovernanceService, PromotionDecisionLog
from evolve.kernel import (
    CampaignController,
    CheckpointManager,
    DurableCostLedger,
)
from evolve.live_campaign import LiveCampaignSpec, run_skill_paired_campaign
from evolve.observers import (
    CostObserver,
    ExternalTraceObserver,
    NativeOutcomeObserver,
    ObserverHub,
)
from evolve.proposals import CandidateCompiler, CompiledRevision, CompileSpec
from evolve.registry import (
    CandidateRegistry,
    CapabilityRegistry,
    RejectedRegistry,
)
from evolve.reporting import AuditVerifier
from evolve.runtime.live_adapters import (
    FrozenSourceWorkspaceManager,
    LegacyOfficialNativeEvaluator,
)
from evolve.runtime.qwen_transport import LegacyQwenCellRunner, LegacyQwenPairTransport
from evolve.strategies import SkillPairedStrategy

_MODEL_IDENTITY_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
)
_DATASETS = {
    "swe-bench-verified": "harness-inputs/swe-bench-verified.jsonl",
    "swe-bench-multilingual": "harness-inputs/swe-bench-multilingual.jsonl",
}
_EXPERIMENT_MECHANISM_ID = "compiled-teacher-candidate-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args), cwd=root, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise ContractViolation(f"Git identity command failed for {root}")
    return completed.stdout.strip()


def _require_clean_head(root: Path) -> str:
    head = _git(root, "rev-parse", "HEAD")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ContractViolation("fresh campaign requires a clean committed HEAD")
    return head


def _load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractViolation("fresh feedback config is unreadable") from error
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ContractViolation("fresh feedback config schema is unsupported")
    tasks = config.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 3:
        raise ContractViolation("fresh feedback config requires exactly three tasks")
    if len({row.get("instance_id") for row in tasks if isinstance(row, dict)}) != 3:
        raise ContractViolation("fresh feedback task identities must be unique")
    if any(
        not isinstance(row, dict) or row.get("cohort") != "feedback" for row in tasks
    ):
        raise ContractViolation("fresh feedback config cannot admit holdout tasks")
    forbidden_fallbacks = {"operator_skill_path", "span_skill_path"} & set(config)
    if forbidden_fallbacks:
        raise ContractViolation(
            "legacy frozen Skill fallback fields are forbidden: "
            + ", ".join(sorted(forbidden_fallbacks))
        )
    return config


def _path(config: Mapping[str, Any], name: str) -> Path:
    value = config.get(name)
    if not isinstance(value, str) or not value:
        raise ContractViolation(f"fresh feedback config path is missing: {name}")
    result = Path(value).expanduser().resolve()
    if not result.exists():
        raise ContractViolation(f"fresh feedback config path is missing: {name}")
    return result


def _launcher(config: Mapping[str, Any], name: str) -> Path:
    value = config.get(name)
    if not isinstance(value, str) or not value:
        raise ContractViolation(f"fresh feedback launcher is missing: {name}")
    result = Path(value).expanduser().absolute()
    if not result.is_file():
        raise ContractViolation(f"fresh feedback launcher is missing: {name}")
    return result


def _model_identity(model_path: Path) -> tuple[ModelIdentity, dict[str, str]]:
    hashes = {}
    for name in _MODEL_IDENTITY_FILES:
        path = model_path / name
        if not path.is_file():
            raise ContractViolation(f"frozen Qwen identity file is missing: {name}")
        hashes[name] = _sha256(path)
    revision = hashlib.sha256(canonical_json(hashes).encode()).hexdigest()
    return (
        ModelIdentity(
            provider="local-mlx",
            model="Qwen3.5-4B-mlx-4bit",
            revision=f"sha256:{revision}",
        ),
        hashes,
    )


def _evaluator_identity(
    *,
    official_source: Path,
    swe_harness_root: Path,
    multi_harness_root: Path,
    pool_root: Path,
    benchmarks: set[str],
) -> tuple[str, dict[str, Any]]:
    dataset_hashes = {}
    for benchmark in sorted(benchmarks):
        relative = _DATASETS.get(benchmark)
        if relative is None:
            raise ContractViolation("fresh task uses an unsupported benchmark")
        dataset = pool_root / relative
        if not dataset.is_file():
            raise ContractViolation(f"frozen native dataset is missing: {benchmark}")
        dataset_hashes[benchmark] = _sha256(dataset)
    identity = {
        "official_patch_evaluator_sha256": _sha256(official_source),
        "swe_harness_revision": _git(swe_harness_root, "rev-parse", "HEAD"),
        "multi_harness_revision": _git(multi_harness_root, "rev-parse", "HEAD"),
        "dataset_sha256": dataset_hashes,
    }
    digest = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    return f"official-native@sha256:{digest}", identity


def _build_tasks(
    rows: list[dict[str, Any]], evaluator_id: str
) -> tuple[tuple[TaskRevision, ...], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    tasks = []
    metadata = {}
    source_inventory = []
    for row in rows:
        checkout = Path(str(row.get("source_uri", ""))).expanduser().resolve()
        base_revision = str(row.get("base_revision", ""))
        if (
            not checkout.is_dir()
            or _git(checkout, "rev-parse", "HEAD") != base_revision
        ):
            raise ContractViolation("fresh feedback source revision drift")
        if _git(checkout, "status", "--porcelain", "--untracked-files=all"):
            raise ContractViolation("fresh feedback source checkout is not clean")
        tree = _git(checkout, "rev-parse", "HEAD^{tree}")
        instance_id = str(row.get("instance_id", ""))
        fingerprint = str(row.get("catalog_fingerprint", ""))
        if len(fingerprint) != 64:
            raise ContractViolation("fresh feedback catalog fingerprint is invalid")
        source_sha = hashlib.sha256(
            canonical_json(
                {
                    "base_revision": base_revision,
                    "git_tree": tree,
                    "instance_id": instance_id,
                    "catalog_fingerprint": fingerprint,
                }
            ).encode()
        ).hexdigest()
        revision_id = f"feedback-{instance_id}@{base_revision[:12]}"
        task = TaskRevision(
            task_id=instance_id,
            revision_id=revision_id,
            project=str(row.get("project", "")),
            cohort=Cohort.FEEDBACK,
            source_sha256=source_sha,
            evaluator_id=evaluator_id,
            source_uri=str(checkout),
        )
        tasks.append(task)
        metadata[revision_id] = {
            "base_revision": base_revision,
            "benchmark_id": str(row.get("benchmark_id", "")),
            "instance_id": instance_id,
            "catalog_fingerprint": fingerprint,
        }
        source_inventory.append(
            {
                "task_revision_id": revision_id,
                "instance_id": instance_id,
                "project": task.project,
                "checkout": str(checkout),
                "base_revision": base_revision,
                "git_tree": tree,
                "source_sha256": source_sha,
                "catalog_fingerprint": fingerprint,
            }
        )
    if len({task.project for task in tasks}) < 2:
        raise ContractViolation(
            "fresh feedback campaign requires at least two projects"
        )
    return tuple(tasks), metadata, source_inventory


def _freeze_harness(config: Mapping[str, Any], output_root: Path) -> Path:
    legacy_root = _path(config, "legacy_root")
    rendered = str(legacy_root)
    sys.path.insert(0, rendered)
    try:
        module = importlib.import_module("official_patch_evaluator")
        receipt = module.freeze_official_harness_runtime(
            swe_python=_launcher(config, "swe_python"),
            multi_python=_launcher(config, "multi_python"),
            swe_harness_root=_path(config, "swe_harness_root"),
            multi_harness_root=_path(config, "multi_harness_root"),
            output_root=output_root / "harness-runtime",
            native_assets_path=_path(config, "native_assets_path"),
        )
    except Exception as error:
        raise ContractViolation(
            f"official harness preflight failed: {type(error).__name__}: {error}"
        ) from error
    finally:
        sys.path.remove(rendered)
    return Path(receipt).resolve()


def _compile_teacher_candidate(*, config: Mapping[str, Any], output_root: Path):
    candidate_id = str(config.get("candidate_id", ""))
    revision_id = str(config.get("candidate_revision_id", ""))
    task_ids = tuple(str(row["instance_id"]) for row in config["tasks"])
    operator_id = str(
        config.get("compiled_operator_id", "apply-compiled-teacher-candidate")
    )
    spec = CompileSpec(
        candidate_id=candidate_id,
        revision_id=revision_id,
        parent_revision_id="qwen-zero-teaching-v1",
        cohort=Cohort.FEEDBACK,
        operator_id=operator_id,
        operator_instruction=str(
            config.get(
                "compiled_operator_instruction",
                "Apply the compiled inactive Teacher teaching to this paired arm.",
            )
        ),
        routes=tuple((task_id, operator_id) for task_id in task_ids),
    )
    return CandidateCompiler().compile(
        request_path=_path(config, "teacher_request"),
        response_path=_path(config, "teacher_response"),
        compile_spec=spec,
        output_root=output_root / "compiled-candidates",
    )


def _freeze_release_candidate_artifacts(
    compiled: CompiledRevision, output_root: Path
) -> None:
    """Expose the self-contained causal chain at the run root without rewriting it."""

    names = (
        "TEACHER-REQUEST.json",
        "TEACHER-RESPONSE.json",
        "MODEL-RECEIPT.json",
        "COST-RECEIPT.json",
        "CANDIDATE-CHANGESET.json",
        "COMPILED-SKILL.json",
        "COMPILED-OPERATOR.json",
        "COMPILED-ROUTER.json",
        "COMPILED-REVISION.json",
    )
    for name in names:
        source = compiled.root / name
        content = source.read_bytes()
        target = output_root / name
        try:
            descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            if target.read_bytes() != content:
                raise ContractViolation(
                    f"immutable release candidate artifact conflict: {name}"
                ) from error
            continue
        try:
            written = os.write(descriptor, content)
            if written != len(content):
                raise ContractViolation(
                    f"partial release candidate artifact write: {name}"
                )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def run_fresh_feedback_e2e(*, config_path: Path, output_root: Path) -> dict[str, Any]:
    """Execute all six arms via the v3 Runtime and seal rebuildable evidence."""

    config = _load_config(config_path.resolve())
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final_commit = _require_clean_head(Path.cwd())
    expected_commit = config.get("final_commit_sha")
    if expected_commit != final_commit:
        raise ContractViolation("fresh campaign config is not bound to final HEAD")

    legacy_root = _path(config, "legacy_root")
    model_path = _path(config, "model_path")
    pool_root = _path(config, "pool_root")
    swe_harness_root = _path(config, "swe_harness_root")
    multi_harness_root = _path(config, "multi_harness_root")
    official_source = legacy_root / "official_patch_evaluator.py"
    evaluator_id, evaluator_identity = _evaluator_identity(
        official_source=official_source,
        swe_harness_root=swe_harness_root,
        multi_harness_root=multi_harness_root,
        pool_root=pool_root,
        benchmarks={str(row["benchmark_id"]) for row in config["tasks"]},
    )
    tasks, task_metadata, source_inventory = _build_tasks(config["tasks"], evaluator_id)
    task_metadata = {
        revision_id: {**metadata, "mechanism_id": _EXPERIMENT_MECHANISM_ID}
        for revision_id, metadata in task_metadata.items()
    }
    model, model_hashes = _model_identity(model_path)
    compiled = _compile_teacher_candidate(config=config, output_root=output_root)
    _freeze_release_candidate_artifacts(compiled, output_root)
    harness_receipt = _freeze_harness(config, output_root)

    campaign_id = str(config.get("campaign_id", ""))
    teacher_ledger = DurableCostLedger(
        output_root / "cost-ledger/events.jsonl",
        campaign_id=campaign_id,
        max_cost_cny=10.0,
        max_model_calls=1,
    )
    teacher_reservation_id = "teacher-reservation-" + compiled.bundle_sha256[:24]
    teacher_result_id = "teacher-result-" + compiled.bundle_sha256[:24]
    teacher_ledger.reserve(
        teacher_reservation_id,
        cost_cny=compiled.cost_cny,
        model_calls=1,
    )
    teacher_ledger.record(
        teacher_reservation_id,
        result_id=teacher_result_id,
        actual_cost_cny=compiled.cost_cny,
        actual_model_calls=1,
    )
    teacher_budget = teacher_ledger.snapshot()
    teacher_events = teacher_ledger.events()
    now = datetime.now(UTC)
    authorization = Authorization(
        authorization_id=f"auth-{campaign_id}",
        campaign_id=campaign_id,
        allowed_cohorts=(Cohort.FEEDBACK,),
        max_cost_cny=0,
        max_model_calls=6,
        expires_at=now + timedelta(hours=8),
        remote_calls_allowed=False,
    )
    checkpoints = CheckpointManager(output_root / "campaign-checkpoints")
    checkpoint_path = checkpoints.path_for(campaign_id)
    controller = (
        CampaignController.from_checkpoint(
            campaign_id=campaign_id,
            authorization=authorization,
            checkpoint_manager=checkpoints,
            now=now,
        )
        if checkpoint_path.exists()
        else CampaignController.create(
            campaign_id=campaign_id,
            authorization=authorization,
            checkpoint_manager=checkpoints,
            now=now,
        )
    )
    receipt_store = ReceiptStore(output_root / "receipt-store")
    graph = EvidenceGraph(output_root / "evidence-graph")
    observer_hub = ObserverHub(
        (
            ExternalTraceObserver(),
            NativeOutcomeObserver(),
            CostObserver(),
        ),
        graph=graph,
    )
    runner = LegacyQwenCellRunner(
        legacy_root=legacy_root,
        model_path=model_path,
        taskset_path=_path(config, "taskset_path"),
        routes_path=_path(config, "routes_path"),
        compiled_revision_root=compiled.root,
    )
    transport = LegacyQwenPairTransport(
        cell_runner=runner, output_root=output_root / "qwen-cells"
    )
    evaluator = LegacyOfficialNativeEvaluator(
        evaluator_id=evaluator_id,
        legacy_root=legacy_root,
        swe_python=_launcher(config, "swe_python"),
        multi_python=_launcher(config, "multi_python"),
        swe_harness_root=swe_harness_root,
        multi_harness_root=multi_harness_root,
        pool_root=pool_root,
        output_root=output_root / "native-official",
        timeout_seconds=int(config.get("native_timeout_seconds", 7200)),
    )
    candidate_id = compiled.change_set.candidate_id
    candidate_revision = compiled.change_set.revision_id
    spend = teacher_budget.spent_cost_cny
    spec = LiveCampaignSpec(
        campaign_id=campaign_id,
        baseline_revision_id="qwen-zero-teaching-v1",
        candidate_id=candidate_id,
        candidate_revision_id=candidate_revision,
        candidate_kind="external-skill",
        candidate_artifact_sha256=compiled.bundle_sha256,
        model=model,
        context_policy_id="frozen-feedback-task-v1",
        tool_policy_id="deterministic-operator-span-v1",
        observer_policy_ids=(
            "external-trace-v1",
            "native-v1",
            "cost-v1",
        ),
        limits=ExecutionLimits(max_tokens=1536, max_seconds=7200, max_cost_cny=0),
        final_commit_sha=final_commit,
        mechanism_id=_EXPERIMENT_MECHANISM_ID,
        human_approval=False,
        generation_config={
            "temperature": 0,
            "seed": 0,
            "thinking": False,
            "max_tokens": 1536,
            "max_context_tokens": 24000,
        },
        task_execution_metadata=task_metadata,
        report_metadata={
            "outcome": "fresh_runtime_closed_loop",
            "final_branch": _git(Path.cwd(), "branch", "--show-current"),
            "actual_api_spend_cny": spend,
            "api_budget_limit_cny": 10.0,
            "api_budget_remaining_cny": round(10.0 - spend, 8),
            "teacher_call_mode": "frozen_real_receipt_replay",
            "teacher_request_sha256": dict(compiled.artifact_sha256)[
                "TEACHER-REQUEST.json"
            ],
            "teacher_response_sha256": dict(compiled.artifact_sha256)[
                "TEACHER-RESPONSE.json"
            ],
            "candidate_sha256": compiled.change_set.source_candidate_sha256,
            "compiled_revision_sha256": compiled.change_set.content_sha256,
            "compiled_bundle_sha256": compiled.bundle_sha256,
            "baseline_program_sha256": hashlib.sha256(
                b"qwen-zero-teaching-v1"
            ).hexdigest(),
            "taught_program_sha256": hashlib.sha256(
                ("qwen-zero-teaching-v1\0" + compiled.bundle_sha256).encode()
            ).hexdigest(),
            "holdout_opened": False,
            "burned_holdout_opened": False,
            "skill_auto_activated": False,
            "model_identity_sha256": model_hashes,
            "evaluator_identity": evaluator_identity,
            "harness_runtime_receipt_sha256": _sha256(harness_receipt),
            "feedback_task_count": 3,
            "project_count": len({task.project for task in tasks}),
        },
    )
    promotion_log = PromotionDecisionLog(
        output_root / "registries/promotion-decisions.jsonl"
    )
    result = run_skill_paired_campaign(
        spec=spec,
        tasks=tasks,
        strategy=SkillPairedStrategy(),
        controller=controller,
        authorization=authorization,
        model_transport=transport,
        workspace_manager=FrozenSourceWorkspaceManager(),
        native_evaluator=evaluator,
        receipt_store=receipt_store,
        observer_hub=observer_hub,
        claim_engine=ClaimEngine(graph),
        evidence_grade_machine=EvidenceGradeMachine(
            graph,
            receipt_store=receipt_store,
        ),
        governance_service=GovernanceService(),
        promotion_decision_log=promotion_log,
        candidate_registry=CandidateRegistry(
            output_root / "registries/candidates.jsonl"
        ),
        capability_registry=CapabilityRegistry(
            output_root / "registries/capabilities.jsonl",
            decision_log=promotion_log,
        ),
        rejected_registry=RejectedRegistry(
            output_root / "registries/rejected.jsonl",
            decision_log=promotion_log,
        ),
        report_root=output_root,
    )
    _atomic_json(
        output_root / "SOURCE-INVENTORY.json",
        {"schema_version": 1, "sources": source_inventory},
    )
    summary = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "final_commit_sha": final_commit,
        "campaign_status": str(result.snapshot.status),
        "execution_statuses": [execution.status for execution in result.executions],
        "execution_replayed": [execution.replayed for execution in result.executions],
        "claims": [
            {
                "task_id": task.task_id,
                "classification": str(claim.classification),
                "claim_id": claim.claim_id,
            }
            for task, claim in zip(tasks, result.claims, strict=True)
        ],
        "evidence_grade_reached": str(result.evidence_state.grade),
        "e3_eligible": result.evidence_state.e3_eligible,
        "promotion_status": str(result.promotion_decision.gate_decision),
        "capability_created": result.capability is not None,
        "capability_active": (
            result.capability.active if result.capability is not None else False
        ),
        "holdout_opened": False,
        "burned_holdout_opened": False,
        "api_spend_cny": spend,
    }
    _atomic_json(output_root / "CAMPAIGN-RESULT.json", summary)
    _atomic_json(
        output_root / "COST-LEDGER-SNAPSHOT.json",
        {
            "schema_version": 1,
            "authority_path": "cost-ledger/events.jsonl",
            "head_path": "cost-ledger/events.jsonl.head.json",
            "event_count": len(teacher_events),
            "head_event_sha256": teacher_events[-1]["event_sha256"],
            "chain_verified_on_recovery": True,
            "budget_cny": teacher_budget.max_cost_cny,
            "reserved_cny": teacher_budget.reserved_cost_cny,
            "actual_spend_cny": teacher_budget.spent_cost_cny,
            "remaining_cny": round(
                teacher_budget.max_cost_cny
                - teacher_budget.spent_cost_cny
                - teacher_budget.reserved_cost_cny,
                8,
            ),
            "teacher": {
                "provider": compiled.provider,
                "model": compiled.model,
                "cost_cny": spend,
                "request_id": teacher_result_id,
                "cost_receipt_sha256": dict(compiled.artifact_sha256)[
                    "COST-RECEIPT.json"
                ],
            },
            "qwen": {"provider": "local-mlx", "api_cost_cny": 0},
        },
    )
    seal_run(output_root)
    return summary


def seal_run(root: Path) -> int:
    """Recompute a literal-SHA manifest after all run-local artifacts settle."""

    root = root.resolve()
    manifest_path = root / "EVIDENCE-MANIFEST.json"
    excluded_prefixes = ("sources/",)
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        relative = path.relative_to(root).as_posix()
        if relative == ".DS_Store" or relative.startswith(excluded_prefixes):
            continue
        entries.append({"path": relative, "sha256": _sha256(path)})
    _atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "entries": entries,
            "excluded_prefixes": list(excluded_prefixes),
        },
    )
    return AuditVerifier().verify_manifest(manifest_path, root=root)


__all__ = ["run_fresh_feedback_e2e", "seal_run"]
