"""Real local-Qwen / official-native executor for autonomous feedback rounds."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from evolve.campaigns import CampaignRunner, CampaignSpec
from evolve.contracts import (
    Authorization,
    Cohort,
    ExecutionLimits,
    canonical_json,
)
from evolve.evidence import EvidenceGraph, ReceiptStore
from evolve.fresh_feedback import (
    _build_tasks,
    _evaluator_identity,
    _model_identity,
    run_fresh_feedback_e2e,
)
from evolve.kernel import CampaignController, CheckpointManager
from evolve.observers import (
    CostObserver,
    ExternalTraceObserver,
    NativeOutcomeObserver,
    ObserverHub,
)
from evolve.runtime import ExecutionRuntime
from evolve.runtime.live_adapters import (
    FrozenSourceWorkspaceManager,
    LegacyOfficialNativeEvaluator,
)
from evolve.runtime.qwen_transport import LegacyQwenCellRunner, LegacyQwenPairTransport
from evolve.strategies import SkillPairedStrategy, StrategyContext, StrategyPhase

from .config import AutonomousEvolutionError
from .output import atomic_json
from .runner import (
    BaselineProbeResult,
    PrescreenResult,
    RoundExecutionRequest,
    RuntimePreflightResult,
)


class FreshFeedbackRoundExecutor:
    """Execute every external effect through existing v3 Runtime authorities."""

    def preflight(self, request: RoundExecutionRequest) -> RuntimePreflightResult:
        runtime = _runtime_metadata(request)
        environment = self._environment(
            request, request.output_root / ".preflight-native"
        )
        self._qwen_runner(
            request,
            candidate_root=request.output_root / ".preflight-no-candidate",
        )
        legacy_root = Path(str(runtime["legacy_root"])).expanduser().resolve()
        for relative in (
            "skill_evolution_loop/operator_student.py",
            "skill_evolution_loop/span_student.py",
        ):
            if not (legacy_root / relative).is_file():
                raise AutonomousEvolutionError(
                    "local Qwen implementation failed preflight"
                )
        if importlib.util.find_spec("mlx_lm") is None:
            raise AutonomousEvolutionError(
                "local Qwen dependency failed preflight"
            )
        for name in ("swe_python", "multi_python"):
            if not Path(str(runtime[name])).expanduser().resolve().is_file():
                raise AutonomousEvolutionError(
                    f"native evaluator interpreter failed preflight: {name}"
                )
        qwen_identity = _identity_sha256(
            {
                "legacy_root": str(legacy_root),
                "model_path": str(request.config.model.model_path),
                "task_pool_sha256": _file_sha256(
                    request.config.swe_bench.task_pool
                ),
                "routes_sha256": _file_sha256(Path(str(runtime["routes_path"]))),
                "selected_task_revision_ids": [
                    task.revision_id for task in environment.tasks
                ],
            }
        )
        native_identity = _identity_sha256(
            {
                "evaluator_id": environment.evaluator.evaluator_id,
                "official_evaluator_sha256": _file_sha256(
                    request.config.swe_bench.official_evaluator
                ),
                "swe_python": str(Path(str(runtime["swe_python"])).resolve()),
                "multi_python": str(Path(str(runtime["multi_python"])).resolve()),
                "selected_task_revision_ids": [
                    task.revision_id for task in environment.tasks
                ],
            }
        )
        return RuntimePreflightResult(
            qwen_identity_sha256=qwen_identity,
            native_identity_sha256=native_identity,
        )

    def baseline(self, request: RoundExecutionRequest) -> BaselineProbeResult:
        environment = self._environment(request, request.output_root / "baseline-probe")
        campaign_id = (
            f"{request.goal_id}-r{request.round_index:04d}-baseline-probe"
        )
        authorization = self._authorization(campaign_id, model_calls=3)
        checkpoints = CheckpointManager(
            request.output_root / "baseline-probe/checkpoints"
        )
        checkpoint = checkpoints.path_for(campaign_id)
        controller = (
            CampaignController.from_checkpoint(
                campaign_id=campaign_id,
                authorization=authorization,
                checkpoint_manager=checkpoints,
                now=datetime.now(UTC),
            )
            if checkpoint.exists()
            else CampaignController.create(
                campaign_id=campaign_id,
                authorization=authorization,
                checkpoint_manager=checkpoints,
                now=datetime.now(UTC),
            )
        )
        receipt_store = ReceiptStore(
            request.output_root / "baseline-probe/receipt-store"
        )
        graph = EvidenceGraph(request.output_root / "baseline-probe/evidence-graph")
        observer = ObserverHub(
            (ExternalTraceObserver(), NativeOutcomeObserver(), CostObserver()),
            graph=graph,
        )
        model_runner = self._qwen_runner(
            request,
            candidate_root=(
                request.candidate.root
                if request.candidate is not None
                else request.output_root / ".no-proposed-candidate"
            ),
        )
        runtime = ExecutionRuntime(
            model_transport=LegacyQwenPairTransport(
                cell_runner=model_runner,
                output_root=request.output_root / "baseline-probe/qwen-cells",
            ),
            workspace_manager=FrozenSourceWorkspaceManager(),
            native_evaluator=environment.evaluator,
            observer_hub=observer,
            receipt_sink=receipt_store,
        )
        contexts = tuple(
            StrategyContext(
                campaign_id=campaign_id,
                task=task,
                model=environment.model,
                context_policy_id="frozen-feedback-task-v1",
                tool_policy_id="deterministic-operator-span-v1",
                observer_policy_ids=(
                    "external-trace-v1",
                    "native-v1",
                    "cost-v1",
                ),
                limits=ExecutionLimits(
                    max_tokens=1536, max_seconds=7200, max_cost_cny=0
                ),
                phase=StrategyPhase.BASELINE_ONLY,
                inputs={
                    "baseline_revision_id": request.baseline_revision_id,
                    "generation_config": _generation_config(request),
                    "plan_metadata": environment.task_metadata[task.revision_id],
                },
            )
            for task in environment.tasks
        )
        result = CampaignRunner(runtime=runtime, controller=controller).run(
            CampaignSpec(
                campaign_id=campaign_id,
                contexts=contexts,
                authorization=authorization,
            ),
            SkillPairedStrategy(),
        )
        if str(result.status) != "completed":
            raise AutonomousEvolutionError("baseline campaign did not complete")
        models = tuple(
            receipt for receipt in result.receipts if receipt.kind == "model"
        )
        native = tuple(
            receipt
            for receipt in result.receipts
            if receipt.kind == "native_evaluation"
        )
        outcomes = tuple(
            {
                "task_revision_id": receipt.payload["task_revision_id"],
                "resolved": bool(receipt.payload.get("resolved", False)),
                "evaluator_error": receipt.payload.get("evaluator_error"),
                "receipt_id": receipt.receipt_id,
            }
            for receipt in native
        )
        failures = tuple(
            {
                "task_revision_id": row["task_revision_id"],
                "signature": (
                    "evaluator_infrastructure_error"
                    if row["evaluator_error"]
                    else "native_unresolved"
                ),
            }
            for row in outcomes
            if not row["resolved"] or row["evaluator_error"]
        )
        return BaselineProbeResult(
            task_ids=tuple(task.task_id for task in environment.tasks),
            model_receipt_ids=tuple(receipt.receipt_id for receipt in models),
            native_receipt_ids=tuple(receipt.receipt_id for receipt in native),
            native_outcomes=outcomes,
            failure_signatures=failures,
            replayed=bool(result.executions)
            and all(execution.replayed for execution in result.executions),
        )

    def prescreen(self, request: RoundExecutionRequest) -> PrescreenResult:
        if request.candidate is None:
            raise AutonomousEvolutionError("prescreen requires a compiled candidate")
        environment = self._environment(request, request.output_root / "prescreen")
        task = environment.tasks[0]
        campaign_id = f"{request.goal_id}-r{request.round_index:04d}-prescreen"
        context = StrategyContext(
            campaign_id=campaign_id,
            task=task,
            model=environment.model,
            context_policy_id="frozen-feedback-task-v1",
            tool_policy_id="deterministic-operator-span-v1",
            observer_policy_ids=("external-trace-v1", "cost-v1"),
            limits=ExecutionLimits(max_tokens=1536, max_seconds=7200, max_cost_cny=0),
            phase=StrategyPhase.EXPERIMENT,
            inputs={
                "baseline_revision_id": request.baseline_revision_id,
                "taught_revision_id": request.candidate.change_set.revision_id,
                "generation_config": _generation_config(request),
                "plan_metadata": {
                    **environment.task_metadata[task.revision_id],
                    "execution_mode": "model-only-prescreen",
                },
            },
        )
        plans = SkillPairedStrategy().plan(context)
        taught = plans[1]
        receipt_store = ReceiptStore(request.output_root / "prescreen/receipt-store")
        graph = EvidenceGraph(request.output_root / "prescreen/evidence-graph")
        model_runner = self._qwen_runner(request, candidate_root=request.candidate.root)
        runtime = ExecutionRuntime(
            model_transport=LegacyQwenPairTransport(
                cell_runner=model_runner,
                output_root=request.output_root / "prescreen/qwen-cells",
            ),
            workspace_manager=FrozenSourceWorkspaceManager(),
            native_evaluator=environment.evaluator,
            observer_hub=ObserverHub(
                (ExternalTraceObserver(), CostObserver()), graph=graph
            ),
            receipt_sink=receipt_store,
        )
        result = runtime.execute(taught, self._authorization(campaign_id, model_calls=1))
        if result.status != "completed":
            raise AutonomousEvolutionError("Qwen prescreen execution did not complete")
        model_receipt = next(
            (receipt for receipt in result.receipts if receipt.kind == "model"), None
        )
        if model_receipt is None:
            raise AutonomousEvolutionError("Qwen prescreen produced no ModelReceipt")
        patch = model_receipt.payload.get("patch")
        applicable = False
        if isinstance(patch, str) and patch.strip():
            source = Path(task.source_uri)
            checked = subprocess.run(
                ("git", "apply", "--check", "-"),
                cwd=source,
                input=patch,
                text=True,
                capture_output=True,
                check=False,
            )
            applicable = checked.returncode == 0
        return PrescreenResult(
            candidate_revision_id=request.candidate.change_set.revision_id,
            candidate_bundle_sha256=request.candidate.bundle_sha256,
            model_receipt_ids=(model_receipt.receipt_id,),
            structural_valid=bool(model_receipt.payload.get("structural_valid")),
            patch_applicable=applicable,
            replayed=result.replayed,
        )

    def paired(self, request: RoundExecutionRequest) -> Mapping[str, Any]:
        if request.candidate is None:
            raise AutonomousEvolutionError("paired campaign requires a candidate")
        config_path = request.output_root / "FRESH-FEEDBACK-CONFIG.json"
        atomic_json(config_path, self._fresh_config(request))
        return run_fresh_feedback_e2e(
            config_path=config_path,
            output_root=request.output_root,
        )

    def _environment(
        self, request: RoundExecutionRequest, evidence_root: Path
    ) -> _Environment:
        runtime = _runtime_metadata(request)
        legacy_root = Path(str(runtime["legacy_root"])).expanduser().resolve()
        pool_root = Path(str(runtime["pool_root"])).expanduser().resolve()
        multi_harness = Path(
            str(runtime["multi_harness_root"])
        ).expanduser().resolve()
        evaluator_id, _identity = _evaluator_identity(
            official_source=request.config.swe_bench.official_evaluator,
            swe_harness_root=request.config.swe_bench.official_harness,
            multi_harness_root=multi_harness,
            pool_root=pool_root,
            benchmarks={str(task["benchmark_id"]) for task in request.selection.tasks},
        )
        tasks, task_metadata, _inventory = _build_tasks(
            [dict(task) for task in request.selection.tasks], evaluator_id
        )
        model, _hashes = _model_identity(request.config.model.model_path)
        evaluator = LegacyOfficialNativeEvaluator(
            evaluator_id=evaluator_id,
            legacy_root=legacy_root,
            swe_python=Path(str(runtime["swe_python"])),
            multi_python=Path(str(runtime["multi_python"])),
            swe_harness_root=request.config.swe_bench.official_harness,
            multi_harness_root=multi_harness,
            pool_root=pool_root,
            output_root=evidence_root / "native-official",
            timeout_seconds=int(runtime.get("native_timeout_seconds", 7200)),
        )
        return _Environment(
            tasks=tasks,
            task_metadata=task_metadata,
            model=model,
            evaluator=evaluator,
        )

    def _qwen_runner(
        self, request: RoundExecutionRequest, *, candidate_root: Path
    ) -> LegacyQwenCellRunner:
        runtime = _runtime_metadata(request)
        arguments: dict[str, Any] = {
            "legacy_root": Path(str(runtime["legacy_root"])),
            "model_path": request.config.model.model_path,
            "taskset_path": request.config.swe_bench.task_pool,
            "routes_path": Path(str(runtime["routes_path"])),
            "compiled_revision_root": candidate_root,
        }
        if request.baseline_compiled_root is not None:
            arguments["baseline_compiled_revision_root"] = (
                request.baseline_compiled_root
            )
        try:
            return LegacyQwenCellRunner(**arguments)
        except TypeError as error:
            if "baseline_compiled_revision_root" in arguments:
                raise AutonomousEvolutionError(
                    "Qwen runtime cannot yet consume the current best baseline Harness"
                ) from error
            arguments.pop("baseline_compiled_revision_root", None)
            return LegacyQwenCellRunner(**arguments)

    def _fresh_config(self, request: RoundExecutionRequest) -> dict[str, Any]:
        assert request.candidate is not None
        runtime = _runtime_metadata(request)
        return {
            "schema_version": 1,
            "campaign_id": f"{request.goal_id}-r{request.round_index:04d}-paired",
            "candidate_id": request.candidate.change_set.candidate_id,
            "candidate_revision_id": request.candidate.change_set.revision_id,
            "parent_revision_id": request.candidate.change_set.parent_revision_id,
            "baseline_revision_id": request.baseline_revision_id,
            "baseline_compiled_revision_root": (
                str(request.baseline_compiled_root)
                if request.baseline_compiled_root is not None
                else None
            ),
            "compiled_revision_root": str(request.candidate.root),
            "final_commit_sha": request.source_commit_sha,
            "source_root": str(request.worktree_root),
            "legacy_root": str(runtime["legacy_root"]),
            "model_path": str(request.config.model.model_path),
            "taskset_path": str(request.config.swe_bench.task_pool),
            "routes_path": str(runtime["routes_path"]),
            "teacher_request": str(
                request.candidate.root / "TEACHER-REQUEST.json"
            ),
            "teacher_response": str(
                request.candidate.root / "TEACHER-RESPONSE.json"
            ),
            "swe_python": str(runtime["swe_python"]),
            "multi_python": str(runtime["multi_python"]),
            "swe_harness_root": str(request.config.swe_bench.official_harness),
            "multi_harness_root": str(runtime["multi_harness_root"]),
            "pool_root": str(runtime["pool_root"]),
            "native_assets_path": str(runtime["native_assets_path"]),
            "native_timeout_seconds": int(runtime.get("native_timeout_seconds", 7200)),
            "teacher_budget_cny": request.config.teacher.budget_cny,
            "tasks": [dict(task) for task in request.selection.tasks],
        }

    @staticmethod
    def _authorization(campaign_id: str, *, model_calls: int) -> Authorization:
        return Authorization(
            authorization_id=f"auth-{campaign_id}",
            campaign_id=campaign_id,
            allowed_cohorts=(Cohort.FEEDBACK,),
            max_cost_cny=0,
            max_model_calls=model_calls,
            expires_at=datetime.now(UTC) + timedelta(hours=8),
            remote_calls_allowed=False,
        )


class _Environment:
    def __init__(self, *, tasks, task_metadata, model, evaluator) -> None:
        self.tasks = tasks
        self.task_metadata = task_metadata
        self.model = model
        self.evaluator = evaluator


def _runtime_metadata(request: RoundExecutionRequest) -> dict[str, Any]:
    try:
        document = json.loads(
            request.config.swe_bench.task_pool.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AutonomousEvolutionError("SWE-bench task pool is unreadable") from error
    runtime = document.get("runtime") if isinstance(document, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise AutonomousEvolutionError(
            "real autonomous execution requires task_pool.runtime metadata"
        )
    result = dict(runtime)
    result.setdefault(
        "legacy_root", str(request.config.swe_bench.official_evaluator.parent)
    )
    required = {
        "legacy_root",
        "routes_path",
        "swe_python",
        "multi_python",
        "multi_harness_root",
        "pool_root",
        "native_assets_path",
    }
    missing = required - set(result)
    if missing:
        raise AutonomousEvolutionError(
            "task_pool.runtime is missing: " + ", ".join(sorted(missing))
        )
    for name in required:
        if not Path(str(result[name])).expanduser().exists():
            raise AutonomousEvolutionError(f"task_pool.runtime path is missing: {name}")
    return result


def _generation_config(request: RoundExecutionRequest) -> dict[str, Any]:
    return {
        "temperature": 0,
        "seed": request.config.execution.seed,
        "thinking": False,
        "max_tokens": 1536,
        "max_context_tokens": 24000,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


__all__ = ["FreshFeedbackRoundExecutor"]
