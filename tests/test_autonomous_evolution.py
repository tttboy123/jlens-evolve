from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import pytest

from evolve.autonomous import (
    AutonomousEvolutionError,
    BaselineProbeResult,
    EvolutionDependencies,
    PrescreenResult,
)
from evolve.autonomous.output import load_best_harness, seal_manifest
from evolve.autonomous.runner import RoundExecutionRequest
from evolve.cli import main
from evolve.contracts import (
    Authorization,
    Cohort,
    ExecutionLimits,
    ExecutionPlan,
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
from evolve.kernel import CampaignController
from evolve.live_campaign import LiveCampaignSpec, run_skill_paired_campaign
from evolve.observers import (
    CostObserver,
    ExternalTraceObserver,
    NativeOutcomeObserver,
    ObserverHub,
)
from evolve.proposals import PricingCnyPerMillionTokens
from evolve.registry import CandidateRegistry, CapabilityRegistry, RejectedRegistry
from evolve.runtime import EvaluatorInfrastructureError, ExecutionRuntime
from evolve.strategies import SkillPairedStrategy

NOW = datetime(2026, 8, 14, 10, 0, tzinfo=UTC)
EVALUATOR_ID = "fixture-native-v1"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_worktree(tmp_path: Path) -> Path:
    root = tmp_path / "worktree"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "Tests")
    _git(root, "config", "user.email", "tests@example.invalid")
    (root / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "fixture")
    return root


def _config(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
    ):
        (model / name).write_text(name, encoding="utf-8")
    harness = tmp_path / "harness"
    harness.mkdir()
    evaluator = tmp_path / "official-evaluator.py"
    evaluator.write_text("# fixture evaluator\n", encoding="utf-8")
    sources = tmp_path / "sources"
    sources.mkdir()
    rows = []
    for index, project in enumerate(("sphinx", "django", "sympy", "pytest"), 1):
        source = sources / project
        source.mkdir()
        rows.append(
            {
                "instance_id": f"{project}__task-{index}",
                "project": project,
                "benchmark_id": "swe-bench-verified",
                "cohort": "feedback",
                "source_uri": str(source),
                "base_revision": f"{index:040x}",
                "catalog_fingerprint": f"{index:064x}",
            }
        )
    pool = tmp_path / "feedback-tasks.json"
    pool.write_text(json.dumps(rows), encoding="utf-8")
    config = {
        "schema_version": 1,
        "goal": {
            "goal_id": "offline-two-round-evolution",
            "description": "Exercise real receipts and claims through the public CLI.",
            "target_native_gains": 3,
            "max_rounds": 2,
            "no_progress_patience": 2,
        },
        "model": {
            "provider": "local-mlx",
            "model_path": str(model),
            "model_identity_files": [
                "config.json",
                "model.safetensors.index.json",
                "tokenizer_config.json",
            ],
        },
        "swe_bench": {
            "task_pool": str(pool),
            "source_pool": str(sources),
            "official_harness": str(harness),
            "official_evaluator": str(evaluator),
            "cohort": "feedback",
        },
        "teacher": {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "endpoint": "https://teacher.invalid/chat/completions",
            "api_key_env": "TEACHER_TEST_KEY",
            "budget_cny": 10.0,
            "max_output_tokens": 1024,
        },
        "execution": {
            "tasks_per_campaign": 3,
            "qwen_prescreen_count": 1,
            "native_finalist_count": 1,
            "seed": 7,
        },
    }
    path = tmp_path / "AUTONOMOUS-EVOLUTION-CONFIG.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


class FixtureWorkspace:
    def materialize(self, plan: ExecutionPlan) -> Mapping[str, Any]:
        return {"workspace_id": f"workspace-{plan.plan_id}"}


class FixtureModel:
    remote = False

    def __init__(self, request: RoundExecutionRequest) -> None:
        self.request = request

    def infer(
        self, plan: ExecutionPlan, workspace: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        patch = f"patch:{self.request.round_index}:{plan.task.task_id}:{plan.arm}"
        prompt = "SYSTEM: baseline repair"
        candidate_prompt = None
        candidate_prompt_sha = None
        compiled = {}
        if plan.arm == "taught":
            assert self.request.candidate is not None
            candidate_prompt = canonical_json(
                {
                    "candidate_revision_id": self.request.candidate.change_set.revision_id,
                    "candidate_bundle_sha256": self.request.candidate.bundle_sha256,
                }
            )
            candidate_prompt_sha = _sha(candidate_prompt)
            prompt = "SYSTEM: repair\nCOMPILED-CANDIDATE:\n" + candidate_prompt
            compiled = {
                "COMPILED-REVISION.json": self.request.candidate.bundle_sha256
            }
        return {
            "arm": plan.arm,
            "plan_id": plan.plan_id,
            "task_revision_id": plan.task.revision_id,
            "patch": patch,
            "patch_sha256": _sha(patch),
            "prompt_texts": [prompt],
            "prompt_sha256": [_sha(prompt)],
            "candidate_consumed": plan.arm == "taught",
            "candidate_revision_id": (
                self.request.candidate.change_set.revision_id
                if plan.arm == "taught" and self.request.candidate is not None
                else None
            ),
            "candidate_bundle_sha256": (
                self.request.candidate.bundle_sha256
                if plan.arm == "taught" and self.request.candidate is not None
                else None
            ),
            "candidate_prompt": candidate_prompt,
            "candidate_prompt_sha256": candidate_prompt_sha,
            "compiled_artifact_sha256": compiled,
            "structural_valid": True,
            "cost_cny": 0,
            "input_tokens": 10,
            "output_tokens": 5,
            "workspace_id": workspace["workspace_id"],
        }


class FixtureNative:
    evaluator_id = EVALUATOR_ID

    def __init__(self, round_index: int) -> None:
        self.round_index = round_index

    def evaluate(self, plan, workspace, model_output):
        resolved = self.round_index == 1 and plan.arm == "taught"
        return {
            "resolved": resolved,
            "exit_code": 0,
            "prediction_sha256": model_output["patch_sha256"],
            "native_error": None if resolved else "empty_patch",
            "native_report_sha256": _sha(f"native-report:{plan.plan_id}"),
            "official_receipt_sha256": _sha(f"official-receipt:{plan.plan_id}"),
        }


class MixedInfrastructureFixtureNative(FixtureNative):
    def __init__(self, round_index: int, failing_task_id: str) -> None:
        super().__init__(round_index)
        self.failing_task_id = failing_task_id

    def evaluate(self, plan, workspace, model_output):
        if plan.task.task_id == self.failing_task_id:
            raise EvaluatorInfrastructureError("fixture evaluator unavailable")
        return super().evaluate(plan, workspace, model_output)


def _tasks(request: RoundExecutionRequest) -> tuple[TaskRevision, ...]:
    return tuple(
        TaskRevision(
            task_id=str(row["instance_id"]),
            revision_id=f"feedback-{row['instance_id']}-r1",
            project=str(row["project"]),
            cohort=Cohort.FEEDBACK,
            source_sha256=_sha(str(row["instance_id"])),
            evaluator_id=EVALUATOR_ID,
            source_uri=str(row["source_uri"]),
        )
        for row in request.selection.tasks
    )


def _authorization(campaign_id: str, calls: int) -> Authorization:
    return Authorization(
        authorization_id=f"auth-{campaign_id}",
        campaign_id=campaign_id,
        allowed_cohorts=(Cohort.FEEDBACK,),
        max_cost_cny=0,
        max_model_calls=calls,
        expires_at=NOW + timedelta(hours=8),
        remote_calls_allowed=False,
    )


class ExecutableFixtureRoundExecutor:
    """Fake only external adapters; Runtime/Claim/Governance remain production."""

    def model_transport(self, request: RoundExecutionRequest) -> FixtureModel:
        return FixtureModel(request)

    def native_evaluator(self, request: RoundExecutionRequest) -> FixtureNative:
        return FixtureNative(request.round_index)

    def baseline(self, request: RoundExecutionRequest) -> BaselineProbeResult:
        task_ids = tuple(task.task_id for task in _tasks(request))
        return BaselineProbeResult(
            task_ids=task_ids,
            model_receipt_ids=tuple(f"baseline-model-{item}" for item in task_ids),
            native_receipt_ids=tuple(f"baseline-native-{item}" for item in task_ids),
            native_outcomes=tuple(
                {"task_id": item, "resolved": False} for item in task_ids
            ),
            failure_signatures=tuple(
                {"task_id": item, "signature": "native_unresolved"}
                for item in task_ids
            ),
            replayed=False,
        )

    def prescreen(self, request: RoundExecutionRequest) -> PrescreenResult:
        assert request.candidate is not None
        task = _tasks(request)[0]
        campaign_id = f"prescreen-{request.round_index}"
        strategy = SkillPairedStrategy()
        baseline, taught = strategy.build_plans(
            campaign_id=campaign_id,
            task=task,
            baseline_revision_id=request.baseline_revision_id,
            taught_revision_id=request.candidate.change_set.revision_id,
            model=ModelIdentity("local-mlx", "fixture-qwen", "frozen-r1"),
            context_policy_id="context-v1",
            tool_policy_id="tools-v1",
            observer_policy_ids=("external-trace-v1", "cost-v1"),
            limits=ExecutionLimits(256, 60, 0),
            plan_metadata={"execution_mode": "model-only-prescreen"},
        )
        del baseline
        store = ReceiptStore(request.output_root / "prescreen/receipt-store")
        graph = EvidenceGraph(request.output_root / "prescreen/evidence-graph")
        runtime = ExecutionRuntime(
            model_transport=self.model_transport(request),
            workspace_manager=FixtureWorkspace(),
            native_evaluator=self.native_evaluator(request),
            observer_hub=ObserverHub(
                (ExternalTraceObserver(), CostObserver()), graph=graph
            ),
            receipt_sink=store,
            clock=lambda: NOW,
        )
        result = runtime.execute(taught, _authorization(campaign_id, 1))
        model = next(receipt for receipt in result.receipts if receipt.kind == "model")
        return PrescreenResult(
            candidate_revision_id=request.candidate.change_set.revision_id,
            candidate_bundle_sha256=request.candidate.bundle_sha256,
            model_receipt_ids=(model.receipt_id,),
            structural_valid=bool(model.payload.get("structural_valid")),
            patch_applicable=(
                bool(model.payload.get("structural_valid"))
                and bool(str(model.payload.get("patch", "")).strip())
            ),
            replayed=result.replayed,
        )

    def paired(self, request: RoundExecutionRequest) -> Mapping[str, Any]:
        assert request.candidate is not None
        campaign_id = f"paired-{request.round_index}"
        authorization = _authorization(campaign_id, 6)
        controller = CampaignController.create(
            campaign_id=campaign_id,
            authorization=authorization,
            now=NOW,
        )
        store = ReceiptStore(request.output_root / "receipt-store")
        graph = EvidenceGraph(request.output_root / "evidence-graph")
        observer = ObserverHub(
            (NativeOutcomeObserver(), CostObserver(), ExternalTraceObserver()),
            graph=graph,
        )
        decision_log = PromotionDecisionLog(
            request.output_root / "registries/promotion-decisions.jsonl"
        )
        tasks = _tasks(request)
        spec = LiveCampaignSpec(
            campaign_id=campaign_id,
            baseline_revision_id=request.baseline_revision_id,
            candidate_id=request.candidate.change_set.candidate_id,
            candidate_revision_id=request.candidate.change_set.revision_id,
            candidate_kind="external-skill",
            candidate_artifact_sha256=request.candidate.bundle_sha256,
            model=ModelIdentity("local-mlx", "fixture-qwen", "frozen-r1"),
            context_policy_id="context-v1",
            tool_policy_id="tools-v1",
            observer_policy_ids=(
                "external-trace-v1",
                "native-v1",
                "cost-v1",
            ),
            limits=ExecutionLimits(256, 60, 0),
            final_commit_sha=request.source_commit_sha,
            generation_config={"temperature": 0, "seed": 7},
            task_execution_metadata={
                task.revision_id: {
                    "base_revision": "1" * 40,
                    "benchmark_id": "swe-bench-verified",
                    "instance_id": task.task_id,
                }
                for task in tasks
            },
        )
        result = run_skill_paired_campaign(
            spec=spec,
            tasks=tasks,
            strategy=SkillPairedStrategy(),
            controller=controller,
            authorization=authorization,
            model_transport=self.model_transport(request),
            workspace_manager=FixtureWorkspace(),
            native_evaluator=self.native_evaluator(request),
            receipt_store=store,
            observer_hub=observer,
            claim_engine=ClaimEngine(graph),
            evidence_grade_machine=EvidenceGradeMachine(graph),
            governance_service=GovernanceService(),
            promotion_decision_log=decision_log,
            candidate_registry=CandidateRegistry(
                request.output_root / "registries/candidates.jsonl"
            ),
            capability_registry=CapabilityRegistry(
                request.output_root / "registries/capabilities.jsonl",
                decision_log=decision_log,
            ),
            rejected_registry=RejectedRegistry(
                request.output_root / "registries/rejected.jsonl",
                decision_log=decision_log,
            ),
            report_root=request.output_root,
            clock=lambda: NOW,
        )
        return {
            "campaign_id": campaign_id,
            "campaign_status": str(result.snapshot.status),
            "execution_statuses": [execution.status for execution in result.executions],
            "claims": [
                {
                    "task_id": task.task_id,
                    "task_revision_id": task.revision_id,
                    "classification": str(claim.classification),
                    "claim_id": claim.claim_id,
                    "grade": str(claim.grade),
                    "counterfactual_pair_sha256": claim.counterfactual_pair_sha256,
                    "counterfactual_receipt_ids": list(
                        claim.counterfactual_receipt_ids
                    ),
                }
                for task, claim in zip(tasks, result.claims, strict=True)
            ],
            "capability_active": False,
            "holdout_opened": False,
            "burned_holdout_opened": False,
        }


class StructuralFailureFixtureModel(FixtureModel):
    def infer(
        self, plan: ExecutionPlan, workspace: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        result = dict(super().infer(plan, workspace))
        if (
            self.request.round_index == 0
            and plan.arm == "taught"
            and plan.metadata.get("execution_mode", "full") == "full"
        ):
            result["structural_valid"] = False
            result["failure_reason"] = "malformed-hunk"
        return result


class StructuralFailureFixtureRoundExecutor(ExecutableFixtureRoundExecutor):
    def model_transport(
        self, request: RoundExecutionRequest
    ) -> StructuralFailureFixtureModel:
        return StructuralFailureFixtureModel(request)


class PrescreenFailureFixtureModel(FixtureModel):
    def infer(
        self, plan: ExecutionPlan, workspace: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        result = dict(super().infer(plan, workspace))
        if plan.metadata.get("execution_mode") == "model-only-prescreen":
            result["structural_valid"] = False
            result["failure_reason"] = "selector-no-match"
        return result


class PrescreenFailureFixtureRoundExecutor(ExecutableFixtureRoundExecutor):
    def model_transport(
        self, request: RoundExecutionRequest
    ) -> PrescreenFailureFixtureModel:
        return PrescreenFailureFixtureModel(request)


SENSITIVE_FAILURE = "/private/eval/gold/holdout/system-prompt.txt"


class SensitiveFailureFixtureModel(StructuralFailureFixtureModel):
    def infer(
        self, plan: ExecutionPlan, workspace: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        result = dict(super().infer(plan, workspace))
        if result.get("failure_reason") is not None:
            result["failure_reason"] = SENSITIVE_FAILURE
        return result


class SensitiveFailureFixtureRoundExecutor(ExecutableFixtureRoundExecutor):
    def model_transport(
        self, request: RoundExecutionRequest
    ) -> SensitiveFailureFixtureModel:
        return SensitiveFailureFixtureModel(request)


class TamperingFixtureRoundExecutor(ExecutableFixtureRoundExecutor):
    def paired(self, request: RoundExecutionRequest) -> Mapping[str, Any]:
        result = dict(super().paired(request))
        claims = [dict(row) for row in result["claims"]]
        claims[0]["task_revision_id"] = "forged-task-revision"
        result["claims"] = claims
        return result


class MixedInfrastructureFixtureRoundExecutor(ExecutableFixtureRoundExecutor):
    def native_evaluator(
        self, request: RoundExecutionRequest
    ) -> MixedInfrastructureFixtureNative:
        return MixedInfrastructureFixtureNative(
            request.round_index,
            request.selection.selected_task_ids[0],
        )


class PhaseInterruptingRoundExecutor:
    def __init__(self, phase: str) -> None:
        self.phase = phase
        self.interrupted = False
        self.delegate = ExecutableFixtureRoundExecutor()

    def baseline(self, request: RoundExecutionRequest) -> BaselineProbeResult:
        return self.delegate.baseline(request)

    def prescreen(self, request: RoundExecutionRequest) -> PrescreenResult:
        if self.phase == "candidate-compiled" and not self.interrupted:
            self.interrupted = True
            raise KeyboardInterrupt("after candidate compilation")
        result = self.delegate.prescreen(request)
        if self.phase == "qwen-run" and not self.interrupted:
            self.interrupted = True
            raise KeyboardInterrupt("after Qwen receipt")
        return result

    def paired(self, request: RoundExecutionRequest) -> Mapping[str, Any]:
        result = self.delegate.paired(request)
        if self.phase == "native-run" and not self.interrupted:
            self.interrupted = True
            raise KeyboardInterrupt("after native receipts")
        return result


def _teacher(
    teacher_requests: list[dict[str, object]], request: dict[str, object]
) -> dict[str, object]:
    teacher_requests.append(request)
    failure = request["failure_package"]
    assert isinstance(failure, dict)
    tasks = failure["selected_tasks"]
    assert isinstance(tasks, list)
    routes = {str(task["instance_id"]): "operator-localize" for task in tasks}
    candidate = {
        "protocol": "skill-harness-v2",
        "prompt_template": "Use the compiled localization harness.",
        "skill_text": "Find the declared symbol before selecting an edit.",
        "operator": {
            "id": "operator-localize",
            "kind": "zero-arg",
            "arguments": [],
            "instruction": "Localize the declaration, then emit one patch.",
        },
        "router": {"routes": routes},
        "memory_policy": None,
        "preconditions": ["feedback task only"],
        "expected_external_effect": {"patch": "more targeted"},
        "expected_internal_effect": {"localization": "earlier"},
        "falsification": {"native": "no gain or regression"},
        "eval_note": "Use only official native paired claims.",
    }
    return {
        "model": "deepseek-v4-flash",
        "choices": [{"message": {"content": json.dumps(candidate)}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


def test_public_cli_runs_two_real_feedback_rounds_and_replays_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []

    def teacher(request: dict[str, object]) -> dict[str, object]:
        return _teacher(teacher_requests, request)

    dependencies = EvolutionDependencies(
        teacher_transport=teacher,
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=ExecutableFixtureRoundExecutor(),
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    config_path = _config(tmp_path)
    config_payload = json.loads(config_path.read_text())
    task_pool = Path(config_payload["swe_bench"]["task_pool"])
    task_rows = json.loads(task_pool.read_text())
    task_rows[0]["future_private_checkout"] = "/private/internal/task.db"
    task_pool.write_text(json.dumps(task_rows), encoding="utf-8")
    output = tmp_path / "output"

    assert main(
        [
            "autonomous-evolve",
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--worktree-root",
            str(worktree),
        ]
    ) == 0
    first_stdout = capsys.readouterr().out
    assert "goal_reached" in first_stdout
    assert len(teacher_requests) == 2
    second_failure = teacher_requests[1]["failure_package"]
    assert isinstance(second_failure, dict)
    assert [
        row["classification"] for row in second_failure["prior_claims"]
    ] == ["neutral", "neutral", "neutral"]
    assert (output / "best/BEST-HARNESS.json").is_file()
    best = load_best_harness(
        output / "best/BEST-HARNESS.json",
        expected_model_identity_sha256=json.loads(
            (output / "GOAL.json").read_text()
        )["model_identity_sha256"],
    )
    assert best is not None
    assert best.change_set.revision_id == json.loads(
        (output / "EVOLUTION-RESULT.json").read_text()
    )["best_candidate_revision_id"]
    assert (output / "EVIDENCE-MANIFEST.json").is_file()
    assert json.loads((output / "EVOLUTION-RESULT.json").read_text())["product_status"] == (
        "offline_e2e_verified"
    )
    first_round = output / "rounds/round-0000"
    preflight = json.loads((first_round / "PREFLIGHT-HEALTH.json").read_text())
    assert preflight["status"] == "healthy"
    assert set(preflight["components"]) == {"teacher", "qwen", "native"}
    assert all(
        len(component["identity_sha256"]) == 64
        for component in preflight["components"].values()
    )
    assert (first_round / "TEACHER-REQUEST.json").is_file()
    assert (first_round / "TEACHER-RESPONSE.json").is_file()
    assert (first_round / "CAMPAIGN-RESULT.json").is_file()
    selection = json.loads((first_round / "TASK-SELECTION.json").read_text())
    assert len(selection["tasks"]) == 3
    assert all("task_fingerprint_sha256" in task for task in selection["tasks"])
    assert len(selection["selection_context_sha256"]) == 64
    assert selection["selection_context"]["mode"] == "stateful-v1"
    assert all(
        "future_private_checkout" not in task
        for task in teacher_requests[0]["failure_package"]["selected_tasks"]
    )

    assert main(
        [
            "autonomous-evolve",
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--worktree-root",
            str(worktree),
        ]
    ) == 0
    capsys.readouterr()
    assert len(teacher_requests) == 2


def test_router_coverage_repair_records_paid_call_and_resumes_without_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A schema-valid but under-covered Router is repaired, not blocked.

    The Teacher response stays untouched; the compiled Router is deterministically
    extended to every selected feedback task so the paid call still produces a
    real taught campaign. The fail-closed gate remains for genuinely invalid
    Routers (covered by the following test).
    """

    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []

    def incomplete_router_teacher(request: dict[str, object]) -> dict[str, object]:
        raw = _teacher(teacher_requests, request)
        choices = raw["choices"]
        assert isinstance(choices, list)
        message = choices[0]["message"]
        assert isinstance(message, dict)
        content = json.loads(str(message["content"]))
        routes = content["router"]["routes"]
        routes.pop(next(iter(routes)))
        message["content"] = json.dumps(content)
        return raw

    dependencies = EvolutionDependencies(
        teacher_transport=incomplete_router_teacher,
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=ExecutableFixtureRoundExecutor(),
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    output = tmp_path / "router-coverage-repair"
    argv = [
        "autonomous-evolve",
        "--config",
        str(_config(tmp_path)),
        "--output",
        str(output),
        "--worktree-root",
        str(worktree),
    ]

    assert main(argv) == 0
    result = json.loads((output / "EVOLUTION-RESULT.json").read_text())
    assert result["status"] == "goal_reached"
    assert result["rounds_completed"] == 2
    assert len(teacher_requests) == 2
    assert not (output / "rounds/round-0001/INTEGRITY-BLOCK.json").is_file()

    repair = json.loads(
        (output / "rounds/round-0001/ROUTER-REPAIR.json").read_text()
    )
    assert repair["schema_version"] == 1
    assert repair["round_index"] == 1
    assert len(repair["teacher_routed_task_ids"]) == 2
    assert len(repair["synthesized_task_ids"]) == 1
    selected = json.loads(
        (output / "rounds/round-0001/TASK-SELECTION.json").read_text()
    )["selected_task_ids"]
    assert set(repair["teacher_routed_task_ids"] + repair["synthesized_task_ids"]) == set(
        selected
    )
    compiled_router = json.loads(
        next(
            (output / "rounds/round-0001/compiled-candidates").glob(
                "*/COMPILED-ROUTER.json"
            )
        ).read_text()
    )
    assert set(route[0] for route in compiled_router["routes"]) == set(selected)
    assert {route[1] for route in compiled_router["routes"]} == {"operator-localize"}

    teacher_ledger = output / "TEACHER-CALL-LEDGER.jsonl"
    cost_ledger = output / "teacher/COST-LEDGER.jsonl"
    ledger_before = teacher_ledger.read_bytes()
    cost_before = cost_ledger.read_bytes()
    assert len(ledger_before.splitlines()) == 2
    second_call = json.loads(ledger_before.splitlines()[1])
    second_payload = second_call["payload"]
    assert second_payload["candidate_revision_id"].endswith("-r0001")
    normalized_response = output / second_payload["response_path"]
    normalized = json.loads(normalized_response.read_text())
    raw_response = normalized_response.with_name("TEACHER-RAW-RESPONSE.json")
    assert second_payload["request_sha256"] == _sha(
        (output / second_payload["request_path"]).read_text()
    )
    assert second_payload["response_sha256"] == hashlib.sha256(
        normalized_response.read_bytes()
    ).hexdigest()
    assert normalized["raw_response_sha256"] == hashlib.sha256(
        raw_response.read_bytes()
    ).hexdigest()

    assert main(argv) == 0
    assert len(teacher_requests) == 2
    assert teacher_ledger.read_bytes() == ledger_before
    assert cost_ledger.read_bytes() == cost_before


def test_invalid_router_reference_fails_closed_and_records_paid_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Router referencing another Operator still fails closed with the call indexed."""

    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []

    def foreign_operator_teacher(request: dict[str, object]) -> dict[str, object]:
        raw = _teacher(teacher_requests, request)
        choices = raw["choices"]
        assert isinstance(choices, list)
        message = choices[0]["message"]
        assert isinstance(message, dict)
        content = json.loads(str(message["content"]))
        routes = content["router"]["routes"]
        assert isinstance(routes, dict)
        content["router"] = {
            "routes": {
                task_id: "operator-not-the-candidate"
                for task_id in routes
            }
        }
        message["content"] = json.dumps(content)
        return raw

    dependencies = EvolutionDependencies(
        teacher_transport=foreign_operator_teacher,
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=ExecutableFixtureRoundExecutor(),
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    output = tmp_path / "router-coverage-failure"
    argv = [
        "autonomous-evolve",
        "--config",
        str(_config(tmp_path)),
        "--output",
        str(output),
        "--worktree-root",
        str(worktree),
    ]

    assert main(argv) == 0
    result = json.loads((output / "EVOLUTION-RESULT.json").read_text())
    assert result["status"] == "blocked_integrity"
    block = json.loads(
        (output / "rounds/round-0000/INTEGRITY-BLOCK.json").read_text()
    )
    assert block["phase"] == "candidate-compilation"
    assert len(teacher_requests) == 1
    # A compile-time contract failure has no compiled revision to index, so the
    # authoritative charge record is the durable cost ledger plus the frozen
    # request/response triplets; the loop stays fail-closed and never re-dispatches.
    cost_ledger = output / "teacher/COST-LEDGER.jsonl"
    cost_before = cost_ledger.read_bytes()
    assert len(cost_before.splitlines()) == 3
    assert b'"event_id":"ledger-open"' in cost_before
    assert b'"event_id":"teacher-result:' in cost_before
    teacher_artifacts = list(
        (output / "teacher").glob("*/TEACHER-REQUEST.json")
    )
    assert len(teacher_artifacts) == 1
    assert list((output / "teacher").glob("*/TEACHER-RAW-RESPONSE.json"))
    assert list((output / "teacher").glob("*/TEACHER-RESPONSE.json"))
    assert not (output / "TEACHER-CALL-LEDGER.jsonl").exists()
    assert not (output / "rounds/round-0000/compiled-candidates").exists()

    assert main(argv) == 0
    assert len(teacher_requests) == 1
    assert cost_ledger.read_bytes() == cost_before


def test_public_cli_selection_binds_complete_history_and_resume_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []
    dependencies = EvolutionDependencies(
        teacher_transport=lambda request: _teacher(teacher_requests, request),
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=StructuralFailureFixtureRoundExecutor(),
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    config_path = _config(tmp_path)
    config = json.loads(config_path.read_text())
    config["goal"]["max_rounds"] = 3
    config["goal"]["no_progress_patience"] = 3
    config["goal"]["max_same_failure_signature"] = 4
    config["goal"]["target_native_gains"] = 99
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "history-bound-selection"
    argv = [
        "autonomous-evolve",
        "--config",
        str(config_path),
        "--output",
        str(output),
        "--worktree-root",
        str(worktree),
    ]

    assert main(argv) == 0
    assert len(teacher_requests) == 3
    third = json.loads(
        (output / "rounds/round-0002/TASK-SELECTION.json").read_text()
    )
    context = third["selection_context"]
    assert len(context["historical_claims"]) == 6
    assert sum(context["task_selection_counts"].values()) == 6
    assert sum(context["failure_signature_counts"].values()) == 3
    assert context["current_best_revision_id"] is not None
    assert context["goal_gap"] == 96
    preserved = canonical_json(third)

    assert main(argv) == 0
    assert canonical_json(
        json.loads((output / "rounds/round-0002/TASK-SELECTION.json").read_text())
    ) == preserved
    assert len(teacher_requests) == 3


def test_public_cli_blocks_tampered_best_projection_before_teacher_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evolve.autonomous.goal import GoalStateStore

    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []
    dependencies = EvolutionDependencies(
        teacher_transport=lambda request: _teacher(teacher_requests, request),
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=ExecutableFixtureRoundExecutor(),
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    original_write = GoalStateStore.write
    interrupted = False

    def interrupt_state_write(self, state):
        nonlocal interrupted
        if state.rounds_completed == 1 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("after sealed accepted round")
        return original_write(self, state)

    monkeypatch.setattr(GoalStateStore, "write", interrupt_state_write)
    output = tmp_path / "tampered-best-resume"
    argv = [
        "autonomous-evolve",
        "--config",
        str(_config(tmp_path)),
        "--output",
        str(output),
        "--worktree-root",
        str(worktree),
    ]
    with pytest.raises(KeyboardInterrupt):
        main(argv)
    best_path = output / "best/BEST-HARNESS.json"
    best = json.loads(best_path.read_text())
    best["supported_task_signatures"] = ["forged-task"]
    best_path.write_text(json.dumps(best), encoding="utf-8")

    assert main(argv) == 0
    result = json.loads((output / "EVOLUTION-RESULT.json").read_text())
    assert result["status"] == "blocked_integrity"
    assert len(teacher_requests) == 1
    block = json.loads(
        (output / "rounds/round-0001/INTEGRITY-BLOCK.json").read_text()
    )
    assert block["phase"] == "best-harness-projection"


def test_public_cli_rejects_resealed_terminal_best_projection_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []
    dependencies = EvolutionDependencies(
        teacher_transport=lambda request: _teacher(teacher_requests, request),
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=ExecutableFixtureRoundExecutor(),
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    output = tmp_path / "terminal-best-forgery"
    argv = [
        "autonomous-evolve",
        "--config",
        str(_config(tmp_path)),
        "--output",
        str(output),
        "--worktree-root",
        str(worktree),
    ]
    assert main(argv) == 0
    assert len(teacher_requests) == 2
    best_path = output / "best/BEST-HARNESS.json"
    best = json.loads(best_path.read_text())
    best["source_claim_ids"] = ["forged-claim"]
    best_path.write_text(json.dumps(best), encoding="utf-8")
    seal_manifest(output)

    assert main(argv) == 0
    result = json.loads((output / "EVOLUTION-RESULT.json").read_text())
    assert result["status"] == "blocked_integrity"
    assert len(teacher_requests) == 2
    block = json.loads(
        (output / "rounds/round-0002/INTEGRITY-BLOCK.json").read_text()
    )
    assert block["phase"] == "best-harness-projection"


def test_public_cli_rejects_unindexed_best_after_crash_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evolve.autonomous.runner as runner_module

    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []
    dependencies = EvolutionDependencies(
        teacher_transport=lambda request: _teacher(teacher_requests, request),
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=ExecutableFixtureRoundExecutor(),
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    original_freeze = runner_module.freeze_json
    interrupted = False

    def interrupt_round_result(path: Path, payload: Mapping[str, Any]) -> Path:
        nonlocal interrupted
        if (
            path.name == "AUTONOMOUS-ROUND-RESULT.json"
            and path.parent.name == "round-0001"
            and not interrupted
        ):
            interrupted = True
            raise KeyboardInterrupt("after BEST projection before round authority")
        return original_freeze(path, payload)

    monkeypatch.setattr(runner_module, "freeze_json", interrupt_round_result)
    output = tmp_path / "unindexed-best-crash"
    argv = [
        "autonomous-evolve",
        "--config",
        str(_config(tmp_path)),
        "--output",
        str(output),
        "--worktree-root",
        str(worktree),
    ]
    with pytest.raises(KeyboardInterrupt):
        main(argv)
    assert json.loads(
        (output / "best/BEST-HARNESS.json").read_text()
    )["harness_kind"] == "compiled-candidate"

    assert main(argv) == 0
    result = json.loads((output / "EVOLUTION-RESULT.json").read_text())
    assert result["status"] == "blocked_integrity"
    assert len(teacher_requests) == 2
    block = json.loads(
        (output / "rounds/round-0001/INTEGRITY-BLOCK.json").read_text()
    )
    assert block["phase"] == "best-harness-projection"


def test_public_cli_carries_authoritative_paired_failures_into_next_teacher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []
    dependencies = EvolutionDependencies(
        teacher_transport=lambda request: _teacher(teacher_requests, request),
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=StructuralFailureFixtureRoundExecutor(),
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    output = tmp_path / "paired-failure-feedback-output"

    assert main(
        [
            "autonomous-evolve",
            "--config",
            str(_config(tmp_path)),
            "--output",
            str(output),
            "--worktree-root",
            str(worktree),
        ]
    ) == 0
    capsys.readouterr()

    round_zero = json.loads(
        (output / "rounds/round-0000/AUTONOMOUS-ROUND-RESULT.json").read_text()
    )
    campaign_feedback = round_zero["campaign_feedback"]
    assert campaign_feedback["campaign_id"] == "paired-0"
    assert campaign_feedback["campaign_status"] == "completed"
    assert len(campaign_feedback["task_pairs"]) == 3
    for pair in campaign_feedback["task_pairs"]:
        assert pair["claim"]["classification"] == "neutral"
        assert len(pair["claim"]["counterfactual_receipt_ids"]) == 6
        baseline = pair["baseline"]
        taught = pair["taught"]
        for arm in (baseline, taught):
            assert arm["plan_id"]
            assert arm["execution_status"] == "completed"
            assert arm["model_receipt_id"]
            assert arm["external_trace_receipt_id"]
            assert arm["native_receipt_id"]
            assert arm["execution_terminal_receipt_id"]
            assert len(arm["patch_sha256"]) == 64
            assert len(arm["native_report_sha256"]) == 64
            assert len(arm["official_receipt_sha256"]) == 64
            assert arm["resolved"] is False
            assert arm["native_error"] == "empty_patch"
        assert baseline["candidate_consumed"] is False
        assert baseline["candidate_revision_id"] is None
        assert baseline["candidate_bundle_sha256"] is None
        assert baseline["structural_valid"] is True
        assert baseline["failure_reason"] is None
        assert taught["candidate_consumed"] is True
        assert taught["candidate_revision_id"] == round_zero["candidate_revision_id"]
        assert taught["candidate_bundle_sha256"] == round_zero["compiled_bundle_sha256"]
        assert taught["structural_valid"] is False
        assert taught["failure_reason"] == "malformed-hunk"

    serialized_feedback = canonical_json(campaign_feedback)
    assert '"patch":' not in serialized_feedback
    assert "prompt_texts" not in serialized_feedback
    assert "candidate_prompt" not in serialized_feedback
    assert "gold" not in serialized_feedback.casefold()

    next_failure_package = json.loads(
        (output / "rounds/round-0001/FAILURE-PACKAGE.json").read_text()
    )
    next_teacher_request = json.loads(
        (output / "rounds/round-0001/TEACHER-REQUEST.json").read_text()
    )
    assert next_failure_package["prior_campaign_feedback"] == campaign_feedback
    assert (
        next_teacher_request["failure_package"]["prior_campaign_feedback"]
        == campaign_feedback
    )
    dispatched_failure_package = teacher_requests[1]["failure_package"]
    assert isinstance(dispatched_failure_package, dict)
    assert dispatched_failure_package["no_progress_rounds"] == 1
    assert all(
        "source_uri" not in task and "source_repository" not in task
        for task in dispatched_failure_package["selected_tasks"]
    )
    assert (
        dispatched_failure_package["prior_campaign_feedback"] == campaign_feedback
    )
    baseline_only_signature = _sha(
        canonical_json(
            json.loads(
                (output / "rounds/round-0000/BASELINE-RESULT.json").read_text()
            )["failure_signatures"]
        )
    )
    assert round_zero["failure_signature_sha256"] != baseline_only_signature


def test_public_cli_carries_prescreen_rejection_into_next_teacher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []
    dependencies = EvolutionDependencies(
        teacher_transport=lambda request: _teacher(teacher_requests, request),
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=PrescreenFailureFixtureRoundExecutor(),
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    output = tmp_path / "prescreen-failure-output"

    assert main(
        [
            "autonomous-evolve",
            "--config",
            str(_config(tmp_path)),
            "--output",
            str(output),
            "--worktree-root",
            str(worktree),
        ]
    ) == 0

    assert len(teacher_requests) == 2
    prior = teacher_requests[1]["failure_package"]["prior_campaign_feedback"]
    assert prior["campaign_status"] == "screened_out"
    assert prior["task_pairs"] == []
    prescreen = prior["prescreen"]
    assert prescreen["candidate_revision_id"]
    assert len(prescreen["candidate_bundle_sha256"]) == 64
    assert prescreen["structural_valid"] is False
    assert prescreen["patch_applicable"] is False
    assert prescreen["failure_reason"] == "selector-no-match"
    assert prescreen["model_receipt_id"]
    assert prescreen["external_trace_receipt_id"]


def test_public_cli_rebuilds_legacy_round_feedback_before_teacher_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evolve.autonomous.goal import GoalStateStore

    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []
    dependencies = EvolutionDependencies(
        teacher_transport=lambda request: _teacher(teacher_requests, request),
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=ExecutableFixtureRoundExecutor(),
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    original_write = GoalStateStore.write
    interrupted = False

    def interrupt_after_index(self, state):
        nonlocal interrupted
        if state.rounds_completed == 1 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("after sealed legacy round index")
        return original_write(self, state)

    monkeypatch.setattr(GoalStateStore, "write", interrupt_after_index)
    output = tmp_path / "legacy-feedback-resume"
    argv = [
        "autonomous-evolve",
        "--config",
        str(_config(tmp_path)),
        "--output",
        str(output),
        "--worktree-root",
        str(worktree),
    ]
    with pytest.raises(KeyboardInterrupt):
        main(argv)

    index_path = output / "ROUND-INDEX.jsonl"
    stored = json.loads(index_path.read_text())
    stored["payload"].pop("campaign_feedback")
    event = {key: value for key, value in stored.items() if key != "event_sha256"}
    stored["event_sha256"] = _sha(canonical_json(event))
    index_path.write_text(canonical_json(stored) + "\n", encoding="utf-8")

    assert main(argv) == 0
    assert len(teacher_requests) == 2
    prior = teacher_requests[1]["failure_package"]["prior_campaign_feedback"]
    assert len(prior["task_pairs"]) == 3
    assert all(pair["claim"]["grade"] == "E2" for pair in prior["task_pairs"])


def test_public_cli_hashes_unknown_sensitive_failure_text_before_teacher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []
    dependencies = EvolutionDependencies(
        teacher_transport=lambda request: _teacher(teacher_requests, request),
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=SensitiveFailureFixtureRoundExecutor(),
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    output = tmp_path / "sensitive-failure-output"

    assert main(
        [
            "autonomous-evolve",
            "--config",
            str(_config(tmp_path)),
            "--output",
            str(output),
            "--worktree-root",
            str(worktree),
        ]
    ) == 0
    capsys.readouterr()

    round_zero = json.loads(
        (output / "rounds/round-0000/AUTONOMOUS-ROUND-RESULT.json").read_text()
    )
    for pair in round_zero["campaign_feedback"]["task_pairs"]:
        assert pair["taught"]["failure_reason"] == "other"
        assert pair["taught"]["failure_reason_raw_sha256"] == _sha(
            SENSITIVE_FAILURE
        )
    second_failure = teacher_requests[1]["failure_package"]
    assert isinstance(second_failure, dict)
    serialized = canonical_json(second_failure["prior_campaign_feedback"])
    assert SENSITIVE_FAILURE not in serialized
    assert "gold" not in serialized.casefold()
    assert "holdout" not in serialized.casefold()
    assert "prompt" not in serialized.casefold()


def test_public_cli_blocks_cross_task_counterfactual_receipts_before_next_teacher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evolve.autonomous.verification import CampaignOutcomeVerifier

    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []
    dependencies = EvolutionDependencies(
        teacher_transport=lambda request: _teacher(teacher_requests, request),
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=ExecutableFixtureRoundExecutor(),
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    original_verify = CampaignOutcomeVerifier.verify

    def cross_bind(self, **kwargs):
        verified = list(original_verify(self, **kwargs))
        verified[0] = replace(
            verified[0],
            counterfactual_receipt_ids=verified[1].counterfactual_receipt_ids,
        )
        return tuple(verified)

    monkeypatch.setattr(CampaignOutcomeVerifier, "verify", cross_bind)
    output = tmp_path / "cross-task-lineage-output"

    assert main(
        [
            "autonomous-evolve",
            "--config",
            str(_config(tmp_path)),
            "--output",
            str(output),
            "--worktree-root",
            str(worktree),
        ]
    ) == 0

    result = json.loads((output / "EVOLUTION-RESULT.json").read_text())
    assert result["status"] == "blocked_integrity"
    assert len(teacher_requests) == 1
    assert not (output / "rounds/round-0001/TEACHER-REQUEST.json").exists()


def test_candidate_task_lineage_tamper_blocks_integrity_and_never_advances_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []
    dependencies = EvolutionDependencies(
        teacher_transport=lambda request: _teacher(teacher_requests, request),
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=TamperingFixtureRoundExecutor(),
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    output = tmp_path / "tampered-output"

    assert main(
        [
            "autonomous-evolve",
            "--config",
            str(_config(tmp_path)),
            "--output",
            str(output),
            "--worktree-root",
            str(worktree),
        ]
    ) == 0

    result = json.loads((output / "EVOLUTION-RESULT.json").read_text())
    assert result["status"] == "blocked_integrity"
    assert result["best_candidate_revision_id"] is None
    best_path = output / "best/BEST-HARNESS.json"
    assert best_path.is_file()
    assert load_best_harness(
        best_path,
        expected_model_identity_sha256=json.loads(
            (output / "GOAL.json").read_text()
        )["model_identity_sha256"],
    ) is None
    assert (output / "rounds/round-0000/INTEGRITY-BLOCK.json").is_file()


def test_public_cli_stops_at_disk_limit_before_teacher_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []
    dependencies = EvolutionDependencies(
        teacher_transport=lambda request: _teacher(teacher_requests, request),
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=ExecutableFixtureRoundExecutor(),
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    config_path = _config(tmp_path)
    config = json.loads(config_path.read_text())
    config["goal"]["disk_limit_bytes"] = 1
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "disk-limited-output"

    assert main(
        [
            "autonomous-evolve",
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--worktree-root",
            str(worktree),
        ]
    ) == 0

    assert teacher_requests == []
    result = json.loads((output / "EVOLUTION-RESULT.json").read_text())
    assert result["status"] == "disk_limit"
    assert result["rounds_completed"] == 0


def test_public_cli_stops_on_repeated_failure_signature_without_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []
    dependencies = EvolutionDependencies(
        teacher_transport=lambda request: _teacher(teacher_requests, request),
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=ExecutableFixtureRoundExecutor(),
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    config_path = _config(tmp_path)
    config = json.loads(config_path.read_text())
    config["goal"]["max_same_failure_signature"] = 1
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "same-failure-output"

    assert main(
        [
            "autonomous-evolve",
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--worktree-root",
            str(worktree),
        ]
    ) == 0

    result = json.loads((output / "EVOLUTION-RESULT.json").read_text())
    assert result["status"] == "max_same_failure_signature"
    assert result["rounds_completed"] == 1
    assert len(teacher_requests) == 1


def test_public_cli_stops_on_authoritative_infra_failures_and_recovers_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evolve.autonomous.goal import GoalStateStore

    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []
    dependencies = EvolutionDependencies(
        teacher_transport=lambda request: _teacher(teacher_requests, request),
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=MixedInfrastructureFixtureRoundExecutor(),
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    config_path = _config(tmp_path)
    config = json.loads(config_path.read_text())
    config["goal"].update(
        {
            "target_native_gains": 99,
            "max_rounds": 10,
            "no_progress_patience": 10,
            "max_same_failure_signature": 10,
            "max_consecutive_infra_failures": 2,
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    original_write = GoalStateStore.write
    interrupted = False

    def interrupt_after_first_checkpoint(self, state):
        nonlocal interrupted
        if state.rounds_completed == 1 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("after first infra checkpoint")
        return original_write(self, state)

    monkeypatch.setattr(GoalStateStore, "write", interrupt_after_first_checkpoint)
    output = tmp_path / "infra-threshold-resume"
    argv = [
        "autonomous-evolve",
        "--config",
        str(config_path),
        "--output",
        str(output),
        "--worktree-root",
        str(worktree),
    ]

    with pytest.raises(KeyboardInterrupt):
        main(argv)
    assert main(argv) == 0

    result = json.loads((output / "EVOLUTION-RESULT.json").read_text())
    state = json.loads((output / "EVOLUTION-STATE.json").read_text())
    assert result["status"] == "max_consecutive_infra_failures"
    assert result["rounds_completed"] == 2
    assert result["native_gain_task_ids"] == []
    assert result["best_candidate_revision_id"] is None
    assert state["consecutive_infra_failures"] == 2
    assert len(teacher_requests) == 2
    resumed_failure_package = teacher_requests[1]["failure_package"]
    assert isinstance(resumed_failure_package, dict)
    prior_feedback = resumed_failure_package["prior_campaign_feedback"]
    assert isinstance(prior_feedback, dict)
    assert prior_feedback["campaign_status"] == "infra_failure"
    assert prior_feedback["infrastructure_failures"]
    for round_index in range(2):
        round_result = json.loads(
            (
                output
                / f"rounds/round-{round_index:04d}/AUTONOMOUS-ROUND-RESULT.json"
            ).read_text()
        )
        assert round_result["round_outcome"] == "infra_failure"
        assert round_result["accepted_as_best"] is False
        assert round_result["claims"] == []
        assert round_result["campaign_feedback"]["infrastructure_failures"]


def test_public_cli_rejects_tampered_preflight_before_external_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []
    executor = PhaseInterruptingRoundExecutor("candidate-compiled")
    dependencies = EvolutionDependencies(
        teacher_transport=lambda request: _teacher(teacher_requests, request),
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=executor,
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    output = tmp_path / "preflight-tamper"
    argv = [
        "autonomous-evolve",
        "--config",
        str(_config(tmp_path)),
        "--output",
        str(output),
        "--worktree-root",
        str(worktree),
    ]
    with pytest.raises(KeyboardInterrupt):
        main(argv)
    assert len(teacher_requests) == 1
    preflight_path = output / "rounds/round-0000/PREFLIGHT-HEALTH.json"
    preflight = json.loads(preflight_path.read_text())
    preflight["components"]["qwen"]["identity_sha256"] = "0" * 64
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

    with pytest.raises(AutonomousEvolutionError, match="immutable .* artifact"):
        main(argv)
    assert len(teacher_requests) == 1


@pytest.mark.parametrize(
    "phase",
    (
        "teacher-frozen",
        "candidate-compiled",
        "qwen-run",
        "native-run",
        "round-completed",
    ),
)
def test_public_cli_resumes_each_persistent_phase_without_repeating_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    from evolve.autonomous.goal import GoalStateStore
    from evolve.proposals import CandidateCompiler

    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []
    executor = PhaseInterruptingRoundExecutor(phase)
    dependencies = EvolutionDependencies(
        teacher_transport=lambda request: _teacher(teacher_requests, request),
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=executor,
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    if phase == "teacher-frozen":
        original_compile = CandidateCompiler.compile
        interrupted = False

        def interrupt_compile(self, *args, **kwargs):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt("after Teacher response freeze")
            return original_compile(self, *args, **kwargs)

        monkeypatch.setattr(CandidateCompiler, "compile", interrupt_compile)
    if phase == "round-completed":
        original_write = GoalStateStore.write
        interrupted_state = False

        def interrupt_state_write(self, state):
            nonlocal interrupted_state
            if state.rounds_completed == 1 and not interrupted_state:
                interrupted_state = True
                raise KeyboardInterrupt("after sealed round index")
            return original_write(self, state)

        monkeypatch.setattr(GoalStateStore, "write", interrupt_state_write)

    output = tmp_path / f"resume-{phase}"
    argv = [
        "autonomous-evolve",
        "--config",
        str(_config(tmp_path)),
        "--output",
        str(output),
        "--worktree-root",
        str(worktree),
    ]
    with pytest.raises(KeyboardInterrupt):
        main(argv)

    preserved: bytes | None = None
    receipt_path: Path | None = None
    if phase == "qwen-run":
        receipt_path = output / "rounds/round-0000/prescreen/receipt-store/receipts.jsonl"
    elif phase == "native-run":
        receipt_path = output / "rounds/round-0000/receipt-store/receipts.jsonl"
    if receipt_path is not None:
        preserved = receipt_path.read_bytes()

    assert main(argv) == 0
    result = json.loads((output / "EVOLUTION-RESULT.json").read_text())
    assert result["status"] == "goal_reached"
    assert len(teacher_requests) == 2
    if receipt_path is not None:
        assert receipt_path.read_bytes() == preserved
    if phase == "qwen-run":
        prescreen = json.loads(
            (output / "rounds/round-0000/PRESCREEN-RESULT.json").read_text()
        )
        assert prescreen["replayed"] is True
    if phase == "round-completed":
        completed = json.loads(
            (output / "rounds/round-0000/AUTONOMOUS-ROUND-RESULT.json").read_text()
        )
        resumed_failure = teacher_requests[1]["failure_package"]
        assert isinstance(resumed_failure, dict)
        assert (
            resumed_failure["prior_campaign_feedback"]
            == completed["campaign_feedback"]
        )


def test_cli_help_exposes_only_the_product_entry(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["autonomous-evolve", "--help"])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "--config" in help_text
    assert "--output" in help_text
    assert "continuous-feedback-evolution" not in help_text


class BaselineFailFixtureRoundExecutor(ExecutableFixtureRoundExecutor):
    """Baseline fails once on the first selected task (student-unresolvable)."""

    def __init__(self) -> None:
        self._failed = False

    def baseline(self, request: RoundExecutionRequest) -> BaselineProbeResult:
        task_ids = tuple(task.task_id for task in _tasks(request))
        if self._failed:
            return super().baseline(request)
        self._failed = True
        first = task_ids[0]
        ok_ids = task_ids[1:]
        return BaselineProbeResult(
            task_ids=task_ids,
            model_receipt_ids=tuple(
                f"baseline-model-{item}" for item in ok_ids
            ),
            native_receipt_ids=tuple(
                f"baseline-native-{item}" for item in ok_ids
            ),
            native_outcomes=tuple(
                {"task_id": item, "resolved": False} for item in ok_ids
            ),
            failure_signatures=tuple(
                {"task_id": item, "signature": "native_unresolved"}
                for item in ok_ids
            ),
            replayed=False,
            failed_task_ids=(first,),
            failure_reasons={first: "operator-class-mismatch"},
        )


def test_baseline_failure_records_neutral_round_and_excludes_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree = _clean_worktree(tmp_path)
    teacher_requests: list[dict[str, object]] = []

    def teacher(request: dict[str, object]) -> dict[str, object]:
        return _teacher(teacher_requests, request)

    dependencies = EvolutionDependencies(
        teacher_transport=teacher,
        teacher_pricing=PricingCnyPerMillionTokens(input=2.0, output=8.0),
        round_executor=BaselineFailFixtureRoundExecutor(),
    )
    monkeypatch.setattr(
        "evolve.autonomous_evolution.build_default_dependencies",
        lambda _config: dependencies,
    )
    config_path = _config(tmp_path)
    config_payload = json.loads(config_path.read_text())
    config_payload["goal"]["max_rounds"] = 3
    config_payload["goal"]["no_progress_patience"] = 3
    config_payload["goal"]["target_native_gains"] = 3
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    output = tmp_path / "output"

    assert (
        main(
            [
                "autonomous-evolve",
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--worktree-root",
                str(worktree),
            ]
        )
        == 0
    )

    r0 = json.loads(
        (output / "rounds/round-0000/AUTONOMOUS-ROUND-RESULT.json").read_text()
    )
    assert r0["campaign_status"] == "baseline_failed"
    assert r0["round_outcome"] == "neutral"
    assert r0["baseline_failure_reasons"] == {
        r0["baseline_failed_task_ids"][0]: "operator-class-mismatch"
    }
    failed_task = r0["baseline_failed_task_ids"][0]
    # Round-1 selection excludes the student-unresolvable task.
    r1_selection = json.loads(
        (output / "rounds/round-0001/TASK-SELECTION.json").read_text()
    )
    assert failed_task not in r1_selection["selected_task_ids"]
    # State records the exclusion.
    state = json.loads((output / "EVOLUTION-STATE.json").read_text())
    assert failed_task in state["student_unresolvable_task_ids"]
    # The failed task is never selected again after round-0000.
    for round_dir in sorted((output / "rounds").glob("round-*")):
        if round_dir.name == "round-0000":
            continue
        selection = json.loads(
            (round_dir / "TASK-SELECTION.json").read_text()
        )
        assert failed_task not in selection["selected_task_ids"]
