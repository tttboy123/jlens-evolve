from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import pytest

from evolve.autonomous import (
    BaselineProbeResult,
    EvolutionDependencies,
    PrescreenResult,
)
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
from evolve.runtime import ExecutionRuntime
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
        return {
            "resolved": self.round_index == 1 and plan.arm == "taught",
            "exit_code": 0,
            "prediction_sha256": model_output["patch_sha256"],
        }


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
            model_transport=FixtureModel(request),
            workspace_manager=FixtureWorkspace(),
            native_evaluator=FixtureNative(request.round_index),
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
            structural_valid=True,
            patch_applicable=True,
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
            model_transport=FixtureModel(request),
            workspace_manager=FixtureWorkspace(),
            native_evaluator=FixtureNative(request.round_index),
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


class TamperingFixtureRoundExecutor(ExecutableFixtureRoundExecutor):
    def paired(self, request: RoundExecutionRequest) -> Mapping[str, Any]:
        result = dict(super().paired(request))
        claims = [dict(row) for row in result["claims"]]
        claims[0]["task_revision_id"] = "forged-task-revision"
        result["claims"] = claims
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
    assert (output / "EVIDENCE-MANIFEST.json").is_file()
    assert json.loads((output / "EVOLUTION-RESULT.json").read_text())["product_status"] == (
        "offline_e2e_verified"
    )
    first_round = output / "rounds/round-0000"
    assert (first_round / "TEACHER-REQUEST.json").is_file()
    assert (first_round / "TEACHER-RESPONSE.json").is_file()
    assert (first_round / "CAMPAIGN-RESULT.json").is_file()
    selection = json.loads((first_round / "TASK-SELECTION.json").read_text())
    assert len(selection["tasks"]) == 3
    assert all("task_fingerprint_sha256" in task for task in selection["tasks"])

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
    assert not (output / "best/BEST-HARNESS.json").exists()
    assert (output / "rounds/round-0000/INTEGRITY-BLOCK.json").is_file()


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
