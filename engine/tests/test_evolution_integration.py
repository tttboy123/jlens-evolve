from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from candidate_tournament import ArmEvaluation
from evolution_controller import EvolutionAuthorization, EvolutionPlan
from evolution_fixture import run_fixture
from evolution_runtime import ExecutionArtifact, ExecutionRequest, run_evolution_search
from mutation_proposer import MutationProposer
from pattern_miner import FrozenObservationEvidence
from real_mutation_proposer import RealProposalError

ORIGINAL = "a" * 64
SEED_PARENT = "b" * 64
EPOCH = "native-adapters-v2.1.0-frozen"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class DeterministicAdapters:
    real_codex_calls = False

    def __init__(self):
        self.proposer = MutationProposer()

    def execute(self, request: ExecutionRequest) -> ExecutionArtifact:
        if request.role == "original":
            score = 0.5
            feature = "stable_shadow"
            delta = 0.0
        elif request.role == "parent":
            score = 0.6
            feature = "tests_before_edit"
            delta = 0.1
        else:
            ordinal = request.candidate_ordinal or 0
            score = 0.9 - ordinal * 0.1
            feature = "tests_before_edit" if ordinal < 2 else "broad_rewrite"
            delta = score - 0.6
            if ordinal == 3:
                delta = -0.1
        evidence_id = (
            f"g{request.generation}-{request.stage}-{request.task_uid}-"
            f"{request.arm_sha256[:8]}"
        )
        contract = _sha(request.task_uid)
        arm = ArmEvaluation.from_dict(
            {
                "schema_version": 1,
                "task_uid": request.task_uid,
                "benchmark_family": "fixture-family",
                "role": request.role,
                "agent_program_sha256": request.arm_sha256,
                "matched_contract_sha256": contract,
                "native_evaluator_epoch": EPOCH,
                "native_score": score,
                "safety_passed": True,
                "cost_units": 1.0,
                "evidence_sha256": _sha(evidence_id),
            }
        )
        observation = None
        if request.role == "candidate" or (
            request.generation == 0 and request.role == "parent"
        ):
            parent_hash = (
                request.original_sha256
                if request.generation == 0
                else request.parent_sha256
            )
            observation = FrozenObservationEvidence.from_dict(
                {
                    "schema_version": 1,
                    "evidence_id": evidence_id,
                    "task_uid": request.task_uid,
                    "benchmark_family": "fixture-family",
                    "agent_program_sha256": request.arm_sha256,
                    "parent_agent_program_sha256": parent_hash,
                    "native_evaluator_epoch": EPOCH,
                    "native_score_delta": delta,
                    "safety_passed": True,
                    "observed_features": [feature],
                    "conditions": ["python", request.stage],
                    "expected_surfaces": [
                        "prompt",
                        "skills",
                        "policy",
                        "constrained_harness_code",
                    ],
                    "evidence": {
                        name: {
                            "path": f"raw/{evidence_id}/{name}.json",
                            "sha256": _sha(f"{evidence_id}-{name}"),
                        }
                        for name in (
                            "trajectory",
                            "tool_events",
                            "native_evaluator",
                            "cost",
                            "safety",
                        )
                    },
                    "causal_boundary": "observational_not_causal",
                    "admission_gate_allowed": False,
                }
            )
        return ExecutionArtifact(arm=arm, observation=observation)

    def propose(self, request, generation: int):
        candidate_hash = _sha(f"g{generation}-{request.request_id}")
        path = {
            "prompt": "AGENTS.md",
            "skills": f".agents/skills/g{generation}/SKILL.md",
            "policy": ".codex/evolution-policy.json",
            "router": ".codex/evolution-policy.json",
            "memory_policy": ".codex/evolution-policy.json",
            "constrained_harness_code": ".codex/harness/runner.py",
        }[request.surface]
        response = {
            "schema_version": 1,
            "changeset_id": f"g{generation}-{request.request_id}",
            "status": "inactive",
            "parent_agent_program_sha256": request.parent_agent_program_sha256,
            "candidate_agent_program_sha256": candidate_hash,
            "hypothesis_ids": list(request.hypothesis_ids),
            "surface": request.surface,
            "operations": [{"op": "replace", "path": path, "after": "new"}],
            "rollback_operations": [{"op": "replace", "path": path, "after": "old"}],
            "proposer": {"platform": "fixture", "model": "deterministic"},
            "native_evaluator_epoch": EPOCH,
            "native_evaluator_authority": "external_fixed",
            "auto_apply": False,
            "production_promotion_allowed": False,
        }
        return self.proposer.propose(
            request=request,
            provider={"platform": "fixture", "model": "deterministic"},
            propose=lambda _prompt: json.dumps(response),
            repair=None,
        )

    def rollback(self, changeset):
        return {
            "forward_patch_sha256": _sha(changeset.changeset_id + "-forward"),
            "rollback_patch_sha256": _sha(changeset.changeset_id + "-rollback"),
            "verified": True,
        }


class FailingProposalAdapters(DeterministicAdapters):
    """Deterministic adapters whose proposer fails for chosen surfaces."""

    def __init__(self, fail_surfaces):
        super().__init__()
        self.fail_surfaces = set(fail_surfaces)

    def propose(self, request, generation: int):
        if request.surface in self.fail_surfaces:
            raise RealProposalError(f"proposal rejected for surface {request.surface}")
        return super().propose(request, generation)


def _run(output: Path):
    return run_fixture(output)


def _run_with_adapters(
    output: Path,
    adapters,
    *,
    plan: EvolutionPlan | None = None,
    authorization: EvolutionAuthorization | None = None,
    worker_count: int = 1,
):
    return run_evolution_search(
        output_dir=output,
        plan=plan
        or EvolutionPlan.build(
            tuple(f"fixture-task-{index:03d}" for index in range(100))
        ),
        authorization=authorization
        or EvolutionAuthorization(
            maximum_unique_search_tasks=100,
            maximum_real_codex_calls=2000,
            maximum_temporary_cloud_instances=1,
            maximum_elapsed_hours=24.0,
            maximum_cloud_cost_cny=30.0,
        ),
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=SEED_PARENT,
        native_evaluator_epoch=EPOCH,
        adapters=adapters,
        worker_count=worker_count,
    )


def test_resume_plan_runs_partial_g0_then_continues_four_generations(tmp_path):
    task_uids = tuple(f"resume-task-{index:03d}" for index in range(91))
    plan = EvolutionPlan.build_resume(task_uids)
    authorization = EvolutionAuthorization(
        maximum_unique_search_tasks=100,
        maximum_real_codex_calls=2000,
        maximum_temporary_cloud_instances=1,
        maximum_elapsed_hours=24.0,
        maximum_cloud_cost_cny=30.0,
    )

    result = _run_with_adapters(
        tmp_path / "resume",
        DeterministicAdapters(),
        plan=plan,
        authorization=authorization,
    )

    assert result["status"] == "completed"
    assert result["completed_generations"] == 4
    assert result["unique_search_tasks_retired"] == 91
    assert result["usage"]["agent_task_calls"] == 326
    assert result["usage"]["real_codex_calls"] == 0
    assert result["final_sealed_opened"] is False
    assert result["claims"]["agent_optimized"] is False
    assert (tmp_path / "resume/generation-0/EXECUTION-LEDGER.jsonl").is_file()


def test_three_candidate_tournament_completes_with_fail_closed_proposals(tmp_path):
    result = _run_with_adapters(
        tmp_path / "three",
        FailingProposalAdapters({"constrained_harness_code"}),
    )

    assert result["status"] == "completed"
    assert result["completed_generations"] == 4
    assert result["candidates"]["proposed"] == 12
    for generation in (1, 2, 3):
        assert (tmp_path / f"three/generation-{generation}/TOURNAMENT.json").is_file()
        failures = tmp_path / f"three/generation-{generation}/PROPOSAL-FAILURES.jsonl"
        assert failures.is_file()
        assert "constrained_harness_code" in failures.read_text(encoding="utf-8")
    assert result["final_sealed_opened"] is False


def test_one_candidate_tournament_completes_with_minimal_candidate(tmp_path):
    result = _run_with_adapters(
        tmp_path / "one",
        FailingProposalAdapters({"skills", "policy", "constrained_harness_code"}),
    )

    assert result["status"] == "completed"
    assert result["completed_generations"] == 4
    assert result["candidates"]["proposed"] == 4
    for generation in (1, 2, 3):
        assert (tmp_path / f"one/generation-{generation}/TOURNAMENT.json").is_file()
    assert result["final_sealed_opened"] is False


def test_zero_candidate_finalizes_gracefully(tmp_path):
    result = _run_with_adapters(
        tmp_path / "zero",
        FailingProposalAdapters(
            {"prompt", "skills", "policy", "constrained_harness_code"}
        ),
    )

    assert result["status"] == "stopped_no_mutation_signal"
    assert result["candidates"]["proposed"] == 0
    assert result["completed_generations"] == 2
    assert (tmp_path / "zero/generation-0/PROPOSAL-FAILURES.jsonl").is_file()
    assert (tmp_path / "zero/generation-1/RESULT.zh-CN.md").is_file()
    assert (tmp_path / "zero/STOP.json").is_file()
    assert result["final_sealed_opened"] is False


def test_four_generation_poc_mines_mutates_tournaments_and_persists_negatives(tmp_path):
    result = _run(tmp_path / "run")

    assert result["status"] == "completed"
    assert result["completed_generations"] == 4
    assert result["unique_search_tasks_retired"] == 100
    assert result["usage"]["agent_task_calls"] == 344
    assert result["usage"]["auxiliary_calls"] == 16
    assert result["usage"]["real_codex_calls"] == 0
    assert result["patterns"]["advantage"] > 0
    assert result["patterns"]["failure"] > 0
    assert result["candidates"]["proposed"] == 16
    assert result["candidates"]["selected"] == 3
    assert result["candidates"]["rejected"] == 9
    assert result["candidates"]["inactive"] == 4
    assert len(result["search_parent_history"]) == 3
    assert result["final_sealed_opened"] is False
    assert result["production_active_ref"] is None
    assert result["claims"]["agent_optimized"] is False
    assert result["claims"]["agentic_rsi"] is False

    report = (tmp_path / "run/REPORT.zh-CN.md").read_text(encoding="utf-8")
    for phrase in (
        "每代发现了什么",
        "PatternCard",
        "inactive ChangeSet",
        "淘汰候选",
        "search parent",
        "证据链",
        "final sealed 未打开",
    ):
        assert phrase in report
    assert (tmp_path / "run/generation-0/PATTERN-CARDS.jsonl").is_file()
    assert (tmp_path / "run/generation-1/TOURNAMENT.json").is_file()
    assert (tmp_path / "run/archive/events.jsonl").is_file()


def test_full_poc_has_one_semantic_fingerprint_across_three_runs(tmp_path):
    results = [_run(tmp_path / f"pass-{index}") for index in range(3)]

    assert len({result["experiment_fingerprint"] for result in results}) == 1
    assert all(result["archive_valid"] for result in results)
    assert all(result["controller_valid"] for result in results)


def test_evolution_runtime_resumes_partial_and_completed_runs_idempotently(tmp_path):
    class InterruptOnceAdapters(DeterministicAdapters):
        def __init__(self):
            super().__init__()
            self.executions = 0

        def execute(self, request: ExecutionRequest) -> ExecutionArtifact:
            self.executions += 1
            if self.executions == 8:
                raise RuntimeError("simulated process interruption")
            return super().execute(request)

    output = tmp_path / "resumable"
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        _run_with_adapters(output, InterruptOnceAdapters())

    resumed = _run_with_adapters(output, DeterministicAdapters())
    repeated = _run_with_adapters(output, DeterministicAdapters())

    assert resumed["status"] == "completed"
    assert repeated["experiment_fingerprint"] == resumed["experiment_fingerprint"]
    assert repeated["usage"]["agent_task_calls"] == 344
    assert repeated["usage"]["auxiliary_calls"] == 16


def test_evolution_stops_cleanly_when_observer_finds_no_mutation_signal(tmp_path):
    class NeutralAdapters(DeterministicAdapters):
        def execute(self, request: ExecutionRequest) -> ExecutionArtifact:
            artifact = super().execute(request)
            if artifact.observation is None:
                return artifact
            neutral = artifact.observation.to_dict()
            neutral["native_score_delta"] = 0.0
            return ExecutionArtifact(
                arm=artifact.arm,
                observation=FrozenObservationEvidence.from_dict(neutral),
            )

    result = _run_with_adapters(tmp_path / "neutral", NeutralAdapters())

    assert result["status"] == "stopped_no_mutation_signal"
    assert result["unique_search_tasks_retired"] == 25
    assert result["task_counts"] == {"unopened": 75, "claimed": 0, "retired": 25}
    assert result["candidates"]["proposed"] == 0
    assert result["stop_reason"]["code"] == "insufficient_mutation_signal"
    assert result["final_sealed_opened"] is False
    assert (tmp_path / "neutral/STOP.json").is_file()


def test_v22_generations_carry_convergence_metrics(tmp_path):
    result = _run(tmp_path / "conv-metrics")

    assert result["status"] == "completed"
    for generation in result["generations"][1:]:
        metrics = generation.get("convergence_metrics")
        assert metrics is not None
        assert isinstance(metrics["mean_abs_delta"], float)
        assert metrics["safety_regression"] is False
        assert metrics["per_candidate"]
        for row in metrics["per_candidate"].values():
            assert "vs_original" in row and "vs_parent" in row
    assert result["terminal_state"] == "completed"


def test_v22_convergence_stops_after_k2_without_safety_regression(tmp_path):
    class ConvergingAdapters(DeterministicAdapters):
        def execute(self, request: ExecutionRequest) -> ExecutionArtifact:
            artifact = super().execute(request)
            arm = artifact.arm.to_dict()
            if arm["role"] == "original":
                arm["native_score"] = 0.5
            elif arm["role"] == "parent":
                arm["native_score"] = 0.51
            else:
                arm["native_score"] = 0.515
            observation = artifact.observation
            if observation is not None:
                obs = observation.to_dict()
                obs["native_score_delta"] = 0.005
                observation = FrozenObservationEvidence.from_dict(obs)
            return ExecutionArtifact(
                arm=ArmEvaluation.from_dict(arm), observation=observation
            )

    result = _run_with_adapters(tmp_path / "converged", ConvergingAdapters())

    assert result["status"] == "converged"
    assert result["terminal_state"] == "converged"
    assert result["completed_generations"] == 3  # G0 + G1 + G2
    assert result["stop_reason"]["code"] == "converged"
    assert result["convergence"]["mean_abs_delta"] < 0.05
    assert result["convergence"]["safety_regression"] is False
    assert (tmp_path / "converged/STOP.json").is_file()
    assert (tmp_path / "converged/RESULT.json").is_file()
    assert result["final_sealed_opened"] is False


def test_v22_pattern_feedback_merge_generalizes_a_fixes_b(tmp_path):
    class GeneralizedAdapters(DeterministicAdapters):
        def execute(self, request: ExecutionRequest) -> ExecutionArtifact:
            artifact = super().execute(request)
            if artifact.observation is None or artifact.arm.role != "candidate":
                return artifact
            if (request.candidate_ordinal or 0) == 0:
                obs = artifact.observation.to_dict()
                features = list(obs["observed_features"])
                if "broad_rewrite" not in features:
                    features.append("broad_rewrite")
                obs["observed_features"] = features
                return ExecutionArtifact(
                    arm=artifact.arm,
                    observation=FrozenObservationEvidence.from_dict(obs),
                )
            return artifact

    result = _run_with_adapters(tmp_path / "generalized", GeneralizedAdapters())

    assert result["patterns"]["generalized"] > 0
    found = False
    for index in (1, 2, 3):
        cards_path = tmp_path / f"generalized/generation-{index}/PATTERN-CARDS.jsonl"
        if not cards_path.is_file():
            continue
        if "generalized_fix" in cards_path.read_text(encoding="utf-8"):
            found = True
            break
    assert found
    # generalized cards remain observational and non-admission
    for index in (1, 2, 3):
        cards_path = tmp_path / f"generalized/generation-{index}/PATTERN-CARDS.jsonl"
        if not cards_path.is_file():
            continue
        for line in cards_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            card = json.loads(line)
            if card["pattern_kind"] == "generalized_fix":
                assert card["causal_boundary"] == "observational_not_causal"
                assert card["admission_gate_allowed"] is False


def test_v22_parallel_candidate_verification_is_contract_identical(tmp_path):
    serial = _run(tmp_path / "serial")

    for workers in (2, 4):
        parallel = _run_with_adapters(
            tmp_path / f"parallel-{workers}",
            DeterministicAdapters(),
            worker_count=workers,
        )
        assert parallel["experiment_fingerprint"] == serial["experiment_fingerprint"]
        assert parallel["status"] == serial["status"] == "completed"
        assert parallel["usage"] == serial["usage"]
        assert parallel["candidates"] == serial["candidates"]
        assert parallel["patterns"] == serial["patterns"]
        assert parallel["archive_valid"] and parallel["controller_valid"]
