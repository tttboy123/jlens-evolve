from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from evolve.contracts import (
    Cohort,
    ExecutionLimits,
    ModelIdentity,
    TaskRevision,
)
from evolve.registry import (
    AgentProgramRecord,
    AgentProgramRegistry,
    CandidateRecord,
    CandidateRegistry,
    CapabilityRecord,
    CapabilityRegistry,
    RegistryBusy,
    RegistryConflict,
    RegistryViolation,
)
from evolve.strategies import (
    AgentProgramSearchStrategy,
    EvolutionStrategy,
    LegacyImportStrategy,
    SkillPairedStrategy,
    StrategyViolation,
)

SHA_A = hashlib.sha256(b"source-a").hexdigest()


def task(*, cohort: Cohort = Cohort.FEEDBACK) -> TaskRevision:
    return TaskRevision(
        task_id="django-13794",
        revision_id="task-r1",
        project="django",
        cohort=cohort,
        source_sha256=SHA_A,
        evaluator_id="swebench-native-v1",
        source_uri="feedback/django-13794",
    )


def model() -> ModelIdentity:
    return ModelIdentity(provider="mlx", model="qwen3.5-4b", revision="frozen-r1")


def limits() -> ExecutionLimits:
    return ExecutionLimits(max_tokens=2048, max_seconds=300, max_cost_cny=0)


def test_skill_paired_strategy_returns_strictly_matched_execution_plans() -> None:
    plans = SkillPairedStrategy().build_plans(
        campaign_id="campaign-1",
        task=task(),
        baseline_revision_id="baseline-r1",
        taught_revision_id="skill-r2",
        model=model(),
        context_policy_id="context-r1",
        tool_policy_id="tools-r1",
        observer_policy_ids=("native", "cost"),
        limits=limits(),
    )

    assert [plan.arm for plan in plans] == ["baseline", "taught"]
    baseline, taught = plans
    assert baseline.plan_id != taught.plan_id
    assert baseline.candidate_revision_id == "baseline-r1"
    assert taught.candidate_revision_id == "skill-r2"
    assert baseline.task == taught.task
    assert baseline.model == taught.model
    assert baseline.context_policy_id == taught.context_policy_id
    assert baseline.tool_policy_id == taught.tool_policy_id
    assert baseline.observer_policy_ids == taught.observer_policy_ids
    assert baseline.native_evaluator_id == taught.native_evaluator_id
    assert baseline.limits == taught.limits
    assert baseline.holdout_scope == taught.holdout_scope == "feedback-only"
    assert SkillPairedStrategy.validate_matched_pair(baseline, taught) is None


def test_skill_paired_strategy_denies_holdout_before_creating_a_plan() -> None:
    with pytest.raises(StrategyViolation, match="feedback"):
        SkillPairedStrategy().build_plans(
            campaign_id="campaign-1",
            task=task(cohort=Cohort.HOLDOUT),
            baseline_revision_id="baseline-r1",
            taught_revision_id="skill-r2",
            model=model(),
            context_policy_id="context-r1",
            tool_policy_id="tools-r1",
            observer_policy_ids=("native",),
            limits=limits(),
        )


def test_matched_pair_rejects_generation_config_drift() -> None:
    baseline, taught = SkillPairedStrategy().build_plans(
        campaign_id="campaign-1",
        task=task(),
        baseline_revision_id="baseline-r1",
        taught_revision_id="skill-r2",
        model=model(),
        context_policy_id="context-r1",
        tool_policy_id="tools-r1",
        observer_policy_ids=("native",),
        limits=limits(),
        generation_config={"temperature": 0, "seed": 7},
    )
    mismatched = replace(taught, metadata={"generation_config": {"seed": 8}})

    with pytest.raises(StrategyViolation, match="generation_config"):
        SkillPairedStrategy.validate_matched_pair(baseline, mismatched)


def test_skill_paired_strategy_preserves_frozen_task_execution_metadata() -> None:
    metadata = {
        "base_revision": "1" * 40,
        "benchmark_id": "swe-bench-verified",
        "instance_id": "django__django-13794",
    }

    baseline, taught = SkillPairedStrategy().build_plans(
        campaign_id="campaign-1",
        task=task(),
        baseline_revision_id="baseline-r1",
        taught_revision_id="skill-r2",
        model=model(),
        context_policy_id="context-r1",
        tool_policy_id="tools-r1",
        observer_policy_ids=("native",),
        limits=limits(),
        generation_config={"temperature": 0},
        plan_metadata=metadata,
    )

    assert baseline.metadata == taught.metadata == {
        **metadata,
        "generation_config": {"temperature": 0},
    }


@pytest.mark.parametrize(
    ("changed_field", "expected_name"),
    [
        ("task", "task"),
        ("model", "model"),
        ("context_policy_id", "context_policy_id"),
        ("tool_policy_id", "tool_policy_id"),
        ("native_evaluator_id", "native_evaluator_id"),
    ],
)
def test_matched_pair_rejects_task_source_evaluator_model_or_config_drift(
    changed_field: str, expected_name: str
) -> None:
    baseline, taught = SkillPairedStrategy().build_plans(
        campaign_id="campaign-1",
        task=task(),
        baseline_revision_id="baseline-r1",
        taught_revision_id="skill-r2",
        model=model(),
        context_policy_id="context-r1",
        tool_policy_id="tools-r1",
        observer_policy_ids=("native",),
        limits=limits(),
    )
    changed_values = {
        "task": replace(
            taught.task, source_sha256=hashlib.sha256(b"other-source").hexdigest()
        ),
        "model": replace(taught.model, revision="other-frozen-revision"),
        "context_policy_id": "other-context",
        "tool_policy_id": "other-tools",
        "native_evaluator_id": "other-evaluator",
    }

    with pytest.raises(StrategyViolation, match=expected_name):
        SkillPairedStrategy.validate_matched_pair(
            baseline, replace(taught, **{changed_field: changed_values[changed_field]})
        )


def test_legacy_import_is_an_executable_feedback_compatibility_plan() -> None:
    strategy = LegacyImportStrategy()
    assert isinstance(strategy, EvolutionStrategy)

    plans = strategy.build_plans(
        campaign_id="campaign-legacy",
        task=task(),
        imported_revision_id="legacy-r106",
        legacy_artifact_sha256=hashlib.sha256(b"r106").hexdigest(),
        provenance_uri="runs/round-6/catalog-entry.json",
        model=model(),
        context_policy_id="legacy-context",
        tool_policy_id="legacy-tools",
        observer_policy_ids=("native",),
        limits=limits(),
    )

    assert len(plans) == 1
    assert plans[0].arm == "legacy-replay"
    assert plans[0].metadata == {
        "legacy_artifact_sha256": hashlib.sha256(b"r106").hexdigest(),
        "provenance_uri": "runs/round-6/catalog-entry.json",
        "compatibility_mode": "replay",
    }


def test_agent_program_search_emits_candidate_dag_tournament_plans() -> None:
    plans = AgentProgramSearchStrategy().build_plans(
        campaign_id="campaign-program",
        task=task(),
        parent_revision_id="program-parent-r3",
        candidate_revision_ids=("program-a-r4", "program-b-r4"),
        tournament_id="tournament-9",
        model=model(),
        context_policy_id="program-context",
        tool_policy_id="program-tools",
        observer_policy_ids=("native", "cost"),
        limits=limits(),
        generation_config={"temperature": 0},
    )

    assert [plan.candidate_revision_id for plan in plans] == [
        "program-parent-r3",
        "program-a-r4",
        "program-b-r4",
    ]
    assert [plan.arm for plan in plans] == ["search-parent", "candidate", "candidate"]
    assert plans[0].metadata["parent_revision_id"] is None
    assert plans[1].metadata["parent_revision_id"] == "program-parent-r3"
    assert plans[2].metadata["parent_revision_id"] == "program-parent-r3"
    assert {plan.metadata["tournament_id"] for plan in plans} == {"tournament-9"}
    assert [plan.metadata["dag_position"] for plan in plans] == [0, 1, 2]


@pytest.mark.parametrize(
    "strategy,kwargs",
    [
        (
            LegacyImportStrategy(),
            {
                "imported_revision_id": "legacy-r1",
                "legacy_artifact_sha256": hashlib.sha256(b"legacy").hexdigest(),
                "provenance_uri": "legacy.json",
            },
        ),
        (
            AgentProgramSearchStrategy(),
            {
                "parent_revision_id": "parent-r1",
                "candidate_revision_ids": ("candidate-r2",),
                "tournament_id": "t1",
            },
        ),
    ],
)
def test_all_strategies_deny_holdout(strategy: EvolutionStrategy, kwargs: dict) -> None:
    with pytest.raises(StrategyViolation, match="feedback"):
        strategy.build_plans(
            campaign_id="campaign-1",
            task=task(cohort=Cohort.HOLDOUT),
            model=model(),
            context_policy_id="context-r1",
            tool_policy_id="tools-r1",
            observer_policy_ids=("native",),
            limits=limits(),
            **kwargs,
        )


def test_candidate_registry_is_append_only_idempotent_and_inactive(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "candidates.jsonl")
    candidate = CandidateRecord(
        candidate_id="candidate-1",
        revision_id="skill-r1",
        candidate_kind="skill",
        source_claim_ids=("claim-e2-1",),
        artifact_sha256=hashlib.sha256(b"skill").hexdigest(),
    )

    assert candidate.active is False
    assert registry.append(candidate) is True
    assert registry.append(candidate) is False
    assert registry.get("candidate-1", "skill-r1") == candidate
    assert registry.all() == (candidate,)

    conflicting = replace(
        candidate, artifact_sha256=hashlib.sha256(b"other").hexdigest()
    )
    with pytest.raises(RegistryConflict, match="conflicting"):
        registry.append(conflicting)


def test_capability_and_agent_program_registries_are_revision_append_only(
    tmp_path,
) -> None:
    capability_registry = CapabilityRegistry(tmp_path / "capabilities.jsonl")
    program_registry = AgentProgramRegistry(tmp_path / "programs.jsonl")
    capability = CapabilityRecord(
        capability_id="cap-symbol-rewrite",
        revision_id="cap-r1",
        capability_kind="operator",
        evidence_claim_ids=("claim-e3-1",),
        artifact_sha256=hashlib.sha256(b"operator").hexdigest(),
    )
    program = AgentProgramRecord(
        program_id="program-default",
        revision_id="program-r1",
        parent_revision_id=None,
        capability_revision_ids=("cap-r1",),
        artifact_sha256=hashlib.sha256(b"program").hexdigest(),
    )

    assert capability.active is False
    assert program.active is False
    assert capability_registry.append(capability) is True
    assert program_registry.append(program) is True
    assert capability_registry.append(capability) is False
    assert program_registry.append(program) is False
    assert capability_registry.get("cap-symbol-rewrite", "cap-r1") == capability
    assert program_registry.get("program-default", "program-r1") == program


def test_registry_detects_hash_tampering_and_honors_single_writer_lease(
    tmp_path,
) -> None:
    path = tmp_path / "candidates.jsonl"
    registry = CandidateRegistry(path)
    candidate = CandidateRecord(
        candidate_id="candidate-1",
        revision_id="skill-r1",
        candidate_kind="skill",
        source_claim_ids=("claim-1",),
        artifact_sha256=hashlib.sha256(b"skill").hexdigest(),
    )
    registry.lease_path.parent.mkdir(parents=True, exist_ok=True)
    registry.lease_path.write_text("held")
    with pytest.raises(RegistryBusy, match="lease"):
        registry.append(candidate)
    registry.lease_path.unlink()
    registry.append(candidate)

    payload = json.loads(path.read_text())
    payload["artifact_sha256"] = hashlib.sha256(b"tampered").hexdigest()
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(RegistryViolation, match="hash mismatch"):
        registry.all()


def test_registry_refuses_pre_activated_assets() -> None:
    with pytest.raises(RegistryViolation, match="inactive"):
        CandidateRecord(
            candidate_id="candidate-1",
            revision_id="skill-r1",
            candidate_kind="skill",
            source_claim_ids=("claim-1",),
            artifact_sha256=hashlib.sha256(b"skill").hexdigest(),
            active=True,
        )


def test_strategy_interpretation_is_observation_only() -> None:
    interpretation = SkillPairedStrategy().interpret(())

    assert interpretation.strategy_id == "skill-paired-v3"
    assert interpretation.receipt_ids == ()
    assert interpretation.observations == {"arm_receipt_count": 0}
    assert not hasattr(interpretation, "promotion")
    assert not hasattr(interpretation, "classification")
