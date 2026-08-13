from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from evolve.contracts import (
    Authorization,
    Claim,
    ClaimClassification,
    ClaimGrade,
    Cohort,
    ContractViolation,
    EvidenceEnvelope,
    ExecutionLimits,
    ExecutionPlan,
    ModelIdentity,
    Receipt,
    TaskRevision,
)

SHA = "a" * 64


def test_execution_plan_is_neutral_and_content_addressable() -> None:
    task = TaskRevision(
        task_id="sphinx-7757",
        revision_id="task-r1",
        project="sphinx",
        cohort=Cohort.FEEDBACK,
        source_sha256=SHA,
        evaluator_id="swebench@sha256:" + SHA,
    )
    plan = ExecutionPlan(
        plan_id="plan-1",
        campaign_id="campaign-1",
        strategy_id="skill-paired-v1",
        task=task,
        candidate_revision_id="baseline",
        arm="baseline",
        model=ModelIdentity(provider="local-mlx", model="Qwen3.5-4B", revision="4bit"),
        context_policy_id="frozen-context-v1",
        tool_policy_id="operator-only-v1",
        observer_policy_ids=("external-trace-v1", "native-v1", "cost-v1"),
        native_evaluator_id="swebench@sha256:" + SHA,
        limits=ExecutionLimits(max_tokens=1024, max_seconds=300, max_cost_cny=0),
        holdout_scope="feedback-only",
    )

    assert plan.content_sha256 == plan.content_sha256
    assert plan.task.cohort is Cohort.FEEDBACK
    with pytest.raises(AttributeError):
        plan.arm = "taught"  # type: ignore[misc]


def test_authorization_denies_holdout_and_expired_or_over_budget_plans() -> None:
    now = datetime.now(UTC)
    authorization = Authorization(
        authorization_id="auth-1",
        campaign_id="campaign-1",
        allowed_cohorts=(Cohort.FEEDBACK,),
        max_cost_cny=10,
        max_model_calls=4,
        expires_at=now + timedelta(hours=1),
        remote_calls_allowed=True,
    )

    authorization.assert_allows(
        cohort=Cohort.FEEDBACK,
        reserved_cost_cny=9.99,
        reserved_model_calls=4,
        remote=True,
        now=now,
    )
    with pytest.raises(ContractViolation, match="cohort"):
        authorization.assert_allows(
            cohort=Cohort.HOLDOUT,
            reserved_cost_cny=0,
            reserved_model_calls=0,
            remote=False,
            now=now,
        )
    with pytest.raises(ContractViolation, match="cost"):
        authorization.assert_allows(
            cohort=Cohort.FEEDBACK,
            reserved_cost_cny=10.01,
            reserved_model_calls=1,
            remote=True,
            now=now,
        )
    with pytest.raises(ContractViolation, match="expired"):
        authorization.assert_allows(
            cohort=Cohort.FEEDBACK,
            reserved_cost_cny=0,
            reserved_model_calls=0,
            remote=False,
            now=authorization.expires_at,
        )


def test_receipt_evidence_and_claim_are_immutable_versioned_facts() -> None:
    receipt = Receipt(
        receipt_id="receipt-1",
        campaign_id="campaign-1",
        plan_id="plan-1",
        sequence=1,
        kind="native_evaluation",
        created_at="2026-08-14T00:00:00Z",
        payload={"resolved": False},
        artifact_sha256=SHA,
    )
    evidence = EvidenceEnvelope(
        evidence_id="evidence-1",
        receipt_ids=(receipt.receipt_id,),
        observer_id="native-v1",
        grade=ClaimGrade.E1,
        payload={"resolved": False},
        artifact_sha256=SHA,
    )
    claim = Claim(
        claim_id="claim-1",
        candidate_id="candidate-1",
        grade=ClaimGrade.E1,
        classification=ClaimClassification.NEUTRAL,
        evidence_ids=(evidence.evidence_id,),
        rationale="baseline and taught are both unresolved",
        supersedes_claim_id=None,
    )

    assert receipt.content_sha256 != evidence.content_sha256
    assert claim.classification is ClaimClassification.NEUTRAL
    with pytest.raises(AttributeError):
        claim.rationale = "changed"  # type: ignore[misc]


def test_contract_rejects_non_literal_sha256_and_burned_cohort_execution() -> None:
    with pytest.raises(ContractViolation, match="SHA-256"):
        Receipt(
            receipt_id="r",
            campaign_id="c",
            plan_id="p",
            sequence=1,
            kind="model",
            created_at="2026-08-14T00:00:00Z",
            payload={},
            artifact_sha256="see manifest",
        )

    with pytest.raises(ContractViolation, match="burned"):
        TaskRevision(
            task_id="r076",
            revision_id="task-r076",
            project="sphinx",
            cohort=Cohort.BURNED,
            source_sha256=SHA,
            evaluator_id="native@sha256:" + SHA,
        )
