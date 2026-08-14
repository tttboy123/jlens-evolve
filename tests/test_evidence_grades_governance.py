from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from evolve.alignment import align_native_pair
from evolve.contracts import (
    Claim,
    ClaimClassification,
    ClaimGrade,
    ContractViolation,
    EvidenceEnvelope,
)
from evolve.evidence import ClaimEngine, EvidenceGradeMachine, EvidenceGraph
from evolve.governance import GateDecision, GovernanceService, PromotionDecisionLog
from evolve.registry import (
    CandidateRecord,
    CapabilityRecord,
    CapabilityRegistry,
    RegistryViolation,
    RejectedRecord,
    RejectedRegistry,
)

SHA = "a" * 64


def _append_claim(
    graph: EvidenceGraph,
    *,
    candidate_id: str,
    task_revision_id: str,
    classification: ClaimClassification,
) -> Claim:
    native_evidence = []
    for arm in ("baseline", "taught"):
        prediction = hashlib.sha256(
            f"{task_revision_id}:{arm}:prediction".encode()
        ).hexdigest()
        evidence = EvidenceEnvelope(
            evidence_id=f"evidence-{task_revision_id}-{arm}-native",
            receipt_ids=(f"receipt-{task_revision_id}-{arm}-native",),
            observer_id="native-v1",
            grade=ClaimGrade.E1,
            payload={
                "task_revision_id": task_revision_id,
                "plan_id": f"plan-{task_revision_id}-{arm}",
                "arm": arm,
                "prediction_sha256": prediction,
            },
            artifact_sha256=SHA,
        )
        graph.append_evidence(evidence)
        native_evidence.append(evidence)
    return graph.append_claim(
        Claim(
            claim_id=f"claim-{task_revision_id}",
            candidate_id=candidate_id,
            grade=(
                ClaimGrade.E1
                if classification is ClaimClassification.INFRA_FAILURE
                else ClaimGrade.E2
            ),
            classification=classification,
            evidence_ids=tuple(row.evidence_id for row in native_evidence),
            rationale="matched native pair",
            supersedes_claim_id=None,
        )
    )


def _append_prediction_evidence(
    graph: EvidenceGraph,
    *,
    task_revision_id: str,
    mechanism_id: str,
    consistent: bool = True,
    matches_native: bool = True,
) -> None:
    for arm in ("baseline", "taught"):
        prediction = hashlib.sha256(
            f"{task_revision_id}:{arm}:prediction".encode()
        ).hexdigest()
        if not matches_native:
            prediction = hashlib.sha256(
                f"{task_revision_id}:{arm}:different-prediction".encode()
            ).hexdigest()
        for observer_id in ("external-trace-v1", "jlens-v1"):
            observation = prediction if consistent else "f" * 64
            graph.append_evidence(
                EvidenceEnvelope(
                    evidence_id=(f"evidence-{task_revision_id}-{arm}-{observer_id}"),
                    receipt_ids=(f"receipt-{task_revision_id}-{arm}-{observer_id}",),
                    observer_id=observer_id,
                    grade=ClaimGrade.E0,
                    payload={
                        "task_revision_id": task_revision_id,
                        "plan_id": f"plan-{task_revision_id}-{arm}",
                        "arm": arm,
                        "mechanism_id": mechanism_id,
                        "prediction_sha256": prediction,
                        "observation_sha256": observation,
                    },
                    artifact_sha256=SHA,
                )
            )


def test_evidence_grade_machine_rebuilds_e0_through_cross_project_e3(
    tmp_path: Path,
) -> None:
    graph = EvidenceGraph(tmp_path / "graph")
    machine = EvidenceGradeMachine(graph)

    empty = machine.aggregate("candidate-1", task_projects={})
    assert empty.grade is ClaimGrade.E0
    assert empty.classification == "insufficient_evidence"

    _append_claim(
        graph,
        candidate_id="candidate-1",
        task_revision_id="sphinx-7757",
        classification=ClaimClassification.GAIN,
    )
    one = machine.aggregate("candidate-1", task_projects={"sphinx-7757": "sphinx"})
    assert (one.grade, one.gain_count, one.task_count) == (ClaimGrade.E2, 1, 1)

    _append_claim(
        graph,
        candidate_id="candidate-1",
        task_revision_id="sphinx-10435",
        classification=ClaimClassification.NEUTRAL,
    )
    two = machine.aggregate(
        "candidate-1",
        task_projects={"sphinx-7757": "sphinx", "sphinx-10435": "sphinx"},
    )
    assert two.grade is ClaimGrade.E2
    assert (two.gain_count, two.neutral_count, two.project_count) == (1, 1, 1)

    _append_claim(
        graph,
        candidate_id="candidate-1",
        task_revision_id="django-13794",
        classification=ClaimClassification.GAIN,
    )
    without_predictions = machine.aggregate(
        "candidate-1",
        task_projects={
            "sphinx-7757": "sphinx",
            "sphinx-10435": "sphinx",
            "django-13794": "django",
        },
        mechanism_id="operator-boundary-v1",
    )
    assert without_predictions.grade is ClaimGrade.E2
    assert without_predictions.e3_eligible is False

    for task_revision in ("sphinx-7757", "sphinx-10435", "django-13794"):
        _append_prediction_evidence(
            graph,
            task_revision_id=task_revision,
            mechanism_id="operator-boundary-v1",
        )
    three = machine.aggregate(
        "candidate-1",
        task_projects={
            "sphinx-7757": "sphinx",
            "sphinx-10435": "sphinx",
            "django-13794": "django",
        },
        mechanism_id="operator-boundary-v1",
    )
    assert three.grade is ClaimGrade.E3
    assert three.classification == "gain"
    assert (three.task_count, three.project_count) == (3, 2)
    assert three.prediction_consistent_task_count == 3
    assert len(three.prediction_evidence_ids) == 18


@pytest.mark.parametrize(
    "classifications,infra_expected",
    [
        ((ClaimClassification.NEUTRAL,) * 3, False),
        (
            (
                ClaimClassification.GAIN,
                ClaimClassification.GAIN,
                ClaimClassification.REGRESSION,
            ),
            False,
        ),
        (
            (
                ClaimClassification.GAIN,
                ClaimClassification.GAIN,
                ClaimClassification.INFRA_FAILURE,
            ),
            True,
        ),
    ],
)
def test_e3_rejects_neutral_regression_and_infra_counterexamples(
    tmp_path: Path,
    classifications: tuple[ClaimClassification, ...],
    infra_expected: bool,
) -> None:
    graph = EvidenceGraph(tmp_path / "graph")
    tasks = ("sphinx-a", "sphinx-b", "django-a")
    for task_revision, classification in zip(tasks, classifications, strict=True):
        _append_claim(
            graph,
            candidate_id="candidate-1",
            task_revision_id=task_revision,
            classification=classification,
        )
        _append_prediction_evidence(
            graph,
            task_revision_id=task_revision,
            mechanism_id="mechanism-v1",
        )
    state = EvidenceGradeMachine(graph).aggregate(
        "candidate-1",
        task_projects={
            "sphinx-a": "sphinx",
            "sphinx-b": "sphinx",
            "django-a": "django",
        },
        mechanism_id="mechanism-v1",
    )

    assert state.grade is (ClaimGrade.E1 if infra_expected else ClaimGrade.E2)
    assert state.e3_eligible is False


def test_e3_rejects_external_or_internal_prediction_mismatch(tmp_path: Path) -> None:
    graph = EvidenceGraph(tmp_path / "graph")
    tasks = ("sphinx-a", "sphinx-b", "django-a")
    for task_revision in tasks:
        _append_claim(
            graph,
            candidate_id="candidate-1",
            task_revision_id=task_revision,
            classification=ClaimClassification.GAIN,
        )
        _append_prediction_evidence(
            graph,
            task_revision_id=task_revision,
            mechanism_id="mechanism-v1",
            consistent=task_revision != "django-a",
        )
    state = EvidenceGradeMachine(graph).aggregate(
        "candidate-1",
        task_projects={
            "sphinx-a": "sphinx",
            "sphinx-b": "sphinx",
            "django-a": "django",
        },
        mechanism_id="mechanism-v1",
    )

    assert state.grade is ClaimGrade.E2
    assert state.prediction_consistent_task_count == 2
    assert state.e3_eligible is False


def test_e3_rejects_native_prediction_mismatch(tmp_path: Path) -> None:
    graph = EvidenceGraph(tmp_path / "graph")
    tasks = ("sphinx-a", "sphinx-b", "django-a")
    for task_revision in tasks:
        _append_claim(
            graph,
            candidate_id="candidate-1",
            task_revision_id=task_revision,
            classification=ClaimClassification.GAIN,
        )
        _append_prediction_evidence(
            graph,
            task_revision_id=task_revision,
            mechanism_id="mechanism-v1",
            matches_native=task_revision != "django-a",
        )
    state = EvidenceGradeMachine(graph).aggregate(
        "candidate-1",
        task_projects={
            "sphinx-a": "sphinx",
            "sphinx-b": "sphinx",
            "django-a": "django",
        },
        mechanism_id="mechanism-v1",
    )

    assert state.grade is ClaimGrade.E2
    assert state.prediction_consistent_task_count == 2
    assert state.e3_eligible is False


def _native_evidence(*, arm: str, resolved: bool, error: str | None = None):
    return EvidenceEnvelope(
        evidence_id=f"native-{arm}-{resolved}-{error}",
        receipt_ids=(f"receipt-{arm}",),
        observer_id="native-v1",
        grade=ClaimGrade.E1,
        payload={
            "arm": arm,
            "task_revision_id": "task-r1",
            "task_source_sha256": SHA,
            "model_identity": "local/Qwen@frozen",
            "native_evaluator_id": "official-v1",
            "execution_config_sha256": "b" * 64,
            "resolved": resolved,
            "evaluator_error": error,
        },
        artifact_sha256=SHA,
    )


def test_claim_engine_assigns_e2_only_to_valid_strict_native_pairs(
    tmp_path: Path,
) -> None:
    graph = EvidenceGraph(tmp_path / "graph")
    engine = ClaimEngine(graph)
    valid = engine.classify_pair(
        "candidate-valid",
        align_native_pair(
            _native_evidence(arm="baseline", resolved=False),
            _native_evidence(arm="taught", resolved=True),
        ),
    )
    infra = engine.classify_pair(
        "candidate-infra",
        align_native_pair(
            _native_evidence(arm="baseline", resolved=False, error="harness failed"),
            _native_evidence(arm="taught", resolved=False),
        ),
    )

    assert (valid.grade, valid.classification) == (
        ClaimGrade.E2,
        ClaimClassification.GAIN,
    )
    assert (infra.grade, infra.classification) == (
        ClaimGrade.E1,
        ClaimClassification.INFRA_FAILURE,
    )


def test_governance_decision_is_immutable_replay_safe_and_projects_asset_semantics(
    tmp_path: Path,
) -> None:
    graph = EvidenceGraph(tmp_path / "graph")
    for task_revision in ("sphinx-7757", "sphinx-10435", "django-13794"):
        _append_claim(
            graph,
            candidate_id="candidate-1",
            task_revision_id=task_revision,
            classification=ClaimClassification.GAIN,
        )
        _append_prediction_evidence(
            graph,
            task_revision_id=task_revision,
            mechanism_id="operator-boundary-v1",
        )
    state = EvidenceGradeMachine(graph).aggregate(
        "candidate-1",
        task_projects={
            "sphinx-7757": "sphinx",
            "sphinx-10435": "sphinx",
            "django-13794": "django",
        },
        mechanism_id="operator-boundary-v1",
    )
    candidate = CandidateRecord(
        candidate_id="candidate-1",
        revision_id="candidate-r1",
        candidate_kind="operator-skill",
        source_claim_ids=state.claim_ids,
        artifact_sha256=hashlib.sha256(b"candidate").hexdigest(),
    )
    log = PromotionDecisionLog(tmp_path / "promotion-decisions.jsonl")
    service = GovernanceService()

    decision = service.decide(
        candidate=candidate,
        evidence=state,
        claims=graph.latest_claims(),
        human_approval=True,
        decided_at="2026-08-14T04:00:00Z",
        log=log,
    )
    replay = service.decide(
        candidate=candidate,
        evidence=state,
        claims=graph.latest_claims(),
        human_approval=True,
        decided_at="2026-08-14T04:00:00Z",
        log=log,
    )

    assert decision.gate_decision is GateDecision.APPROVED
    assert decision.prediction_evidence_ids == state.prediction_evidence_ids
    assert replay == decision
    assert log.all() == (decision,)
    capability = service.to_capability(
        candidate=candidate,
        decision=decision,
        capability_id="capability-operator-1",
    )
    assert isinstance(capability, CapabilityRecord)
    assert capability.active is False
    assert capability.promotion_decision_id == decision.decision_id
    assert capability.source_candidate_id == candidate.candidate_id
    capability_registry = CapabilityRegistry(
        tmp_path / "capabilities.jsonl", decision_log=log
    )
    assert capability_registry.append(capability) is True
    assert capability_registry.append(capability) is False
    with pytest.raises(RegistryViolation, match="promotion decision identity"):
        capability_registry.append(
            CapabilityRecord(
                capability_id="capability-bypass",
                revision_id="candidate-r1",
                capability_kind="operator-skill",
                evidence_claim_ids=state.claim_ids,
                artifact_sha256=candidate.artifact_sha256,
            )
        )
    with pytest.raises(RegistryViolation, match="decision log"):
        CapabilityRegistry(tmp_path / "bypass.jsonl").append(capability)
    with pytest.raises(ContractViolation, match="E3 evidence"):
        service.decide(
            candidate=candidate,
            evidence=replace(state, prediction_evidence_ids=()),
            claims=graph.latest_claims(),
            human_approval=True,
            decided_at="2026-08-14T04:00:01Z",
            log=log,
        )

    regression_graph = EvidenceGraph(tmp_path / "regression-graph")
    _append_claim(
        regression_graph,
        candidate_id="candidate-bad",
        task_revision_id="sphinx-regression",
        classification=ClaimClassification.REGRESSION,
    )
    bad_state = EvidenceGradeMachine(regression_graph).aggregate(
        "candidate-bad", task_projects={"sphinx-regression": "sphinx"}
    )
    bad_candidate = CandidateRecord(
        candidate_id="candidate-bad",
        revision_id="candidate-bad-r1",
        candidate_kind="operator-skill",
        source_claim_ids=bad_state.claim_ids,
        artifact_sha256=hashlib.sha256(b"bad-candidate").hexdigest(),
    )
    rejected_decision = service.decide(
        candidate=bad_candidate,
        evidence=bad_state,
        claims=regression_graph.latest_claims(),
        human_approval=True,
        decided_at="2026-08-14T04:01:00Z",
        log=log,
    )
    rejected = service.to_rejected(candidate=bad_candidate, decision=rejected_decision)
    assert rejected_decision.gate_decision is GateDecision.REJECTED
    assert isinstance(rejected, RejectedRecord)
    assert rejected.active is False
    assert rejected.promotion_decision_id == rejected_decision.decision_id
    rejected_registry = RejectedRegistry(tmp_path / "rejected.jsonl", decision_log=log)
    assert rejected_registry.append(rejected) is True
    assert rejected_registry.append(rejected) is False
