from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from evolve.alignment import align_native_pair
from evolve.contracts import (
    Claim,
    ClaimClassification,
    ClaimGrade,
    ContractViolation,
    CounterfactualArmEvidence,
    EvidenceEnvelope,
    MatchedCounterfactualPair,
    MechanismPrediction,
    Receipt,
    canonical_json,
)
from evolve.evidence import ClaimEngine, EvidenceGradeMachine, EvidenceGraph
from evolve.governance import (
    GateDecision,
    GovernanceDecisionAuthority,
    GovernanceService,
    PromotionDecisionLog,
)
from evolve.observers import (
    TrustedJacobianLensObserver,
    TrustedObserverIdentity,
    TrustedObserverKeyring,
    derive_structured_jlens_observation,
    issue_trusted_observation_attestation,
)
from evolve.registry import (
    CandidateRecord,
    CapabilityRecord,
    CapabilityRegistry,
    RegistryViolation,
    RejectedRecord,
    RejectedRegistry,
)

SHA = "a" * 64
TRUST_SECRET = b"test-only-trusted-observer-secret"
TRUST_IDENTITY = TrustedObserverIdentity(
    key_id="jlens-key-2026-08",
    implementation_id="jlens-runtime-v3",
    implementation_sha256=hashlib.sha256(b"trusted jlens implementation").hexdigest(),
)
TRUST_KEYRING = TrustedObserverKeyring({TRUST_IDENTITY: TRUST_SECRET})


def _grade_machine(graph: EvidenceGraph) -> EvidenceGradeMachine:
    return EvidenceGradeMachine(graph, trusted_observer_verifier=TRUST_KEYRING)


def _append_claim(
    graph: EvidenceGraph,
    *,
    candidate_id: str,
    task_revision_id: str,
    classification: ClaimClassification,
    mechanism_id: str = "mechanism-v1",
    external_matches_native: bool = True,
) -> Claim:
    claim_evidence = []
    counterfactual_receipts = []
    arms: dict[str, CounterfactualArmEvidence] = {}
    for arm in ("baseline", "taught"):
        prediction = hashlib.sha256(
            f"{task_revision_id}:{arm}:prediction".encode()
        ).hexdigest()
        external_prediction = prediction
        if not external_matches_native:
            external_prediction = hashlib.sha256(
                f"{task_revision_id}:{arm}:different-prediction".encode()
            ).hexdigest()
        plan_id = f"plan-{task_revision_id}-{arm}"
        model_receipt_id = f"receipt-{task_revision_id}-{arm}-model"
        model_artifact_sha256 = hashlib.sha256(
            f"model:{task_revision_id}:{arm}".encode()
        ).hexdigest()
        evidence = EvidenceEnvelope(
            evidence_id=f"evidence-{task_revision_id}-{arm}-native",
            receipt_ids=(f"receipt-{task_revision_id}-{arm}-native",),
            observer_id="native-v1",
            grade=ClaimGrade.E1,
            payload={
                "task_revision_id": task_revision_id,
                "plan_id": plan_id,
                "arm": arm,
                "prediction_sha256": prediction,
                "model_receipt_id": model_receipt_id,
                "model_artifact_sha256": model_artifact_sha256,
            },
            artifact_sha256=SHA,
        )
        graph.append_evidence(evidence)
        external = EvidenceEnvelope(
            evidence_id=f"evidence-{task_revision_id}-{arm}-external-trace-v1",
            receipt_ids=(f"receipt-{task_revision_id}-{arm}-external-trace-v1",),
            observer_id="external-trace-v1",
            grade=ClaimGrade.E0,
            payload={
                "task_revision_id": task_revision_id,
                "plan_id": plan_id,
                "arm": arm,
                "mechanism_id": mechanism_id,
                "prediction_sha256": external_prediction,
                "model_receipt_id": model_receipt_id,
                "model_artifact_sha256": model_artifact_sha256,
            },
            artifact_sha256=SHA,
        )
        graph.append_evidence(external)
        claim_evidence.extend((external, evidence))
        arms[arm] = CounterfactualArmEvidence(
            arm=arm,
            campaign_id="campaign-1",
            plan_id=plan_id,
            model_receipt_id=model_receipt_id,
            model_receipt_sha256=hashlib.sha256(
                f"model-receipt:{task_revision_id}:{arm}".encode()
            ).hexdigest(),
            model_artifact_sha256=model_artifact_sha256,
            external_trace_evidence_id=external.evidence_id,
            external_trace_evidence_sha256=external.content_sha256,
            external_trace_receipt_id=external.receipt_ids[0],
            external_trace_artifact_sha256=external.artifact_sha256,
            native_outcome_evidence_id=evidence.evidence_id,
            native_outcome_evidence_sha256=evidence.content_sha256,
            native_outcome_receipt_id=evidence.receipt_ids[0],
            native_outcome_artifact_sha256=evidence.artifact_sha256,
            prediction_sha256=prediction,
            prompt_bundle_sha256=hashlib.sha256(
                f"prompt:{task_revision_id}:{arm}".encode()
            ).hexdigest(),
            candidate_prompt_sha256=("e" * 64 if arm == "taught" else None),
            candidate_consumed=arm == "taught",
            candidate_revision_id="candidate-r1" if arm == "taught" else None,
            candidate_bundle_sha256="b" * 64 if arm == "taught" else None,
        )
        counterfactual_receipts.extend(
            (
                model_receipt_id,
                f"receipt-{task_revision_id}-{arm}-external-trace-v1",
                evidence.receipt_ids[0],
            )
        )
    pair = MatchedCounterfactualPair(
        candidate_id=candidate_id,
        candidate_revision_id="candidate-r1",
        candidate_bundle_sha256="b" * 64,
        campaign_id="campaign-1",
        task_revision_id=task_revision_id,
        task_source_sha256="c" * 64,
        model_identity="local-mlx/qwen@frozen-r1",
        native_evaluator_id="swebench@v1",
        execution_config_sha256="d" * 64,
        baseline=arms["baseline"],
        taught=arms["taught"],
    )
    claim = Claim(
        claim_id=f"claim-{task_revision_id}",
        candidate_id=candidate_id,
        grade=(
            ClaimGrade.E1
            if classification is ClaimClassification.INFRA_FAILURE
            else ClaimGrade.E2
        ),
        classification=classification,
        evidence_ids=tuple(row.evidence_id for row in claim_evidence),
        rationale="matched native pair",
        supersedes_claim_id=None,
        counterfactual_pair_sha256=(
            None
            if classification is ClaimClassification.INFRA_FAILURE
            else pair.content_sha256
        ),
        counterfactual_receipt_ids=(
            ()
            if classification is ClaimClassification.INFRA_FAILURE
            else tuple(counterfactual_receipts)
        ),
    )
    return graph.append_claim(
        claim,
        counterfactual_pair=(
            None if classification is ClaimClassification.INFRA_FAILURE else pair
        ),
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
        plan_id = f"plan-{task_revision_id}-{arm}"
        model_receipt_id = f"receipt-{task_revision_id}-{arm}-model"
        model_artifact_sha256 = hashlib.sha256(
            f"model:{task_revision_id}:{arm}".encode()
        ).hexdigest()
        external_receipt_id = f"receipt-{task_revision_id}-{arm}-external-trace-v1"
        external = EvidenceEnvelope(
            evidence_id=f"evidence-{task_revision_id}-{arm}-external-trace-v1",
            receipt_ids=(external_receipt_id,),
            observer_id="external-trace-v1",
            grade=ClaimGrade.E0,
            payload={
                "task_revision_id": task_revision_id,
                "plan_id": plan_id,
                "arm": arm,
                "mechanism_id": mechanism_id,
                "prediction_sha256": prediction,
                "model_receipt_id": model_receipt_id,
                "model_artifact_sha256": model_artifact_sha256,
            },
            artifact_sha256=SHA,
        )
        existing = {row.evidence_id: row for row in graph.list_evidence()}.get(
            external.evidence_id
        )
        if existing is None:
            graph.append_evidence(external)
        else:
            assert existing == external
        observed_prediction = prediction if consistent else "f" * 64
        mechanism_prediction = MechanismPrediction.create(
            prediction_id="prediction-role-commitment-v1",
            candidate_revision_id="candidate-r1",
            mechanism_id=mechanism_id,
            observer_config_sha256=hashlib.sha256(b"observer-config").hexdigest(),
            expected_internal_effect={
                "concept": "declared-role",
                "phase": "symbol-selection",
                "min_final_score": 0.7,
                "min_location_count": 2,
                "require_non_decreasing": True,
            },
        )
        mechanism_prediction_payload = mechanism_prediction.as_payload()
        mechanism_prediction_receipt = Receipt(
            receipt_id=f"receipt-{plan_id}-mechanism-prediction",
            campaign_id="campaign-1",
            plan_id=plan_id,
            sequence=1,
            kind="mechanism_prediction",
            created_at="2026-08-14T05:59:00Z",
            payload=mechanism_prediction_payload,
            artifact_sha256=hashlib.sha256(
                canonical_json(mechanism_prediction_payload).encode()
            ).hexdigest(),
        )
        scores = (0.6, 0.4) if arm == "baseline" else (0.4, 0.9)
        raw_trace = json.dumps(
            {
                "locations": [
                    {
                        "layer": 8 + index,
                        "token_position": 120 + index,
                        "phase": "symbol-selection",
                        "concept_scores": {"declared-role": score},
                    }
                    for index, score in enumerate(scores)
                ]
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        observation = derive_structured_jlens_observation(
            raw_trace_artifact=raw_trace,
            mechanism_prediction_receipt=mechanism_prediction_receipt,
            prediction_sha256=observed_prediction,
        )
        trusted_receipt_id = f"receipt-{task_revision_id}-{arm}-trusted-jlens"
        attestation = issue_trusted_observation_attestation(
            identity=TRUST_IDENTITY,
            secret_key=TRUST_SECRET,
            observation_artifact=observation,
            observation_receipt_id=trusted_receipt_id,
            plan_id=plan_id,
            task_revision_id=task_revision_id,
            model_receipt_id=model_receipt_id,
            model_artifact_sha256=model_artifact_sha256,
            observed_at="2026-08-14T06:00:00Z",
            nonce=f"nonce-{plan_id}",
        )
        trusted = TrustedJacobianLensObserver(
            keyring=TRUST_KEYRING,
            artifact_reader=lambda digest, data=observation: (
                data if digest == hashlib.sha256(data).hexdigest() else b"missing"
            ),
        ).observe(
            Receipt(
                receipt_id=trusted_receipt_id,
                campaign_id="campaign-1",
                plan_id=plan_id,
                sequence=4,
                kind="trusted_jlens_observation",
                created_at="2026-08-14T06:00:00Z",
                payload={"attestation": attestation},
                artifact_sha256=hashlib.sha256(observation).hexdigest(),
            )
        )
        assert trusted is not None
        graph.append_evidence(trusted)


def test_evidence_grade_machine_does_not_promote_unreplayed_claims_to_e3(
    tmp_path: Path,
) -> None:
    graph = EvidenceGraph(tmp_path / "graph")
    machine = _grade_machine(graph)

    empty = machine.aggregate("candidate-1", task_projects={})
    assert empty.grade is ClaimGrade.E0
    assert empty.classification == "insufficient_evidence"

    _append_claim(
        graph,
        candidate_id="candidate-1",
        task_revision_id="sphinx-7757",
        classification=ClaimClassification.GAIN,
        mechanism_id="operator-boundary-v1",
    )
    one = machine.aggregate("candidate-1", task_projects={"sphinx-7757": "sphinx"})
    assert (one.grade, one.gain_count, one.task_count) == (ClaimGrade.E2, 1, 1)

    _append_claim(
        graph,
        candidate_id="candidate-1",
        task_revision_id="sphinx-10435",
        classification=ClaimClassification.NEUTRAL,
        mechanism_id="operator-boundary-v1",
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
        mechanism_id="operator-boundary-v1",
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
    assert three.grade is ClaimGrade.E2
    assert three.e3_eligible is False
    assert three.classification == "gain"
    assert (three.task_count, three.project_count) == (3, 2)
    assert three.prediction_consistent_task_count == 0
    assert three.prediction_evidence_ids == ()


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
    state = _grade_machine(graph).aggregate(
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
    state = _grade_machine(graph).aggregate(
        "candidate-1",
        task_projects={
            "sphinx-a": "sphinx",
            "sphinx-b": "sphinx",
            "django-a": "django",
        },
        mechanism_id="mechanism-v1",
    )

    assert state.grade is ClaimGrade.E2
    assert state.prediction_consistent_task_count == 0
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
            external_matches_native=task_revision != "django-a",
        )
        _append_prediction_evidence(
            graph,
            task_revision_id=task_revision,
            mechanism_id="mechanism-v1",
            matches_native=task_revision != "django-a",
        )
    state = _grade_machine(graph).aggregate(
        "candidate-1",
        task_projects={
            "sphinx-a": "sphinx",
            "sphinx-b": "sphinx",
            "django-a": "django",
        },
        mechanism_id="mechanism-v1",
    )

    assert state.grade is ClaimGrade.E2
    assert state.prediction_consistent_task_count == 0
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


def test_claim_engine_native_only_pair_cannot_mint_e2(
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
        ClaimGrade.E1,
        ClaimClassification.GAIN,
    )
    assert valid.counterfactual_pair_sha256 is None
    assert valid.counterfactual_receipt_ids == ()
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
            mechanism_id="operator-boundary-v1",
        )
        _append_prediction_evidence(
            graph,
            task_revision_id=task_revision,
            mechanism_id="operator-boundary-v1",
        )
    state = _grade_machine(graph).aggregate(
        "candidate-1",
        task_projects={
            "sphinx-7757": "sphinx",
            "sphinx-10435": "sphinx",
            "django-13794": "django",
        },
        mechanism_id="operator-boundary-v1",
    )
    # This test exercises Governance/Registry projection.  The independent
    # ReceiptStore-backed positive E3 path is covered by
    # test_trusted_observer_e3; treat its output as the upstream authority.
    state = replace(
        state,
        grade=ClaimGrade.E3,
        e3_eligible=True,
        prediction_consistent_task_count=3,
        prediction_evidence_ids=tuple(
            f"authoritative-upstream-evidence-{index}" for index in range(18)
        ),
    )
    candidate = CandidateRecord(
        candidate_id="candidate-1",
        revision_id="candidate-r1",
        candidate_kind="operator-skill",
        source_claim_ids=state.claim_ids,
        artifact_sha256=hashlib.sha256(b"candidate").hexdigest(),
    )
    authority = GovernanceDecisionAuthority(
        key_id="governance-test-key",
        secret_key=b"g" * 32,
    )
    log = PromotionDecisionLog(
        tmp_path / "promotion-decisions.jsonl", authority=authority
    )
    service = GovernanceService(authority=authority)

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
