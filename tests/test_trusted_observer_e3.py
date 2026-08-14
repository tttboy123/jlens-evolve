from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evolve.alignment import align_native_pair
from evolve.contracts import (
    ClaimGrade,
    EvidenceEnvelope,
    Receipt,
    canonical_json,
)
from evolve.evidence import (
    ClaimEngine,
    EvidenceGradeMachine,
    EvidenceGraph,
    IntegrityError,
    ReceiptStore,
    build_matched_counterfactual_pair,
)
from evolve.observers import (
    JacobianLensObserver,
    TrustedJacobianLensObserver,
    TrustedObserverIdentity,
    TrustedObserverKeyring,
    issue_trusted_observation_attestation,
)
from evolve.observers.observer_hub import ReceiptObserver


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


SECRET = b"test-only-trusted-observer-secret"
IDENTITY = TrustedObserverIdentity(
    key_id="jlens-key-2026-08",
    implementation_id="jlens-runtime-v3",
    implementation_sha256=_sha(b"trusted jlens implementation"),
)


def _keyring() -> TrustedObserverKeyring:
    return TrustedObserverKeyring({IDENTITY: SECRET})


def _append_task(
    graph: EvidenceGraph,
    store: ReceiptStore,
    *,
    task_revision_id: str,
    project: str,
    keyring: TrustedObserverKeyring,
    forged_mode: str | None = None,
) -> str:
    del project
    mechanism_id = "mechanism-v1"
    arms: dict[str, tuple[Receipt, EvidenceEnvelope, EvidenceEnvelope]] = {}
    for arm in ("baseline", "taught"):
        plan_id = f"plan-{task_revision_id}-{arm}"
        model_receipt_id = f"receipt-{plan_id}-model"
        prediction_sha256 = _sha(f"prediction:{plan_id}".encode())
        candidate_consumed = arm == "taught"
        model_payload = {
            "patch_sha256": prediction_sha256,
            "provider": "local-mlx",
            "model": "qwen",
            "revision": "frozen-r1",
            "candidate_consumed": candidate_consumed,
            "candidate_revision_id": "candidate-r1" if candidate_consumed else None,
            "candidate_bundle_sha256": _sha(b"candidate-bundle")
            if candidate_consumed
            else None,
        }
        model = Receipt(
            receipt_id=model_receipt_id,
            campaign_id="campaign-1",
            plan_id=plan_id,
            sequence=1,
            kind="model",
            created_at="2026-08-14T06:00:00Z",
            payload=model_payload,
            artifact_sha256=_sha(canonical_json(model_payload).encode()),
        )
        store.append(model, canonical_json(model_payload).encode())
        model_artifact_sha256 = model.artifact_sha256
        external_payload = {
            "task_revision_id": task_revision_id,
            "arm": arm,
            "mechanism_id": mechanism_id,
            "prediction_sha256": prediction_sha256,
            "model_receipt_id": model_receipt_id,
            "model_artifact_sha256": model_artifact_sha256,
            "candidate_consumed": candidate_consumed,
            "candidate_revision_id": "candidate-r1" if candidate_consumed else None,
            "candidate_bundle_sha256": _sha(b"candidate-bundle")
            if candidate_consumed
            else None,
        }
        external_receipt_id = f"receipt-{plan_id}-external"
        external_receipt = Receipt(
            receipt_id=external_receipt_id,
            campaign_id="campaign-1",
            plan_id=plan_id,
            sequence=2,
            kind="external_trace",
            created_at="2026-08-14T06:00:00Z",
            payload=external_payload,
            artifact_sha256=_sha(canonical_json(external_payload).encode()),
        )
        store.append(external_receipt, canonical_json(external_payload).encode())
        native_payload = {
            "task_revision_id": task_revision_id,
            "task_source_sha256": _sha(f"source:{task_revision_id}".encode()),
            "model_identity": "local-mlx/qwen@frozen-r1",
            "native_evaluator_id": "swebench@v1",
            "execution_config_sha256": _sha(b"execution-config"),
            "arm": arm,
            "prediction_sha256": prediction_sha256,
            "model_receipt_id": model_receipt_id,
            "model_artifact_sha256": model_artifact_sha256,
            "resolved": arm == "taught",
            "evaluator_error": None,
        }
        native_receipt = Receipt(
            receipt_id=f"receipt-{plan_id}-native",
            campaign_id="campaign-1",
            plan_id=plan_id,
            sequence=3,
            kind="native_evaluation",
            created_at="2026-08-14T06:00:00Z",
            payload=native_payload,
            artifact_sha256=_sha(canonical_json(native_payload).encode()),
        )
        store.append(native_receipt, canonical_json(native_payload).encode())
        native = EvidenceEnvelope(
            evidence_id=f"evidence-{plan_id}-native",
            receipt_ids=(native_receipt.receipt_id,),
            observer_id="native-v1",
            grade=ClaimGrade.E1,
            payload={
                "campaign_id": "campaign-1",
                "receipt_kind": "native_evaluation",
                **native_payload,
                "task_revision_id": task_revision_id,
                "plan_id": plan_id,
            },
            artifact_sha256=native_receipt.artifact_sha256,
        )
        graph.append_evidence(native)
        external = EvidenceEnvelope(
            evidence_id=f"evidence-{plan_id}-external",
            receipt_ids=(external_receipt_id,),
            observer_id="external-trace-v1",
            grade=ClaimGrade.E0,
            payload={
                "campaign_id": "campaign-1",
                "receipt_kind": "external_trace",
                **external_payload,
                "task_revision_id": task_revision_id,
                "plan_id": plan_id,
            },
            artifact_sha256=external_receipt.artifact_sha256,
        )
        graph.append_evidence(external)
        arms[arm] = (model, external, native)

        observation = json.dumps(
            {
                "mechanism_id": mechanism_id,
                "prediction_sha256": prediction_sha256,
                "trace_summary": {"sensitive_nodes": ["node-a", "node-b"]},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        receipt_id = f"receipt-{plan_id}-trusted-jlens"
        attestation = issue_trusted_observation_attestation(
            identity=IDENTITY,
            secret_key=SECRET,
            observation_artifact=observation,
            observation_receipt_id=receipt_id,
            plan_id=plan_id,
            task_revision_id=task_revision_id,
            model_receipt_id=model_receipt_id,
            model_artifact_sha256=model_artifact_sha256,
            observed_at="2026-08-14T06:00:00Z",
            nonce=f"nonce-{plan_id}",
        )
        if forged_mode == "signature":
            attestation = {**attestation, "signature_hmac_sha256": "0" * 64}
        elif forged_mode == "renamed_receipt":
            receipt_id += "-renamed"
        elif forged_mode == "model_subject_mismatch":
            attestation = issue_trusted_observation_attestation(
                identity=IDENTITY,
                secret_key=SECRET,
                observation_artifact=observation,
                observation_receipt_id=receipt_id,
                plan_id=plan_id,
                task_revision_id=task_revision_id,
                model_receipt_id=model_receipt_id,
                model_artifact_sha256="f" * 64,
                observed_at="2026-08-14T06:00:00Z",
                nonce=f"nonce-{plan_id}",
            )
        receipt = Receipt(
            receipt_id=receipt_id,
            campaign_id="campaign-1",
            plan_id=plan_id,
            sequence=4,
            kind="trusted_jlens_observation",
            created_at="2026-08-14T06:00:00Z",
            payload={"attestation": attestation},
            artifact_sha256=_sha(observation),
        )
        store.append(receipt, observation)
        trusted = TrustedJacobianLensObserver(
            keyring=keyring,
            artifact_reader=store.read_artifact,
        )
        if forged_mode in {"signature", "renamed_receipt"}:
            with pytest.raises(IntegrityError):
                trusted.observe(receipt)
        else:
            envelope = trusted.observe(receipt)
            assert envelope is not None
            if forged_mode == "model_subject_mismatch":
                # The attestation itself is valid, but it cannot align with the
                # external/native evidence for this model execution.
                pass
            graph.append_evidence(envelope)

    baseline = arms["baseline"]
    taught = arms["taught"]
    pair = build_matched_counterfactual_pair(
        candidate_id="candidate-1",
        candidate_revision_id="candidate-r1",
        candidate_bundle_sha256=_sha(b"candidate-bundle"),
        baseline_model_receipt=baseline[0],
        baseline_external_evidence=baseline[1],
        baseline_native_evidence=baseline[2],
        taught_model_receipt=taught[0],
        taught_external_evidence=taught[1],
        taught_native_evidence=taught[2],
    )
    claim = ClaimEngine(graph).classify_pair(
        "candidate-1",
        align_native_pair(baseline[2], taught[2]),
        counterfactual_pair=pair,
    )
    return claim.claim_id


def _aggregate_three_tasks(
    tmp_path: Path,
    *,
    forged_mode: str | None = None,
    replay_counterfactual: bool = True,
):
    graph = EvidenceGraph(tmp_path / "graph")
    store = ReceiptStore(tmp_path / "receipts")
    keyring = _keyring()
    for task in ("sphinx-a", "sphinx-b", "django-a"):
        _append_task(
            graph,
            store,
            task_revision_id=task,
            project="django" if task.startswith("django") else "sphinx",
            keyring=keyring,
            forged_mode=forged_mode if task == "django-a" else None,
        )
    return EvidenceGradeMachine(
        graph,
        trusted_observer_verifier=keyring,
        receipt_store=store if replay_counterfactual else None,
    ).aggregate(
        "candidate-1",
        task_projects={
            "sphinx-a": "sphinx",
            "sphinx-b": "sphinx",
            "django-a": "django",
        },
        mechanism_id="mechanism-v1",
    )


def test_e3_requires_valid_independently_signed_trusted_observations(
    tmp_path: Path,
) -> None:
    state = _aggregate_three_tasks(tmp_path)

    assert state.grade is ClaimGrade.E3
    assert state.e3_eligible is True
    assert state.prediction_consistent_task_count == 3
    assert len(state.prediction_evidence_ids) == 18


def test_signed_observations_cannot_promote_claims_without_receipt_replay(
    tmp_path: Path,
) -> None:
    state = _aggregate_three_tasks(tmp_path, replay_counterfactual=False)

    assert state.grade is ClaimGrade.E2
    assert state.e3_eligible is False
    assert state.prediction_consistent_task_count == 3
    assert "counterfactual_rebuilt=False" in state.rationale


def test_forged_signature_and_renamed_receipt_fail_closed(tmp_path: Path) -> None:
    for mode in ("signature", "renamed_receipt"):
        state = _aggregate_three_tasks(tmp_path / mode, forged_mode=mode)
        assert state.grade is ClaimGrade.E2
        assert state.e3_eligible is False
        assert state.prediction_consistent_task_count == 2


def test_valid_attestation_for_different_model_subject_stays_e2(
    tmp_path: Path,
) -> None:
    state = _aggregate_three_tasks(tmp_path, forged_mode="model_subject_mismatch")

    assert state.grade is ClaimGrade.E2
    assert state.e3_eligible is False
    assert state.prediction_consistent_task_count == 2


def test_observation_artifact_bytes_are_verified_not_just_metadata(
    tmp_path: Path,
) -> None:
    del tmp_path
    observation = b'{"mechanism_id":"mechanism-v1","prediction_sha256":"' + (
        b"a" * 64
    ) + b'"}'
    receipt_id = "receipt-trusted"
    attestation = issue_trusted_observation_attestation(
        identity=IDENTITY,
        secret_key=SECRET,
        observation_artifact=observation,
        observation_receipt_id=receipt_id,
        plan_id="plan-a",
        task_revision_id="task-a",
        model_receipt_id="model-a",
        model_artifact_sha256="b" * 64,
        observed_at="2026-08-14T06:00:00Z",
        nonce="nonce-a",
    )
    receipt = Receipt(
        receipt_id=receipt_id,
        campaign_id="campaign-a",
        plan_id="plan-a",
        sequence=4,
        kind="trusted_jlens_observation",
        created_at="2026-08-14T06:00:00Z",
        payload={"attestation": attestation},
        artifact_sha256=_sha(observation),
    )
    observer = TrustedJacobianLensObserver(
        keyring=_keyring(), artifact_reader=lambda _: observation + b"tampered"
    )

    with pytest.raises(IntegrityError, match="observation artifact"):
        observer.observe(receipt)


def test_generic_jlens_name_and_self_reported_hash_never_reach_e3(
    tmp_path: Path,
) -> None:
    graph = EvidenceGraph(tmp_path / "graph")
    store = ReceiptStore(tmp_path / "receipts")
    for task in ("sphinx-a", "sphinx-b", "django-a"):
        _append_task(
            graph,
            store,
            task_revision_id=task,
            project="django" if task.startswith("django") else "sphinx",
            keyring=_keyring(),
            forged_mode="signature",
        )
        for arm in ("baseline", "taught"):
            plan_id = f"plan-{task}-{arm}"
            prediction = _sha(f"prediction:{plan_id}".encode())
            receipt = Receipt(
                receipt_id=f"receipt-{plan_id}-self-reported",
                campaign_id="campaign-1",
                plan_id=plan_id,
                sequence=4,
                kind="internal_trace",
                created_at="2026-08-14T06:00:00Z",
                payload={
                    "task_revision_id": task,
                    "plan_id": plan_id,
                    "mechanism_id": "mechanism-v1",
                    "prediction_sha256": prediction,
                    "observation_sha256": prediction,
                    "trust_verified": True,
                },
                artifact_sha256=_sha(b"self-reported"),
            )
            store.append(receipt, b"self-reported")
            envelope = JacobianLensObserver().observe(receipt)
            assert envelope is not None
            graph.append_evidence(envelope)

    state = EvidenceGradeMachine(
        graph,
        trusted_observer_verifier=_keyring(),
        receipt_store=store,
    ).aggregate(
        "candidate-1",
        task_projects={
            "sphinx-a": "sphinx",
            "sphinx-b": "sphinx",
            "django-a": "django",
        },
        mechanism_id="mechanism-v1",
    )

    assert state.grade is ClaimGrade.E2
    assert state.e3_eligible is False
    assert state.prediction_consistent_task_count == 0


def test_trusted_observer_id_and_metadata_cannot_bypass_keyring(
    tmp_path: Path,
) -> None:
    receipt = Receipt(
        receipt_id="receipt-spoof",
        campaign_id="campaign-1",
        plan_id="plan-spoof",
        sequence=4,
        kind="trusted_jlens_observation",
        created_at="2026-08-14T06:00:00Z",
        payload={
            "attestation_verified": True,
            "task_revision_id": "task-spoof",
            "model_receipt_id": "model-spoof",
            "model_artifact_sha256": "a" * 64,
            "prediction_sha256": "b" * 64,
            "mechanism_id": "mechanism-v1",
        },
        artifact_sha256=_sha(b"spoof"),
    )
    spoof = ReceiptObserver(
        observer_id="trusted-jlens-v1",
        receipt_kinds=("trusted_jlens_observation",),
    ).observe(receipt)

    assert spoof is not None
    assert _keyring().verify_evidence(spoof) is False

    graph = EvidenceGraph(tmp_path / "graph")
    graph.append_evidence(spoof)
    assert graph.evidence_by_observer("trusted-jlens-v1") == (spoof,)
