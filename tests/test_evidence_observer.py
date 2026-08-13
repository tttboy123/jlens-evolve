from __future__ import annotations

import hashlib

import pytest

from evolve.alignment import AlignmentError, align_native_pair
from evolve.contracts import ClaimClassification, ClaimGrade, Receipt
from evolve.evidence import (
    ClaimEngine,
    ConcurrentWriterError,
    EvidenceGraph,
    IntegrityError,
    ReceiptConflict,
    ReceiptStore,
)
from evolve.observers import (
    CostObserver,
    ExternalTraceObserver,
    NativeOutcomeObserver,
    ObserverHub,
    SafetyObserver,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _receipt(
    *,
    receipt_id: str = "receipt-1",
    sequence: int = 1,
    kind: str = "native_evaluation",
    artifact: bytes = b'{"status":"failed"}\n',
    payload: dict[str, object] | None = None,
) -> Receipt:
    return Receipt(
        receipt_id=receipt_id,
        campaign_id="campaign-1",
        plan_id="plan-1",
        sequence=sequence,
        kind=kind,
        created_at="2026-08-14T00:00:00Z",
        payload=payload or {"status": "failed"},
        artifact_sha256=_sha(artifact),
    )


def test_receipt_store_is_content_addressed_idempotent_and_fail_closed(
    tmp_path,
) -> None:
    store = ReceiptStore(tmp_path / "receipts")
    artifact = b'{"status":"failed"}\n'
    receipt = _receipt(artifact=artifact)

    assert store.append(receipt, artifact) == receipt
    assert store.append(receipt, artifact) == receipt
    assert store.list_receipts() == (receipt,)
    assert store.read_artifact(receipt.artifact_sha256) == artifact

    with pytest.raises(ReceiptConflict, match="receipt-1"):
        store.append(
            _receipt(receipt_id="receipt-1", kind="cost", artifact=artifact),
            artifact,
        )

    with pytest.raises(IntegrityError, match="artifact SHA-256"):
        store.append(_receipt(receipt_id="receipt-2", artifact=artifact), b"tampered")


def test_observer_hub_emits_evidence_only_and_graph_rebuilds_from_files(
    tmp_path,
) -> None:
    receipt_store = ReceiptStore(tmp_path / "receipts")
    graph = EvidenceGraph(tmp_path / "graph")
    hub = ObserverHub(
        (
            ExternalTraceObserver(),
            NativeOutcomeObserver(),
            CostObserver(),
            SafetyObserver(),
        ),
        graph=graph,
    )
    native_artifact = b'{"resolved":true}\n'
    native = _receipt(
        artifact=native_artifact,
        payload={
            "arm": "taught",
            "resolved": True,
            "evaluator_error": None,
            "task_revision_id": "task-r1",
        },
    )
    receipt_store.append(native, native_artifact)

    evidence = hub.observe(native)

    assert len(evidence) == 1
    assert evidence[0].observer_id == "native-v1"
    assert evidence[0].grade is ClaimGrade.E1
    assert not hasattr(hub, "promote")

    rebuilt = EvidenceGraph.rebuild(tmp_path / "graph", receipt_store)
    assert rebuilt.evidence_for_receipt(native.receipt_id) == evidence
    assert rebuilt.evidence_by_observer("native-v1") == evidence
    assert rebuilt.evidence_for_plan("plan-1") == evidence


def _native_evidence(
    tmp_path,
    *,
    arm: str,
    resolved: bool,
    receipt_id: str,
    evaluator_error: str | None = None,
    config_sha256: str = "c" * 64,
):
    artifact = f'{{"arm":"{arm}","resolved":{str(resolved).lower()}}}\n'.encode()
    receipt = _receipt(
        receipt_id=receipt_id,
        artifact=artifact,
        payload={
            "arm": arm,
            "resolved": resolved,
            "evaluator_error": evaluator_error,
            "task_revision_id": "task-r1",
            "task_source_sha256": "a" * 64,
            "model_identity": "local-mlx/Qwen3.5-4B@4bit",
            "native_evaluator_id": "swebench@v1",
            "execution_config_sha256": config_sha256,
        },
    )
    store = ReceiptStore(tmp_path / "receipts")
    store.append(receipt, artifact)
    graph = EvidenceGraph(tmp_path / "graph")
    return ObserverHub((NativeOutcomeObserver(),), graph=graph).observe(receipt)[0]


@pytest.mark.parametrize(
    ("baseline_resolved", "taught_resolved", "error", "expected"),
    [
        (False, True, None, ClaimClassification.GAIN),
        (True, True, None, ClaimClassification.NEUTRAL),
        (False, False, None, ClaimClassification.NEUTRAL),
        (True, False, None, ClaimClassification.REGRESSION),
        (False, True, "container failed", ClaimClassification.INFRA_FAILURE),
    ],
)
def test_claim_engine_strictly_classifies_matched_native_pairs(
    tmp_path, baseline_resolved, taught_resolved, error, expected
) -> None:
    baseline = _native_evidence(
        tmp_path,
        arm="baseline",
        resolved=baseline_resolved,
        receipt_id="baseline-native",
        evaluator_error=error,
    )
    taught = _native_evidence(
        tmp_path,
        arm="taught",
        resolved=taught_resolved,
        receipt_id="taught-native",
    )
    graph = EvidenceGraph(tmp_path / "graph")

    pair = align_native_pair(baseline, taught)
    claim = ClaimEngine(graph).classify_pair("candidate-1", pair)

    assert claim.classification is expected
    assert graph.latest_claims() == (claim,)
    assert graph.classification_counts() == {str(expected): 1}


def test_alignment_rejects_unmatched_config_and_claim_correction_is_superseding(
    tmp_path,
) -> None:
    baseline = _native_evidence(
        tmp_path,
        arm="baseline",
        resolved=False,
        receipt_id="baseline-native",
    )
    unmatched = _native_evidence(
        tmp_path,
        arm="taught",
        resolved=True,
        receipt_id="unmatched-native",
        config_sha256="d" * 64,
    )
    with pytest.raises(AlignmentError, match="execution_config_sha256"):
        align_native_pair(baseline, unmatched)

    taught_failure = _native_evidence(
        tmp_path,
        arm="taught",
        resolved=False,
        receipt_id="taught-failure-native",
    )
    graph = EvidenceGraph(tmp_path / "graph")
    engine = ClaimEngine(graph)
    initial = engine.classify_pair(
        "candidate-1", align_native_pair(baseline, taught_failure)
    )

    taught_success = _native_evidence(
        tmp_path,
        arm="taught",
        resolved=True,
        receipt_id="taught-success-native",
    )
    corrected = engine.classify_pair(
        "candidate-1",
        align_native_pair(baseline, taught_success),
        supersedes=initial,
    )

    assert corrected.classification is ClaimClassification.GAIN
    assert corrected.supersedes_claim_id == initial.claim_id
    assert len(graph.list_claims()) == 2
    assert graph.latest_claims() == (corrected,)


def test_receipt_store_rejects_concurrent_writer_and_detects_file_tampering(
    tmp_path,
) -> None:
    store = ReceiptStore(tmp_path / "receipts")
    artifact = b'{"status":"failed"}\n'
    receipt = _receipt(artifact=artifact)

    with store.acquire_writer():
        with pytest.raises(ConcurrentWriterError, match="already held"):
            store.append(receipt, artifact)

    store.append(receipt, artifact)
    original = store.log_path.read_text(encoding="utf-8")
    store.log_path.write_text(
        original.replace('"kind":"native_evaluation"', '"kind":"cost"'),
        encoding="utf-8",
    )
    with pytest.raises(IntegrityError, match="receipt hash mismatch"):
        store.list_receipts()


def test_evidence_graph_detects_immutable_record_hash_mismatch(tmp_path) -> None:
    envelope = _native_evidence(
        tmp_path,
        arm="baseline",
        resolved=False,
        receipt_id="baseline-native",
    )
    graph = EvidenceGraph(tmp_path / "graph")
    original = graph.evidence_path.read_text(encoding="utf-8")
    graph.evidence_path.write_text(
        original.replace('"resolved":false', '"resolved":true'),
        encoding="utf-8",
    )

    with pytest.raises(IntegrityError, match="graph record hash mismatch"):
        graph.evidence_for_receipt(envelope.receipt_ids[0])
