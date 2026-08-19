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
    EvidenceEnvelope,
    Receipt,
    canonical_json,
)
from evolve.evidence import (
    ClaimEngine,
    EvidenceGraph,
    IntegrityError,
    ReceiptStore,
    build_matched_counterfactual_pair,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _receipt(*, arm: str, kind: str, payload: dict[str, object]) -> Receipt:
    complete = dict(payload)
    return Receipt(
        receipt_id=f"receipt-{arm}-{kind}",
        campaign_id="campaign-1",
        plan_id=f"plan-{arm}",
        sequence={"model": 1, "external_trace": 2, "native_evaluation": 3}[kind],
        kind=kind,
        created_at="2026-08-14T06:00:00Z",
        payload=complete,
        artifact_sha256=_sha(canonical_json(complete)),
    )


def _arm(
    *,
    arm: str,
    resolved: bool,
    candidate: bool,
    prompt_mode: str = "valid",
):
    prediction = _sha(f"prediction:{arm}")
    candidate_bundle_sha256 = _sha("candidate")
    candidate_prompt = canonical_json(
        {
            "candidate_revision_id": "candidate-r2",
            "candidate_bundle_sha256": candidate_bundle_sha256,
        }
    )
    parent_harness_revision_id = "parent-r1"
    parent_harness_bundle_sha256 = _sha("parent-harness")
    parent_harness_prompt = canonical_json(
        {
            "revision_id": parent_harness_revision_id,
            "bundle_sha256": parent_harness_bundle_sha256,
        }
    )
    parent_harness_prompt_sha256 = _sha(parent_harness_prompt)
    parent_lineage: dict[str, object] = {
        "parent_harness_revision_id": parent_harness_revision_id,
        "parent_harness_bundle_sha256": parent_harness_bundle_sha256,
        "parent_harness_prompt": parent_harness_prompt,
        "parent_harness_prompt_sha256": parent_harness_prompt_sha256,
    }
    prompt_text = (
        f"SYSTEM: taught repair\nCOMPILED-CANDIDATE:\n{candidate_prompt}"
        if candidate
        else f"SYSTEM: baseline repair\nBASELINE-HARNESS:\n{parent_harness_prompt}"
    )
    if prompt_mode == "baseline-leak":
        prompt_text = f"SYSTEM: baseline repair\n{candidate_prompt}"
    elif prompt_mode == "taught-omission":
        prompt_text = "SYSTEM: taught repair without compiled candidate"
    prompt_lineage: dict[str, object] = {
        "prompt_texts": [prompt_text],
        "prompt_sha256": [_sha(prompt_text)],
        "candidate_prompt": candidate_prompt if candidate else None,
        "candidate_prompt_sha256": _sha(candidate_prompt) if candidate else None,
        "compiled_artifact_sha256": (
            {"COMPILED-REVISION.json": candidate_bundle_sha256} if candidate else {}
        ),
    }
    if prompt_mode == "missing":
        prompt_lineage = {}
    external_prompt_lineage = dict(prompt_lineage)
    if prompt_mode == "external-mismatch":
        external_prompt_lineage["prompt_texts"] = ["forged external prompt"]
        external_prompt_lineage["prompt_sha256"] = [_sha("forged external prompt")]
    model = _receipt(
        arm=arm,
        kind="model",
        payload={
            "patch_sha256": prediction,
            "provider": "local-mlx",
            "model": "qwen",
            "revision": "frozen-r1",
            "candidate_consumed": candidate,
            "candidate_revision_id": "candidate-r2" if candidate else None,
            "candidate_bundle_sha256": candidate_bundle_sha256 if candidate else None,
            **parent_lineage,
            **prompt_lineage,
        },
    )
    external_receipt = _receipt(
        arm=arm,
        kind="external_trace",
        payload={
            "arm": arm,
            "task_revision_id": "task-r1",
            "model_receipt_id": model.receipt_id,
            "model_artifact_sha256": model.artifact_sha256,
            "prediction_sha256": prediction,
            "candidate_consumed": candidate,
            "candidate_revision_id": "candidate-r2" if candidate else None,
            "candidate_bundle_sha256": _sha("candidate") if candidate else None,
            **parent_lineage,
            **external_prompt_lineage,
        },
    )
    native_receipt = _receipt(
        arm=arm,
        kind="native_evaluation",
        payload={
            "arm": arm,
            "task_revision_id": "task-r1",
            "task_source_sha256": _sha("task-source"),
            "model_identity": "local-mlx/qwen@frozen-r1",
            "native_evaluator_id": "swebench@v1",
            "execution_config_sha256": _sha("execution-config"),
            "model_receipt_id": model.receipt_id,
            "model_artifact_sha256": model.artifact_sha256,
            "prediction_sha256": prediction,
            **parent_lineage,
            "resolved": resolved,
            "evaluator_error": None,
        },
    )

    def evidence(receipt: Receipt, observer_id: str, grade: ClaimGrade):
        return EvidenceEnvelope(
            evidence_id=f"evidence-{arm}-{receipt.kind}",
            receipt_ids=(receipt.receipt_id,),
            observer_id=observer_id,
            grade=grade,
            payload={
                "campaign_id": receipt.campaign_id,
                "plan_id": receipt.plan_id,
                "receipt_kind": receipt.kind,
                **receipt.payload,
            },
            artifact_sha256=receipt.artifact_sha256,
        )

    return (
        model,
        external_receipt,
        evidence(external_receipt, "external-trace-v1", ClaimGrade.E0),
        native_receipt,
        evidence(native_receipt, "native-v1", ClaimGrade.E1),
    )


@pytest.mark.parametrize(
    ("arm", "candidate", "prompt_mode"),
    (
        ("baseline", False, "missing"),
        ("baseline", False, "baseline-leak"),
        ("taught", True, "taught-omission"),
        ("taught", True, "external-mismatch"),
    ),
)
def test_e2_rejects_missing_or_forged_prompt_lineage(
    arm: str,
    candidate: bool,
    prompt_mode: str,
) -> None:
    baseline = _arm(
        arm="baseline",
        resolved=False,
        candidate=False,
        prompt_mode=prompt_mode if arm == "baseline" else "valid",
    )
    taught = _arm(
        arm="taught",
        resolved=True,
        candidate=True,
        prompt_mode=prompt_mode if arm == "taught" else "valid",
    )

    with pytest.raises(ContractViolation, match="prompt|candidate projection"):
        build_matched_counterfactual_pair(
            candidate_id="candidate-1",
            candidate_revision_id="candidate-r2",
            candidate_bundle_sha256=_sha("candidate"),
            baseline_model_receipt=baseline[0],
            baseline_external_evidence=baseline[2],
            baseline_native_evidence=baseline[4],
            taught_model_receipt=taught[0],
            taught_external_evidence=taught[2],
            taught_native_evidence=taught[4],
        )


def test_e2_claim_binds_complete_matched_counterfactual_pair(tmp_path: Path) -> None:
    baseline = _arm(arm="baseline", resolved=False, candidate=False)
    taught = _arm(arm="taught", resolved=True, candidate=True)
    pair = build_matched_counterfactual_pair(
        candidate_id="candidate-1",
        candidate_revision_id="candidate-r2",
        candidate_bundle_sha256=_sha("candidate"),
        baseline_model_receipt=baseline[0],
        baseline_external_evidence=baseline[2],
        baseline_native_evidence=baseline[4],
        taught_model_receipt=taught[0],
        taught_external_evidence=taught[2],
        taught_native_evidence=taught[4],
    )
    graph = EvidenceGraph(tmp_path / "graph")
    for envelope in (baseline[2], baseline[4], taught[2], taught[4]):
        graph.append_evidence(envelope)

    claim = ClaimEngine(graph).classify_pair(
        "candidate-1",
        align_native_pair(baseline[4], taught[4]),
        counterfactual_pair=pair,
    )

    assert claim.grade is ClaimGrade.E2
    assert claim.classification is ClaimClassification.GAIN
    assert claim.counterfactual_pair_sha256 == pair.content_sha256
    assert claim.evidence_ids == pair.evidence_ids
    assert claim.counterfactual_receipt_ids == pair.receipt_ids


def test_new_e2_claim_cannot_be_minted_from_native_metadata_only() -> None:
    with pytest.raises(ContractViolation, match="complete counterfactual lineage"):
        Claim(
            claim_id="claim-unbound",
            candidate_id="candidate-1",
            grade=ClaimGrade.E2,
            classification=ClaimClassification.GAIN,
            evidence_ids=("native-baseline", "native-taught"),
            rationale="native hashes happen to match",
            supersedes_claim_id=None,
        )


def test_new_e3_claim_cannot_bypass_counterfactual_lineage() -> None:
    with pytest.raises(ContractViolation, match="complete counterfactual lineage"):
        Claim(
            claim_id="claim-unbound-e3",
            candidate_id="candidate-1",
            grade=ClaimGrade.E3,
            classification=ClaimClassification.GAIN,
            evidence_ids=("native-baseline", "native-taught"),
            rationale="observer id and prediction hashes self-report equality",
            supersedes_claim_id=None,
        )


def test_graph_rejects_direct_e2_append_without_pair_object(tmp_path: Path) -> None:
    forged = Claim(
        claim_id="claim-shaped-but-unverified",
        candidate_id="candidate-1",
        grade=ClaimGrade.E2,
        classification=ClaimClassification.GAIN,
        evidence_ids=tuple(f"evidence-{index}" for index in range(4)),
        rationale="four ids and six receipts are still only metadata",
        supersedes_claim_id=None,
        counterfactual_pair_sha256="a" * 64,
        counterfactual_receipt_ids=tuple(f"receipt-{index}" for index in range(6)),
    )

    with pytest.raises(IntegrityError, match="complete counterfactual pair"):
        EvidenceGraph(tmp_path / "graph").append_claim(forged)


def test_graph_rebuild_rejects_counterfactual_pair_hash_tampering(
    tmp_path: Path,
) -> None:
    baseline = _arm(arm="baseline", resolved=False, candidate=False)
    taught = _arm(arm="taught", resolved=True, candidate=True)
    pair = build_matched_counterfactual_pair(
        candidate_id="candidate-1",
        candidate_revision_id="candidate-r2",
        candidate_bundle_sha256=_sha("candidate"),
        baseline_model_receipt=baseline[0],
        baseline_external_evidence=baseline[2],
        baseline_native_evidence=baseline[4],
        taught_model_receipt=taught[0],
        taught_external_evidence=taught[2],
        taught_native_evidence=taught[4],
    )
    store = ReceiptStore(tmp_path / "receipts")
    graph = EvidenceGraph(tmp_path / "graph")
    for row in (*baseline, *taught):
        if isinstance(row, Receipt):
            store.append(row, canonical_json(row.payload).encode())
        elif isinstance(row, EvidenceEnvelope):
            graph.append_evidence(row)
    ClaimEngine(graph).classify_pair(
        "candidate-1",
        align_native_pair(baseline[4], taught[4]),
        counterfactual_pair=pair,
    )
    record = json.loads(graph.claims_path.read_text(encoding="utf-8"))
    record["value"]["counterfactual_pair_sha256"] = "f" * 64
    record["content_sha256"] = _sha(canonical_json(record["value"]))
    graph.claims_path.write_text(canonical_json(record) + "\n", encoding="utf-8")

    with pytest.raises(IntegrityError, match="counterfactual pair hash"):
        EvidenceGraph.rebuild(tmp_path / "graph", store)


@pytest.mark.parametrize(
    ("target_kind", "expected"),
    (
        ("external_trace", "external receipt kind"),
        ("native_evaluation", "native receipt kind"),
    ),
)
def test_graph_rebuild_rejects_counterfactual_evidence_backed_by_wrong_kind(
    tmp_path: Path,
    target_kind: str,
    expected: str,
) -> None:
    baseline = _arm(arm="baseline", resolved=False, candidate=False)
    taught = _arm(arm="taught", resolved=True, candidate=True)
    pair = build_matched_counterfactual_pair(
        candidate_id="candidate-1",
        candidate_revision_id="candidate-r2",
        candidate_bundle_sha256=_sha("candidate"),
        baseline_model_receipt=baseline[0],
        baseline_external_evidence=baseline[2],
        baseline_native_evidence=baseline[4],
        taught_model_receipt=taught[0],
        taught_external_evidence=taught[2],
        taught_native_evidence=taught[4],
    )
    store = ReceiptStore(tmp_path / "receipts")
    graph = EvidenceGraph(tmp_path / "graph")
    for row in (*baseline, *taught):
        if isinstance(row, Receipt):
            receipt = row
            if receipt.kind == target_kind and receipt.plan_id == "plan-baseline":
                receipt = replace(receipt, kind="model")
            store.append(receipt, canonical_json(receipt.payload).encode())
        elif isinstance(row, EvidenceEnvelope):
            graph.append_evidence(row)
    ClaimEngine(graph).classify_pair(
        "candidate-1",
        align_native_pair(baseline[4], taught[4]),
        counterfactual_pair=pair,
    )

    with pytest.raises(IntegrityError, match=expected):
        EvidenceGraph.rebuild(tmp_path / "graph", store)


def test_graph_rebuild_rejects_claim_classification_forged_over_native_outcomes(
    tmp_path: Path,
) -> None:
    baseline = _arm(arm="baseline", resolved=False, candidate=False)
    taught = _arm(arm="taught", resolved=True, candidate=True)
    pair = build_matched_counterfactual_pair(
        candidate_id="candidate-1",
        candidate_revision_id="candidate-r2",
        candidate_bundle_sha256=_sha("candidate"),
        baseline_model_receipt=baseline[0],
        baseline_external_evidence=baseline[2],
        baseline_native_evidence=baseline[4],
        taught_model_receipt=taught[0],
        taught_external_evidence=taught[2],
        taught_native_evidence=taught[4],
    )
    store = ReceiptStore(tmp_path / "receipts")
    graph = EvidenceGraph(tmp_path / "graph")
    for row in (*baseline, *taught):
        if isinstance(row, Receipt):
            store.append(row, canonical_json(row.payload).encode())
        elif isinstance(row, EvidenceEnvelope):
            graph.append_evidence(row)
    valid = ClaimEngine(graph).classify_pair(
        "candidate-1",
        align_native_pair(baseline[4], taught[4]),
        counterfactual_pair=pair,
    )
    record = json.loads(graph.claims_path.read_text(encoding="utf-8"))
    record["value"]["classification"] = "neutral"
    record["value"]["claim_id"] = valid.claim_id + "-forged"
    record["content_sha256"] = _sha(canonical_json(record["value"]))
    graph.claims_path.write_text(canonical_json(record) + "\n", encoding="utf-8")

    with pytest.raises(IntegrityError, match="classification"):
        EvidenceGraph.rebuild(tmp_path / "graph", store)


def test_claim_engine_rejects_pair_evidence_not_frozen_in_its_graph(
    tmp_path: Path,
) -> None:
    baseline = _arm(arm="baseline", resolved=False, candidate=False)
    taught = _arm(arm="taught", resolved=True, candidate=True)
    pair = build_matched_counterfactual_pair(
        candidate_id="candidate-1",
        candidate_revision_id="candidate-r2",
        candidate_bundle_sha256=_sha("candidate"),
        baseline_model_receipt=baseline[0],
        baseline_external_evidence=baseline[2],
        baseline_native_evidence=baseline[4],
        taught_model_receipt=taught[0],
        taught_external_evidence=taught[2],
        taught_native_evidence=taught[4],
    )

    with pytest.raises(ContractViolation, match="not frozen in EvidenceGraph"):
        ClaimEngine(EvidenceGraph(tmp_path / "graph")).classify_pair(
            "candidate-1",
            align_native_pair(baseline[4], taught[4]),
            counterfactual_pair=pair,
        )


def test_claim_engine_rejects_same_evidence_id_with_drifted_content(
    tmp_path: Path,
) -> None:
    baseline = _arm(arm="baseline", resolved=False, candidate=False)
    taught = _arm(arm="taught", resolved=True, candidate=True)
    pair = build_matched_counterfactual_pair(
        candidate_id="candidate-1",
        candidate_revision_id="candidate-r2",
        candidate_bundle_sha256=_sha("candidate"),
        baseline_model_receipt=baseline[0],
        baseline_external_evidence=baseline[2],
        baseline_native_evidence=baseline[4],
        taught_model_receipt=taught[0],
        taught_external_evidence=taught[2],
        taught_native_evidence=taught[4],
    )
    original = baseline[2]
    assert isinstance(original, EvidenceEnvelope)
    drifted_payload = {**original.payload, "structural_valid": False}
    receipt_payload = dict(drifted_payload)
    for name in ("campaign_id", "plan_id", "receipt_kind"):
        receipt_payload.pop(name)
    drifted = replace(
        original,
        payload=drifted_payload,
        artifact_sha256=_sha(canonical_json(receipt_payload)),
    )
    graph = EvidenceGraph(tmp_path / "graph")
    for envelope in (drifted, baseline[4], taught[2], taught[4]):
        assert isinstance(envelope, EvidenceEnvelope)
        graph.append_evidence(envelope)

    with pytest.raises(ContractViolation, match="not frozen in EvidenceGraph"):
        ClaimEngine(graph).classify_pair(
            "candidate-1",
            align_native_pair(baseline[4], taught[4]),
            counterfactual_pair=pair,
        )


def test_builder_rejects_baseline_candidate_and_native_model_rebinding() -> None:
    baseline = list(_arm(arm="baseline", resolved=False, candidate=False))
    taught = _arm(arm="taught", resolved=True, candidate=True)
    baseline_model = baseline[0]
    assert isinstance(baseline_model, Receipt)
    rebound_payload = {
        **baseline_model.payload,
        "candidate_consumed": True,
        "candidate_revision_id": "candidate-r2",
        "candidate_bundle_sha256": _sha("candidate"),
    }
    baseline[0] = _receipt(arm="baseline", kind="model", payload=rebound_payload)
    with pytest.raises(ContractViolation, match="does not bind model artifact"):
        build_matched_counterfactual_pair(
            candidate_id="candidate-1",
            candidate_revision_id="candidate-r2",
            candidate_bundle_sha256=_sha("candidate"),
            baseline_model_receipt=baseline[0],
            baseline_external_evidence=baseline[2],
            baseline_native_evidence=baseline[4],
            taught_model_receipt=taught[0],
            taught_external_evidence=taught[2],
            taught_native_evidence=taught[4],
        )


def test_legacy_unbound_e2_claim_requires_explicit_compatibility_reader(
    tmp_path: Path,
) -> None:
    value = {
        "claim_id": "legacy-claim",
        "candidate_id": "legacy-candidate",
        "grade": "E2",
        "classification": "gain",
        "evidence_ids": ["legacy-baseline", "legacy-taught"],
        "rationale": "sealed before counterfactual lineage contract",
        "supersedes_claim_id": None,
    }
    record = {"content_sha256": _sha(canonical_json(value)), "value": value}
    graph = EvidenceGraph(tmp_path / "graph")
    graph.claims_path.write_text(canonical_json(record) + "\n", encoding="utf-8")

    with pytest.raises(IntegrityError, match="invalid graph record"):
        graph.list_claims()

    legacy_graph = EvidenceGraph(tmp_path / "graph", allow_legacy_unbound_claims=True)
    (claim,) = legacy_graph.list_claims()

    assert claim.grade is ClaimGrade.E2
    assert claim.counterfactual_pair_sha256 is None
    assert claim.content_sha256 == record["content_sha256"]


def test_prompt_lineage_requires_embedded_prompt_texts() -> None:
    """Regression: frozen qwen cell RESULTs carried prompt_sha256 but no
    prompt_texts, so any natively-resolved baseline hit
    'baseline prompt evidence is missing or invalid'.  The lineage validator
    must accept embedded prompt_texts and reject the hash-only shape."""
    from evolve.contracts import ContractViolation
    from evolve.evidence.counterfactual import _validate_prompt_lineage

    prompt = "SYSTEM: baseline repair\nUSER: fix the bug"
    ok = {
        "prompt_texts": [prompt],
        "prompt_sha256": [_sha(prompt)],
        "candidate_prompt": None,
        "candidate_prompt_sha256": None,
        "compiled_artifact_sha256": {},
    }
    # With embedded prompt_texts the lineage validates (returns bundle hash).
    bundle_hash, bundled = _validate_prompt_lineage(
        "baseline",
        ok,
        candidate_revision_id="candidate-r1",
        candidate_bundle_sha256="a" * 64,
    )
    assert isinstance(bundle_hash, str) and bundle_hash

    # The old frozen shape (hashes only, no prompt_texts) is rejected.
    broken = {
        "prompt_texts": None,
        "prompt_sha256": [_sha(prompt)],
        "candidate_prompt": None,
        "candidate_prompt_sha256": None,
        "compiled_artifact_sha256": {},
    }
    with pytest.raises(ContractViolation):
        _validate_prompt_lineage(
            "baseline",
            broken,
            candidate_revision_id="candidate-r1",
            candidate_bundle_sha256="a" * 64,
        )


def test_prompt_lineage_rejects_empty_prompt_text_entries() -> None:
    """Regression: the operator clause gate appended an empty-string prompt to
    the trace; frozen baseline cells then carried prompt_texts with an empty
    entry, failing 'baseline prompt evidence is missing or invalid'."""
    from evolve.contracts import ContractViolation
    from evolve.evidence.counterfactual import _validate_prompt_lineage

    real_prompt = "SYSTEM: baseline repair\nUSER: fix it"
    broken = {
        "prompt_texts": [real_prompt, ""],
        "prompt_sha256": [_sha(real_prompt), _sha("")],
        "candidate_prompt": None,
        "candidate_prompt_sha256": None,
        "compiled_artifact_sha256": {},
    }
    with pytest.raises(ContractViolation):
        _validate_prompt_lineage(
            "baseline", broken,
            candidate_revision_id="candidate-r1", candidate_bundle_sha256="a" * 64,
        )

    ok = {
        "prompt_texts": [real_prompt],
        "prompt_sha256": [_sha(real_prompt)],
        "candidate_prompt": None,
        "candidate_prompt_sha256": None,
        "compiled_artifact_sha256": {},
    }
    bundle_hash, _ = _validate_prompt_lineage(
        "baseline", ok,
        candidate_revision_id="candidate-r1", candidate_bundle_sha256="a" * 64,
    )
    assert isinstance(bundle_hash, str) and bundle_hash
