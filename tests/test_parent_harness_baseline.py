from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from test_counterfactual_lineage import _arm, _sha
from test_qwen_transport import _compiled, _plan

from evolve.contracts import (
    ContractViolation,
    EvidenceEnvelope,
    Receipt,
    canonical_json,
)
from evolve.evidence import build_matched_counterfactual_pair
from evolve.runtime.qwen_transport import LegacyQwenCellRunner


def _runner(tmp_path: Path, *, proposed: Path, parent: Path | None):
    legacy_root = tmp_path / "legacy"
    model_root = tmp_path / "model"
    legacy_root.mkdir()
    model_root.mkdir()
    taskset = tmp_path / "TASKSET.json"
    routes = tmp_path / "ROUTES.json"
    taskset.write_text('{"tasks": []}\n', encoding="utf-8")
    routes.write_text('{"routes": {}}\n', encoding="utf-8")
    return LegacyQwenCellRunner(
        legacy_root=legacy_root,
        model_path=model_root,
        taskset_path=taskset,
        routes_path=routes,
        compiled_revision_root=proposed,
        baseline_compiled_revision_root=parent,
    )


def test_baseline_loads_current_parent_without_reading_proposed_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_root = tmp_path / "parent"
    proposed_root = tmp_path / "proposed"
    parent = _compiled(parent_root, "Use the current best harness.")
    parent = replace(
        parent,
        change_set=replace(
            parent.change_set,
            revision_id="parent-r1",
            parent_revision_id="empty-harness-v1",
        ),
        skill=replace(parent.skill, revision_id="parent-r1"),
        operator=replace(parent.operator, revision_id="parent-r1"),
        router=replace(parent.router, revision_id="parent-r1"),
    )
    loaded: list[Path] = []

    def load(root: str | Path):
        resolved = Path(root).resolve()
        loaded.append(resolved)
        if resolved == proposed_root.resolve():
            raise AssertionError("baseline read proposed candidate root")
        return parent

    monkeypatch.setattr("evolve.runtime.qwen_transport.CompiledRevision.load", load)
    runner = _runner(tmp_path, proposed=proposed_root, parent=parent_root)

    compiled = runner._compiled_for_plan(
        replace(_plan(arm="baseline"), candidate_revision_id="parent-r1")
    )
    lineage = runner._parent_harness_lineage(
        replace(_plan(arm="baseline"), candidate_revision_id="parent-r1"), compiled
    )

    assert compiled is parent
    assert loaded == [parent_root.resolve()]
    assert lineage["parent_harness_revision_id"] == "parent-r1"
    assert lineage["parent_harness_bundle_sha256"] == parent.bundle_sha256
    assert lineage["parent_harness_prompt"].startswith("{")
    assert (
        lineage["parent_harness_prompt_sha256"]
        == hashlib.sha256(lineage["parent_harness_prompt"].encode()).hexdigest()
    )


def test_runner_rejects_aliasing_parent_and_proposed_roots(tmp_path: Path) -> None:
    shared = tmp_path / "shared-revision"

    with pytest.raises(ContractViolation, match="separate roots"):
        _runner(tmp_path, proposed=shared, parent=shared)


def test_baseline_condition_projects_parent_with_explicit_marker(
    tmp_path: Path,
) -> None:
    class Adapter:
        @staticmethod
        def experiment_config():
            return {"temperature": 0}

    captured: dict[str, str] = {}

    def builder(*, taught_skill: str, **_kwargs):
        captured["teaching"] = taught_skill
        from types import SimpleNamespace

        return (
            SimpleNamespace(
                mechanism="operator", teaching="baseline", revision="baseline"
            ),
        )

    parent = _compiled(tmp_path / "parent", "Use the current best harness.")
    LegacyQwenCellRunner._condition_for_plan(
        plan=replace(_plan(arm="baseline"), candidate_revision_id="candidate-taught"),
        mechanism="operator",
        adapter=Adapter(),
        compiled=parent,
        builder=builder,
    )

    assert captured["teaching"].startswith("BASELINE-HARNESS:\n{")
    assert "Use the current best harness." in captured["teaching"]


def test_pair_rejects_parent_harness_mix_across_native_lineage() -> None:
    baseline = _arm(arm="baseline", resolved=False, candidate=False)
    taught = _arm(arm="taught", resolved=True, candidate=True)
    forged_native = replace(
        taught[4],
        payload={
            **taught[4].payload,
            "parent_harness_bundle_sha256": _sha("different-parent"),
        },
        artifact_sha256=_sha(
            canonical_json(
                {
                    **taught[3].payload,
                    "parent_harness_bundle_sha256": _sha("different-parent"),
                }
            )
        ),
    )

    with pytest.raises(ContractViolation, match="parent harness"):
        build_matched_counterfactual_pair(
            candidate_id="candidate-1",
            candidate_revision_id="candidate-r2",
            candidate_bundle_sha256=_sha("candidate"),
            baseline_model_receipt=baseline[0],
            baseline_external_evidence=baseline[2],
            baseline_native_evidence=baseline[4],
            taught_model_receipt=taught[0],
            taught_external_evidence=taught[2],
            taught_native_evidence=forged_native,
        )


def test_round_zero_empty_parent_lineage_remains_supported() -> None:
    baseline = _arm(arm="baseline", resolved=False, candidate=False)
    taught = _arm(arm="taught", resolved=True, candidate=True)
    names = {
        "parent_harness_revision_id",
        "parent_harness_bundle_sha256",
        "parent_harness_prompt_sha256",
        "parent_harness_prompt",
    }

    def without_parent(item: Receipt | EvidenceEnvelope):
        payload = {
            key: value for key, value in item.payload.items() if key not in names
        }
        receipt_payload = dict(payload)
        for projected in ("campaign_id", "plan_id", "receipt_kind"):
            receipt_payload.pop(projected, None)
        return replace(
            item, payload=payload, artifact_sha256=_sha(canonical_json(receipt_payload))
        )

    def round_zero(arm):
        model = without_parent(arm[0])
        external = without_parent(
            replace(
                arm[2],
                payload={
                    **arm[2].payload,
                    "model_artifact_sha256": model.artifact_sha256,
                },
            )
        )
        native = without_parent(
            replace(
                arm[4],
                payload={
                    **arm[4].payload,
                    "model_artifact_sha256": model.artifact_sha256,
                },
            )
        )
        return model, external, native

    baseline_model, baseline_external, baseline_native = round_zero(baseline)
    taught_model, taught_external, taught_native = round_zero(taught)
    pair = build_matched_counterfactual_pair(
        candidate_id="candidate-1",
        candidate_revision_id="candidate-r2",
        candidate_bundle_sha256=_sha("candidate"),
        baseline_model_receipt=baseline_model,
        baseline_external_evidence=baseline_external,
        baseline_native_evidence=baseline_native,
        taught_model_receipt=taught_model,
        taught_external_evidence=taught_external,
        taught_native_evidence=taught_native,
    )

    assert pair.parent_harness_revision_id is None
    assert pair.parent_harness_bundle_sha256 is None
    assert pair.parent_harness_prompt_sha256 is None
