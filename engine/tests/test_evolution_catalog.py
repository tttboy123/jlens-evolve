from __future__ import annotations

import json

import pytest

from skill_evolution_loop.evolution_catalog import (
    CatalogConflict,
    ContractError,
    EvolutionCatalog,
    EvolutionRecord,
)


def _record(*, record_id: str = "r073-issue-seeding") -> EvolutionRecord:
    return EvolutionRecord.create(
        record_type="mechanisms",
        record_id=record_id,
        title="Issue-anchored candidate seeding",
        status="implemented",
        capability_tags=("localization",),
        task_tags=("swe-bench",),
        failure_mode_tags=("wrong-target",),
        source_model="deepseek-chat",
        source_runtime="openai-compatible",
        payload={"round": "r073", "mechanism": "issue-anchor"},
        evidence_refs=(
            {
                "path": "runs/skill-evolution-loop/r073/SUMMARY.json",
                "sha256": "a" * 64,
            },
        ),
        cross_model_validations=(),
    )


def test_catalog_is_append_only_deduplicated_and_searchable(tmp_path) -> None:
    catalog = EvolutionCatalog(tmp_path)
    first = catalog.append(_record())
    duplicate = catalog.append(_record(record_id="r074-same-mechanism"))

    assert first.created is True
    assert duplicate.created is False
    assert duplicate.path == first.path
    assert catalog.search(
        record_types=("mechanisms",),
        capability_tags=("localization",),
        statuses=("implemented",),
    ) == (_record(),)

    index = json.loads((tmp_path / "indexes" / "CATALOG.json").read_text())
    assert index["record_count"] == 1
    assert index["evidence_sha256"]


def test_catalog_refuses_record_id_reuse_with_different_content(tmp_path) -> None:
    catalog = EvolutionCatalog(tmp_path)
    catalog.append(_record())
    changed = EvolutionRecord.create(
        **{
            **_record().content_dict(),
            "title": "Different mechanism",
        }
    )

    with pytest.raises(CatalogConflict, match="record_id"):
        catalog.append(changed)


def test_proposal_context_surfaces_existing_mechanisms_before_parent_call(
    tmp_path,
) -> None:
    catalog = EvolutionCatalog(tmp_path)
    catalog.append(_record())

    context = catalog.proposal_context(
        capability_tags=("localization",),
        failure_mode_tags=("wrong-target",),
        limit=10,
    )

    assert context["implemented_mechanisms"][0]["record_id"] == ("r073-issue-seeding")
    assert context["query_fingerprint"]


def test_cross_model_validation_is_model_specific_but_skill_is_reusable() -> None:
    record = EvolutionRecord.create(
        record_type="skills",
        record_id="skill-localization-001",
        title="Anchor edit selection to issue evidence",
        status="candidate",
        capability_tags=("localization",),
        task_tags=("swe-bench",),
        failure_mode_tags=("wrong-target",),
        source_model="deepseek-chat",
        source_runtime="api",
        payload={"skill_sha256": "b" * 64},
        evidence_refs=(),
        cross_model_validations=(
            {
                "target_model": "qwen-cuda",
                "target_runtime": "vllm",
                "outcome": "pending",
                "evidence_sha256": None,
            },
        ),
    )

    assert record.source_model == "deepseek-chat"
    assert record.cross_model_validations[0]["target_model"] == "qwen-cuda"


def test_catalog_writer_lease_fails_fast_and_cleans_up(tmp_path) -> None:
    catalog = EvolutionCatalog(tmp_path)
    record = EvolutionRecord.create(
        record_type="mechanisms",
        record_id="lease-mechanism-001",
        title="lease",
        status="implemented",
        capability_tags=(),
        task_tags=(),
        failure_mode_tags=(),
        source_model="none",
        source_runtime="local",
        payload={},
        evidence_refs=(),
        cross_model_validations=(),
    )
    catalog.append(record)
    assert not catalog._lock_path.exists()

    # manually hold the lease, then a second append must fail fast
    catalog._lock_path.parent.mkdir(parents=True, exist_ok=True)
    catalog._lock_path.write_text("held")
    second = EvolutionRecord.create(
        record_type="mechanisms",
        record_id="lease-mechanism-002",
        title="lease two",
        status="implemented",
        capability_tags=(),
        task_tags=(),
        failure_mode_tags=(),
        source_model="none",
        source_runtime="local",
        payload={},
        evidence_refs=(),
        cross_model_validations=(),
    )
    with pytest.raises(ContractError, match="writer lease"):
        catalog.append(second)
    # lease remains held by the external holder
    assert catalog._lock_path.exists()
    catalog._lock_path.unlink()

    # after the external holder releases, append succeeds and cleans up
    catalog.append(second)
    assert not catalog._lock_path.exists()
