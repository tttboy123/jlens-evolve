from __future__ import annotations

from pathlib import Path

import pytest

from verify_continuous_ab import verify_stage


@pytest.mark.xfail(
    strict=True,
    reason=(
        "documented operator modification of the v2.1.0-continuous-ab frozen "
        "source-snapshot on 2026-08-05 (amendment 024; incident "
        "INCIDENT-OPERATOR-SNAPSHOT-OVERWRITE-2026-08-05): 4 pinned files "
        "irrecoverable, execution_bridge_sources_match is expected False. "
        "XPASS (files restored) means the marker must be removed."
    ),
)
def test_frozen_v21_stage_is_internally_recoverable_and_tamper_evident():
    root = Path(__file__).resolve().parents[1]
    stage = root / "artifacts/v2.1.0/v2.1.0-continuous-ab"

    result = verify_stage(stage)

    assert result["status"] == "verified"
    assert result["checks"]
    assert all(result["checks"].values())
    assert result["facts"]["selected_task_count"] == 300
    assert result["facts"]["final_sealed_unopened"] == 60
    assert {
        "service_manifest_tamper_evident",
        "changeset_registry_tamper_evident",
        "terminal_prediction_boundary_wrapper",
        "pilot_predeclared_without_opening_tasks",
        "pilot_paid_actions_human_required",
        "integrity_pass3_stable",
        "execution_bridge_covers_all_adapters",
        "execution_bridge_is_zero_cost_only",
        "execution_bridge_sources_match",
    }.issubset(result["checks"])
