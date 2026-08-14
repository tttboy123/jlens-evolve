from __future__ import annotations

from pathlib import Path

import pytest

from evolve.autonomous import AutonomousEvolutionError
from evolve.autonomous.verification import CampaignOutcomeVerifier


def _result(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "campaign_id": "campaign-1",
        "campaign_status": "completed",
        "execution_statuses": ["completed"] * 6,
        "claims": [
            {
                "task_id": f"task-{index}",
                "classification": "neutral",
                "claim_id": f"claim-{index}",
            }
            for index in range(3)
        ],
        "capability_active": False,
        "holdout_opened": False,
        "burned_holdout_opened": False,
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("holdout_opened", True),
        ("burned_holdout_opened", True),
        ("capability_active", True),
    ),
)
def test_campaign_verifier_rejects_safety_state_self_reports(
    tmp_path: Path, field: str, value: bool
) -> None:
    with pytest.raises(AutonomousEvolutionError, match="safety invariant"):
        CampaignOutcomeVerifier().verify(
            round_root=tmp_path,
            result=_result(**{field: value}),
            selected_task_ids=("task-0", "task-1", "task-2"),
            candidate_id="candidate-1",
            candidate_revision_id="candidate-r1",
            candidate_bundle_sha256="a" * 64,
        )

def test_campaign_verifier_rejects_claim_summary_without_authoritative_graph(
    tmp_path: Path,
) -> None:
    with pytest.raises(AutonomousEvolutionError, match="authoritative"):
        CampaignOutcomeVerifier().verify(
            round_root=tmp_path,
            result=_result(),
            selected_task_ids=("task-0", "task-1", "task-2"),
            candidate_id="candidate-1",
            candidate_revision_id="candidate-r1",
            candidate_bundle_sha256="a" * 64,
        )
