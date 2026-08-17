from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from skill_evolution_loop.contracts import canonical_json, sha256_json
from skill_evolution_loop.round1_strategy import (
    create_round1_strategy_authorization,
    dispatch_round1_localization_strategy,
    freeze_round1_localization_strategy_request,
)
from teacher_api import TeacherConfig, TeacherProvider, TeacherResponse


class _Client:
    config = TeacherConfig(
        provider=TeacherProvider.DEEPSEEK,
        api_base="https://example.invalid",
        model="deepseek-v4-flash",
        api_key_env="TEST_KEY",
    )

    def complete(self, _sample):
        strategy = {
            "failure_diagnosis": "Retrieval recall and target choice are conflated.",
            "recommended_architecture": "Freeze hierarchical localization upstream.",
            "localization_stages": ["retrieve files", "rank symbols"],
            "leakage_controls": ["never read gold"],
            "causal_evaluation": "Use identical receipts across both arms.",
            "rollout_plan": ["qualify recall", "run feedback", "open holdout once"],
        }
        return TeacherResponse(
            provider=TeacherProvider.DEEPSEEK,
            model="deepseek-v4-flash",
            text=canonical_json(strategy),
            usage={"prompt_tokens": 30, "completion_tokens": 70, "total_tokens": 100},
        )


def _audit(path: Path) -> None:
    audit = {
        "schema_version": 1,
        "evaluator_only": True,
        "student_visible": False,
        "answer_content_serialized": False,
        "audit": {"audited_tasks": 60, "ready_tasks": 24, "rows": ["private"]},
    }
    path.write_text(
        canonical_json({**audit, "evidence_sha256": sha256_json(audit)}) + "\n",
        encoding="utf-8",
    )


def test_round1_strategy_request_contains_only_aggregate_failure(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "audit.json"
    research = tmp_path / "research.md"
    output = tmp_path / "request.json"
    _audit(audit_path)
    research.write_text("primary-source findings", encoding="utf-8")

    request = freeze_round1_localization_strategy_request(
        audit_path=audit_path,
        research_path=research,
        output_path=output,
    )

    serialized = json.dumps(request)
    assert request["request"]["aggregate_failure"]["covered_tasks"] == 24
    assert "private" not in serialized
    assert request["gold_paths_included"] is False


def test_round1_strategy_dispatch_is_audited_and_replay_safe(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    research = tmp_path / "research.md"
    request_path = tmp_path / "request.json"
    authorization_path = tmp_path / "authorization.json"
    response_path = tmp_path / "response.json"
    _audit(audit_path)
    research.write_text("primary-source findings", encoding="utf-8")
    freeze_round1_localization_strategy_request(
        audit_path=audit_path,
        research_path=research,
        output_path=request_path,
    )
    create_round1_strategy_authorization(
        request_path=request_path,
        output_path=authorization_path,
        expires_at=datetime.now(timezone.utc) + timedelta(days=1),
    )

    first = dispatch_round1_localization_strategy(
        request_path=request_path,
        authorization_path=authorization_path,
        ledger_path=tmp_path / "ledger.json",
        output_path=response_path,
        client=_Client(),
    )
    second = dispatch_round1_localization_strategy(
        request_path=request_path,
        authorization_path=authorization_path,
        ledger_path=tmp_path / "ledger.json",
        output_path=response_path,
        client=_Client(),
    )

    assert first == second
    assert first["campaign_tokens_charged"] == 100
    assert first["campaign_tokens_after"] == 155_327
    assert first["candidate_status"] == "advisory_inactive"
    assert first["auto_apply"] is False
