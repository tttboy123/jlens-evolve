"""Offline tests for the authorized P1 patch-realization strategy call."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from skill_evolution_loop.contracts import canonical_json, sha256_json
from skill_evolution_loop.p1_parent import P1ParentCallAuthorization
from skill_evolution_loop.p1_strategy import (
    dispatch_p1_realization_strategy_call,
    freeze_p1_realization_strategy_request,
)
from teacher_api import TeacherClient, TeacherConfig, TeacherProvider


class _FakeResponse:
    status_code = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def json(self) -> dict[str, Any]:
        return self.payload


def _checkpoint(path: Path) -> None:
    value = {
        "schema_version": 1,
        "status": "capability_gate_not_met_sampling_stopped",
        "fixed_inputs": {
            "taskset_fingerprint": "a" * 64,
            "qualification_fingerprint": "b" * 64,
            "skill_revision_fingerprint": "c" * 64,
            "student_model": "Qwen3.5-4B-mlx-4bit",
        },
        "experiments": [
            {
                "run": "symbol-v2",
                "task_id": "feedback-one",
                "action_space": "python-symbol-rewrite-v1",
                "taught_raw_sha256": "d" * 64,
                "taught_structural_valid": False,
                "finding": "The diagnosis is correct but replacement copies the symbol.",
            }
        ],
        "success_gate": {
            "baseline_native_fail_to_taught_native_resolved": 0,
            "met": False,
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def test_strategy_request_and_call_are_frozen_advisory_and_token_audited(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    research = tmp_path / "research.md"
    request_path = tmp_path / "request.json"
    _checkpoint(checkpoint)
    research.write_text("Primary-source research synthesis.", encoding="utf-8")
    request = freeze_p1_realization_strategy_request(
        checkpoint_path=checkpoint,
        research_path=research,
        output_path=request_path,
    )
    approval = P1ParentCallAuthorization.create(
        request_sha256=request["strategy_request_sha256"],
        model="deepseek-v4-flash",
        maximum_output_tokens=64_000,
        authorization_id="p1-realization-r6",
        approved_by="user",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(
        canonical_json(approval.to_dict()) + "\n", encoding="utf-8"
    )
    calls: list[dict[str, Any]] = []

    def transport(url, headers=None, json=None, timeout=None):
        calls.append(json)
        return _FakeResponse(
            {
                "model": "deepseek-v4-flash",
                "usage": {"prompt_tokens": 1_000, "completion_tokens": 2_000},
                "choices": [
                    {
                        "message": {
                            "content": canonical_json(
                                {
                                    "failure_analysis": "Diagnosis is not executed.",
                                    "recommended_action_space": "Typed edit operators.",
                                    "operator_catalog": [
                                        {
                                            "name": "replace_symbol_body",
                                            "arguments": {
                                                "file": "relative path",
                                                "symbol": "qualified symbol",
                                                "body": "typed AST blueprint",
                                            },
                                            "postconditions": [
                                                "tree changed",
                                                "syntax valid",
                                            ],
                                        }
                                    ],
                                    "realization_loop": [
                                        "localize",
                                        "instantiate",
                                        "materialize",
                                    ],
                                    "verifier_policy": ["reject no-op", "run native"],
                                    "causal_eval": "Freeze mechanism, vary only Skill.",
                                    "compiled_skill_requirements": [
                                        "operator selection",
                                        "no-op rejection",
                                    ],
                                }
                            )
                        }
                    }
                ],
            }
        )

    client = TeacherClient(
        TeacherConfig(
            provider=TeacherProvider.DEEPSEEK,
            api_base="https://api.deepseek.com",
            model="deepseek-v4-flash",
            api_key_env="P1_TEST_KEY",
        ),
        api_key="test-key",
        transport=transport,
    )
    output_path = tmp_path / "response.json"
    report = dispatch_p1_realization_strategy_call(
        request_evidence_path=request_path,
        authorization_path=authorization_path,
        ledger_path=tmp_path / "ledger.json",
        output_path=output_path,
        call_id="p1-realization-round-006",
        client=client,
    )

    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 64_000
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["thinking"] == {"type": "enabled"}
    assert report["event_type"] == "parent-strategy-response"
    assert report["candidate_status"] == "advisory_inactive"
    assert report["auto_apply"] is False
    assert report["tokens_charged"] == 3_000
    assert report["goal_parent_token_limit"] == 3_000_000
    content = {k: v for k, v in report.items() if k != "evidence_sha256"}
    assert report["evidence_sha256"] == sha256_json(content)
