import json
from pathlib import Path

import pytest

import evolve_service
from evolve_service import ServiceCLIError, inspect_result, rollback_plan, run_cli

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/v1.0.0/MANIFEST.json"
RC_CONFIG = ROOT / "artifacts/v1.0.0/v0.9.0-release-candidate/configs/experiment.json"
RC_RESULT = (
    ROOT / "artifacts/v1.0.0/v0.9.0-release-candidate/runs/rc-pass3-1/result.json"
)


def test_inspect_and_verify_commands_return_machine_readable_results(capsys):
    inspected = inspect_result(RC_RESULT)
    assert inspected["decision"] == "accepted"
    assert inspected["stage"] == "v0.9.0-release-candidate"
    assert inspected["checks_passed"] == inspected["checks_total"] == 12

    assert run_cli(["verify", "--manifest", str(MANIFEST)]) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["valid"] is True
    assert stdout["failures"] == []


@pytest.mark.parametrize("kind", ["agent-program", "agent-code", "skill", "evaluator"])
def test_rollback_plan_is_explicitly_non_mutating(kind: str):
    plan = rollback_plan(kind=kind, evidence_root=ROOT / "artifacts/v1.0.0")
    assert plan["kind"] == kind
    assert plan["applied"] is False
    assert plan["requires_new_operation"] is True
    assert plan["steps"]
    assert all(Path(path).is_file() for path in plan["evidence_paths"])


def test_run_command_completes_rc_and_refuses_nonempty_output(tmp_path: Path, capsys):
    output = tmp_path / "service-run"

    assert (
        run_cli(
            [
                "run",
                "--config",
                str(RC_CONFIG),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["decision"] == "accepted"
    assert (output / "result.json").is_file()

    with pytest.raises(ServiceCLIError, match="non-empty"):
        run_cli(
            [
                "run",
                "--config",
                str(RC_CONFIG),
                "--output",
                str(output),
            ]
        )


def test_multimodel_and_swe_probe_commands_are_machine_readable(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
):
    registry = tmp_path / "models.json"
    suite = tmp_path / "suite.json"
    registry.write_text("{}", encoding="utf-8")
    suite.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        evolve_service,
        "probe_model_registry",
        lambda path: [
            {"id": "qwen4b", "required": True, "enabled": True, "status": "available"}
        ],
    )
    monkeypatch.setattr(
        evolve_service,
        "run_multi_model_suite",
        lambda **kwargs: {
            "summary": {"decision_scope": "diagnostic_not_model_promotion"},
            "cells": [{"safe": True}],
        },
    )
    monkeypatch.setattr(
        evolve_service,
        "probe_swe_environment",
        lambda path: {
            "ready": False,
            "status": "adapter_ready_runtime_blocked",
            "blockers": ["docker_daemon_unavailable"],
        },
    )

    assert run_cli(["model-probe", "--registry", str(registry)]) == 0
    assert json.loads(capsys.readouterr().out)["models"][0]["status"] == "available"

    output = tmp_path / "matrix"
    assert (
        run_cli(
            [
                "benchmark-run",
                "--registry",
                str(registry),
                "--suite",
                str(suite),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["cells"] == 1

    assert run_cli(["swe-probe", "--path", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["ready"] is False
