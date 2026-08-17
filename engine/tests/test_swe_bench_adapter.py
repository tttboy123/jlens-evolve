from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from swe_bench_adapter import (
    build_harness_command,
    evaluate_readiness,
    probe_environment,
    validate_prediction,
    write_predictions,
)


def _prediction() -> dict[str, str]:
    return {
        "instance_id": "django__django-123",
        "model_name_or_path": "qwen-coder",
        "model_patch": (
            "diff --git a/django/core/demo.py b/django/core/demo.py\n"
            "--- a/django/core/demo.py\n+++ b/django/core/demo.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        ),
    }


def test_prediction_writer_matches_official_three_field_contract(tmp_path: Path):
    prediction = validate_prediction(_prediction())
    output = tmp_path / "predictions.jsonl"

    write_predictions([prediction], output)

    assert json.loads(output.read_text(encoding="utf-8")) == prediction


def test_prediction_preflight_rejects_test_poisoning_and_extra_fields():
    poisoned = _prediction()
    poisoned["model_patch"] = poisoned["model_patch"].replace(
        "django/core/demo.py", "tests/test_demo.py"
    )
    with pytest.raises(ValueError, match="test path"):
        validate_prediction(poisoned)

    extra = {**_prediction(), "reasoning": "hidden"}
    with pytest.raises(ValueError, match="exactly"):
        validate_prediction(extra)


def test_empty_model_patch_is_preserved_as_an_unresolved_prediction():
    empty = {**_prediction(), "model_patch": ""}

    prediction = validate_prediction(empty)

    assert prediction["model_patch"] == ""


def test_harness_command_is_non_shell_single_worker_and_arm_local_build(tmp_path: Path):
    command = build_harness_command(
        python_executable=tmp_path / ".venv/bin/python",
        dataset_name="princeton-nlp/SWE-bench_Lite",
        split="test",
        predictions_path=tmp_path / "predictions.jsonl",
        run_id="qwen-smoke",
        instance_ids=("sympy__sympy-20590",),
        max_workers=1,
        arm64_macos=True,
    )

    assert command[:3] == [
        str(tmp_path / ".venv/bin/python"),
        "-m",
        "swebench.harness.run_evaluation",
    ]
    assert "--max_workers" in command
    assert command[command.index("--max_workers") + 1] == "1"
    assert command[command.index("--namespace") + 1] == ""
    assert ";" not in " ".join(command)


def test_readiness_requires_harness_docker_and_recommended_disk():
    blocked = evaluate_readiness(
        has_swebench=False,
        has_datasets=False,
        has_docker_cli=False,
        has_docker_daemon=False,
        free_disk_gib=23,
        recommended_free_disk_gib=120,
        arm64=True,
    )
    assert blocked["ready"] is False
    assert set(blocked["blockers"]) == {
        "swebench_missing",
        "datasets_missing",
        "docker_missing",
        "insufficient_disk",
    }
    assert blocked["arm64_experimental"] is True


def test_environment_probe_rejects_installed_cli_when_docker_daemon_is_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("swe_bench_adapter.shutil.which", lambda name: "/bin/docker")
    monkeypatch.setattr(
        "swe_bench_adapter.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )

    readiness = probe_environment(tmp_path, recommended_free_disk_gib=0)

    assert readiness["ready"] is False
    assert "docker_daemon_unavailable" in readiness["blockers"]
