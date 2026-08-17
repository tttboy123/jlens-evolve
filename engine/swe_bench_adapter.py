"""Fail-closed adapter for SWE-bench prediction files and official harness commands."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

PREDICTION_FIELDS = frozenset({"instance_id", "model_name_or_path", "model_patch"})


def _patch_paths(patch: str) -> tuple[str, ...]:
    paths: list[str] = []
    for line in patch.splitlines():
        match = re.fullmatch(r"diff --git a/(.+) b/(.+)", line)
        if not match:
            continue
        old_path, new_path = match.groups()
        if old_path != new_path:
            raise ValueError("renamed paths are not allowed by the local preflight")
        candidate = PurePosixPath(old_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("unsafe patch path")
        paths.append(old_path)
    return tuple(paths)


def validate_prediction(value: dict[str, Any]) -> dict[str, str]:
    """Validate the official three-field prediction shape plus local anti-poisoning."""

    if set(value) != PREDICTION_FIELDS:
        raise ValueError(
            "SWE-bench prediction must contain exactly the three official fields"
        )
    prediction: dict[str, str] = {}
    for key in sorted(PREDICTION_FIELDS):
        item = value.get(key)
        if not isinstance(item, str) or (key != "model_patch" and not item.strip()):
            raise ValueError(f"prediction field must be a non-empty string: {key}")
        prediction[key] = item
    paths = _patch_paths(prediction["model_patch"])
    if prediction["model_patch"].strip() and not paths:
        raise ValueError("model_patch must contain at least one git diff path")
    for path in paths:
        parts = PurePosixPath(path).parts
        if "tests" in parts or PurePosixPath(path).name.startswith("test_"):
            raise ValueError(f"model patch touches test path: {path}")
    return {
        key: prediction[key]
        for key in ("instance_id", "model_name_or_path", "model_patch")
    }


def write_predictions(predictions: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Write validated JSONL consumable by the official SWE-bench harness."""

    rows = [validate_prediction(prediction) for prediction in predictions]
    if not rows:
        raise ValueError("at least one prediction is required")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_harness_command(
    *,
    python_executable: Path,
    dataset_name: str,
    split: str,
    predictions_path: Path,
    run_id: str,
    instance_ids: tuple[str, ...] = (),
    max_workers: int = 1,
    arm64_macos: bool = False,
) -> list[str]:
    """Build, but do not execute, the official non-shell harness command."""

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if not run_id or any(character.isspace() for character in run_id):
        raise ValueError("run_id must be a non-empty token")
    command = [
        str(python_executable),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
    ]
    if instance_ids:
        command.extend(["--instance_ids", *instance_ids])
    if arm64_macos:
        command.extend(["--namespace", ""])
    return command


def evaluate_readiness(
    *,
    has_swebench: bool,
    has_datasets: bool,
    has_docker_cli: bool,
    has_docker_daemon: bool,
    free_disk_gib: float,
    recommended_free_disk_gib: float,
    arm64: bool,
) -> dict[str, Any]:
    """Evaluate external runtime prerequisites without installing or mutating them."""

    blockers = []
    if not has_swebench:
        blockers.append("swebench_missing")
    if not has_datasets:
        blockers.append("datasets_missing")
    if not has_docker_cli:
        blockers.append("docker_missing")
    elif not has_docker_daemon:
        blockers.append("docker_daemon_unavailable")
    if free_disk_gib < recommended_free_disk_gib:
        blockers.append("insufficient_disk")
    return {
        "ready": not blockers,
        "status": "ready" if not blockers else "adapter_ready_runtime_blocked",
        "blockers": blockers,
        "free_disk_gib": round(free_disk_gib, 3),
        "recommended_free_disk_gib": recommended_free_disk_gib,
        "arm64_experimental": arm64,
    }


def probe_environment(
    path: Path, *, recommended_free_disk_gib: float = 120
) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    docker_executable = shutil.which("docker")
    docker_daemon_ready = False
    if docker_executable:
        try:
            docker_daemon_ready = (
                subprocess.run(
                    [docker_executable, "info", "--format", "{{.ServerVersion}}"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                ).returncode
                == 0
            )
        except (OSError, subprocess.TimeoutExpired):
            docker_daemon_ready = False
    return evaluate_readiness(
        has_swebench=importlib.util.find_spec("swebench") is not None,
        has_datasets=importlib.util.find_spec("datasets") is not None,
        has_docker_cli=docker_executable is not None,
        has_docker_daemon=docker_daemon_ready,
        free_disk_gib=usage.free / (1024**3),
        recommended_free_disk_gib=recommended_free_disk_gib,
        arm64=platform.machine().lower() in {"arm64", "aarch64"},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=Path.cwd())
    parser.add_argument("--recommended-free-disk-gib", type=float, default=120)
    args = parser.parse_args()
    print(
        json.dumps(
            probe_environment(
                args.probe,
                recommended_free_disk_gib=args.recommended_free_disk_gib,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
