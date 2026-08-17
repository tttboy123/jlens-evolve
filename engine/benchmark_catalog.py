"""Pinned real benchmark catalog and normalized task-contract builders."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from benchmark_adapters import BenchmarkContractError, BenchmarkTask


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_commit(value: str, *, field: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise BenchmarkContractError(f"{field} must be a pinned 40-character commit")


@dataclass(frozen=True)
class BenchmarkSource:
    adapter_id: str
    dataset_repo: str
    dataset_revision: str
    split: str
    task_family: str
    default_language: str
    license_id: str
    harness_repo: str
    harness_revision: str
    harness_kind: str

    def __post_init__(self) -> None:
        _require_commit(self.dataset_revision, field="dataset_revision")
        _require_commit(self.harness_revision, field="harness_revision")
        if self.harness_kind not in {"swebench", "multi_swe_bench", "harbor"}:
            raise BenchmarkContractError("unsupported benchmark harness kind")


SWE_HARNESS_REVISION = "f7bbbb2ccdf479001d6467c9e34af59e44a840f9"
MULTI_SWE_HARNESS_REVISION = "24f493f8a103e72312ded4f6b9c89f081d69cb09"
HARBOR_HARNESS_REVISION = "72bc40b1e58b47a9cc6e0f14c29aced3a9e53767"

PINNED_SOURCES = {
    "swe-bench-verified": BenchmarkSource(
        adapter_id="swe-bench-verified",
        dataset_repo="SWE-bench/SWE-bench_Verified",
        dataset_revision="91aa3ed51b709be6457e12d00300a6a596d4c6a3",
        split="test",
        task_family="repo_issue",
        default_language="python",
        license_id="dataset-card-unspecified",
        harness_repo="SWE-bench/SWE-bench",
        harness_revision=SWE_HARNESS_REVISION,
        harness_kind="swebench",
    ),
    "swe-bench-multilingual": BenchmarkSource(
        adapter_id="swe-bench-multilingual",
        dataset_repo="SWE-bench/SWE-bench_Multilingual",
        dataset_revision="e5c585e008e2cb5eecc7c64192d855c53279d788",
        split="test",
        task_family="repo_issue",
        default_language="polyglot",
        license_id="mit",
        harness_repo="SWE-bench/SWE-bench",
        harness_revision=SWE_HARNESS_REVISION,
        harness_kind="swebench",
    ),
    "multi-swe-bench-flash": BenchmarkSource(
        adapter_id="multi-swe-bench-flash",
        dataset_repo="ByteDance-Seed/Multi-SWE-bench-flash",
        dataset_revision="b0485dbebaf8a1317ebf140e80e6fc6c02d3502b",
        split="train",
        task_family="repo_pull_request",
        default_language="polyglot",
        license_id="other",
        harness_repo="multi-swe-bench/multi-swe-bench",
        harness_revision=MULTI_SWE_HARNESS_REVISION,
        harness_kind="multi_swe_bench",
    ),
    "terminal-bench-2": BenchmarkSource(
        adapter_id="terminal-bench-2",
        dataset_repo="harborframework/terminal-bench-2.0",
        dataset_revision="f2e8c75e23add71613117eecc9498f53bcd7e04e",
        split="test",
        task_family="terminal_environment",
        default_language="shell",
        license_id="apache-2.0",
        harness_repo="harbor-framework/harbor",
        harness_revision=HARBOR_HARNESS_REVISION,
        harness_kind="harbor",
    ),
}


def _hf_ref(source: BenchmarkSource, suffix: str) -> str:
    return (
        f"hf://datasets/{source.dataset_repo}@{source.dataset_revision}/"
        f"{suffix.lstrip('/')}"
    )


def _source_url(source: BenchmarkSource) -> str:
    return (
        f"https://huggingface.co/datasets/{source.dataset_repo}/tree/"
        f"{source.dataset_revision}"
    )


def normalize_swe_task(
    source: BenchmarkSource,
    row: dict[str, Any],
    *,
    row_index: int,
) -> BenchmarkTask:
    """Normalize a SWE-bench row while keeping gold patches outside the contract."""

    if source.harness_kind != "swebench":
        raise BenchmarkContractError("SWE row requires a SWE-bench source")
    repo = str(row.get("repo", "")).strip()
    instance_id = str(row.get("instance_id", "")).strip()
    base_commit = str(row.get("base_commit", "")).strip()
    if not repo or not instance_id or not base_commit:
        raise BenchmarkContractError("invalid SWE-bench row identity")
    issue_match = re.search(r"-(\d+)$", instance_id)
    overlap_keys = [f"instance://{instance_id.lower()}"]
    if issue_match:
        overlap_keys.append(f"https://github.com/{repo}/issues/{issue_match.group(1)}")
    return BenchmarkTask(
        benchmark_id=source.adapter_id,
        benchmark_revision=source.dataset_revision,
        instance_id=instance_id,
        task_family=source.task_family,
        language=str(row.get("language") or source.default_language).lower(),
        repo=repo,
        base_commit=base_commit,
        environment_ref=f"swebench://{repo}@{base_commit}",
        grader_ref=(
            f"git+https://github.com/{source.harness_repo}.git@"
            f"{source.harness_revision}#swebench.harness.run_evaluation"
        ),
        instruction_ref=_hf_ref(
            source,
            f"{source.split}#row={row_index}&field=problem_statement",
        ),
        source_url=_source_url(source),
        license_id=source.license_id,
        overlap_keys=tuple(overlap_keys),
        content_sha256=_sha256_json(row),
    )


def normalize_multi_swe_task(
    source: BenchmarkSource,
    row: dict[str, Any],
    *,
    row_index: int,
) -> BenchmarkTask:
    """Normalize a Multi-SWE-bench Flash pull-request instance."""

    if source.harness_kind != "multi_swe_bench":
        raise BenchmarkContractError("Multi-SWE row requires its native source")
    org = str(row.get("org", "")).strip()
    repo_name = str(row.get("repo", "")).strip()
    instance_id = str(row.get("instance_id", "")).strip()
    number = str(row.get("number", "")).strip()
    base = row.get("base") if isinstance(row.get("base"), dict) else {}
    base_commit = str(base.get("sha", "")).strip()
    if not all((org, repo_name, instance_id, number, base_commit)):
        raise BenchmarkContractError("invalid Multi-SWE-bench row identity")
    repo = f"{org}/{repo_name}"
    overlap_keys = [
        f"instance://{instance_id.lower()}",
        f"https://github.com/{repo}/pull/{number}",
    ]
    for issue in row.get("resolved_issues", []):
        if isinstance(issue, dict) and str(issue.get("number", "")).strip():
            overlap_keys.append(f"https://github.com/{repo}/issues/{issue['number']}")
    return BenchmarkTask(
        benchmark_id=source.adapter_id,
        benchmark_revision=source.dataset_revision,
        instance_id=instance_id,
        task_family=source.task_family,
        language=str(row.get("language") or source.default_language).lower(),
        repo=repo,
        base_commit=base_commit,
        environment_ref=f"multi-swe://{repo}@{base_commit}",
        grader_ref=(
            f"git+https://github.com/{source.harness_repo}.git@"
            f"{source.harness_revision}#multi_swe_bench.harness.run_evaluation"
        ),
        instruction_ref=_hf_ref(source, f"{source.split}#row={row_index}"),
        source_url=_source_url(source),
        license_id=source.license_id,
        overlap_keys=tuple(dict.fromkeys(overlap_keys)),
        content_sha256=_sha256_json(row),
    )


def normalize_terminal_tasks(
    source: BenchmarkSource,
    sibling_paths: list[str],
) -> tuple[BenchmarkTask, ...]:
    """Create Terminal-Bench contracts from a pinned repository file index."""

    if source.harness_kind != "harbor":
        raise BenchmarkContractError("terminal tasks require a Harbor source")
    by_task: dict[str, list[str]] = {}
    for path in sibling_paths:
        if "/" not in path:
            continue
        task_name = path.split("/", 1)[0]
        by_task.setdefault(task_name, []).append(path)
    tasks = []
    for task_name, paths in sorted(by_task.items()):
        path_set = set(paths)
        required = {
            f"{task_name}/instruction.md",
            f"{task_name}/task.toml",
            f"{task_name}/environment/Dockerfile",
        }
        if not required.issubset(path_set) or not any(
            path.startswith(f"{task_name}/tests/") for path in paths
        ):
            continue
        tasks.append(
            BenchmarkTask(
                benchmark_id=source.adapter_id,
                benchmark_revision=source.dataset_revision,
                instance_id=task_name,
                task_family=source.task_family,
                language=source.default_language,
                repo=source.dataset_repo,
                base_commit=source.dataset_revision,
                environment_ref=_hf_ref(source, f"{task_name}/environment"),
                grader_ref=_hf_ref(source, f"{task_name}/tests"),
                instruction_ref=_hf_ref(source, f"{task_name}/instruction.md"),
                source_url=_source_url(source),
                license_id=source.license_id,
                overlap_keys=(f"terminal-task://{task_name}",),
                content_sha256=_sha256_json(
                    {
                        "dataset_revision": source.dataset_revision,
                        "task_name": task_name,
                        "paths": sorted(paths),
                    }
                ),
            )
        )
    if not tasks:
        raise BenchmarkContractError("no executable Terminal-Bench tasks found")
    return tuple(tasks)


def build_execution_command(
    source: BenchmarkSource,
    *,
    predictions_path: str,
    run_id: str,
    instance_ids: tuple[str, ...],
    dataset_path: str | None = None,
    model: str | None = None,
    reasoning: str | None = None,
    arm: str | None = None,
    agent_program_sha256: str | None = None,
    baseline_contract_sha256: str | None = None,
) -> tuple[str, ...]:
    """Build a non-shell native-harness command for one frozen task batch."""

    if not run_id or not instance_ids:
        raise BenchmarkContractError("execution requires run_id and instance_ids")
    pinned_path = dataset_path or (
        f"/workspace/eval-datasets/{source.adapter_id}@{source.dataset_revision}"
    )
    if source.harness_kind == "swebench":
        if dataset_path is None:
            pinned_path += ".jsonl"
        return (
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            pinned_path,
            "--split",
            source.split,
            "--predictions_path",
            predictions_path,
            "--max_workers",
            "1",
            "--run_id",
            run_id,
            "--instance_ids",
            *instance_ids,
        )
    if source.harness_kind == "multi_swe_bench":
        return (
            "python",
            "-m",
            "multi_swe_bench.harness.run_evaluation",
            "--config",
            f"{predictions_path}.config.json",
        )
    if len(instance_ids) != 1:
        raise BenchmarkContractError("Harbor matched rounds execute one task at a time")
    if not model or not reasoning or arm not in {"baseline", "evolved"}:
        raise BenchmarkContractError(
            "Harbor execution requires a matched model, reasoning, and arm"
        )
    for field, value in {
        "agent_program_sha256": agent_program_sha256,
        "baseline_contract_sha256": baseline_contract_sha256,
    }.items():
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise BenchmarkContractError(f"Harbor {field} must be a SHA-256")
    return (
        "harbor",
        "run",
        "--path",
        pinned_path,
        "--task-name",
        instance_ids[0],
        "--agent",
        "evolve_jlens_harbor:FrozenCodexAgent",
        "--model",
        model,
        "--ak",
        f"reasoning_effort={reasoning}",
        "--ak",
        f"round_id={run_id.rsplit('-', 1)[0]}",
        "--ak",
        f"arm={arm}",
        "--ak",
        f"agent_program_sha256={agent_program_sha256}",
        "--ak",
        f"baseline_contract_sha256={baseline_contract_sha256}",
        "--ak",
        f"harness_revision={source.harness_revision}",
        "--job-name",
        run_id,
    )
