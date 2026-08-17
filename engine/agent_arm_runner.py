"""Matched Codex arm contracts and fail-closed patch prediction execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from continuous_ab import BaselineContract

_ARM_NAME = re.compile(r"[a-z][a-z0-9-]{0,63}")


class AgentExecutionError(ValueError):
    """Raised before dispatch when an Agent arm crosses a frozen contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _profile_tree_hash(root: Path) -> str:
    root = root.resolve()
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise AgentExecutionError("AgentProgram profile cannot contain symlinks")
        relative = path.relative_to(root).as_posix()
        allowed = (
            relative == "AGENTS.md"
            or relative == ".codex/evolution-policy.json"
            or (relative.startswith(".codex/harness/") and relative.endswith(".py"))
            or (
                relative.startswith(".agents/skills/")
                and relative.endswith("/SKILL.md")
            )
        )
        if not allowed:
            raise AgentExecutionError(
                f"AgentProgram profile path is not allowed: {relative}"
            )
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    if not rows:
        raise AgentExecutionError("AgentProgram profile is empty")
    return _sha256_json(rows)


def profile_tree_hash(root: Path) -> str:
    """Return the authoritative hash for one allowed AgentProgram profile tree."""

    return _profile_tree_hash(root.resolve())


def build_agent_prompt(
    materialized_task: dict[str, Any],
    profile_root: Path,
    expected_agent_sha256: str,
) -> str:
    """Compose Prompt/Skill/Policy without copying profile files into the target repo."""

    profile_root = profile_root.resolve()
    if _profile_tree_hash(profile_root) != expected_agent_sha256:
        raise AgentExecutionError("AgentProgram profile tree hash mismatch")
    sections = []
    for path in sorted(item for item in profile_root.rglob("*") if item.is_file()):
        relative = path.relative_to(profile_root).as_posix()
        sections.append(f"### AgentProgram surface: {relative}\n{path.read_text()}")
    instruction = materialized_task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise AgentExecutionError("materialized task instruction is empty")
    return (
        "You are one arm of a frozen matched benchmark. Follow the project-local "
        "AgentProgram below, solve only the task, do not edit tests, and leave all "
        "source changes in the current Git workspace.\n\n"
        + "\n\n".join(sections)
        + "\n\n### Benchmark task\n"
        + instruction.strip()
        + "\n"
    )


@dataclass(frozen=True)
class AgentInvocationContract:
    round_id: str
    arm: str
    task_uid: str
    benchmark_id: str
    instance_id: str
    repo: str
    base_commit: str
    instruction_sha256: str
    agent_program_sha256: str
    profile_root: str
    baseline_contract_sha256: str
    model: str
    reasoning: str
    token_budget: int
    timeout_seconds: int
    tools: tuple[str, ...]
    retries: int
    evaluator_epoch: str
    workspace: str
    evidence_dir: str

    @property
    def matched_contract_sha256(self) -> str:
        payload = asdict(self)
        for key in (
            "arm",
            "agent_program_sha256",
            "profile_root",
            "workspace",
            "evidence_dir",
        ):
            payload.pop(key)
        payload["tools"] = list(self.tools)
        return _sha256_json(payload)


@dataclass(frozen=True)
class ArmProgram:
    """One isolated AgentProgram arm in a multi-candidate matched execution."""

    name: str
    agent_program_sha256: str
    profile_root: Path
    workspace: Path

    def __post_init__(self) -> None:
        if _ARM_NAME.fullmatch(self.name) is None:
            raise AgentExecutionError(f"invalid Agent arm name: {self.name}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.agent_program_sha256):
            raise AgentExecutionError("AgentProgram hash must be a lowercase SHA-256")


def build_multi_arm_invocations(
    *,
    baseline: BaselineContract,
    materialized_task: dict[str, Any],
    arms: tuple[ArmProgram, ...],
    evidence_root: Path,
) -> dict[str, AgentInvocationContract]:
    """Freeze N isolated arms against one identical native execution contract."""

    if baseline.retries != 0:
        raise AgentExecutionError("matched evolution requires retries=0")
    if len(arms) < 2:
        raise AgentExecutionError("multi-arm evolution requires at least two arms")
    if len({arm.name for arm in arms}) != len(arms):
        raise AgentExecutionError("Agent arm names must be unique")
    if len({arm.agent_program_sha256 for arm in arms}) != len(arms):
        raise AgentExecutionError("AgentProgram arm hashes must be unique")
    if arms[0].name != "original":
        raise AgentExecutionError("first multi-arm entry must be permanent original")
    if arms[0].agent_program_sha256 != baseline.agent_program_sha256:
        raise AgentExecutionError("permanent original differs from baseline contract")
    workspace_paths = [arm.workspace.resolve() for arm in arms]
    if len(set(workspace_paths)) != len(workspace_paths):
        raise AgentExecutionError("Agent arms require isolated workspaces")
    for arm in arms:
        if _profile_tree_hash(arm.profile_root) != arm.agent_program_sha256:
            raise AgentExecutionError(f"{arm.name} AgentProgram tree hash mismatch")
    common = {
        "round_id": materialized_task["round_id"],
        "task_uid": materialized_task["task_uid"],
        "benchmark_id": materialized_task["benchmark_id"],
        "instance_id": materialized_task["instance_id"],
        "repo": materialized_task["repo"],
        "base_commit": materialized_task["base_commit"],
        "instruction_sha256": materialized_task["instruction_sha256"],
        "baseline_contract_sha256": baseline.contract_sha256,
        "model": baseline.model,
        "reasoning": baseline.reasoning,
        "token_budget": baseline.token_budget,
        "timeout_seconds": baseline.timeout_seconds,
        "tools": baseline.tools,
        "retries": baseline.retries,
        "evaluator_epoch": baseline.evaluator_epoch,
    }
    invocations = {
        arm.name: AgentInvocationContract(
            **common,
            arm=arm.name,
            agent_program_sha256=arm.agent_program_sha256,
            profile_root=str(arm.profile_root.resolve()),
            workspace=str(arm.workspace.resolve()),
            evidence_dir=str((evidence_root / arm.name).resolve()),
        )
        for arm in arms
    }
    if (
        len({contract.matched_contract_sha256 for contract in invocations.values()})
        != 1
    ):
        raise AgentExecutionError("Agent arms are not matched")
    return invocations


def build_matched_invocations(
    *,
    baseline: BaselineContract,
    materialized_task: dict[str, Any],
    baseline_profile: Path,
    evolved_profile: Path,
    evolved_agent_sha256: str,
    baseline_workspace: Path,
    evolved_workspace: Path,
    evidence_root: Path,
) -> dict[str, AgentInvocationContract]:
    """Freeze the same execution surface for both arms, except AgentProgram."""

    if baseline.retries != 0:
        raise AgentExecutionError("matched pilot requires retries=0")
    if _profile_tree_hash(baseline_profile) != baseline.agent_program_sha256:
        raise AgentExecutionError("baseline AgentProgram tree hash mismatch")
    if _profile_tree_hash(evolved_profile) != evolved_agent_sha256:
        raise AgentExecutionError("evolved AgentProgram tree hash mismatch")
    common = {
        "round_id": materialized_task["round_id"],
        "task_uid": materialized_task["task_uid"],
        "benchmark_id": materialized_task["benchmark_id"],
        "instance_id": materialized_task["instance_id"],
        "repo": materialized_task["repo"],
        "base_commit": materialized_task["base_commit"],
        "instruction_sha256": materialized_task["instruction_sha256"],
        "baseline_contract_sha256": baseline.contract_sha256,
        "model": baseline.model,
        "reasoning": baseline.reasoning,
        "token_budget": baseline.token_budget,
        "timeout_seconds": baseline.timeout_seconds,
        "tools": baseline.tools,
        "retries": baseline.retries,
        "evaluator_epoch": baseline.evaluator_epoch,
    }
    pair = {
        "baseline": AgentInvocationContract(
            **common,
            arm="baseline",
            agent_program_sha256=baseline.agent_program_sha256,
            profile_root=str(baseline_profile.resolve()),
            workspace=str(baseline_workspace.resolve()),
            evidence_dir=str((evidence_root / "baseline").resolve()),
        ),
        "evolved": AgentInvocationContract(
            **common,
            arm="evolved",
            agent_program_sha256=evolved_agent_sha256,
            profile_root=str(evolved_profile.resolve()),
            workspace=str(evolved_workspace.resolve()),
            evidence_dir=str((evidence_root / "evolved").resolve()),
        ),
    }
    if (
        pair["baseline"].matched_contract_sha256
        != pair["evolved"].matched_contract_sha256
    ):
        raise AgentExecutionError("Agent arms are not matched")
    return pair


def build_codex_argv(
    contract: AgentInvocationContract, *, codex_executable: Path
) -> tuple[str, ...]:
    evidence_dir = Path(contract.evidence_dir)
    argv = [
        str(codex_executable),
        "exec",
        "--json",
        "--ephemeral",
    ]
    # A custom model provider (e.g. DeepSeek) configured in the instance
    # ~/.codex/config.toml must survive the hermetic flags; the caller sets
    # EVOLVE_CODEX_REQUIRE_PROVIDER_CONFIG=1 to keep the provider config.
    if os.environ.get("EVOLVE_CODEX_REQUIRE_PROVIDER_CONFIG") != "1":
        argv.append("--ignore-user-config")
    sandbox_args = (
        ["--dangerously-bypass-approvals-and-sandbox"]
        if os.environ.get("EVOLVE_CODEX_NO_SANDBOX") == "1"
        else ["--sandbox", "workspace-write"]
    )
    argv.extend(
        [
            "--ignore-rules",
            "--disable",
            "apps",
            "--disable",
            "enable_mcp_apps",
            "--disable",
            "plugins",
            "--disable",
            "remote_plugin",
            "--disable",
            "browser_use",
            "--disable",
            "computer_use",
            "--disable",
            "multi_agent",
            "--disable",
            "memories",
            "--skip-git-repo-check",
            "--model",
            contract.model,
            "-c",
            'model_provider="custom"',
            "-c",
            f'model_reasoning_effort="{contract.reasoning}"',
            "-c",
            'web_search="disabled"',
            "-c",
            "apps._default.enabled=false",
            "-c",
            "include_apps_instructions=false",
            "-c",
            "mcp_servers={}",
            "-c",
            "tool_output_token_limit=4096",
            *sandbox_args,
            "-C",
            contract.workspace,
            "--output-last-message",
            str(evidence_dir / "last-message.txt"),
            "-",
        ]
    )
    return tuple(argv)


def _usage_from_jsonl(text: str) -> dict[str, int]:
    usage: dict[str, int] = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
    }
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        candidate = event.get("usage") if isinstance(event, dict) else None
        if not isinstance(candidate, dict):
            continue
        for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
            value = candidate.get(key)
            if isinstance(value, int) and value >= 0:
                usage[key] = max(usage[key], value)
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def _git_output(workspace: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )


def _stage_intent_to_add_untracked(workspace: Path) -> tuple[str, ...]:
    """Make untracked files visible to git diff without adding their contents."""

    untracked = _git_output(
        workspace, ["ls-files", "--others", "--exclude-standard", "-z"]
    )
    if untracked.returncode != 0:
        raise AgentExecutionError("cannot enumerate untracked Agent output")
    paths = tuple(path for path in untracked.stdout.split("\0") if path)
    for path in paths:
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise AgentExecutionError("Agent created an unsafe untracked path")
    if paths:
        staged = _git_output(workspace, ["add", "--intent-to-add", "--", *paths])
        if staged.returncode != 0:
            raise AgentExecutionError("cannot freeze untracked Agent output")
    return paths


def _frozen_workspace_source(
    contract: AgentInvocationContract, workspace: Path, actual_head: str
) -> tuple[str, dict[str, Any], Path | None, str | None]:
    record_path = workspace / ".git" / "evolve-source.json"
    if not record_path.exists():
        if actual_head != contract.base_commit:
            raise AgentExecutionError("workspace is not at the frozen base commit")
        provenance = {
            "archive_sha256": None,
            "base_commit": contract.base_commit,
            "repo": contract.repo,
            "source": "git-fetch-pinned-commit",
            "synthetic_commit": None,
        }
        return contract.base_commit, provenance, None, None
    try:
        provenance = json.loads(record_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise AgentExecutionError("workspace source record is unreadable") from exc
    if not isinstance(provenance, dict):
        raise AgentExecutionError("workspace source record must be an object")
    if provenance.get("base_commit") != contract.base_commit:
        raise AgentExecutionError("workspace source record base commit mismatch")
    if provenance.get("repo") != contract.repo:
        raise AgentExecutionError("workspace source record repository mismatch")
    source = provenance.get("source")
    if source == "github-codeload-pinned-archive":
        archive_sha256 = provenance.get("archive_sha256")
        synthetic_commit = provenance.get("synthetic_commit")
        if (
            not isinstance(archive_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", archive_sha256) is None
        ):
            raise AgentExecutionError("workspace archive hash is invalid")
        if (
            not isinstance(synthetic_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", synthetic_commit) is None
        ):
            raise AgentExecutionError("workspace synthetic commit is invalid")
        expected_head = synthetic_commit
    elif source == "git-fetch-pinned-commit":
        if (
            provenance.get("archive_sha256") is not None
            or provenance.get("synthetic_commit") is not None
        ):
            raise AgentExecutionError("Git workspace source record is invalid")
        expected_head = contract.base_commit
    else:
        raise AgentExecutionError("workspace source transport is not allowed")
    if actual_head != expected_head:
        raise AgentExecutionError("workspace is not at its frozen source snapshot")
    return expected_head, provenance, record_path, _sha256_file(record_path)


def _codex_subprocess_environment() -> tuple[dict[str, str], dict[str, Any]]:
    environment = dict(os.environ)
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        environment.pop(key, None)
    proxy = os.environ.get("EVOLVE_CODEX_HTTPS_PROXY")
    if proxy is None:
        return environment, {
            "loopback_proxy_configured": False,
            "proxy_sha256": None,
        }
    if re.fullmatch(r"http://127\.0\.0\.1:[1-9][0-9]{3,4}", proxy) is None:
        raise AgentExecutionError("Codex transport proxy must be loopback HTTP")
    port = int(proxy.rsplit(":", 1)[1])
    if port > 65535:
        raise AgentExecutionError("Codex transport proxy port is invalid")
    environment["HTTP_PROXY"] = proxy
    environment["HTTPS_PROXY"] = proxy
    environment["NO_PROXY"] = "127.0.0.1,localhost"
    return environment, {
        "loopback_proxy_configured": True,
        "proxy_sha256": hashlib.sha256(proxy.encode()).hexdigest(),
    }


def codex_subprocess_environment() -> tuple[dict[str, str], dict[str, Any]]:
    """Return the frozen network environment shared by Agent and proposer calls."""

    return _codex_subprocess_environment()


def validate_patch_arm_pre_dispatch(
    contract: AgentInvocationContract,
    *,
    prompt: str,
    authorization: dict[str, Any],
) -> tuple[Path, str, dict[str, Any], Path | None, str | None]:
    """Validate every local invariant before a real-call reservation is created."""

    if authorization.get("status") != "authorized" or not authorization.get(
        "effective_caps"
    ):
        raise AgentExecutionError("explicit pilot authorization is required")
    if authorization["effective_caps"].get("maximum_agent_calls", 0) < 1:
        raise AgentExecutionError("authorization has no Agent call capacity")
    if contract.retries != 0:
        raise AgentExecutionError("run_patch_arm does not permit retries")
    workspace = Path(contract.workspace).resolve()
    head = _git_output(workspace, ["rev-parse", "HEAD"])
    if head.returncode != 0:
        raise AgentExecutionError("workspace is not at the frozen base commit")
    frozen_head, workspace_source, source_record_path, source_record_sha256 = (
        _frozen_workspace_source(contract, workspace, head.stdout.strip())
    )
    if not prompt.strip():
        raise AgentExecutionError("Agent prompt is empty")
    _codex_subprocess_environment()
    return (
        workspace,
        frozen_head,
        workspace_source,
        source_record_path,
        source_record_sha256,
    )


def run_patch_arm(
    contract: AgentInvocationContract,
    *,
    prompt: str,
    codex_executable: Path,
    authorization: dict[str, Any],
) -> dict[str, Any]:
    """Execute one authorized arm once and freeze even an empty/failing patch."""

    (
        workspace,
        frozen_head,
        workspace_source,
        source_record_path,
        source_record_sha256,
    ) = validate_patch_arm_pre_dispatch(
        contract,
        prompt=prompt,
        authorization=authorization,
    )
    evidence_dir = Path(contract.evidence_dir).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    argv = build_codex_argv(contract, codex_executable=codex_executable)
    subprocess_environment, network_transport = _codex_subprocess_environment()
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(argv),
            input=prompt,
            capture_output=True,
            text=True,
            timeout=contract.timeout_seconds,
            check=False,
            env=subprocess_environment,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as error:
        returncode = 124
        stdout = error.stdout if isinstance(error.stdout, str) else ""
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        timed_out = True
    elapsed = time.monotonic() - started
    stdout_path = evidence_dir / "events.jsonl"
    stderr_path = evidence_dir / "stderr.txt"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    untracked_paths = _stage_intent_to_add_untracked(workspace)
    post_head = _git_output(workspace, ["rev-parse", "HEAD"])
    if post_head.returncode != 0:
        raise AgentExecutionError(
            "workspace Git history disappeared after Agent execution"
        )
    current_head = post_head.stdout.strip()
    if source_record_path is not None and (
        not source_record_path.exists()
        or _sha256_file(source_record_path) != source_record_sha256
    ):
        raise AgentExecutionError("workspace source record changed during execution")
    diff = _git_output(
        workspace,
        ["diff", "--binary", "--no-ext-diff", frozen_head, "--"],
    )
    if diff.returncode != 0:
        raise AgentExecutionError("git diff failed after Agent execution")
    prediction_path = evidence_dir / "prediction.patch"
    prediction_path.write_text(diff.stdout, encoding="utf-8")
    usage = _usage_from_jsonl(stdout)
    receipt = {
        "schema_version": "1.0",
        "round_id": contract.round_id,
        "arm": contract.arm,
        "task_uid": contract.task_uid,
        "benchmark_id": contract.benchmark_id,
        "instance_id": contract.instance_id,
        "agent_program_sha256": contract.agent_program_sha256,
        "baseline_contract_sha256": contract.baseline_contract_sha256,
        "matched_contract_sha256": contract.matched_contract_sha256,
        "evaluator_epoch": contract.evaluator_epoch,
        "argv": list(argv),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 6),
        "execution_success": returncode == 0,
        "workspace_head_unchanged": current_head == frozen_head,
        "workspace_source": workspace_source,
        "workspace_source_record_sha256": source_record_sha256,
        "network_transport": network_transport,
        "untracked_paths_frozen": list(untracked_paths),
        "usage": usage,
        "token_budget": contract.token_budget,
        "token_budget_exceeded": usage["total_tokens"] > contract.token_budget,
        "prediction": {
            "path": str(prediction_path),
            "bytes": prediction_path.stat().st_size,
            "sha256": _sha256_file(prediction_path),
            "frozen": True,
        },
        "raw_events": {
            "path": str(stdout_path),
            "bytes": stdout_path.stat().st_size,
            "sha256": _sha256_file(stdout_path),
        },
        "stderr": {
            "path": str(stderr_path),
            "bytes": stderr_path.stat().st_size,
            "sha256": _sha256_file(stderr_path),
        },
    }
    receipt["integrity_sha256"] = _sha256_json(receipt)
    receipt_path = evidence_dir / "agent-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return receipt
