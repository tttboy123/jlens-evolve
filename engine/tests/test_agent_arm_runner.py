from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from agent_arm_runner import (
    AgentExecutionError,
    build_agent_prompt,
    build_codex_argv,
    build_matched_invocations,
    run_patch_arm,
)
from continuous_ab import BaselineContract


def _tree_hash(root: Path) -> str:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _baseline(agent_sha: str) -> BaselineContract:
    return BaselineContract(
        experiment_id="agent-runner-test",
        agent_program_sha256=agent_sha,
        model="gpt-test",
        reasoning="low",
        token_budget=4096,
        timeout_seconds=30,
        tools=("shell", "apply_patch"),
        retries=0,
        evaluator_epoch="native-v1",
    )


def _profile(path: Path, text: str) -> str:
    path.mkdir(parents=True)
    (path / "AGENTS.md").write_text(text, encoding="utf-8")
    return _tree_hash(path)


def _git_workspace(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    (path / "source.txt").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        cwd=path,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_matched_invocations_differ_only_by_agent_program_and_isolated_paths(
    tmp_path: Path,
):
    baseline_profile = tmp_path / "baseline-profile"
    evolved_profile = tmp_path / "evolved-profile"
    baseline_sha = _profile(baseline_profile, "Baseline policy.\n")
    evolved_sha = _profile(evolved_profile, "Evolved policy.\n")
    materialized = {
        "round_id": "round-000001",
        "task_uid": "d" * 64,
        "benchmark_id": "swe-bench-verified",
        "instance_id": "repo__repo-1",
        "repo": "example/repo",
        "base_commit": "deadbeef",
        "instruction": "Fix the issue.",
        "instruction_sha256": hashlib.sha256(b"Fix the issue.").hexdigest(),
    }

    pair = build_matched_invocations(
        baseline=_baseline(baseline_sha),
        materialized_task=materialized,
        baseline_profile=baseline_profile,
        evolved_profile=evolved_profile,
        evolved_agent_sha256=evolved_sha,
        baseline_workspace=tmp_path / "baseline-workspace",
        evolved_workspace=tmp_path / "evolved-workspace",
        evidence_root=tmp_path / "evidence",
    )

    assert (
        pair["baseline"].matched_contract_sha256
        == pair["evolved"].matched_contract_sha256
    )
    assert pair["baseline"].workspace != pair["evolved"].workspace
    assert pair["baseline"].agent_program_sha256 == baseline_sha
    assert pair["evolved"].agent_program_sha256 == evolved_sha
    assert "Baseline policy." in build_agent_prompt(
        materialized, baseline_profile, baseline_sha
    )
    argv = build_codex_argv(pair["baseline"], codex_executable=Path("/opt/codex"))
    assert argv[0] == "/opt/codex"
    assert "--ignore-user-config" in argv
    assert "--ephemeral" in argv
    assert "workspace-write" in argv
    assert "apps" in argv
    assert "apps._default.enabled=false" in argv
    assert "include_apps_instructions=false" in argv
    assert "mcp_servers={}" in argv
    assert "tool_output_token_limit=4096" in argv
    assert argv[-1] == "-"


def test_build_codex_argv_keeps_provider_config_when_required(monkeypatch, tmp_path):
    baseline_profile = tmp_path / "baseline-profile"
    evolved_profile = tmp_path / "evolved-profile"
    baseline_sha = _profile(baseline_profile, "Baseline policy.\n")
    evolved_sha = _profile(evolved_profile, "Evolved policy.\n")
    materialized = {
        "round_id": "round-000001",
        "task_uid": "d" * 64,
        "benchmark_id": "swe-bench-verified",
        "instance_id": "repo__repo-1",
        "repo": "example/repo",
        "base_commit": "deadbeef",
        "instruction": "Fix the issue.",
        "instruction_sha256": hashlib.sha256(b"Fix the issue.").hexdigest(),
    }
    pair = build_matched_invocations(
        baseline=_baseline(baseline_sha),
        materialized_task=materialized,
        baseline_profile=baseline_profile,
        evolved_profile=evolved_profile,
        evolved_agent_sha256=evolved_sha,
        baseline_workspace=tmp_path / "baseline-workspace",
        evolved_workspace=tmp_path / "evolved-workspace",
        evidence_root=tmp_path / "evidence",
    )
    monkeypatch.setenv("EVOLVE_CODEX_REQUIRE_PROVIDER_CONFIG", "1")
    argv = build_codex_argv(pair["baseline"], codex_executable=Path("/opt/codex"))
    assert "--ignore-user-config" not in argv
    assert 'model_provider="custom"' in argv


def test_build_codex_argv_bypasses_internal_sandbox_when_requested(
    monkeypatch, tmp_path
):
    baseline_profile = tmp_path / "baseline-profile"
    evolved_profile = tmp_path / "evolved-profile"
    baseline_sha = _profile(baseline_profile, "Baseline policy.\n")
    evolved_sha = _profile(evolved_profile, "Evolved policy.\n")
    materialized = {
        "round_id": "round-000001",
        "task_uid": "d" * 64,
        "benchmark_id": "swe-bench-verified",
        "instance_id": "repo__repo-1",
        "repo": "example/repo",
        "base_commit": "deadbeef",
        "instruction": "Fix the issue.",
        "instruction_sha256": hashlib.sha256(b"Fix the issue.").hexdigest(),
    }
    pair = build_matched_invocations(
        baseline=_baseline(baseline_sha),
        materialized_task=materialized,
        baseline_profile=baseline_profile,
        evolved_profile=evolved_profile,
        evolved_agent_sha256=evolved_sha,
        baseline_workspace=tmp_path / "baseline-workspace",
        evolved_workspace=tmp_path / "evolved-workspace",
        evidence_root=tmp_path / "evidence",
    )
    monkeypatch.setenv("EVOLVE_CODEX_NO_SANDBOX", "1")
    argv = build_codex_argv(pair["baseline"], codex_executable=Path("/opt/codex"))
    assert "--dangerously-bypass-approvals-and-sandbox" in argv
    assert "--sandbox" not in argv


def test_patch_arm_requires_authorization_and_freezes_failure_or_success(
    tmp_path: Path,
):
    profile = tmp_path / "profile"
    profile_sha = _profile(profile, "Policy.\n")
    workspace = tmp_path / "workspace"
    base_commit = _git_workspace(workspace)
    materialized = {
        "round_id": "round-000001",
        "task_uid": "d" * 64,
        "benchmark_id": "swe-bench-verified",
        "instance_id": "repo__repo-1",
        "repo": "example/repo",
        "base_commit": base_commit,
        "instruction": "Fix the issue.",
        "instruction_sha256": hashlib.sha256(b"Fix the issue.").hexdigest(),
    }
    invocation = build_matched_invocations(
        baseline=_baseline(profile_sha),
        materialized_task=materialized,
        baseline_profile=profile,
        evolved_profile=profile,
        evolved_agent_sha256=profile_sha,
        baseline_workspace=workspace,
        evolved_workspace=tmp_path / "unused-workspace",
        evidence_root=tmp_path / "evidence",
    )["baseline"]
    fake_codex = tmp_path / "fake-codex"
    fake_codex.write_text(
        "#!/bin/sh\n"
        "while [ $# -gt 0 ]; do\n"
        '  if [ "$1" = "-C" ]; then shift; work=$1; fi\n'
        "  shift\n"
        "done\n"
        'printf \'%s\\n\' \'{"type":"turn.completed","usage":{"input_tokens":120,"cached_input_tokens":20,"output_tokens":30}}\'\n'
        "printf '%s\\n' changed > \"$work/source.txt\"\n"
        "printf '%s\\n' added > \"$work/new-file.txt\"\n",
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)

    with pytest.raises(AgentExecutionError, match="authorization"):
        run_patch_arm(
            invocation,
            prompt="Fix the issue.",
            codex_executable=fake_codex,
            authorization={"status": "HUMAN_REQUIRED"},
        )

    receipt = run_patch_arm(
        invocation,
        prompt="Fix the issue.",
        codex_executable=fake_codex,
        authorization={
            "status": "authorized",
            "effective_caps": {"maximum_agent_calls": 20},
        },
    )

    assert receipt["returncode"] == 0
    assert receipt["prediction"]["frozen"] is True
    assert receipt["prediction"]["bytes"] > 0
    assert "new-file.txt" in Path(receipt["prediction"]["path"]).read_text()
    assert receipt["usage"] == {
        "input_tokens": 120,
        "cached_input_tokens": 20,
        "output_tokens": 30,
        "total_tokens": 150,
    }
    assert receipt["token_budget_exceeded"] is False


def test_patch_arm_accepts_verified_pinned_archive_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    profile = tmp_path / "profile"
    profile_sha = _profile(profile, "Policy.\n")
    workspace = tmp_path / "workspace"
    synthetic_commit = _git_workspace(workspace)
    base_commit = "a" * 40
    provenance = {
        "archive_sha256": "b" * 64,
        "base_commit": base_commit,
        "repo": "example/repo",
        "source": "github-codeload-pinned-archive",
        "synthetic_commit": synthetic_commit,
    }
    (workspace / ".git/evolve-source.json").write_text(
        json.dumps(provenance, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    materialized = {
        "round_id": "round-archive",
        "task_uid": "e" * 64,
        "benchmark_id": "swe-bench-verified",
        "instance_id": "repo__repo-archive",
        "repo": "example/repo",
        "base_commit": base_commit,
        "instruction": "Fix the issue.",
        "instruction_sha256": hashlib.sha256(b"Fix the issue.").hexdigest(),
    }
    invocation = build_matched_invocations(
        baseline=_baseline(profile_sha),
        materialized_task=materialized,
        baseline_profile=profile,
        evolved_profile=profile,
        evolved_agent_sha256=profile_sha,
        baseline_workspace=workspace,
        evolved_workspace=tmp_path / "unused-workspace",
        evidence_root=tmp_path / "evidence",
    )["baseline"]
    fake_codex = tmp_path / "fake-codex"
    fake_codex.write_text(
        "#!/bin/sh\n"
        "while [ $# -gt 0 ]; do\n"
        '  if [ "$1" = "-C" ]; then shift; work=$1; fi\n'
        "  shift\n"
        "done\n"
        "printf '%s\\n' changed > \"$work/source.txt\"\n",
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)
    proxy_url = "http://127.0.0.1:43128"
    monkeypatch.setenv("EVOLVE_CODEX_HTTPS_PROXY", proxy_url)

    receipt = run_patch_arm(
        invocation,
        prompt="Fix the issue.",
        codex_executable=fake_codex,
        authorization={
            "status": "authorized",
            "effective_caps": {"maximum_agent_calls": 20},
        },
    )

    assert receipt["workspace_source"] == provenance
    assert receipt["workspace_head_unchanged"] is True
    assert receipt["prediction"]["bytes"] > 0
    assert receipt["network_transport"] == {
        "loopback_proxy_configured": True,
        "proxy_sha256": hashlib.sha256(proxy_url.encode()).hexdigest(),
    }
