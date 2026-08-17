"""Resumable public source prefetch for the frozen Round 1 selection."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_json, sha256_json
from .round1_selection import _verified

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(
    args: list[str], *, cwd: Path, timeout: int = 1800
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _has_commit(repo: Path, commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=repo,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def prefetch_round1_sources(
    *,
    selection_path: Path,
    cache_root: Path,
    evidence_root: Path,
    max_repos: int | None = None,
    repository_url_template: str = "https://github.com/{repo}.git",
) -> dict[str, Any]:
    """Clone/fetch every required public base commit with per-repo receipts."""

    if max_repos is not None and (type(max_repos) is not int or max_repos < 1):
        raise ContractError("Round 1 source max_repos must be a positive integer")
    if repository_url_template.count("{repo}") != 1:
        raise ContractError("Round 1 repository URL template is invalid")
    selection_raw, selection = _verified(selection_path, label="Round 1 selection")
    tasks = selection.get("tasks")
    if (
        selection.get("status") != "frozen"
        or type(selection.get("task_count")) is not int
        or selection["task_count"] < 1
        or not isinstance(tasks, list)
        or selection["task_count"] != len(tasks)
    ):
        raise ContractError("Round 1 selection is not frozen")
    required: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        if not isinstance(task, dict):
            raise ContractError("Round 1 selected source task is invalid")
        repo = task.get("repo")
        commit = task.get("base_commit")
        if (
            not isinstance(repo, str)
            or _REPO.fullmatch(repo) is None
            or not isinstance(commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        ):
            raise ContractError("Round 1 source identity is invalid")
        required[repo].add(commit)

    cache = cache_root.resolve()
    evidence = evidence_root.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    completed_rows: list[dict[str, Any]] = []
    executed = 0
    for repo in sorted(required):
        org, name = repo.split("/", 1)
        target = cache / org / name
        receipt_dir = evidence / "repos" / org / name
        receipt_path = receipt_dir / "SOURCE-RECEIPT.json"
        commits = sorted(required[repo])
        if receipt_path.is_file():
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
            content = {
                key: value
                for key, value in existing.items()
                if key != "evidence_sha256"
            }
            if (
                existing.get("evidence_sha256") != sha256_json(content)
                or existing.get("repo") != repo
                or existing.get("required_commits") != commits
                or not (target / ".git").is_dir()
                or any(not _has_commit(target, commit) for commit in commits)
            ):
                raise ContractError("Round 1 source receipt does not match cache")
            completed_rows.append(existing)
            continue
        if max_repos is not None and executed >= max_repos:
            continue
        if target.exists() and not (target / ".git").is_dir():
            raise ContractError("Round 1 source cache is incomplete")
        receipt_dir.mkdir(parents=True, exist_ok=True)
        commands: list[dict[str, Any]] = []
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            url = repository_url_template.format(repo=repo)
            args = [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                url,
                str(target),
            ]
            result = _run(args, cwd=target.parent)
            stdout = receipt_dir / "clone.stdout.log"
            stderr = receipt_dir / "clone.stderr.log"
            stdout.write_text(result.stdout, encoding="utf-8")
            stderr.write_text(result.stderr, encoding="utf-8")
            commands.append(
                {
                    "kind": "clone",
                    "returncode": result.returncode,
                    "stdout_sha256": _sha256_file(stdout),
                    "stderr_sha256": _sha256_file(stderr),
                }
            )
            if result.returncode != 0 or not (target / ".git").is_dir():
                raise ContractError(f"Round 1 source clone failed: {repo}")
        for index, commit in enumerate(commits):
            if _has_commit(target, commit):
                continue
            args = ["git", "fetch", "--filter=blob:none", "origin", commit]
            result = _run(args, cwd=target)
            stdout = receipt_dir / f"fetch-{index:03d}.stdout.log"
            stderr = receipt_dir / f"fetch-{index:03d}.stderr.log"
            stdout.write_text(result.stdout, encoding="utf-8")
            stderr.write_text(result.stderr, encoding="utf-8")
            commands.append(
                {
                    "kind": "fetch",
                    "commit": commit,
                    "returncode": result.returncode,
                    "stdout_sha256": _sha256_file(stdout),
                    "stderr_sha256": _sha256_file(stderr),
                }
            )
            if result.returncode != 0 or not _has_commit(target, commit):
                raise ContractError(f"Round 1 source commit fetch failed: {repo}")
        content = {
            "schema_version": 1,
            "status": "ready",
            "repo": repo,
            "cache_path": str(target),
            "required_commits": commits,
            "verified_commits": [
                commit for commit in commits if _has_commit(target, commit)
            ],
            "commands": commands,
            "network_calls_performed": repository_url_template.startswith(
                ("http://", "https://")
            ),
        }
        report = {**content, "evidence_sha256": sha256_json(content)}
        receipt_path.write_text(canonical_json(report) + "\n", encoding="utf-8")
        completed_rows.append(report)
        executed += 1
    planned = len(required)
    content = {
        "schema_version": 1,
        "status": "complete" if len(completed_rows) == planned else "partial",
        "planned_repositories": planned,
        "completed_repositories": len(completed_rows),
        "required_commit_count": sum(len(commits) for commits in required.values()),
        "verified_commit_count": sum(
            len(row["verified_commits"]) for row in completed_rows
        ),
        "selection_file_sha256": hashlib.sha256(selection_raw).hexdigest(),
        "selection_evidence_sha256": selection["evidence_sha256"],
        "cache_root": str(cache),
        "repositories": [row["repo"] for row in completed_rows],
        "network_calls_performed": any(
            row["network_calls_performed"] for row in completed_rows
        ),
    }
    summary = {**content, "summary_sha256": sha256_json(content)}
    name = "SUMMARY.json" if content["status"] == "complete" else "PROGRESS.json"
    (evidence / name).write_text(canonical_json(summary) + "\n", encoding="utf-8")
    return summary
