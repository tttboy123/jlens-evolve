"""Shared-mirror Git workspace materialization for isolated real Agent arms."""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any


class WorkspaceMaterializationError(RuntimeError):
    """Raised when an arm cannot be pinned to its exact benchmark commit."""


class UnsafeArchiveError(WorkspaceMaterializationError):
    """Raised when a source archive violates extraction safety constraints."""


def _run(
    args: list[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, check=False, timeout=600
    )
    if completed.returncode != 0:
        raise WorkspaceMaterializationError(
            f"Git command failed ({completed.returncode}): {' '.join(args[:3])}"
        )
    return completed


def _run_network(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    failures = []
    for _attempt in range(3):
        completed = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        if completed.returncode == 0:
            return completed
        failures.append(completed.stderr.strip())
    raise WorkspaceMaterializationError(
        "Git network fetch failed after three attempts: " + " | ".join(failures)
    )


def _download_github_archive(repo: str, commit: str) -> bytes:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise WorkspaceMaterializationError("GitHub repository identity is invalid")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise WorkspaceMaterializationError("GitHub archive commit must be a full SHA")
    url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
    except OSError as exc:
        raise WorkspaceMaterializationError(
            "pinned GitHub archive download failed"
        ) from exc
    if not payload:
        raise WorkspaceMaterializationError("pinned GitHub archive is empty")
    return payload


def _safe_component(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")[:48] or "item"
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"{stem}-{digest}"


class GitWorkspaceFactory:
    """Create one isolated checkout per task/arm while sharing repository objects."""

    def __init__(
        self,
        *,
        root: Path,
        remote_resolver: Callable[[str], str] | None = None,
        network_fetcher: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        archive_fetcher: Callable[[str, str], bytes] | None = None,
        prefer_archive_for_github: bool = True,
    ) -> None:
        self.root = root.resolve()
        self.mirrors = self.root / "mirrors"
        self.checkouts = self.root / "checkouts"
        self.remote_resolver = remote_resolver or (
            lambda repo: f"https://github.com/{repo}.git"
        )
        self.network_fetcher = network_fetcher or _run_network
        self.archive_fetcher = archive_fetcher or _download_github_archive
        self.prefer_archive_for_github = prefer_archive_for_github
        self.mirrors.mkdir(parents=True, exist_ok=True)
        self.checkouts.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _head(path: Path) -> str:
        return _run(["git", "rev-parse", "HEAD"], cwd=path).stdout.strip()

    @staticmethod
    def _source_record_path(mirror: Path, base_commit: str) -> Path:
        return mirror / "evolve-sources" / f"{base_commit}.json"

    @classmethod
    def _read_source_record(
        cls, mirror: Path, base_commit: str
    ) -> dict[str, Any] | None:
        path = cls._source_record_path(mirror, base_commit)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("base_commit") != base_commit:
            raise WorkspaceMaterializationError("archive source record is inconsistent")
        return record

    @staticmethod
    def _validate_archive_members(members: list[tarfile.TarInfo]) -> str:
        roots = set()
        for member in members:
            parts = Path(member.name).parts
            if (
                not parts
                or member.name.startswith("/")
                or ".." in parts
                or ".git" in parts
                or member.isdev()
                or member.isfifo()
            ):
                raise UnsafeArchiveError(f"unsafe archive member: {member.name}")
            roots.add(parts[0])
        if len(roots) != 1:
            raise WorkspaceMaterializationError(
                "pinned archive must contain exactly one root directory"
            )
        return roots.pop()

    def _install_archive_snapshot(
        self, *, mirror: Path, repo: str, base_commit: str
    ) -> dict[str, Any]:
        payload = self.archive_fetcher(repo, base_commit)
        archive_sha256 = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            temporary_root = Path(temporary)
            try:
                with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                    members = archive.getmembers()
                    archive_root_name = self._validate_archive_members(members)
                    archive.extractall(temporary_root, members=members, filter="data")
            except (tarfile.TarError, OSError) as exc:
                raise WorkspaceMaterializationError(
                    "pinned GitHub archive could not be safely extracted"
                ) from exc
            snapshot = temporary_root / archive_root_name
            if not snapshot.is_dir():
                raise WorkspaceMaterializationError(
                    "pinned archive root is not a directory"
                )
            _run(["git", "init", "-q"], cwd=snapshot)
            _run(["git", "add", "-f", "--all"], cwd=snapshot)
            _run(
                [
                    "git",
                    "-c",
                    "user.name=Evolve Snapshot",
                    "-c",
                    "user.email=evolve-snapshot@example.invalid",
                    "commit",
                    "-qm",
                    f"frozen source snapshot {repo}@{base_commit}",
                ],
                cwd=snapshot,
            )
            synthetic_commit = self._head(snapshot)
            _run(
                ["git", "fetch", "--no-tags", str(snapshot), synthetic_commit],
                cwd=mirror,
            )
            _run(
                [
                    "git",
                    "update-ref",
                    f"refs/evolve/archive/{base_commit}",
                    synthetic_commit,
                ],
                cwd=mirror,
            )
        record = {
            "archive_sha256": archive_sha256,
            "base_commit": base_commit,
            "repo": repo,
            "source": "github-codeload-pinned-archive",
            "synthetic_commit": synthetic_commit,
        }
        record_path = self._source_record_path(mirror, base_commit)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return record

    def _mirror(self, repo: str, base_commit: str) -> tuple[Path, str, dict[str, Any]]:
        mirror = self.mirrors / f"{_safe_component(repo)}.git"
        remote = self.remote_resolver(repo)
        if not mirror.exists():
            _run(["git", "init", "--bare", str(mirror)])
            _run(["git", "remote", "add", "origin", remote], cwd=mirror)
        elif not mirror.is_dir():
            raise WorkspaceMaterializationError(
                "repository mirror path is not a directory"
            )
        configured_remote = _run(
            ["git", "remote", "get-url", "origin"], cwd=mirror
        ).stdout.strip()
        if configured_remote != remote:
            raise WorkspaceMaterializationError("repository mirror remote is immutable")
        archived = self._read_source_record(mirror, base_commit)
        if archived is not None:
            return mirror, str(archived["synthetic_commit"]), archived
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{base_commit}^{{commit}}"],
            cwd=mirror,
            capture_output=True,
            text=True,
            check=False,
        )
        if exists.returncode != 0:
            is_github = remote.startswith("https://github.com/")
            archive_error: Exception | None = None
            if is_github and self.prefer_archive_for_github:
                try:
                    archived = self._install_archive_snapshot(
                        mirror=mirror, repo=repo, base_commit=base_commit
                    )
                    return mirror, str(archived["synthetic_commit"]), archived
                except UnsafeArchiveError:
                    raise
                except (WorkspaceMaterializationError, OSError) as exc:
                    archive_error = exc
            try:
                self.network_fetcher(
                    [
                        "git",
                        "-c",
                        "http.version=HTTP/1.1",
                        "fetch",
                        "--no-tags",
                        "--depth=1",
                        "origin",
                        base_commit,
                    ],
                    cwd=mirror,
                )
            except (WorkspaceMaterializationError, subprocess.TimeoutExpired) as exc:
                if is_github and not self.prefer_archive_for_github:
                    archived = self._install_archive_snapshot(
                        mirror=mirror, repo=repo, base_commit=base_commit
                    )
                    return mirror, str(archived["synthetic_commit"]), archived
                if archive_error is not None:
                    raise WorkspaceMaterializationError(
                        "both pinned archive and Git fetch transports failed"
                    ) from exc
                raise
        provenance = {
            "archive_sha256": None,
            "base_commit": base_commit,
            "repo": repo,
            "source": "git-fetch-pinned-commit",
            "synthetic_commit": None,
        }
        return mirror, base_commit, provenance

    def __call__(self, materialized: dict[str, Any], arm: Any) -> Path:
        repo = materialized.get("repo")
        task_uid = materialized.get("task_uid")
        base_commit = materialized.get("base_commit")
        arm_name = getattr(arm, "name", None)
        if not all(
            isinstance(value, str) and value.strip()
            for value in (repo, task_uid, base_commit, arm_name)
        ):
            raise WorkspaceMaterializationError("workspace identity is incomplete")
        mirror, checkout_commit, provenance = self._mirror(repo, base_commit)
        destination = (
            self.checkouts / _safe_component(task_uid) / _safe_component(arm_name)
        )
        if destination.exists():
            record_path = destination / ".git" / "evolve-source.json"
            record = (
                json.loads(record_path.read_text(encoding="utf-8"))
                if record_path.exists()
                else None
            )
            if (
                not destination.is_dir()
                or self._head(destination) != checkout_commit
                or (record is not None and record.get("base_commit") != base_commit)
            ):
                raise WorkspaceMaterializationError(
                    "persisted arm workspace differs from frozen base commit"
                )
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--shared", str(mirror), str(destination)])
        _run(["git", "fetch", "--no-tags", "origin", checkout_commit], cwd=destination)
        _run(["git", "checkout", "--detach", checkout_commit], cwd=destination)
        if self._head(destination) != checkout_commit:
            raise WorkspaceMaterializationError("arm workspace checkout is not pinned")
        (destination / ".git" / "evolve-source.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination
