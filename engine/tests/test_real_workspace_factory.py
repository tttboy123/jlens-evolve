from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from real_workspace_factory import GitWorkspaceFactory, WorkspaceMaterializationError


def _source_repo(path: Path) -> tuple[Path, str]:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    (path / "source.py").write_text("value = 1\n", encoding="utf-8")
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
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return path, commit


def test_workspace_factory_reuses_mirror_but_isolates_task_arms(tmp_path: Path):
    source, commit = _source_repo(tmp_path / "source")
    factory = GitWorkspaceFactory(
        root=tmp_path / "workspaces",
        remote_resolver=lambda _repo: str(source),
    )
    task = {
        "task_uid": "task-001",
        "repo": "example/repo",
        "base_commit": commit,
    }
    original = factory(task, SimpleNamespace(name="original"))
    parent = factory(task, SimpleNamespace(name="parent"))

    assert original != parent
    assert (original / "source.py").read_text() == "value = 1\n"
    assert (parent / "source.py").read_text() == "value = 1\n"
    (original / "source.py").write_text("value = 2\n", encoding="utf-8")
    assert (parent / "source.py").read_text() == "value = 1\n"
    assert (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == commit
    )
    mirrors = list((tmp_path / "workspaces/mirrors").glob("*.git"))
    assert len(mirrors) == 1
    advertised_refs = subprocess.run(
        ["git", "show-ref"],
        cwd=mirrors[0],
        capture_output=True,
        text=True,
        check=False,
    )
    assert advertised_refs.stdout == ""


def test_workspace_factory_reopens_clean_existing_workspace(tmp_path: Path):
    source, commit = _source_repo(tmp_path / "source")
    factory = GitWorkspaceFactory(
        root=tmp_path / "workspaces",
        remote_resolver=lambda _repo: str(source),
    )
    task = {
        "task_uid": "task-001",
        "repo": "example/repo",
        "base_commit": commit,
    }
    arm = SimpleNamespace(name="original")

    first = factory(task, arm)
    second = factory(task, arm)

    assert first == second


def _archive_bytes(*, root: str = "repo-base") -> bytes:
    payload = b"value = 7\n"
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        member = tarfile.TarInfo(f"{root}/source.py")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    return stream.getvalue()


def test_workspace_factory_uses_pinned_github_archive_fallback(tmp_path: Path):
    base_commit = "a" * 40
    requested = []

    def failing_fetch(_args, *, cwd):
        raise WorkspaceMaterializationError(f"blocked git transport: {cwd}")

    def archive_fetcher(repo: str, commit: str) -> bytes:
        requested.append((repo, commit))
        return _archive_bytes()

    factory = GitWorkspaceFactory(
        root=tmp_path / "workspaces",
        remote_resolver=lambda _repo: "https://github.com/example/repo.git",
        network_fetcher=failing_fetch,
        archive_fetcher=archive_fetcher,
    )
    task = {
        "task_uid": "task-archive",
        "repo": "example/repo",
        "base_commit": base_commit,
    }

    original = factory(task, SimpleNamespace(name="original"))
    parent = factory(task, SimpleNamespace(name="parent"))

    assert requested == [("example/repo", base_commit)]
    assert (original / "source.py").read_text() == "value = 7\n"
    assert (parent / "source.py").read_text() == "value = 7\n"
    provenance = json.loads((original / ".git/evolve-source.json").read_text())
    assert provenance["base_commit"] == base_commit
    assert provenance["source"] == "github-codeload-pinned-archive"
    assert provenance["archive_sha256"]
    assert (
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=original,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )


def test_workspace_factory_rejects_archive_path_traversal(tmp_path: Path):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        payload = b"escape"
        member = tarfile.TarInfo("repo-base/../../escape.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    factory = GitWorkspaceFactory(
        root=tmp_path / "workspaces",
        remote_resolver=lambda _repo: "https://github.com/example/repo.git",
        network_fetcher=lambda _args, *, cwd: (_ for _ in ()).throw(
            WorkspaceMaterializationError(f"blocked git transport: {cwd}")
        ),
        archive_fetcher=lambda _repo, _commit: stream.getvalue(),
    )

    with pytest.raises(WorkspaceMaterializationError, match="unsafe archive member"):
        factory(
            {
                "task_uid": "task-unsafe",
                "repo": "example/repo",
                "base_commit": "b" * 40,
            },
            SimpleNamespace(name="original"),
        )
