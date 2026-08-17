from __future__ import annotations

import subprocess
from pathlib import Path

from skill_evolution_loop.contracts import canonical_json, sha256_json
from skill_evolution_loop.round1_sources import prefetch_round1_sources


def _remote(path: Path) -> str:
    source = path / "source"
    source.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    (source / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=source,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    remote = path / "remotes/org/repo"
    remote.parent.mkdir(parents=True)
    subprocess.run(
        ["git", "clone", "-q", "--bare", str(source), str(remote)], check=True
    )
    return commit


def test_round1_source_prefetch_is_resumable_and_commit_pinned(
    tmp_path: Path,
) -> None:
    commit = _remote(tmp_path)
    tasks = [
        {
            "repo": "org/repo",
            "base_commit": commit,
        }
        for _ in range(60)
    ]
    content = {"status": "frozen", "task_count": 60, "tasks": tasks}
    selection = tmp_path / "selection.json"
    selection.write_text(
        canonical_json({**content, "evidence_sha256": sha256_json(content)}) + "\n",
        encoding="utf-8",
    )
    fields = {
        "selection_path": selection,
        "cache_root": tmp_path / "cache",
        "evidence_root": tmp_path / "evidence",
        "repository_url_template": f"file://{tmp_path}/remotes/{{repo}}",
    }

    first = prefetch_round1_sources(**fields)
    resumed = prefetch_round1_sources(**fields)

    assert first == resumed
    assert first["status"] == "complete"
    assert first["planned_repositories"] == 1
    assert first["verified_commit_count"] == 1
    assert first["network_calls_performed"] is False
