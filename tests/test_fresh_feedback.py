from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

from evolve.contracts import Cohort, ContractViolation, canonical_json
from evolve.evidence import ReceiptStore
from evolve.fresh_feedback import (
    _build_tasks,
    _launcher,
    _load_config,
    _require_clean_head,
    _trusted_jlens_runtime,
    seal_run,
)
from evolve.proposals import CandidateCompiler, CompiledRevision, CompileSpec


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source(tmp_path: Path, name: str) -> tuple[Path, str]:
    root = tmp_path / name
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "Tests")
    _git(root, "config", "user.email", "tests@example.invalid")
    (root / "source.txt").write_text(name, encoding="utf-8")
    _git(root, "add", "source.txt")
    _git(root, "commit", "-m", "source")
    return root, _git(root, "rev-parse", "HEAD")


def test_fresh_config_denies_any_non_feedback_task(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tasks": [
                    {"instance_id": "a", "cohort": "feedback"},
                    {"instance_id": "b", "cohort": "holdout"},
                    {"instance_id": "c", "cohort": "feedback"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ContractViolation, match="holdout"):
        _load_config(path)


@pytest.mark.parametrize("field", ["operator_skill_path", "span_skill_path"])
def test_fresh_config_denies_legacy_frozen_skill_fallbacks(
    tmp_path: Path, field: str
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tasks": [
                    {"instance_id": "a", "cohort": "feedback"},
                    {"instance_id": "b", "cohort": "feedback"},
                    {"instance_id": "c", "cohort": "feedback"},
                ],
                field: "/legacy/frozen-skill.json",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ContractViolation, match="fallback"):
        _load_config(path)


def test_trusted_jlens_requires_a_process_local_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("JLENS_TEST_SECRET", raising=False)
    config = {
        "trusted_jlens": {
            "secret_env": "JLENS_TEST_SECRET",
        }
    }

    with pytest.raises(ContractViolation, match="environment variable is missing"):
        _trusted_jlens_runtime(
            config=config,
            compiled=cast(CompiledRevision, object()),
            receipt_store=ReceiptStore(tmp_path / "receipts"),
        )


def test_build_tasks_binds_exact_clean_git_tree_and_feedback_identity(
    tmp_path: Path,
) -> None:
    rows = []
    for index, project in enumerate(("sphinx", "phpoffice", "laravel"), 1):
        source, revision = _source(tmp_path, project)
        rows.append(
            {
                "source_uri": str(source),
                "base_revision": revision,
                "instance_id": f"task-{index}",
                "project": project,
                "benchmark_id": "swe-bench-verified",
                "catalog_fingerprint": hashlib.sha256(project.encode()).hexdigest(),
            }
        )

    tasks, metadata, inventory = _build_tasks(rows, "native-v1")

    assert len(tasks) == len(metadata) == len(inventory) == 3
    assert all(str(task.cohort) == "feedback" for task in tasks)
    assert all(task.evaluator_id == "native-v1" for task in tasks)
    assert metadata[tasks[0].revision_id]["base_revision"] == rows[0]["base_revision"]


def test_seal_run_hashes_every_non_source_artifact_and_verifies(tmp_path: Path) -> None:
    (tmp_path / "artifact.txt").write_text("sealed", encoding="utf-8")
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources/large-cache.txt").write_text("cache", encoding="utf-8")

    assert seal_run(tmp_path) == 1
    manifest = json.loads((tmp_path / "EVIDENCE-MANIFEST.json").read_text())
    assert manifest["entries"] == [
        {
            "path": "artifact.txt",
            "sha256": hashlib.sha256(b"sealed").hexdigest(),
        }
    ]
    assert manifest["excluded_prefixes"] == ["sources/"]


def test_release_identity_rejects_a_dirty_worktree(tmp_path: Path) -> None:
    source, revision = _source(tmp_path, "release")
    assert _require_clean_head(source) == revision
    (source / "source.txt").write_text("dirty", encoding="utf-8")

    with pytest.raises(ContractViolation, match="clean committed HEAD"):
        _require_clean_head(source)


def test_python_launcher_preserves_venv_symlink(tmp_path: Path) -> None:
    target = tmp_path / "python-real"
    target.write_text("binary", encoding="utf-8")
    launcher = tmp_path / "venv-python"
    launcher.symlink_to(target)

    assert _launcher({"python": str(launcher)}, "python") == launcher.absolute()


def test_fresh_campaign_consumes_precompiled_candidate_with_parent_lineage(
    tmp_path: Path,
) -> None:
    request = {
        "request_id": "teacher-r1",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "failure_package": {"feedback_only": True},
    }
    request_path = tmp_path / "TEACHER-REQUEST.json"
    request_path.write_text(canonical_json(request) + "\n", encoding="utf-8")
    response = {
        "schema_version": 1,
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "candidate": {
            "protocol": "skill-v1",
            "prompt_template": "Repair {task}",
            "skill_text": "Localize the declared symbol before editing.",
            "eval_note": "Native feedback paired evaluation.",
        },
        "candidate_status": "inactive",
        "auto_activate": False,
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        "network_calls": 1,
        "pricing_cny_per_million": {"input": 2.0, "output": 8.0},
        "estimated_cost_cny": 0.00006,
    }
    response_path = tmp_path / "TEACHER-RESPONSE.json"
    response_path.write_text(canonical_json(response) + "\n", encoding="utf-8")
    compiled = CandidateCompiler().compile(
        request_path=request_path,
        response_path=response_path,
        compile_spec=CompileSpec(
            candidate_id="candidate-r1",
            revision_id="candidate-r1-revision",
            parent_revision_id="best-parent-r0",
            cohort=Cohort.FEEDBACK,
            operator_id="operator-r1",
            operator_instruction="Apply skill.",
            routes=(("task-a", "operator-r1"),),
        ),
        output_root=tmp_path / "compiled",
    )

    from evolve.fresh_feedback import _compile_teacher_candidate

    loaded = _compile_teacher_candidate(
        config={
            "compiled_revision_root": str(compiled.root),
            "candidate_id": compiled.change_set.candidate_id,
            "candidate_revision_id": compiled.change_set.revision_id,
            "parent_revision_id": "best-parent-r0",
            "tasks": [{"instance_id": "task-a"}],
        },
        output_root=tmp_path / "run",
    )

    assert loaded.bundle_sha256 == compiled.bundle_sha256
    assert loaded.change_set.parent_revision_id == "best-parent-r0"
