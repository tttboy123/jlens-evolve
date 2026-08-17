"""Deterministic zero-network smoke run for the complete feedback loop."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .contracts import LoopAuthorization, LoopRevision, canonical_json
from .evaluator import (
    EvaluationBaseline,
    EvaluationPolicy,
    LoopEvaluator,
    NativeOutcome,
)
from .ledger import ParentCallLedger
from .loop import LoopConfig, LoopDriver
from .parent_model import ParentModelAdapter
from .registry import LoopRevisionRegistry
from .student_adapter import StudentAdapter, StudentTask


def _fixture_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    source = path / "src/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Offline Smoke",
            "-c",
            "user.email=smoke@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=path,
        check=True,
    )
    return path


def run_offline_smoke(output_root: Path) -> dict[str, Any]:
    """Run failure -> feedback -> revision -> convergence without network I/O."""
    root = output_root.resolve()
    if root.exists():
        raise FileExistsError(f"offline smoke output already exists: {root}")
    root.mkdir(parents=True)
    feedback_repo = _fixture_repo(root / "fixtures/feedback")
    holdout_repo = _fixture_repo(root / "fixtures/holdout")

    initial = LoopRevision.create(
        skill_id="offline-structured-edit",
        revision_id="offline-rev-000",
        parent_revision_id=None,
        source_round=0,
        protocol="structured-search-replace-v1",
        skill_text="Return a bounded edit.",
        prompt_template="Edit the allowed target.",
        eval_note="offline baseline",
    )

    def student(task: StudentTask, revision: LoopRevision) -> str:
        if revision.revision_id == initial.revision_id:
            return "The value should change, but no machine-applicable edit is emitted."
        return json.dumps(
            {
                "file": "src/example.py",
                "search": "x = 1",
                "replace": "x = 2",
                "diagnostic": f"bounded edit for {task.task_id}",
            }
        )

    def parent(_request: Any) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "protocol": "structured-search-replace-v1",
            "skill_text": "Copy one exact span and return one bounded JSON edit.",
            "prompt_template": "Return JSON for the allowed non-test target.",
            "eval_note": "Converted reasoning-only output into an edit contract.",
            "usage": {"total_tokens": 0, "transport": "offline-fixture"},
        }

    authorization = LoopAuthorization.create(
        authorization_id="offline-smoke",
        approved_by="local-fixture",
        maximum_parent_calls=1,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    driver = LoopDriver(
        student=StudentAdapter(generator=student),
        evaluator=LoopEvaluator(EvaluationPolicy.strict()),
        parent=ParentModelAdapter(
            ledger=ParentCallLedger(root / "parent-calls.json", authorization),
            transport=parent,
        ),
        registry=LoopRevisionRegistry(root / "registry"),
        evidence_root=root / "evidence",
        authorization=authorization,
        config=LoopConfig(max_rounds=3, no_progress_patience=2),
    )
    tasks = [
        StudentTask.create(
            task_id="feedback-001",
            checkout=feedback_repo,
            instruction="Change x from one to two.",
            allowed_targets=["src/example.py"],
            cohort="feedback",
        ),
        StudentTask.create(
            task_id="holdout-001",
            checkout=holdout_repo,
            instruction="Change x from one to two.",
            allowed_targets=["src/example.py"],
            cohort="holdout",
        ),
    ]
    result = driver.run(
        initial_revision=initial,
        tasks=tasks,
        native_evaluator=lambda attempt: NativeOutcome(
            resolved=attempt.structural_valid,
            safe=True,
            detail="offline native fixture",
        ),
        baseline=EvaluationBaseline(
            feedback_native_rate=0.0,
            holdout_native_rate=0.0,
        ),
    )
    report = {
        "schema_version": 1,
        "network_calls_performed": False,
        "parent_transport": "deterministic-local-fixture",
        "result": result.to_dict(),
    }
    (root / "RESULT.json").write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report
