"""Fail-fast local Qwen experiment for the AST-anchored symbol action space."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import ContractError
from .eval_manifest import EvaluationTaskSet
from .experiment import PairedExperimentRunner
from .p1_local_feedback import load_frozen_p1_local_feedback_revision
from .p1_parent import load_frozen_p1_parent_revision
from .symbol_rewrite import (
    MlxSymbolRewriteGenerator,
    SymbolRewriteAdapter,
    build_symbol_conditions,
)
from .target_selection import TargetSelectionManifest


def run_local_qwen_symbol_p1(
    *,
    manifest_path: Path,
    skill_revision_path: Path,
    target_selection_path: Path,
    evidence_root: Path,
    workspace_root: Path,
    model_path: Path,
    max_cells: int | None,
    max_tokens: int = 1536,
    context_chars: int = 8_000,
    task_ids: set[str] | None = None,
    condition_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Run/resume a strict baseline/taught symbol-rewrite comparison."""
    try:
        taskset = EvaluationTaskSet.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        wrapper = json.loads(skill_revision_path.read_text(encoding="utf-8"))
        target_selection = TargetSelectionManifest.from_dict(
            json.loads(target_selection_path.read_text(encoding="utf-8")),
            taskset=taskset,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("symbol P1 inputs are unreadable") from exc
    if wrapper.get("candidate_source") == "local-feedback-compiler-v1":
        taught_revision = load_frozen_p1_local_feedback_revision(skill_revision_path)
    else:
        taught_revision = load_frozen_p1_parent_revision(skill_revision_path)
    if not model_path.is_dir():
        raise ContractError(f"local Qwen model is missing: {model_path}")

    generator = MlxSymbolRewriteGenerator(
        model_path=str(model_path.resolve()),
        max_tokens=max_tokens,
        max_context_chars=context_chars,
    )
    adapter = SymbolRewriteAdapter(generator=generator)
    generation_config = adapter.experiment_config()
    conditions = build_symbol_conditions(
        taught_skill=taught_revision.skill_text,
        parent_revision_id=taught_revision.revision_id,
        source_round=taught_revision.source_round,
        generation_config=generation_config,
    )
    if condition_ids is not None:
        known = {condition.condition_id for condition in conditions}
        if not condition_ids <= known:
            raise ContractError("unknown symbol experiment condition filter")
        conditions = [
            condition
            for condition in conditions
            if condition.condition_id in condition_ids
        ]
    runner = PairedExperimentRunner(
        taskset=taskset,
        adapters={"symbol": adapter},
        conditions=conditions,
        evidence_root=evidence_root,
        workspace_root=workspace_root,
        qualification_fingerprint=target_selection.fingerprint,
    )
    return runner.run(
        max_cells=max_cells,
        task_ids=task_ids,
        condition_ids=condition_ids,
    )
