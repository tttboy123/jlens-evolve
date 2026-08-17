"""Generate and materialize project-local Codex AgentChangeSets."""

from __future__ import annotations

import difflib
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codex_target_runtime import CodexHistoryTask, CodexProjectProfile

ALLOWED_CHANGE_PATHS = (
    "AGENTS.md",
    ".agents/skills/evidence-to-agent-change/SKILL.md",
    ".codex/evolution-policy.json",
)


class AgentChangeSetError(ValueError):
    """Raised when a ChangeSet or its materialization crosses the frozen policy."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _profile_tree_hash(root: Path) -> str:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return _sha256_text(_canonical_json(rows))


def _safe_target(root: Path, relative: str) -> Path:
    if relative.startswith("/") or relative not in ALLOWED_CHANGE_PATHS:
        raise AgentChangeSetError(f"change path is not allowed: {relative}")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise AgentChangeSetError(f"unsafe change path: {relative}")
    return target


@dataclass(frozen=True)
class AgentFileChange:
    path: str
    surface: str
    operation: str
    before: str | None
    after: str | None
    rationale: str

    @property
    def before_sha256(self) -> str | None:
        return None if self.before is None else _sha256_text(self.before)

    @property
    def after_sha256(self) -> str | None:
        return None if self.after is None else _sha256_text(self.after)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "surface": self.surface,
            "operation": self.operation,
            "before": self.before,
            "after": self.after,
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class AgentChangeSet:
    schema_version: int
    changeset_id: str
    status: str
    parent_tree_hash: str
    changes: tuple[AgentFileChange, ...]
    evidence_refs: tuple[dict[str, Any], ...]
    risks: tuple[str, ...]
    model_calls: int = 0
    auto_apply: bool = False

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "changeset_id": self.changeset_id,
            "status": self.status,
            "parent_tree_hash": self.parent_tree_hash,
            "changes": [change.to_dict() for change in self.changes],
            "evidence_refs": list(self.evidence_refs),
            "risks": list(self.risks),
            "model_calls": self.model_calls,
            "auto_apply": self.auto_apply,
        }

    @property
    def sha256(self) -> str:
        return _sha256_text(_canonical_json(self._payload()))

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "changeset_hash": self.sha256}

    def validate(self, baseline_root: Path) -> None:
        if self.schema_version != 1 or self.status != "candidate":
            raise AgentChangeSetError("unsupported ChangeSet contract")
        if self.auto_apply or self.model_calls != 0:
            raise AgentChangeSetError("ChangeSet must be offline and non-applying")
        if tuple(change.path for change in self.changes) != ALLOWED_CHANGE_PATHS:
            raise AgentChangeSetError("ChangeSet must cover the three frozen paths")
        if {change.surface for change in self.changes} != {"prompt", "skill", "policy"}:
            raise AgentChangeSetError("ChangeSet must cover prompt, skill, and policy")
        if _profile_tree_hash(baseline_root) != self.parent_tree_hash:
            raise AgentChangeSetError("baseline tree hash does not match parent")
        for change in self.changes:
            target = _safe_target(baseline_root, change.path)
            actual = target.read_text(encoding="utf-8") if target.is_file() else None
            if actual != change.before:
                raise AgentChangeSetError(f"before content mismatch: {change.path}")
            if change.operation == "add" and change.before is not None:
                raise AgentChangeSetError(
                    f"add change has existing content: {change.path}"
                )
            if change.operation == "modify" and (
                change.before is None or change.after is None
            ):
                raise AgentChangeSetError(f"modify change is incomplete: {change.path}")


_PROMPT_BLOCK = """
## Evolution delivery contract

For Agent-evolution work, lead with the changed target behavior and its decision.
Separate observed evidence, causal limits, exact Prompt/Skill/Policy mappings,
verification, limitations, and the next safe action.  Include the runnable process
and artifact paths when handing off a result.  JLens observations never establish
causality without a matched intervention.

<!-- evolve-capabilities: ["outcome_contract"] -->
"""

_SKILL = """---
name: evidence-to-agent-change
description: Convert real user corrections into auditable application-layer changes.
---

# Evidence to Agent Change

Cluster repeated user corrections by failure hypothesis.  For each cluster, map the
failure to one of Prompt, Skill, or Policy; state the causal boundary; propose the
smallest ChangeSet; remove unnecessary components; and define matched verification.
Keep rejected candidates and never treat an observer correlation as the intervention.

<!-- evolve-capabilities: ["causal_change", "complexity_control"] -->
"""


def propose_changeset(
    baseline_root: Path, public_tasks: tuple[CodexHistoryTask, ...]
) -> AgentChangeSet:
    """Create one deterministic three-surface candidate from public evidence only."""

    if not public_tasks or any(task.partition != "public" for task in public_tasks):
        raise AgentChangeSetError("proposer accepts public history only")
    families = {task.task_family for task in public_tasks}
    required_families = {
        "result_legibility",
        "evidence_to_change",
        "operation_contract",
        "complexity_control",
        "plugin_boundary",
    }
    if not required_families.issubset(families):
        raise AgentChangeSetError(
            "public evidence does not cover frozen failure families"
        )
    profile = CodexProjectProfile.from_path(baseline_root)
    agents_path = _safe_target(baseline_root, "AGENTS.md")
    policy_path = _safe_target(baseline_root, ".codex/evolution-policy.json")
    before_agents = agents_path.read_text(encoding="utf-8")
    before_policy = policy_path.read_text(encoding="utf-8")
    policy = json.loads(before_policy)
    policy["capabilities"] = ["plugin_governance", "rollback_safety"]
    policy["candidate_frozen_before_sealed"] = True
    policy["matched_ab_required"] = True
    policy["preserve_rejected_candidates"] = True
    after_policy = (
        json.dumps(policy, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    evidence_refs = tuple(
        {
            "ordinal": task.ordinal,
            "message_sha256": task.text_sha256,
            "partition": task.partition,
            "task_family": task.task_family,
        }
        for task in public_tasks
    )
    return AgentChangeSet(
        schema_version=1,
        changeset_id="codex-real-history-001",
        status="candidate",
        parent_tree_hash=profile.tree_hash,
        changes=(
            AgentFileChange(
                path="AGENTS.md",
                surface="prompt",
                operation="modify",
                before=before_agents,
                after=before_agents.rstrip() + "\n" + _PROMPT_BLOCK,
                rationale="Repeated requests could not identify the result, evidence, or next action.",
            ),
            AgentFileChange(
                path=".agents/skills/evidence-to-agent-change/SKILL.md",
                surface="skill",
                operation="add",
                before=None,
                after=_SKILL,
                rationale="Repeated evidence-to-change and complexity failures need a reusable procedure.",
            ),
            AgentFileChange(
                path=".codex/evolution-policy.json",
                surface="policy",
                operation="modify",
                before=before_policy,
                after=after_policy,
                rationale="Plugin authority, sealed ordering, rollback, and non-application need enforcement.",
            ),
        ),
        evidence_refs=evidence_refs,
        risks=(
            "offline contract replay does not prove model instruction-following",
            "selected history comes from one project thread",
            "candidate must remain project-local until a live audit is separately authorized",
        ),
    )


def _apply_changes(changeset: AgentChangeSet, root: Path, *, reverse: bool) -> None:
    changes = reversed(changeset.changes) if reverse else changeset.changes
    for change in changes:
        target = _safe_target(root, change.path)
        expected = change.after if reverse else change.before
        replacement = change.before if reverse else change.after
        actual = target.read_text(encoding="utf-8") if target.is_file() else None
        if actual != expected:
            raise AgentChangeSetError(
                f"materialization content mismatch: {change.path}"
            )
        if replacement is None:
            target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(replacement, encoding="utf-8")


def _diff_for_change(change: AgentFileChange, *, reverse: bool) -> str:
    before = change.after if reverse else change.before
    after = change.before if reverse else change.after
    fromfile = f"a/{change.path}" if before is not None else "/dev/null"
    tofile = f"b/{change.path}" if after is not None else "/dev/null"
    return "".join(
        difflib.unified_diff(
            [] if before is None else before.splitlines(keepends=True),
            [] if after is None else after.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


def _run_git_apply(root: Path, patch: Path, *, check: bool) -> None:
    args = ["git", "apply"]
    if check:
        args.append("--check")
    args.append(str(patch.resolve()))
    completed = subprocess.run(
        args,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if completed.returncode:
        raise AgentChangeSetError(
            f"patch verification failed ({completed.returncode}): {completed.stderr.strip()}"
        )


def materialize_changeset(
    *, changeset: AgentChangeSet, baseline_root: Path, output_dir: Path
) -> dict[str, Any]:
    changeset.validate(baseline_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise AgentChangeSetError(f"refusing non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_snapshot = output_dir / "baseline-profile"
    candidate_snapshot = output_dir / "candidate-profile"
    shutil.copytree(baseline_root, baseline_snapshot)
    shutil.copytree(baseline_root, candidate_snapshot)
    _apply_changes(changeset, candidate_snapshot, reverse=False)

    apply_patch = output_dir / "apply.patch"
    rollback_patch = output_dir / "rollback.patch"
    apply_patch.write_text(
        "".join(
            _diff_for_change(change, reverse=False) for change in changeset.changes
        ),
        encoding="utf-8",
    )
    rollback_patch.write_text(
        "".join(
            _diff_for_change(change, reverse=True)
            for change in reversed(changeset.changes)
        ),
        encoding="utf-8",
    )
    (output_dir / "AgentChangeSet.json").write_text(
        json.dumps(changeset.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    baseline_hash = _profile_tree_hash(baseline_snapshot)
    candidate_hash = _profile_tree_hash(candidate_snapshot)
    with tempfile.TemporaryDirectory(prefix="codex-changeset-") as temporary:
        isolated = Path(temporary) / "profile"
        shutil.copytree(baseline_snapshot, isolated)
        _run_git_apply(isolated, apply_patch, check=True)
        _run_git_apply(isolated, apply_patch, check=False)
        apply_equal = _profile_tree_hash(isolated) == candidate_hash
        _run_git_apply(isolated, rollback_patch, check=True)
        _run_git_apply(isolated, rollback_patch, check=False)
        rollback_equal = _profile_tree_hash(isolated) == baseline_hash
    verification = {
        "changeset_hash": changeset.sha256,
        "baseline_tree_hash": baseline_hash,
        "candidate_tree_hash": candidate_hash,
        "apply_patch_check": apply_equal,
        "rollback_patch_check": rollback_equal,
        "rollback_tree_hash_equal": rollback_equal,
        "auto_applied_to_live_profile": False,
    }
    (output_dir / "patch-verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return verification
