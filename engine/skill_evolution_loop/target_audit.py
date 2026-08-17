"""Evaluator-only audit that keeps impossible target scopes out of P1."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ContractError, sha256_json
from .eval_manifest import EvaluationTaskSet

_PATCH_TARGET = re.compile(r"^\+\+\+ b/([^\t\n]+)(?:\t.*)?$", re.MULTILINE)
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$")
_TEST_PARTS = frozenset({"test", "tests", "__tests__"})
_DOCUMENTATION_PARTS = frozenset({"doc", "docs", "documentation"})


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_implementation_target(relative: str) -> bool:
    path = Path(relative)
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if parts & (_TEST_PARTS | _DOCUMENTATION_PARTS):
        return False
    if (
        name.startswith("test_")
        or ".test." in name
        or ".spec." in name
        or name.startswith("readme")
        or name.startswith("changelog")
        or name in {"authors", "changes", "contributing", "news"}
    ):
        return False
    return True


def reference_implementation_targets(patch_text: str) -> tuple[str, ...]:
    """Return evaluator-only implementation paths from a unified diff."""

    return tuple(
        sorted(
            {
                target
                for target in _PATCH_TARGET.findall(patch_text)
                if target != "/dev/null" and _is_implementation_target(target)
            }
        )
    )


@dataclass(frozen=True)
class GoldPatchReference:
    """Private evaluator input; never serialized into a Student task."""

    task_id: str
    patch_path: Path


@dataclass(frozen=True)
class TargetCoverageRow:
    task_id: str
    reference_patch_sha256: str
    implementation_targets: tuple[str, ...]
    allowed_targets: tuple[str, ...]
    missing_targets: tuple[str, ...]
    mechanism_max_files: int
    within_mechanism_capacity: bool
    ready: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "reference_patch_sha256": self.reference_patch_sha256,
            "implementation_targets": list(self.implementation_targets),
            "allowed_targets": list(self.allowed_targets),
            "missing_targets": list(self.missing_targets),
            "mechanism_max_files": self.mechanism_max_files,
            "within_mechanism_capacity": self.within_mechanism_capacity,
            "ready": self.ready,
        }


@dataclass(frozen=True)
class TargetCoverageAudit:
    schema_version: int
    taskset_id: str
    taskset_fingerprint: str
    evaluator_only: bool
    audited_tasks: int
    ready_tasks: int
    ready: bool
    rows: tuple[TargetCoverageRow, ...]
    network_calls_performed: bool

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "taskset_id": self.taskset_id,
            "taskset_fingerprint": self.taskset_fingerprint,
            "evaluator_only": self.evaluator_only,
            "audited_tasks": self.audited_tasks,
            "ready_tasks": self.ready_tasks,
            "ready": self.ready,
            "rows": [row.to_dict() for row in self.rows],
            "network_calls_performed": self.network_calls_performed,
        }

    @property
    def audit_sha256(self) -> str:
        return sha256_json(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "audit_sha256": self.audit_sha256}


@dataclass(frozen=True)
class MechanismCapacityPolicy:
    """Frozen output limits for the two Student adapter contracts."""

    structured_max_tokens: int
    tokenizer_path: Path
    hunk_max_hunks: int = 3
    hunk_max_changed_lines: int = 50
    hunk_max_lines: int = 80

    def validate(self) -> None:
        limits = (
            self.structured_max_tokens,
            self.hunk_max_hunks,
            self.hunk_max_changed_lines,
            self.hunk_max_lines,
        )
        if any(type(value) is not int or value < 1 for value in limits):
            raise ContractError("mechanism capacity limits must be positive integers")
        if not self.tokenizer_path.resolve().is_file():
            raise ContractError("mechanism capacity tokenizer is unavailable")

    @property
    def tokenizer_sha256(self) -> str:
        return _file_sha256(self.tokenizer_path.resolve())

    def to_dict(self) -> dict[str, Any]:
        return {
            "structured_max_tokens": self.structured_max_tokens,
            "tokenizer_sha256": self.tokenizer_sha256,
            "hunk_max_hunks": self.hunk_max_hunks,
            "hunk_max_changed_lines": self.hunk_max_changed_lines,
            "hunk_max_lines": self.hunk_max_lines,
        }


@dataclass(frozen=True)
class MechanismCapacityRow:
    task_id: str
    reference_patch_sha256: str
    implementation_targets: tuple[str, ...]
    allowed_targets: tuple[str, ...]
    missing_targets: tuple[str, ...]
    implementation_hunks: int
    implementation_changed_lines: int
    implementation_patch_lines: int
    structured_output_tokens: int | None
    structured_unique_search: bool
    structured_ready: bool
    hunk_ready: bool
    reasons: tuple[str, ...]
    ready: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "reference_patch_sha256": self.reference_patch_sha256,
            "implementation_targets": list(self.implementation_targets),
            "allowed_targets": list(self.allowed_targets),
            "missing_targets": list(self.missing_targets),
            "implementation_hunks": self.implementation_hunks,
            "implementation_changed_lines": self.implementation_changed_lines,
            "implementation_patch_lines": self.implementation_patch_lines,
            "structured_output_tokens": self.structured_output_tokens,
            "structured_unique_search": self.structured_unique_search,
            "structured_ready": self.structured_ready,
            "hunk_ready": self.hunk_ready,
            "reasons": list(self.reasons),
            "ready": self.ready,
        }


@dataclass(frozen=True)
class MechanismCapacityAudit:
    schema_version: int
    taskset_id: str
    taskset_fingerprint: str
    evaluator_only: bool
    policy: MechanismCapacityPolicy
    audited_tasks: int
    ready_tasks: int
    ready: bool
    rows: tuple[MechanismCapacityRow, ...]
    network_calls_performed: bool

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "taskset_id": self.taskset_id,
            "taskset_fingerprint": self.taskset_fingerprint,
            "evaluator_only": self.evaluator_only,
            "policy": self.policy.to_dict(),
            "audited_tasks": self.audited_tasks,
            "ready_tasks": self.ready_tasks,
            "ready": self.ready,
            "rows": [row.to_dict() for row in self.rows],
            "network_calls_performed": self.network_calls_performed,
        }

    @property
    def audit_sha256(self) -> str:
        return sha256_json(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "audit_sha256": self.audit_sha256}


@dataclass(frozen=True)
class _PatchHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _FilePatch:
    target: str
    hunks: tuple[_PatchHunk, ...]


def _parse_file_patches(patch_text: str) -> tuple[_FilePatch, ...]:
    """Parse the bounded unified-diff subset used by benchmark fix patches."""
    lines = patch_text.splitlines(keepends=True)
    files: list[_FilePatch] = []
    target: str | None = None
    hunks: list[_PatchHunk] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("+++ "):
            if target is not None:
                files.append(_FilePatch(target=target, hunks=tuple(hunks)))
            raw_target = line[4:].split("\t", 1)[0].strip()
            target = raw_target[2:] if raw_target.startswith("b/") else raw_target
            hunks = []
            index += 1
            continue
        match = _HUNK_HEADER.match(line.rstrip("\n"))
        if match and target is not None:
            body: list[str] = []
            index += 1
            while index < len(lines):
                candidate = lines[index]
                if candidate.startswith(("diff --git ", "--- ", "+++ ", "@@ ")):
                    break
                if candidate.startswith((" ", "+", "-", "\\")):
                    body.append(candidate)
                index += 1
            hunks.append(
                _PatchHunk(
                    old_start=int(match.group(1)),
                    old_count=int(match.group(2) or "1"),
                    new_start=int(match.group(3)),
                    new_count=int(match.group(4) or "1"),
                    lines=tuple(body),
                )
            )
            continue
        index += 1
    if target is not None:
        files.append(_FilePatch(target=target, hunks=tuple(hunks)))
    return tuple(files)


def _apply_file_patch(source: str, file_patch: _FilePatch) -> str:
    before = source.splitlines(keepends=True)
    after: list[str] = []
    cursor = 0
    for hunk in file_patch.hunks:
        old_index = hunk.old_start - 1
        if old_index < cursor or old_index > len(before):
            raise ContractError("mechanism capacity reference hunk is invalid")
        after.extend(before[cursor:old_index])
        consumed: list[str] = []
        replacement: list[str] = []
        for line in hunk.lines:
            if line.startswith("\\"):
                continue
            marker, content = line[0], line[1:]
            if marker in {" ", "-"}:
                consumed.append(content)
            if marker in {" ", "+"}:
                replacement.append(content)
        if len(consumed) != hunk.old_count or len(replacement) != hunk.new_count:
            raise ContractError("mechanism capacity reference hunk counts disagree")
        if before[old_index : old_index + hunk.old_count] != consumed:
            raise ContractError("mechanism capacity reference does not match source")
        after.extend(replacement)
        cursor = old_index + hunk.old_count
    after.extend(before[cursor:])
    return "".join(after)


def _shortest_unique_edit(before: str, after: str) -> tuple[str, str, bool]:
    """Return one line-bounded search/replace spanning every reference change."""
    old = before.splitlines(keepends=True)
    new = after.splitlines(keepends=True)
    prefix = 0
    while prefix < min(len(old), len(new)) and old[prefix] == new[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < len(old) - prefix
        and suffix < len(new) - prefix
        and old[len(old) - suffix - 1] == new[len(new) - suffix - 1]
    ):
        suffix += 1
    old_end = len(old) - suffix
    new_end = len(new) - suffix
    left = prefix
    right_extension = 0
    while True:
        search = "".join(old[left : old_end + right_extension])
        replace = "".join(new[left : new_end + right_extension])
        if search and before.count(search) == 1:
            return search, replace, True
        if left > 0:
            left -= 1
            continue
        maximum_extension = min(len(old) - old_end, len(new) - new_end)
        if right_extension < maximum_extension:
            right_extension += 1
            continue
        return search, replace, False


def _token_counter(tokenizer_path: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise ContractError(
            "mechanism capacity tokenizer runtime is unavailable"
        ) from exc
    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path.resolve()))
    except Exception as exc:  # pragma: no cover - library-specific parse failures
        raise ContractError("mechanism capacity tokenizer is unreadable") from exc
    return lambda value: len(tokenizer.encode(value).ids)


def audit_target_coverage(
    taskset: EvaluationTaskSet,
    references: list[GoldPatchReference],
    *,
    mechanism_max_files: int = 1,
    mechanism_max_files_by_task: dict[str, int] | None = None,
) -> TargetCoverageAudit:
    """Check that bounded edit scopes cover reference implementation files.

    The report intentionally includes only patch hashes and target paths. Patch
    contents remain outside the TaskSet, runner, revisions, and feedback loop.
    """
    taskset.validate()
    if type(mechanism_max_files) is not int or mechanism_max_files < 1:
        raise ContractError("target coverage mechanism capacity must be positive")
    capacities = mechanism_max_files_by_task or {}
    if any(
        task_id not in {task.task_id for task in taskset.tasks}
        or type(limit) is not int
        or limit < 1
        for task_id, limit in capacities.items()
    ):
        raise ContractError("target coverage task capacity is invalid")
    if not references:
        raise ContractError("target coverage audit requires references")
    task_ids = [row.task_id for row in references]
    if len(set(task_ids)) != len(task_ids):
        raise ContractError("target coverage audit task ids must be unique")
    tasks = {row.task_id: row for row in taskset.tasks}
    unknown = sorted(set(task_ids) - set(tasks))
    if unknown:
        raise ContractError(f"target coverage audit has unknown task: {unknown[0]}")

    rows: list[TargetCoverageRow] = []
    for reference in references:
        patch_path = reference.patch_path.resolve()
        if not patch_path.is_file():
            raise ContractError(
                f"target coverage reference patch is unavailable: {reference.task_id}"
            )
        try:
            patch_text = patch_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ContractError(
                f"target coverage reference patch is unavailable: {reference.task_id}"
            ) from exc
        targets = reference_implementation_targets(patch_text)
        if not targets:
            raise ContractError(
                f"target coverage reference has no implementation target: "
                f"{reference.task_id}"
            )
        task = tasks[reference.task_id]
        allowed = tuple(sorted(task.allowed_targets))
        missing = tuple(sorted(set(targets) - set(allowed)))
        task_capacity = capacities.get(reference.task_id, mechanism_max_files)
        within_capacity = len(targets) <= task_capacity
        rows.append(
            TargetCoverageRow(
                task_id=reference.task_id,
                reference_patch_sha256=_file_sha256(patch_path),
                implementation_targets=targets,
                allowed_targets=allowed,
                missing_targets=missing,
                mechanism_max_files=task_capacity,
                within_mechanism_capacity=within_capacity,
                ready=not missing and within_capacity,
            )
        )
    ordered = tuple(sorted(rows, key=lambda row: row.task_id))
    ready_tasks = sum(row.ready for row in ordered)
    return TargetCoverageAudit(
        schema_version=1,
        taskset_id=taskset.taskset_id,
        taskset_fingerprint=taskset.fingerprint,
        evaluator_only=True,
        audited_tasks=len(ordered),
        ready_tasks=ready_tasks,
        ready=ready_tasks == len(ordered),
        rows=ordered,
        network_calls_performed=False,
    )


def audit_mechanism_capacity(
    taskset: EvaluationTaskSet,
    references: list[GoldPatchReference],
    *,
    policy: MechanismCapacityPolicy,
) -> MechanismCapacityAudit:
    """Reconcile gold fixes against the frozen structured and hunk contracts.

    Gold content is used only inside this evaluator. The serialized result keeps
    hashes and aggregate capacity measurements, never source or patch text.
    """
    taskset.validate()
    policy.validate()
    if not references:
        raise ContractError("mechanism capacity audit requires references")
    task_ids = [row.task_id for row in references]
    if len(set(task_ids)) != len(task_ids):
        raise ContractError("mechanism capacity audit task ids must be unique")
    tasks = {row.task_id: row for row in taskset.tasks}
    unknown = sorted(set(task_ids) - set(tasks))
    if unknown:
        raise ContractError(f"mechanism capacity audit has unknown task: {unknown[0]}")

    count_tokens = _token_counter(policy.tokenizer_path)
    rows: list[MechanismCapacityRow] = []
    for reference in references:
        patch_path = reference.patch_path.resolve()
        if not patch_path.is_file():
            raise ContractError(
                f"mechanism capacity reference patch is unavailable: "
                f"{reference.task_id}"
            )
        try:
            patch_text = patch_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ContractError(
                f"mechanism capacity reference patch is unavailable: "
                f"{reference.task_id}"
            ) from exc
        implementation = tuple(
            file_patch
            for file_patch in _parse_file_patches(patch_text)
            if file_patch.target != "/dev/null"
            and _is_implementation_target(file_patch.target)
        )
        if not implementation:
            raise ContractError(
                f"mechanism capacity reference has no implementation target: "
                f"{reference.task_id}"
            )
        task = tasks[reference.task_id]
        targets = tuple(sorted(file_patch.target for file_patch in implementation))
        allowed = tuple(sorted(task.allowed_targets))
        missing = tuple(sorted(set(targets) - set(allowed)))
        hunk_count = sum(len(file_patch.hunks) for file_patch in implementation)
        changed_lines = sum(
            sum(
                line.startswith(("+", "-"))
                for hunk in file_patch.hunks
                for line in hunk.lines
            )
            for file_patch in implementation
        )
        patch_lines = sum(
            2
            + len(file_patch.hunks)
            + sum(len(hunk.lines) for hunk in file_patch.hunks)
            for file_patch in implementation
        )

        structured_tokens: int | None = None
        unique_search = False
        if len(implementation) == 1 and not missing:
            file_patch = implementation[0]
            source_path = task.source_repository / file_patch.target
            try:
                before = source_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ContractError(
                    f"mechanism capacity task source is unavailable: "
                    f"{reference.task_id}"
                ) from exc
            after = _apply_file_patch(before, file_patch)
            search, replace, unique_search = _shortest_unique_edit(before, after)
            if unique_search:
                minimum_object = json.dumps(
                    {
                        "file": file_patch.target,
                        "search": search,
                        "replace": replace,
                        "diagnostic": "x",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                structured_tokens = count_tokens(minimum_object)

        structured_ready = bool(
            not missing
            and len(implementation) == 1
            and unique_search
            and structured_tokens is not None
            and structured_tokens <= policy.structured_max_tokens
        )
        hunk_ready = bool(
            not missing
            and len(implementation) == 1
            and hunk_count <= policy.hunk_max_hunks
            and changed_lines <= policy.hunk_max_changed_lines
            and patch_lines <= policy.hunk_max_lines
        )
        reasons: list[str] = []
        if missing:
            reasons.append("reference-target-not-allowed")
        if len(implementation) != 1:
            reasons.append("implementation-file-count")
        if not unique_search:
            reasons.append("structured-search-not-unique")
        if (
            structured_tokens is not None
            and structured_tokens > policy.structured_max_tokens
        ):
            reasons.append("structured-token-budget")
        if hunk_count > policy.hunk_max_hunks:
            reasons.append("hunk-count")
        if changed_lines > policy.hunk_max_changed_lines:
            reasons.append("hunk-changed-lines")
        if patch_lines > policy.hunk_max_lines:
            reasons.append("hunk-total-lines")
        ready = structured_ready and hunk_ready
        rows.append(
            MechanismCapacityRow(
                task_id=reference.task_id,
                reference_patch_sha256=_file_sha256(patch_path),
                implementation_targets=targets,
                allowed_targets=allowed,
                missing_targets=missing,
                implementation_hunks=hunk_count,
                implementation_changed_lines=changed_lines,
                implementation_patch_lines=patch_lines,
                structured_output_tokens=structured_tokens,
                structured_unique_search=unique_search,
                structured_ready=structured_ready,
                hunk_ready=hunk_ready,
                reasons=tuple(reasons),
                ready=ready,
            )
        )
    ordered = tuple(sorted(rows, key=lambda row: row.task_id))
    ready_tasks = sum(row.ready for row in ordered)
    return MechanismCapacityAudit(
        schema_version=1,
        taskset_id=taskset.taskset_id,
        taskset_fingerprint=taskset.fingerprint,
        evaluator_only=True,
        policy=policy,
        audited_tasks=len(ordered),
        ready_tasks=ready_tasks,
        ready=ready_tasks == len(ordered),
        rows=ordered,
        network_calls_performed=False,
    )
