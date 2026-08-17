"""Structured reasoning-to-edit adapter with fail-closed patch construction."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ContractError, LoopRevision

_COHORTS = {"feedback", "holdout"}
_EDIT_FIELDS = frozenset({"file", "search", "replace", "diagnostic"})


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _implementation_fingerprint(file: str, source: str) -> str:
    """Hash executable structure, ignoring Python comments and formatting."""

    if Path(file).suffix == ".py":
        try:
            semantic = ast.dump(ast.parse(source), include_attributes=False)
        except SyntaxError as exc:
            raise ContractError(
                "structurally valid Python output is not parseable"
            ) from exc
    else:
        semantic = source
    return _sha256_text(semantic)


def parse_unresolved_abstention(raw: str) -> str | None:
    """Return a bounded diagnostic for the one explicit fail-closed sentinel."""

    candidate = raw.strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "status",
        "diagnostic",
    }:
        return None
    diagnostic = value.get("diagnostic")
    if (
        value.get("schema_version") != 1
        or value.get("status") != "unresolved"
        or not isinstance(diagnostic, str)
        or not diagnostic.strip()
        or len(diagnostic) > 500
    ):
        return None
    return diagnostic.strip()


@dataclass(frozen=True)
class StudentTask:
    """One isolated task checkout and its role in the feedback experiment."""

    task_id: str
    checkout: Path
    instruction: str
    allowed_targets: tuple[str, ...]
    cohort: str

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        checkout: Path,
        instruction: str,
        allowed_targets: list[str],
        cohort: str,
    ) -> StudentTask:
        task = cls(
            task_id=task_id,
            checkout=checkout.resolve(),
            instruction=instruction,
            allowed_targets=tuple(allowed_targets),
            cohort=cohort,
        )
        task.validate()
        return task

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.task_id):
            raise ContractError("invalid student task id")
        if not self.checkout.is_dir():
            raise ContractError("student checkout is missing")
        if not self.instruction.strip():
            raise ContractError("student instruction must be non-empty")
        if self.cohort not in _COHORTS:
            raise ContractError("student cohort must be feedback or holdout")
        if len(set(self.allowed_targets)) != len(self.allowed_targets):
            raise ContractError("allowed targets must be unique")
        for target in self.allowed_targets:
            self.resolve_target(target)

    def resolve_target(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ContractError("student target escapes checkout")
        resolved = (self.checkout / path).resolve()
        if not resolved.is_relative_to(self.checkout) or not resolved.is_file():
            raise ContractError("student target is not a checkout file")
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "checkout": str(self.checkout),
            "instruction": self.instruction,
            "allowed_targets": list(self.allowed_targets),
            "cohort": self.cohort,
        }


@dataclass(frozen=True)
class StructuredEdit:
    file: str
    search: str
    replace: str
    diagnostic: str

    @classmethod
    def from_model_output(cls, raw: str) -> StructuredEdit:
        candidate = raw.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
        if fenced:
            candidate = fenced.group(1)
        else:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start < 0 or end <= start:
                raise ContractError("student output contains no edit object")
            candidate = candidate[start : end + 1]
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ContractError("student edit object is malformed JSON") from exc
        if not isinstance(data, dict) or set(data) != _EDIT_FIELDS:
            raise ContractError("student edit object fields are invalid")
        edit = cls(
            file=str(data["file"]),
            search=str(data["search"]),
            replace=str(data["replace"]),
            diagnostic=str(data["diagnostic"]),
        )
        edit.validate()
        return edit

    def validate(self) -> None:
        if (
            not self.file.strip()
            or not self.search.strip()
            or not self.diagnostic.strip()
        ):
            raise ContractError("student edit fields must be non-empty")
        if self.search == self.replace:
            raise ContractError("student search and replacement are identical")

    def to_dict(self) -> dict[str, str]:
        return {
            "file": self.file,
            "search": self.search,
            "replace": self.replace,
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True)
class StudentAttempt:
    task: StudentTask
    revision_id: str
    raw_output: str
    raw_output_sha256: str
    edit: StructuredEdit | None
    patch: str
    patch_sha256: str | None
    target_file: str | None
    before_sha256: str | None
    after_sha256: str | None
    implementation_fingerprint: str | None
    structural_valid: bool
    failure_reason: str | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "revision_id": self.revision_id,
            "raw_output_sha256": self.raw_output_sha256,
            "edit": self.edit.to_dict() if self.edit else None,
            "patch": self.patch,
            "patch_sha256": self.patch_sha256,
            "target_file": self.target_file,
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
            "implementation_fingerprint": self.implementation_fingerprint,
            "structural_valid": self.structural_valid,
            "failure_reason": self.failure_reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class MajorityResult:
    """Outcome of a multi-sample majority vote over one A/B arm."""

    attempt: StudentAttempt
    samples: int
    votes: dict[str, int]
    selected_seed: int | None


StudentGenerator = Callable[[StudentTask, LoopRevision], str]


class StudentAdapter:
    """Turn one model response into an auditable, non-test unified diff."""

    def __init__(self, *, generator: StudentGenerator) -> None:
        self.generator = generator

    def experiment_config(self) -> dict[str, Any]:
        configured = getattr(self.generator, "generation_config", None)
        return {
            "adapter": type(self).__name__,
            "adapter_contract": "structured-search-replace-v1",
            "generator": configured() if configured is not None else {},
        }

    def run(self, task: StudentTask, revision: LoopRevision) -> StudentAttempt:
        task.validate()
        revision.validate()
        try:
            raw = self.generator(task, revision)
        except Exception as exc:
            return self._failure(
                task, revision, "", "eval-infra", f"generator failed: {exc}"
            )
        if not isinstance(raw, str):
            return self._failure(
                task, revision, "", "eval-infra", "generator returned non-text"
            )
        preliminary = self._classify_unstructured(raw)
        if preliminary is not None:
            return self._failure(task, revision, raw, preliminary, preliminary)
        try:
            edit = StructuredEdit.from_model_output(raw)
        except ContractError as exc:
            return self._failure(task, revision, raw, "malformed-hunk", str(exc))

        allowed = task.allowed_targets or self._discover_targets(task)
        if edit.file not in allowed or self._is_test_path(edit.file):
            return self._failure(
                task,
                revision,
                raw,
                "wrong-target",
                f"target is not allowed: {edit.file}",
                edit=edit,
            )
        try:
            target = task.resolve_target(edit.file)
            before = target.read_text(encoding="utf-8")
        except (ContractError, OSError, UnicodeError) as exc:
            return self._failure(
                task, revision, raw, "wrong-target", str(exc), edit=edit
            )
        occurrences = before.count(edit.search)
        if occurrences != 1:
            return self._failure(
                task,
                revision,
                raw,
                "apply-fail",
                f"search span must match exactly once; matches={occurrences}",
                edit=edit,
            )
        after = before.replace(edit.search, edit.replace, 1)
        patch = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{edit.file}",
                tofile=f"b/{edit.file}",
            )
        )
        if not patch.strip() or not self._git_apply_check(task.checkout, patch):
            return self._failure(
                task,
                revision,
                raw,
                "apply-fail",
                "constructed patch does not apply cleanly",
                edit=edit,
            )
        return StudentAttempt(
            task=task,
            revision_id=revision.revision_id,
            raw_output=raw,
            raw_output_sha256=_sha256_text(raw),
            edit=edit,
            patch=patch,
            patch_sha256=_sha256_text(patch),
            target_file=edit.file,
            before_sha256=_sha256_text(before),
            after_sha256=_sha256_text(after),
            implementation_fingerprint=_implementation_fingerprint(edit.file, after),
            structural_valid=True,
            failure_reason=None,
            detail=edit.diagnostic,
        )

    def run_majority(
        self,
        task: StudentTask,
        revision: LoopRevision,
        *,
        samples: int = 3,
        seed_base: int = 0,
    ) -> MajorityResult:
        """Generate N deterministic samples and select the majority patch.

        Only structural-valid attempts vote; the patch SHA256 is the ballot.
        Ties keep the first sample's patch so results are stable. The selected
        seed is the seed of the first winning sample.
        """
        count = max(1, samples)
        attempts: list[StudentAttempt] = []
        for index in range(count):
            if hasattr(self.generator, "seed"):
                try:
                    self.generator.seed = seed_base + index
                except Exception:
                    pass
            attempts.append(self.run(task, revision))
        ballots: list[tuple[int, StudentAttempt]] = []
        for index, attempt in enumerate(attempts):
            if attempt.structural_valid and attempt.patch_sha256:
                ballots.append((index, attempt))
        if not ballots:
            return MajorityResult(attempts[0], count, {}, None)
        tallies: dict[str, int] = {}
        for _index, attempt in ballots:
            tallies[attempt.patch_sha256] = tallies.get(attempt.patch_sha256, 0) + 1
        first_index_of: dict[str, int] = {}
        for index, attempt in ballots:
            first_index_of.setdefault(attempt.patch_sha256, index)
        winner_sha = max(
            tallies,
            key=lambda key: (tallies[key], -first_index_of[key]),
        )
        winner = next(
            attempt for _i, attempt in ballots if attempt.patch_sha256 == winner_sha
        )
        return MajorityResult(
            winner,
            count,
            dict(sorted(tallies.items())),
            seed_base + first_index_of[winner_sha],
        )

    def apply(self, attempt: StudentAttempt) -> None:
        if not attempt.structural_valid or not attempt.patch:
            raise ContractError("cannot apply an invalid student attempt")
        if not self._git_apply_check(attempt.task.checkout, attempt.patch):
            raise ContractError("student patch no longer applies cleanly")
        completed = subprocess.run(
            ["git", "apply", "-"],
            cwd=attempt.task.checkout,
            input=attempt.patch,
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            raise ContractError(f"student patch application failed: {completed.stderr}")

    @staticmethod
    def _git_apply_check(checkout: Path, patch: str) -> bool:
        completed = subprocess.run(
            ["git", "apply", "--check", "-"],
            cwd=checkout,
            input=patch,
            capture_output=True,
            check=False,
            text=True,
        )
        return completed.returncode == 0

    @staticmethod
    def _is_test_path(relative: str) -> bool:
        path = Path(relative)
        lowered = [part.lower() for part in path.parts]
        return bool(
            {"test", "tests", "__tests__"}.intersection(lowered)
            or ".test." in path.name.lower()
            or ".spec." in path.name.lower()
            or path.name.lower().startswith("test_")
        )

    @staticmethod
    def _classify_unstructured(raw: str) -> str | None:
        stripped = raw.strip()
        if not stripped:
            return "no-diff"
        if "```file" in stripped:
            return "full-file"
        if "--- a/" in stripped:
            if "+++ b/" not in stripped:
                return "bad-header"
            if "@@" not in stripped:
                return "malformed-hunk"
        if "{" not in stripped or "}" not in stripped:
            return "reasoning-only"
        return None

    @staticmethod
    def _discover_targets(task: StudentTask) -> tuple[str, ...]:
        from .legacy_swe_4b import _pick_target_files

        targets, _ = _pick_target_files(task.checkout, task.instruction)
        return tuple(path.relative_to(task.checkout).as_posix() for path in targets)

    @staticmethod
    def _failure(
        task: StudentTask,
        revision: LoopRevision,
        raw: str,
        reason: str,
        detail: str,
        *,
        edit: StructuredEdit | None = None,
        patch: str = "",
        target_file: str | None = None,
        before_sha256: str | None = None,
    ) -> StudentAttempt:
        return StudentAttempt(
            task=task,
            revision_id=revision.revision_id,
            raw_output=raw,
            raw_output_sha256=_sha256_text(raw),
            edit=edit,
            patch=patch,
            patch_sha256=_sha256_text(patch) if patch else None,
            target_file=target_file or (edit.file if edit else None),
            before_sha256=before_sha256,
            after_sha256=None,
            implementation_fingerprint=None,
            structural_valid=False,
            failure_reason=reason,
            detail=detail,
        )


class HunkStudentAdapter(StudentAdapter):
    """Validate one model-produced unified diff under the same attempt contract."""

    def experiment_config(self) -> dict[str, Any]:
        config = super().experiment_config()
        return {
            **config,
            "adapter_contract": "unified-diff-hunk-v2",
            "limits": {"max_hunks": 3, "max_changed_lines": 50, "max_lines": 80},
        }

    def run(self, task: StudentTask, revision: LoopRevision) -> StudentAttempt:
        from .legacy_swe_4b import _extract_hunk

        task.validate()
        revision.validate()
        try:
            raw = self.generator(task, revision)
        except Exception as exc:
            return self._failure(
                task, revision, "", "eval-infra", f"generator failed: {exc}"
            )
        if not isinstance(raw, str):
            return self._failure(
                task, revision, "", "eval-infra", "generator returned non-text"
            )
        if not raw.strip():
            return self._failure(task, revision, raw, "no-diff", "empty output")
        if "```file" in raw:
            return self._failure(
                task, revision, raw, "full-file", "full-file output is prohibited"
            )
        patch, reason = _extract_hunk(raw)
        if patch is None:
            if reason == "no-diff":
                reason = "reasoning-only"
            return self._failure(
                task, revision, raw, reason or "no-diff", reason or "no-diff"
            )

        old_paths = re.findall(r"^--- a/([^\n]+)$", patch, re.MULTILINE)
        new_paths = re.findall(r"^\+\+\+ b/([^\n]+)$", patch, re.MULTILINE)
        if len(old_paths) != 1 or old_paths != new_paths:
            return self._failure(
                task,
                revision,
                raw,
                "wrong-target",
                "hunk output must edit exactly one matching file",
                patch=patch,
            )
        relative = old_paths[0]
        allowed = task.allowed_targets or self._discover_targets(task)
        if relative not in allowed or self._is_test_path(relative):
            return self._failure(
                task,
                revision,
                raw,
                "wrong-target",
                f"target is not allowed: {relative}",
                patch=patch,
                target_file=relative,
            )
        try:
            before = task.resolve_target(relative).read_text(
                encoding="utf-8", errors="replace"
            )
        except (ContractError, OSError) as exc:
            return self._failure(
                task,
                revision,
                raw,
                "wrong-target",
                str(exc),
                patch=patch,
                target_file=relative,
            )
        lines = patch.splitlines()
        hunk_count = sum(line.startswith("@@") for line in lines)
        changed_lines = sum(
            line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
            for line in lines
        )
        if hunk_count > 3 or changed_lines > 50 or len(lines) > 80:
            return self._failure(
                task,
                revision,
                raw,
                "malformed-hunk",
                "hunk output exceeds protocol limits",
                patch=patch,
                target_file=relative,
                before_sha256=_sha256_text(before),
            )
        if not self._git_apply_check(task.checkout, patch):
            return self._failure(
                task,
                revision,
                raw,
                "apply-fail",
                "extracted hunk does not apply cleanly",
                patch=patch,
                target_file=relative,
                before_sha256=_sha256_text(before),
            )
        return StudentAttempt(
            task=task,
            revision_id=revision.revision_id,
            raw_output=raw,
            raw_output_sha256=_sha256_text(raw),
            edit=None,
            patch=patch,
            patch_sha256=_sha256_text(patch),
            target_file=relative,
            before_sha256=_sha256_text(before),
            after_sha256=None,
            implementation_fingerprint=None,
            structural_valid=True,
            failure_reason=None,
            detail="validated unified diff",
        )
