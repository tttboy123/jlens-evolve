"""Language-agnostic exact-span renderer for frozen Round 1 search tasks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_json, sha256_json

_LANGUAGE_EXTENSIONS = {
    "c": frozenset({".c", ".h"}),
    "c++": frozenset({".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}),
    "go": frozenset({".go"}),
    "java": frozenset({".java"}),
    "javascript": frozenset({".js", ".jsx", ".mjs", ".cjs"}),
    "php": frozenset({".php"}),
    "ruby": frozenset({".rb"}),
    "rust": frozenset({".rs"}),
    "typescript": frozenset({".ts", ".tsx", ".mts", ".cts"}),
}
_ALLOWED_EXTENSIONS = frozenset().union(*_LANGUAGE_EXTENSIONS.values())
_MAX_OPERATIONS = 4
_MAX_BUNDLE_FILES = 2
_MAX_SPAN_COMBINED_CHARS = 600
_MAX_SPAN_LINES = 80


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_test_path(path: Path) -> bool:
    lowered = {part.lower() for part in path.parts}
    name = path.name.lower()
    return bool(
        lowered.intersection({"test", "tests", "__tests__"})
        or name.startswith("test_")
        or ".test." in name
        or ".spec." in name
    )


@dataclass(frozen=True)
class SpanEditIntent:
    defect: str
    trigger: str
    desired_boundary: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpanEditIntent:
        if not isinstance(data, dict) or set(data) != {
            "defect",
            "trigger",
            "desired_boundary",
        }:
            raise ContractError("span edit intent fields are invalid")
        intent = cls(
            defect=str(data["defect"]),
            trigger=str(data["trigger"]),
            desired_boundary=str(data["desired_boundary"]),
        )
        intent.validate()
        return intent

    def validate(self) -> None:
        if any(not value.strip() for value in self.to_dict().values()):
            raise ContractError("span edit intent fields must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "defect": self.defect,
            "trigger": self.trigger,
            "desired_boundary": self.desired_boundary,
        }


@dataclass(frozen=True)
class SpanOperation:
    before: str
    after: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpanOperation:
        if not isinstance(data, dict) or set(data) != {"before", "after"}:
            raise ContractError("span operation fields are invalid")
        operation = cls(before=str(data["before"]), after=str(data["after"]))
        operation.validate()
        return operation

    def validate(self) -> None:
        if not self.before:
            raise ContractError("span operation before field must be non-empty")
        if self.before == self.after:
            raise ContractError("span operation must change source")
        if len(self.before) + len(self.after) > _MAX_SPAN_COMBINED_CHARS:
            raise ContractError("span operation exceeds 600 characters")
        for value in (self.before, self.after):
            if len(value.splitlines()) > _MAX_SPAN_LINES:
                raise ContractError("span operation exceeds its bounded size")

    def to_dict(self) -> dict[str, str]:
        return {"before": self.before, "after": self.after}


@dataclass(frozen=True)
class SpanPlan:
    schema_version: int
    file: str
    intent: SpanEditIntent
    operations: tuple[SpanOperation, ...]
    diagnostic: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpanPlan:
        if not isinstance(data, dict) or set(data) != {
            "schema_version",
            "file",
            "intent",
            "operations",
            "diagnostic",
        }:
            raise ContractError("span plan fields are invalid")
        operations = data["operations"]
        if not isinstance(operations, list):
            raise ContractError("span plan operations must be a list")
        plan = cls(
            schema_version=data["schema_version"],
            file=str(data["file"]),
            intent=SpanEditIntent.from_dict(data["intent"]),
            operations=tuple(SpanOperation.from_dict(row) for row in operations),
            diagnostic=str(data["diagnostic"]),
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported span plan schema")
        path = Path(self.file)
        if (
            path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or path.suffix.lower() not in _ALLOWED_EXTENSIONS
            or _is_test_path(path)
        ):
            raise ContractError("span plan target is not an allowlisted source path")
        self.intent.validate()
        if not 1 <= len(self.operations) <= _MAX_OPERATIONS:
            raise ContractError("span plan requires one to four operations")
        for operation in self.operations:
            operation.validate()
        if not self.diagnostic.strip():
            raise ContractError("span plan diagnostic must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "file": self.file,
            "intent": self.intent.to_dict(),
            "operations": [row.to_dict() for row in self.operations],
            "diagnostic": self.diagnostic,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class SpanBundlePlan:
    """One atomic plan spanning at most two source files."""

    schema_version: int
    plans: tuple[SpanPlan, ...]
    diagnostic: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpanBundlePlan:
        if not isinstance(data, dict) or set(data) != {
            "schema_version",
            "plans",
            "diagnostic",
        }:
            raise ContractError("span bundle fields are invalid")
        plans = data["plans"]
        if not isinstance(plans, list):
            raise ContractError("span bundle plans must be a list")
        diagnostic = str(data["diagnostic"])
        normalized_plans = []
        required_without_diagnostic = {
            "schema_version",
            "file",
            "intent",
            "operations",
        }
        for row in plans:
            if isinstance(row, dict) and set(row) == required_without_diagnostic:
                # Bundle v3 removes a redundant failure mode for small models:
                # one shared diagnosis is sufficient when a file-local duplicate
                # is omitted. The canonical plan still receives a non-empty
                # diagnostic and all other fields remain exact/fail-closed.
                row = {**row, "diagnostic": diagnostic}
            normalized_plans.append(row)
        bundle = cls(
            schema_version=data["schema_version"],
            plans=tuple(SpanPlan.from_dict(row) for row in normalized_plans),
            diagnostic=diagnostic,
        )
        bundle.validate()
        return bundle

    @classmethod
    def from_plan(cls, plan: SpanPlan) -> SpanBundlePlan:
        return cls(
            schema_version=1,
            plans=(plan,),
            diagnostic=plan.diagnostic,
        )

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported span bundle schema")
        if not 1 <= len(self.plans) <= _MAX_BUNDLE_FILES:
            raise ContractError("span bundle requires one or two files")
        files = [plan.file for plan in self.plans]
        if len(set(files)) != len(files):
            raise ContractError("span bundle file targets must be unique")
        for plan in self.plans:
            plan.validate()
        if not self.diagnostic.strip():
            raise ContractError("span bundle diagnostic must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plans": [plan.to_dict() for plan in self.plans],
            "diagnostic": self.diagnostic,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class SpanMaterializationResult:
    before: str
    after: str
    accepted: bool
    gates: dict[str, bool]
    plan_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_sha256": _sha256_text(self.before),
            "after_sha256": _sha256_text(self.after),
            "accepted": self.accepted,
            "gates": self.gates,
            "plan_fingerprint": self.plan_fingerprint,
        }


@dataclass(frozen=True)
class SpanBundleMaterializationResult:
    results: tuple[tuple[str, SpanMaterializationResult], ...]
    accepted: bool
    bundle_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "bundle_fingerprint": self.bundle_fingerprint,
            "files": {file: result.to_dict() for file, result in self.results},
        }


@dataclass(frozen=True)
class _ResolvedSpan:
    start: int
    end: int
    replacement: str


def _match_offsets(source: str, selector: str) -> list[int]:
    offsets: list[int] = []
    start = 0
    while True:
        found = source.find(selector, start)
        if found < 0:
            return offsets
        offsets.append(found)
        start = found + 1


def materialize_span_plan(source: str, plan: SpanPlan) -> SpanMaterializationResult:
    """Resolve unique exact spans against the original source and render edits."""

    plan.validate()
    resolved: list[_ResolvedSpan] = []
    for operation in plan.operations:
        offsets = _match_offsets(source, operation.before)
        if len(offsets) != 1:
            raise ContractError(
                f"span selector must match exactly once; matches={len(offsets)}"
            )
        start = offsets[0]
        resolved.append(
            _ResolvedSpan(start, start + len(operation.before), operation.after)
        )
    ordered = sorted(resolved, key=lambda row: (row.start, row.end))
    for left, right in zip(ordered, ordered[1:], strict=False):
        if right.start < left.end:
            raise ContractError("span operations overlap")
    after = source
    for edit in reversed(ordered):
        after = after[: edit.start] + edit.replacement + after[edit.end :]
    gates = {
        "selectors_unique": True,
        "non_overlapping": True,
        "bounded_change": True,
        "non_copy": after != source,
        "utf8_roundtrip": after.encode("utf-8").decode("utf-8") == after,
    }
    if not all(gates.values()):
        raise ContractError("span materialization gate rejected the plan")
    return SpanMaterializationResult(
        before=source,
        after=after,
        accepted=True,
        gates=gates,
        plan_fingerprint=plan.fingerprint,
    )


def materialize_span_bundle(
    sources: dict[str, str], bundle: SpanBundlePlan
) -> SpanBundleMaterializationResult:
    """Materialize every file before accepting the bundle atomically."""

    bundle.validate()
    results: list[tuple[str, SpanMaterializationResult]] = []
    for plan in bundle.plans:
        source = sources.get(plan.file)
        if source is None:
            raise ContractError(f"span bundle source is unavailable: {plan.file}")
        results.append((plan.file, materialize_span_plan(source, plan)))
    accepted = len(results) == len(bundle.plans) and all(
        result.accepted for _file, result in results
    )
    if not accepted:
        raise ContractError("span bundle materialization was not atomic")
    return SpanBundleMaterializationResult(
        results=tuple(results),
        accepted=True,
        bundle_fingerprint=bundle.fingerprint,
    )


def _fixture_plan(file: str, before: str, after: str) -> SpanPlan:
    return SpanPlan(
        schema_version=1,
        file=file,
        intent=SpanEditIntent("fixture defect", "fixture trigger", "fixture boundary"),
        operations=(SpanOperation(before, after),),
        diagnostic="renderer qualification fixture",
    )


def _qualification_cases() -> list[tuple[str, str, str, SpanPlan, str]]:
    fixtures = {
        "c": ("src/a.c", "int limit = 1;\n", "limit = 1", "limit = 2"),
        "c++": ("src/a.cpp", "auto limit = 1;\n", "limit = 1", "limit = 2"),
        "go": ("src/a.go", "limit := 1\n", "limit := 1", "limit := 2"),
        "java": ("src/A.java", "int limit = 1;\n", "limit = 1", "limit = 2"),
        "javascript": (
            "src/a.js",
            "const limit = 1;\n",
            "limit = 1",
            "limit = 2",
        ),
        "php": ("src/a.php", "$limit = 1;\n", "limit = 1", "limit = 2"),
        "ruby": ("src/a.rb", "limit = 1\n", "limit = 1", "limit = 2"),
        "rust": ("src/a.rs", "let limit = 1;\n", "limit = 1", "limit = 2"),
        "typescript": (
            "src/a.ts",
            "const limit: number = 1;\n",
            "number = 1",
            "number = 2",
        ),
    }
    cases: list[tuple[str, str, str, SpanPlan, str]] = []
    for language, (file, source, before, after) in fixtures.items():
        for index in range(3):
            prefix = (
                f"// fixture {index}\n" if language != "go" else f"// fixture {index}\n"
            )
            case_source = prefix + source
            plan = _fixture_plan(file, before, after)
            expected = case_source.replace(before, after)
            cases.append((f"{language}-{index}", language, case_source, plan, expected))
    return cases


def run_span_renderer_qualification(output_path: Path) -> dict[str, Any]:
    """Run and freeze the supported-language exact-span renderer capacity suite."""

    cases = _qualification_cases()
    rows: list[dict[str, Any]] = []
    language_passes: dict[str, int] = {name: 0 for name in _LANGUAGE_EXTENSIONS}
    for case_id, language, source, plan, expected in cases:
        result = materialize_span_plan(source, plan)
        passed = result.accepted and result.after == expected
        if passed:
            language_passes[language] += 1
        rows.append(
            {
                "case_id": case_id,
                "language": language,
                "source_sha256": _sha256_text(source),
                "expected_sha256": _sha256_text(expected),
                "plan_fingerprint": plan.fingerprint,
                "passed": passed,
                "result": result.to_dict(),
            }
        )
    supported = sorted(
        language for language, passed in language_passes.items() if passed == 3
    )
    accepted = sum(row["passed"] is True for row in rows)
    content = {
        "schema_version": 1,
        "suite_id": "multilanguage-exact-span-renderer-synthetic-v1",
        "renderer_contract": "multilanguage-exact-span-plan-v1",
        "status": (
            "qualified"
            if accepted == len(cases) and len(supported) == len(_LANGUAGE_EXTENSIONS)
            else "rejected"
        ),
        "planned_cases": len(cases),
        "accepted_cases": accepted,
        "supported_languages": supported,
        "language_passes": dict(sorted(language_passes.items())),
        "rows": rows,
        "scope": "renderer_capacity_only_not_student_or_skill_capability",
        "holdout_task_ids_included": False,
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    output = output_path.resolve()
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError("span renderer qualification is unreadable") from exc
        if existing != report:
            raise ContractError("frozen span qualification does not match replay")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report


def run_span_bundle_renderer_qualification(output_path: Path) -> dict[str, Any]:
    """Qualify atomic two-file rendering once for every supported language."""

    rows: list[dict[str, Any]] = []
    supported: list[str] = []
    for language, suffixes in sorted(_LANGUAGE_EXTENSIONS.items()):
        suffix = sorted(suffixes, key=lambda value: (len(value), value))[0]
        left = f"src/left{suffix}"
        right = f"src/right{suffix}"
        sources = {left: "limit = 1\n", right: "limit = 1\n"}
        bundle = SpanBundlePlan(
            schema_version=1,
            plans=(
                _fixture_plan(left, "limit = 1", "limit = 2"),
                _fixture_plan(right, "limit = 1", "limit = 2"),
            ),
            diagnostic="atomic two-file renderer fixture",
        )
        result = materialize_span_bundle(sources, bundle)
        passed = result.accepted and all(
            item.after == "limit = 2\n" for _file, item in result.results
        )
        if passed:
            supported.append(language)
        rows.append(
            {
                "case_id": f"{language}-two-file",
                "language": language,
                "source_sha256": _sha256_text(canonical_json(sources)),
                "bundle_fingerprint": bundle.fingerprint,
                "passed": passed,
                "result": result.to_dict(),
            }
        )
    content = {
        "schema_version": 1,
        "suite_id": "multilanguage-exact-span-bundle-synthetic-v2",
        "renderer_contract": "multilanguage-exact-span-bundle-v2",
        "status": "qualified"
        if len(supported) == len(_LANGUAGE_EXTENSIONS)
        else "rejected",
        "planned_cases": len(_LANGUAGE_EXTENSIONS),
        "accepted_cases": sum(row["passed"] is True for row in rows),
        "supported_languages": supported,
        "max_bundle_files": 2,
        "atomic_apply_required": True,
        "rows": rows,
        "scope": "renderer_capacity_only_not_student_or_skill_capability",
        "holdout_task_ids_included": False,
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    output = output_path.resolve()
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError("span bundle qualification is unreadable") from exc
        if existing != report:
            raise ContractError(
                "frozen span bundle qualification does not match replay"
            )
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report
