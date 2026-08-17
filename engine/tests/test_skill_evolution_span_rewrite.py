from pathlib import Path

import pytest

from skill_evolution_loop.contracts import ContractError
from skill_evolution_loop.span_rewrite import (
    SpanBundlePlan,
    SpanEditIntent,
    SpanOperation,
    SpanPlan,
    materialize_span_bundle,
    materialize_span_plan,
    run_span_bundle_renderer_qualification,
    run_span_renderer_qualification,
)


def _plan(*operations: SpanOperation, file: str = "src/example.ts") -> SpanPlan:
    return SpanPlan(
        schema_version=1,
        file=file,
        intent=SpanEditIntent(
            defect="wrong boundary",
            trigger="edge case",
            desired_boundary="preserve the valid branch",
        ),
        operations=operations,
        diagnostic="bounded exact-span rewrite",
    )


def test_span_renderer_applies_unique_non_overlapping_edits() -> None:
    source = "const low = 1;\nconst high = 2;\n"
    plan = _plan(
        SpanOperation(before="low = 1", after="low = 0"),
        SpanOperation(before="high = 2", after="high = 3"),
    )

    result = materialize_span_plan(source, plan)

    assert result.accepted is True
    assert result.after == "const low = 0;\nconst high = 3;\n"
    assert all(result.gates.values())


def test_span_renderer_supports_exact_deletion() -> None:
    source = "before();\nremove_me();\nafter();\n"

    result = materialize_span_plan(
        source,
        _plan(SpanOperation(before="remove_me();\n", after="")),
    )

    assert result.accepted is True
    assert result.after == "before();\nafter();\n"


def test_span_renderer_rejects_ambiguous_overlap_and_noop() -> None:
    with pytest.raises(ContractError, match="exactly once"):
        materialize_span_plan(
            "value = 1\nvalue = 1\n",
            _plan(SpanOperation(before="value = 1", after="value = 2")),
        )
    with pytest.raises(ContractError, match="overlap"):
        materialize_span_plan(
            "const value = 1;\n",
            _plan(
                SpanOperation(before="const value = 1", after="const value = 2"),
                SpanOperation(before="value = 1", after="value = 3"),
            ),
        )
    with pytest.raises(ContractError, match="must change"):
        SpanOperation(before="same", after="same").validate()


def test_span_operation_rejects_more_than_600_combined_characters() -> None:
    with pytest.raises(ContractError, match="600 characters"):
        SpanOperation(before="a" * 301, after="b" * 300).validate()


def test_span_plan_rejects_unsafe_or_unknown_target() -> None:
    for target in ("../src/a.js", "/tmp/a.js", "src/a.py", "tests/a.js"):
        with pytest.raises(ContractError):
            _plan(SpanOperation(before="a", after="b"), file=target).validate()


def test_span_bundle_materializes_two_files_atomically() -> None:
    left = _plan(
        SpanOperation(before="low = 1", after="low = 0"),
        file="src/left.ts",
    )
    right = _plan(
        SpanOperation(before="high = 2", after="high = 3"),
        file="src/right.ts",
    )
    bundle = SpanBundlePlan(
        schema_version=1,
        plans=(left, right),
        diagnostic="two-file invariant",
    )

    result = materialize_span_bundle(
        {
            "src/left.ts": "const low = 1;\n",
            "src/right.ts": "const high = 2;\n",
        },
        bundle,
    )

    assert result.accepted is True
    assert dict(result.results)["src/left.ts"].after == "const low = 0;\n"
    assert dict(result.results)["src/right.ts"].after == "const high = 3;\n"


def test_span_bundle_canonicalizes_missing_file_diagnostic() -> None:
    plan = _plan(SpanOperation(before="low = 1", after="low = 0"))
    plan_data = plan.to_dict()
    del plan_data["diagnostic"]

    bundle = SpanBundlePlan.from_dict(
        {
            "schema_version": 1,
            "plans": [plan_data],
            "diagnostic": "shared bundle diagnosis",
        }
    )

    assert bundle.plans[0].diagnostic == "shared bundle diagnosis"
    assert bundle.to_dict()["plans"][0]["diagnostic"] == ("shared bundle diagnosis")


def test_span_bundle_rejects_duplicate_or_more_than_two_files() -> None:
    plan = _plan(SpanOperation(before="a", after="b"))
    with pytest.raises(ContractError, match="unique"):
        SpanBundlePlan(1, (plan, plan), "duplicate").validate()
    with pytest.raises(ContractError, match="one or two"):
        SpanBundlePlan(1, (plan, plan, plan), "too many").validate()


def test_multilanguage_renderer_qualification_is_complete(tmp_path: Path) -> None:
    result = run_span_renderer_qualification(tmp_path / "QUALIFICATION.json")

    assert result["status"] == "qualified"
    assert result["accepted_cases"] == result["planned_cases"] == 27
    assert result["supported_languages"] == [
        "c",
        "c++",
        "go",
        "java",
        "javascript",
        "php",
        "ruby",
        "rust",
        "typescript",
    ]
    assert result["scope"] == "renderer_capacity_only_not_student_or_skill_capability"


def test_multilanguage_two_file_bundle_qualification_is_complete(
    tmp_path: Path,
) -> None:
    result = run_span_bundle_renderer_qualification(
        tmp_path / "BUNDLE-QUALIFICATION.json"
    )

    assert result["status"] == "qualified"
    assert result["accepted_cases"] == result["planned_cases"] == 9
    assert result["max_bundle_files"] == 2
    assert result["atomic_apply_required"] is True
    (materialize_span_bundle,)
