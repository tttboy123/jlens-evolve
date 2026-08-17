"""Offline regression gates for localization recall and selector harvest.

Golden fix locations are used ONLY as a dev-side measurement oracle; none of
these fixtures are ever injected into a Student prompt.
"""

from __future__ import annotations

from skill_evolution_loop.operator_student import _harvest_typed_selector_candidates
from skill_evolution_loop.symbol_rewrite import (
    qualified_symbol_candidates,
    qualified_symbol_for_issue,
)


def test_symbol_candidates_include_zero_overlap_golden_site() -> None:
    # The issue text shares zero lexical tokens with the real fix site
    # (``visit_literal``); single-symbol localization would miss it.
    source = (
        "class InlineWrapper:\n"
        "    def render(self):\n"
        "        return self.escape(self.body)\n"
        "    def visit_literal(self, node):\n"
        "        return node.astext()\n"
        "    def visit_comment(self, node):\n"
        "        return node.astext()\n"
    )
    instruction = (
        "Inline TeX wrapper boundaries around percent-newline breaks are wrong."
    )
    candidates = qualified_symbol_candidates(source, instruction, top_n=4)
    assert "InlineWrapper.visit_literal" in candidates
    # Best-first ordering must not drop the golden site behind unrelated defs.
    assert candidates[0] in ("InlineWrapper.visit_literal", "InlineWrapper.render")


def test_symbol_candidates_default_top_n() -> None:
    source = (
        "class Alpha:\n"
        "    def first(self):\n"
        "        return 1\n"
        "    def second(self):\n"
        "        return 2\n"
    )
    candidates = qualified_symbol_candidates(source, "second item must be returned")
    assert candidates[0] == "Alpha.second"
    assert "Alpha" in candidates


def test_single_symbol_api_still_returns_best_or_none() -> None:
    source = (
        "class Alpha:\n"
        "    def first(self):\n"
        "        return 1\n"
        "    def second(self):\n"
        "        return 2\n"
    )
    assert qualified_symbol_for_issue(source, "second item must be returned") == (
        "Alpha.second"
    )
    assert qualified_symbol_for_issue(
        source, "unrelated prose with no code anchor"
    ) in (
        "Alpha.second",
        "Alpha.first",
        None,
    )


def test_harvest_ranks_by_issue_line_proximity() -> None:
    source = (
        "def visit_literal(self, node):\n"
        "    self.body.append(node.astext())\n"
        "    hlcode = node.highlight()\n"
        "    trailer = self.tail\n"
    )
    candidates = _harvest_typed_selector_candidates(
        source, instruction="hlcode highlighting must preserve its output"
    )
    by_source = {row["source"]: row for row in candidates}
    assert by_source["hlcode = node.highlight()"]["issue_line_proximity"] == 0
    assert by_source["trailer = self.tail"]["issue_line_proximity"] == 1
    order = [row["source"] for row in candidates]
    assert order.index("hlcode = node.highlight()") < order.index("trailer = self.tail")


def test_harvest_default_candidate_limit_is_128() -> None:
    source = "def many():\n" + "".join(f"    v{i} = {i}\n" for i in range(200))
    candidates = _harvest_typed_selector_candidates(source, instruction="assign all")
    assert len(candidates) == 128
