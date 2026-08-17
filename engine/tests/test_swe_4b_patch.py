"""Regression tests for swe_4b_patch target selection (v2.5 Qwen teaching)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swe_4b_patch import _clean_keywords, _pick_target_files

CHECKOUTS = Path("/private/tmp/qwen-sphinx")
PROBLEMS_PATH = CHECKOUTS / "problems.json"
PROBLEMS = (
    json.loads(PROBLEMS_PATH.read_text(encoding="utf-8"))
    if PROBLEMS_PATH.is_file()
    else {}
)

GOLD = {
    "sphinx-doc__sphinx-7757": ("7757", "sphinx/util/inspect.py"),
    "sphinx-doc__sphinx-9658": ("9658", "sphinx/ext/autodoc/mock.py"),
    "sphinx-doc__sphinx-10435": ("10435", "sphinx/writers/latex.py"),
}


@pytest.mark.skipif(
    not PROBLEMS_PATH.is_file(),
    reason="optional /private/tmp/qwen-sphinx integration fixture is unavailable",
)
def test_target_selection_hits_gold_fix_files() -> None:
    """The picker must surface the real fix source (not theme JS/templates)."""
    for inst, (task, gold_fix) in GOLD.items():
        checkout = CHECKOUTS / f"sphinx-{task}"
        targets, _context = _pick_target_files(
            checkout, PROBLEMS[inst]["problem_statement"]
        )
        rels = [str(path.relative_to(checkout)) for path in targets]
        assert gold_fix in rels, f"{inst}: expected {gold_fix}, got {rels}"


def test_keyword_cleaning_drops_template_words() -> None:
    kws = _clean_keywords("Describe the bug To Reproduce Build following document")
    for banned in ("describe", "reproduce", "build"):
        assert banned not in kws
