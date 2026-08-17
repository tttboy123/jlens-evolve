from __future__ import annotations

from pathlib import Path

import pytest

from skill_evolution_loop.contracts import ContractError
from skill_evolution_loop.round1_qualification import (
    _qualification_cell_evidence_fingerprint,
    infer_repo_language,
    qualify_round1_candidates,
)
from skill_evolution_loop.target_audit import reference_implementation_targets


def test_round1_language_inference_uses_source_not_reference(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.rs").write_text("fn main() {}\n" * 20, encoding="utf-8")
    (tmp_path / "src/tiny.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/large.py").write_text("x = 1\n" * 100, encoding="utf-8")

    assert infer_repo_language(tmp_path) == "rust"


def test_reference_target_parser_excludes_tests_and_docs() -> None:
    patch = """--- a/src/main.rs
+++ b/src/main.rs
@@ -1 +1 @@
-old
+new
--- a/tests/main.rs
+++ b/tests/main.rs
@@ -1 +1 @@
-old
+new
"""

    assert reference_implementation_targets(patch) == ("src/main.rs",)


def test_round1_qualification_caps_evaluator_only_retrieval_at_64(
    tmp_path: Path,
) -> None:
    with pytest.raises(ContractError, match="candidate limit"):
        qualify_round1_candidates(
            candidate_path=tmp_path / "missing-candidates.json",
            source_summary_path=tmp_path / "missing-sources.json",
            pool_root=tmp_path,
            evidence_root=tmp_path / "evidence",
            workspace_root=tmp_path / "workspace",
            max_candidates=65,
        )


def test_qualification_fingerprint_binds_cell_identity_not_only_counts() -> None:
    first = [
        {"task_id": "task-a", "evidence_sha256": "a" * 64},
        {"task_id": "task-b", "evidence_sha256": "b" * 64},
    ]
    changed = [
        {"task_id": "task-a", "evidence_sha256": "a" * 64},
        {"task_id": "task-b", "evidence_sha256": "c" * 64},
    ]

    assert _qualification_cell_evidence_fingerprint(first) != (
        _qualification_cell_evidence_fingerprint(changed)
    )
