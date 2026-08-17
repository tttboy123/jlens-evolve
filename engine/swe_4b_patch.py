"""Compatibility CLI for the legacy Qwen3.5-4B patch experiment.

New feedback-loop orchestration lives in :mod:`skill_evolution_loop`.  The
legacy modes remain importable so existing runbooks and regression tests keep
working while this file stays a deliberately thin command-line boundary.
"""

from skill_evolution_loop.legacy_swe_4b import (
    MODEL,
    _clean_keywords,
    _pick_target_files,
    apply_and_diff,
    apply_and_diff_hunk,
    edits_to_diff,
    generate_edit,
    generate_hunk,
    generate_patch,
    main,
)

__all__ = [
    "MODEL",
    "_clean_keywords",
    "_pick_target_files",
    "apply_and_diff",
    "apply_and_diff_hunk",
    "edits_to_diff",
    "generate_edit",
    "generate_hunk",
    "generate_patch",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
