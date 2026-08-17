#!/usr/bin/env python3
"""Freeze one feedback-only inactive PatternCard trigger refinement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skill_evolution_loop.contracts import ContractError, LoopRevision, canonical_json
from skill_evolution_loop.pattern_card_revision import refine_pattern_card_symptoms


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("parent PatternCard revision is unreadable") from exc
    if not isinstance(value, dict):
        raise ContractError("parent PatternCard revision must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--source-evidence-sha256", required=True)
    parser.add_argument("--revision-id", required=True)
    parser.add_argument("--card-number", required=True, type=int)
    parser.add_argument("--symptom", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    wrapper = _load(args.parent)
    parent = LoopRevision.from_dict(wrapper.get("next_revision"))
    report = refine_pattern_card_symptoms(
        parent=parent,
        replacements={args.card_number: args.symptom},
        source_evidence_sha256=args.source_evidence_sha256,
        revision_id=args.revision_id,
    )
    if args.output.exists():
        if _load(args.output) != report:
            raise ContractError("existing PatternCard refinement does not match")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
