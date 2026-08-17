"""Append-only storage for loop-local revisions."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

from .contracts import ContractError, LoopRevision, canonical_json


class LoopRevisionRegistry:
    """Store immutable revisions without conflating them with promoted Skills."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "revisions.jsonl"

    def read_revisions(self) -> list[LoopRevision]:
        if not self.path.exists():
            return []
        return [
            LoopRevision.from_dict(json.loads(line))
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def latest(self, skill_id: str) -> LoopRevision | None:
        matches = [row for row in self.read_revisions() if row.skill_id == skill_id]
        return matches[-1] if matches else None

    def append(self, revision: LoopRevision) -> bool:
        revision.validate()
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            existing = [
                LoopRevision.from_dict(json.loads(line))
                for line in handle
                if line.strip()
            ]
            by_fingerprint = {row.fingerprint for row in existing}
            if revision.fingerprint in by_fingerprint:
                return False
            same_skill = [row for row in existing if row.skill_id == revision.skill_id]
            expected_parent = same_skill[-1].revision_id if same_skill else None
            if revision.parent_revision_id != expected_parent:
                raise ContractError(
                    "loop revision parent must match the latest stored revision"
                )
            if same_skill and revision.source_round <= same_skill[-1].source_round:
                raise ContractError("source_round must increase along revision lineage")
            handle.seek(0, os.SEEK_END)
            handle.write(canonical_json(revision.to_dict()) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return True
