"""Append-only persistence for evolution events and reusable lessons."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
    return rows


class ExperienceStore:
    """Durable project-scoped archive with idempotent append semantics."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "archive.jsonl"
        self.lessons_path = self.root / "lessons.jsonl"
        self.skills_dir = self.root / "skills"

    @staticmethod
    def _append_unique(path: Path, row: dict[str, Any], id_key: str) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            seen = {
                parsed.get(id_key)
                for line in handle
                if line.strip()
                for parsed in (json.loads(line),)
            }
            if row[id_key] in seen:
                return False
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return True

    def append_event(self, event: dict[str, Any]) -> bool:
        row = dict(event)
        row.setdefault("schema_version", SCHEMA_VERSION)
        row.setdefault("recorded_at", datetime.now(UTC).isoformat())
        row.setdefault("event_id", _json_hash(row))
        return self._append_unique(self.events_path, row, "event_id")

    def read_events(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.events_path)

    def read_lessons(self) -> list[dict[str, Any]]:
        """Return the latest append-only revision of every lesson."""
        latest: dict[str, dict[str, Any]] = {}
        for row in _read_jsonl(self.lessons_path):
            latest[row["lesson_id"]] = row
        return list(latest.values())

    def distill_lessons(self, *, min_evidence: int = 2) -> int:
        """Promote repeatedly holdout-verified improvements into lesson records."""
        if min_evidence < 1:
            raise ValueError("min_evidence must be at least one")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in self.read_events():
            lesson = event.get("lesson")
            if (
                not event.get("accepted")
                or not event.get("holdout_verified")
                or not lesson
            ):
                continue
            key = _json_hash(
                {
                    "task_family": event.get("task_family"),
                    "lesson": lesson,
                    "tags": sorted(event.get("tags", [])),
                }
            )
            grouped[key].append(event)

        existing = {row["lesson_id"]: row for row in self.read_lessons()}
        promoted = 0
        for lesson_id, evidence in grouped.items():
            source_tasks = sorted({str(row.get("task_id")) for row in evidence})
            evidence_ids = sorted({str(row["event_id"]) for row in evidence})
            if len(evidence_ids) < min_evidence:
                continue
            exemplar = evidence[-1]
            row = {
                "schema_version": SCHEMA_VERSION,
                "lesson_id": lesson_id,
                "revision_id": _json_hash([lesson_id, evidence_ids]),
                "task_family": exemplar.get("task_family"),
                "lesson": exemplar["lesson"],
                "tags": sorted(set(exemplar.get("tags", []))),
                "evidence_count": len(evidence_ids),
                "evidence_event_ids": evidence_ids,
                "source_task_ids": source_tasks,
                "updated_at": datetime.now(UTC).isoformat(),
                "status": "candidate",
            }
            old = existing.get(lesson_id)
            if old and old.get("revision_id") == row["revision_id"]:
                continue
            # Revisions are append-only and are keyed separately from the stable
            # lesson id.  Readers collapse them to the latest revision.
            self._append_unique(self.lessons_path, row, "revision_id")
            promoted += 1
        return promoted

    def retrieve_lessons(
        self,
        *,
        task_family: str,
        tags: Iterable[str] = (),
        exclude_task_id: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve validated family-compatible lessons with deterministic ranking."""
        wanted = set(tags)
        candidates = []
        for lesson in self.read_lessons():
            if lesson.get("task_family") != task_family:
                continue
            sources = set(lesson.get("source_task_ids", []))
            if exclude_task_id is not None and sources == {exclude_task_id}:
                continue
            overlap = len(wanted.intersection(lesson.get("tags", [])))
            candidates.append((overlap, int(lesson.get("evidence_count", 0)), lesson))
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]["lesson_id"]))
        return [item[2] for item in candidates[:limit]]

    def render_skill_candidate(self, task_family: str) -> Path:
        """Render a reviewable local skill candidate; never install it globally."""
        lessons = self.retrieve_lessons(task_family=task_family, limit=20)
        slug = re.sub(r"[^a-z0-9]+", "-", task_family.lower()).strip("-") or "evolve"
        skill_dir = self.skills_dir / slug
        skill_dir.mkdir(parents=True, exist_ok=True)
        path = skill_dir / "SKILL.md"
        lines = [
            "---",
            f"name: evolve-{slug}",
            f"description: Holdout-verified candidate lessons for {task_family}.",
            "status: candidate",
            "---",
            "",
            f"# Evolve lessons: {task_family}",
            "",
            "This file is project-scoped and requires human review before installation.",
            "",
        ]
        if lessons:
            for lesson in lessons:
                lines.extend(
                    [
                        f"- {lesson['lesson']}",
                        (
                            f"  Evidence: {lesson['evidence_count']} verified events; "
                            f"tasks {', '.join(lesson['source_task_ids'])}."
                        ),
                    ]
                )
        else:
            lines.append("No lesson has met the promotion threshold yet.")
        content = "\n".join(lines) + "\n"
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=skill_dir, delete=False
        ) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        temporary.replace(path)
        return path
