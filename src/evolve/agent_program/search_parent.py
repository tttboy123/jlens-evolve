"""Append-only search-parent advances derived from tournament decisions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from evolve.contracts import canonical_json, content_sha256

from .revision import AgentProgramViolation
from .tournament import TournamentDecision


@dataclass(frozen=True, slots=True)
class SearchParentEvent:
    sequence: int
    program_id: str
    tournament_id: str
    execution_scope: str
    previous_parent_revision_id: str
    selected_revision_id: str
    decision_sha256: str
    previous_event_sha256: str | None
    event_sha256: str

    def identity_payload(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "program_id": self.program_id,
            "tournament_id": self.tournament_id,
            "execution_scope": self.execution_scope,
            "previous_parent_revision_id": self.previous_parent_revision_id,
            "selected_revision_id": self.selected_revision_id,
            "decision_sha256": self.decision_sha256,
            "previous_event_sha256": self.previous_event_sha256,
        }


class SearchParentLog:
    """Single-program hash chain; search state only, never activation state."""

    def __init__(self, path: str | Path, *, program_id: str) -> None:
        if not isinstance(program_id, str) or not program_id.strip():
            raise AgentProgramViolation("search-parent program_id is invalid")
        self.path = Path(path)
        self.program_id = program_id
        self.lease_path = self.path.with_suffix(self.path.suffix + ".writer.lock")

    def append(self, decision: TournamentDecision) -> bool:
        # Recompute even for an already-instantiated object so forged construction
        # cannot bypass the decision's self-authenticating contract.
        if content_sha256(decision.identity_payload()) != decision.decision_sha256:
            raise AgentProgramViolation("tournament decision hash mismatch")
        if decision.program_id != self.program_id:
            raise AgentProgramViolation("search-parent program identity drift")
        if decision.execution_scope != "fixture":
            raise AgentProgramViolation("search-parent accepts fixture decisions only")
        events = self._read()
        matching = [
            event
            for event in events
            if event.tournament_id == decision.tournament_id
        ]
        if matching:
            if (
                len(matching) == 1
                and matching[0].decision_sha256 == decision.decision_sha256
            ):
                return False
            raise AgentProgramViolation("conflicting tournament decision replay")
        current = (
            events[-1].selected_revision_id
            if events
            else decision.parent_revision_id
        )
        if current != decision.parent_revision_id:
            raise AgentProgramViolation("search-parent fork detected")
        identity = {
            "sequence": len(events) + 1,
            "program_id": self.program_id,
            "tournament_id": decision.tournament_id,
            "execution_scope": "fixture",
            "previous_parent_revision_id": decision.parent_revision_id,
            "selected_revision_id": decision.winner_revision_id,
            "decision_sha256": decision.decision_sha256,
            "previous_event_sha256": events[-1].event_sha256 if events else None,
        }
        payload = {**identity, "event_sha256": content_sha256(identity)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            lease_fd = os.open(
                self.lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError as error:
            raise AgentProgramViolation("search-parent writer lease is busy") from error
        try:
            # Recheck under the lease so two writers cannot create the same next
            # sequence or diverging heads.
            if self._read() != events:
                raise AgentProgramViolation("search-parent changed during append")
            fd = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            try:
                os.write(fd, (canonical_json(payload) + "\n").encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            os.close(lease_fd)
            self.lease_path.unlink(missing_ok=True)
        return True

    def all(self) -> tuple[SearchParentEvent, ...]:
        return tuple(self._read())

    def current_parent_revision_id(self) -> str | None:
        events = self._read()
        return events[-1].selected_revision_id if events else None

    def _read(self) -> list[SearchParentEvent]:
        if not self.path.exists():
            return []
        events: list[SearchParentEvent] = []
        expected_fields = {
            "sequence",
            "program_id",
            "tournament_id",
            "execution_scope",
            "previous_parent_revision_id",
            "selected_revision_id",
            "decision_sha256",
            "previous_event_sha256",
            "event_sha256",
        }
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), 1
        ):
            try:
                payload = json.loads(line)
                if not isinstance(payload, Mapping) or set(payload) != expected_fields:
                    raise ValueError("invalid fields")
                event = SearchParentEvent(**payload)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise AgentProgramViolation(
                    f"invalid search-parent event at line {line_number}"
                ) from error
            if event.program_id != self.program_id:
                raise AgentProgramViolation("search-parent program identity drift")
            if event.execution_scope != "fixture":
                raise AgentProgramViolation("search-parent scope drift")
            if event.sequence != line_number:
                raise AgentProgramViolation("search-parent sequence drift")
            expected_previous = events[-1].event_sha256 if events else None
            if event.previous_event_sha256 != expected_previous:
                raise AgentProgramViolation("search-parent hash chain fork")
            if event.event_sha256 != content_sha256(event.identity_payload()):
                raise AgentProgramViolation("search-parent event hash mismatch")
            if events and (
                event.previous_parent_revision_id
                != events[-1].selected_revision_id
            ):
                raise AgentProgramViolation("search-parent revision chain fork")
            events.append(event)
        return events


__all__ = ["SearchParentEvent", "SearchParentLog"]
