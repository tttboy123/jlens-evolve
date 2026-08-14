"""Append-only run indexes used to rebuild autonomous state after interruption."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from evolve.contracts import canonical_json

from .config import AutonomousEvolutionError

_GENESIS = "0" * 64


class HashChainIndex:
    """Single-writer JSONL index with deterministic replay and idempotent append."""

    def __init__(self, path: str | Path, *, index_id: str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.index_id = index_id
        self.lease_path = self.path.with_suffix(self.path.suffix + ".writer.lock")

    def rows(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise AutonomousEvolutionError("autonomous index has a partial tail")
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        previous = _GENESIS
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            try:
                stored = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise AutonomousEvolutionError(
                    f"autonomous index line {line_number} is invalid"
                ) from error
            if not isinstance(stored, dict):
                raise AutonomousEvolutionError("autonomous index row must be an object")
            digest = stored.get("event_sha256")
            event = {key: value for key, value in stored.items() if key != "event_sha256"}
            if (
                event.get("index_id") != self.index_id
                or event.get("sequence") != len(rows)
                or event.get("previous_event_sha256") != previous
                or not isinstance(digest, str)
                or hashlib.sha256(canonical_json(event).encode()).hexdigest() != digest
            ):
                raise AutonomousEvolutionError(
                    f"autonomous index chain mismatch at line {line_number}"
                )
            event_id = event.get("event_id")
            if not isinstance(event_id, str) or not event_id or event_id in seen:
                raise AutonomousEvolutionError("autonomous index event identity conflict")
            seen.add(event_id)
            previous = digest
            rows.append(stored)
        return tuple(rows)

    def append(self, *, event_id: str, payload: Mapping[str, Any]) -> bool:
        if not event_id:
            raise AutonomousEvolutionError("autonomous index event_id is empty")
        with self._writer():
            rows = self.rows()
            prior = next((row for row in rows if row["event_id"] == event_id), None)
            if prior is not None:
                if prior.get("payload") != dict(payload):
                    raise AutonomousEvolutionError(
                        "autonomous index immutable event conflict"
                    )
                return False
            previous = rows[-1]["event_sha256"] if rows else _GENESIS
            event = {
                "index_id": self.index_id,
                "sequence": len(rows),
                "previous_event_sha256": previous,
                "event_id": event_id,
                "payload": dict(payload),
            }
            stored = {
                **event,
                "event_sha256": hashlib.sha256(
                    canonical_json(event).encode()
                ).hexdigest(),
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600
            )
            try:
                encoded = (canonical_json(stored) + "\n").encode()
                if os.write(descriptor, encoded) != len(encoded):
                    raise AutonomousEvolutionError("partial autonomous index append")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return True

    @contextmanager
    def _writer(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError as error:
            raise AutonomousEvolutionError(
                f"autonomous index writer is already active: {self.path.name}"
            ) from error
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode())
            os.fsync(descriptor)
            yield
        finally:
            os.close(descriptor)
            self.lease_path.unlink(missing_ok=True)
