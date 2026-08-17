"""Tamper-evident, reserve-before-dispatch parent-call ledger."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    ContractError,
    LoopAuthorization,
    canonical_json,
    sha256_json,
)

_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_STATUSES = {"reserved", "completed", "aborted"}


@dataclass(frozen=True)
class ParentCallRecord:
    call_id: str
    request_sha256: str
    status: str
    response_sha256: str | None
    response: dict[str, Any] | None
    usage: dict[str, Any]
    error: str | None

    _FIELDS = frozenset(
        {
            "call_id",
            "request_sha256",
            "status",
            "response_sha256",
            "response",
            "usage",
            "error",
        }
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParentCallRecord:
        if not isinstance(data, dict) or set(data) != cls._FIELDS:
            raise ContractError("invalid parent call record fields")
        record = cls(
            call_id=str(data["call_id"]),
            request_sha256=str(data["request_sha256"]),
            status=str(data["status"]),
            response_sha256=data["response_sha256"],
            response=data["response"],
            usage=dict(data["usage"]),
            error=data["error"],
        )
        record.validate()
        return record

    def validate(self) -> None:
        if _IDENTIFIER.fullmatch(self.call_id) is None:
            raise ContractError("invalid call_id")
        if _SHA256.fullmatch(self.request_sha256) is None:
            raise ContractError("invalid request sha256")
        if self.status not in _STATUSES:
            raise ContractError("invalid parent call status")
        if self.status == "completed":
            if (
                not isinstance(self.response, dict)
                or self.response_sha256 is None
                or _SHA256.fullmatch(self.response_sha256) is None
                or sha256_json(self.response) != self.response_sha256
            ):
                raise ContractError("invalid completed parent call evidence")
        elif self.response is not None or self.response_sha256 is not None:
            raise ContractError("non-completed parent call cannot have a response")
        if self.status == "aborted" and not self.error:
            raise ContractError("aborted parent call requires an error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "request_sha256": self.request_sha256,
            "status": self.status,
            "response_sha256": self.response_sha256,
            "response": self.response,
            "usage": self.usage,
            "error": self.error,
        }


class ParentCallLedger:
    """Persist call reservations; aborted calls consume budget and are not resent."""

    _STATE_FIELDS = frozenset(
        {"schema_version", "authorization", "records", "state_sha256"}
    )

    def __init__(self, path: Path, authorization: LoopAuthorization) -> None:
        authorization.validate()
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.authorization = authorization
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self.path.exists():
                stored_authorization, _ = self._load_unlocked()
                if stored_authorization != authorization:
                    raise ContractError("authorization does not match parent ledger")
            else:
                self._write_unlocked([])

    def records(self) -> list[ParentCallRecord]:
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            _, records = self._load_unlocked()
            return records

    def get(self, call_id: str) -> ParentCallRecord | None:
        return next((row for row in self.records() if row.call_id == call_id), None)

    def reserve(self, *, call_id: str, request_sha256: str) -> ParentCallRecord:
        self.authorization.assert_active()
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            _, records = self._load_unlocked()
            existing = next((row for row in records if row.call_id == call_id), None)
            if existing is not None:
                if existing.request_sha256 != request_sha256:
                    raise ContractError(
                        "call_id already belongs to a different request"
                    )
                return existing
            if len(records) >= self.authorization.maximum_parent_calls:
                raise ContractError("parent call budget exceeded")
            record = ParentCallRecord(
                call_id=call_id,
                request_sha256=request_sha256,
                status="reserved",
                response_sha256=None,
                response=None,
                usage={},
                error=None,
            )
            record.validate()
            self._write_unlocked([*records, record])
            return record

    def complete(
        self,
        *,
        call_id: str,
        response_sha256: str,
        usage: dict[str, Any],
        response: dict[str, Any] | None = None,
    ) -> ParentCallRecord:
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            _, records = self._load_unlocked()
            index, current = self._required_record(records, call_id)
            if current.status != "reserved":
                raise ContractError("parent call is already terminal")
            if response is None:
                response = {}
            if sha256_json(response) != response_sha256:
                raise ContractError("response sha256 mismatch")
            completed = ParentCallRecord(
                call_id=current.call_id,
                request_sha256=current.request_sha256,
                status="completed",
                response_sha256=response_sha256,
                response=response,
                usage=dict(usage),
                error=None,
            )
            completed.validate()
            records[index] = completed
            self._write_unlocked(records)
            return completed

    def abort(self, *, call_id: str, reason: str) -> ParentCallRecord:
        if not reason.strip():
            raise ContractError("abort reason must be non-empty")
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            _, records = self._load_unlocked()
            index, current = self._required_record(records, call_id)
            if current.status == "aborted":
                return current
            if current.status != "reserved":
                raise ContractError("parent call is already terminal")
            aborted = ParentCallRecord(
                call_id=current.call_id,
                request_sha256=current.request_sha256,
                status="aborted",
                response_sha256=None,
                response=None,
                usage={},
                error=reason,
            )
            aborted.validate()
            records[index] = aborted
            self._write_unlocked(records)
            return aborted

    @staticmethod
    def _required_record(
        records: list[ParentCallRecord], call_id: str
    ) -> tuple[int, ParentCallRecord]:
        for index, record in enumerate(records):
            if record.call_id == call_id:
                return index, record
        raise ContractError(f"unknown parent call: {call_id}")

    def _load_unlocked(self) -> tuple[LoopAuthorization, list[ParentCallRecord]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError("parent ledger is unreadable") from exc
        if not isinstance(data, dict) or set(data) != self._STATE_FIELDS:
            raise ContractError("invalid parent ledger fields")
        content = {key: value for key, value in data.items() if key != "state_sha256"}
        if data["state_sha256"] != sha256_json(content):
            raise ContractError("parent ledger state sha256 mismatch")
        if data["schema_version"] != 1:
            raise ContractError("unsupported parent ledger schema")
        authorization = LoopAuthorization.from_dict(data["authorization"])
        records = [ParentCallRecord.from_dict(row) for row in data["records"]]
        if len({row.call_id for row in records}) != len(records):
            raise ContractError("duplicate call_id in parent ledger")
        return authorization, records

    def _write_unlocked(self, records: list[ParentCallRecord]) -> None:
        content = {
            "schema_version": 1,
            "authorization": self.authorization.to_dict(),
            "records": [row.to_dict() for row in records],
        }
        data = {**content, "state_sha256": sha256_json(content)}
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as handle:
            handle.write(canonical_json(data) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(self.path)
