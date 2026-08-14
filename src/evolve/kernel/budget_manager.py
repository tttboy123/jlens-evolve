"""Campaign-wide cost and model-call accounting."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from evolve.contracts import ContractViolation, canonical_json


class BudgetExceeded(ContractViolation):
    """A reservation or final charge would exceed campaign authorization."""


class LedgerIntegrityError(ContractViolation):
    """The durable cost ledger is incomplete, malformed, or hash-invalid."""


class LedgerConflict(ContractViolation):
    """An immutable event identity was reused with different facts."""


class LedgerBusy(ContractViolation):
    """Another writer owns the durable cost ledger lease."""


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    max_cost_cny: float
    max_model_calls: int
    reserved_cost_cny: float
    reserved_model_calls: int
    spent_cost_cny: float
    spent_model_calls: int


class BudgetManager:
    """Reserve before dispatch and reconcile only after a recorded result."""

    def __init__(
        self,
        *,
        max_cost_cny: float,
        max_model_calls: int,
        reserved_cost_cny: float = 0,
        reserved_model_calls: int = 0,
        spent_cost_cny: float = 0,
        spent_model_calls: int = 0,
    ) -> None:
        self._max_cost_cny = max_cost_cny
        self._max_model_calls = max_model_calls
        self._reserved_cost_cny = reserved_cost_cny
        self._reserved_model_calls = reserved_model_calls
        self._spent_cost_cny = spent_cost_cny
        self._spent_model_calls = spent_model_calls

    def reserve(self, *, cost_cny: float, model_calls: int) -> None:
        if cost_cny < 0 or model_calls < 0:
            raise ContractViolation("budget reservations cannot be negative")
        prospective_cost = self._spent_cost_cny + self._reserved_cost_cny + cost_cny
        prospective_calls = (
            self._spent_model_calls + self._reserved_model_calls + model_calls
        )
        if prospective_cost > self._max_cost_cny:
            raise BudgetExceeded("campaign cost budget exhausted")
        if prospective_calls > self._max_model_calls:
            raise BudgetExceeded("campaign model call budget exhausted")
        self._reserved_cost_cny += cost_cny
        self._reserved_model_calls += model_calls

    def record(
        self,
        *,
        reserved_cost_cny: float,
        reserved_model_calls: int,
        actual_cost_cny: float,
        actual_model_calls: int,
    ) -> None:
        values = (
            reserved_cost_cny,
            reserved_model_calls,
            actual_cost_cny,
            actual_model_calls,
        )
        if any(value < 0 for value in values):
            raise ContractViolation("budget charges cannot be negative")
        if (
            reserved_cost_cny > self._reserved_cost_cny
            or reserved_model_calls > self._reserved_model_calls
        ):
            raise ContractViolation("result exceeds its outstanding reservation")
        if self._spent_cost_cny + actual_cost_cny > self._max_cost_cny:
            raise BudgetExceeded("actual campaign cost exceeds authorization")
        if self._spent_model_calls + actual_model_calls > self._max_model_calls:
            raise BudgetExceeded("actual campaign model calls exceed authorization")
        self._reserved_cost_cny -= reserved_cost_cny
        self._reserved_model_calls -= reserved_model_calls
        self._spent_cost_cny += actual_cost_cny
        self._spent_model_calls += actual_model_calls

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            max_cost_cny=self._max_cost_cny,
            max_model_calls=self._max_model_calls,
            reserved_cost_cny=self._reserved_cost_cny,
            reserved_model_calls=self._reserved_model_calls,
            spent_cost_cny=self._spent_cost_cny,
            spent_model_calls=self._spent_model_calls,
        )


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DispatchResult = TypeVar("_DispatchResult")


class DurableCostLedger:
    """Append-only reservation/charge authority rebuilt on every operation."""

    def __init__(
        self,
        path: str | Path,
        *,
        campaign_id: str,
        max_cost_cny: float,
        max_model_calls: int,
    ) -> None:
        if not isinstance(campaign_id, str) or not campaign_id.strip():
            raise ContractViolation("cost ledger campaign_id must be non-empty")
        _validate_amount(max_cost_cny, "max_cost_cny")
        _validate_calls(max_model_calls, "max_model_calls")
        self.path = Path(path)
        self.lease_path = self.path.with_suffix(self.path.suffix + ".writer.lock")
        self.campaign_id = campaign_id
        self.max_cost_cny = float(max_cost_cny)
        self.max_model_calls = max_model_calls
        self._initialize()

    def reserve(
        self, reservation_id: str, *, cost_cny: float, model_calls: int
    ) -> bool:
        _validate_event_id(reservation_id, "reservation_id")
        _validate_amount(cost_cny, "reservation cost")
        _validate_calls(model_calls, "reservation model_calls")
        event = {
            "event_id": reservation_id,
            "kind": "reservation",
            "campaign_id": self.campaign_id,
            "cost_cny": float(cost_cny),
            "model_calls": model_calls,
        }

        def validate(events: tuple[dict[str, Any], ...]) -> None:
            snapshot = self._snapshot(events)
            if (
                snapshot.spent_cost_cny + snapshot.reserved_cost_cny + float(cost_cny)
                > self.max_cost_cny
            ):
                raise BudgetExceeded("campaign cost budget exhausted")
            if (
                snapshot.spent_model_calls + snapshot.reserved_model_calls + model_calls
                > self.max_model_calls
            ):
                raise BudgetExceeded("campaign model call budget exhausted")

        return self._append_mutation(event, validate)

    def record(
        self,
        reservation_id: str,
        *,
        result_id: str,
        actual_cost_cny: float,
        actual_model_calls: int,
    ) -> bool:
        _validate_event_id(reservation_id, "reservation_id")
        _validate_event_id(result_id, "result_id")
        _validate_amount(actual_cost_cny, "actual cost")
        _validate_calls(actual_model_calls, "actual model_calls")
        event = {
            "event_id": result_id,
            "kind": "charge",
            "campaign_id": self.campaign_id,
            "reservation_id": reservation_id,
            "actual_cost_cny": float(actual_cost_cny),
            "actual_model_calls": actual_model_calls,
        }

        def validate(events: tuple[dict[str, Any], ...]) -> None:
            reservation = next(
                (
                    item
                    for item in events
                    if item["kind"] == "reservation"
                    and item["event_id"] == reservation_id
                ),
                None,
            )
            if reservation is None:
                raise ContractViolation("charge references unknown reservation")
            prior_charge = next(
                (
                    item
                    for item in events
                    if item["kind"] == "charge"
                    and item["reservation_id"] == reservation_id
                ),
                None,
            )
            if prior_charge is not None:
                raise LedgerConflict(
                    f"reservation {reservation_id} already has a result"
                )
            snapshot = self._snapshot(events)
            if snapshot.spent_cost_cny + float(actual_cost_cny) > self.max_cost_cny:
                raise BudgetExceeded("actual campaign cost exceeds authorization")
            if snapshot.spent_model_calls + actual_model_calls > self.max_model_calls:
                raise BudgetExceeded("actual campaign model calls exceed authorization")

        return self._append_mutation(event, validate)

    def dispatch_once(
        self,
        reservation_id: str,
        *,
        cost_cny: float,
        model_calls: int,
        dispatch: Callable[[], _DispatchResult],
    ) -> tuple[bool, _DispatchResult | None]:
        """Reserve durably before dispatch and suppress duplicate side effects."""

        written = self.reserve(
            reservation_id, cost_cny=cost_cny, model_calls=model_calls
        )
        if not written:
            return False, None
        return True, dispatch()

    def snapshot(self) -> BudgetSnapshot:
        return self._snapshot(self._read_events())

    def events(self) -> tuple[dict[str, Any], ...]:
        return self._read_events()

    def _initialize(self) -> None:
        if self.path.exists():
            self._validate_header(self._read_events())
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._writer_lease():
            if self.path.exists():
                self._validate_header(self._read_events())
                return
            self._append_event(
                {
                    "event_id": "ledger-open",
                    "kind": "opened",
                    "campaign_id": self.campaign_id,
                    "max_cost_cny": self.max_cost_cny,
                    "max_model_calls": self.max_model_calls,
                }
            )

    def _append_mutation(
        self,
        event: dict[str, Any],
        validate: Callable[[tuple[dict[str, Any], ...]], None],
    ) -> bool:
        with self._writer_lease():
            events = self._read_events()
            self._validate_header(events)
            prior = next(
                (item for item in events if item["event_id"] == event["event_id"]),
                None,
            )
            if prior is not None:
                if prior == event:
                    return False
                raise LedgerConflict(
                    f"conflicting immutable ledger event {event['event_id']}"
                )
            validate(events)
            self._append_event(event)
            return True

    def _read_events(self) -> tuple[dict[str, Any], ...]:
        if not self.path.is_file():
            raise LedgerIntegrityError("durable cost ledger is missing")
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise LedgerIntegrityError("durable cost ledger has a partial final event")
        events = []
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            try:
                payload = json.loads(line)
                expected_sha256 = payload.pop("event_sha256")
            except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
                raise LedgerIntegrityError(
                    f"invalid cost ledger event at line {line_number}"
                ) from error
            if (
                not isinstance(payload, dict)
                or not isinstance(expected_sha256, str)
                or _SHA256.fullmatch(expected_sha256) is None
                or _event_sha256(payload) != expected_sha256
            ):
                raise LedgerIntegrityError(
                    f"cost ledger hash mismatch at line {line_number}"
                )
            self._validate_event(payload, line_number)
            events.append(payload)
        if not events:
            raise LedgerIntegrityError("durable cost ledger is empty")
        event_ids = [event["event_id"] for event in events]
        if len(event_ids) != len(set(event_ids)):
            raise LedgerIntegrityError("durable cost ledger has duplicate event ids")
        return tuple(events)

    def _validate_header(self, events: tuple[dict[str, Any], ...]) -> None:
        header = events[0]
        if (
            header.get("kind") != "opened"
            or header.get("event_id") != "ledger-open"
            or header.get("campaign_id") != self.campaign_id
            or header.get("max_cost_cny") != self.max_cost_cny
            or header.get("max_model_calls") != self.max_model_calls
        ):
            raise LedgerIntegrityError("cost ledger authorization identity mismatch")

    def _validate_event(self, event: dict[str, Any], line_number: int) -> None:
        try:
            _validate_event_id(event["event_id"], "event_id")
            if event["campaign_id"] != self.campaign_id:
                raise LedgerIntegrityError("cost ledger campaign identity mismatch")
            if event["kind"] == "opened":
                _validate_amount(event["max_cost_cny"], "max_cost_cny")
                _validate_calls(event["max_model_calls"], "max_model_calls")
            elif event["kind"] == "reservation":
                _validate_amount(event["cost_cny"], "reservation cost")
                _validate_calls(event["model_calls"], "reservation model_calls")
            elif event["kind"] == "charge":
                _validate_event_id(event["reservation_id"], "reservation_id")
                _validate_amount(event["actual_cost_cny"], "actual cost")
                _validate_calls(event["actual_model_calls"], "actual model_calls")
            else:
                raise LedgerIntegrityError("cost ledger event kind is invalid")
        except (KeyError, ContractViolation) as error:
            raise LedgerIntegrityError(
                f"invalid cost ledger event at line {line_number}"
            ) from error

    def _snapshot(self, events: tuple[dict[str, Any], ...]) -> BudgetSnapshot:
        self._validate_header(events)
        charged = {
            event["reservation_id"] for event in events if event["kind"] == "charge"
        }
        reservations = [
            event
            for event in events
            if event["kind"] == "reservation" and event["event_id"] not in charged
        ]
        charges = [event for event in events if event["kind"] == "charge"]
        return BudgetSnapshot(
            max_cost_cny=self.max_cost_cny,
            max_model_calls=self.max_model_calls,
            reserved_cost_cny=sum(event["cost_cny"] for event in reservations),
            reserved_model_calls=sum(event["model_calls"] for event in reservations),
            spent_cost_cny=sum(event["actual_cost_cny"] for event in charges),
            spent_model_calls=sum(event["actual_model_calls"] for event in charges),
        )

    def _append_event(self, event: dict[str, Any]) -> None:
        stored = {**event, "event_sha256": _event_sha256(event)}
        encoded = (canonical_json(stored) + "\n").encode()
        fd = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
        try:
            offset = 0
            while offset < len(encoded):
                offset += os.write(fd, encoded[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)

    @contextmanager
    def _writer_lease(self) -> Iterator[None]:
        try:
            lease_fd = os.open(
                self.lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
            )
        except FileExistsError as error:
            raise LedgerBusy("durable cost ledger writer lease is held") from error
        try:
            os.write(lease_fd, f"pid={os.getpid()}\n".encode())
            os.fsync(lease_fd)
            yield
        finally:
            os.close(lease_fd)
            self.lease_path.unlink(missing_ok=True)


def _event_sha256(event: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(event).encode()).hexdigest()


def _validate_event_id(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractViolation(f"{field} must be non-empty")


def _validate_amount(value: object, field: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ContractViolation(f"{field} must be finite and non-negative")


def _validate_calls(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractViolation(f"{field} must be a non-negative integer")
