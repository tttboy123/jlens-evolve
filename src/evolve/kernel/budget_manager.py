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
_GENESIS_PREVIOUS_SHA256 = "0" * 64
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
        self.head_path = self.path.with_suffix(self.path.suffix + ".head.json")
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
        return self._snapshot(self._read_events(recover_head=True))

    def events(self) -> tuple[dict[str, Any], ...]:
        return self._read_events(recover_head=True)

    def _initialize(self) -> None:
        if self.path.exists():
            self._validate_header(self._read_events(recover_head=True))
            return
        if self.head_path.exists():
            raise LedgerIntegrityError("cost ledger head exists without event log")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._writer_lease():
            if self.path.exists():
                self._validate_header(
                    self._read_events(
                        recover_head=True, writer_lease_held=True
                    )
                )
                return
            self._append_event(
                {
                    "event_id": "ledger-open",
                    "kind": "opened",
                    "campaign_id": self.campaign_id,
                    "max_cost_cny": self.max_cost_cny,
                    "max_model_calls": self.max_model_calls,
                },
                (),
            )

    def _append_mutation(
        self,
        event: dict[str, Any],
        validate: Callable[[tuple[dict[str, Any], ...]], None],
    ) -> bool:
        with self._writer_lease():
            events = self._read_events(recover_head=True, writer_lease_held=True)
            self._validate_header(events)
            prior = next(
                (item for item in events if item["event_id"] == event["event_id"]),
                None,
            )
            if prior is not None:
                if _event_facts(prior) == event:
                    return False
                raise LedgerConflict(
                    f"conflicting immutable ledger event {event['event_id']}"
                )
            validate(events)
            self._append_event(event, events)
            return True

    def _read_events(
        self,
        *,
        recover_head: bool = False,
        writer_lease_held: bool = False,
        validate_head: bool = True,
    ) -> tuple[dict[str, Any], ...]:
        if not self.path.is_file():
            raise LedgerIntegrityError("durable cost ledger is missing")
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise LedgerIntegrityError("durable cost ledger has a partial final event")
        events: list[dict[str, Any]] = []
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            try:
                stored = json.loads(line)
                if not isinstance(stored, dict):
                    raise TypeError("event must be an object")
                expected_sha256 = stored.get("event_sha256")
                payload = {
                    key: value
                    for key, value in stored.items()
                    if key != "event_sha256"
                }
            except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
                raise LedgerIntegrityError(
                    f"invalid cost ledger event at line {line_number}"
                ) from error
            if (
                not isinstance(stored, dict)
                or not isinstance(expected_sha256, str)
                or _SHA256.fullmatch(expected_sha256) is None
                or _event_sha256(payload) != expected_sha256
            ):
                raise LedgerIntegrityError(
                    f"cost ledger hash mismatch at line {line_number}"
                )
            self._validate_event(stored, line_number)
            expected_sequence = len(events)
            expected_previous = (
                _GENESIS_PREVIOUS_SHA256
                if not events
                else events[-1]["event_sha256"]
            )
            if stored.get("sequence") != expected_sequence:
                raise LedgerIntegrityError(
                    f"cost ledger sequence mismatch at line {line_number}"
                )
            if stored.get("previous_event_sha256") != expected_previous:
                raise LedgerIntegrityError(
                    f"cost ledger chain mismatch at line {line_number}"
                )
            events.append(stored)
        if not events:
            raise LedgerIntegrityError("durable cost ledger is empty")
        event_ids = [event["event_id"] for event in events]
        if len(event_ids) != len(set(event_ids)):
            raise LedgerIntegrityError("durable cost ledger has duplicate event ids")
        self._validate_replay(tuple(events))
        if validate_head:
            self._validate_head(
                tuple(events),
                recover_head=recover_head,
                writer_lease_held=writer_lease_held,
            )
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
            _validate_calls(event["sequence"], "sequence")
            if _SHA256.fullmatch(event["previous_event_sha256"]) is None:
                raise ContractViolation(
                    "previous_event_sha256 must be a literal lowercase SHA-256"
                )
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
        except (KeyError, TypeError, ContractViolation) as error:
            raise LedgerIntegrityError(
                f"invalid cost ledger event at line {line_number}"
            ) from error

    def _validate_replay(self, events: tuple[dict[str, Any], ...]) -> None:
        """Replay domain invariants so a valid chain is also a valid budget history."""

        self._validate_header(events)
        reservations: dict[str, dict[str, Any]] = {}
        charged_reservations: set[str] = set()
        spent_cost = 0.0
        spent_calls = 0
        for line_number, event in enumerate(events[1:], 2):
            if event["kind"] == "opened":
                raise LedgerIntegrityError(
                    f"duplicate cost ledger genesis at line {line_number}"
                )
            if event["kind"] == "reservation":
                reservations[event["event_id"]] = event
            elif event["kind"] == "charge":
                reservation_id = event["reservation_id"]
                if reservation_id not in reservations:
                    raise LedgerIntegrityError(
                        f"charge references unknown reservation at line {line_number}"
                    )
                if reservation_id in charged_reservations:
                    raise LedgerIntegrityError(
                        f"reservation has duplicate charge at line {line_number}"
                    )
                charged_reservations.add(reservation_id)
                spent_cost += event["actual_cost_cny"]
                spent_calls += event["actual_model_calls"]
            reserved_cost = sum(
                reservation["cost_cny"]
                for reservation_id, reservation in reservations.items()
                if reservation_id not in charged_reservations
            )
            reserved_calls = sum(
                reservation["model_calls"]
                for reservation_id, reservation in reservations.items()
                if reservation_id not in charged_reservations
            )
            if spent_cost + reserved_cost > self.max_cost_cny:
                raise LedgerIntegrityError(
                    f"cost ledger exceeds authorization at line {line_number}"
                )
            if spent_calls + reserved_calls > self.max_model_calls:
                raise LedgerIntegrityError(
                    f"cost ledger model calls exceed authorization at line {line_number}"
                )

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

    def _append_event(
        self, event: dict[str, Any], events: tuple[dict[str, Any], ...]
    ) -> None:
        chained = {
            **event,
            "sequence": len(events),
            "previous_event_sha256": (
                events[-1]["event_sha256"]
                if events
                else _GENESIS_PREVIOUS_SHA256
            ),
        }
        stored = {**chained, "event_sha256": _event_sha256(chained)}
        encoded = (canonical_json(stored) + "\n").encode()
        fd = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
        try:
            offset = 0
            while offset < len(encoded):
                offset += os.write(fd, encoded[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)
        self._write_head(stored)

    def _write_head(self, event: dict[str, Any]) -> None:
        payload = {
            "schema_version": 1,
            "campaign_id": self.campaign_id,
            "event_count": event["sequence"] + 1,
            "head_event_sha256": event["event_sha256"],
        }
        stored = {**payload, "anchor_sha256": _event_sha256(payload)}
        encoded = (canonical_json(stored) + "\n").encode("utf-8")
        temporary = self.head_path.with_name(
            f"{self.head_path.name}.tmp.{os.getpid()}"
        )
        try:
            fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                offset = 0
                while offset < len(encoded):
                    offset += os.write(fd, encoded[offset:])
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(temporary, self.head_path)
            directory_fd = os.open(self.head_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def _validate_head(
        self,
        events: tuple[dict[str, Any], ...],
        *,
        recover_head: bool,
        writer_lease_held: bool,
    ) -> None:
        if not self.head_path.is_file():
            raise LedgerIntegrityError("durable cost ledger head is missing")
        raw = self.head_path.read_bytes()
        if not raw.endswith(b"\n"):
            raise LedgerIntegrityError("durable cost ledger head is partial")
        try:
            stored = json.loads(raw.decode("utf-8"))
            if not isinstance(stored, dict):
                raise TypeError("head must be an object")
            anchor_sha256 = stored.get("anchor_sha256")
            payload = {
                key: value for key, value in stored.items() if key != "anchor_sha256"
            }
        except (TypeError, UnicodeError, json.JSONDecodeError) as error:
            raise LedgerIntegrityError("durable cost ledger head is invalid") from error
        if (
            not isinstance(stored, dict)
            or not isinstance(anchor_sha256, str)
            or _SHA256.fullmatch(anchor_sha256) is None
            or _event_sha256(payload) != anchor_sha256
        ):
            raise LedgerIntegrityError("durable cost ledger head hash mismatch")
        if (
            payload.get("schema_version") != 1
            or payload.get("campaign_id") != self.campaign_id
        ):
            raise LedgerIntegrityError("durable cost ledger head mismatch")
        if (
            payload.get("event_count") == len(events)
            and payload.get("head_event_sha256") == events[-1]["event_sha256"]
        ):
            return
        head_count = payload.get("event_count")
        strict_one_event_prefix = (
            isinstance(head_count, int)
            and not isinstance(head_count, bool)
            and head_count >= 1
            and head_count + 1 == len(events)
            and payload.get("head_event_sha256")
            == events[head_count - 1]["event_sha256"]
        )
        if not recover_head or not strict_one_event_prefix:
            raise LedgerIntegrityError("durable cost ledger head mismatch")
        if writer_lease_held:
            self._write_head(events[-1])
            return
        with self._writer_lease():
            # Re-read under the single-writer lease: another process may have
            # already recovered the head while this caller waited.
            refreshed = self._read_events(validate_head=False)
            self._validate_head(
                refreshed,
                recover_head=True,
                writer_lease_held=True,
            )

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


def _event_facts(event: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable business event without its storage-chain envelope."""

    return {
        key: value
        for key, value in event.items()
        if key not in {"sequence", "previous_event_sha256", "event_sha256"}
    }


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
