"""Append-only candidate archive and experimental AgentProgram lineage."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from mutation_proposer import InactiveChangeSet

_SHA256 = re.compile(r"[0-9a-f]{64}")


class ArchiveContractError(ValueError):
    """Raised when candidate history would become mutable or ambiguous."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _validate_sha256(value: str, *, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ArchiveContractError(f"invalid {name} sha256")


class CandidateArchive:
    """Persist every candidate and state transition without deleting failures."""

    _TRANSITIONS: ClassVar[dict[str, frozenset[str]]] = {
        "inactive": frozenset({"evaluating", "failed"}),
        "evaluating": frozenset({"selected", "rejected", "failed"}),
        "selected": frozenset(),
        "rejected": frozenset(),
        "failed": frozenset(),
    }

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.authority_path = self.root / "AUTHORITY.json"
        self.events_path = self.root / "events.jsonl"
        self.candidates_dir = self.root / "candidates"
        if not self.authority_path.is_file():
            raise ArchiveContractError(f"archive authority does not exist: {self.root}")

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        original_agent_program_sha256: str,
        seed_parent_agent_program_sha256: str,
    ) -> CandidateArchive:
        _validate_sha256(original_agent_program_sha256, name="original AgentProgram")
        _validate_sha256(
            seed_parent_agent_program_sha256, name="seed parent AgentProgram"
        )
        root = root.resolve()
        authority_path = root / "AUTHORITY.json"
        expected = {
            "schema_version": 1,
            "original_agent_program_sha256": original_agent_program_sha256,
            "search_parent_sha256": seed_parent_agent_program_sha256,
            "seed_parent_agent_program_sha256": seed_parent_agent_program_sha256,
            "production_active_ref": None,
            "search_parent_history": [],
        }
        if authority_path.exists():
            current = json.loads(authority_path.read_text(encoding="utf-8"))
            if current != expected and not (
                current.get("original_agent_program_sha256")
                == original_agent_program_sha256
                and current.get("seed_parent_agent_program_sha256")
                == seed_parent_agent_program_sha256
            ):
                raise ArchiveContractError(
                    "archive authority already exists with different roots"
                )
        else:
            root.mkdir(parents=True, exist_ok=True)
            (root / "candidates").mkdir(parents=True, exist_ok=True)
            _atomic_json(authority_path, expected)
            (root / "events.jsonl").touch()
        return cls(root)

    def authority(self) -> dict[str, Any]:
        return json.loads(self.authority_path.read_text(encoding="utf-8"))

    @property
    def search_parent_sha256(self) -> str:
        return str(self.authority()["search_parent_sha256"])

    def events(self) -> tuple[dict[str, Any], ...]:
        if not self.events_path.exists():
            return ()
        return tuple(
            json.loads(line)
            for line in self.events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    def _append_event(self, event: dict[str, Any]) -> dict[str, Any]:
        rows = self.events()
        previous = rows[-1]["event_sha256"] if rows else None
        payload = {
            "schema_version": 1,
            "sequence": len(rows) + 1,
            "previous_event_sha256": previous,
            **event,
        }
        payload["event_sha256"] = _digest(payload)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return payload

    def _candidate_path(self, candidate_sha256: str) -> Path:
        _validate_sha256(candidate_sha256, name="candidate AgentProgram")
        return self.candidates_dir / candidate_sha256 / "CHANGESET.json"

    def candidates(self) -> tuple[str, ...]:
        if not self.candidates_dir.exists():
            return ()
        return tuple(
            sorted(
                path.parent.name
                for path in self.candidates_dir.glob("*/CHANGESET.json")
            )
        )

    def _known_programs(self) -> set[str]:
        authority = self.authority()
        return {
            authority["original_agent_program_sha256"],
            authority["seed_parent_agent_program_sha256"],
            *self.candidates(),
        }

    def register_candidate(self, changeset: InactiveChangeSet) -> dict[str, Any]:
        if changeset.status != "inactive" or changeset.auto_apply is not False:
            raise ArchiveContractError("candidate must be an inactive ChangeSet")
        if changeset.production_promotion_allowed is not False:
            raise ArchiveContractError("candidate cannot allow production promotion")
        candidate = changeset.candidate_agent_program_sha256
        parent = changeset.parent_agent_program_sha256
        if candidate == parent:
            raise ArchiveContractError("candidate lineage self-cycle is forbidden")
        if parent not in self._known_programs():
            raise ArchiveContractError(
                "candidate parent is not present in archive lineage"
            )
        payload = changeset.to_dict()
        payload["archive_blob_sha256"] = _digest(payload)
        path = self._candidate_path(candidate)
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8"))
            if current != payload:
                raise ArchiveContractError("candidate blob is immutable")
            return current
        path.parent.mkdir(parents=True, exist_ok=False)
        _atomic_json(path, payload)
        self._append_event(
            {
                "event_type": "candidate_registered",
                "candidate_agent_program_sha256": candidate,
                "parent_agent_program_sha256": parent,
                "changeset_id": changeset.changeset_id,
                "candidate_blob_sha256": payload["archive_blob_sha256"],
                "from_state": None,
                "to_state": "inactive",
                "reason": "inactive_changeset_registered",
                "evidence_sha256": None,
            }
        )
        return payload

    def candidate_state(self, candidate_sha256: str) -> str:
        if not self._candidate_path(candidate_sha256).is_file():
            raise ArchiveContractError("candidate is not registered")
        state = "inactive"
        for event in self.events():
            if event.get("candidate_agent_program_sha256") == candidate_sha256:
                target = event.get("to_state")
                if target in self._TRANSITIONS:
                    state = target
        return state

    def transition(
        self,
        candidate_sha256: str,
        target_state: str,
        *,
        reason: str,
        evidence_sha256: str,
    ) -> dict[str, Any]:
        if not reason.strip():
            raise ArchiveContractError("candidate transition reason cannot be empty")
        _validate_sha256(evidence_sha256, name="transition evidence")
        current = self.candidate_state(candidate_sha256)
        if target_state not in self._TRANSITIONS[current]:
            raise ArchiveContractError(
                f"illegal candidate transition: {current}->{target_state}"
            )
        proposal = json.loads(
            self._candidate_path(candidate_sha256).read_text(encoding="utf-8")
        )
        return self._append_event(
            {
                "event_type": "candidate_state_changed",
                "candidate_agent_program_sha256": candidate_sha256,
                "parent_agent_program_sha256": proposal["parent_agent_program_sha256"],
                "from_state": current,
                "to_state": target_state,
                "reason": reason,
                "evidence_sha256": evidence_sha256,
            }
        )

    def record_rollback(
        self,
        candidate_sha256: str,
        *,
        forward_patch_sha256: str,
        rollback_patch_sha256: str,
        verified: bool,
    ) -> dict[str, Any]:
        self.candidate_state(candidate_sha256)
        _validate_sha256(forward_patch_sha256, name="forward patch")
        _validate_sha256(rollback_patch_sha256, name="rollback patch")
        if not isinstance(verified, bool):
            raise ArchiveContractError("rollback verified must be boolean")
        return self._append_event(
            {
                "event_type": "rollback_recorded",
                "candidate_agent_program_sha256": candidate_sha256,
                "forward_patch_sha256": forward_patch_sha256,
                "rollback_patch_sha256": rollback_patch_sha256,
                "rollback_verified": verified,
                "from_state": self.candidate_state(candidate_sha256),
                "to_state": self.candidate_state(candidate_sha256),
                "reason": "rollback_roundtrip_recorded",
                "evidence_sha256": rollback_patch_sha256,
            }
        )

    def _rollback_verified(self, candidate_sha256: str) -> bool:
        return any(
            event.get("event_type") == "rollback_recorded"
            and event.get("candidate_agent_program_sha256") == candidate_sha256
            and event.get("rollback_verified") is True
            for event in self.events()
        )

    def advance_search_parent(
        self,
        candidate_sha256: str,
        *,
        decision_sha256: str,
        production: bool = False,
    ) -> dict[str, Any]:
        if production:
            raise ArchiveContractError("production/global promotion is prohibited")
        _validate_sha256(decision_sha256, name="parent decision")
        if self.candidate_state(candidate_sha256) != "selected":
            raise ArchiveContractError("search parent candidate must be selected")
        if not self._rollback_verified(candidate_sha256):
            raise ArchiveContractError(
                "verified rollback is required before parent advance"
            )
        authority = self.authority()
        previous = authority["search_parent_sha256"]
        if previous == candidate_sha256:
            return authority["search_parent_history"][-1]
        decision = {
            "previous_parent_sha256": previous,
            "search_parent_sha256": candidate_sha256,
            "decision_sha256": decision_sha256,
            "scope": "experimental_search_lineage_only",
            "production_promoted": False,
        }
        authority["search_parent_sha256"] = candidate_sha256
        authority["search_parent_history"].append(decision)
        _atomic_json(self.authority_path, authority)
        self._append_event(
            {
                "event_type": "search_parent_advanced",
                "candidate_agent_program_sha256": candidate_sha256,
                "previous_parent_sha256": previous,
                "from_state": "selected",
                "to_state": "selected",
                "reason": "experimental_search_parent_gate_passed",
                "evidence_sha256": decision_sha256,
                "production_promoted": False,
            }
        )
        return decision

    def verify(self) -> dict[str, Any]:
        errors: list[str] = []
        previous = None
        for index, event in enumerate(self.events(), start=1):
            if event.get("sequence") != index:
                errors.append(f"event {index} sequence mismatch")
            if event.get("previous_event_sha256") != previous:
                errors.append(f"event {index} previous hash mismatch")
            claimed = event.get("event_sha256")
            payload = dict(event)
            payload.pop("event_sha256", None)
            actual = _digest(payload)
            if claimed != actual:
                errors.append(f"event {index} content hash mismatch")
            previous = claimed
        for candidate in self.candidates():
            path = self._candidate_path(candidate)
            value = json.loads(path.read_text(encoding="utf-8"))
            claimed = value.pop("archive_blob_sha256", None)
            if claimed != _digest(value):
                errors.append(f"candidate {candidate} blob hash mismatch")
        authority = self.authority()
        if authority.get("production_active_ref") is not None:
            errors.append("production active ref must remain unset")
        return {
            "valid": not errors,
            "errors": errors,
            "event_count": len(self.events()),
            "candidate_count": len(self.candidates()),
            "production_active_ref": authority.get("production_active_ref"),
        }
