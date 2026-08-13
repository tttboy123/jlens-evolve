"""Single-writer file leases created with O_EXCL."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from evolve.contracts import ContractViolation, canonical_json


class LeaseContended(ContractViolation):
    """Another writer owns the campaign lease."""


@dataclass(frozen=True, slots=True)
class Lease:
    campaign_id: str
    owner_id: str
    token: str
    acquired_at: str
    expires_at: str


class FileLeaseManager:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, campaign_id: str) -> Path:
        if not campaign_id or "/" in campaign_id or "\\" in campaign_id:
            raise ContractViolation("campaign_id is not safe for a lease path")
        return self._root / f"{campaign_id}.lease"

    def acquire(
        self,
        campaign_id: str,
        *,
        owner_id: str,
        now: datetime,
        ttl: timedelta,
    ) -> Lease:
        if not owner_id.strip():
            raise ContractViolation("owner_id must be non-empty text")
        if now.tzinfo is None or ttl <= timedelta(0):
            raise ContractViolation("lease requires aware time and positive ttl")
        lease = Lease(
            campaign_id=campaign_id,
            owner_id=owner_id,
            token=uuid.uuid4().hex,
            acquired_at=now.isoformat(),
            expires_at=(now + ttl).isoformat(),
        )
        path = self._path(campaign_id)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                owner = existing.get("owner_id", "unknown owner")
            except (OSError, json.JSONDecodeError):
                owner = "an unreadable lease"
            raise LeaseContended(f"campaign lease is held by {owner}") from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(asdict(lease)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return lease

    def release(self, campaign_id: str, token: str) -> None:
        path = self._path(campaign_id)
        if not path.exists():
            raise ContractViolation("campaign lease does not exist")
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ContractViolation("campaign lease cannot be verified") from error
        if existing.get("token") != token:
            raise ContractViolation("lease token does not own this campaign")
        path.unlink()
