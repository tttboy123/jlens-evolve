"""Content-verified atomic campaign checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from evolve.contracts import ContractViolation, canonical_json


class CheckpointManager:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def path_for(self, campaign_id: str) -> Path:
        if not campaign_id or "/" in campaign_id or "\\" in campaign_id:
            raise ContractViolation("campaign_id is not safe for a checkpoint path")
        return self._root / f"{campaign_id}.json"

    def save(self, campaign_id: str, payload: Mapping[str, Any]) -> None:
        payload_bytes = canonical_json(payload).encode("utf-8")
        document = {
            "schema_version": 1,
            "payload": payload,
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        }
        encoded = (canonical_json(document) + "\n").encode("utf-8")
        destination = self.path_for(campaign_id)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self._root, prefix=f".{campaign_id}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        finally:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()

    def load(self, campaign_id: str) -> Mapping[str, Any]:
        path = self.path_for(campaign_id)
        if not path.exists():
            raise ContractViolation(f"checkpoint does not exist for {campaign_id}")
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema_version") != 1:
            raise ContractViolation("unsupported checkpoint schema")
        payload = document.get("payload")
        actual = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        if actual != document.get("payload_sha256"):
            raise ContractViolation("checkpoint hash mismatch")
        if not isinstance(payload, Mapping):
            raise ContractViolation("checkpoint payload must be an object")
        return payload
