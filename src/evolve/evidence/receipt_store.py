"""Append-only, content-addressed storage for v3 execution receipts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from evolve.contracts import Receipt, canonical_json


class ReceiptStoreError(RuntimeError):
    """Base class for receipt persistence failures."""


class IntegrityError(ReceiptStoreError):
    """Stored or supplied content does not match its immutable identity."""


class ReceiptConflict(ReceiptStoreError):
    """A receipt identifier was reused for different content."""


class ConcurrentWriterError(ReceiptStoreError):
    """The single-writer lease is already held."""


class ReceiptStore:
    """Filesystem receipt authority with one append writer at a time.

    The JSONL log is the only receipt source of truth. Artifacts are addressed by
    their literal SHA-256. Every read verifies both the receipt record hash and
    every referenced artifact, so corruption is never silently projected.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.artifacts_dir = self.root / "artifacts" / "sha256"
        self.log_path = self.root / "receipts.jsonl"
        self.lease_path = self.root / ".writer.lock"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def acquire_writer(self) -> Iterator[None]:
        try:
            descriptor = os.open(
                self.lease_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as error:
            raise ConcurrentWriterError(
                f"receipt writer lease already held: {self.lease_path}"
            ) from error
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode())
            os.fsync(descriptor)
            yield
        finally:
            os.close(descriptor)
            self.lease_path.unlink(missing_ok=True)

    def append(self, receipt: Receipt, artifact: bytes) -> Receipt:
        artifact_digest = hashlib.sha256(artifact).hexdigest()
        if artifact_digest != receipt.artifact_sha256:
            raise IntegrityError(
                "artifact SHA-256 mismatch: "
                f"expected {receipt.artifact_sha256}, got {artifact_digest}"
            )

        with self.acquire_writer():
            existing = {item.receipt_id: item for item in self.list_receipts()}
            if prior := existing.get(receipt.receipt_id):
                if prior != receipt:
                    raise ReceiptConflict(
                        f"receipt {receipt.receipt_id} already has different content"
                    )
                self._verify_artifact(prior.artifact_sha256, expected=artifact)
                return prior

            artifact_path = self._artifact_path(receipt.artifact_sha256)
            self._write_content_addressed(artifact_path, artifact)
            record = {
                "receipt": dataclasses.asdict(receipt),
                "receipt_sha256": receipt.content_sha256,
            }
            encoded = (canonical_json(record) + "\n").encode("utf-8")
            descriptor = os.open(
                self.log_path,
                os.O_CREAT | os.O_APPEND | os.O_WRONLY,
                0o600,
            )
            try:
                written = os.write(descriptor, encoded)
                if written != len(encoded):
                    raise ReceiptStoreError("partial receipt append")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return receipt

    def list_receipts(self) -> tuple[Receipt, ...]:
        if not self.log_path.exists():
            return ()
        receipts: list[Receipt] = []
        seen: dict[str, Receipt] = {}
        with self.log_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                    receipt = Receipt(**record["receipt"])
                    expected = record["receipt_sha256"]
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise IntegrityError(
                        f"invalid receipt record at line {line_number}"
                    ) from error
                if receipt.content_sha256 != expected:
                    raise IntegrityError(
                        f"receipt hash mismatch at line {line_number}: "
                        f"{receipt.receipt_id}"
                    )
                if prior := seen.get(receipt.receipt_id):
                    if prior != receipt:
                        raise ReceiptConflict(
                            f"receipt {receipt.receipt_id} appears with "
                            "conflicting content"
                        )
                    continue
                self._verify_artifact(receipt.artifact_sha256)
                seen[receipt.receipt_id] = receipt
                receipts.append(receipt)
        return tuple(receipts)

    def receipts_for(self, plan_id: str) -> tuple[Receipt, ...]:
        return tuple(
            receipt for receipt in self.list_receipts() if receipt.plan_id == plan_id
        )

    def read_artifact(self, artifact_sha256: str) -> bytes:
        return self._verify_artifact(artifact_sha256)

    def _artifact_path(self, digest: str) -> Path:
        return self.artifacts_dir / digest

    def _verify_artifact(self, digest: str, *, expected: bytes | None = None) -> bytes:
        path = self._artifact_path(digest)
        try:
            content = path.read_bytes()
        except FileNotFoundError as error:
            raise IntegrityError(f"missing artifact {digest}") from error
        actual = hashlib.sha256(content).hexdigest()
        if actual != digest:
            raise IntegrityError(
                f"artifact hash mismatch for {digest}: stored content is {actual}"
            )
        if expected is not None and content != expected:
            raise IntegrityError(f"artifact content mismatch for {digest}")
        return content

    @staticmethod
    def _write_content_addressed(path: Path, content: bytes) -> None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            existing = path.read_bytes()
            if existing != content:
                raise IntegrityError(f"artifact collision at {path.name}")
            return
        try:
            written = os.write(descriptor, content)
            if written != len(content):
                raise ReceiptStoreError("partial artifact write")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
