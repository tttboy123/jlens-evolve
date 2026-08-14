"""Cryptographically attested, independent JLens observation boundary.

The HMAC key is an execution capability and is deliberately never placed in a
receipt or evidence envelope.  A signed attestation binds an observation's
literal bytes to the exact model execution it observed.  Merely choosing a
trusted-looking observer id, copying hashes, or embedding an ``internal_trace``
in model output cannot mint this capability.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from typing import Callable, Mapping

from evolve.contracts import (
    ClaimGrade,
    EvidenceEnvelope,
    Receipt,
    canonical_json,
)
from evolve.evidence.receipt_store import IntegrityError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ATTESTATION_VERSION = "trusted-jlens-attestation-v1"
_RECEIPT_KIND = "trusted_jlens_observation"
_OBSERVER_ID = "trusted-jlens-v1"


@dataclass(frozen=True, slots=True)
class TrustedObserverIdentity:
    """Public identity of one trusted observer implementation/key capability."""

    key_id: str
    implementation_id: str
    implementation_sha256: str

    def __post_init__(self) -> None:
        _require_text("key_id", self.key_id)
        _require_text("implementation_id", self.implementation_id)
        _require_sha256("implementation_sha256", self.implementation_sha256)


class TrustedObserverKeyring:
    """In-memory trust root used to verify observer attestations.

    Secrets have no serialization API.  Production callers should construct the
    keyring from process-local secret material rather than artifacts under the
    campaign root.
    """

    def __init__(
        self, identities: Mapping[TrustedObserverIdentity, bytes]
    ) -> None:
        entries: dict[str, tuple[TrustedObserverIdentity, bytes]] = {}
        for identity, secret in identities.items():
            if not isinstance(secret, bytes) or len(secret) < 16:
                raise ValueError("trusted observer secret must be at least 16 bytes")
            if identity.key_id in entries:
                raise ValueError(f"duplicate trusted observer key id: {identity.key_id}")
            entries[identity.key_id] = (identity, secret)
        self._entries = entries

    def verify_attestation(self, attestation: Mapping[str, object]) -> bool:
        try:
            body, signature = _split_attestation(attestation)
            key_id = _require_text_value(body, "key_id")
            identity, secret = self._entries[key_id]
            if body["observer_implementation_id"] != identity.implementation_id:
                return False
            if (
                body["observer_implementation_sha256"]
                != identity.implementation_sha256
            ):
                return False
            expected = _signature(secret, body)
            return hmac.compare_digest(signature, expected)
        except (KeyError, TypeError, ValueError):
            return False

    def verify_evidence(self, envelope: EvidenceEnvelope) -> bool:
        """Re-verify a persisted envelope without trusting its metadata flags."""

        if envelope.observer_id != _OBSERVER_ID:
            return False
        attestation = envelope.payload.get("attestation")
        if not isinstance(attestation, Mapping) or not self.verify_attestation(
            attestation
        ):
            return False
        try:
            body, _ = _split_attestation(attestation)
            if body["attestation_version"] != _ATTESTATION_VERSION:
                return False
            if body["observation_receipt_kind"] != _RECEIPT_KIND:
                return False
            if envelope.receipt_ids != (body["observation_receipt_id"],):
                return False
            if envelope.artifact_sha256 != body["observation_artifact_sha256"]:
                return False
            for field in (
                "plan_id",
                "task_revision_id",
                "model_receipt_id",
                "model_artifact_sha256",
                "observation_artifact_sha256",
                "prediction_sha256",
                "mechanism_id",
                "observer_implementation_id",
                "observer_implementation_sha256",
                "observed_at",
                "nonce",
            ):
                if envelope.payload.get(field) != body[field]:
                    return False
            return True
        except (KeyError, TypeError):
            return False


def issue_trusted_observation_attestation(
    *,
    identity: TrustedObserverIdentity,
    secret_key: bytes,
    observation_artifact: bytes,
    observation_receipt_id: str,
    plan_id: str,
    task_revision_id: str,
    model_receipt_id: str,
    model_artifact_sha256: str,
    observed_at: str,
    nonce: str,
) -> dict[str, object]:
    """Sign a concrete observation artifact for one exact model execution."""

    if not isinstance(secret_key, bytes) or len(secret_key) < 16:
        raise ValueError("trusted observer secret must be at least 16 bytes")
    observation = _parse_observation(observation_artifact)
    prediction_sha256 = _require_sha256_value(observation, "prediction_sha256")
    mechanism_id = _require_text_value(observation, "mechanism_id")
    for name, value in (
        ("observation_receipt_id", observation_receipt_id),
        ("plan_id", plan_id),
        ("task_revision_id", task_revision_id),
        ("model_receipt_id", model_receipt_id),
        ("observed_at", observed_at),
        ("nonce", nonce),
    ):
        _require_text(name, value)
    _require_sha256("model_artifact_sha256", model_artifact_sha256)
    body: dict[str, object] = {
        "attestation_version": _ATTESTATION_VERSION,
        "key_id": identity.key_id,
        "observer_implementation_id": identity.implementation_id,
        "observer_implementation_sha256": identity.implementation_sha256,
        "observer_capability": "independent_runtime_observation",
        "observation_receipt_id": observation_receipt_id,
        "observation_receipt_kind": _RECEIPT_KIND,
        "plan_id": plan_id,
        "task_revision_id": task_revision_id,
        "model_receipt_id": model_receipt_id,
        "model_artifact_sha256": model_artifact_sha256,
        "observation_artifact_sha256": hashlib.sha256(
            observation_artifact
        ).hexdigest(),
        "prediction_sha256": prediction_sha256,
        "mechanism_id": mechanism_id,
        "observed_at": observed_at,
        "nonce": nonce,
    }
    return {**body, "signature_hmac_sha256": _signature(secret_key, body)}


class TrustedJacobianLensObserver:
    """Emit E3-eligible evidence only after bytes and attestation verify."""

    observer_id = _OBSERVER_ID

    def __init__(
        self,
        *,
        keyring: TrustedObserverKeyring,
        artifact_reader: Callable[[str], bytes],
    ) -> None:
        self._keyring = keyring
        self._artifact_reader = artifact_reader

    def observe(self, receipt: Receipt) -> EvidenceEnvelope | None:
        if receipt.kind != _RECEIPT_KIND:
            return None
        attestation = receipt.payload.get("attestation")
        if not isinstance(attestation, Mapping):
            raise IntegrityError("trusted observation attestation is missing")
        if not self._keyring.verify_attestation(attestation):
            raise IntegrityError("trusted observation attestation is invalid")
        body, signature = _split_attestation(attestation)
        if body["attestation_version"] != _ATTESTATION_VERSION:
            raise IntegrityError("trusted observation attestation version is invalid")
        if body["observation_receipt_kind"] != receipt.kind:
            raise IntegrityError("trusted observation receipt kind mismatch")
        if body["observation_receipt_id"] != receipt.receipt_id:
            raise IntegrityError("trusted observation receipt identity mismatch")
        if body["plan_id"] != receipt.plan_id:
            raise IntegrityError("trusted observation plan identity mismatch")

        artifact = self._artifact_reader(receipt.artifact_sha256)
        actual_artifact_sha256 = hashlib.sha256(artifact).hexdigest()
        if actual_artifact_sha256 != receipt.artifact_sha256:
            raise IntegrityError("trusted observation artifact SHA-256 mismatch")
        if body["observation_artifact_sha256"] != actual_artifact_sha256:
            raise IntegrityError("trusted observation artifact attestation mismatch")
        observation = _parse_observation(artifact)
        if body["prediction_sha256"] != _require_sha256_value(
            observation, "prediction_sha256"
        ):
            raise IntegrityError("trusted observation prediction mismatch")
        if body["mechanism_id"] != _require_text_value(observation, "mechanism_id"):
            raise IntegrityError("trusted observation mechanism mismatch")

        payload = {
            field: body[field]
            for field in (
                "plan_id",
                "task_revision_id",
                "model_receipt_id",
                "model_artifact_sha256",
                "observation_artifact_sha256",
                "prediction_sha256",
                "mechanism_id",
                "observer_implementation_id",
                "observer_implementation_sha256",
                "observed_at",
                "nonce",
            )
        }
        payload.update(
            {
                "attestation": dict(attestation),
                "attestation_verified": True,
                "observation_sha256": actual_artifact_sha256,
            }
        )
        identity = hashlib.sha256(
            (
                f"{self.observer_id}\0{receipt.content_sha256}\0{signature}"
            ).encode()
        ).hexdigest()
        envelope = EvidenceEnvelope(
            evidence_id=f"evidence-{identity}",
            receipt_ids=(receipt.receipt_id,),
            observer_id=self.observer_id,
            grade=ClaimGrade.E0,
            payload=payload,
            artifact_sha256=actual_artifact_sha256,
        )
        if not self._keyring.verify_evidence(envelope):
            raise IntegrityError("trusted observation evidence projection is invalid")
        return envelope


def _parse_observation(artifact: bytes) -> Mapping[str, object]:
    try:
        value = json.loads(artifact.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError("trusted observation artifact is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise IntegrityError("trusted observation artifact must be an object")
    return value


def _signature(secret: bytes, body: Mapping[str, object]) -> str:
    return hmac.new(
        secret,
        canonical_json(body).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _split_attestation(
    attestation: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    body = dict(attestation)
    signature = body.pop("signature_hmac_sha256")
    if not isinstance(signature, str) or _SHA256.fullmatch(signature) is None:
        raise ValueError("invalid attestation signature")
    required = {
        "attestation_version",
        "key_id",
        "observer_implementation_id",
        "observer_implementation_sha256",
        "observer_capability",
        "observation_receipt_id",
        "observation_receipt_kind",
        "plan_id",
        "task_revision_id",
        "model_receipt_id",
        "model_artifact_sha256",
        "observation_artifact_sha256",
        "prediction_sha256",
        "mechanism_id",
        "observed_at",
        "nonce",
    }
    if set(body) != required:
        raise ValueError("invalid attestation fields")
    for field in (
        "observer_implementation_sha256",
        "model_artifact_sha256",
        "observation_artifact_sha256",
        "prediction_sha256",
    ):
        _require_sha256(field, body[field])
    for field in required - {
        "observer_implementation_sha256",
        "model_artifact_sha256",
        "observation_artifact_sha256",
        "prediction_sha256",
    }:
        _require_text(field, body[field])
    if body["observer_capability"] != "independent_runtime_observation":
        raise ValueError("invalid observer capability")
    return body, signature


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _require_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a literal lowercase SHA-256")


def _require_text_value(value: Mapping[str, object], name: str) -> str:
    result = value[name]
    _require_text(name, result)
    return result  # type: ignore[return-value]


def _require_sha256_value(value: Mapping[str, object], name: str) -> str:
    result = value[name]
    _require_sha256(name, result)
    return result  # type: ignore[return-value]
