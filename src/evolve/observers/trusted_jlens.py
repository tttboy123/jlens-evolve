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
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable, Mapping

from evolve.contracts import (
    ClaimGrade,
    EvidenceEnvelope,
    ExecutionPlan,
    MechanismPrediction,
    Receipt,
    canonical_json,
)
from evolve.evidence.receipt_store import IntegrityError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ATTESTATION_VERSION = "trusted-jlens-attestation-v1"
_RECEIPT_KIND = "trusted_jlens_observation"
_OBSERVER_ID = "trusted-jlens-v1"
_OBSERVATION_SCHEMA = "structured-jlens-observation-v1"


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

    def __init__(self, identities: Mapping[TrustedObserverIdentity, bytes]) -> None:
        entries: dict[str, tuple[TrustedObserverIdentity, bytes]] = {}
        for identity, secret in identities.items():
            if not isinstance(secret, bytes) or len(secret) < 16:
                raise ValueError("trusted observer secret must be at least 16 bytes")
            if identity.key_id in entries:
                raise ValueError(
                    f"duplicate trusted observer key id: {identity.key_id}"
                )
            entries[identity.key_id] = (identity, secret)
        self._entries = entries

    def verify_attestation(self, attestation: Mapping[str, object]) -> bool:
        try:
            body, signature = _split_attestation(attestation)
            key_id = _require_text_value(body, "key_id")
            identity, secret = self._entries[key_id]
            if body["observer_implementation_id"] != identity.implementation_id:
                return False
            if body["observer_implementation_sha256"] != identity.implementation_sha256:
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
                "mechanism_prediction_receipt_id",
                "mechanism_prediction_artifact_sha256",
                "observer_config_sha256",
                "raw_trace_sha256",
                "expected_internal_effect_sha256",
                "observed_internal_effect_sha256",
                "effect_consistent",
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

    def verify_receipt_lineage(
        self,
        *,
        envelope: EvidenceEnvelope,
        observation_receipt: Receipt,
        prediction_receipt: Receipt,
        model_receipt: Receipt,
        artifact_reader: Callable[[str], bytes],
    ) -> bool:
        """Rebuild a trusted observation from its exact persisted receipts."""

        try:
            if not self.verify_evidence(envelope):
                return False
            attestation = envelope.payload["attestation"]
            if not isinstance(attestation, Mapping):
                return False
            body, _ = _split_attestation(attestation)
            if (
                observation_receipt.kind != _RECEIPT_KIND
                or observation_receipt.receipt_id != body["observation_receipt_id"]
                or observation_receipt.plan_id != body["plan_id"]
                or observation_receipt.payload != {"attestation": dict(attestation)}
                or observation_receipt.artifact_sha256
                != body["observation_artifact_sha256"]
            ):
                return False
            if (
                prediction_receipt.kind != "mechanism_prediction"
                or prediction_receipt.receipt_id
                != body["mechanism_prediction_receipt_id"]
                or prediction_receipt.plan_id != body["plan_id"]
                or prediction_receipt.artifact_sha256
                != body["mechanism_prediction_artifact_sha256"]
            ):
                return False
            if (
                model_receipt.kind != "model"
                or model_receipt.receipt_id != body["model_receipt_id"]
                or model_receipt.plan_id != body["plan_id"]
                or model_receipt.artifact_sha256 != body["model_artifact_sha256"]
            ):
                return False
            if not (
                prediction_receipt.sequence
                < model_receipt.sequence
                < observation_receipt.sequence
            ):
                return False
            if (
                len(
                    {
                        prediction_receipt.campaign_id,
                        model_receipt.campaign_id,
                        observation_receipt.campaign_id,
                    }
                )
                != 1
            ):
                return False
            prediction = MechanismPrediction.from_payload(prediction_receipt.payload)
            artifact = artifact_reader(observation_receipt.artifact_sha256)
            observation = _parse_structured_observation(artifact)
            if (
                observation["mechanism_prediction_receipt_id"]
                != prediction_receipt.receipt_id
                or observation["mechanism_prediction_artifact_sha256"]
                != prediction_receipt.artifact_sha256
                or observation["mechanism_id"] != prediction.mechanism_id
                or observation["observer_config_sha256"]
                != prediction.observer_config_sha256
                or observation["expected_internal_effect_sha256"]
                != prediction.expected_internal_effect_sha256
            ):
                return False
            return hashlib.sha256(artifact).hexdigest() == envelope.artifact_sha256
        except (KeyError, TypeError, ValueError, IntegrityError):
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
    observation = _parse_structured_observation(observation_artifact)
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
        "observation_artifact_sha256": hashlib.sha256(observation_artifact).hexdigest(),
        "prediction_sha256": prediction_sha256,
        "mechanism_id": mechanism_id,
        "mechanism_prediction_receipt_id": observation[
            "mechanism_prediction_receipt_id"
        ],
        "mechanism_prediction_artifact_sha256": observation[
            "mechanism_prediction_artifact_sha256"
        ],
        "observer_config_sha256": observation["observer_config_sha256"],
        "raw_trace_sha256": observation["raw_trace_sha256"],
        "expected_internal_effect_sha256": observation[
            "expected_internal_effect_sha256"
        ],
        "observed_internal_effect_sha256": observation[
            "observed_internal_effect_sha256"
        ],
        "effect_consistent": observation["effect_consistent"],
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
        observation = _parse_structured_observation(artifact)
        if body["prediction_sha256"] != _require_sha256_value(
            observation, "prediction_sha256"
        ):
            raise IntegrityError("trusted observation prediction mismatch")
        if body["mechanism_id"] != _require_text_value(observation, "mechanism_id"):
            raise IntegrityError("trusted observation mechanism mismatch")
        for field in (
            "observer_config_sha256",
            "raw_trace_sha256",
            "expected_internal_effect_sha256",
            "observed_internal_effect_sha256",
            "effect_consistent",
        ):
            if body[field] != observation[field]:
                raise IntegrityError(f"trusted observation {field} mismatch")

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
                "mechanism_prediction_receipt_id",
                "mechanism_prediction_artifact_sha256",
                "observer_config_sha256",
                "raw_trace_sha256",
                "expected_internal_effect_sha256",
                "observed_internal_effect_sha256",
                "effect_consistent",
                "observer_implementation_id",
                "observer_implementation_sha256",
                "observed_at",
                "nonce",
            )
        }
        raw_trace = observation["raw_trace"]
        if not isinstance(raw_trace, Mapping):  # validated above; narrows type
            raise IntegrityError("trusted observation raw trace is invalid")
        payload.update(
            {
                "attestation": dict(attestation),
                "attestation_verified": True,
                "observation_sha256": actual_artifact_sha256,
                "locations": raw_trace["locations"],
                "expected_internal_effect": observation["expected_internal_effect"],
                "observed_internal_effect": observation["observed_internal_effect"],
            }
        )
        identity = hashlib.sha256(
            (f"{self.observer_id}\0{receipt.content_sha256}\0{signature}").encode()
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


class TrustedJLensReceiptIssuer:
    """Issue one trusted observation receipt from a plan-bound raw trace source."""

    def __init__(
        self,
        *,
        identity: TrustedObserverIdentity,
        secret_key: bytes,
        raw_trace_reader: Callable[[ExecutionPlan, Receipt], bytes],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(secret_key, bytes) or len(secret_key) < 16:
            raise ValueError("trusted observer secret must be at least 16 bytes")
        self._identity = identity
        self._secret_key = secret_key
        self._raw_trace_reader = raw_trace_reader
        self._clock = clock or (lambda: datetime.now(UTC))

    def issue(
        self,
        plan: ExecutionPlan,
        receipts: tuple[Receipt, ...],
    ) -> tuple[Receipt, bytes]:
        prediction_receipt = _only_receipt(
            receipts, "mechanism_prediction", plan.plan_id
        )
        model_receipt = _only_receipt(receipts, "model", plan.plan_id)
        raw_trace = self._raw_trace_reader(plan, model_receipt)
        observation = derive_structured_jlens_observation(
            raw_trace_artifact=raw_trace,
            mechanism_prediction_receipt=prediction_receipt,
            prediction_sha256=_require_sha256_value(
                model_receipt.payload, "patch_sha256"
            ),
        )
        observed_at = _isoformat(self._clock())
        identity = hashlib.sha256(
            canonical_json(
                {
                    "plan_id": plan.plan_id,
                    "prediction_receipt_id": prediction_receipt.receipt_id,
                    "model_receipt_id": model_receipt.receipt_id,
                    "observation_sha256": hashlib.sha256(observation).hexdigest(),
                }
            ).encode("utf-8")
        ).hexdigest()
        receipt_id = f"receipt-trusted-jlens-{identity}"
        attestation = issue_trusted_observation_attestation(
            identity=self._identity,
            secret_key=self._secret_key,
            observation_artifact=observation,
            observation_receipt_id=receipt_id,
            plan_id=plan.plan_id,
            task_revision_id=plan.task.revision_id,
            model_receipt_id=model_receipt.receipt_id,
            model_artifact_sha256=model_receipt.artifact_sha256,
            observed_at=observed_at,
            nonce=f"nonce-{identity}",
        )
        receipt = Receipt(
            receipt_id=receipt_id,
            campaign_id=plan.campaign_id,
            plan_id=plan.plan_id,
            sequence=max(receipt.sequence for receipt in receipts) + 1,
            kind=_RECEIPT_KIND,
            created_at=observed_at,
            payload={"attestation": attestation},
            artifact_sha256=hashlib.sha256(observation).hexdigest(),
        )
        return receipt, observation


def derive_structured_jlens_observation(
    *,
    raw_trace_artifact: bytes,
    mechanism_prediction_receipt: Receipt,
    prediction_sha256: str,
) -> bytes:
    """Deterministically derive an attestable observation from raw JLens rows."""

    if mechanism_prediction_receipt.kind != "mechanism_prediction":
        raise IntegrityError("mechanism prediction receipt kind is invalid")
    if (
        mechanism_prediction_receipt.artifact_sha256
        != hashlib.sha256(
            canonical_json(mechanism_prediction_receipt.payload).encode("utf-8")
        ).hexdigest()
    ):
        raise IntegrityError("mechanism prediction receipt does not bind payload")
    try:
        mechanism_prediction = MechanismPrediction.from_payload(
            mechanism_prediction_receipt.payload
        )
    except (KeyError, TypeError, ValueError) as error:
        raise IntegrityError(
            "mechanism prediction receipt payload is invalid"
        ) from error
    _require_sha256("prediction_sha256", prediction_sha256)
    raw_trace = _parse_json_object(raw_trace_artifact, "raw JLens trace")
    canonical_raw = canonical_json(raw_trace).encode("utf-8")
    if raw_trace_artifact != canonical_raw:
        raise IntegrityError("raw JLens trace must use canonical JSON bytes")
    locations = _normalize_locations(raw_trace)
    normalized_raw = {"locations": locations}
    if raw_trace != normalized_raw:
        raise IntegrityError("raw JLens trace contains unsupported fields")
    expected = _normalize_expected_effect(mechanism_prediction.expected_internal_effect)
    observed = _derive_observed_effect(locations, expected)
    effect_consistent = _effect_is_consistent(observed, expected)
    body: dict[str, object] = {
        "schema_version": _OBSERVATION_SCHEMA,
        "observer_config_sha256": mechanism_prediction.observer_config_sha256,
        "mechanism_id": mechanism_prediction.mechanism_id,
        "mechanism_prediction_receipt_id": mechanism_prediction_receipt.receipt_id,
        "mechanism_prediction_artifact_sha256": (
            mechanism_prediction_receipt.artifact_sha256
        ),
        "prediction_sha256": prediction_sha256,
        "raw_trace": normalized_raw,
        "raw_trace_sha256": hashlib.sha256(canonical_raw).hexdigest(),
        "expected_internal_effect": expected,
        "expected_internal_effect_sha256": _json_sha256(expected),
        "observed_internal_effect": observed,
        "observed_internal_effect_sha256": _json_sha256(observed),
        "effect_consistent": effect_consistent,
    }
    body["derivation_sha256"] = _json_sha256(body)
    return canonical_json(body).encode("utf-8")


def _parse_json_object(artifact: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(artifact.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError(f"{label} is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{label} must be an object")
    return value


def _parse_structured_observation(artifact: bytes) -> Mapping[str, object]:
    value = _parse_json_object(artifact, "trusted observation artifact")
    required = {
        "schema_version",
        "observer_config_sha256",
        "mechanism_id",
        "mechanism_prediction_receipt_id",
        "mechanism_prediction_artifact_sha256",
        "prediction_sha256",
        "raw_trace",
        "raw_trace_sha256",
        "expected_internal_effect",
        "expected_internal_effect_sha256",
        "observed_internal_effect",
        "observed_internal_effect_sha256",
        "effect_consistent",
        "derivation_sha256",
    }
    if set(value) != required or value.get("schema_version") != _OBSERVATION_SCHEMA:
        raise IntegrityError("trusted observation structured schema is invalid")
    raw_trace = value.get("raw_trace")
    expected_effect = value.get("expected_internal_effect")
    observed_effect = value.get("observed_internal_effect")
    if not isinstance(raw_trace, Mapping) or not isinstance(expected_effect, Mapping):
        raise IntegrityError("trusted observation structured fields are invalid")
    locations = _normalize_locations(raw_trace)
    expected = _normalize_expected_effect(expected_effect)
    observed = _derive_observed_effect(locations, expected)
    if observed_effect != observed:
        raise IntegrityError("trusted observation deterministic derivation mismatch")
    canonical_raw = canonical_json({"locations": locations}).encode("utf-8")
    checks = {
        "raw_trace_sha256": hashlib.sha256(canonical_raw).hexdigest(),
        "expected_internal_effect_sha256": _json_sha256(expected),
        "observed_internal_effect_sha256": _json_sha256(observed),
    }
    for field, expected_value in checks.items():
        if value.get(field) != expected_value:
            raise IntegrityError(f"trusted observation {field} mismatch")
    if value.get("effect_consistent") is not _effect_is_consistent(observed, expected):
        raise IntegrityError("trusted observation effect consistency mismatch")
    derivation = dict(value)
    derivation.pop("derivation_sha256")
    if value.get("derivation_sha256") != _json_sha256(derivation):
        raise IntegrityError("trusted observation derivation SHA-256 mismatch")
    _require_sha256_value(value, "observer_config_sha256")
    _require_sha256_value(value, "mechanism_prediction_artifact_sha256")
    _require_sha256_value(value, "prediction_sha256")
    _require_text_value(value, "mechanism_id")
    _require_text_value(value, "mechanism_prediction_receipt_id")
    return value


def _normalize_locations(raw_trace: Mapping[str, object]) -> list[dict[str, object]]:
    if set(raw_trace) != {"locations"}:
        raise IntegrityError("raw JLens trace must contain only locations")
    rows = raw_trace.get("locations")
    if not isinstance(rows, list) or not rows:
        raise IntegrityError("raw JLens trace locations are missing")
    normalized: list[dict[str, object]] = []
    prior_key: tuple[int, int] | None = None
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "layer",
            "token_position",
            "phase",
            "concept_scores",
        }:
            raise IntegrityError("raw JLens location schema is invalid")
        layer = row["layer"]
        token_position = row["token_position"]
        phase = row["phase"]
        scores = row["concept_scores"]
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer < 0
            or isinstance(token_position, bool)
            or not isinstance(token_position, int)
            or token_position < 0
        ):
            raise IntegrityError("raw JLens layer/token position is invalid")
        _require_text("phase", phase)
        if not isinstance(scores, Mapping) or not scores:
            raise IntegrityError("raw JLens concept scores are missing")
        normalized_scores: dict[str, float] = {}
        for concept, score in sorted(scores.items()):
            _require_text("concept", concept)
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0 <= float(score) <= 1
            ):
                raise IntegrityError("raw JLens concept score is invalid")
            normalized_scores[str(concept)] = float(score)
        key = (layer, token_position)
        if prior_key is not None and key <= prior_key:
            raise IntegrityError("raw JLens locations must be strictly ordered")
        prior_key = key
        normalized.append(
            {
                "layer": layer,
                "token_position": token_position,
                "phase": phase,
                "concept_scores": normalized_scores,
            }
        )
    return normalized


def _normalize_expected_effect(value: Mapping[str, object]) -> dict[str, object]:
    required = {
        "concept",
        "phase",
        "min_final_score",
        "min_location_count",
        "require_non_decreasing",
    }
    if set(value) != required:
        raise IntegrityError("expected internal effect schema is invalid")
    concept = value["concept"]
    phase = value["phase"]
    threshold = value["min_final_score"]
    count = value["min_location_count"]
    monotonic = value["require_non_decreasing"]
    _require_text("concept", concept)
    _require_text("phase", phase)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0 <= float(threshold) <= 1
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or not isinstance(monotonic, bool)
    ):
        raise IntegrityError("expected internal effect values are invalid")
    return {
        "concept": concept,
        "phase": phase,
        "min_final_score": float(threshold),
        "min_location_count": count,
        "require_non_decreasing": monotonic,
    }


def _derive_observed_effect(
    locations: list[dict[str, object]], expected: Mapping[str, object]
) -> dict[str, object]:
    concept = str(expected["concept"])
    phase = str(expected["phase"])
    scores: list[float] = []
    for row in locations:
        concept_scores = row["concept_scores"]
        if not isinstance(concept_scores, Mapping):
            raise IntegrityError("normalized JLens concept scores are invalid")
        if row["phase"] == phase and concept in concept_scores:
            scores.append(float(concept_scores[concept]))
    if not scores:
        return {
            "concept": concept,
            "phase": phase,
            "location_count": 0,
            "first_score": None,
            "final_score": None,
            "minimum_score": None,
            "non_decreasing": False,
        }
    return {
        "concept": concept,
        "phase": phase,
        "location_count": len(scores),
        "first_score": scores[0],
        "final_score": scores[-1],
        "minimum_score": min(scores),
        "non_decreasing": all(
            current >= previous for previous, current in zip(scores, scores[1:])
        ),
    }


def _effect_is_consistent(
    observed: Mapping[str, object], expected: Mapping[str, object]
) -> bool:
    final_score = observed["final_score"]
    location_count = observed["location_count"]
    minimum_count = expected["min_location_count"]
    minimum_score = expected["min_final_score"]
    require_non_decreasing = expected["require_non_decreasing"]
    return (
        isinstance(final_score, float)
        and isinstance(location_count, int)
        and isinstance(minimum_count, int)
        and location_count >= minimum_count
        and isinstance(minimum_score, float)
        and final_score >= minimum_score
        and isinstance(require_non_decreasing, bool)
        and (not require_non_decreasing or observed["non_decreasing"] is True)
    )


def _json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _only_receipt(receipts: tuple[Receipt, ...], kind: str, plan_id: str) -> Receipt:
    matches = tuple(receipt for receipt in receipts if receipt.kind == kind)
    if len(matches) != 1 or matches[0].plan_id != plan_id:
        raise IntegrityError(f"trusted JLens issuer requires one {kind} receipt")
    return matches[0]


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("trusted observer clock must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


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
        "mechanism_prediction_receipt_id",
        "mechanism_prediction_artifact_sha256",
        "observer_config_sha256",
        "raw_trace_sha256",
        "expected_internal_effect_sha256",
        "observed_internal_effect_sha256",
        "effect_consistent",
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
        "mechanism_prediction_artifact_sha256",
        "observer_config_sha256",
        "raw_trace_sha256",
        "expected_internal_effect_sha256",
        "observed_internal_effect_sha256",
    ):
        _require_sha256(field, body[field])
    for field in required - {
        "observer_implementation_sha256",
        "model_artifact_sha256",
        "observation_artifact_sha256",
        "prediction_sha256",
        "mechanism_prediction_artifact_sha256",
        "observer_config_sha256",
        "raw_trace_sha256",
        "expected_internal_effect_sha256",
        "observed_internal_effect_sha256",
        "effect_consistent",
    }:
        _require_text(field, body[field])
    if not isinstance(body["effect_consistent"], bool):
        raise ValueError("effect_consistent must be boolean")
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


__all__ = [
    "TrustedJacobianLensObserver",
    "TrustedJLensReceiptIssuer",
    "TrustedObserverIdentity",
    "TrustedObserverKeyring",
    "derive_structured_jlens_observation",
    "issue_trusted_observation_attestation",
]
