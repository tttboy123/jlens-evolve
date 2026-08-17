"""Mine score-labeled but non-causal Agent trajectory patterns."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
_SURFACE_ORDER = (
    "prompt",
    "skills",
    "policy",
    "router",
    "memory_policy",
    "constrained_harness_code",
)
_ALLOWED_SURFACES = frozenset(_SURFACE_ORDER)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class EvidenceFile:
    path: str
    sha256: str

    @classmethod
    def from_dict(cls, value: Any, *, name: str) -> EvidenceFile:
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise ValueError(f"invalid {name} evidence reference")
        path = value["path"]
        digest = value["sha256"]
        if not isinstance(path, str) or not path.strip() or path.startswith("/"):
            raise ValueError(f"invalid {name} evidence path")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"invalid {name} evidence sha256")
        return cls(path=path, sha256=digest)

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class FrozenObservationEvidence:
    schema_version: int
    evidence_id: str
    task_uid: str
    benchmark_family: str
    agent_program_sha256: str
    parent_agent_program_sha256: str
    native_evaluator_epoch: str
    native_score_delta: float
    safety_passed: bool
    observed_features: tuple[str, ...]
    conditions: tuple[str, ...]
    expected_surfaces: tuple[str, ...]
    evidence: dict[str, EvidenceFile]
    causal_boundary: str
    admission_gate_allowed: bool

    _FIELDS = frozenset(
        {
            "schema_version",
            "evidence_id",
            "task_uid",
            "benchmark_family",
            "agent_program_sha256",
            "parent_agent_program_sha256",
            "native_evaluator_epoch",
            "native_score_delta",
            "safety_passed",
            "observed_features",
            "conditions",
            "expected_surfaces",
            "evidence",
            "causal_boundary",
            "admission_gate_allowed",
        }
    )
    _EVIDENCE_FIELDS = frozenset(
        {"trajectory", "tool_events", "native_evaluator", "cost", "safety"}
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrozenObservationEvidence:
        if not isinstance(data, dict):
            raise TypeError("observation evidence must be a mapping")
        unknown = sorted(data.keys() - cls._FIELDS)
        missing = sorted(cls._FIELDS - data.keys())
        if unknown:
            raise ValueError(f"unknown evidence fields: {unknown}")
        if missing:
            raise ValueError(f"missing evidence fields: {missing}")
        if data["schema_version"] != 1:
            raise ValueError("unsupported observation evidence schema")
        for field in ("evidence_id", "task_uid", "benchmark_family"):
            if (
                not isinstance(data[field], str)
                or _IDENTIFIER.fullmatch(data[field]) is None
            ):
                raise ValueError(f"invalid {field}")
        for field in ("agent_program_sha256", "parent_agent_program_sha256"):
            if (
                not isinstance(data[field], str)
                or _SHA256.fullmatch(data[field]) is None
            ):
                raise ValueError(f"invalid {field}")
        if (
            not isinstance(data["native_evaluator_epoch"], str)
            or not data["native_evaluator_epoch"].strip()
        ):
            raise ValueError("invalid native evaluator epoch")
        if not isinstance(data["native_score_delta"], (int, float)):
            raise TypeError("native_score_delta must be numeric")
        if not isinstance(data["safety_passed"], bool):
            raise TypeError("safety_passed must be boolean")
        if data["causal_boundary"] != "observational_not_causal":
            raise ValueError("observation evidence lost causal boundary")
        if data["admission_gate_allowed"] is not False:
            raise ValueError("observation evidence cannot claim admission authority")
        evidence = data["evidence"]
        if not isinstance(evidence, dict) or set(evidence) != cls._EVIDENCE_FIELDS:
            raise ValueError("observation evidence file inventory mismatch")
        parsed_evidence = {
            name: EvidenceFile.from_dict(evidence[name], name=name)
            for name in sorted(evidence)
        }
        list_fields: dict[str, tuple[str, ...]] = {}
        for field in ("observed_features", "conditions", "expected_surfaces"):
            value = data[field]
            if (
                not isinstance(value, list)
                or not value
                or not all(
                    isinstance(item, str) and _IDENTIFIER.fullmatch(item)
                    for item in value
                )
                or len(set(value)) != len(value)
            ):
                raise ValueError(f"{field} must be a unique non-empty identifier list")
            list_fields[field] = tuple(value)
        unknown_surfaces = set(list_fields["expected_surfaces"]) - _ALLOWED_SURFACES
        if unknown_surfaces:
            raise ValueError(
                f"unsupported expected surfaces: {sorted(unknown_surfaces)}"
            )
        return cls(
            schema_version=1,
            evidence_id=data["evidence_id"],
            task_uid=data["task_uid"],
            benchmark_family=data["benchmark_family"],
            agent_program_sha256=data["agent_program_sha256"],
            parent_agent_program_sha256=data["parent_agent_program_sha256"],
            native_evaluator_epoch=data["native_evaluator_epoch"],
            native_score_delta=float(data["native_score_delta"]),
            safety_passed=data["safety_passed"],
            observed_features=list_fields["observed_features"],
            conditions=list_fields["conditions"],
            expected_surfaces=list_fields["expected_surfaces"],
            evidence=parsed_evidence,
            causal_boundary=data["causal_boundary"],
            admission_gate_allowed=False,
        )

    @property
    def outcome(self) -> str:
        if not self.safety_passed or self.native_score_delta < 0:
            return "failure"
        if self.native_score_delta > 0:
            return "advantage"
        return "neutral"

    @property
    def evidence_sha256s(self) -> tuple[str, ...]:
        return tuple(sorted({item.sha256 for item in self.evidence.values()}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "task_uid": self.task_uid,
            "benchmark_family": self.benchmark_family,
            "agent_program_sha256": self.agent_program_sha256,
            "parent_agent_program_sha256": self.parent_agent_program_sha256,
            "native_evaluator_epoch": self.native_evaluator_epoch,
            "native_score_delta": self.native_score_delta,
            "safety_passed": self.safety_passed,
            "observed_features": list(self.observed_features),
            "conditions": list(self.conditions),
            "expected_surfaces": list(self.expected_surfaces),
            "evidence": {
                name: value.to_dict() for name, value in self.evidence.items()
            },
            "causal_boundary": self.causal_boundary,
            "admission_gate_allowed": self.admission_gate_allowed,
        }


@dataclass(frozen=True)
class PatternCard:
    schema_version: int
    pattern_id: str
    pattern_kind: str
    observed_feature: str
    evidence_ids: tuple[str, ...]
    evidence_sha256s: tuple[str, ...]
    counterexample_evidence_ids: tuple[str, ...]
    support_count: int
    counterexample_count: int
    conditions: tuple[str, ...]
    expected_surfaces: tuple[str, ...]
    confidence: float
    causal_boundary: str
    admission_gate_allowed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pattern_id": self.pattern_id,
            "pattern_kind": self.pattern_kind,
            "observed_feature": self.observed_feature,
            "evidence_ids": list(self.evidence_ids),
            "evidence_sha256s": list(self.evidence_sha256s),
            "counterexample_evidence_ids": list(self.counterexample_evidence_ids),
            "support_count": self.support_count,
            "counterexample_count": self.counterexample_count,
            "conditions": list(self.conditions),
            "expected_surfaces": list(self.expected_surfaces),
            "confidence": self.confidence,
            "causal_boundary": self.causal_boundary,
            "admission_gate_allowed": self.admission_gate_allowed,
        }


class PatternAdvantageMiner:
    """Aggregate every observed success and failure without selecting candidates."""

    @staticmethod
    def _confidence(support: int, counterexamples: int) -> float:
        purity = support / (support + counterexamples)
        sample_weight = min(1.0, support / 3.0)
        return round(purity * sample_weight, 6)

    def mine(
        self, evidence: Iterable[FrozenObservationEvidence]
    ) -> tuple[PatternCard, ...]:
        rows = tuple(sorted(evidence, key=lambda item: item.evidence_id))
        ids = [row.evidence_id for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate observation evidence id")
        epochs = {row.native_evaluator_epoch for row in rows}
        if len(epochs) > 1:
            raise ValueError("pattern mining cannot mix native evaluator epochs")
        features = sorted(
            {feature for row in rows for feature in row.observed_features}
        )
        cards: list[PatternCard] = []
        for feature in features:
            members = tuple(row for row in rows if feature in row.observed_features)
            for kind in ("advantage", "failure"):
                support = tuple(row for row in members if row.outcome == kind)
                if not support:
                    continue
                counterexamples = tuple(row for row in members if row.outcome != kind)
                condition_sets = [set(row.conditions) for row in support]
                conditions = tuple(sorted(set.intersection(*condition_sets)))
                surfaces = tuple(
                    sorted(
                        {
                            surface
                            for row in support
                            for surface in row.expected_surfaces
                        },
                        key=_SURFACE_ORDER.index,
                    )
                )
                seed = {
                    "kind": kind,
                    "feature": feature,
                    "support": [row.evidence_id for row in support],
                    "counterexamples": [row.evidence_id for row in counterexamples],
                    "conditions": conditions,
                    "surfaces": surfaces,
                }
                pattern_id = (
                    f"pattern-{kind}-"
                    f"{hashlib.sha256(_canonical_json(seed).encode()).hexdigest()[:16]}"
                )
                cards.append(
                    PatternCard(
                        schema_version=1,
                        pattern_id=pattern_id,
                        pattern_kind=kind,
                        observed_feature=feature,
                        evidence_ids=tuple(row.evidence_id for row in support),
                        evidence_sha256s=tuple(
                            sorted(
                                {
                                    digest
                                    for row in support
                                    for digest in row.evidence_sha256s
                                }
                            )
                        ),
                        counterexample_evidence_ids=tuple(
                            row.evidence_id for row in counterexamples
                        ),
                        support_count=len(support),
                        counterexample_count=len(counterexamples),
                        conditions=conditions,
                        expected_surfaces=surfaces,
                        confidence=self._confidence(len(support), len(counterexamples)),
                        causal_boundary="observational_not_causal",
                        admission_gate_allowed=False,
                    )
                )
        return tuple(sorted(cards, key=lambda card: card.pattern_id))
