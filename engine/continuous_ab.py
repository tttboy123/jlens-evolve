"""Permanent shadow baseline, matched A/B rounds, promotion, and sealed audit."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


class ABContractError(ValueError):
    """Raised when an A/B or sealed-audit contract is violated."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ABContractError(f"{field} must be a lowercase SHA-256")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _seal_integrity(payload: dict[str, Any]) -> dict[str, Any]:
    sealed = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    sealed["integrity_sha256"] = _sha256_json(sealed)
    return sealed


def _verify_integrity(payload: dict[str, Any], *, label: str) -> None:
    expected = payload.get("integrity_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "integrity_sha256"
    }
    if expected != _sha256_json(unsigned):
        raise ABContractError(f"{label} was tampered")


@dataclass(frozen=True)
class BaselineContract:
    experiment_id: str
    agent_program_sha256: str
    model: str
    reasoning: str
    token_budget: int
    timeout_seconds: int
    tools: tuple[str, ...]
    retries: int
    evaluator_epoch: str
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ABContractError("unsupported baseline contract schema")
        _require_sha256(self.agent_program_sha256, field="agent_program_sha256")
        if not self.experiment_id or not self.model or not self.reasoning:
            raise ABContractError("baseline identity fields must be non-empty")
        if self.token_budget < 1 or self.timeout_seconds < 1 or self.retries < 0:
            raise ABContractError("baseline budgets are invalid")
        if not self.tools or len(set(self.tools)) != len(self.tools):
            raise ABContractError("baseline tools must be unique and non-empty")
        if not self.evaluator_epoch:
            raise ABContractError("evaluator_epoch must be non-empty")

    @property
    def contract_sha256(self) -> str:
        return _sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tools"] = list(self.tools)
        return payload

    def replace(self, **changes: Any) -> BaselineContract:
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BaselineContract:
        return cls(**{**payload, "tools": tuple(payload["tools"])})


class PermanentBaselineAuthority:
    """Freeze one shadow baseline for the lifetime of an experiment."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def freeze(self, baseline: BaselineContract) -> dict[str, Any]:
        if self.path.exists():
            payload = self.load()
            if payload["baseline_contract_sha256"] != baseline.contract_sha256:
                raise ABContractError("permanent shadow baseline cannot be replaced")
            return payload
        payload = {
            "schema_version": "1.0",
            "baseline": baseline.to_dict(),
            "baseline_contract_sha256": baseline.contract_sha256,
            "freeze_count": 1,
            "immutable": True,
        }
        _atomic_json(self.path, payload)
        return payload

    def load(self) -> dict[str, Any]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1.0":
            raise ABContractError("unsupported baseline authority schema")
        baseline = BaselineContract.from_dict(payload["baseline"])
        if payload.get("baseline_contract_sha256") != baseline.contract_sha256:
            raise ABContractError("permanent shadow baseline authority was tampered")
        if payload.get("freeze_count") != 1 or payload.get("immutable") is not True:
            raise ABContractError("permanent shadow baseline authority is invalid")
        return payload


@dataclass(frozen=True)
class ArmResult:
    arm: str
    resolved: bool
    regression_failures: int
    safe: bool
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    benchmark_family: str
    evaluator_epoch: str

    def __post_init__(self) -> None:
        if self.arm not in {"baseline", "evolved"}:
            raise ABContractError("unknown arm")
        if self.regression_failures < 0:
            raise ABContractError("regression_failures cannot be negative")
        if self.input_tokens < 0 or self.output_tokens < 0 or self.elapsed_seconds < 0:
            raise ABContractError("result costs cannot be negative")
        if not self.benchmark_family or not self.evaluator_epoch:
            raise ABContractError("result family and evaluator must be non-empty")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def replace(self, **changes: Any) -> ArmResult:
        return replace(self, **changes)


class MatchedRoundLedger:
    PHASES = ("planned", "predictions_frozen", "evaluated", "retired")

    def __init__(self, path: Path, payload: dict[str, Any]) -> None:
        self.path = path.resolve()
        self.payload = payload

    @property
    def phase(self) -> str:
        return str(self.payload["phase"])

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        round_id: str,
        task_uid: str,
        baseline: BaselineContract,
        evolved_agent_sha256: str,
    ) -> MatchedRoundLedger:
        _require_sha256(task_uid, field="task_uid") if len(task_uid) == 64 else None
        _require_sha256(evolved_agent_sha256, field="evolved_agent_sha256")
        if not round_id or not task_uid:
            raise ABContractError("round_id and task_uid must be non-empty")
        payload = {
            "schema_version": "1.0",
            "round_id": round_id,
            "task_uid": task_uid,
            "phase": "planned",
            "baseline": baseline.to_dict(),
            "baseline_contract_sha256": baseline.contract_sha256,
            "evolved_agent_sha256": evolved_agent_sha256,
            "arm_contracts": {
                "baseline": baseline.to_dict(),
                "evolved": baseline.replace(
                    agent_program_sha256=evolved_agent_sha256
                ).to_dict(),
            },
            "events": [{"phase": "planned"}],
        }
        ledger = cls(path, payload)
        ledger._persist()
        return ledger

    @classmethod
    def load(cls, path: Path) -> MatchedRoundLedger:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _verify_integrity(payload, label="round ledger")
        if payload.get("schema_version") != "1.0":
            raise ABContractError("unsupported round ledger schema")
        if payload.get("phase") not in cls.PHASES:
            raise ABContractError("invalid round phase")
        baseline = BaselineContract.from_dict(payload["baseline"])
        if payload.get("baseline_contract_sha256") != baseline.contract_sha256:
            raise ABContractError("permanent shadow baseline contract was tampered")
        expected_arm_contracts = {
            "baseline": baseline.to_dict(),
            "evolved": baseline.replace(
                agent_program_sha256=payload.get("evolved_agent_sha256", "")
            ).to_dict(),
        }
        if payload.get("arm_contracts") != expected_arm_contracts:
            raise ABContractError("matched arm contracts were tampered")
        return cls(path, payload)

    @classmethod
    def resume(cls, path: Path, *, baseline: BaselineContract) -> MatchedRoundLedger:
        ledger = cls.load(path)
        if ledger.payload["baseline_contract_sha256"] != baseline.contract_sha256:
            raise ABContractError("permanent shadow baseline cannot be replaced")
        return ledger

    def _persist(self) -> None:
        self.payload = _seal_integrity(self.payload)
        _atomic_json(self.path, self.payload)

    def _advance(self, target: str, **evidence: Any) -> None:
        current = self.PHASES.index(self.phase)
        expected = self.PHASES[current + 1] if current + 1 < len(self.PHASES) else None
        if target != expected:
            raise ABContractError(
                f"invalid round transition: {self.phase} -> {target}; expected={expected}"
            )
        self.payload["phase"] = target
        self.payload["events"].append({"phase": target, **evidence})
        self._persist()

    def freeze_predictions(self, predictions: dict[str, Path]) -> None:
        if set(predictions) != {"baseline", "evolved"}:
            raise ABContractError("both arms must be frozen together")
        if self.phase != "planned":
            raise ABContractError("predictions can only be frozen once")
        rows = []
        for arm in ("baseline", "evolved"):
            path = predictions[arm].resolve()
            if not path.is_file():
                raise ABContractError(f"prediction missing for {arm}")
            rows.append(
                {
                    "arm": arm,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                    "frozen": True,
                }
            )
        self.payload["predictions"] = rows
        self._advance("predictions_frozen", prediction_count=2)

    def record_task_materialization(self, path: Path) -> dict[str, Any]:
        if self.phase != "planned":
            raise ABContractError("task input must be frozen before predictions")
        path = path.resolve()
        if not path.is_file():
            raise ABContractError("materialized task input is missing")
        evidence = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "frozen": True,
        }
        existing = self.payload.get("task_input")
        if existing is not None:
            if existing != evidence:
                raise ABContractError("materialized task input cannot be replaced")
            return existing
        self.payload["task_input"] = evidence
        self.payload["events"].append(
            {"phase": "planned", "event": "task_input_frozen", **evidence}
        )
        self._persist()
        return evidence

    def record_native_evidence(
        self,
        evidence_paths: dict[str, Path],
        results: dict[str, ArmResult],
    ) -> None:
        """Freeze both normalized native reports before admitting arm results."""

        if self.phase != "predictions_frozen":
            raise ABContractError("native evidence requires frozen predictions")
        if set(evidence_paths) != {"baseline", "evolved"} or set(results) != {
            "baseline",
            "evolved",
        }:
            raise ABContractError("both arms require native evidence")
        if "native_evidence" in self.payload:
            raise ABContractError("native evidence can only be frozen once")
        rows = []
        for arm in ("baseline", "evolved"):
            path = evidence_paths[arm].resolve()
            if not path.is_file():
                raise ABContractError(f"native evidence is missing for {arm}")
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ABContractError("native evidence is not valid JSON") from error
            if not isinstance(payload, dict):
                raise ABContractError("native evidence must be a JSON object")
            _verify_integrity(payload, label="native evidence")
            identity = payload.get("identity", payload.get("contract"))
            if not isinstance(identity, dict):
                raise ABContractError("native evidence identity is missing")
            expected_program = self.payload["arm_contracts"][arm][
                "agent_program_sha256"
            ]
            expected_identity = {
                "round_id": self.payload["round_id"],
                "arm": arm,
                "agent_program_sha256": expected_program,
                "baseline_contract_sha256": self.payload["baseline_contract_sha256"],
                "evaluator_epoch": self.payload["baseline"]["evaluator_epoch"],
            }
            if any(
                identity.get(key) != value for key, value in expected_identity.items()
            ):
                raise ABContractError("native evidence contract mismatch")
            if payload.get("result") != asdict(results[arm]):
                raise ABContractError("native evidence result mismatch")
            rows.append(
                {
                    "arm": arm,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                    "frozen": True,
                }
            )
        self.payload["native_evidence"] = rows
        self.payload["events"].append(
            {
                "phase": "predictions_frozen",
                "event": "native_evidence_frozen",
                "evidence_count": 2,
            }
        )
        self._persist()

    def record_results(self, results: dict[str, ArmResult]) -> None:
        if self.phase != "predictions_frozen":
            raise ABContractError("results require frozen predictions")
        if set(results) != {"baseline", "evolved"}:
            raise ABContractError("both arm results are required")
        for prediction in self.payload.get("predictions", []):
            path = Path(prediction["path"])
            if (
                not path.is_file()
                or path.stat().st_size != prediction["bytes"]
                or _sha256_file(path) != prediction["sha256"]
            ):
                raise ABContractError(
                    f"frozen prediction for {prediction['arm']} was tampered"
                )
        for evidence in self.payload.get("native_evidence", []):
            path = Path(evidence["path"])
            if (
                not path.is_file()
                or path.stat().st_size != evidence["bytes"]
                or _sha256_file(path) != evidence["sha256"]
            ):
                raise ABContractError(
                    f"native evidence for {evidence['arm']} was tampered"
                )
        baseline_contract = BaselineContract.from_dict(self.payload["baseline"])
        for arm, result in results.items():
            if result.arm != arm:
                raise ABContractError("result arm label mismatch")
            if result.evaluator_epoch != baseline_contract.evaluator_epoch:
                raise ABContractError("result evaluator does not match frozen epoch")
        self.payload["results"] = {
            arm: asdict(results[arm]) for arm in ("baseline", "evolved")
        }
        self._advance("evaluated", matched=True)

    def retire(self) -> None:
        if self.phase != "evaluated":
            raise ABContractError("only an evaluated round can retire")
        self._advance("retired", task_uid=self.payload["task_uid"])


@dataclass(frozen=True)
class ChangeSetCadence:
    interval: int = 10

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ABContractError("proposal interval must be positive")

    def can_propose(self, *, completed_rounds: int, last_proposal_round: int) -> bool:
        if completed_rounds < 0 or last_proposal_round < 0:
            raise ABContractError("round counters cannot be negative")
        return completed_rounds >= last_proposal_round + self.interval


@dataclass(frozen=True)
class ContinuousSchedule:
    """Predeclared 300-round cycle: observe, propose, promote, then seal."""

    cycles: int = 8
    search_rounds_per_cycle: int = 20
    promotion_rounds_per_cycle: int = 10
    final_sealed_rounds: int = 60
    proposal_interval_floor: int = 10

    def __post_init__(self) -> None:
        values = (
            self.cycles,
            self.search_rounds_per_cycle,
            self.promotion_rounds_per_cycle,
            self.final_sealed_rounds,
            self.proposal_interval_floor,
        )
        if any(value < 1 for value in values):
            raise ABContractError("continuous schedule values must be positive")

    def build(self) -> dict[str, Any]:
        rounds: list[dict[str, Any]] = []
        proposal_after_rounds: list[int] = []
        round_number = 0
        for cycle in range(1, self.cycles + 1):
            for cycle_round in range(1, self.search_rounds_per_cycle + 1):
                round_number += 1
                rounds.append(
                    {
                        "round_number": round_number,
                        "cycle": cycle,
                        "cycle_round": cycle_round,
                        "partition": "search",
                    }
                )
            proposal_after_rounds.append(round_number)
            for cycle_round in range(1, self.promotion_rounds_per_cycle + 1):
                round_number += 1
                rounds.append(
                    {
                        "round_number": round_number,
                        "cycle": cycle,
                        "cycle_round": cycle_round,
                        "partition": "promotion",
                    }
                )
        for sealed_round in range(1, self.final_sealed_rounds + 1):
            round_number += 1
            rounds.append(
                {
                    "round_number": round_number,
                    "cycle": "final",
                    "cycle_round": sealed_round,
                    "partition": "final_sealed",
                }
            )
        cadence = ChangeSetCadence(interval=self.proposal_interval_floor)
        last = 0
        for proposal_round in proposal_after_rounds:
            if not cadence.can_propose(
                completed_rounds=proposal_round,
                last_proposal_round=last,
            ):
                raise ABContractError("schedule violates ChangeSet cadence")
            last = proposal_round
        return {
            "schema_version": "1.0",
            "rounds": rounds,
            "proposal_after_rounds": proposal_after_rounds,
            "partition_counts": {
                "search": self.cycles * self.search_rounds_per_cycle,
                "promotion": self.cycles * self.promotion_rounds_per_cycle,
                "final_sealed": self.final_sealed_rounds,
            },
            "max_changesets": self.cycles,
            "final_sealed_opens_after_round": (
                self.cycles
                * (self.search_rounds_per_cycle + self.promotion_rounds_per_cycle)
            ),
        }


class ChangeSetRegistry:
    """Persistent proposal, promotion, and rollback authority."""

    def __init__(self, path: Path, payload: dict[str, Any]) -> None:
        self.path = path.resolve()
        self.payload = payload

    @property
    def active_agent_sha256(self) -> str:
        return str(self.payload["active_agent_sha256"])

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        initial_agent_sha256: str,
        cadence: ChangeSetCadence,
    ) -> ChangeSetRegistry:
        path = path.resolve()
        _require_sha256(initial_agent_sha256, field="initial_agent_sha256")
        if path.exists():
            registry = cls.load(path)
            if registry.payload["initial_agent_sha256"] != initial_agent_sha256:
                raise ABContractError("ChangeSet registry initial agent mismatch")
            if registry.payload["cadence_interval"] != cadence.interval:
                raise ABContractError("ChangeSet registry cadence mismatch")
            return registry
        payload = {
            "schema_version": "1.0",
            "initial_agent_sha256": initial_agent_sha256,
            "active_agent_sha256": initial_agent_sha256,
            "cadence_interval": cadence.interval,
            "last_proposal_round": 0,
            "proposals": [],
            "events": [
                {
                    "event": "registry_created",
                    "active_agent_sha256": initial_agent_sha256,
                }
            ],
        }
        registry = cls(path, payload)
        registry._persist()
        return registry

    @classmethod
    def load(cls, path: Path) -> ChangeSetRegistry:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _verify_integrity(payload, label="ChangeSet registry")
        if payload.get("schema_version") != "1.0":
            raise ABContractError("unsupported ChangeSet registry schema")
        _require_sha256(
            payload.get("initial_agent_sha256", ""), field="initial_agent_sha256"
        )
        _require_sha256(
            payload.get("active_agent_sha256", ""), field="active_agent_sha256"
        )
        ChangeSetCadence(interval=payload.get("cadence_interval", 0))
        if payload.get("last_proposal_round", -1) < 0:
            raise ABContractError("invalid ChangeSet proposal counter")
        if not isinstance(payload.get("proposals"), list):
            raise ABContractError("invalid ChangeSet proposal ledger")
        return cls(path, payload)

    def _persist(self) -> None:
        self.payload = _seal_integrity(self.payload)
        _atomic_json(self.path, self.payload)

    def _proposal(self, proposal_id: str) -> dict[str, Any]:
        matches = [
            proposal
            for proposal in self.payload["proposals"]
            if proposal["proposal_id"] == proposal_id
        ]
        if len(matches) != 1:
            raise ABContractError(f"unknown ChangeSet proposal: {proposal_id}")
        return matches[0]

    def propose(
        self,
        *,
        completed_rounds: int,
        candidate_agent_sha256: str,
        forward_patch: Path,
        rollback_patch: Path,
    ) -> dict[str, Any]:
        _require_sha256(candidate_agent_sha256, field="candidate_agent_sha256")
        cadence = ChangeSetCadence(interval=self.payload["cadence_interval"])
        last_round = self.payload["last_proposal_round"]
        if not cadence.can_propose(
            completed_rounds=completed_rounds,
            last_proposal_round=last_round,
        ):
            if completed_rounds == last_round:
                raise ABContractError(
                    "at most one ChangeSet is allowed per cadence boundary"
                )
            raise ABContractError("ChangeSet proposal cadence has not elapsed")
        paths = {
            "forward": forward_patch.resolve(),
            "rollback": rollback_patch.resolve(),
        }
        if any(not path.is_file() for path in paths.values()):
            raise ABContractError("ChangeSet forward and rollback patches are required")
        parent_sha256 = self.active_agent_sha256
        patch_evidence = {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        }
        proposal_id = _sha256_json(
            {
                "completed_rounds": completed_rounds,
                "parent_agent_sha256": parent_sha256,
                "candidate_agent_sha256": candidate_agent_sha256,
                "patch_evidence": patch_evidence,
            }
        )
        proposal = {
            "proposal_id": proposal_id,
            "completed_rounds": completed_rounds,
            "parent_agent_sha256": parent_sha256,
            "candidate_agent_sha256": candidate_agent_sha256,
            "patch_evidence": patch_evidence,
            "status": "proposed",
        }
        self.payload["proposals"].append(proposal)
        self.payload["last_proposal_round"] = completed_rounds
        self.payload["events"].append({"event": "proposed", "proposal_id": proposal_id})
        self._persist()
        return proposal

    def promote(
        self,
        proposal_id: str,
        decision: PromotionDecision,
    ) -> dict[str, Any]:
        proposal = self._proposal(proposal_id)
        if proposal["status"] != "proposed":
            raise ABContractError("ChangeSet is not awaiting promotion")
        if not decision.approved:
            raise ABContractError("failed promotion gate cannot be promoted")
        if proposal["parent_agent_sha256"] != self.active_agent_sha256:
            raise ABContractError("active agent changed after ChangeSet proposal")
        proposal["promotion_decision"] = asdict(decision)
        proposal["status"] = "promoted"
        self.payload["active_agent_sha256"] = proposal["candidate_agent_sha256"]
        self.payload["events"].append({"event": "promoted", "proposal_id": proposal_id})
        self._persist()
        return {
            "proposal_id": proposal_id,
            "active_agent_sha256": self.active_agent_sha256,
        }

    def reject(
        self,
        proposal_id: str,
        decision: PromotionDecision,
    ) -> dict[str, Any]:
        proposal = self._proposal(proposal_id)
        if proposal["status"] != "proposed":
            raise ABContractError("ChangeSet is not awaiting promotion")
        if decision.approved:
            raise ABContractError(
                "an approved ChangeSet cannot be recorded as rejected"
            )
        proposal["promotion_decision"] = asdict(decision)
        proposal["status"] = "rejected"
        self.payload["events"].append({"event": "rejected", "proposal_id": proposal_id})
        self._persist()
        return {
            "proposal_id": proposal_id,
            "active_agent_sha256": self.active_agent_sha256,
        }

    def rollback(self, proposal_id: str, *, reason: str) -> dict[str, Any]:
        proposal = self._proposal(proposal_id)
        if not reason.strip():
            raise ABContractError("rollback reason must be non-empty")
        if proposal["status"] != "promoted":
            raise ABContractError("only a promoted ChangeSet can be rolled back")
        if self.active_agent_sha256 != proposal["candidate_agent_sha256"]:
            raise ABContractError("ChangeSet is not the active agent")
        self.payload["active_agent_sha256"] = proposal["parent_agent_sha256"]
        proposal["status"] = "rolled_back"
        proposal["rollback_reason"] = reason
        self.payload["events"].append(
            {"event": "rolled_back", "proposal_id": proposal_id, "reason": reason}
        )
        self._persist()
        return {
            "proposal_id": proposal_id,
            "active_agent_sha256": self.active_agent_sha256,
        }


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    wins: int
    losses: int
    ties: int
    one_sided_sign_p: float
    quality_uplift: float
    cost_ratio: float
    checks: dict[str, bool]


class PromotionGate:
    """Predeclared paired quality, regression, safety, and cost admission gate."""

    def evaluate(self, pairs: list[tuple[ArmResult, ArmResult]]) -> PromotionDecision:
        if not pairs:
            raise ABContractError("promotion requires paired results")
        for baseline, evolved in pairs:
            if baseline.arm != "baseline" or evolved.arm != "evolved":
                raise ABContractError("promotion pairs must be baseline/evolved")
            if baseline.evaluator_epoch != evolved.evaluator_epoch:
                raise ABContractError("promotion pair evaluator mismatch")
            if baseline.benchmark_family != evolved.benchmark_family:
                raise ABContractError("promotion pair benchmark mismatch")

        wins = sum(
            not baseline.resolved and evolved.resolved for baseline, evolved in pairs
        )
        losses = sum(
            baseline.resolved and not evolved.resolved for baseline, evolved in pairs
        )
        ties = len(pairs) - wins - losses
        directional = wins + losses
        if directional:
            sign_p = sum(
                math.comb(directional, successes)
                for successes in range(wins, directional + 1)
            ) / (2**directional)
        else:
            sign_p = 1.0
        baseline_resolved = sum(baseline.resolved for baseline, _ in pairs)
        evolved_resolved = sum(evolved.resolved for _, evolved in pairs)
        uplift = (evolved_resolved - baseline_resolved) / len(pairs)
        baseline_tokens = sum(baseline.total_tokens for baseline, _ in pairs)
        evolved_tokens = sum(evolved.total_tokens for _, evolved in pairs)
        cost_ratio = (
            evolved_tokens / baseline_tokens if baseline_tokens else float("inf")
        )

        families = sorted({baseline.benchmark_family for baseline, _ in pairs})
        family_nonregression = all(
            sum(
                evolved.resolved
                for baseline, evolved in pairs
                if baseline.benchmark_family == family
            )
            >= sum(
                baseline.resolved
                for baseline, _ in pairs
                if baseline.benchmark_family == family
            )
            for family in families
        )
        checks = {
            "complete_promotion_block": len(pairs) >= 10,
            "wins_exceed_losses_by_three": wins - losses >= 3,
            "one_sided_exact_sign_p_lte_0_05": sign_p <= 0.05,
            "benchmark_family_nonregression": family_nonregression,
            "regression_failures_nonincrease": sum(
                evolved.regression_failures for _, evolved in pairs
            )
            <= sum(baseline.regression_failures for baseline, _ in pairs),
            "candidate_safety_100_percent": all(evolved.safe for _, evolved in pairs),
            "cost_gate": cost_ratio <= 1.10 or uplift >= 0.10,
        }
        return PromotionDecision(
            approved=all(checks.values()),
            wins=wins,
            losses=losses,
            ties=ties,
            one_sided_sign_p=sign_p,
            quality_uplift=uplift,
            cost_ratio=cost_ratio,
            checks=checks,
        )


class FinalSealedAuditor:
    """One-shot lock for a candidate frozen before opening final tasks."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def open(
        self,
        *,
        candidate_sha256: str,
        task_uids: list[str],
        used_task_uids: set[str] | frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        if self.path.exists():
            raise ABContractError("final sealed audit already opened")
        _require_sha256(candidate_sha256, field="candidate_sha256")
        if len(task_uids) < 60:
            raise ABContractError("final sealed audit requires at least 60 tasks")
        if len(task_uids) != len(set(task_uids)):
            raise ABContractError("final sealed task identities must be unique")
        overlap = sorted(set(task_uids) & set(used_task_uids))
        if overlap:
            raise ABContractError("final sealed tasks were previously used")
        payload = {
            "schema_version": "1.0",
            "opened_once": True,
            "candidate_sha256": candidate_sha256,
            "candidate_frozen_before_open": True,
            "task_count": len(task_uids),
            "task_uids": task_uids,
            "task_set_sha256": _sha256_json(task_uids),
        }
        _atomic_json(self.path, payload)
        return payload
