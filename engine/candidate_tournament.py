"""Native-evaluator-only candidate tournaments and search-parent selection."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ROLES = frozenset({"original", "parent", "candidate"})


class TournamentContractError(ValueError):
    """Raised when tournament evidence is incomplete, unmatched, or unsafe."""


def _validate_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TournamentContractError(f"invalid {name} sha256")
    return value


@dataclass(frozen=True)
class ArmEvaluation:
    schema_version: int
    task_uid: str
    benchmark_family: str
    role: str
    agent_program_sha256: str
    matched_contract_sha256: str
    native_evaluator_epoch: str
    native_score: float
    safety_passed: bool
    cost_units: float
    evidence_sha256: str

    _FIELDS = frozenset(
        {
            "schema_version",
            "task_uid",
            "benchmark_family",
            "role",
            "agent_program_sha256",
            "matched_contract_sha256",
            "native_evaluator_epoch",
            "native_score",
            "safety_passed",
            "cost_units",
            "evidence_sha256",
        }
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArmEvaluation:
        if not isinstance(data, dict):
            raise TournamentContractError("arm evaluation must be a mapping")
        unknown = sorted(data.keys() - cls._FIELDS)
        missing = sorted(cls._FIELDS - data.keys())
        if unknown:
            raise TournamentContractError(f"unknown arm fields: {unknown}")
        if missing:
            raise TournamentContractError(f"missing arm fields: {missing}")
        if data["schema_version"] != 1:
            raise TournamentContractError("unsupported arm evaluation schema")
        for field in ("task_uid", "benchmark_family", "native_evaluator_epoch"):
            if not isinstance(data[field], str) or not data[field].strip():
                raise TournamentContractError(f"invalid arm {field}")
        if data["role"] not in _ROLES:
            raise TournamentContractError("invalid arm role")
        if not isinstance(data["native_score"], (int, float)):
            raise TournamentContractError("native_score must be numeric")
        if not isinstance(data["safety_passed"], bool):
            raise TournamentContractError("safety_passed must be boolean")
        if not isinstance(data["cost_units"], (int, float)) or data["cost_units"] < 0:
            raise TournamentContractError("cost_units must be non-negative")
        return cls(
            schema_version=1,
            task_uid=data["task_uid"],
            benchmark_family=data["benchmark_family"],
            role=data["role"],
            agent_program_sha256=_validate_sha(
                data["agent_program_sha256"], "agent program"
            ),
            matched_contract_sha256=_validate_sha(
                data["matched_contract_sha256"], "matched contract"
            ),
            native_evaluator_epoch=data["native_evaluator_epoch"],
            native_score=float(data["native_score"]),
            safety_passed=data["safety_passed"],
            cost_units=float(data["cost_units"]),
            evidence_sha256=_validate_sha(data["evidence_sha256"], "evidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_uid": self.task_uid,
            "benchmark_family": self.benchmark_family,
            "role": self.role,
            "agent_program_sha256": self.agent_program_sha256,
            "matched_contract_sha256": self.matched_contract_sha256,
            "native_evaluator_epoch": self.native_evaluator_epoch,
            "native_score": self.native_score,
            "safety_passed": self.safety_passed,
            "cost_units": self.cost_units,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True)
class TournamentResult:
    schema_version: int
    stage: str
    task_uids: tuple[str, ...]
    original_sha256: str
    parent_sha256: str
    candidate_sha256s: tuple[str, ...]
    ranking: tuple[str, ...]
    finalists: tuple[str, ...]
    metrics: dict[str, dict[str, Any]]
    comparisons: dict[str, dict[str, Any]]
    native_evaluator_epoch: str
    ranking_authority: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "task_uids": list(self.task_uids),
            "original_sha256": self.original_sha256,
            "parent_sha256": self.parent_sha256,
            "candidate_sha256s": list(self.candidate_sha256s),
            "ranking": list(self.ranking),
            "finalists": list(self.finalists),
            "metrics": self.metrics,
            "comparisons": self.comparisons,
            "native_evaluator_epoch": self.native_evaluator_epoch,
            "ranking_authority": self.ranking_authority,
        }


@dataclass(frozen=True)
class ParentDecision:
    advance: bool
    previous_parent_sha256: str
    search_parent_sha256: str
    champion_sha256: str
    reasons: tuple[str, ...]
    scope: str = "experimental_search_lineage_only"
    production_promoted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "advance": self.advance,
            "previous_parent_sha256": self.previous_parent_sha256,
            "search_parent_sha256": self.search_parent_sha256,
            "champion_sha256": self.champion_sha256,
            "reasons": list(self.reasons),
            "scope": self.scope,
            "production_promoted": self.production_promoted,
        }


class CandidateTournament:
    """Rank candidates without accepting observer-derived score fields."""

    @staticmethod
    def _comparison(
        candidate_rows: tuple[ArmEvaluation, ...],
        baseline_rows: tuple[ArmEvaluation, ...],
    ) -> dict[str, Any]:
        candidate_by_task = {row.task_uid: row for row in candidate_rows}
        baseline_by_task = {row.task_uid: row for row in baseline_rows}
        wins = losses = ties = 0
        for task_uid in sorted(candidate_by_task):
            delta = (
                candidate_by_task[task_uid].native_score
                - baseline_by_task[task_uid].native_score
            )
            if delta > 1e-12:
                wins += 1
            elif delta < -1e-12:
                losses += 1
            else:
                ties += 1
        family_metrics = {}
        for family in sorted({row.benchmark_family for row in candidate_rows}):
            candidate_scores = [
                row.native_score
                for row in candidate_rows
                if row.benchmark_family == family
            ]
            baseline_scores = [
                row.native_score
                for row in baseline_rows
                if row.benchmark_family == family
            ]
            candidate_mean = fmean(candidate_scores)
            baseline_mean = fmean(baseline_scores)
            family_metrics[family] = {
                "candidate_mean": candidate_mean,
                "baseline_mean": baseline_mean,
                "delta": candidate_mean - baseline_mean,
                "noninferior": candidate_mean + 1e-12 >= baseline_mean,
            }
        candidate_cost = fmean(row.cost_units for row in candidate_rows)
        baseline_cost = fmean(row.cost_units for row in baseline_rows)
        cost_increase = (
            (candidate_cost - baseline_cost) / baseline_cost
            if baseline_cost > 0
            else 0.0
            if candidate_cost == 0
            else float("inf")
        )
        quality_delta = fmean(row.native_score for row in candidate_rows) - fmean(
            row.native_score for row in baseline_rows
        )
        return {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "quality_delta": quality_delta,
            "candidate_mean_cost": candidate_cost,
            "baseline_mean_cost": baseline_cost,
            "cost_increase_fraction": cost_increase,
            "families": family_metrics,
            "all_families_noninferior": all(
                value["noninferior"] for value in family_metrics.values()
            ),
        }

    def evaluate_stage(
        self,
        *,
        stage: str,
        evaluations: Iterable[ArmEvaluation],
        task_uids: tuple[str, ...],
        original_sha256: str,
        parent_sha256: str,
        candidate_sha256s: tuple[str, ...],
        advance_count: int,
    ) -> TournamentResult:
        if stage not in {"scout", "semifinal", "confirmation"}:
            raise TournamentContractError("unsupported tournament stage")
        if len(set(task_uids)) != len(task_uids) or not task_uids:
            raise TournamentContractError("task_uids must be unique and non-empty")
        _validate_sha(original_sha256, "original")
        _validate_sha(parent_sha256, "parent")
        if original_sha256 == parent_sha256:
            raise TournamentContractError("original and parent roles must be distinct")
        if (
            len(set(candidate_sha256s)) != len(candidate_sha256s)
            or not candidate_sha256s
        ):
            raise TournamentContractError(
                "candidate hashes must be unique and non-empty"
            )
        if not 1 <= advance_count <= len(candidate_sha256s):
            raise TournamentContractError("invalid tournament advance_count")
        for candidate in candidate_sha256s:
            _validate_sha(candidate, "candidate")
            if candidate in {original_sha256, parent_sha256}:
                raise TournamentContractError("candidate duplicates a baseline arm")
        rows = tuple(evaluations)
        expected_agents = {original_sha256, parent_sha256, *candidate_sha256s}
        expected_pairs = {
            (task, agent) for task in task_uids for agent in expected_agents
        }
        actual_pairs = {(row.task_uid, row.agent_program_sha256) for row in rows}
        if len(rows) != len(actual_pairs) or actual_pairs != expected_pairs:
            raise TournamentContractError(
                "tournament arm inventory is incomplete or duplicated"
            )
        epochs = {row.native_evaluator_epoch for row in rows}
        if len(epochs) != 1:
            raise TournamentContractError(
                "tournament cannot mix native evaluator epochs"
            )
        for task_uid in task_uids:
            task_rows = [row for row in rows if row.task_uid == task_uid]
            contracts = {row.matched_contract_sha256 for row in task_rows}
            families = {row.benchmark_family for row in task_rows}
            if len(contracts) != 1:
                raise TournamentContractError(
                    "matched contract differs across task arms"
                )
            if len(families) != 1:
                raise TournamentContractError(
                    "benchmark family differs across task arms"
                )
            for row in task_rows:
                expected_role = (
                    "original"
                    if row.agent_program_sha256 == original_sha256
                    else "parent"
                    if row.agent_program_sha256 == parent_sha256
                    else "candidate"
                )
                if row.role != expected_role:
                    raise TournamentContractError(
                        "arm role does not match AgentProgram hash"
                    )
        by_agent = {
            agent: tuple(
                sorted(
                    (row for row in rows if row.agent_program_sha256 == agent),
                    key=lambda row: row.task_uid,
                )
            )
            for agent in expected_agents
        }
        metrics = {}
        comparisons = {}
        for candidate in candidate_sha256s:
            candidate_rows = by_agent[candidate]
            metrics[candidate] = {
                "mean_native_score": fmean(row.native_score for row in candidate_rows),
                "mean_cost_units": fmean(row.cost_units for row in candidate_rows),
                "safety_failures": sum(not row.safety_passed for row in candidate_rows),
                "completed_tasks": len(candidate_rows),
            }
            comparisons[candidate] = {
                "vs_original": self._comparison(
                    candidate_rows, by_agent[original_sha256]
                ),
                "vs_parent": self._comparison(candidate_rows, by_agent[parent_sha256]),
            }
        ranking = tuple(
            sorted(
                candidate_sha256s,
                key=lambda candidate: (
                    metrics[candidate]["safety_failures"],
                    -metrics[candidate]["mean_native_score"],
                    metrics[candidate]["mean_cost_units"],
                    candidate,
                ),
            )
        )
        return TournamentResult(
            schema_version=1,
            stage=stage,
            task_uids=tuple(task_uids),
            original_sha256=original_sha256,
            parent_sha256=parent_sha256,
            candidate_sha256s=tuple(candidate_sha256s),
            ranking=ranking,
            finalists=ranking[:advance_count],
            metrics=metrics,
            comparisons=comparisons,
            native_evaluator_epoch=next(iter(epochs)),
            ranking_authority="fixed_native_evaluator_safety_cost_only",
        )


class SearchParentSelector:
    """Apply the frozen 12-task gate to an experimental lineage only."""

    def decide(self, result: TournamentResult) -> ParentDecision:
        if result.stage != "confirmation":
            raise TournamentContractError("search parent requires confirmation stage")
        champion = result.ranking[0]
        metrics = result.metrics[champion]
        vs_parent = result.comparisons[champion]["vs_parent"]
        vs_original = result.comparisons[champion]["vs_original"]
        reasons = []
        if len(result.task_uids) != 12 or metrics["completed_tasks"] != 12:
            reasons.append("confirmation_requires_exactly_12_complete_tasks")
        if metrics["safety_failures"] != 0:
            reasons.append("safety_failure")
        if not vs_parent["all_families_noninferior"]:
            reasons.append("benchmark_family_regression_vs_parent")
        if not vs_original["all_families_noninferior"]:
            reasons.append("benchmark_family_regression_vs_original")
        if vs_parent["wins"] < vs_parent["losses"] + 2:
            reasons.append("candidate_vs_parent_win_margin_below_two")
        if vs_original["wins"] < vs_original["losses"]:
            reasons.append("candidate_regresses_vs_original_shadow")
        if (
            vs_parent["cost_increase_fraction"] > 0.10 + 1e-12
            and vs_parent["quality_delta"] < 0.10 - 1e-12
        ):
            reasons.append("cost_increase_exceeds_quality_exception")
        advance = not reasons
        return ParentDecision(
            advance=advance,
            previous_parent_sha256=result.parent_sha256,
            search_parent_sha256=(champion if advance else result.parent_sha256),
            champion_sha256=champion,
            reasons=tuple(reasons),
        )


def result_sha256(result: TournamentResult) -> str:
    """Return a stable content digest for archive evidence links."""
    import hashlib

    payload = json.dumps(
        result.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
