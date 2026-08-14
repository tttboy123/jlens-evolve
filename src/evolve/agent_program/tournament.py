"""Matched, hash-bound tournament authority for AgentProgram fixtures."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from evolve.contracts import ExecutionPlan, Receipt, content_sha256
from evolve.runtime import ExecutionResult

from .revision import AgentProgramViolation


@dataclass(frozen=True, slots=True)
class TournamentDecision:
    decision_id: str
    decision_sha256: str
    tournament_id: str
    campaign_id: str
    program_id: str
    execution_scope: str
    parent_revision_id: str
    participant_revision_ids: tuple[str, ...]
    task_revision_ids: tuple[str, ...]
    plan_ids: tuple[str, ...]
    receipt_ids: tuple[str, ...]
    program_bundle_sha256: tuple[tuple[str, str], ...]
    tournament_config_sha256: str
    matched_config_sha256: str
    score_policy_id: str
    scores: tuple[tuple[str, float], ...]
    winner_revision_id: str
    advanced: bool

    def __post_init__(self) -> None:
        expected = content_sha256(self.identity_payload())
        if self.decision_sha256 != expected:
            raise AgentProgramViolation("tournament decision hash mismatch")
        if self.decision_id != f"tournament-decision-{expected[:24]}":
            raise AgentProgramViolation("tournament decision identity mismatch")
        if self.execution_scope != "fixture":
            raise AgentProgramViolation("fixture tournament scope mismatch")
        if self.participant_revision_ids[0] != self.parent_revision_id:
            raise AgentProgramViolation("tournament parent ordering mismatch")
        if self.winner_revision_id not in self.participant_revision_ids:
            raise AgentProgramViolation("tournament winner is not a participant")
        if self.advanced != (self.winner_revision_id != self.parent_revision_id):
            raise AgentProgramViolation("tournament advance outcome mismatch")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "tournament_id": self.tournament_id,
            "campaign_id": self.campaign_id,
            "program_id": self.program_id,
            "execution_scope": self.execution_scope,
            "parent_revision_id": self.parent_revision_id,
            "participant_revision_ids": list(self.participant_revision_ids),
            "task_revision_ids": list(self.task_revision_ids),
            "plan_ids": list(self.plan_ids),
            "receipt_ids": list(self.receipt_ids),
            "program_bundle_sha256": [list(row) for row in self.program_bundle_sha256],
            "tournament_config_sha256": self.tournament_config_sha256,
            "matched_config_sha256": self.matched_config_sha256,
            "score_policy_id": self.score_policy_id,
            "scores": [list(row) for row in self.scores],
            "winner_revision_id": self.winner_revision_id,
            "advanced": self.advanced,
        }


class TournamentAuthority:
    """Select one fixture search parent only from a complete matched matrix."""

    score_policy_id = "fixture-mean-score-v1"

    def decide(
        self,
        *,
        plans: Sequence[ExecutionPlan],
        executions: Sequence[ExecutionResult],
    ) -> TournamentDecision:
        if not plans or len(plans) != len(executions):
            raise AgentProgramViolation("tournament execution matrix is incomplete")
        plan_ids = tuple(plan.plan_id for plan in plans)
        if len(set(plan_ids)) != len(plan_ids):
            raise AgentProgramViolation("tournament contains duplicate plans")
        campaign_ids = {plan.campaign_id for plan in plans}
        tournament_ids = {plan.metadata.get("tournament_id") for plan in plans}
        tournament_config_hashes = {
            plan.metadata.get("tournament_config_sha256") for plan in plans
        }
        if len(campaign_ids) != 1 or len(tournament_ids) != 1:
            raise AgentProgramViolation("tournament campaign identity drift")
        if len(tournament_config_hashes) != 1:
            raise AgentProgramViolation("tournament configuration drift")
        tournament_id = next(iter(tournament_ids))
        tournament_config_sha256 = next(iter(tournament_config_hashes))
        if not isinstance(tournament_id, str) or not tournament_id:
            raise AgentProgramViolation("tournament_id is missing")
        if (
            not isinstance(tournament_config_sha256, str)
            or len(tournament_config_sha256) != 64
        ):
            raise AgentProgramViolation("tournament config hash is invalid")
        matched_configs = {_matched_config(plan) for plan in plans}
        if len(matched_configs) != 1:
            raise AgentProgramViolation("tournament matched execution config drift")
        matched_config_sha256 = next(iter(matched_configs))

        parent_revisions = {
            plan.candidate_revision_id for plan in plans if plan.arm == "search-parent"
        }
        candidate_revisions = tuple(
            dict.fromkeys(
                plan.candidate_revision_id for plan in plans if plan.arm == "candidate"
            )
        )
        if len(parent_revisions) != 1 or not candidate_revisions:
            raise AgentProgramViolation("tournament requires parent and candidates")
        if any(plan.arm not in {"search-parent", "candidate"} for plan in plans):
            raise AgentProgramViolation("tournament arm is invalid")
        parent_revision_id = next(iter(parent_revisions))
        participants = (parent_revision_id, *candidate_revisions)
        if len(set(participants)) != len(participants):
            raise AgentProgramViolation("tournament revisions are duplicated")
        tasks = tuple(dict.fromkeys(plan.task.revision_id for plan in plans))
        cells = [(plan.task.revision_id, plan.candidate_revision_id) for plan in plans]
        expected_cells = {(task, revision) for task in tasks for revision in participants}
        if len(cells) != len(set(cells)):
            raise AgentProgramViolation("tournament contains duplicate execution cells")
        if set(cells) != expected_cells:
            raise AgentProgramViolation("tournament execution matrix is incomplete")

        scores: dict[str, list[float]] = {revision: [] for revision in participants}
        program_ids: set[str] = set()
        bundles: dict[str, str] = {}
        receipt_ids: list[str] = []
        for plan, execution in zip(plans, executions, strict=True):
            if execution.status != "completed":
                raise AgentProgramViolation("tournament contains a partial execution")
            model = _model_receipt(plan, execution)
            payload = model.payload
            if payload.get("execution_scope") != "fixture":
                raise AgentProgramViolation("tournament execution scope drift")
            if payload.get("revision_id") != plan.candidate_revision_id:
                raise AgentProgramViolation("tournament program revision drift")
            if payload.get("program_bundle_sha256") != plan.metadata.get(
                "program_bundle_sha256"
            ):
                raise AgentProgramViolation("tournament program bundle drift")
            if plan.arm == "candidate" and payload.get(
                "parent_revision_id"
            ) != parent_revision_id:
                raise AgentProgramViolation("candidate tournament parent drift")
            for name in (
                "program_prompt_sha256",
                "program_context_sha256",
                "program_tool_policy_sha256",
                "program_capabilities_sha256",
            ):
                if payload.get(name) != plan.metadata.get(name):
                    raise AgentProgramViolation(f"tournament {name} drift")
            score = payload.get("fixture_score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise AgentProgramViolation("tournament fixture score is invalid")
            revision_id = plan.candidate_revision_id
            scores[revision_id].append(float(score))
            program_id = payload.get("program_id")
            if not isinstance(program_id, str) or not program_id:
                raise AgentProgramViolation("tournament program_id is missing")
            program_ids.add(program_id)
            bundle = str(payload["program_bundle_sha256"])
            prior = bundles.setdefault(revision_id, bundle)
            if prior != bundle:
                raise AgentProgramViolation("tournament revision bundle drift")
            receipt_ids.append(model.receipt_id)
        if len(program_ids) != 1:
            raise AgentProgramViolation("tournament mixes AgentProgram identities")
        aggregate = tuple(
            (revision, sum(scores[revision]) / len(scores[revision]))
            for revision in participants
        )
        highest = max(score for _, score in aggregate)
        tied = [revision for revision, score in aggregate if score == highest]
        winner = parent_revision_id if parent_revision_id in tied else min(tied)
        identity = {
            "tournament_id": tournament_id,
            "campaign_id": next(iter(campaign_ids)),
            "program_id": next(iter(program_ids)),
            "execution_scope": "fixture",
            "parent_revision_id": parent_revision_id,
            "participant_revision_ids": list(participants),
            "task_revision_ids": list(tasks),
            "plan_ids": list(plan_ids),
            "receipt_ids": receipt_ids,
            "program_bundle_sha256": [
                [revision, bundles[revision]] for revision in participants
            ],
            "tournament_config_sha256": tournament_config_sha256,
            "matched_config_sha256": matched_config_sha256,
            "score_policy_id": self.score_policy_id,
            "scores": [list(row) for row in aggregate],
            "winner_revision_id": winner,
            "advanced": winner != parent_revision_id,
        }
        decision_sha256 = content_sha256(identity)
        return TournamentDecision(
            decision_id=f"tournament-decision-{decision_sha256[:24]}",
            decision_sha256=decision_sha256,
            tournament_id=tournament_id,
            campaign_id=next(iter(campaign_ids)),
            program_id=next(iter(program_ids)),
            execution_scope="fixture",
            parent_revision_id=parent_revision_id,
            participant_revision_ids=participants,
            task_revision_ids=tasks,
            plan_ids=plan_ids,
            receipt_ids=tuple(receipt_ids),
            program_bundle_sha256=tuple(
                (revision, bundles[revision]) for revision in participants
            ),
            tournament_config_sha256=tournament_config_sha256,
            matched_config_sha256=matched_config_sha256,
            score_policy_id=self.score_policy_id,
            scores=aggregate,
            winner_revision_id=winner,
            advanced=winner != parent_revision_id,
        )


def _model_receipt(plan: ExecutionPlan, execution: ExecutionResult) -> Receipt:
    if any(
        receipt.plan_id != plan.plan_id or receipt.campaign_id != plan.campaign_id
        for receipt in execution.receipts
    ):
        raise AgentProgramViolation("tournament receipt identity drift")
    models = tuple(receipt for receipt in execution.receipts if receipt.kind == "model")
    if len(models) != 1:
        raise AgentProgramViolation("tournament requires one model receipt per cell")
    model = models[0]
    if model.artifact_sha256 != content_sha256(model.payload):
        raise AgentProgramViolation("tournament model receipt artifact drift")
    return model


def _matched_config(plan: ExecutionPlan) -> str:
    return content_sha256(
        {
            "strategy_id": plan.strategy_id,
            "model": plan.model,
            "context_policy_id": plan.context_policy_id,
            "tool_policy_id": plan.tool_policy_id,
            "observer_policy_ids": plan.observer_policy_ids,
            "native_evaluator_id": plan.native_evaluator_id,
            "limits": plan.limits,
            "holdout_scope": plan.holdout_scope,
            "execution_profile": plan.metadata.get("execution_profile"),
            "tournament_config_sha256": plan.metadata.get(
                "tournament_config_sha256"
            ),
        }
    )


__all__ = ["TournamentAuthority", "TournamentDecision"]
