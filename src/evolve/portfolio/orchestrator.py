"""Evidence-bound round trips across Skill and AgentProgram strategies.

The orchestrator owns coordination, not fact authority.  Failure Claims,
Skill native Claims, Governance decisions, and AgentProgram tournament Claims
must already be produced by their respective public boundaries.  This module
only verifies and links those immutable products.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from evolve.agent_program import AgentProgramRevision, AgentProgramViolation
from evolve.campaigns import CampaignRunResult, CampaignRunStatus
from evolve.contracts import (
    Claim,
    ClaimClassification,
    ClaimGrade,
    ContractViolation,
    Receipt,
    canonical_json,
    content_sha256,
)
from evolve.evidence import EvidenceGraph, IntegrityError, ReceiptStore
from evolve.governance import GateDecision, PromotionDecisionLog
from evolve.registry import (
    AgentProgramRecord,
    AgentProgramRegistry,
    CandidateRecord,
    CandidateRegistry,
    CapabilityRecord,
    CapabilityRegistry,
    RegistryViolation,
)
from evolve.strategies import StrategyStatus

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PortfolioViolation(ContractViolation):
    """A cross-strategy link is missing an authoritative immutable edge."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PortfolioViolation(f"{name} must be non-empty text")
    return value


def _sha(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PortfolioViolation(f"{name} must be a literal SHA-256")
    return value


def compiled_bundle_sha256(root: str | Path) -> str:
    """Hash a compiled bundle from its relative paths and exact file bytes."""

    bundle_root = Path(root).resolve()
    if not bundle_root.is_dir() or bundle_root.is_symlink():
        raise PortfolioViolation("compiled bundle root is not a directory")
    rows: list[tuple[str, str]] = []
    for path in sorted(bundle_root.rglob("*")):
        if path.is_symlink():
            raise PortfolioViolation("compiled bundle cannot contain symlinks")
        if path.is_file():
            rows.append(
                (
                    path.relative_to(bundle_root).as_posix(),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    if not rows:
        raise PortfolioViolation("compiled bundle is empty")
    return content_sha256(rows)


@dataclass(frozen=True, slots=True)
class CompiledSkillCandidate:
    candidate_id: str
    revision_id: str
    bundle_sha256: str
    bundle_root: Path
    source_gap_sha256: str
    active: bool = False

    def __post_init__(self) -> None:
        _text("candidate_id", self.candidate_id)
        _text("revision_id", self.revision_id)
        _sha("bundle_sha256", self.bundle_sha256)
        _sha("source_gap_sha256", self.source_gap_sha256)
        object.__setattr__(self, "bundle_root", Path(self.bundle_root).resolve())


@dataclass(frozen=True, slots=True)
class SkillValidationAuthority:
    candidate: CandidateRecord
    capability: CapabilityRecord
    claims: tuple[Claim, ...]
    receipt_store_root: Path
    evidence_graph_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "receipt_store_root", Path(self.receipt_store_root).resolve()
        )
        object.__setattr__(
            self, "evidence_graph_root", Path(self.evidence_graph_root).resolve()
        )


@dataclass(frozen=True, slots=True)
class TournamentAuthority:
    result: CampaignRunResult
    receipt_store_root: Path
    evidence_graph_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "receipt_store_root", Path(self.receipt_store_root).resolve()
        )
        object.__setattr__(
            self, "evidence_graph_root", Path(self.evidence_graph_root).resolve()
        )


class TeacherCompiler(Protocol):
    def compile(self, gap: CapabilityGap) -> CompiledSkillCandidate: ...


class SkillAuthority(Protocol):
    def validate(
        self, gap: CapabilityGap, candidate: CompiledSkillCandidate
    ) -> SkillValidationAuthority: ...


class TournamentRunner(Protocol):
    def run(
        self, parent: AgentProgramRevision, candidate: AgentProgramRevision
    ) -> TournamentAuthority: ...


@dataclass(frozen=True, slots=True)
class PortfolioRequest:
    round_trip_id: str
    parent_program_root: Path
    failure_claim: Claim
    failure_receipt_store_root: Path
    failure_evidence_graph_root: Path

    def __post_init__(self) -> None:
        _text("round_trip_id", self.round_trip_id)
        for name in (
            "parent_program_root",
            "failure_receipt_store_root",
            "failure_evidence_graph_root",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())


@dataclass(frozen=True, slots=True)
class CapabilityGap:
    gap_id: str
    round_trip_id: str
    program_id: str
    failed_program_revision_id: str
    failed_program_bundle_sha256: str
    failure_claim_id: str
    failure_claim_sha256: str
    failure_classification: str
    evidence_ids: tuple[str, ...]
    receipt_ids: tuple[str, ...]
    failure_signature_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "gap_id",
            "round_trip_id",
            "program_id",
            "failed_program_revision_id",
            "failure_claim_id",
            "failure_classification",
        ):
            _text(name, getattr(self, name))
        for name in (
            "failed_program_bundle_sha256",
            "failure_claim_sha256",
            "failure_signature_sha256",
        ):
            _sha(name, getattr(self, name))
        if not self.evidence_ids or not self.receipt_ids:
            raise PortfolioViolation("CapabilityGap authority references are incomplete")

    @property
    def content_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, slots=True)
class PortfolioDecision:
    decision_id: str
    round_trip_id: str
    gap_sha256: str
    skill_candidate_id: str
    skill_candidate_revision_id: str
    skill_candidate_bundle_sha256: str
    skill_candidate_bundle_root: str
    capability_id: str
    capability_revision_id: str
    capability_sha256: str
    promotion_decision_id: str
    agent_program_id: str
    parent_program_revision_id: str
    candidate_program_revision_id: str
    candidate_program_root: str
    candidate_program_bundle_sha256: str
    skill_claim_ids: tuple[str, ...]
    skill_claim_sha256: tuple[tuple[str, str], ...]
    skill_receipt_store_root: str
    skill_evidence_graph_root: str
    tournament_campaign_id: str
    tournament_id: str
    tournament_decision_sha256: str
    tournament_action: str
    search_parent_revision_id: str
    tournament_claim_ids: tuple[str, ...]
    tournament_claim_sha256: tuple[tuple[str, str], ...]
    tournament_receipt_store_root: str
    tournament_evidence_graph_root: str

    def __post_init__(self) -> None:
        for name in (
            "decision_id",
            "round_trip_id",
            "skill_candidate_id",
            "skill_candidate_revision_id",
            "skill_candidate_bundle_root",
            "capability_id",
            "capability_revision_id",
            "promotion_decision_id",
            "agent_program_id",
            "parent_program_revision_id",
            "candidate_program_revision_id",
            "candidate_program_root",
            "tournament_campaign_id",
            "tournament_id",
            "tournament_action",
            "search_parent_revision_id",
        ):
            _text(name, getattr(self, name))
        for name in (
            "gap_sha256",
            "skill_candidate_bundle_sha256",
            "capability_sha256",
            "candidate_program_bundle_sha256",
            "tournament_decision_sha256",
        ):
            _sha(name, getattr(self, name))
        if not self.skill_claim_ids or not self.tournament_claim_ids:
            raise PortfolioViolation("PortfolioDecision Claim authority is incomplete")

    @property
    def content_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, slots=True)
class PortfolioResult:
    gap: CapabilityGap
    decision: PortfolioDecision
    capability: CapabilityRecord
    program: AgentProgramRevision
    replayed: bool


class PortfolioOrchestrator:
    """Coordinate one authority-bound cross-strategy round trip."""

    def __init__(
        self,
        root: str | Path,
        *,
        candidate_registry: CandidateRegistry,
        capability_registry: CapabilityRegistry,
        agent_program_registry: AgentProgramRegistry,
        promotion_decision_log: PromotionDecisionLog,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._candidate_registry = candidate_registry
        self._capability_registry = capability_registry
        self._agent_program_registry = agent_program_registry
        self._promotion_log = promotion_decision_log
        self._gaps = _PortfolioLog(
            self.root / "capability-gaps.jsonl", CapabilityGap, "gap_id"
        )
        self._decisions = _PortfolioLog(
            self.root / "portfolio-decisions.jsonl",
            PortfolioDecision,
            "round_trip_id",
        )

    def run(
        self,
        request: PortfolioRequest,
        *,
        teacher: TeacherCompiler,
        skill_authority: SkillAuthority,
        tournament: TournamentRunner,
    ) -> PortfolioResult:
        parent = self._load_parent(request.parent_program_root)
        gap = self._capability_gap(request, parent)
        self._gaps.append(gap)
        prior = self._decisions.get(request.round_trip_id)
        if prior is not None:
            return self._replay(gap, prior, parent)

        candidate = teacher.compile(gap)
        self._validate_compiled_candidate(candidate, gap)
        validation = skill_authority.validate(gap, candidate)
        self._validate_compiled_candidate(candidate, gap)
        capability = self._validate_skill_authority(candidate, validation)
        program = self._compose_program(parent, gap, candidate, capability)
        tournament_result = tournament.run(parent, program)
        self._validate_compiled_candidate(candidate, gap)
        action, winner, claims, tournament_decision_sha256 = self._validate_tournament(
            parent, program, tournament_result
        )
        decision = self._decision(
            request=request,
            gap=gap,
            candidate=candidate,
            validation=validation,
            capability=capability,
            program=program,
            tournament=tournament_result,
            tournament_action=action,
            tournament_winner=winner,
            tournament_claims=claims,
            tournament_decision_sha256=tournament_decision_sha256,
        )
        self._decisions.append(decision)
        return PortfolioResult(gap, decision, capability, program, replayed=False)

    def _load_parent(self, root: Path) -> AgentProgramRevision:
        try:
            parent = AgentProgramRevision.load(root)
        except AgentProgramViolation as error:
            raise PortfolioViolation("parent AgentProgram failed verification") from error
        record = self._agent_program_registry.get(parent.program_id, parent.revision_id)
        expected = AgentProgramRecord(
            program_id=parent.program_id,
            revision_id=parent.revision_id,
            parent_revision_id=parent.parent_revision_id,
            capability_revision_ids=parent.capability_revision_ids,
            artifact_sha256=parent.bundle_sha256,
        )
        if record != expected:
            raise PortfolioViolation("parent AgentProgram is not registry-authoritative")
        return parent

    def _capability_gap(
        self, request: PortfolioRequest, parent: AgentProgramRevision
    ) -> CapabilityGap:
        try:
            store = ReceiptStore(request.failure_receipt_store_root)
            graph = EvidenceGraph.rebuild(request.failure_evidence_graph_root, store)
            latest = {
                claim.claim_id: claim for claim in graph.latest_claims()
            }.get(request.failure_claim.claim_id)
            if (
                latest != request.failure_claim
                or latest is None
                or latest.content_sha256 != request.failure_claim.content_sha256
            ):
                raise PortfolioViolation("failure Claim is not authoritative")
            if latest.grade < ClaimGrade.E1:
                raise PortfolioViolation("failure Claim evidence grade is insufficient")
            if latest.candidate_id != parent.revision_id or latest.classification not in {
                ClaimClassification.NEUTRAL,
                ClaimClassification.REGRESSION,
            }:
                raise PortfolioViolation("failure Claim does not bind the parent")
            evidence = {
                row.evidence_id: row for row in graph.list_evidence()
            }
            selected = tuple(evidence.get(item) for item in latest.evidence_ids)
            if any(row is None for row in selected):
                raise PortfolioViolation("failure Claim evidence is incomplete")
            receipts = {
                row.receipt_id: row for row in store.list_receipts()
            }
            native: list[Receipt] = []
            receipt_ids: set[str] = set()
            for envelope in selected:
                if envelope is None:
                    raise PortfolioViolation("failure Claim evidence is incomplete")
                receipt_ids.update(envelope.receipt_ids)
                native.extend(
                    receipt
                    for receipt_id in envelope.receipt_ids
                    if (receipt := receipts.get(receipt_id)) is not None
                    and receipt.kind == "native_evaluation"
                )
            if len(native) != 1:
                raise PortfolioViolation("failure Claim must bind one native outcome")
            outcome = native[0]
            model_id = outcome.payload.get("model_receipt_id")
            model = receipts.get(model_id) if isinstance(model_id, str) else None
            if (
                outcome.payload.get("resolved") is not False
                or outcome.payload.get("evaluator_error") not in (None, "")
                or model is None
                or model.kind != "model"
                or outcome.payload.get("model_artifact_sha256")
                != model.artifact_sha256
                or model.payload.get("revision_id") != parent.revision_id
                or model.payload.get("program_bundle_sha256")
                != parent.bundle_sha256
            ):
                raise PortfolioViolation("failure native/model lineage is invalid")
            receipt_ids.add(model.receipt_id)
            signature = content_sha256(
                {
                    "claim_sha256": latest.content_sha256,
                    "native_receipt_sha256": outcome.content_sha256,
                    "model_receipt_sha256": model.content_sha256,
                }
            )
        except PortfolioViolation:
            raise
        except (OSError, ValueError, IntegrityError) as error:
            raise PortfolioViolation("failure Claim is not authoritative") from error
        payload = {
            "round_trip_id": request.round_trip_id,
            "program_id": parent.program_id,
            "failed_program_revision_id": parent.revision_id,
            "failed_program_bundle_sha256": parent.bundle_sha256,
            "failure_claim_id": latest.claim_id,
            "failure_claim_sha256": latest.content_sha256,
            "failure_classification": str(latest.classification),
            "evidence_ids": list(latest.evidence_ids),
            "receipt_ids": sorted(receipt_ids),
            "failure_signature_sha256": signature,
        }
        return CapabilityGap(
            gap_id="gap-" + content_sha256(payload),
            round_trip_id=request.round_trip_id,
            program_id=parent.program_id,
            failed_program_revision_id=parent.revision_id,
            failed_program_bundle_sha256=parent.bundle_sha256,
            failure_claim_id=latest.claim_id,
            failure_claim_sha256=latest.content_sha256,
            failure_classification=str(latest.classification),
            evidence_ids=latest.evidence_ids,
            receipt_ids=tuple(sorted(receipt_ids)),
            failure_signature_sha256=signature,
        )

    @staticmethod
    def _validate_compiled_candidate(
        candidate: CompiledSkillCandidate, gap: CapabilityGap
    ) -> None:
        if not isinstance(candidate, CompiledSkillCandidate):
            raise PortfolioViolation("Teacher returned an invalid compiled candidate")
        if candidate.active:
            raise PortfolioViolation("Teacher candidate must remain inactive")
        if candidate.source_gap_sha256 != gap.content_sha256:
            raise PortfolioViolation("Teacher candidate gap lineage mismatch")
        if compiled_bundle_sha256(candidate.bundle_root) != candidate.bundle_sha256:
            raise PortfolioViolation("Teacher candidate bundle hash mismatch")

    def _validate_skill_authority(
        self,
        compiled: CompiledSkillCandidate,
        authority: SkillValidationAuthority,
    ) -> CapabilityRecord:
        if not isinstance(authority, SkillValidationAuthority):
            raise PortfolioViolation("Skill validation authority is invalid")
        candidate = authority.candidate
        capability = authority.capability
        claims = tuple(authority.claims)
        if (
            candidate.candidate_id != compiled.candidate_id
            or candidate.revision_id != compiled.revision_id
            or candidate.artifact_sha256 != compiled.bundle_sha256
            or candidate.active
        ):
            raise PortfolioViolation("validated Candidate bundle identity mismatch")
        if (
            not claims
            or any(
                claim.candidate_id != compiled.candidate_id
                or claim.classification is not ClaimClassification.GAIN
                or claim.grade < ClaimGrade.E2
                for claim in claims
            )
            or set(candidate.source_claim_ids)
            != {claim.claim_id for claim in claims}
        ):
            raise PortfolioViolation("Skill validation lacks native GAIN Claims")
        self._verify_claims(
            claims,
            receipt_store_root=authority.receipt_store_root,
            evidence_graph_root=authority.evidence_graph_root,
        )
        self._verify_skill_claim_lineage(compiled, claims, authority.receipt_store_root)
        if (
            capability.active
            or capability.source_candidate_id != compiled.candidate_id
            or capability.revision_id != compiled.revision_id
            or capability.artifact_sha256 != compiled.bundle_sha256
            or set(capability.evidence_claim_ids)
            != {claim.claim_id for claim in claims}
            or capability.promotion_decision_id is None
        ):
            raise PortfolioViolation("capability bundle identity mismatch")
        try:
            decision = self._promotion_log.verified_approved(
                capability.promotion_decision_id
            )
        except ContractViolation as error:
            raise PortfolioViolation("capability lacks Governance approval") from error
        if (
            decision.gate_decision is not GateDecision.APPROVED
            or decision.evidence_grade is not ClaimGrade.E3
            or decision.candidate_id != compiled.candidate_id
            or decision.candidate_revision_id != compiled.revision_id
            or set(decision.claim_ids) != {claim.claim_id for claim in claims}
        ):
            raise PortfolioViolation("capability Governance identity mismatch")
        try:
            self._candidate_registry.append(candidate)
            self._capability_registry.append(capability)
        except RegistryViolation as error:
            raise PortfolioViolation("authoritative registry projection failed") from error
        return capability

    @staticmethod
    def _verify_skill_claim_lineage(
        compiled: CompiledSkillCandidate,
        claims: tuple[Claim, ...],
        receipt_store_root: Path,
    ) -> None:
        receipts = {
            receipt.receipt_id: receipt
            for receipt in ReceiptStore(receipt_store_root).list_receipts()
        }
        for claim in claims:
            bound = tuple(
                receipts.get(receipt_id)
                for receipt_id in claim.counterfactual_receipt_ids
            )
            models = tuple(
                receipt
                for receipt in bound
                if receipt is not None and receipt.kind == "model"
            )
            if len(models) != 2:
                raise PortfolioViolation("Skill Claim model lineage is incomplete")
            baseline = tuple(
                receipt
                for receipt in models
                if receipt.payload.get("candidate_consumed") is False
            )
            taught = tuple(
                receipt
                for receipt in models
                if receipt.payload.get("candidate_consumed") is True
            )
            if (
                len(baseline) != 1
                or len(taught) != 1
                or baseline[0].payload.get("candidate_revision_id") is not None
                or baseline[0].payload.get("candidate_bundle_sha256") is not None
                or taught[0].payload.get("candidate_revision_id")
                != compiled.revision_id
                or taught[0].payload.get("candidate_bundle_sha256")
                != compiled.bundle_sha256
            ):
                raise PortfolioViolation("Skill Claim Candidate bundle mismatch")

    def _compose_program(
        self,
        parent: AgentProgramRevision,
        gap: CapabilityGap,
        candidate: CompiledSkillCandidate,
        capability: CapabilityRecord,
    ) -> AgentProgramRevision:
        if capability.revision_id in parent.capability_revision_ids:
            raise PortfolioViolation("parent already references the capability")
        capabilities = (*parent.capability_revision_ids, capability.revision_id)
        lineage = {
            "gap_sha256": gap.content_sha256,
            "skill_candidate_id": candidate.candidate_id,
            "skill_candidate_revision_id": candidate.revision_id,
            "skill_candidate_bundle_sha256": candidate.bundle_sha256,
            "capability_id": capability.capability_id,
            "capability_record_sha256": capability.content_sha256,
        }
        revision_id = "program-" + content_sha256(
            {
                "parent_bundle_sha256": parent.bundle_sha256,
                "capabilities": list(capabilities),
                "portfolio_lineage": lineage,
            }
        )[:24]
        output = self.root / "agent-program-revisions" / revision_id
        context = dict(parent.context)
        context["portfolio_lineage"] = lineage
        if output.exists():
            try:
                program = AgentProgramRevision.load(output)
            except AgentProgramViolation as error:
                raise PortfolioViolation("existing AgentProgram is corrupt") from error
        else:
            program = AgentProgramRevision.freeze(
                output,
                program_id=parent.program_id,
                revision_id=revision_id,
                parent_revision_id=parent.revision_id,
                program_prompt=parent.program_prompt,
                context=context,
                tool_policy=parent.tool_policy,
                capability_revision_ids=capabilities,
            )
        if (
            program.program_id != parent.program_id
            or program.parent_revision_id != parent.revision_id
            or program.program_prompt != parent.program_prompt
            or dict(program.context) != context
            or program.tool_policy != parent.tool_policy
            or program.capability_revision_ids != capabilities
        ):
            raise PortfolioViolation("AgentProgram composition identity mismatch")
        record = AgentProgramRecord(
            program_id=program.program_id,
            revision_id=program.revision_id,
            parent_revision_id=program.parent_revision_id,
            capability_revision_ids=program.capability_revision_ids,
            artifact_sha256=program.bundle_sha256,
        )
        try:
            self._agent_program_registry.append(record)
        except RegistryViolation as error:
            raise PortfolioViolation("AgentProgram registry projection failed") from error
        return program

    def _validate_tournament(
        self,
        parent: AgentProgramRevision,
        candidate: AgentProgramRevision,
        authority: TournamentAuthority,
    ) -> tuple[str, str, tuple[Claim, ...], str]:
        if not isinstance(authority, TournamentAuthority):
            raise PortfolioViolation("tournament authority is invalid")
        result = authority.result
        if (
            result.status is not CampaignRunStatus.COMPLETED
            or len(result.plans) != 2
            or len(result.decisions) != 1
            or tuple(plan.arm for plan in result.plans)
            != ("search-parent", "candidate")
            or tuple(plan.candidate_revision_id for plan in result.plans)
            != (parent.revision_id, candidate.revision_id)
        ):
            raise PortfolioViolation("live AgentProgram tournament is incomplete")
        expected_bundles = {
            parent.revision_id: parent.bundle_sha256,
            candidate.revision_id: candidate.bundle_sha256,
        }
        if any(
            plan.metadata.get("execution_profile") != "live"
            or plan.metadata.get("program_bundle_sha256")
            != expected_bundles[plan.candidate_revision_id]
            for plan in result.plans
        ):
            raise PortfolioViolation("tournament program bundle mismatch")
        claims = tuple(result.claims)
        if (
            len(claims) != 2
            or {claim.candidate_id for claim in claims}
            != {parent.revision_id, candidate.revision_id}
        ):
            raise PortfolioViolation("tournament Claims are incomplete")
        self._verify_claims(
            claims,
            receipt_store_root=authority.receipt_store_root,
            evidence_graph_root=authority.evidence_graph_root,
        )
        store = ReceiptStore(authority.receipt_store_root)
        receipts = {row.receipt_id: row for row in store.list_receipts()}
        evidence = {
            row.evidence_id: row
            for row in EvidenceGraph(authority.evidence_graph_root).list_evidence()
        }
        plans = {plan.candidate_revision_id: plan for plan in result.plans}
        for claim in claims:
            plan = plans[claim.candidate_id]
            selected = tuple(evidence.get(item) for item in claim.evidence_ids)
            native = [
                receipt
                for envelope in selected
                if envelope is not None
                and envelope.payload.get("plan_id") == plan.plan_id
                for receipt_id in envelope.receipt_ids
                if (receipt := receipts.get(receipt_id)) is not None
                and receipt.kind == "native_evaluation"
            ]
            if len(native) != 1:
                raise PortfolioViolation("tournament Claim lacks native authority")
            model_id = native[0].payload.get("model_receipt_id")
            model = receipts.get(model_id) if isinstance(model_id, str) else None
            if (
                model is None
                or model.kind != "model"
                or native[0].payload.get("model_artifact_sha256")
                != model.artifact_sha256
                or model.payload.get("revision_id") != claim.candidate_id
                or model.payload.get("program_bundle_sha256")
                != expected_bundles[claim.candidate_id]
            ):
                raise PortfolioViolation("tournament model/native lineage mismatch")
            evaluator_error = native[0].payload.get("evaluator_error")
            resolved = native[0].payload.get("resolved")
            if evaluator_error not in (None, ""):
                expected_classification = ClaimClassification.INFRA_FAILURE
            elif resolved is True:
                expected_classification = ClaimClassification.GAIN
            elif resolved is False:
                expected_classification = ClaimClassification.NEUTRAL
            else:
                raise PortfolioViolation("tournament native outcome is not boolean")
            if (
                claim.grade is not ClaimGrade.E1
                or claim.classification is not expected_classification
            ):
                raise PortfolioViolation(
                    "tournament Claim classification contradicts native outcome"
                )
        ordered_claims = tuple(sorted(claims, key=lambda claim: claim.claim_id))
        weights = {
            ClaimClassification.GAIN: 2,
            ClaimClassification.NEUTRAL: 0,
            ClaimClassification.REGRESSION: -2,
            ClaimClassification.INFRA_FAILURE: -1,
        }
        scores = {
            revision_id: next(
                weights[claim.classification]
                for claim in ordered_claims
                if claim.candidate_id == revision_id
            )
            for revision_id in (parent.revision_id, candidate.revision_id)
        }
        highest = max(scores.values())
        tied = {
            revision_id for revision_id, score in scores.items() if score == highest
        }
        winner = (
            parent.revision_id
            if parent.revision_id in tied
            else min(tied)
        )
        decision = result.decisions[0]
        decision_sha256 = content_sha256(
            {
                "tournament_id": result.plans[0].metadata.get("tournament_id"),
                "execution_scope": "live",
                "parent_revision_id": parent.revision_id,
                "participant_revision_ids": [
                    parent.revision_id,
                    candidate.revision_id,
                ],
                "program_bundle_sha256": [
                    [plan.candidate_revision_id, plan.metadata["program_bundle_sha256"]]
                    for plan in result.plans
                ],
                "claim_ids": [claim.claim_id for claim in ordered_claims],
                "claim_sha256": [
                    [claim.claim_id, claim.content_sha256]
                    for claim in ordered_claims
                ],
                "scores": [
                    [revision_id, scores[revision_id]]
                    for revision_id in (parent.revision_id, candidate.revision_id)
                ],
                "winner_revision_id": winner,
            }
        )
        expected_reason = (
            f"decision={decision_sha256};winner={winner};"
            "scope=live;promotion_claimed=false"
        )
        expected_action = (
            "advance-search-parent"
            if winner == candidate.revision_id
            else "reject-candidates"
        )
        if (
            decision.status is not StrategyStatus.LIVE
            or decision.action != expected_action
            or decision.claim_ids
            != tuple(claim.claim_id for claim in ordered_claims)
            or decision.reason != expected_reason
        ):
            raise PortfolioViolation("tournament decision is not authority-bound")
        return decision.action, winner, claims, decision_sha256

    @staticmethod
    def _verify_claims(
        claims: tuple[Claim, ...],
        *,
        receipt_store_root: Path,
        evidence_graph_root: Path,
    ) -> None:
        try:
            store = ReceiptStore(receipt_store_root)
            graph = EvidenceGraph.rebuild(evidence_graph_root, store)
            persisted = {claim.claim_id: claim for claim in graph.latest_claims()}
        except (OSError, ValueError, IntegrityError) as error:
            raise PortfolioViolation("Claim authority replay failed") from error
        if (
            len({claim.claim_id for claim in claims}) != len(claims)
            or any(
                persisted.get(claim.claim_id) != claim
                or persisted[claim.claim_id].content_sha256 != claim.content_sha256
                for claim in claims
            )
        ):
            raise PortfolioViolation("Claim is not authoritative")

    def _decision(
        self,
        *,
        request: PortfolioRequest,
        gap: CapabilityGap,
        candidate: CompiledSkillCandidate,
        validation: SkillValidationAuthority,
        capability: CapabilityRecord,
        program: AgentProgramRevision,
        tournament: TournamentAuthority,
        tournament_action: str,
        tournament_winner: str,
        tournament_claims: tuple[Claim, ...],
        tournament_decision_sha256: str,
    ) -> PortfolioDecision:
        payload = {
            "round_trip_id": request.round_trip_id,
            "gap_sha256": gap.content_sha256,
            "skill_candidate_bundle_sha256": candidate.bundle_sha256,
            "capability_sha256": capability.content_sha256,
            "candidate_program_bundle_sha256": program.bundle_sha256,
            "tournament_campaign_id": tournament.result.campaign_id,
            "tournament_action": tournament_action,
            "tournament_decision_sha256": tournament_decision_sha256,
            "search_parent_revision_id": tournament_winner,
            "skill_claim_sha256": sorted(
                (claim.claim_id, claim.content_sha256)
                for claim in validation.claims
            ),
            "tournament_claim_sha256": sorted(
                (claim.claim_id, claim.content_sha256)
                for claim in tournament_claims
            ),
        }
        return PortfolioDecision(
            decision_id="portfolio-decision-" + content_sha256(payload),
            round_trip_id=request.round_trip_id,
            gap_sha256=gap.content_sha256,
            skill_candidate_id=candidate.candidate_id,
            skill_candidate_revision_id=candidate.revision_id,
            skill_candidate_bundle_sha256=candidate.bundle_sha256,
            skill_candidate_bundle_root=str(candidate.bundle_root),
            capability_id=capability.capability_id,
            capability_revision_id=capability.revision_id,
            capability_sha256=capability.content_sha256,
            promotion_decision_id=capability.promotion_decision_id or "",
            agent_program_id=program.program_id,
            parent_program_revision_id=program.parent_revision_id or "",
            candidate_program_revision_id=program.revision_id,
            candidate_program_root=str(program.root),
            candidate_program_bundle_sha256=program.bundle_sha256,
            skill_claim_ids=tuple(
                sorted(claim.claim_id for claim in validation.claims)
            ),
            skill_claim_sha256=tuple(
                sorted(
                    (claim.claim_id, claim.content_sha256)
                    for claim in validation.claims
                )
            ),
            skill_receipt_store_root=str(validation.receipt_store_root),
            skill_evidence_graph_root=str(validation.evidence_graph_root),
            tournament_campaign_id=tournament.result.campaign_id,
            tournament_id=str(tournament.result.plans[0].metadata["tournament_id"]),
            tournament_decision_sha256=tournament_decision_sha256,
            tournament_action=tournament_action,
            search_parent_revision_id=tournament_winner,
            tournament_claim_ids=tuple(
                sorted(claim.claim_id for claim in tournament_claims)
            ),
            tournament_claim_sha256=tuple(
                sorted(
                    (claim.claim_id, claim.content_sha256)
                    for claim in tournament_claims
                )
            ),
            tournament_receipt_store_root=str(tournament.receipt_store_root),
            tournament_evidence_graph_root=str(tournament.evidence_graph_root),
        )

    def _replay(
        self,
        gap: CapabilityGap,
        decision: PortfolioDecision,
        parent: AgentProgramRevision,
    ) -> PortfolioResult:
        if (
            decision.gap_sha256 != gap.content_sha256
            or decision.parent_program_revision_id != parent.revision_id
        ):
            raise PortfolioViolation("portfolio replay input identity drift")
        capability = self._capability_registry.get(
            decision.capability_id, decision.capability_revision_id
        )
        if (
            capability is None
            or capability.content_sha256 != decision.capability_sha256
            or capability.active
            or capability.promotion_decision_id != decision.promotion_decision_id
        ):
            raise PortfolioViolation("portfolio replay capability drift")
        try:
            approved = self._promotion_log.verified_approved(
                decision.promotion_decision_id
            )
        except ContractViolation as error:
            raise PortfolioViolation("portfolio replay Governance drift") from error
        if approved.candidate_id != decision.skill_candidate_id:
            raise PortfolioViolation("portfolio replay Candidate drift")
        candidate = self._candidate_registry.get(
            decision.skill_candidate_id, decision.skill_candidate_revision_id
        )
        if (
            candidate is None
            or candidate.artifact_sha256
            != decision.skill_candidate_bundle_sha256
            or compiled_bundle_sha256(decision.skill_candidate_bundle_root)
            != decision.skill_candidate_bundle_sha256
        ):
            raise PortfolioViolation("portfolio replay Candidate registry drift")
        try:
            program = AgentProgramRevision.load(decision.candidate_program_root)
        except AgentProgramViolation as error:
            raise PortfolioViolation("portfolio replay AgentProgram drift") from error
        record = self._agent_program_registry.get(
            decision.agent_program_id, decision.candidate_program_revision_id
        )
        if (
            program.bundle_sha256 != decision.candidate_program_bundle_sha256
            or program.parent_revision_id != parent.revision_id
            or capability.revision_id not in program.capability_revision_ids
            or record is None
            or record.artifact_sha256 != program.bundle_sha256
        ):
            raise PortfolioViolation("portfolio replay AgentProgram drift")
        skill_graph = EvidenceGraph(decision.skill_evidence_graph_root)
        skill_claims = tuple(
            claim
            for claim in skill_graph.latest_claims()
            if claim.claim_id in decision.skill_claim_ids
        )
        self._verify_replayed_claim_hashes(
            skill_claims,
            decision.skill_claim_sha256,
            receipt_store_root=Path(decision.skill_receipt_store_root),
            evidence_graph_root=Path(decision.skill_evidence_graph_root),
        )
        tournament_graph = EvidenceGraph(decision.tournament_evidence_graph_root)
        tournament_claims = tuple(
            claim
            for claim in tournament_graph.latest_claims()
            if claim.claim_id in decision.tournament_claim_ids
        )
        self._verify_replayed_claim_hashes(
            tournament_claims,
            decision.tournament_claim_sha256,
            receipt_store_root=Path(decision.tournament_receipt_store_root),
            evidence_graph_root=Path(decision.tournament_evidence_graph_root),
        )
        self._verify_replayed_tournament_decision(
            decision,
            parent,
            program,
            tournament_claims,
            receipt_store_root=Path(decision.tournament_receipt_store_root),
            evidence_graph_root=Path(decision.tournament_evidence_graph_root),
        )
        return PortfolioResult(gap, decision, capability, program, replayed=True)

    @staticmethod
    def _verify_replayed_tournament_decision(
        decision: PortfolioDecision,
        parent: AgentProgramRevision,
        candidate: AgentProgramRevision,
        claims: tuple[Claim, ...],
        *,
        receipt_store_root: Path,
        evidence_graph_root: Path,
    ) -> None:
        if {claim.candidate_id for claim in claims} != {
            parent.revision_id,
            candidate.revision_id,
        }:
            raise PortfolioViolation("portfolio replay tournament participants drift")
        receipts = {
            receipt.receipt_id: receipt
            for receipt in ReceiptStore(receipt_store_root).list_receipts()
        }
        evidence = {
            envelope.evidence_id: envelope
            for envelope in EvidenceGraph(evidence_graph_root).list_evidence()
        }
        expected_bundles = {
            parent.revision_id: parent.bundle_sha256,
            candidate.revision_id: candidate.bundle_sha256,
        }
        for claim in claims:
            native = [
                receipt
                for evidence_id in claim.evidence_ids
                if (envelope := evidence.get(evidence_id)) is not None
                for receipt_id in envelope.receipt_ids
                if (receipt := receipts.get(receipt_id)) is not None
                and receipt.kind == "native_evaluation"
            ]
            if len(native) != 1:
                raise PortfolioViolation("portfolio replay native Claim drift")
            model_id = native[0].payload.get("model_receipt_id")
            model = receipts.get(model_id) if isinstance(model_id, str) else None
            if (
                model is None
                or model.kind != "model"
                or native[0].payload.get("model_artifact_sha256")
                != model.artifact_sha256
                or model.payload.get("revision_id") != claim.candidate_id
                or model.payload.get("program_bundle_sha256")
                != expected_bundles[claim.candidate_id]
            ):
                raise PortfolioViolation("portfolio replay native/model drift")
            evaluator_error = native[0].payload.get("evaluator_error")
            resolved = native[0].payload.get("resolved")
            expected_classification = (
                ClaimClassification.INFRA_FAILURE
                if evaluator_error not in (None, "")
                else ClaimClassification.GAIN
                if resolved is True
                else ClaimClassification.NEUTRAL
                if resolved is False
                else None
            )
            if (
                expected_classification is None
                or claim.grade is not ClaimGrade.E1
                or claim.classification is not expected_classification
            ):
                raise PortfolioViolation(
                    "portfolio replay Claim classification/native drift"
                )
        ordered_claims = tuple(sorted(claims, key=lambda claim: claim.claim_id))
        weights = {
            ClaimClassification.GAIN: 2,
            ClaimClassification.NEUTRAL: 0,
            ClaimClassification.REGRESSION: -2,
            ClaimClassification.INFRA_FAILURE: -1,
        }
        scores = {
            revision_id: next(
                weights[claim.classification]
                for claim in ordered_claims
                if claim.candidate_id == revision_id
            )
            for revision_id in (parent.revision_id, candidate.revision_id)
        }
        highest = max(scores.values())
        tied = {
            revision_id for revision_id, score in scores.items() if score == highest
        }
        winner = parent.revision_id if parent.revision_id in tied else min(tied)
        action = (
            "advance-search-parent"
            if winner == candidate.revision_id
            else "reject-candidates"
        )
        decision_sha256 = content_sha256(
            {
                "tournament_id": decision.tournament_id,
                "execution_scope": "live",
                "parent_revision_id": parent.revision_id,
                "participant_revision_ids": [
                    parent.revision_id,
                    candidate.revision_id,
                ],
                "program_bundle_sha256": [
                    [parent.revision_id, parent.bundle_sha256],
                    [candidate.revision_id, candidate.bundle_sha256],
                ],
                "claim_ids": [claim.claim_id for claim in ordered_claims],
                "claim_sha256": [
                    [claim.claim_id, claim.content_sha256]
                    for claim in ordered_claims
                ],
                "scores": [
                    [revision_id, scores[revision_id]]
                    for revision_id in (parent.revision_id, candidate.revision_id)
                ],
                "winner_revision_id": winner,
            }
        )
        if (
            decision.search_parent_revision_id != winner
            or decision.tournament_action != action
            or decision.tournament_decision_sha256 != decision_sha256
        ):
            raise PortfolioViolation("portfolio replay tournament decision drift")

    def _verify_replayed_claim_hashes(
        self,
        claims: tuple[Claim, ...],
        expected: tuple[tuple[str, str], ...],
        *,
        receipt_store_root: Path,
        evidence_graph_root: Path,
    ) -> None:
        self._verify_claims(
            claims,
            receipt_store_root=receipt_store_root,
            evidence_graph_root=evidence_graph_root,
        )
        if tuple(sorted((row.claim_id, row.content_sha256) for row in claims)) != expected:
            raise PortfolioViolation("portfolio replay Claim drift")


class _PortfolioLog:
    """Small append-only authority for gap and portfolio-decision projections."""

    def __init__(self, path: Path, record_type: type, identity_field: str) -> None:
        self.path = path
        self.record_type = record_type
        self.identity_field = identity_field
        self.lock_path = path.with_suffix(path.suffix + ".writer.lock")

    def get(self, identity: str):
        return {getattr(row, self.identity_field): row for row in self.all()}.get(
            identity
        )

    def append(self, value) -> bool:
        identity = getattr(value, self.identity_field)
        prior = self.get(identity)
        if prior is not None:
            if prior == value:
                return False
            raise PortfolioViolation("conflicting immutable portfolio record")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError as error:
            raise PortfolioViolation("portfolio writer lease is held") from error
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode())
            os.fsync(descriptor)
            prior = self.get(identity)
            if prior is not None:
                if prior == value:
                    return False
                raise PortfolioViolation("conflicting immutable portfolio record")
            record = {
                "content_sha256": value.content_sha256,
                "value": dataclasses.asdict(value),
            }
            encoded = (canonical_json(record) + "\n").encode()
            stream = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            try:
                if os.write(stream, encoded) != len(encoded):
                    raise PortfolioViolation("partial portfolio append")
                os.fsync(stream)
            finally:
                os.close(stream)
        finally:
            os.close(descriptor)
            self.lock_path.unlink(missing_ok=True)
        return True

    def all(self) -> tuple:
        if not self.path.exists():
            return ()
        rows = []
        for line_number, line in enumerate(self.path.read_text().splitlines(), 1):
            try:
                record = json.loads(line)
                payload = record["value"]
                for field in dataclasses.fields(self.record_type):
                    if field.name in payload and field.name in {
                        "evidence_ids",
                        "receipt_ids",
                        "skill_claim_ids",
                        "tournament_claim_ids",
                    }:
                        payload[field.name] = tuple(payload[field.name])
                    elif field.name in payload and field.name in {
                        "skill_claim_sha256",
                        "tournament_claim_sha256",
                    }:
                        payload[field.name] = tuple(
                            tuple(item) for item in payload[field.name]
                        )
                value = self.record_type(**payload)
                if value.content_sha256 != record["content_sha256"]:
                    raise PortfolioViolation("portfolio record hash mismatch")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise PortfolioViolation(
                    f"invalid portfolio record at line {line_number}"
                ) from error
            rows.append(value)
        identities = [getattr(row, self.identity_field) for row in rows]
        if len(identities) != len(set(identities)):
            raise PortfolioViolation("duplicate portfolio record identity")
        return tuple(rows)


__all__ = [
    "CapabilityGap",
    "CompiledSkillCandidate",
    "PortfolioDecision",
    "PortfolioOrchestrator",
    "PortfolioRequest",
    "PortfolioResult",
    "PortfolioViolation",
    "SkillValidationAuthority",
    "SkillAuthority",
    "TeacherCompiler",
    "TournamentAuthority",
    "TournamentRunner",
    "compiled_bundle_sha256",
]
