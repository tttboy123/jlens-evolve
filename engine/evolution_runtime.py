"""Orchestrate Observer -> patterns -> mutations -> tournaments across generations."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

from candidate_tournament import (
    ArmEvaluation,
    CandidateTournament,
    SearchParentSelector,
    result_sha256,
)
from codex_mutation_caller import CodexMutationCallError
from evolution_archive import CandidateArchive
from evolution_controller import (
    EvolutionAuthorization,
    EvolutionController,
    EvolutionPlan,
)
from evolution_report import render_final_report, render_generation_report
from mutation_proposer import (
    InactiveChangeSet,
    MutationContractError,
    MutationProposer,
    MutationRequest,
    ProposalResult,
)
from pattern_miner import (
    _SURFACE_ORDER,
    FrozenObservationEvidence,
    PatternAdvantageMiner,
    PatternCard,
)
from real_mutation_proposer import RealProposalError


@dataclass(frozen=True)
class ExecutionRequest:
    generation: int
    stage: str
    task_uid: str
    role: str
    arm_sha256: str
    original_sha256: str
    parent_sha256: str
    candidate_ordinal: int | None


@dataclass(frozen=True)
class ExecutionArtifact:
    arm: ArmEvaluation
    observation: FrozenObservationEvidence | None


class EvolutionAdapters(Protocol):
    real_codex_calls: bool

    def execute(self, request: ExecutionRequest) -> ExecutionArtifact: ...

    def propose(self, request: MutationRequest, generation: int) -> ProposalResult: ...

    def rollback(self, changeset: InactiveChangeSet) -> dict[str, Any]: ...


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(_canonical_json(value) + "\n" for value in values),
        encoding="utf-8",
    )


def _load_proposal_failures(generation_dir: Path) -> list[dict[str, Any]]:
    path = generation_dir / "PROPOSAL-FAILURES.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


_CONVERGENCE_EPSILON = 0.05
_CONVERGENCE_K = 2


def _paired_delta(
    evaluations: list[ArmEvaluation],
    *,
    candidate_sha256: str,
    baseline_sha256: str,
) -> dict[str, Any] | None:
    """Paired native-score/cost/safety delta of one candidate vs one baseline."""
    candidate = {
        row.task_uid: row
        for row in evaluations
        if row.agent_program_sha256 == candidate_sha256
    }
    baseline = {
        row.task_uid: row
        for row in evaluations
        if row.agent_program_sha256 == baseline_sha256
    }
    pairs = sorted(set(candidate) & set(baseline))
    if not pairs:
        return None
    score_deltas = [
        candidate[task].native_score - baseline[task].native_score for task in pairs
    ]
    cost_deltas = [
        candidate[task].cost_units - baseline[task].cost_units for task in pairs
    ]
    safety_regression = any(
        not candidate[task].safety_passed and baseline[task].safety_passed
        for task in pairs
    )
    return {
        "paired_tasks": len(pairs),
        "native_score_delta_mean": round(fmean(score_deltas), 6),
        "cost_delta_mean": round(fmean(cost_deltas), 6),
        "safety_regression": safety_regression,
    }


def _convergence_metrics(
    evaluations: list[ArmEvaluation],
    *,
    original_sha256: str,
    parent_sha256: str,
    candidate_sha256s: tuple[str, ...],
) -> dict[str, Any]:
    """Candidate-vs-original / candidate-vs-parent deltas for one generation."""
    per_candidate: dict[str, Any] = {}
    deltas: list[float] = []
    safety_regression = False
    for candidate in candidate_sha256s:
        row: dict[str, Any] = {"vs_original": None, "vs_parent": None}
        for key, baseline in (
            ("vs_original", original_sha256),
            ("vs_parent", parent_sha256),
        ):
            metric = _paired_delta(
                evaluations, candidate_sha256=candidate, baseline_sha256=baseline
            )
            row[key] = metric
            if metric is not None:
                deltas.append(abs(metric["native_score_delta_mean"]))
                safety_regression = safety_regression or metric["safety_regression"]
        per_candidate[candidate] = row
    return {
        "per_candidate": per_candidate,
        "mean_abs_delta": round(fmean(deltas), 6) if deltas else None,
        "safety_regression": safety_regression,
        "epsilon": _CONVERGENCE_EPSILON,
        "k_consecutive": _CONVERGENCE_K,
    }


def _convergence_stop(
    previous_summaries: list[dict[str, Any]],
    current_metrics: dict[str, Any],
) -> bool:
    """K=2 consecutive |mean delta| < epsilon with no safety regression -> stop."""
    tail = [
        summary.get("convergence_metrics")
        for summary in previous_summaries[-(_CONVERGENCE_K - 1) :]
        if summary.get("convergence_metrics")
    ]
    tail.append(current_metrics)
    tail = [item for item in tail if item]
    if len(tail) < _CONVERGENCE_K:
        return False
    return all(
        item.get("mean_abs_delta") is not None
        and item["mean_abs_delta"] < _CONVERGENCE_EPSILON
        and not item.get("safety_regression")
        for item in tail
    )


def _generalize_patterns(
    *,
    champion_sha256: str,
    champion_hypothesis_ids: tuple[str, ...],
    previous_cards: tuple[PatternCard, ...],
    current_cards: tuple[PatternCard, ...],
    observations: list[FrozenObservationEvidence],
) -> tuple[PatternCard, ...]:
    """A fixes B -> generalized PatternCard (observational_not_causal only)."""
    if not champion_hypothesis_ids:
        return ()
    prior_by_id = {card.pattern_id: card for card in previous_cards}
    current_failures = [
        card for card in current_cards if card.pattern_kind == "failure"
    ]
    if not current_failures:
        return ()
    champion_advantage = tuple(
        row
        for row in observations
        if row.agent_program_sha256 == champion_sha256 and row.outcome == "advantage"
    )
    if not champion_advantage:
        return ()
    generalized: list[PatternCard] = []
    for source_id in champion_hypothesis_ids:
        source = prior_by_id.get(source_id)
        if source is None:
            continue
        for target in current_failures:
            fixed = any(
                target.observed_feature in row.observed_features
                for row in champion_advantage
            )
            if not fixed:
                continue
            merged_evidence = tuple(
                dict.fromkeys(source.evidence_ids + target.evidence_ids)
            )
            counterexamples = tuple(
                dict.fromkeys(
                    source.counterexample_evidence_ids
                    + target.counterexample_evidence_ids
                )
            )
            conditions = tuple(sorted(set(source.conditions) | set(target.conditions)))
            surfaces = tuple(
                sorted(
                    set(source.expected_surfaces) | set(target.expected_surfaces),
                    key=_SURFACE_ORDER.index,
                )
            )
            observed_feature = f"{source.observed_feature}|{target.observed_feature}"
            seed = {
                "generalized": True,
                "source": source.pattern_id,
                "target": target.pattern_id,
                "feature": observed_feature,
                "evidence": list(merged_evidence),
                "conditions": conditions,
                "surfaces": surfaces,
            }
            pattern_id = (
                "pattern-generalized-"
                + hashlib.sha256(_canonical_json(seed).encode()).hexdigest()[:16]
            )
            generalized.append(
                PatternCard(
                    schema_version=1,
                    pattern_id=pattern_id,
                    pattern_kind="generalized_fix",
                    observed_feature=observed_feature,
                    evidence_ids=merged_evidence,
                    evidence_sha256s=tuple(
                        sorted(
                            {
                                digest
                                for card in (source, target)
                                for digest in card.evidence_sha256s
                            }
                        )
                    ),
                    counterexample_evidence_ids=counterexamples,
                    support_count=len(merged_evidence),
                    counterexample_count=len(counterexamples),
                    conditions=conditions,
                    expected_surfaces=surfaces,
                    confidence=min(source.confidence, target.confidence),
                    causal_boundary="observational_not_causal",
                    admission_gate_allowed=False,
                )
            )
    return tuple(sorted(generalized, key=lambda card: card.pattern_id))


def _write_changeset_artifacts(
    generation_dir: Path, changeset: InactiveChangeSet
) -> None:
    root = generation_dir / "candidates" / changeset.changeset_id
    _write_json(root / "CHANGESET.json", changeset.to_dict())
    (root / "forward.patch").write_text(
        _canonical_json(list(changeset.operations)) + "\n", encoding="utf-8"
    )
    (root / "rollback.patch").write_text(
        _canonical_json(list(changeset.rollback_operations)) + "\n", encoding="utf-8"
    )


def _validate_execution(request: ExecutionRequest, artifact: ExecutionArtifact) -> None:
    arm = artifact.arm
    if (
        arm.task_uid != request.task_uid
        or arm.agent_program_sha256 != request.arm_sha256
    ):
        raise ValueError("execution artifact does not match frozen task/arm request")
    if arm.role != request.role:
        raise ValueError("execution artifact role does not match request")


def _propose_generation(
    *,
    generation: int,
    cards: tuple[PatternCard, ...],
    parent_sha256: str,
    native_evaluator_epoch: str,
    generation_dir: Path,
    archive: CandidateArchive,
    controller: EvolutionController,
    adapters: EvolutionAdapters,
) -> tuple[InactiveChangeSet, ...]:
    requests = MutationProposer().build_requests(
        cards,
        parent_agent_program_sha256=parent_sha256,
        native_evaluator_epoch=native_evaluator_epoch,
        maximum_candidates=4,
    )
    proposed = []
    failures: list[dict[str, Any]] = []
    for request in requests:
        try:
            result = adapters.propose(request, generation)
        except (
            MutationContractError,
            RealProposalError,
            CodexMutationCallError,
        ) as exc:
            failures.append(
                {
                    "request_id": request.request_id,
                    "surface": request.surface,
                    "error_type": type(exc).__name__,
                    "reason": str(exc)[:500],
                }
            )
            continue
        changeset = result.changeset
        already_registered = (
            changeset.candidate_agent_program_sha256 in archive.candidates()
        )
        if (
            not getattr(adapters, "accounts_auxiliary_calls", False)
            and not already_registered
        ):
            controller.record_auxiliary_calls(
                len(result.raw_responses),
                real_codex_calls=(
                    len(result.raw_responses) if adapters.real_codex_calls else 0
                ),
            )
        if changeset.parent_agent_program_sha256 != parent_sha256:
            raise ValueError(
                "proposed ChangeSet is not a child of current search parent"
            )
        archive.register_candidate(changeset)
        _write_changeset_artifacts(generation_dir, changeset)
        proposed.append(changeset)
    if failures:
        _write_jsonl(
            generation_dir / "PROPOSAL-FAILURES.jsonl",
            failures,
        )
    return tuple(proposed)


def _patterns_summary(cards: tuple[PatternCard, ...]) -> dict[str, int]:
    return {
        "total": len(cards),
        "advantage": sum(card.pattern_kind == "advantage" for card in cards),
        "failure": sum(card.pattern_kind == "failure" for card in cards),
        "generalized": sum(card.pattern_kind == "generalized_fix" for card in cards),
    }


def _mutation_request_count(
    cards: tuple[PatternCard, ...],
    *,
    parent_sha256: str,
    native_evaluator_epoch: str,
) -> int:
    return len(
        MutationProposer().build_requests(
            cards,
            parent_agent_program_sha256=parent_sha256,
            native_evaluator_epoch=native_evaluator_epoch,
            maximum_candidates=4,
        )
    )


def _finalize_no_signal(
    *,
    output_dir: Path,
    generation_summaries: list[dict[str, Any]],
    all_cards: list[PatternCard],
    archive: CandidateArchive,
    controller: EvolutionController,
    generation: int,
    available_mutation_requests: int,
) -> dict[str, Any]:
    controller_status = controller.inspect()
    states = {
        candidate: archive.candidate_state(candidate)
        for candidate in archive.candidates()
    }
    reason = {
        "code": "insufficient_mutation_signal",
        "generation": generation,
        "required_distinct_mutation_requests": 4,
        "available_distinct_mutation_requests": available_mutation_requests,
        "action": "stop_without_fabricating_patterns_or_lowering_gates",
    }
    result = {
        "schema_version": 1,
        "stage": "v2.1.1-jlens-evolution",
        "status": "stopped_no_mutation_signal",
        "terminal_state": "exhausted",
        "completed_generations": len(generation_summaries),
        "unique_search_tasks_retired": controller_status["task_counts"]["retired"],
        "generations": generation_summaries,
        "patterns": _patterns_summary(tuple(all_cards)),
        "candidates": {
            "proposed": len(states),
            "selected": sum(state == "selected" for state in states.values()),
            "rejected": sum(state == "rejected" for state in states.values()),
            "failed": sum(state == "failed" for state in states.values()),
            "inactive": sum(state == "inactive" for state in states.values()),
        },
        "search_parent_history": controller_status["parent_history"],
        "usage": controller_status["usage"],
        "task_counts": controller_status["task_counts"],
        "archive_valid": archive.verify()["valid"],
        "controller_valid": controller.verify()["valid"],
        "stop_reason": reason,
        "final_sealed_opened": controller_status["final_sealed_opened"],
        "production_active_ref": controller_status["production_active_ref"],
        "claims": {
            "evolution_engine": True,
            "jlens_observer_only": True,
            "experimental_search_parent_advance": bool(
                controller_status["parent_history"]
            ),
            "agent_optimized": False,
            "production_promoted": False,
            "final_sealed_generalization": False,
            "agentic_rsi": False,
        },
    }
    result["experiment_fingerprint"] = _sha(
        {
            "generations": result["generations"],
            "patterns": result["patterns"],
            "candidates": result["candidates"],
            "search_parent_history": result["search_parent_history"],
            "usage": result["usage"],
            "task_counts": result["task_counts"],
            "stop_reason": reason,
        }
    )
    _write_json(output_dir / "STOP.json", reason)
    _write_json(output_dir / "RESULT.json", result)
    (output_dir / "REPORT.zh-CN.md").write_text(
        render_final_report(result), encoding="utf-8"
    )
    return result


def _finalize_converged(
    *,
    output_dir: Path,
    generation_summaries: list[dict[str, Any]],
    all_cards: list[PatternCard],
    archive: CandidateArchive,
    controller: EvolutionController,
    generation: int,
    convergence_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Terminate the search at a declared convergence boundary (K=2, no safety regression)."""
    controller_status = controller.inspect()
    states = {
        candidate: archive.candidate_state(candidate)
        for candidate in archive.candidates()
    }
    reason = {
        "code": "converged",
        "generation": generation,
        "epsilon": _CONVERGENCE_EPSILON,
        "k_consecutive": _CONVERGENCE_K,
        "mean_abs_delta": convergence_metrics.get("mean_abs_delta"),
        "safety_regression": convergence_metrics.get("safety_regression"),
        "action": "stop_search_converged",
    }
    result = {
        "schema_version": 1,
        "stage": "v2.1.1-jlens-evolution",
        "status": "converged",
        "terminal_state": "converged",
        "completed_generations": len(generation_summaries),
        "unique_search_tasks_retired": controller_status["task_counts"]["retired"],
        "generations": generation_summaries,
        "patterns": _patterns_summary(tuple(all_cards)),
        "candidates": {
            "proposed": len(states),
            "selected": sum(state == "selected" for state in states.values()),
            "rejected": sum(state == "rejected" for state in states.values()),
            "failed": sum(state == "failed" for state in states.values()),
            "inactive": sum(state == "inactive" for state in states.values()),
        },
        "search_parent_history": controller_status["parent_history"],
        "usage": controller_status["usage"],
        "task_counts": controller_status["task_counts"],
        "archive_valid": archive.verify()["valid"],
        "controller_valid": controller.verify()["valid"],
        "stop_reason": reason,
        "convergence": convergence_metrics,
        "final_sealed_opened": controller_status["final_sealed_opened"],
        "production_active_ref": controller_status["production_active_ref"],
        "claims": {
            "evolution_engine": True,
            "jlens_observer_only": True,
            "experimental_search_parent_advance": bool(
                controller_status["parent_history"]
            ),
            "agent_optimized": False,
            "production_promoted": False,
            "final_sealed_generalization": False,
            "agentic_rsi": False,
        },
    }
    result["experiment_fingerprint"] = _sha(
        {
            "generations": result["generations"],
            "patterns": result["patterns"],
            "candidates": result["candidates"],
            "search_parent_history": result["search_parent_history"],
            "usage": result["usage"],
            "task_counts": result["task_counts"],
            "stop_reason": reason,
            "convergence": convergence_metrics,
        }
    )
    _write_json(output_dir / "STOP.json", reason)
    _write_json(output_dir / "RESULT.json", result)
    (output_dir / "REPORT.zh-CN.md").write_text(
        render_final_report(result), encoding="utf-8"
    )
    return result


def _ensure_candidate_transition(
    archive: CandidateArchive,
    candidate_sha256: str,
    target_state: str,
    *,
    reason: str,
    evidence_sha256: str,
) -> None:
    current = archive.candidate_state(candidate_sha256)
    if current == target_state:
        return
    if target_state == "evaluating" and current in {"selected", "rejected", "failed"}:
        return
    archive.transition(
        candidate_sha256,
        target_state,
        reason=reason,
        evidence_sha256=evidence_sha256,
    )


def _ensure_rollback(
    archive: CandidateArchive,
    candidate_sha256: str,
    rollback: dict[str, Any],
) -> None:
    matching = [
        event
        for event in archive.events()
        if event.get("event_type") == "rollback_recorded"
        and event.get("candidate_agent_program_sha256") == candidate_sha256
    ]
    if matching:
        expected = {
            "forward_patch_sha256": rollback["forward_patch_sha256"],
            "rollback_patch_sha256": rollback["rollback_patch_sha256"],
            "rollback_verified": rollback["verified"],
        }
        if any(
            event.get(key) != value
            for key, value in expected.items()
            for event in matching
        ):
            raise ValueError("persisted rollback evidence is immutable")
        return
    archive.record_rollback(
        candidate_sha256,
        forward_patch_sha256=rollback["forward_patch_sha256"],
        rollback_patch_sha256=rollback["rollback_patch_sha256"],
        verified=rollback["verified"],
    )


def _ensure_parent_advance(
    archive: CandidateArchive,
    candidate_sha256: str,
    *,
    previous_parent_sha256: str,
    decision_sha256: str,
) -> None:
    history = archive.authority()["search_parent_history"]
    existing = [
        row
        for row in history
        if row.get("search_parent_sha256") == candidate_sha256
        and row.get("previous_parent_sha256") == previous_parent_sha256
    ]
    if existing:
        if existing[0].get("decision_sha256") != decision_sha256:
            raise ValueError("persisted search-parent decision is immutable")
        return
    if archive.search_parent_sha256 != previous_parent_sha256:
        raise ValueError("search-parent lineage cannot be resumed from a stale parent")
    archive.advance_search_parent(candidate_sha256, decision_sha256=decision_sha256)


def _run_claimed_stage(
    *,
    generation: int,
    stage: str,
    candidate_sha256s: tuple[str, ...],
    controller: EvolutionController,
    adapters: EvolutionAdapters,
    worker_count: int = 1,
) -> tuple[list[ArmEvaluation], list[FrozenObservationEvidence]]:
    """Run a claimed stage; tasks may execute concurrently (contract identical).

    Each arm's evidence is frozen independently the moment its execution and
    validation complete; controller mutations are serialized by an internal
    lock, so reservations and the evidence ledger stay consistent even when
    worker_count > 1.
    """
    claim = controller.claim_stage(
        generation, stage, candidate_sha256s=candidate_sha256s
    )
    original = claim.arm_sha256s[0]
    parent = claim.arm_sha256s[1]

    def _run_task(
        task_uid: str,
    ) -> tuple[str, list[tuple[ExecutionRequest, ExecutionArtifact]]]:
        results: list[tuple[ExecutionRequest, ExecutionArtifact]] = []
        for arm_index, arm in enumerate(claim.arm_sha256s):
            role = (
                "original"
                if arm_index == 0
                else "parent"
                if arm_index == 1
                else "candidate"
            )
            request = ExecutionRequest(
                generation=generation,
                stage=stage,
                task_uid=task_uid,
                role=role,
                arm_sha256=arm,
                original_sha256=original,
                parent_sha256=parent,
                candidate_ordinal=(arm_index - 2 if role == "candidate" else None),
            )
            artifact = adapters.execute(request)
            _validate_execution(request, artifact)
            existing = controller.arm_evidence_status(
                generation=generation,
                stage=stage,
                task_uid=task_uid,
                arm_sha256=arm,
            )
            if existing is None:
                controller.record_arm_evidence(
                    generation=generation,
                    stage=stage,
                    task_uid=task_uid,
                    arm_sha256=arm,
                    evidence_sha256=artifact.arm.evidence_sha256,
                    real_codex_calls=1 if adapters.real_codex_calls else 0,
                )
            elif existing["evidence_sha256"] != artifact.arm.evidence_sha256:
                raise ValueError("resumed arm evidence differs from frozen controller")
            results.append((request, artifact))
        return task_uid, results

    if worker_count <= 1:
        ordered = [_run_task(task_uid) for task_uid in claim.task_uids]
    else:
        workers = max(1, min(worker_count, len(claim.task_uids)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            ordered = list(pool.map(_run_task, claim.task_uids))

    evaluations = []
    observations = []
    for _task_uid, results in ordered:
        for _request, artifact in results:
            evaluations.append(artifact.arm)
            if artifact.observation is not None:
                observations.append(artifact.observation)
    controller.complete_stage(generation, stage)
    return evaluations, observations


def run_evolution_search(
    *,
    output_dir: Path,
    plan: EvolutionPlan,
    authorization: EvolutionAuthorization,
    original_agent_program_sha256: str,
    seed_parent_agent_program_sha256: str,
    native_evaluator_epoch: str,
    adapters: EvolutionAdapters,
    controller: EvolutionController | None = None,
    worker_count: int = 1,
) -> dict[str, Any]:
    """Run the frozen four-generation protocol with injected external adapters.

    worker_count controls candidate/task execution concurrency inside each
    claimed stage (1 = serial; 2..4 = parallel tasks, identical contract,
    evidence frozen per arm).
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if controller is None:
        controller = EvolutionController.initialize(
            output_dir / "controller",
            plan=plan,
            authorization=authorization,
            original_agent_program_sha256=original_agent_program_sha256,
            seed_parent_agent_program_sha256=seed_parent_agent_program_sha256,
            native_evaluator_epoch=native_evaluator_epoch,
        )
    elif controller.root != (output_dir / "controller").resolve():
        raise ValueError("injected controller root differs from evolution output")
    archive = CandidateArchive.create(
        output_dir / "archive",
        original_agent_program_sha256=original_agent_program_sha256,
        seed_parent_agent_program_sha256=seed_parent_agent_program_sha256,
    )
    generation_summaries = []
    all_cards: list[PatternCard] = []

    g0_dir = output_dir / "generation-0"
    g0_evaluations, g0_observations = _run_claimed_stage(
        generation=0,
        stage="observe",
        candidate_sha256s=(),
        controller=controller,
        adapters=adapters,
        worker_count=worker_count,
    )
    _write_jsonl(
        g0_dir / "EXECUTION-LEDGER.jsonl", [row.to_dict() for row in g0_evaluations]
    )
    _write_jsonl(
        g0_dir / "OBSERVER-EVIDENCE.jsonl",
        [
            {
                "evidence_id": row.evidence_id,
                "task_uid": row.task_uid,
                "agent_program_sha256": row.agent_program_sha256,
                "parent_agent_program_sha256": row.parent_agent_program_sha256,
                "native_score_delta": row.native_score_delta,
                "outcome": row.outcome,
                "causal_boundary": row.causal_boundary,
                "admission_gate_allowed": row.admission_gate_allowed,
            }
            for row in g0_observations
        ],
    )
    cards = PatternAdvantageMiner().mine(g0_observations)
    all_cards.extend(cards)
    _write_jsonl(g0_dir / "PATTERN-CARDS.jsonl", [card.to_dict() for card in cards])
    _write_json(
        g0_dir / "HYPOTHESES.json", {"cards": [card.pattern_id for card in cards]}
    )
    g0_request_count = _mutation_request_count(
        cards,
        parent_sha256=seed_parent_agent_program_sha256,
        native_evaluator_epoch=native_evaluator_epoch,
    )
    if g0_request_count == 0:
        generation_summary = {
            "generation": 0,
            "tasks_retired": 25,
            "patterns": _patterns_summary(cards),
            "proposed_candidates": [],
            "rejected_candidates": [],
            "proposal_failures": [],
            "selected_candidate": None,
            "parent_decision": None,
            "tournaments": {},
        }
        (g0_dir / "RESULT.zh-CN.md").write_text(
            render_generation_report(generation_summary), encoding="utf-8"
        )
        generation_summaries.append(generation_summary)
        return _finalize_no_signal(
            output_dir=output_dir,
            generation_summaries=generation_summaries,
            all_cards=all_cards,
            archive=archive,
            controller=controller,
            generation=0,
            available_mutation_requests=g0_request_count,
        )
    g0_proposed = _propose_generation(
        generation=0,
        cards=cards,
        parent_sha256=seed_parent_agent_program_sha256,
        native_evaluator_epoch=native_evaluator_epoch,
        generation_dir=g0_dir,
        archive=archive,
        controller=controller,
        adapters=adapters,
    )
    generation_summary = {
        "generation": 0,
        "tasks_retired": 25,
        "patterns": _patterns_summary(cards),
        "proposed_candidates": [
            item.candidate_agent_program_sha256 for item in g0_proposed
        ],
        "rejected_candidates": [],
        "proposal_failures": _load_proposal_failures(g0_dir),
        "selected_candidate": None,
        "parent_decision": None,
        "tournaments": {},
    }
    (g0_dir / "RESULT.zh-CN.md").write_text(
        render_generation_report(generation_summary), encoding="utf-8"
    )
    generation_summaries.append(generation_summary)
    current_candidates = g0_proposed

    for generation in range(1, 4):
        generation_dir = output_dir / f"generation-{generation}"
        candidate_by_hash = {
            item.candidate_agent_program_sha256: item for item in current_candidates
        }
        candidate_hashes = tuple(candidate_by_hash)
        if not candidate_hashes:
            generation_dir.mkdir(parents=True, exist_ok=True)
            generation_summary = {
                "generation": generation,
                "tasks_retired": 0,
                "patterns": _patterns_summary(tuple(all_cards)),
                "proposed_candidates": [],
                "rejected_candidates": [],
                "proposal_failures": _load_proposal_failures(generation_dir),
                "selected_candidate": None,
                "parent_decision": None,
                "tournaments": {},
            }
            (generation_dir / "RESULT.zh-CN.md").write_text(
                render_generation_report(generation_summary), encoding="utf-8"
            )
            generation_summaries.append(generation_summary)
            return _finalize_no_signal(
                output_dir=output_dir,
                generation_summaries=generation_summaries,
                all_cards=all_cards,
                archive=archive,
                controller=controller,
                generation=generation,
                available_mutation_requests=0,
            )
        scout_advance = min(2, len(candidate_hashes))
        for candidate in candidate_hashes:
            _ensure_candidate_transition(
                archive,
                candidate,
                "evaluating",
                reason=f"generation_{generation}_scout",
                evidence_sha256=_sha(
                    {"generation": generation, "candidate": candidate}
                ),
            )
        all_evaluations: list[ArmEvaluation] = []
        all_observations: list[FrozenObservationEvidence] = []
        tournaments = {}

        scout_eval, scout_obs = _run_claimed_stage(
            generation=generation,
            stage="scout",
            candidate_sha256s=candidate_hashes,
            controller=controller,
            adapters=adapters,
            worker_count=worker_count,
        )
        all_evaluations.extend(scout_eval)
        all_observations.extend(scout_obs)
        scout = CandidateTournament().evaluate_stage(
            stage="scout",
            evaluations=scout_eval,
            task_uids=plan.stage(generation, "scout").task_uids,
            original_sha256=controller.inspect()["original_agent_program_sha256"],
            parent_sha256=controller.inspect()["claims"][f"g{generation}:scout"][
                "arm_sha256s"
            ][1],
            candidate_sha256s=candidate_hashes,
            advance_count=scout_advance,
        )
        tournaments["scout"] = scout.to_dict()
        rejected = [
            candidate
            for candidate in candidate_hashes
            if candidate not in scout.finalists
        ]
        for candidate in rejected:
            _ensure_candidate_transition(
                archive,
                candidate,
                "rejected",
                reason="scout_elimination",
                evidence_sha256=result_sha256(scout),
            )

        semi_eval, semi_obs = _run_claimed_stage(
            generation=generation,
            stage="semifinal",
            candidate_sha256s=scout.finalists,
            controller=controller,
            adapters=adapters,
            worker_count=worker_count,
        )
        all_evaluations.extend(semi_eval)
        all_observations.extend(semi_obs)
        semifinal = CandidateTournament().evaluate_stage(
            stage="semifinal",
            evaluations=semi_eval,
            task_uids=plan.stage(generation, "semifinal").task_uids,
            original_sha256=controller.inspect()["original_agent_program_sha256"],
            parent_sha256=controller.inspect()["claims"][f"g{generation}:semifinal"][
                "arm_sha256s"
            ][1],
            candidate_sha256s=scout.finalists,
            advance_count=1,
        )
        tournaments["semifinal"] = semifinal.to_dict()
        semifinal_rejected = [
            candidate
            for candidate in scout.finalists
            if candidate not in semifinal.finalists
        ]
        rejected.extend(semifinal_rejected)
        for candidate in semifinal_rejected:
            _ensure_candidate_transition(
                archive,
                candidate,
                "rejected",
                reason="semifinal_elimination",
                evidence_sha256=result_sha256(semifinal),
            )

        confirm_eval, confirm_obs = _run_claimed_stage(
            generation=generation,
            stage="confirmation",
            candidate_sha256s=semifinal.finalists,
            controller=controller,
            adapters=adapters,
            worker_count=worker_count,
        )
        all_evaluations.extend(confirm_eval)
        all_observations.extend(confirm_obs)
        confirmation = CandidateTournament().evaluate_stage(
            stage="confirmation",
            evaluations=confirm_eval,
            task_uids=plan.stage(generation, "confirmation").task_uids,
            original_sha256=controller.inspect()["original_agent_program_sha256"],
            parent_sha256=controller.inspect()["claims"][f"g{generation}:confirmation"][
                "arm_sha256s"
            ][1],
            candidate_sha256s=semifinal.finalists,
            advance_count=1,
        )
        tournaments["confirmation"] = confirmation.to_dict()
        decision = SearchParentSelector().decide(confirmation)
        champion = decision.champion_sha256
        if decision.advance:
            _ensure_candidate_transition(
                archive,
                champion,
                "selected",
                reason="confirmation_gate_passed",
                evidence_sha256=result_sha256(confirmation),
            )
            rollback = adapters.rollback(candidate_by_hash[champion])
            _ensure_rollback(
                archive,
                champion,
                rollback,
            )
            _ensure_parent_advance(
                archive,
                champion,
                previous_parent_sha256=decision.previous_parent_sha256,
                decision_sha256=result_sha256(confirmation),
            )
        else:
            rejected.append(champion)
            _ensure_candidate_transition(
                archive,
                champion,
                "rejected",
                reason="confirmation_gate_failed:" + ",".join(decision.reasons),
                evidence_sha256=result_sha256(confirmation),
            )
        controller.record_parent_decision(
            generation=generation,
            previous_parent_sha256=decision.previous_parent_sha256,
            search_parent_sha256=decision.search_parent_sha256,
            decision_sha256=result_sha256(confirmation),
            advance=decision.advance,
        )

        _write_jsonl(
            generation_dir / "EXECUTION-LEDGER.jsonl",
            [row.to_dict() for row in all_evaluations],
        )
        _write_jsonl(
            generation_dir / "OBSERVER-EVIDENCE.jsonl",
            [
                {
                    "evidence_id": row.evidence_id,
                    "task_uid": row.task_uid,
                    "agent_program_sha256": row.agent_program_sha256,
                    "parent_agent_program_sha256": row.parent_agent_program_sha256,
                    "native_score_delta": row.native_score_delta,
                    "outcome": row.outcome,
                    "causal_boundary": row.causal_boundary,
                    "admission_gate_allowed": row.admission_gate_allowed,
                }
                for row in all_observations
            ],
        )
        previous_cards = tuple(all_cards)
        cards = PatternAdvantageMiner().mine(all_observations)
        all_cards.extend(cards)
        if decision.advance:
            generalized = _generalize_patterns(
                champion_sha256=champion,
                champion_hypothesis_ids=tuple(
                    candidate_by_hash[champion].hypothesis_ids
                ),
                previous_cards=previous_cards,
                current_cards=cards,
                observations=all_observations,
            )
            if generalized:
                cards = tuple(
                    sorted(cards + generalized, key=lambda card: card.pattern_id)
                )
                all_cards.extend(generalized)
        _write_jsonl(
            generation_dir / "PATTERN-CARDS.jsonl",
            [card.to_dict() for card in cards],
        )
        _write_json(
            generation_dir / "HYPOTHESES.json",
            {"cards": [card.pattern_id for card in cards]},
        )
        _write_json(generation_dir / "TOURNAMENT.json", tournaments)
        _write_json(generation_dir / "PARENT-DECISION.json", decision.to_dict())
        conv_metrics = _convergence_metrics(
            confirm_eval,
            original_sha256=controller.inspect()["original_agent_program_sha256"],
            parent_sha256=decision.previous_parent_sha256,
            candidate_sha256s=tuple(semifinal.finalists),
        )
        next_request_count = _mutation_request_count(
            cards,
            parent_sha256=decision.search_parent_sha256,
            native_evaluator_epoch=native_evaluator_epoch,
        )
        if next_request_count == 0:
            generation_summary = {
                "generation": generation,
                "tasks_retired": 25,
                "patterns": _patterns_summary(cards),
                "convergence_metrics": conv_metrics,
                "proposed_candidates": [],
                "rejected_candidates": rejected,
                "proposal_failures": _load_proposal_failures(generation_dir),
                "selected_candidate": champion if decision.advance else None,
                "parent_decision": decision.to_dict(),
                "tournaments": tournaments,
            }
            (generation_dir / "RESULT.zh-CN.md").write_text(
                render_generation_report(generation_summary), encoding="utf-8"
            )
            generation_summaries.append(generation_summary)
            return _finalize_no_signal(
                output_dir=output_dir,
                generation_summaries=generation_summaries,
                all_cards=all_cards,
                archive=archive,
                controller=controller,
                generation=generation,
                available_mutation_requests=next_request_count,
            )
        if generation >= 2 and _convergence_stop(generation_summaries, conv_metrics):
            generation_summary = {
                "generation": generation,
                "tasks_retired": 25,
                "patterns": _patterns_summary(cards),
                "convergence_metrics": conv_metrics,
                "proposed_candidates": [],
                "rejected_candidates": rejected,
                "proposal_failures": _load_proposal_failures(generation_dir),
                "selected_candidate": champion if decision.advance else None,
                "parent_decision": decision.to_dict(),
                "tournaments": tournaments,
            }
            (generation_dir / "RESULT.zh-CN.md").write_text(
                render_generation_report(generation_summary), encoding="utf-8"
            )
            generation_summaries.append(generation_summary)
            return _finalize_converged(
                output_dir=output_dir,
                generation_summaries=generation_summaries,
                all_cards=all_cards,
                archive=archive,
                controller=controller,
                generation=generation,
                convergence_metrics=conv_metrics,
            )
        current_candidates = _propose_generation(
            generation=generation,
            cards=cards,
            parent_sha256=decision.search_parent_sha256,
            native_evaluator_epoch=native_evaluator_epoch,
            generation_dir=generation_dir,
            archive=archive,
            controller=controller,
            adapters=adapters,
        )
        generation_summary = {
            "generation": generation,
            "tasks_retired": 25,
            "patterns": _patterns_summary(cards),
            "convergence_metrics": conv_metrics,
            "proposed_candidates": [
                item.candidate_agent_program_sha256 for item in current_candidates
            ],
            "rejected_candidates": rejected,
            "proposal_failures": _load_proposal_failures(generation_dir),
            "selected_candidate": champion if decision.advance else None,
            "parent_decision": decision.to_dict(),
            "tournaments": tournaments,
        }
        (generation_dir / "RESULT.zh-CN.md").write_text(
            render_generation_report(generation_summary), encoding="utf-8"
        )
        generation_summaries.append(generation_summary)

    states = {
        candidate: archive.candidate_state(candidate)
        for candidate in archive.candidates()
    }
    controller_status = controller.inspect()
    archive_verification = archive.verify()
    controller_verification = controller.verify()
    pattern_counts = _patterns_summary(tuple(all_cards))
    candidate_counts = {
        "proposed": len(states),
        "selected": sum(state == "selected" for state in states.values()),
        "rejected": sum(state == "rejected" for state in states.values()),
        "failed": sum(state == "failed" for state in states.values()),
        "inactive": sum(state == "inactive" for state in states.values()),
    }
    semantic = {
        "generations": generation_summaries,
        "patterns": pattern_counts,
        "candidates": candidate_counts,
        "search_parent_history": controller_status["parent_history"],
        "usage": controller_status["usage"],
        "task_counts": controller_status["task_counts"],
        "archive_valid": archive_verification["valid"],
        "controller_valid": controller_verification["valid"],
    }
    result = {
        "schema_version": 1,
        "stage": "v2.1.1-jlens-evolution",
        "status": "completed",
        "terminal_state": "completed",
        "completed_generations": 4,
        "unique_search_tasks_retired": controller_status["task_counts"]["retired"],
        **semantic,
        "final_sealed_opened": controller_status["final_sealed_opened"],
        "production_active_ref": controller_status["production_active_ref"],
        "claims": {
            "evolution_engine": True,
            "jlens_observer_only": True,
            "experimental_search_parent_advance": True,
            "agent_optimized": False,
            "production_promoted": False,
            "final_sealed_generalization": False,
            "agentic_rsi": False,
        },
        "experiment_fingerprint": _sha(semantic),
    }
    _write_json(output_dir / "RESULT.json", result)
    (output_dir / "REPORT.zh-CN.md").write_text(
        render_final_report(result), encoding="utf-8"
    )
    return result
