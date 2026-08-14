"""Persistent outer loop for real baseline→Teacher→candidate→native evolution."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from evolve.contracts import Cohort, ContractViolation, canonical_json
from evolve.kernel import BudgetExceeded
from evolve.proposals import (
    CandidateCompiler,
    CandidateProposer,
    CompiledRevision,
    CompileSpec,
    PricingCnyPerMillionTokens,
    ProposalResult,
    Transport,
)

from .config import AutonomousEvolutionConfig, AutonomousEvolutionError
from .goal import GoalRunStatus, GoalState, GoalStateStore
from .output import (
    atomic_json,
    export_best_harness,
    export_empty_harness,
    file_sha256,
    freeze_json,
    model_identity,
    seal_manifest,
)
from .state import HashChainIndex
from .task_selector import FeedbackTaskSelector, TaskSelection
from .verification import CampaignOutcomeVerifier, VerifiedCampaignClaim


@dataclass(frozen=True, slots=True)
class BaselineProbeResult:
    task_ids: tuple[str, ...]
    model_receipt_ids: tuple[str, ...]
    native_receipt_ids: tuple[str, ...]
    native_outcomes: tuple[Mapping[str, Any], ...]
    failure_signatures: tuple[Mapping[str, Any], ...]
    replayed: bool


@dataclass(frozen=True, slots=True)
class PrescreenResult:
    candidate_revision_id: str
    candidate_bundle_sha256: str
    model_receipt_ids: tuple[str, ...]
    structural_valid: bool
    patch_applicable: bool
    replayed: bool
    status: str = "completed"


@dataclass(frozen=True, slots=True)
class RoundExecutionRequest:
    goal_id: str
    round_index: int
    output_root: Path
    selection: TaskSelection
    config: AutonomousEvolutionConfig
    source_commit_sha: str
    worktree_root: Path
    baseline_revision_id: str
    baseline_compiled_root: Path | None
    candidate: CompiledRevision | None = None


@runtime_checkable
class EvolutionRoundExecutor(Protocol):
    """External-effect boundary used by the autonomous product loop."""

    def baseline(self, request: RoundExecutionRequest) -> BaselineProbeResult: ...

    def prescreen(self, request: RoundExecutionRequest) -> PrescreenResult: ...

    def paired(self, request: RoundExecutionRequest) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class EvolutionDependencies:
    teacher_transport: Transport
    teacher_pricing: PricingCnyPerMillionTokens
    round_executor: EvolutionRoundExecutor
    evidence_mode: str = "test-fixture"


def build_default_dependencies(
    config: AutonomousEvolutionConfig,
) -> EvolutionDependencies:
    """Bind the product config to the real Teacher/Qwen/native adapters."""

    from evolve.teachers import build_teacher_transport

    from .fresh_executor import FreshFeedbackRoundExecutor

    pricing = _provider_pricing(config.teacher.provider, config.teacher.model)
    return EvolutionDependencies(
        teacher_transport=build_teacher_transport(
            provider=config.teacher.provider,
            model=config.teacher.model,
            endpoint=config.teacher.endpoint,
            api_key_env=config.teacher.api_key_env,
        ),
        teacher_pricing=pricing,
        round_executor=FreshFeedbackRoundExecutor(),
        evidence_mode=(
            "real-frozen-teacher-replay"
            if config.teacher.provider.casefold() in {"frozen", "frozen-replay"}
            else "real-live-teacher"
        ),
    )


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args), cwd=root, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise AutonomousEvolutionError(f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def _freeze_copy(source: Path, target: Path) -> None:
    content = source.read_bytes()
    if target.exists():
        if target.read_bytes() != content:
            raise AutonomousEvolutionError(
                f"immutable round artifact conflict: {target.name}"
            )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        if os.write(descriptor, content) != len(content):
            raise AutonomousEvolutionError(
                f"partial round artifact write: {target.name}"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class AutonomousEvolutionRunner:
    def __init__(
        self,
        *,
        config: AutonomousEvolutionConfig,
        output_root: str | Path,
        worktree_root: str | Path,
        dependencies: EvolutionDependencies,
    ) -> None:
        self.config = config
        self.output_root = Path(output_root).expanduser().resolve()
        self.worktree_root = Path(worktree_root).expanduser().resolve()
        self.dependencies = dependencies
        self.selector = FeedbackTaskSelector(
            config.swe_bench.task_pool,
            source_pool=config.swe_bench.source_pool,
        )
        self.state_store = GoalStateStore(self.output_root / "EVOLUTION-STATE.json")
        self.selection_index = HashChainIndex(
            self.output_root / "TASK-SELECTION-INDEX.jsonl",
            index_id=f"{config.goal.goal_id}:task-selections:v1",
        )
        self.teacher_index = HashChainIndex(
            self.output_root / "TEACHER-CALL-LEDGER.jsonl",
            index_id=f"{config.goal.goal_id}:teacher-calls:v1",
        )
        self.round_index = HashChainIndex(
            self.output_root / "ROUND-INDEX.jsonl",
            index_id=f"{config.goal.goal_id}:rounds:v1",
        )

    def run(self) -> dict[str, Any]:
        head = _git(self.worktree_root, "rev-parse", "HEAD")
        if _git(self.worktree_root, "status", "--porcelain", "--untracked-files=all"):
            raise AutonomousEvolutionError(
                "autonomous evolution requires a clean committed worktree"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        identity_sha, identity_files = model_identity(self.config.model)
        config_identity = hashlib.sha256(
            canonical_json(_config_identity_payload(self.config)).encode()
        ).hexdigest()
        freeze_json(
            self.output_root / "GOAL.json",
            {
                "schema_version": 1,
                **asdict(self.config.goal),
                "model_identity_sha256": identity_sha,
                "model_identity_files": identity_files,
                "source_commit_sha": head,
                "config_sha256": config_identity,
                "feedback_only": True,
                "no_weight_training": True,
            },
        )
        best_path = self.output_root / "best/BEST-HARNESS.json"
        if not best_path.is_file():
            export_empty_harness(
                output_root=self.output_root,
                model_identity_sha256=identity_sha,
            )
        state = self.state_store.load_or_create(goal_id=self.config.goal.goal_id)
        replay = self._rebuild_completed_rounds(state)
        state = replay.state
        if state.status is not GoalRunStatus.ACTIVE:
            return self._terminal_result(state)

        proposer = CandidateProposer(
            root=self.output_root / "teacher",
            provider=self.config.teacher.provider,
            model=self.config.teacher.model,
            transport=self.dependencies.teacher_transport,
            pricing=self.dependencies.teacher_pricing,
            hard_budget_cny=self.config.teacher.budget_cny,
        )
        prior_claims = replay.prior_claims
        rejected_candidate_ids = list(replay.rejected_candidate_ids)
        best_fitness = replay.best_fitness
        best_compiled_root = replay.best_round_root
        gained_tasks = set(state.native_gain_task_ids)

        for current_round in range(
            state.next_round_index, self.config.goal.max_rounds
        ):
            round_root = self.output_root / "rounds" / f"round-{current_round:04d}"
            round_root.mkdir(parents=True, exist_ok=True)
            selection = self.selector.select(
                round_index=current_round,
                count=self.config.execution.tasks_per_campaign,
                prior_claims=prior_claims,
            )
            selection_payload = _selection_payload(selection)
            freeze_json(round_root / "TASK-SELECTION.json", selection_payload)
            self.selection_index.append(
                event_id=selection.selection_id, payload=selection_payload
            )
            baseline_revision_id = (
                state.best_candidate_revision_id or "empty-harness-v1"
            )
            baseline_root = (
                best_compiled_root
                if state.best_candidate_revision_id is not None
                else None
            )
            probe_request = RoundExecutionRequest(
                goal_id=self.config.goal.goal_id,
                round_index=current_round,
                output_root=round_root,
                selection=selection,
                config=self.config,
                source_commit_sha=head,
                worktree_root=self.worktree_root,
                baseline_revision_id=baseline_revision_id,
                baseline_compiled_root=baseline_root,
            )
            baseline_path = round_root / "BASELINE-RESULT.json"
            if baseline_path.is_file():
                baseline = _load_baseline_result(baseline_path)
            else:
                baseline = self.dependencies.round_executor.baseline(probe_request)
                _validate_baseline(baseline, selection)
                freeze_json(baseline_path, _baseline_payload(baseline))
            _validate_baseline(baseline, selection)
            baseline_payload = _baseline_payload(baseline)
            failure_package = {
                "schema_version": 1,
                "goal": asdict(self.config.goal),
                "round_index": current_round,
                "model_identity_sha256": identity_sha,
                "current_best_harness": {
                    "candidate_revision_id": state.best_candidate_revision_id,
                    "compiled_bundle_sha256": state.best_bundle_sha256,
                },
                "selected_tasks": [dict(task) for task in selection.tasks],
                "baseline": baseline_payload,
                "prior_claims": [dict(row) for row in prior_claims],
                "failure_signatures": [dict(row) for row in baseline.failure_signatures],
                "regressions": [
                    dict(row)
                    for row in prior_claims
                    if row.get("classification") == "regression"
                ],
                "rejected_candidate_ids": rejected_candidate_ids,
                "constraints": {
                    "feedback_only": True,
                    "no_weight_training": True,
                    "candidate_inactive": True,
                    "do_not_modify_evaluator": True,
                },
            }
            freeze_json(round_root / "FAILURE-PACKAGE.json", failure_package)
            request_id = f"{self.config.goal.goal_id}-round-{current_round:04d}"
            try:
                proposal = proposer.propose(
                    request_id=request_id,
                    failure_package=failure_package,
                    max_output_tokens=self.config.teacher.max_output_tokens,
                )
            except BudgetExceeded:
                state = replace(state, status=GoalRunStatus.BUDGET_EXHAUSTED)
                self.state_store.write(state)
                return self._write_result(state, proposer=proposer)
            except ContractViolation as error:
                return self._block_integrity(
                    state=state,
                    proposer=proposer,
                    round_root=round_root,
                    phase="teacher-proposal",
                    error=error,
                )
            try:
                candidate = self._compile_candidate(
                    proposal=proposal,
                    round_index=current_round,
                    selection=selection,
                    parent_revision_id=baseline_revision_id,
                    round_root=round_root,
                )
            except ContractViolation as error:
                return self._block_integrity(
                    state=state,
                    proposer=proposer,
                    round_root=round_root,
                    phase="candidate-compilation",
                    error=error,
                )
            self.teacher_index.append(
                event_id=request_id,
                payload={
                    "request_path": str(proposal.request_path.relative_to(self.output_root)),
                    "request_sha256": file_sha256(proposal.request_path),
                    "response_path": str(proposal.response_path.relative_to(self.output_root)),
                    "response_sha256": file_sha256(proposal.response_path),
                    "provider": proposal.usage.provider,
                    "model": proposal.usage.model,
                    "input_tokens": proposal.usage.input_tokens,
                    "output_tokens": proposal.usage.output_tokens,
                    "estimated_cost_cny": proposal.usage.estimated_cost_cny,
                    "candidate_id": candidate.change_set.candidate_id,
                    "candidate_revision_id": candidate.change_set.revision_id,
                    "compiled_bundle_sha256": candidate.bundle_sha256,
                },
            )
            for name, _ in candidate.artifact_sha256:
                _freeze_copy(candidate.root / name, round_root / name)
            _freeze_copy(candidate.manifest_path, round_root / candidate.manifest_path.name)
            candidate_request = replace(probe_request, candidate=candidate)
            prescreen_path = round_root / "PRESCREEN-RESULT.json"
            try:
                if prescreen_path.is_file():
                    prescreen = _load_prescreen_result(prescreen_path)
                    if prescreen.status == "completed":
                        _validate_prescreen(prescreen, candidate)
                elif self.config.execution.qwen_prescreen_count == 0:
                    prescreen = PrescreenResult(
                        candidate_revision_id=candidate.change_set.revision_id,
                        candidate_bundle_sha256=candidate.bundle_sha256,
                        model_receipt_ids=(),
                        structural_valid=True,
                        patch_applicable=True,
                        replayed=False,
                        status="skipped",
                    )
                else:
                    prescreen = self.dependencies.round_executor.prescreen(
                        candidate_request
                    )
                    _validate_prescreen(prescreen, candidate)
                freeze_json(prescreen_path, _prescreen_payload(prescreen))
                if not (prescreen.structural_valid and prescreen.patch_applicable):
                    claims: tuple[VerifiedCampaignClaim, ...] = ()
                    campaign_result: Mapping[str, Any] = {
                        "campaign_status": "screened_out",
                        "claims": [],
                    }
                else:
                    campaign_path = round_root / "CAMPAIGN-RESULT.json"
                    campaign_result = (
                        _load_json_object(campaign_path)
                        if campaign_path.is_file()
                        else self.dependencies.round_executor.paired(
                            candidate_request
                        )
                    )
                    freeze_json(campaign_path, dict(campaign_result))
                    claims = CampaignOutcomeVerifier().verify(
                        round_root=round_root,
                        result=campaign_result,
                        selected_task_ids=selection.selected_task_ids,
                        candidate_id=candidate.change_set.candidate_id,
                        candidate_revision_id=candidate.change_set.revision_id,
                        candidate_bundle_sha256=candidate.bundle_sha256,
                    )
            except ContractViolation as error:
                return self._block_integrity(
                    state=state,
                    proposer=proposer,
                    round_root=round_root,
                    phase="model-native-lineage",
                    error=error,
                )
            freeze_json(
                round_root / "CAMPAIGN-RESULT.json", dict(campaign_result)
            )
            claim_rows = [asdict(claim) for claim in claims]
            fitness = _fitness(claims)
            accepted = _acceptable_advance(claims) and (
                best_fitness is None or fitness > best_fitness
            )
            best_revision: str | None
            best_bundle: str | None
            if accepted:
                best_fitness = fitness
                best_compiled_root = candidate.root
                gained_tasks.update(
                    claim.task_id for claim in claims if claim.classification == "gain"
                )
                export_best_harness(
                    output_root=self.output_root,
                    round_root=round_root,
                    model_identity_sha256=identity_sha,
                    candidate_id=candidate.change_set.candidate_id,
                    candidate_revision_id=candidate.change_set.revision_id,
                    bundle_sha256=candidate.bundle_sha256,
                    claims=claims,
                )
                no_progress = 0
                best_revision = candidate.change_set.revision_id
                best_bundle = candidate.bundle_sha256
            else:
                rejected_candidate_ids.append(candidate.change_set.candidate_id)
                no_progress = state.no_progress_rounds + 1
                best_revision = state.best_candidate_revision_id
                best_bundle = state.best_bundle_sha256
            infra = bool(claims) and all(
                claim.classification == "infra_failure" for claim in claims
            )
            consecutive_infra = state.consecutive_infra_failures + 1 if infra else 0
            round_payload = {
                "schema_version": 1,
                "round_index": current_round,
                "selection_id": selection.selection_id,
                "baseline_replayed": baseline.replayed,
                "candidate_id": candidate.change_set.candidate_id,
                "candidate_revision_id": candidate.change_set.revision_id,
                "parent_revision_id": baseline_revision_id,
                "compiled_bundle_sha256": candidate.bundle_sha256,
                "prescreen": _prescreen_payload(prescreen),
                "claims": claim_rows,
                "fitness": fitness,
                "accepted_as_best": accepted,
                "best_candidate_revision_id": best_revision,
                "best_bundle_sha256": best_bundle,
                "campaign_status": campaign_result.get("campaign_status"),
            }
            freeze_json(
                round_root / "AUTONOMOUS-ROUND-RESULT.json", round_payload
            )
            seal_manifest(round_root)
            self.round_index.append(
                event_id=f"round-{current_round:04d}", payload=round_payload
            )
            prior_claims = tuple(claim_rows)
            state = GoalState(
                schema_version=1,
                goal_id=state.goal_id,
                status=GoalRunStatus.ACTIVE,
                next_round_index=current_round + 1,
                rounds_completed=current_round + 1,
                no_progress_rounds=no_progress,
                consecutive_infra_failures=consecutive_infra,
                native_gain_task_ids=tuple(sorted(gained_tasks)),
                best_candidate_revision_id=best_revision,
                best_bundle_sha256=best_bundle,
            )
            if len(gained_tasks) >= self.config.goal.target_native_gains:
                state = replace(state, status=GoalRunStatus.GOAL_REACHED)
            elif no_progress >= self.config.goal.no_progress_patience:
                state = replace(state, status=GoalRunStatus.NO_PROGRESS)
            elif consecutive_infra >= 3:
                state = replace(
                    state, status=GoalRunStatus.BLOCKED_INFRASTRUCTURE
                )
            self.state_store.write(state)
            if state.status is not GoalRunStatus.ACTIVE:
                return self._write_result(state, proposer=proposer)

        state = replace(state, status=GoalRunStatus.MAX_ROUNDS_REACHED)
        self.state_store.write(state)
        return self._write_result(state, proposer=proposer)

    def _compile_candidate(
        self,
        *,
        proposal: ProposalResult,
        round_index: int,
        selection: TaskSelection,
        parent_revision_id: str,
        round_root: Path,
    ) -> CompiledRevision:
        candidate = proposal.candidate
        operator = getattr(candidate, "operator", None)
        revision_id = f"{candidate.candidate_id}-r{round_index:04d}"
        compile_spec = CompileSpec(
            candidate_id=candidate.candidate_id,
            revision_id=revision_id,
            parent_revision_id=parent_revision_id,
            cohort=Cohort.FEEDBACK,
        )
        if not isinstance(operator, Mapping):
            operator_id = f"operator-{candidate.candidate_id[-16:]}"
            compile_spec = CompileSpec(
                candidate_id=candidate.candidate_id,
                revision_id=revision_id,
                parent_revision_id=parent_revision_id,
                cohort=Cohort.FEEDBACK,
                operator_id=operator_id,
                operator_instruction=candidate.skill_text,
                routes=tuple(
                    (task_id, operator_id)
                    for task_id in selection.selected_task_ids
                ),
            )
        compiled = CandidateCompiler().compile(
            request_path=proposal.request_path,
            response_path=proposal.response_path,
            compile_spec=compile_spec,
            output_root=round_root / "compiled-candidates",
        )
        if not set(selection.selected_task_ids).issubset(dict(compiled.router.routes)):
            raise AutonomousEvolutionError(
                "compiled Teacher Router does not cover selected feedback tasks"
            )
        return compiled

    def _rebuild_completed_rounds(self, state: GoalState) -> _ReplayState:
        rows = self.round_index.rows()
        if len(rows) not in {state.rounds_completed, state.rounds_completed + 1}:
            raise AutonomousEvolutionError(
                "evolution state disagrees with the round index"
            )
        prior: tuple[Mapping[str, Any], ...] = ()
        rejected: list[str] = []
        best_fitness: int | None = None
        best_round: Path | None = None
        best_candidate: str | None = None
        for sequence, row in enumerate(rows):
            payload = row["payload"]
            if payload.get("round_index") != sequence:
                raise AutonomousEvolutionError("round index sequence drift")
            round_root = self.output_root / "rounds" / f"round-{sequence:04d}"
            from evolve.reporting import AuditVerifier

            AuditVerifier().verify_manifest(
                round_root / "EVIDENCE-MANIFEST.json", root=round_root
            )
            prior = tuple(payload.get("claims", ()))
            if payload.get("accepted_as_best"):
                best_fitness = int(payload["fitness"])
                best_round = (
                    round_root
                    / "compiled-candidates"
                    / str(payload["candidate_revision_id"])
                )
                best_candidate = str(payload["candidate_id"])
            else:
                rejected.append(str(payload["candidate_id"]))
        if len(rows) == state.rounds_completed + 1:
            state = self._recover_state_from_rounds(state, rows)
            self.state_store.write(state)
        return _ReplayState(
            state=state,
            prior_claims=prior,
            rejected_candidate_ids=tuple(rejected),
            best_fitness=best_fitness,
            best_round_root=best_round,
            best_candidate_id=best_candidate,
        )

    def _recover_state_from_rounds(
        self, state: GoalState, rows: Sequence[Mapping[str, Any]]
    ) -> GoalState:
        """Recover the one legal crash window: sealed round indexed before state."""

        gained = {
            str(claim["task_id"])
            for row in rows
            if row["payload"].get("accepted_as_best")
            for claim in row["payload"].get("claims", ())
            if claim.get("classification") == "gain"
        }
        tail_no_progress = 0
        tail_infra = 0
        for row in reversed(rows):
            payload = row["payload"]
            if payload.get("accepted_as_best"):
                break
            tail_no_progress += 1
        for row in reversed(rows):
            claims = row["payload"].get("claims", ())
            if claims and all(
                claim.get("classification") == "infra_failure" for claim in claims
            ):
                tail_infra += 1
            else:
                break
        last = rows[-1]["payload"]
        recovered = GoalState(
            schema_version=state.schema_version,
            goal_id=state.goal_id,
            status=GoalRunStatus.ACTIVE,
            next_round_index=len(rows),
            rounds_completed=len(rows),
            no_progress_rounds=tail_no_progress,
            consecutive_infra_failures=tail_infra,
            native_gain_task_ids=tuple(sorted(gained)),
            best_candidate_revision_id=last.get("best_candidate_revision_id"),
            best_bundle_sha256=last.get("best_bundle_sha256"),
        )
        if len(gained) >= self.config.goal.target_native_gains:
            return replace(recovered, status=GoalRunStatus.GOAL_REACHED)
        if tail_no_progress >= self.config.goal.no_progress_patience:
            return replace(recovered, status=GoalRunStatus.NO_PROGRESS)
        if tail_infra >= 3:
            return replace(recovered, status=GoalRunStatus.BLOCKED_INFRASTRUCTURE)
        return recovered

    def _terminal_result(self, state: GoalState) -> dict[str, Any]:
        path = self.output_root / "EVOLUTION-RESULT.json"
        if not path.is_file():
            raise AutonomousEvolutionError("terminal state has no evolution result")
        from evolve.reporting import AuditVerifier

        AuditVerifier().verify_manifest(
            self.output_root / "EVIDENCE-MANIFEST.json", root=self.output_root
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def _block_integrity(
        self,
        *,
        state: GoalState,
        proposer: CandidateProposer,
        round_root: Path,
        phase: str,
        error: Exception,
    ) -> dict[str, Any]:
        blocked = replace(state, status=GoalRunStatus.BLOCKED_INTEGRITY)
        freeze_json(
            round_root / "INTEGRITY-BLOCK.json",
            {
                "schema_version": 1,
                "phase": phase,
                "error_type": type(error).__name__,
                "error": str(error),
                "best_candidate_revision_id": state.best_candidate_revision_id,
                "best_bundle_sha256": state.best_bundle_sha256,
                "best_advanced": False,
            },
        )
        seal_manifest(round_root)
        self.state_store.write(blocked)
        return self._write_result(blocked, proposer=proposer)

    def _write_result(
        self, state: GoalState, *, proposer: CandidateProposer
    ) -> dict[str, Any]:
        best_path = self.output_root / "best" / "BEST-HARNESS.json"
        result = {
            "schema_version": 1,
            "goal_id": state.goal_id,
            "status": str(state.status),
            "rounds_completed": state.rounds_completed,
            "native_gain_task_count": len(state.native_gain_task_ids),
            "native_gain_task_ids": list(state.native_gain_task_ids),
            "best_candidate_revision_id": state.best_candidate_revision_id,
            "best_bundle_sha256": state.best_bundle_sha256,
            "best_harness_path": str(best_path) if best_path.is_file() else None,
            "teacher_provider": self.config.teacher.provider,
            "teacher_model": self.config.teacher.model,
            "teacher_spend_cny": proposer.cost_ledger.snapshot().spent_cost_cny,
            "holdout_opened": False,
            "skill_active": False,
            "evidence_mode": self.dependencies.evidence_mode,
            "product_status": (
                "autonomous_loop_verified"
                if state.status is GoalRunStatus.GOAL_REACHED
                and self.dependencies.evidence_mode.startswith("real-")
                else "offline_e2e_verified"
                if state.status is GoalRunStatus.GOAL_REACHED
                else "partial"
            ),
        }
        atomic_json(self.output_root / "EVOLUTION-RESULT.json", result)
        seal_manifest(self.output_root)
        return result


@dataclass(frozen=True, slots=True)
class _ReplayState:
    state: GoalState
    prior_claims: tuple[Mapping[str, Any], ...]
    rejected_candidate_ids: tuple[str, ...]
    best_fitness: int | None
    best_round_root: Path | None
    best_candidate_id: str | None


def _selection_payload(selection: TaskSelection) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "selection_id": selection.selection_id,
        "round_index": selection.round_index,
        "selected_task_ids": list(selection.selected_task_ids),
        "selected_projects": list(selection.selected_projects),
        "selection_reason": list(selection.selection_reason),
        "excluded": list(selection.excluded),
        "tasks": [dict(task) for task in selection.tasks],
    }


def _baseline_payload(result: BaselineProbeResult) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_ids": list(result.task_ids),
        "model_receipt_ids": list(result.model_receipt_ids),
        "native_receipt_ids": list(result.native_receipt_ids),
        "native_outcomes": [dict(row) for row in result.native_outcomes],
        "failure_signatures": [dict(row) for row in result.failure_signatures],
        "replayed": result.replayed,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AutonomousEvolutionError(
            f"frozen autonomous artifact is unreadable: {path.name}"
        ) from error
    if not isinstance(payload, dict):
        raise AutonomousEvolutionError(
            f"frozen autonomous artifact is not an object: {path.name}"
        )
    return payload


def _load_baseline_result(path: Path) -> BaselineProbeResult:
    payload = _load_json_object(path)
    try:
        return BaselineProbeResult(
            task_ids=tuple(payload["task_ids"]),
            model_receipt_ids=tuple(payload["model_receipt_ids"]),
            native_receipt_ids=tuple(payload["native_receipt_ids"]),
            native_outcomes=tuple(payload["native_outcomes"]),
            failure_signatures=tuple(payload["failure_signatures"]),
            replayed=bool(payload["replayed"]),
        )
    except (KeyError, TypeError) as error:
        raise AutonomousEvolutionError("frozen baseline result is invalid") from error


def _config_identity_payload(config: AutonomousEvolutionConfig) -> dict[str, Any]:
    """Return the path-normalized product input bound into GOAL.json."""

    return {
        "schema_version": config.schema_version,
        "goal": asdict(config.goal),
        "model": {
            **asdict(config.model),
            "model_path": str(config.model.model_path),
        },
        "swe_bench": {
            **asdict(config.swe_bench),
            "task_pool": str(config.swe_bench.task_pool),
            "source_pool": str(config.swe_bench.source_pool),
            "official_harness": str(config.swe_bench.official_harness),
            "official_evaluator": str(config.swe_bench.official_evaluator),
        },
        "teacher": asdict(config.teacher),
        "execution": asdict(config.execution),
    }


def _prescreen_payload(result: PrescreenResult) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "candidate_revision_id": result.candidate_revision_id,
        "candidate_bundle_sha256": result.candidate_bundle_sha256,
        "model_receipt_ids": list(result.model_receipt_ids),
        "structural_valid": result.structural_valid,
        "patch_applicable": result.patch_applicable,
        "replayed": result.replayed,
        "status": result.status,
    }


def _load_prescreen_result(path: Path) -> PrescreenResult:
    payload = _load_json_object(path)
    try:
        return PrescreenResult(
            candidate_revision_id=str(payload["candidate_revision_id"]),
            candidate_bundle_sha256=str(payload["candidate_bundle_sha256"]),
            model_receipt_ids=tuple(payload["model_receipt_ids"]),
            structural_valid=bool(payload["structural_valid"]),
            patch_applicable=bool(payload["patch_applicable"]),
            replayed=bool(payload["replayed"]),
            status=str(payload["status"]),
        )
    except (KeyError, TypeError) as error:
        raise AutonomousEvolutionError("frozen prescreen result is invalid") from error


def _validate_baseline(result: BaselineProbeResult, selection: TaskSelection) -> None:
    if set(result.task_ids) != set(selection.selected_task_ids):
        raise AutonomousEvolutionError("baseline result task selection mismatch")
    if len(result.model_receipt_ids) != len(selection.selected_task_ids):
        raise AutonomousEvolutionError("baseline did not produce real model receipts")
    if len(result.native_receipt_ids) != len(selection.selected_task_ids):
        raise AutonomousEvolutionError("baseline did not produce native receipts")


def _validate_prescreen(
    result: PrescreenResult, candidate: CompiledRevision
) -> None:
    if (
        result.candidate_revision_id != candidate.change_set.revision_id
        or result.candidate_bundle_sha256 != candidate.bundle_sha256
        or not result.model_receipt_ids
    ):
        raise AutonomousEvolutionError(
            "Qwen prescreen did not produce candidate-bound model receipts"
        )


def _fitness(claims: Sequence[VerifiedCampaignClaim]) -> int:
    weights = {"gain": 10, "neutral": 0, "regression": -100}
    return sum(weights.get(claim.classification, 0) for claim in claims)


def _acceptable_advance(claims: Sequence[VerifiedCampaignClaim]) -> bool:
    classes = [claim.classification for claim in claims]
    return (
        "gain" in classes
        and "regression" not in classes
        and "infra_failure" not in classes
    )


def _provider_pricing(
    provider: str, model: str
) -> PricingCnyPerMillionTokens:
    """Conservative provider pricing used for pre-dispatch authorization.

    Unknown live providers fail closed instead of performing an unpriced call.
    Frozen replay is zero-cost because it performs no network dispatch.
    """

    if provider in {"frozen", "frozen-replay"}:
        return PricingCnyPerMillionTokens(input=0, output=0)
    if provider.casefold() == "deepseek" and model:
        return PricingCnyPerMillionTokens(input=2.0, output=8.0)
    raise AutonomousEvolutionError(
        "live Teacher provider pricing is not configured for budget authorization"
    )


__all__ = [
    "AutonomousEvolutionRunner",
    "BaselineProbeResult",
    "EvolutionDependencies",
    "EvolutionRoundExecutor",
    "PrescreenResult",
    "RoundExecutionRequest",
    "build_default_dependencies",
]
