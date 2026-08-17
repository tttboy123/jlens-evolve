"""Fail-closed bridge from the v2.1 evolution engine to real Codex/native evals."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent_arm_runner import (
    ArmProgram,
    build_agent_prompt,
    build_multi_arm_invocations,
    run_patch_arm,
    validate_patch_arm_pre_dispatch,
)
from benchmark_adapters import TaskPool
from benchmark_execution import materialize_task_contract
from candidate_tournament import ArmEvaluation
from continuous_ab import BaselineContract
from evolution_controller import EvolutionController
from evolution_runtime import ExecutionArtifact, ExecutionRequest
from mutation_proposer import InactiveChangeSet, MutationRequest, ProposalResult
from native_result_adapter import PatchAdmissionContract, normalize_patch_result
from pattern_miner import FrozenObservationEvidence


class BridgeContractError(ValueError):
    """Raised before a real call when execution evidence is incomplete or unmatched."""


_REAL_PATCH_BENCHMARKS = frozenset(
    {"swe-bench-verified", "swe-bench-multilingual", "multi-swe-bench-flash"}
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _stage_for(index: int) -> tuple[int, str]:
    generation, within = divmod(index, 25)
    if generation == 0:
        return generation, "observe"
    if within < 5:
        return generation, "scout"
    if within < 13:
        return generation, "semifinal"
    return generation, "confirmation"


def freeze_search_task_schedule(
    frozen_pool_path: Path, output_path: Path
) -> dict[str, Any]:
    """Select 100 unopened search identities without reading task instructions/gold."""

    frozen_pool_path = frozen_pool_path.resolve()
    source_sha256 = _sha256_file(frozen_pool_path)
    pool = TaskPool.load(frozen_pool_path)
    records = [
        record
        for record in pool.records
        if record.assigned_partition == "search"
        and record.state == "unopened"
        and record.benchmark_id in _REAL_PATCH_BENCHMARKS
    ]
    if len(records) < 100:
        raise BridgeContractError(
            f"frozen pool has fewer than 100 unopened search tasks: {len(records)}"
        )
    tasks = []
    for ordinal, record in enumerate(records[:100]):
        generation, stage = _stage_for(ordinal)
        tasks.append(
            {
                "ordinal": ordinal + 1,
                "generation": generation,
                "stage": stage,
                "task_uid": record.task_uid,
                "benchmark_id": record.benchmark_id,
                "instance_id": record.instance_id,
                "task_contract_sha256": _sha256_json(record.task_contract),
            }
        )
    allocation = Counter(row["benchmark_id"] for row in tasks)
    unsigned = {
        "schema_version": "1.0",
        "status": "frozen_before_task_materialization",
        "source_pool_path": str(frozen_pool_path),
        "source_pool_sha256": source_sha256,
        "tasks": tasks,
        "benchmark_allocation": dict(sorted(allocation.items())),
        "excluded_search_benchmark_families": ["terminal-bench-2"],
        "exclusion_reason": (
            "real v2.1.1 bridge freezes patch benchmarks only; Harbor executes its "
            "Agent inside the evaluator and is not contract-equivalent to patch arms"
        ),
        "unique_search_tasks": 100,
        "promotion_tasks_opened": 0,
        "final_sealed_tasks_opened": 0,
        "task_content_included": False,
        "gold_fields_included": False,
    }
    payload = {**unsigned, "semantic_fingerprint": _sha256_json(unsigned)}
    output_path = output_path.resolve()
    if output_path.exists():
        existing = load_search_task_schedule(output_path)
        if existing != payload:
            raise BridgeContractError("frozen search task schedule is immutable")
        return existing
    _atomic_json(output_path, payload)
    return payload


def load_search_task_schedule(path: Path) -> dict[str, Any]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BridgeContractError("unsupported search task schedule")
    schema = payload.get("schema_version")
    if schema == "1.0":
        resume = None
        expected_count = 100
    elif schema == "1.1-resume":
        resume = payload.get("resume")
        if (
            not isinstance(resume, dict)
            or not isinstance(resume.get("of_semantic_fingerprint"), str)
            or not isinstance(resume.get("lost_task_uids"), list)
            or len(resume.get("lost_task_uids", [])) != 9
            or not isinstance(resume.get("incident_reference"), str)
        ):
            raise BridgeContractError("resume schedule metadata is incomplete")
        expected_count = 91
    else:
        raise BridgeContractError("unsupported search task schedule")
    fingerprint = payload.get("semantic_fingerprint")
    unsigned = {
        key: value for key, value in payload.items() if key != "semantic_fingerprint"
    }
    if fingerprint != _sha256_json(unsigned):
        raise BridgeContractError("search task schedule fingerprint mismatch")
    tasks = payload.get("tasks")
    if (
        not isinstance(tasks, list)
        or len(tasks) != expected_count
        or len({row.get("task_uid") for row in tasks if isinstance(row, dict)})
        != expected_count
    ):
        raise BridgeContractError(
            f"search task schedule must contain {expected_count} unique tasks"
        )
    if resume is not None:
        lost = {uid for uid in resume["lost_task_uids"] if isinstance(uid, str)}
        if len(lost) != 9 or any(
            isinstance(row, dict) and row.get("task_uid") in lost for row in tasks
        ):
            raise BridgeContractError(
                "resume lost tasks must be absent from the resume schedule"
            )
    if (
        payload.get("promotion_tasks_opened") != 0
        or payload.get("final_sealed_tasks_opened") != 0
        or payload.get("task_content_included") is not False
        or payload.get("gold_fields_included") is not False
    ):
        raise BridgeContractError(
            "search task schedule crossed a sealed/content boundary"
        )
    return payload


def materialize_evolution_claimed_task(
    *,
    controller: EvolutionController,
    schedule_path: Path,
    task_uid: str,
    pool_root: Path,
    output_path: Path,
    terminal_dataset_root: Path | None = None,
) -> dict[str, Any]:
    """Open one task only after the new EvolutionController has claimed its stage."""

    output_path = output_path.resolve()
    task_state = controller.task_state(task_uid)
    resuming_retired = task_state == "retired" and output_path.is_file()
    if not controller.task_materialization_allowed(task_uid) and not resuming_retired:
        raise BridgeContractError(
            "evolution task must be claimed before materialization"
        )
    schedule = load_search_task_schedule(schedule_path)
    scheduled = [row for row in schedule["tasks"] if row["task_uid"] == task_uid]
    if len(scheduled) != 1:
        raise BridgeContractError("claimed task is absent from frozen search schedule")
    scheduled_row = scheduled[0]
    pool = TaskPool.load(pool_root.resolve() / "TASK_POOL.json")
    records = [record for record in pool.records if record.task_uid == task_uid]
    if len(records) != 1:
        raise BridgeContractError("scheduled task is absent from frozen task pool")
    record = records[0]
    if record.assigned_partition != "search" or record.state != "unopened":
        raise BridgeContractError("only an unopened search-pool source may be bridged")
    if _sha256_json(record.task_contract) != scheduled_row["task_contract_sha256"]:
        raise BridgeContractError("scheduled task contract hash mismatch")
    round_id = (
        f"g{scheduled_row['generation']}-{scheduled_row['stage']}-{task_uid[:16]}"
    )
    payload = materialize_task_contract(
        task_contract=record.task_contract,
        round_id=round_id,
        pool_root=pool_root,
        terminal_dataset_root=terminal_dataset_root,
    )
    payload["evolution_generation"] = scheduled_row["generation"]
    payload["evolution_stage"] = scheduled_row["stage"]
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise BridgeContractError("materialized task evidence is immutable")
        return existing
    _atomic_json(output_path, payload)
    return payload


@dataclass(frozen=True)
class ObservationFeatures:
    observed_features: tuple[str, ...]
    conditions: tuple[str, ...]
    expected_surfaces: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, values in (
            ("observed_features", self.observed_features),
            ("conditions", self.conditions),
            ("expected_surfaces", self.expected_surfaces),
        ):
            if not values or len(set(values)) != len(values):
                raise BridgeContractError(
                    f"observer {name} must be unique and non-empty"
                )


@dataclass(frozen=True)
class ObservationContext:
    request: ExecutionRequest
    materialized_task: dict[str, Any]
    trajectory_path: Path
    tool_events_path: Path
    native_evidence_path: Path
    cost_path: Path
    safety_path: Path


@dataclass(frozen=True)
class _ArmDescriptor:
    name: str
    agent_program_sha256: str
    profile_root: Path


class RealEvolutionAdapters:
    """Inject real Codex/native evaluator evidence into the generic evolution loop."""

    real_codex_calls = True
    accounts_auxiliary_calls = True

    def __init__(
        self,
        *,
        controller: EvolutionController,
        schedule_path: Path,
        pool_root: Path,
        baseline: BaselineContract,
        profile_roots: dict[str, Path],
        run_root: Path,
        codex_executable: Path,
        authorization: dict[str, Any],
        workspace_factory: Callable[[dict[str, Any], _ArmDescriptor], Path],
        native_evaluator: Callable[[Any, dict[str, Any], dict[str, Any]], Path],
        observer: Callable[[ObservationContext], ObservationFeatures],
        proposer: Any,
        terminal_dataset_root: Path | None = None,
    ) -> None:
        self.controller = controller
        self.schedule_path = schedule_path.resolve()
        self.pool_root = pool_root.resolve()
        self.baseline = baseline
        self.profile_roots = {
            digest: path.resolve() for digest, path in profile_roots.items()
        }
        self.run_root = run_root.resolve()
        self.codex_executable = codex_executable.resolve()
        self.workspace_factory = workspace_factory
        self.native_evaluator = native_evaluator
        self.observer = observer
        self.proposer_adapter = proposer
        self.terminal_dataset_root = (
            None if terminal_dataset_root is None else terminal_dataset_root.resolve()
        )
        self.execution_authorization = self._translate_authorization(authorization)
        self._invocations: dict[str, dict[str, Any]] = {}
        self._native_scores: dict[tuple[str, str], float] = {}
        load_search_task_schedule(self.schedule_path)

    def _translate_authorization(self, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("status") != "AUTHORIZED_WITH_HARD_CAPS":
            raise BridgeContractError("real evolution authorization is missing")
        expected = asdict(self.controller.authorization)
        scope = value.get("scope")
        if not isinstance(scope, dict) or any(
            scope.get(key) != expected_value for key, expected_value in expected.items()
        ):
            raise BridgeContractError("real evolution authorization caps mismatch")
        return {
            "status": "authorized",
            "effective_caps": {
                "maximum_agent_calls": expected["maximum_real_codex_calls"]
            },
        }

    @staticmethod
    def require_native_report(path: Path) -> Path:
        path = path.resolve()
        if not path.is_file():
            raise BridgeContractError("native evaluator report is missing")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BridgeContractError("native evaluator report is invalid") from error
        if not isinstance(payload, dict):
            raise BridgeContractError("native evaluator report must be an object")
        return path

    def _claim_arm_names(self, request: ExecutionRequest) -> dict[str, str]:
        key = f"g{request.generation}:{request.stage}"
        claim = self.controller.inspect()["claims"].get(key)
        if claim is None or claim.get("status") not in {"claimed", "complete"}:
            raise BridgeContractError("execution stage is not claimed")
        names = {}
        for index, digest in enumerate(claim["arm_sha256s"]):
            name = (
                "original"
                if index == 0
                else "parent"
                if index == 1
                else f"candidate-{index - 2}"
            )
            names[digest] = name
        if request.arm_sha256 not in names:
            raise BridgeContractError("execution arm is outside the claimed stage")
        return names

    def _task_invocations(
        self, request: ExecutionRequest, materialized: dict[str, Any]
    ) -> dict[str, Any]:
        existing = self._invocations.get(request.task_uid)
        if existing is not None:
            return existing
        names = self._claim_arm_names(request)
        arms = []
        for digest, name in names.items():
            profile_root = self.profile_roots.get(digest)
            if profile_root is None:
                raise BridgeContractError(f"AgentProgram profile is missing: {digest}")
            descriptor = _ArmDescriptor(name, digest, profile_root)
            workspace = self.workspace_factory(materialized, descriptor).resolve()
            arms.append(ArmProgram(name, digest, profile_root, workspace))
        order = {"original": 0, "parent": 1}
        arms.sort(key=lambda arm: (order.get(arm.name, 2), arm.name))
        evidence_root = (
            self.run_root
            / "evidence"
            / f"generation-{request.generation}"
            / request.stage
            / request.task_uid
        )
        invocations = build_multi_arm_invocations(
            baseline=self.baseline,
            materialized_task=materialized,
            arms=tuple(arms),
            evidence_root=evidence_root,
        )
        self._invocations[request.task_uid] = invocations
        return invocations

    @staticmethod
    def _write_tool_events(trajectory_path: Path, output_path: Path) -> None:
        def is_tool_event(value: dict[str, Any]) -> bool:
            stack: list[Any] = [value]
            while stack:
                current = stack.pop()
                if isinstance(current, dict):
                    kind = str(current.get("type", "")).lower()
                    if (
                        "tool" in kind
                        or kind
                        in {
                            "command_execution",
                            "file_change",
                            "file_edit",
                            "apply_patch",
                        }
                        or "tool" in current
                        or "command" in current
                    ):
                        return True
                    stack.extend(current.values())
                elif isinstance(current, list):
                    stack.extend(current)
            return False

        events = []
        for line in trajectory_path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and is_tool_event(value):
                events.append(value)
        _atomic_json(output_path, {"schema_version": 1, "events": events})

    def _evidence_ref(self, path: Path) -> dict[str, str]:
        path = path.resolve()
        if not path.is_relative_to(self.run_root):
            raise BridgeContractError("observer evidence escaped real run root")
        return {
            "path": path.relative_to(self.run_root).as_posix(),
            "sha256": _sha256_file(path),
        }

    @staticmethod
    def _artifact_payload(artifact: ExecutionArtifact) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "arm": artifact.arm.to_dict(),
            "observation": (
                None if artifact.observation is None else artifact.observation.to_dict()
            ),
        }

    def _load_execution_artifact(
        self, path: Path, request: ExecutionRequest
    ) -> ExecutionArtifact | None:
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or set(payload) != {
            "schema_version",
            "arm",
            "observation",
        }:
            raise BridgeContractError("persisted execution artifact is invalid")
        arm = ArmEvaluation.from_dict(payload["arm"])
        observation_payload = payload["observation"]
        observation = (
            None
            if observation_payload is None
            else FrozenObservationEvidence.from_dict(observation_payload)
        )
        artifact = ExecutionArtifact(arm=arm, observation=observation)
        if (
            arm.task_uid != request.task_uid
            or arm.agent_program_sha256 != request.arm_sha256
        ):
            raise BridgeContractError("persisted execution artifact is mismatched")
        reservation = self.controller.arm_call_status(
            generation=request.generation,
            stage=request.stage,
            task_uid=request.task_uid,
            arm_sha256=request.arm_sha256,
        )
        if reservation is None:
            raise BridgeContractError("execution artifact has no call reservation")
        if reservation["status"] == "completed" and (
            reservation["evidence_sha256"] != arm.evidence_sha256
        ):
            raise BridgeContractError("execution artifact differs from call ledger")
        self._native_scores[(request.task_uid, request.arm_sha256)] = arm.native_score
        return artifact

    @staticmethod
    def _resume_agent_receipt(
        *,
        invocation: Any,
        prompt: str,
        reservation: dict[str, Any],
    ) -> dict[str, Any]:
        evidence_dir = Path(invocation.evidence_dir).resolve()
        receipt_path = evidence_dir / "agent-receipt.json"
        if not receipt_path.is_file():
            raise BridgeContractError(
                "real Codex call is reserved without a complete Agent receipt"
            )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        claimed = receipt.get("integrity_sha256")
        unsigned = {
            key: value for key, value in receipt.items() if key != "integrity_sha256"
        }
        if claimed != _sha256_json(unsigned):
            raise BridgeContractError("persisted Agent receipt integrity mismatch")
        expected = {
            "round_id": invocation.round_id,
            "arm": invocation.arm,
            "task_uid": invocation.task_uid,
            "benchmark_id": invocation.benchmark_id,
            "instance_id": invocation.instance_id,
            "agent_program_sha256": invocation.agent_program_sha256,
            "baseline_contract_sha256": invocation.baseline_contract_sha256,
            "matched_contract_sha256": invocation.matched_contract_sha256,
            "evaluator_epoch": invocation.evaluator_epoch,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise BridgeContractError("persisted Agent receipt contract mismatch")
        for field in ("prediction", "raw_events", "stderr"):
            reference = receipt.get(field)
            path = (
                Path(reference.get("path", "")).resolve()
                if isinstance(reference, dict)
                else None
            )
            if (
                path is None
                or not path.is_file()
                or not path.is_relative_to(evidence_dir)
                or reference.get("sha256") != _sha256_file(path)
            ):
                raise BridgeContractError(f"persisted Agent {field} evidence mismatch")
        if reservation.get("status") not in {"reserved", "completed"}:
            raise BridgeContractError("Agent call reservation is invalid")
        return receipt

    def execute(self, request: ExecutionRequest) -> ExecutionArtifact:
        task_input_path = self.run_root / "tasks" / request.task_uid / "task-input.json"
        materialized = materialize_evolution_claimed_task(
            controller=self.controller,
            schedule_path=self.schedule_path,
            task_uid=request.task_uid,
            pool_root=self.pool_root,
            output_path=task_input_path,
            terminal_dataset_root=self.terminal_dataset_root,
        )
        invocations = self._task_invocations(request, materialized)
        name = self._claim_arm_names(request)[request.arm_sha256]
        invocation = invocations[name]
        prompt = build_agent_prompt(
            materialized,
            Path(invocation.profile_root),
            invocation.agent_program_sha256,
        )
        evidence_dir = Path(invocation.evidence_dir).resolve()
        artifact_path = evidence_dir / "execution-artifact.json"
        resumed_artifact = self._load_execution_artifact(artifact_path, request)
        if resumed_artifact is not None:
            return resumed_artifact
        validate_patch_arm_pre_dispatch(
            invocation,
            prompt=prompt,
            authorization=self.execution_authorization,
        )
        reservation = self.controller.reserve_arm_call(
            generation=request.generation,
            stage=request.stage,
            task_uid=request.task_uid,
            arm_sha256=request.arm_sha256,
        )
        if reservation["dispatch_allowed"] is True:
            receipt = run_patch_arm(
                invocation,
                prompt=prompt,
                codex_executable=self.codex_executable,
                authorization=self.execution_authorization,
            )
        else:
            receipt = self._resume_agent_receipt(
                invocation=invocation,
                prompt=prompt,
                reservation=reservation,
            )
        report_path = self.require_native_report(
            self.native_evaluator(invocation, materialized, receipt)
        )
        admission = normalize_patch_result(
            contract=PatchAdmissionContract(
                round_id=invocation.round_id,
                arm=invocation.arm,
                task_uid=invocation.task_uid,
                benchmark_id=invocation.benchmark_id,
                instance_id=invocation.instance_id,
                agent_program_sha256=invocation.agent_program_sha256,
                baseline_contract_sha256=invocation.baseline_contract_sha256,
                evaluator_epoch=invocation.evaluator_epoch,
            ),
            prediction_path=Path(receipt["prediction"]["path"]),
            agent_receipt_path=evidence_dir / "agent-receipt.json",
            evaluator_report_path=report_path,
            evidence_path=evidence_dir / "native-admission.json",
        )
        native = admission.result
        native_score = (
            1.0 if native.resolved and native.regression_failures == 0 else 0.0
        )
        safety_passed = native.safe and native.regression_failures == 0
        cost_units = float(native.total_tokens)
        evidence_sha256 = str(admission.evidence["integrity_sha256"])
        arm = ArmEvaluation.from_dict(
            {
                "schema_version": 1,
                "task_uid": request.task_uid,
                "benchmark_family": native.benchmark_family,
                "role": request.role,
                "agent_program_sha256": request.arm_sha256,
                "matched_contract_sha256": invocation.matched_contract_sha256,
                "native_evaluator_epoch": native.evaluator_epoch,
                "native_score": native_score,
                "safety_passed": safety_passed,
                "cost_units": cost_units,
                "evidence_sha256": evidence_sha256,
            }
        )
        self._native_scores[(request.task_uid, request.arm_sha256)] = native_score

        trajectory_path = Path(receipt["raw_events"]["path"]).resolve()
        tool_events_path = evidence_dir / "tool-events.json"
        self._write_tool_events(trajectory_path, tool_events_path)
        cost_path = evidence_dir / "cost.json"
        _atomic_json(
            cost_path,
            {
                "schema_version": 1,
                "input_tokens": native.input_tokens,
                "output_tokens": native.output_tokens,
                "total_tokens": native.total_tokens,
                "elapsed_seconds": native.elapsed_seconds,
            },
        )
        safety_path = evidence_dir / "safety.json"
        _atomic_json(
            safety_path,
            {
                "schema_version": 1,
                "execution_safe": native.safe,
                "regression_failures": native.regression_failures,
                "safety_passed": safety_passed,
            },
        )
        context = ObservationContext(
            request=request,
            materialized_task=materialized,
            trajectory_path=trajectory_path,
            tool_events_path=tool_events_path,
            native_evidence_path=admission.evidence_path,
            cost_path=cost_path,
            safety_path=safety_path,
        )
        features = self.observer(context)
        if not isinstance(features, ObservationFeatures):
            raise BridgeContractError("observer did not return ObservationFeatures")
        if request.role == "original":
            reference_sha256 = request.original_sha256
            reference_score = native_score
        elif request.role == "parent":
            reference_sha256 = request.original_sha256
            reference_score = self._native_scores.get(
                (request.task_uid, request.original_sha256)
            )
        else:
            reference_sha256 = request.parent_sha256
            reference_score = self._native_scores.get(
                (request.task_uid, request.parent_sha256)
            )
        if reference_score is None:
            raise BridgeContractError(
                "observer reference arm native evidence is missing"
            )
        evidence_id = (
            f"g{request.generation}-{request.stage}-{request.task_uid[:16]}-"
            f"{request.arm_sha256[:12]}"
        )
        observation = FrozenObservationEvidence.from_dict(
            {
                "schema_version": 1,
                "evidence_id": evidence_id,
                "task_uid": request.task_uid,
                "benchmark_family": native.benchmark_family,
                "agent_program_sha256": request.arm_sha256,
                "parent_agent_program_sha256": reference_sha256,
                "native_evaluator_epoch": native.evaluator_epoch,
                "native_score_delta": native_score - reference_score,
                "safety_passed": safety_passed,
                "observed_features": list(features.observed_features),
                "conditions": list(features.conditions),
                "expected_surfaces": list(features.expected_surfaces),
                "evidence": {
                    "trajectory": self._evidence_ref(trajectory_path),
                    "tool_events": self._evidence_ref(tool_events_path),
                    "native_evaluator": self._evidence_ref(admission.evidence_path),
                    "cost": self._evidence_ref(cost_path),
                    "safety": self._evidence_ref(safety_path),
                },
                "causal_boundary": "observational_not_causal",
                "admission_gate_allowed": False,
            }
        )
        artifact = ExecutionArtifact(arm=arm, observation=observation)
        _atomic_json(artifact_path, self._artifact_payload(artifact))
        return artifact

    def propose(self, request: MutationRequest, generation: int) -> ProposalResult:
        if self.proposer_adapter is None:
            raise BridgeContractError("real MutationProposer adapter is not configured")
        result = self.proposer_adapter.propose(request, generation)
        if not isinstance(result, ProposalResult):
            raise BridgeContractError(
                "real MutationProposer returned an invalid result"
            )
        profile = self.proposer_adapter.profile_root(
            result.changeset.candidate_agent_program_sha256
        )
        self.profile_roots[result.changeset.candidate_agent_program_sha256] = Path(
            profile
        ).resolve()
        return result

    def rollback(self, changeset: InactiveChangeSet) -> dict[str, Any]:
        if self.proposer_adapter is None:
            raise BridgeContractError("real MutationProposer adapter is not configured")
        result = self.proposer_adapter.rollback(changeset)
        required = {"forward_patch_sha256", "rollback_patch_sha256", "verified"}
        if not isinstance(result, dict) or set(result) != required:
            raise BridgeContractError("rollback verification result is invalid")
        return result
