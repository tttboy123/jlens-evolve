"""Recoverable four-generation search controller with fail-closed budgets."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")


class ControllerContractError(ValueError):
    """Raised when generation, task, or authority state violates the plan."""


class BudgetContractError(ControllerContractError):
    """Raised before a hard authorization cap would be exceeded."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_sha(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ControllerContractError(f"invalid {name} sha256")
    return value


def _locked(method):
    """Serialize controller mutations so parallel arm execution stays atomic."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


@dataclass(frozen=True)
class StagePlan:
    name: str
    task_uids: tuple[str, ...]
    candidate_count: int
    expected_agent_calls: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task_uids": list(self.task_uids),
            "candidate_count": self.candidate_count,
            "expected_agent_calls": self.expected_agent_calls,
        }


@dataclass(frozen=True)
class GenerationPlan:
    generation: int
    task_uids: tuple[str, ...]
    stages: tuple[StagePlan, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "task_uids": list(self.task_uids),
            "stages": [stage.to_dict() for stage in self.stages],
        }


@dataclass(frozen=True)
class EvolutionPlan:
    schema_version: int
    generations: tuple[GenerationPlan, ...]
    unique_search_tasks: int
    planned_agent_task_calls: int
    maximum_proposer_calls: int
    maximum_reviewer_calls: int
    planned_real_codex_calls: int

    @classmethod
    def build(cls, task_uids: tuple[str, ...]) -> EvolutionPlan:
        if len(task_uids) != 100:
            raise ControllerContractError("evolution plan requires exactly 100 tasks")
        return cls._build_generations(
            task_uids,
            generation_sizes=(25, 25, 25, 25),
            unique_search_tasks=100,
        )

    @classmethod
    def build_resume(cls, task_uids: tuple[str, ...]) -> EvolutionPlan:
        """Build the 91-task resume plan: G0 observe keeps only 16 tasks after
        9 irrecoverably lost evidence tasks, G1-G3 keep 25 tasks each."""

        if len(task_uids) != 91:
            raise ControllerContractError(
                "resume evolution plan requires exactly 91 tasks"
            )
        return cls._build_generations(
            task_uids,
            generation_sizes=(16, 25, 25, 25),
            unique_search_tasks=91,
        )

    @classmethod
    def _build_generations(
        cls,
        task_uids: tuple[str, ...],
        *,
        generation_sizes: tuple[int, int, int, int],
        unique_search_tasks: int,
    ) -> EvolutionPlan:
        if len(set(task_uids)) != len(task_uids):
            raise ControllerContractError("evolution plan tasks must be unique")
        if not all(isinstance(task, str) and task.strip() for task in task_uids):
            raise ControllerContractError(
                "evolution plan task IDs must be non-empty strings"
            )
        generations = []
        offset = 0
        for generation, size in enumerate(generation_sizes):
            generation_tasks = task_uids[offset : offset + size]
            offset += size
            if generation == 0:
                stages = (
                    StagePlan(
                        name="observe",
                        task_uids=generation_tasks,
                        candidate_count=0,
                        expected_agent_calls=2 * size,
                    ),
                )
            else:
                stages = (
                    StagePlan(
                        name="scout",
                        task_uids=generation_tasks[:5],
                        candidate_count=4,
                        expected_agent_calls=30,
                    ),
                    StagePlan(
                        name="semifinal",
                        task_uids=generation_tasks[5:13],
                        candidate_count=2,
                        expected_agent_calls=32,
                    ),
                    StagePlan(
                        name="confirmation",
                        task_uids=generation_tasks[13:],
                        candidate_count=1,
                        expected_agent_calls=36,
                    ),
                )
            generations.append(
                GenerationPlan(
                    generation=generation,
                    task_uids=generation_tasks,
                    stages=stages,
                )
            )
        planned_agent = sum(
            stage.expected_agent_calls
            for generation in generations
            for stage in generation.stages
        )
        return cls(
            schema_version=1,
            generations=tuple(generations),
            unique_search_tasks=unique_search_tasks,
            planned_agent_task_calls=planned_agent,
            maximum_proposer_calls=32,
            maximum_reviewer_calls=8,
            planned_real_codex_calls=planned_agent + 32 + 8,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvolutionPlan:
        tasks = tuple(
            task
            for generation in data["generations"]
            for task in generation["task_uids"]
        )
        if len(tasks) == 100:
            plan = cls.build(tasks)
        elif len(tasks) == 91:
            plan = cls.build_resume(tasks)
        else:
            raise ControllerContractError(
                "persisted evolution plan has unsupported task count"
            )
        if plan.to_dict() != data:
            raise ControllerContractError(
                "persisted evolution plan differs from frozen plan"
            )
        return plan

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generations": [generation.to_dict() for generation in self.generations],
            "unique_search_tasks": self.unique_search_tasks,
            "planned_agent_task_calls": self.planned_agent_task_calls,
            "maximum_proposer_calls": self.maximum_proposer_calls,
            "maximum_reviewer_calls": self.maximum_reviewer_calls,
            "planned_real_codex_calls": self.planned_real_codex_calls,
        }

    def stage(self, generation: int, name: str) -> StagePlan:
        try:
            generation_plan = self.generations[generation]
        except IndexError as error:
            raise ControllerContractError("unknown evolution generation") from error
        for stage in generation_plan.stages:
            if stage.name == name:
                return stage
        raise ControllerContractError(f"unknown generation stage: {generation}/{name}")

    @property
    def stage_sequence(self) -> tuple[tuple[int, str], ...]:
        return tuple(
            (generation.generation, stage.name)
            for generation in self.generations
            for stage in generation.stages
        )


@dataclass(frozen=True)
class EvolutionAuthorization:
    maximum_unique_search_tasks: int
    maximum_real_codex_calls: int
    maximum_temporary_cloud_instances: int
    maximum_elapsed_hours: float
    maximum_cloud_cost_cny: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvolutionAuthorization:
        return cls(**data)

    def validate(self, plan: EvolutionPlan) -> None:
        if plan.unique_search_tasks > self.maximum_unique_search_tasks:
            raise BudgetContractError(
                f"plan exceeds {self.maximum_unique_search_tasks} search-task cap"
            )
        if plan.planned_real_codex_calls > self.maximum_real_codex_calls:
            raise BudgetContractError(
                f"plan exceeds {self.maximum_real_codex_calls} real Codex-call cap"
            )
        if self.maximum_temporary_cloud_instances != 1:
            raise BudgetContractError(
                "authorization must allow exactly one cloud instance"
            )
        if self.maximum_elapsed_hours != 24.0:
            raise BudgetContractError("authorization elapsed-time cap must be 24 hours")
        if self.maximum_cloud_cost_cny != 30.0:
            raise BudgetContractError("authorization cloud cap must be CNY 30")


@dataclass(frozen=True)
class StageClaim:
    generation: int
    stage: str
    task_uids: tuple[str, ...]
    arm_sha256s: tuple[str, ...]
    matched_arm_count: int


class EvolutionController:
    """Own task lifecycle, call accounting, and experimental lineage state."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.state_path = self.root / "STATE.json"
        self.state_digest_path = self.root / "STATE.sha256"
        self._lock = threading.RLock()
        if not self.state_path.is_file() or not self.state_digest_path.is_file():
            raise ControllerContractError(
                f"controller state does not exist: {self.root}"
            )
        self._load_verified()

    @classmethod
    def initialize(
        cls,
        root: Path,
        *,
        plan: EvolutionPlan,
        authorization: EvolutionAuthorization,
        original_agent_program_sha256: str,
        seed_parent_agent_program_sha256: str,
        native_evaluator_epoch: str,
    ) -> EvolutionController:
        authorization.validate(plan)
        _validate_sha(original_agent_program_sha256, name="original AgentProgram")
        _validate_sha(seed_parent_agent_program_sha256, name="seed parent AgentProgram")
        if original_agent_program_sha256 == seed_parent_agent_program_sha256:
            raise ControllerContractError("original and seed parent must be distinct")
        if not native_evaluator_epoch.strip():
            raise ControllerContractError("native evaluator epoch cannot be empty")
        root = root.resolve()
        state = {
            "schema_version": 1,
            "plan": plan.to_dict(),
            "authorization": asdict(authorization),
            "original_agent_program_sha256": original_agent_program_sha256,
            "seed_parent_agent_program_sha256": seed_parent_agent_program_sha256,
            "search_parent_sha256": seed_parent_agent_program_sha256,
            "native_evaluator_epoch": native_evaluator_epoch,
            "task_states": {
                task: {"state": "unopened", "claim": None}
                for generation in plan.generations
                for task in generation.task_uids
            },
            "claims": {},
            "arm_evidence": {},
            "call_reservations": {},
            "auxiliary_call_reservations": {},
            "usage": {
                "real_codex_calls": 0,
                "agent_task_calls": 0,
                "auxiliary_calls": 0,
                "cloud_instance_ids": [],
                "elapsed_hours": 0.0,
                "cloud_cost_cny": 0.0,
            },
            "parent_history": [],
            "final_sealed_opened": False,
            "production_active_ref": None,
            "external_actions": 0,
        }
        state_path = root / "STATE.json"
        if state_path.exists():
            controller = cls(root)
            immutable_fields = (
                "schema_version",
                "plan",
                "authorization",
                "original_agent_program_sha256",
                "seed_parent_agent_program_sha256",
                "native_evaluator_epoch",
            )
            if any(
                controller._state.get(field) != state[field]
                for field in immutable_fields
            ):
                raise ControllerContractError(
                    "controller already exists with a different frozen contract"
                )
            return controller
        root.mkdir(parents=True, exist_ok=True)
        controller = object.__new__(cls)
        controller.root = root
        controller.state_path = state_path
        controller.state_digest_path = root / "STATE.sha256"
        controller._lock = threading.RLock()
        controller._state = state
        controller._persist()
        return cls(root)

    def _load_verified(self) -> dict[str, Any]:
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        expected = self.state_digest_path.read_text(encoding="utf-8").strip()
        actual = _digest(state)
        if expected != actual:
            raise ControllerContractError("controller state hash mismatch")
        self._state = state
        EvolutionPlan.from_dict(state["plan"])
        return state

    def _persist(self) -> None:
        _atomic_json(self.state_path, self._state)
        self.state_digest_path.write_text(_digest(self._state) + "\n", encoding="utf-8")

    @property
    def plan(self) -> EvolutionPlan:
        return EvolutionPlan.from_dict(self._state["plan"])

    @property
    def authorization(self) -> EvolutionAuthorization:
        return EvolutionAuthorization.from_dict(self._state["authorization"])

    @staticmethod
    def _claim_key(generation: int, stage: str) -> str:
        return f"g{generation}:{stage}"

    def _assert_next_stage(self, generation: int, stage: str) -> None:
        target = (generation, stage)
        for item in self.plan.stage_sequence:
            claim = self._state["claims"].get(self._claim_key(*item))
            if item == target:
                return
            if claim is None or claim["status"] != "complete":
                raise ControllerContractError(
                    f"prior stage is not complete before {generation}/{stage}: {item}"
                )
        raise ControllerContractError(f"stage is outside frozen plan: {target}")

    @_locked
    def claim_stage(
        self,
        generation: int,
        stage: str,
        *,
        candidate_sha256s: tuple[str, ...],
    ) -> StageClaim:
        stage_plan = self.plan.stage(generation, stage)
        if len(candidate_sha256s) > stage_plan.candidate_count:
            raise ControllerContractError(
                f"{generation}/{stage} allows at most {stage_plan.candidate_count} candidates"
            )
        if len(set(candidate_sha256s)) != len(candidate_sha256s):
            raise ControllerContractError("stage candidate hashes must be unique")
        for candidate in candidate_sha256s:
            _validate_sha(candidate, name="stage candidate")
        key = self._claim_key(generation, stage)
        existing = self._state["claims"].get(key)
        if existing is not None:
            if (
                existing.get("generation") != generation
                or existing.get("stage") != stage
                or tuple(existing.get("task_uids", ())) != stage_plan.task_uids
                or tuple(existing.get("arm_sha256s", ()))[2:] != candidate_sha256s
                or existing.get("matched_arm_count")
                != len(existing.get("task_uids", ()))
                * len(existing.get("arm_sha256s", ()))
                or existing.get("status") not in {"claimed", "complete"}
            ):
                raise ControllerContractError("existing stage claim is immutable")
            return StageClaim(
                generation=generation,
                stage=stage,
                task_uids=tuple(existing["task_uids"]),
                arm_sha256s=tuple(existing["arm_sha256s"]),
                matched_arm_count=existing["matched_arm_count"],
            )
        arms = (
            self._state["original_agent_program_sha256"],
            self._state["search_parent_sha256"],
            *candidate_sha256s,
        )
        if len(set(arms)) != len(arms):
            raise ControllerContractError("stage arms must be distinct")
        expected_calls = len(stage_plan.task_uids) * len(arms)
        # Graceful degradation: a generation may run with fewer candidates
        # than the frozen maximum (0..stage_plan.candidate_count). The frozen
        # expected_agent_calls is treated as the full-budget ceiling; the
        # actual claim budget is derived from the number of arms executed.
        if expected_calls > stage_plan.expected_agent_calls:
            raise ControllerContractError("stage arm budget exceeds frozen plan")
        claim_payload = {
            "generation": generation,
            "stage": stage,
            "task_uids": list(stage_plan.task_uids),
            "arm_sha256s": list(arms),
            "matched_arm_count": expected_calls,
            "status": "claimed",
        }
        self._assert_next_stage(generation, stage)
        for task_uid in stage_plan.task_uids:
            task = self._state["task_states"][task_uid]
            if task["state"] != "unopened":
                raise ControllerContractError("search task cannot be reused")
            task["state"] = "claimed"
            task["claim"] = key
        self._state["claims"][key] = claim_payload
        self._persist()
        return StageClaim(
            generation=generation,
            stage=stage,
            task_uids=stage_plan.task_uids,
            arm_sha256s=tuple(arms),
            matched_arm_count=expected_calls,
        )

    def task_state(self, task_uid: str) -> str:
        try:
            return str(self._state["task_states"][task_uid]["state"])
        except KeyError as error:
            raise ControllerContractError(
                "task is not in frozen search plan"
            ) from error

    def task_materialization_allowed(self, task_uid: str) -> bool:
        return self.task_state(task_uid) == "claimed"

    def _check_call_budget(self, increment: int) -> None:
        current = self._state["usage"]["real_codex_calls"]
        cap = self.authorization.maximum_real_codex_calls
        if current + increment > cap:
            raise BudgetContractError(f"real Codex calls would exceed {cap} hard cap")

    @_locked
    def reserve_arm_call(
        self,
        *,
        generation: int,
        stage: str,
        task_uid: str,
        arm_sha256: str,
    ) -> dict[str, Any]:
        """Charge one real call before dispatch; a crash cannot silently retry it."""

        _validate_sha(arm_sha256, name="arm")
        claim_key = self._claim_key(generation, stage)
        claim = self._state["claims"].get(claim_key)
        if claim is None or claim["status"] != "claimed":
            raise ControllerContractError(
                "stage must be claimed before call reservation"
            )
        if task_uid not in claim["task_uids"] or arm_sha256 not in claim["arm_sha256s"]:
            raise ControllerContractError(
                "call reservation is outside frozen stage claim"
            )
        reservation_key = f"{claim_key}|{task_uid}|{arm_sha256}"
        reservations = self._state.setdefault("call_reservations", {})
        existing = reservations.get(reservation_key)
        if existing is not None:
            if existing.get("status") in {
                "aborted_pre_dispatch",
                "aborted_without_service_dispatch",
                "aborted_infrastructure_invalid",
            }:
                self._check_call_budget(1)
                existing["status"] = "reserved"
                existing["evidence_sha256"] = None
                existing["reservation_count"] = existing.get("reservation_count", 1) + 1
                self._state["usage"]["real_codex_calls"] += 1
                self._persist()
                return {**existing, "dispatch_allowed": True}
            return {
                "reservation_count": 1,
                "pre_dispatch_aborts": [],
                **existing,
                "dispatch_allowed": False,
            }
        self._check_call_budget(1)
        value = {
            "generation": generation,
            "stage": stage,
            "task_uid": task_uid,
            "arm_sha256": arm_sha256,
            "status": "reserved",
            "evidence_sha256": None,
            "reservation_count": 1,
            "pre_dispatch_aborts": [],
        }
        reservations[reservation_key] = value
        self._state["usage"]["real_codex_calls"] += 1
        self._persist()
        return {**value, "dispatch_allowed": True}

    @_locked
    def abort_arm_call_pre_dispatch(
        self,
        *,
        generation: int,
        stage: str,
        task_uid: str,
        arm_sha256: str,
        reason_code: str,
        evidence_sha256: str,
    ) -> dict[str, Any]:
        """Audit a proven no-dispatch failure without refunding reserved budget."""

        _validate_sha(arm_sha256, name="arm")
        _validate_sha(evidence_sha256, name="pre-dispatch evidence")
        if re.fullmatch(r"[a-z][a-z0-9-]{2,63}", reason_code) is None:
            raise ControllerContractError("pre-dispatch reason code is invalid")
        key = f"{self._claim_key(generation, stage)}|{task_uid}|{arm_sha256}"
        reservation = self._state.setdefault("call_reservations", {}).get(key)
        if reservation is None:
            raise ControllerContractError("arm call was not reserved")
        abort = {
            "evidence_sha256": evidence_sha256,
            "reason_code": reason_code,
            "reservation_count": reservation.get("reservation_count", 1),
        }
        if reservation.get("status") == "aborted_pre_dispatch":
            if reservation.get("pre_dispatch_aborts", [])[-1:] != [abort]:
                raise ControllerContractError(
                    "pre-dispatch abort evidence is immutable"
                )
            return dict(reservation)
        if reservation.get("status") != "reserved":
            raise ControllerContractError(
                "only a reserved arm call can abort before dispatch"
            )
        if reservation.get("evidence_sha256") is not None:
            raise ControllerContractError(
                "completed evidence cannot be marked as pre-dispatch"
            )
        reservation.setdefault("pre_dispatch_aborts", []).append(abort)
        reservation.setdefault("reservation_count", 1)
        reservation["status"] = "aborted_pre_dispatch"
        self._persist()
        return dict(reservation)

    @_locked
    def invalidate_arm_call_infrastructure(
        self,
        *,
        generation: int,
        stage: str,
        task_uid: str,
        arm_sha256: str,
        reason_code: str,
        evidence_sha256: str,
    ) -> dict[str, Any]:
        """Exclude a dispatched call whose execution substrate was invalid."""

        _validate_sha(arm_sha256, name="arm")
        _validate_sha(evidence_sha256, name="infrastructure evidence")
        if re.fullmatch(r"[a-z][a-z0-9-]{2,63}", reason_code) is None:
            raise ControllerContractError("infrastructure reason code is invalid")
        key = f"{self._claim_key(generation, stage)}|{task_uid}|{arm_sha256}"
        reservation = self._state.setdefault("call_reservations", {}).get(key)
        if reservation is None:
            raise ControllerContractError("arm call was not reserved")
        abort = {
            "evidence_sha256": evidence_sha256,
            "reason_code": reason_code,
            "reservation_count": reservation.get("reservation_count", 1),
        }
        if reservation.get("status") == "aborted_infrastructure_invalid":
            if reservation.get("infrastructure_aborts", [])[-1:] != [abort]:
                raise ControllerContractError(
                    "infrastructure abort evidence is immutable"
                )
            return dict(reservation)
        if reservation.get("status") != "reserved":
            raise ControllerContractError(
                "only a reserved arm call can be infrastructure-invalidated"
            )
        if reservation.get("evidence_sha256") is not None:
            raise ControllerContractError(
                "completed evidence cannot be infrastructure-invalidated"
            )
        reservation.setdefault("infrastructure_aborts", []).append(abort)
        reservation.setdefault("reservation_count", 1)
        reservation["status"] = "aborted_infrastructure_invalid"
        self._persist()
        return dict(reservation)

    @_locked
    def abort_arm_call_without_service_dispatch(
        self,
        *,
        generation: int,
        stage: str,
        task_uid: str,
        arm_sha256: str,
        reason_code: str,
        evidence_sha256: str,
    ) -> dict[str, Any]:
        """Audit a launched process proven never to have reached the LLM service."""

        _validate_sha(arm_sha256, name="arm")
        _validate_sha(evidence_sha256, name="no-service evidence")
        if re.fullmatch(r"[a-z][a-z0-9-]{2,63}", reason_code) is None:
            raise ControllerContractError("no-service reason code is invalid")
        key = f"{self._claim_key(generation, stage)}|{task_uid}|{arm_sha256}"
        reservation = self._state.setdefault("call_reservations", {}).get(key)
        if reservation is None:
            raise ControllerContractError("arm call was not reserved")
        abort = {
            "evidence_sha256": evidence_sha256,
            "reason_code": reason_code,
            "reservation_count": reservation.get("reservation_count", 1),
        }
        if reservation.get("status") == "aborted_without_service_dispatch":
            if reservation.get("non_dispatch_aborts", [])[-1:] != [abort]:
                raise ControllerContractError("no-service abort evidence is immutable")
            return dict(reservation)
        if reservation.get("status") != "reserved":
            raise ControllerContractError(
                "only a reserved arm call can abort without service dispatch"
            )
        if reservation.get("evidence_sha256") is not None:
            raise ControllerContractError(
                "completed evidence cannot be marked as no-service dispatch"
            )
        reservation.setdefault("non_dispatch_aborts", []).append(abort)
        reservation.setdefault("reservation_count", 1)
        reservation["status"] = "aborted_without_service_dispatch"
        self._persist()
        return dict(reservation)

    @_locked
    def arm_call_status(
        self,
        *,
        generation: int,
        stage: str,
        task_uid: str,
        arm_sha256: str,
    ) -> dict[str, Any] | None:
        key = f"{self._claim_key(generation, stage)}|{task_uid}|{arm_sha256}"
        value = self._state.get("call_reservations", {}).get(key)
        return None if value is None else dict(value)

    @_locked
    def arm_evidence_status(
        self,
        *,
        generation: int,
        stage: str,
        task_uid: str,
        arm_sha256: str,
    ) -> dict[str, Any] | None:
        key = f"{self._claim_key(generation, stage)}|{task_uid}|{arm_sha256}"
        value = self._state.get("arm_evidence", {}).get(key)
        return None if value is None else dict(value)

    @_locked
    def record_arm_evidence(
        self,
        *,
        generation: int,
        stage: str,
        task_uid: str,
        arm_sha256: str,
        evidence_sha256: str,
        real_codex_calls: int = 1,
    ) -> dict[str, Any]:
        _validate_sha(arm_sha256, name="arm")
        _validate_sha(evidence_sha256, name="native evidence")
        if not isinstance(real_codex_calls, int) or real_codex_calls not in {0, 1}:
            raise BudgetContractError("one arm may consume zero or one real Codex call")
        key = self._claim_key(generation, stage)
        claim = self._state["claims"].get(key)
        if claim is None or claim["status"] != "claimed":
            raise ControllerContractError(
                "stage must be claimed before recording evidence"
            )
        if task_uid not in claim["task_uids"] or arm_sha256 not in claim["arm_sha256s"]:
            raise ControllerContractError("arm evidence is outside frozen stage claim")
        evidence_key = f"{key}|{task_uid}|{arm_sha256}"
        existing = self._state["arm_evidence"].get(evidence_key)
        value = {
            "generation": generation,
            "stage": stage,
            "task_uid": task_uid,
            "arm_sha256": arm_sha256,
            "evidence_sha256": evidence_sha256,
            "native_evaluator_epoch": self._state["native_evaluator_epoch"],
            "real_codex_calls": real_codex_calls,
        }
        if existing is not None:
            if existing != value:
                raise ControllerContractError("frozen native evidence is immutable")
            return existing
        reservations = self._state.setdefault("call_reservations", {})
        reservation = reservations.get(evidence_key)
        if reservation is not None:
            if real_codex_calls != 1:
                raise BudgetContractError(
                    "reserved real arm evidence must consume exactly one call"
                )
            if reservation["status"] not in {"reserved", "completed"}:
                raise ControllerContractError("real arm call reservation is invalid")
            if (
                reservation["status"] == "completed"
                and reservation["evidence_sha256"] != evidence_sha256
            ):
                raise ControllerContractError("completed call evidence is immutable")
            reservation["status"] = "completed"
            reservation["evidence_sha256"] = evidence_sha256
            charge = 0
        else:
            charge = real_codex_calls
            self._check_call_budget(charge)
        self._state["arm_evidence"][evidence_key] = value
        self._state["usage"]["real_codex_calls"] += charge
        self._state["usage"]["agent_task_calls"] += 1
        self._persist()
        return value

    def quarantine_completed_arm_evidence(
        self,
        *,
        generation: int,
        stage: str,
        task_uid: str,
        arm_sha256: str,
        evidence_sha256: str,
        reason_code: str,
        incident_sha256: str,
    ) -> dict[str, Any]:
        """Tombstone infrastructure-invalid evidence without refunding its call.

        The frozen Agent receipt remains the recovery authority. Only the invalid
        native admission is removed from the active stage so the evaluator can be
        rerun without another model dispatch.
        """

        _validate_sha(arm_sha256, name="arm")
        _validate_sha(evidence_sha256, name="quarantined evidence")
        _validate_sha(incident_sha256, name="quarantine incident")
        if re.fullmatch(r"[a-z][a-z0-9-]{2,63}", reason_code) is None:
            raise ControllerContractError("quarantine reason code is invalid")
        claim_key = self._claim_key(generation, stage)
        claim = self._state.get("claims", {}).get(claim_key)
        if claim is None or claim.get("status") != "claimed":
            raise ControllerContractError(
                "only an incomplete claimed stage may quarantine evidence"
            )
        evidence_key = f"{claim_key}|{task_uid}|{arm_sha256}"
        entry = {
            "generation": generation,
            "stage": stage,
            "task_uid": task_uid,
            "arm_sha256": arm_sha256,
            "evidence_sha256": evidence_sha256,
            "reason_code": reason_code,
            "incident_sha256": incident_sha256,
        }
        quarantine = self._state.setdefault("quarantined_arm_evidence", [])
        if entry in quarantine:
            return dict(entry)
        active = self._state.get("arm_evidence", {}).get(evidence_key)
        reservation = self._state.get("call_reservations", {}).get(evidence_key)
        if (
            active is None
            or active.get("evidence_sha256") != evidence_sha256
            or reservation is None
            or reservation.get("status") != "completed"
            or reservation.get("evidence_sha256") != evidence_sha256
        ):
            raise ControllerContractError(
                "only matching completed arm evidence may be quarantined"
            )
        del self._state["arm_evidence"][evidence_key]
        reservation["status"] = "reserved"
        reservation["evidence_sha256"] = None
        reservation.setdefault("completed_evidence_quarantines", []).append(entry)
        quarantine.append(entry)
        usage = self._state["usage"]
        if usage["agent_task_calls"] < 1:
            raise ControllerContractError("agent task usage cannot be quarantined")
        usage["agent_task_calls"] -= 1
        self._persist()
        return dict(entry)

    @_locked
    def complete_stage(self, generation: int, stage: str) -> dict[str, Any]:
        key = self._claim_key(generation, stage)
        claim = self._state["claims"].get(key)
        if claim is None:
            raise ControllerContractError("stage has not been claimed")
        if claim["status"] == "complete":
            return claim
        missing = [
            (task_uid, arm)
            for task_uid in claim["task_uids"]
            for arm in claim["arm_sha256s"]
            if f"{key}|{task_uid}|{arm}" not in self._state["arm_evidence"]
        ]
        if missing:
            raise ControllerContractError(
                f"stage lacks frozen native evidence for {len(missing)} matched arms"
            )
        claim["status"] = "complete"
        for task_uid in claim["task_uids"]:
            task = self._state["task_states"][task_uid]
            if task != {"state": "claimed", "claim": key}:
                raise ControllerContractError(
                    "task lifecycle changed before retirement"
                )
            task["state"] = "retired"
        self._persist()
        return claim

    @_locked
    def record_auxiliary_calls(
        self, count: int, *, real_codex_calls: int | None = None
    ) -> None:
        if not isinstance(count, int) or count < 0:
            raise BudgetContractError("auxiliary call count must be non-negative")
        consumed = count if real_codex_calls is None else real_codex_calls
        if not isinstance(consumed, int) or not 0 <= consumed <= count:
            raise BudgetContractError(
                "real auxiliary Codex calls must be between zero and executions"
            )
        self._check_call_budget(consumed)
        self._state["usage"]["real_codex_calls"] += consumed
        self._state["usage"]["auxiliary_calls"] += count
        self._persist()

    @_locked
    def reserve_auxiliary_call(
        self, *, reservation_id: str, kind: str
    ) -> dict[str, Any]:
        """Charge a proposer/reviewer call before dispatch and make retry explicit."""

        if (
            not isinstance(reservation_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9|:._-]{0,255}", reservation_id) is None
        ):
            raise ControllerContractError("invalid auxiliary reservation ID")
        if kind not in {"mutation_proposer", "reviewer"}:
            raise ControllerContractError("invalid auxiliary call kind")
        reservations = self._state.setdefault("auxiliary_call_reservations", {})
        existing = reservations.get(reservation_id)
        if existing is not None:
            if existing["kind"] != kind:
                raise ControllerContractError(
                    "auxiliary call reservation kind is immutable"
                )
            if existing["status"] == "aborted_without_service_dispatch":
                self._check_call_budget(1)
                existing["status"] = "reserved"
                existing["evidence_sha256"] = None
                existing.setdefault("aborts", [])
                existing["reservation_count"] = existing.get("reservation_count", 1) + 1
                self._state["usage"]["real_codex_calls"] += 1
                self._state["usage"]["auxiliary_calls"] += 1
                self._persist()
                return {**existing, "dispatch_allowed": True}
            return {**existing, "dispatch_allowed": False}
        self._check_call_budget(1)
        value = {
            "reservation_id": reservation_id,
            "kind": kind,
            "status": "reserved",
            "evidence_sha256": None,
            "aborts": [],
            "reservation_count": 1,
        }
        reservations[reservation_id] = value
        self._state["usage"]["real_codex_calls"] += 1
        self._state["usage"]["auxiliary_calls"] += 1
        self._persist()
        return {**value, "dispatch_allowed": True}

    @_locked
    def abort_auxiliary_call(
        self, *, reservation_id: str, reason_code: str, evidence_sha256: str
    ) -> dict[str, Any]:
        """Audit an auxiliary call proven never to have reached the model."""

        _validate_sha(evidence_sha256, name="auxiliary abort evidence")
        if re.fullmatch(r"[a-z][a-z0-9-]{2,63}", reason_code) is None:
            raise ControllerContractError("auxiliary abort reason code is invalid")
        reservations = self._state.setdefault("auxiliary_call_reservations", {})
        reservation = reservations.get(reservation_id)
        if reservation is None:
            raise ControllerContractError("auxiliary call was not reserved")
        abort = {
            "evidence_sha256": evidence_sha256,
            "reason_code": reason_code,
            "reservation_count": reservation.get("reservation_count", 1),
        }
        if reservation["status"] == "aborted_without_service_dispatch":
            if reservation.get("aborts", [])[-1:] != [abort]:
                raise ControllerContractError("auxiliary abort evidence is immutable")
            return dict(reservation)
        if reservation["status"] != "reserved":
            raise ControllerContractError(
                "only a reserved auxiliary call can abort without dispatch"
            )
        if reservation.get("evidence_sha256") is not None:
            raise ControllerContractError(
                "completed auxiliary evidence cannot be aborted"
            )
        reservation.setdefault("aborts", []).append(abort)
        reservation["status"] = "aborted_without_service_dispatch"
        self._persist()
        return dict(reservation)

    @_locked
    def reset_auxiliary_call(
        self, *, reservation_id: str, reason_code: str, evidence_sha256: str
    ) -> dict[str, Any]:
        """Explicitly reconcile an auxiliary call whose evidence was invalid.

        The completed (or reserved) reservation is moved to
        aborted_without_service_dispatch with the invalid evidence preserved in
        the abort record so a later re-dispatch is explicit and audited.
        """

        _validate_sha(evidence_sha256, name="auxiliary reconciliation evidence")
        if re.fullmatch(r"[a-z][a-z0-9-]{2,63}", reason_code) is None:
            raise ControllerContractError("auxiliary reason code is invalid")
        reservations = self._state.setdefault("auxiliary_call_reservations", {})
        reservation = reservations.get(reservation_id)
        if reservation is None:
            raise ControllerContractError("auxiliary call was not reserved")
        if reservation["status"] == "aborted_without_service_dispatch":
            return dict(reservation)
        if reservation["status"] not in {"reserved", "completed"}:
            raise ControllerContractError(
                "only reserved or completed auxiliary calls can be reconciled"
            )
        abort = {
            "evidence_sha256": evidence_sha256,
            "reason_code": reason_code,
            "previous_evidence_sha256": reservation.get("evidence_sha256"),
            "reservation_count": reservation.get("reservation_count", 1),
        }
        reservation.setdefault("aborts", []).append(abort)
        reservation["status"] = "aborted_without_service_dispatch"
        self._persist()
        return dict(reservation)

    @_locked
    def complete_auxiliary_call(
        self, *, reservation_id: str, evidence_sha256: str
    ) -> dict[str, Any]:
        _validate_sha(evidence_sha256, name="auxiliary evidence")
        reservation = self._state.setdefault("auxiliary_call_reservations", {}).get(
            reservation_id
        )
        if reservation is None:
            raise ControllerContractError("auxiliary call was not reserved")
        if reservation["status"] == "completed":
            if reservation["evidence_sha256"] != evidence_sha256:
                raise ControllerContractError("auxiliary call evidence is immutable")
            return dict(reservation)
        if reservation["status"] != "reserved":
            raise ControllerContractError("auxiliary call reservation is invalid")
        reservation["status"] = "completed"
        reservation["evidence_sha256"] = evidence_sha256
        self._persist()
        return dict(reservation)

    def register_cloud_usage(
        self,
        *,
        instance_id: str,
        elapsed_hours: float,
        cloud_cost_cny: float,
    ) -> None:
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise BudgetContractError("cloud instance ID cannot be empty")
        usage = self._state["usage"]
        instance_ids = set(usage["cloud_instance_ids"])
        if (
            instance_id not in instance_ids
            and len(instance_ids)
            >= self.authorization.maximum_temporary_cloud_instances
        ):
            raise BudgetContractError(
                "cloud instance count would exceed one-instance cap"
            )
        if elapsed_hours > self.authorization.maximum_elapsed_hours:
            raise BudgetContractError("elapsed time would exceed 24-hour hard cap")
        if cloud_cost_cny > self.authorization.maximum_cloud_cost_cny:
            raise BudgetContractError("cloud cost would exceed CNY 30 hard cap")
        if (
            elapsed_hours < usage["elapsed_hours"]
            or cloud_cost_cny < usage["cloud_cost_cny"]
        ):
            raise BudgetContractError("cloud usage counters must be monotonic")
        if instance_id not in instance_ids:
            usage["cloud_instance_ids"].append(instance_id)
            usage["cloud_instance_ids"].sort()
            self._state["external_actions"] += 1
        usage["elapsed_hours"] = float(elapsed_hours)
        usage["cloud_cost_cny"] = float(cloud_cost_cny)
        self._persist()

    @_locked
    def record_parent_decision(
        self,
        *,
        generation: int,
        previous_parent_sha256: str,
        search_parent_sha256: str,
        decision_sha256: str,
        advance: bool,
    ) -> None:
        _validate_sha(previous_parent_sha256, name="previous parent")
        _validate_sha(search_parent_sha256, name="search parent")
        _validate_sha(decision_sha256, name="parent decision")
        value = {
            "generation": generation,
            "previous_parent_sha256": previous_parent_sha256,
            "search_parent_sha256": search_parent_sha256,
            "decision_sha256": decision_sha256,
            "advance": advance,
            "scope": "experimental_search_lineage_only",
            "production_promoted": False,
        }
        existing = next(
            (
                item
                for item in self._state["parent_history"]
                if item["generation"] == generation
            ),
            None,
        )
        if existing is not None:
            if existing != value:
                raise ControllerContractError("parent decision is immutable")
            return
        if previous_parent_sha256 != self._state["search_parent_sha256"]:
            raise ControllerContractError(
                "parent decision starts from stale search parent"
            )
        if not advance and search_parent_sha256 != previous_parent_sha256:
            raise ControllerContractError(
                "retained parent decision cannot change parent"
            )
        self._state["parent_history"].append(value)
        self._state["search_parent_sha256"] = search_parent_sha256
        self._persist()

    def open_final_sealed(self) -> None:
        raise ControllerContractError(
            "final sealed open is prohibited by this authorization"
        )

    def promote_production(self, _candidate_sha256: str) -> None:
        raise ControllerContractError("production/global promotion is prohibited")

    def inspect(self) -> dict[str, Any]:
        task_counts = {state: 0 for state in ("unopened", "claimed", "retired")}
        for value in self._state["task_states"].values():
            task_counts[value["state"]] += 1
        return {
            "schema_version": self._state["schema_version"],
            "search_parent_sha256": self._state["search_parent_sha256"],
            "original_agent_program_sha256": self._state[
                "original_agent_program_sha256"
            ],
            "native_evaluator_epoch": self._state["native_evaluator_epoch"],
            "task_counts": task_counts,
            "claims": self._state["claims"],
            "usage": self._state["usage"],
            "parent_history": self._state["parent_history"],
            "call_reservations": self._state.get("call_reservations", {}),
            "auxiliary_call_reservations": self._state.get(
                "auxiliary_call_reservations", {}
            ),
            "final_sealed_opened": self._state["final_sealed_opened"],
            "production_active_ref": self._state["production_active_ref"],
            "external_actions": self._state["external_actions"],
        }

    def verify(self) -> dict[str, Any]:
        errors = []
        try:
            self._load_verified()
        except (OSError, ValueError) as error:
            return {"valid": False, "errors": [str(error)], "external_actions": 0}
        plan = self.plan
        authorization = self.authorization
        try:
            authorization.validate(plan)
        except ValueError as error:
            errors.append(str(error))
        if len(self._state["task_states"]) != plan.unique_search_tasks:
            errors.append(
                "controller must retain exactly "
                f"{plan.unique_search_tasks} search tasks"
            )
        if self._state["final_sealed_opened"] is not False:
            errors.append("final sealed must remain unopened")
        if self._state["production_active_ref"] is not None:
            errors.append("production active ref must remain unset")
        usage = self._state["usage"]
        if usage["real_codex_calls"] > authorization.maximum_real_codex_calls:
            errors.append("real Codex call budget exceeded")
        if (
            len(usage["cloud_instance_ids"])
            > authorization.maximum_temporary_cloud_instances
        ):
            errors.append("cloud instance budget exceeded")
        if usage["elapsed_hours"] > authorization.maximum_elapsed_hours:
            errors.append("elapsed-time budget exceeded")
        if usage["cloud_cost_cny"] > authorization.maximum_cloud_cost_cny:
            errors.append("cloud cost budget exceeded")
        return {
            "valid": not errors,
            "errors": errors,
            "external_actions": 0,
            "state_sha256": _digest(self._state),
            "task_counts": self.inspect()["task_counts"],
            "usage": usage,
            "final_sealed_opened": False,
            "production_active_ref": None,
        }
