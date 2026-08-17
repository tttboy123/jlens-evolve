"""Recoverable coordinator for continuous matched A/B task rounds."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmark_adapters import BenchmarkContractError, TaskPool
from continuous_ab import (
    ABContractError,
    ArmResult,
    BaselineContract,
    ChangeSetCadence,
    ChangeSetRegistry,
    ContinuousSchedule,
    FinalSealedAuditor,
    MatchedRoundLedger,
    PermanentBaselineAuthority,
    PromotionGate,
)
from native_result_adapter import NormalizedAdmission

ROUND_ID = re.compile(r"round-[0-9]{6}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _service_integrity(payload: dict[str, Any]) -> str:
    unsigned = {
        key: value for key, value in payload.items() if key != "integrity_sha256"
    }
    encoded = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_service(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    sealed = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    sealed["integrity_sha256"] = _service_integrity(sealed)
    _atomic_json(path, sealed)
    return sealed


def _verify_service(payload: dict[str, Any]) -> None:
    if payload.get("integrity_sha256") != _service_integrity(payload):
        raise ABContractError("service manifest was tampered")


class ContinuousABService:
    """Single local state authority for task claims and paired result ledgers."""

    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
        self.root = root.resolve()
        self.manifest = manifest
        self.pool_path = self.root / "TASK_POOL.json"
        self.rounds_dir = self.root / "rounds"
        self.baseline_path = self.root / "permanent-baseline.json"
        self.changesets_path = self.root / "changesets.json"
        self.final_sealed_path = self.root / "FINAL_SEALED_AUDIT.json"

    @classmethod
    def initialize(
        cls,
        root: Path,
        *,
        frozen_pool_path: Path,
        baseline: BaselineContract,
        initial_active_agent_sha256: str,
    ) -> ContinuousABService:
        root = root.resolve()
        manifest_path = root / "SERVICE.json"
        if manifest_path.exists():
            service = cls.load(root)
            PermanentBaselineAuthority(service.baseline_path).freeze(baseline)
            if (
                ChangeSetRegistry.load(service.changesets_path).payload[
                    "initial_agent_sha256"
                ]
                != initial_active_agent_sha256
            ):
                raise ABContractError("initial active agent cannot be replaced")
            return service
        pool = TaskPool.load(frozen_pool_path)
        if any(record.state != "unopened" for record in pool.records):
            raise BenchmarkContractError(
                "runtime must start from an unopened task pool"
            )
        root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(frozen_pool_path.resolve(), root / "TASK_POOL.json")
        PermanentBaselineAuthority(root / "permanent-baseline.json").freeze(baseline)
        ChangeSetRegistry.create(
            root / "changesets.json",
            initial_agent_sha256=initial_active_agent_sha256,
            cadence=ChangeSetCadence(interval=10),
        )
        manifest = {
            "schema_version": "1.0",
            "status": "ready",
            "frozen_pool_source": str(frozen_pool_path.resolve()),
            "baseline_contract_sha256": baseline.contract_sha256,
            "rounds_directory": "rounds",
            "final_sealed_opened": False,
            "search_cycle_candidates": {},
        }
        manifest = _write_service(manifest_path, manifest)
        return cls(root, manifest)

    @classmethod
    def load(cls, root: Path) -> ContinuousABService:
        root = root.resolve()
        manifest = json.loads((root / "SERVICE.json").read_text(encoding="utf-8"))
        _verify_service(manifest)
        if manifest.get("schema_version") != "1.0":
            raise ABContractError("unsupported continuous A/B service schema")
        service = cls(root, manifest)
        TaskPool.load(service.pool_path)
        baseline = PermanentBaselineAuthority(service.baseline_path).load()
        if baseline["baseline_contract_sha256"] != manifest["baseline_contract_sha256"]:
            raise ABContractError("service baseline authority mismatch")
        ChangeSetRegistry.load(service.changesets_path)
        return service

    @property
    def completed_round_count(self) -> int:
        if not self.rounds_dir.exists():
            return 0
        return sum(
            MatchedRoundLedger.load(path).phase == "retired"
            for path in self.rounds_dir.glob("round-*.json")
        )

    def _round_path(self, round_id: str) -> Path:
        if ROUND_ID.fullmatch(round_id) is None:
            raise ABContractError("invalid round id")
        return self.rounds_dir / f"{round_id}.json"

    def _baseline(self) -> BaselineContract:
        authority = PermanentBaselineAuthority(self.baseline_path).load()
        return BaselineContract.from_dict(authority["baseline"])

    def plan_round(
        self,
        *,
        partition: str,
        evolved_agent_sha256: str,
    ) -> dict[str, Any]:
        if partition not in {"search", "promotion", "final_sealed"}:
            raise ABContractError(
                "round partition must be search, promotion, or final_sealed"
            )
        self.reconcile_all_rounds()
        round_contract = self._expected_round_contract()
        expected_partition = str(round_contract["partition"])
        if partition != expected_partition:
            raise ABContractError(
                f"expected partition is {expected_partition}; requested={partition}"
            )
        registry = ChangeSetRegistry.load(self.changesets_path)
        pending = [
            proposal
            for proposal in registry.payload["proposals"]
            if proposal["status"] == "proposed"
        ]
        if partition == "search":
            if pending:
                raise ABContractError(
                    "pending ChangeSet must be decided before the next search cycle"
                )
            cycle = str(round_contract["cycle"])
            frozen = self.manifest["search_cycle_candidates"].get(cycle)
            if frozen is None:
                self.manifest["search_cycle_candidates"][cycle] = evolved_agent_sha256
                self.manifest = _write_service(
                    self.root / "SERVICE.json", self.manifest
                )
            elif frozen != evolved_agent_sha256:
                raise ABContractError("search cycle candidate cannot be replaced")
        elif partition == "promotion":
            if len(pending) != 1:
                raise ABContractError(
                    "promotion rounds require exactly one pending ChangeSet"
                )
            if pending[0]["candidate_agent_sha256"] != evolved_agent_sha256:
                raise ABContractError(
                    "evolved arm is not the pending ChangeSet candidate"
                )
        if partition == "final_sealed":
            if not self.final_sealed_path.exists():
                raise ABContractError("final sealed audit has not been opened")
            audit = json.loads(self.final_sealed_path.read_text(encoding="utf-8"))
            if audit["candidate_sha256"] != evolved_agent_sha256:
                raise ABContractError("evolved arm is not the frozen final candidate")
        pool = TaskPool.load(self.pool_path)
        available = [
            record
            for record in pool.records
            if record.assigned_partition == partition and record.state == "unopened"
        ]
        if not available:
            raise ABContractError(f"no unopened {partition} tasks remain")
        completed_and_open = (
            list(self.rounds_dir.glob("round-*.json"))
            if self.rounds_dir.exists()
            else []
        )
        round_id = f"round-{len(completed_and_open) + 1:06d}"
        task = available[0]
        ledger = MatchedRoundLedger.create(
            self._round_path(round_id),
            round_id=round_id,
            task_uid=task.task_uid,
            baseline=self._baseline(),
            evolved_agent_sha256=evolved_agent_sha256,
        )
        pool.claim(task.task_uid, partition)
        pool.save(self.pool_path)
        return {
            "round_id": round_id,
            "task_uid": task.task_uid,
            "benchmark_id": task.benchmark_id,
            "instance_id": task.instance_id,
            "partition": partition,
            "phase": ledger.phase,
            "task_contract": task.task_contract,
        }

    def _expected_round_contract(self) -> dict[str, Any]:
        pool = TaskPool.load(self.pool_path)
        counts = {
            partition: sum(
                record.assigned_partition == partition for record in pool.records
            )
            for partition in ("search", "promotion", "final_sealed")
        }
        if counts == {"search": 160, "promotion": 80, "final_sealed": 60}:
            schedule = ContinuousSchedule().build()["rounds"]
        else:
            schedule = [
                {
                    "partition": partition,
                    "cycle": 1 if partition != "final_sealed" else "final",
                }
                for partition in ("search", "promotion", "final_sealed")
                for _ in range(counts[partition])
            ]
        round_count = (
            len(list(self.rounds_dir.glob("round-*.json")))
            if self.rounds_dir.exists()
            else 0
        )
        if round_count >= len(schedule):
            raise ABContractError("all predeclared rounds have already been planned")
        return schedule[round_count]

    def propose_changeset(
        self,
        *,
        candidate_agent_sha256: str,
        forward_patch: Path,
        rollback_patch: Path,
    ) -> dict[str, Any]:
        self.reconcile_all_rounds()
        planned = (
            len(list(self.rounds_dir.glob("round-*.json")))
            if self.rounds_dir.exists()
            else 0
        )
        completed = self.completed_round_count
        if planned != completed:
            raise ABContractError("all planned rounds must retire before a proposal")
        if self._expected_round_contract()["partition"] != "promotion":
            raise ABContractError("ChangeSet can only be proposed before promotion")
        registry = ChangeSetRegistry.load(self.changesets_path)
        return registry.propose(
            completed_rounds=completed,
            candidate_agent_sha256=candidate_agent_sha256,
            forward_patch=forward_patch,
            rollback_patch=rollback_patch,
        )

    def decide_changeset(self, proposal_id: str) -> dict[str, Any]:
        self.reconcile_all_rounds()
        registry = ChangeSetRegistry.load(self.changesets_path)
        proposal = next(
            (
                item
                for item in registry.payload["proposals"]
                if item["proposal_id"] == proposal_id
            ),
            None,
        )
        if proposal is None or proposal["status"] != "proposed":
            raise ABContractError("unknown or inactive ChangeSet proposal")
        pool = TaskPool.load(self.pool_path)
        by_uid = {record.task_uid: record for record in pool.records}
        pairs = []
        for path in sorted(self.rounds_dir.glob("round-*.json")):
            ledger = MatchedRoundLedger.load(path)
            round_number = int(str(ledger.payload["round_id"]).split("-")[-1])
            if round_number <= proposal["completed_rounds"]:
                continue
            task = by_uid[str(ledger.payload["task_uid"])]
            if task.assigned_partition != "promotion":
                continue
            if ledger.phase != "retired":
                raise ABContractError("promotion pair is not retired")
            if (
                ledger.payload["evolved_agent_sha256"]
                != proposal["candidate_agent_sha256"]
            ):
                raise ABContractError("promotion pair used a different candidate")
            pairs.append(
                (
                    ArmResult(**ledger.payload["results"]["baseline"]),
                    ArmResult(**ledger.payload["results"]["evolved"]),
                )
            )
        if len(pairs) != 10:
            raise ABContractError("ChangeSet decision requires exactly 10 new pairs")
        decision = PromotionGate().evaluate(pairs)
        if decision.approved:
            outcome = registry.promote(proposal_id, decision)
            status = "promoted"
        else:
            outcome = registry.reject(proposal_id, decision)
            status = "rejected"
        return {"status": status, "decision": asdict(decision), **outcome}

    def open_final_sealed(self, *, candidate_sha256: str) -> dict[str, Any]:
        pool = TaskPool.load(self.pool_path)
        unfinished = [
            record.task_uid
            for record in pool.records
            if record.assigned_partition != "final_sealed" and record.state != "retired"
        ]
        if unfinished:
            raise ABContractError(
                "search and promotion tasks must retire before final sealed opens"
            )
        final_tasks = [
            record.task_uid
            for record in pool.records
            if record.assigned_partition == "final_sealed"
            and record.state == "unopened"
        ]
        used_task_uids = (
            {
                str(MatchedRoundLedger.load(path).payload["task_uid"])
                for path in self.rounds_dir.glob("round-*.json")
            }
            if self.rounds_dir.exists()
            else set()
        )
        audit = FinalSealedAuditor(self.final_sealed_path).open(
            candidate_sha256=candidate_sha256,
            task_uids=final_tasks,
            used_task_uids=used_task_uids,
        )
        self.manifest["final_sealed_opened"] = True
        self.manifest["final_candidate_sha256"] = candidate_sha256
        self.manifest = _write_service(self.root / "SERVICE.json", self.manifest)
        return audit

    def freeze_predictions(
        self,
        round_id: str,
        predictions: dict[str, Path],
    ) -> None:
        ledger = MatchedRoundLedger.load(self._round_path(round_id))
        ledger.freeze_predictions(predictions)

    def record_results(
        self,
        round_id: str,
        results: dict[str, ArmResult],
    ) -> None:
        path = self._round_path(round_id)
        ledger = MatchedRoundLedger.load(path)
        ledger.record_results(results)
        pool = TaskPool.load(self.pool_path)
        pool.retire(str(ledger.payload["task_uid"]))
        pool.save(self.pool_path)
        ledger.retire()

    def record_native_results(
        self,
        round_id: str,
        admissions: dict[str, NormalizedAdmission],
    ) -> None:
        """Admit a complete pair only after normalized native evidence is frozen."""

        if set(admissions) != {"baseline", "evolved"}:
            raise ABContractError("both native arm admissions are required")
        path = self._round_path(round_id)
        ledger = MatchedRoundLedger.load(path)
        results = {arm: admissions[arm].result for arm in admissions}
        ledger.record_native_evidence(
            {arm: admissions[arm].evidence_path for arm in admissions},
            results,
        )
        ledger.record_results(results)
        pool = TaskPool.load(self.pool_path)
        pool.retire(str(ledger.payload["task_uid"]))
        pool.save(self.pool_path)
        ledger.retire()

    def round_state(self, round_id: str) -> dict[str, Any]:
        return MatchedRoundLedger.load(self._round_path(round_id)).payload

    def reconcile_round(self, round_id: str) -> dict[str, Any]:
        """Repair the two safe crash windows between ledger and pool writes."""

        path = self._round_path(round_id)
        ledger = MatchedRoundLedger.load(path)
        pool = TaskPool.load(self.pool_path)
        task_uid = str(ledger.payload["task_uid"])
        task = next(
            (record for record in pool.records if record.task_uid == task_uid),
            None,
        )
        if task is None:
            raise ABContractError("round task is missing from the runtime pool")
        changed = False
        if task.state == "unopened":
            pool.claim(task_uid, task.assigned_partition)
            pool.save(self.pool_path)
            changed = True
        elif task.state not in {task.assigned_partition, "retired"}:
            raise ABContractError("round task and pool state disagree")

        if ledger.phase == "evaluated":
            if task.state != "retired":
                pool = TaskPool.load(self.pool_path)
                pool.retire(task_uid)
                pool.save(self.pool_path)
            ledger.retire()
            changed = True
        elif ledger.phase == "retired" and task.state != "retired":
            pool = TaskPool.load(self.pool_path)
            pool.retire(task_uid)
            pool.save(self.pool_path)
            changed = True
        payload = MatchedRoundLedger.load(path).payload
        return {**payload, "reconciled": changed}

    def reconcile_all_rounds(self) -> list[dict[str, Any]]:
        if not self.rounds_dir.exists():
            return []
        return [
            self.reconcile_round(path.stem)
            for path in sorted(self.rounds_dir.glob("round-*.json"))
        ]
