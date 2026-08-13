"""Strategy-neutral Campaign lifecycle and checkpoint projection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping

from evolve.contracts import Authorization, ContractViolation, ExecutionPlan

from .budget_manager import BudgetManager, BudgetSnapshot
from .checkpoint_manager import CheckpointManager


class CampaignStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


TERMINAL_STATUSES = frozenset(
    {
        CampaignStatus.COMPLETED,
        CampaignStatus.FAILED,
        CampaignStatus.BLOCKED,
        CampaignStatus.CANCELLED,
        CampaignStatus.PARTIAL,
    }
)


@dataclass(frozen=True, slots=True)
class WorkItemSnapshot:
    plan_id: str
    plan_sha256: str
    arm: str
    status: str
    reserved_cost_cny: float
    reserved_model_calls: int
    actual_cost_cny: float | None = None
    actual_model_calls: int | None = None


@dataclass(frozen=True, slots=True)
class CampaignSnapshot:
    campaign_id: str
    authorization_id: str
    status: CampaignStatus
    reason: str | None
    budget: BudgetSnapshot
    work_items: tuple[WorkItemSnapshot, ...]


class CampaignController:
    """Own lifecycle and resource admission without interpreting Strategy arms."""

    def __init__(
        self,
        *,
        campaign_id: str,
        authorization: Authorization,
        now: datetime,
        checkpoint_manager: CheckpointManager | None,
        status: CampaignStatus,
        reason: str | None,
        budget: BudgetManager,
        work_items: Mapping[str, WorkItemSnapshot],
    ) -> None:
        if authorization.campaign_id != campaign_id:
            raise ContractViolation("authorization campaign does not match controller")
        authorization.assert_allows(
            cohort=authorization.allowed_cohorts[0],
            reserved_cost_cny=0,
            reserved_model_calls=0,
            remote=False,
            now=now,
        )
        self._campaign_id = campaign_id
        self._authorization = authorization
        self._now = now
        self._checkpoints = checkpoint_manager
        self._status = status
        self._reason = reason
        self._budget = budget
        self._work_items = dict(work_items)

    @classmethod
    def create(
        cls,
        *,
        campaign_id: str,
        authorization: Authorization,
        now: datetime,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> CampaignController:
        controller = cls(
            campaign_id=campaign_id,
            authorization=authorization,
            now=now,
            checkpoint_manager=checkpoint_manager,
            status=CampaignStatus.CREATED,
            reason=None,
            budget=BudgetManager(
                max_cost_cny=authorization.max_cost_cny,
                max_model_calls=authorization.max_model_calls,
            ),
            work_items={},
        )
        controller._save()
        return controller

    @classmethod
    def from_checkpoint(
        cls,
        *,
        campaign_id: str,
        authorization: Authorization,
        checkpoint_manager: CheckpointManager,
        now: datetime,
    ) -> CampaignController:
        payload = checkpoint_manager.load(campaign_id)
        if payload.get("campaign_id") != campaign_id:
            raise ContractViolation("checkpoint campaign identity mismatch")
        if payload.get("authorization_id") != authorization.authorization_id:
            raise ContractViolation("checkpoint authorization identity mismatch")
        budget_payload = payload["budget"]
        work_items = {
            item["plan_id"]: WorkItemSnapshot(**item)
            for item in payload.get("work_items", ())
        }
        return cls(
            campaign_id=campaign_id,
            authorization=authorization,
            now=now,
            checkpoint_manager=checkpoint_manager,
            status=CampaignStatus(payload["status"]),
            reason=payload.get("reason"),
            budget=BudgetManager(**budget_payload),
            work_items=work_items,
        )

    def start(self) -> None:
        if self._status not in {CampaignStatus.CREATED, CampaignStatus.PAUSED}:
            raise ContractViolation(f"cannot start campaign from {self._status}")
        self._status = CampaignStatus.RUNNING
        self._reason = None
        self._save()

    def pause(self, reason: str) -> None:
        self._require_running()
        self._status = CampaignStatus.PAUSED
        self._reason = reason
        self._save()

    def submit(
        self,
        plan: ExecutionPlan,
        *,
        reserved_model_calls: int = 1,
        remote: bool | None = None,
    ) -> bool:
        self._require_running()
        if plan.campaign_id != self._campaign_id:
            raise ContractViolation("execution plan campaign does not match controller")
        existing = self._work_items.get(plan.plan_id)
        if existing is not None:
            if existing.plan_sha256 != plan.content_sha256:
                raise ContractViolation("duplicate plan_id has conflicting content")
            return False
        remote_call = (
            remote
            if remote is not None
            else plan.model.provider
            not in {
                "local",
                "local-mlx",
                "mlx",
            }
        )
        self._authorization.assert_allows(
            cohort=plan.task.cohort,
            reserved_cost_cny=0,
            reserved_model_calls=0,
            remote=remote_call,
            now=self._now,
        )
        self._budget.reserve(
            cost_cny=plan.limits.max_cost_cny,
            model_calls=reserved_model_calls,
        )
        self._work_items[plan.plan_id] = WorkItemSnapshot(
            plan_id=plan.plan_id,
            plan_sha256=plan.content_sha256,
            arm=plan.arm,
            status="pending",
            reserved_cost_cny=plan.limits.max_cost_cny,
            reserved_model_calls=reserved_model_calls,
        )
        self._save()
        return True

    def record_result(
        self,
        plan_id: str,
        *,
        actual_cost_cny: float,
        actual_model_calls: int,
        succeeded: bool = True,
    ) -> None:
        self._require_running()
        item = self._work_items.get(plan_id)
        if item is None:
            raise ContractViolation("cannot record an unknown execution plan")
        if item.status in {"completed", "failed"}:
            if (
                item.actual_cost_cny == actual_cost_cny
                and item.actual_model_calls == actual_model_calls
                and item.status == ("completed" if succeeded else "failed")
            ):
                return
            raise ContractViolation(
                "execution result already recorded with other facts"
            )
        self._budget.record(
            reserved_cost_cny=item.reserved_cost_cny,
            reserved_model_calls=item.reserved_model_calls,
            actual_cost_cny=actual_cost_cny,
            actual_model_calls=actual_model_calls,
        )
        self._work_items[plan_id] = WorkItemSnapshot(
            plan_id=item.plan_id,
            plan_sha256=item.plan_sha256,
            arm=item.arm,
            status="completed" if succeeded else "failed",
            reserved_cost_cny=item.reserved_cost_cny,
            reserved_model_calls=item.reserved_model_calls,
            actual_cost_cny=actual_cost_cny,
            actual_model_calls=actual_model_calls,
        )
        self._save()

    def finalize(self, status: CampaignStatus, reason: str | None = None) -> None:
        if self._status in TERMINAL_STATUSES:
            raise ContractViolation("campaign is already terminal")
        if status not in TERMINAL_STATUSES:
            raise ContractViolation("final campaign status must be terminal")
        if status is CampaignStatus.COMPLETED and any(
            item.status != "completed" for item in self._work_items.values()
        ):
            raise ContractViolation("completed campaign has unfinished work")
        self._status = status
        self._reason = reason
        self._save()

    def mark_partial(self, reason: str) -> None:
        self.finalize(CampaignStatus.PARTIAL, reason)

    def snapshot(self) -> CampaignSnapshot:
        return CampaignSnapshot(
            campaign_id=self._campaign_id,
            authorization_id=self._authorization.authorization_id,
            status=self._status,
            reason=self._reason,
            budget=self._budget.snapshot(),
            work_items=tuple(self._work_items.values()),
        )

    def _require_running(self) -> None:
        if self._status in TERMINAL_STATUSES:
            raise ContractViolation("campaign is terminal")
        if self._status is not CampaignStatus.RUNNING:
            raise ContractViolation("campaign must be running")

    def _save(self) -> None:
        if self._checkpoints is None:
            return
        snapshot = self.snapshot()
        payload: dict[str, Any] = asdict(snapshot)
        payload["status"] = str(snapshot.status)
        self._checkpoints.save(self._campaign_id, payload)
