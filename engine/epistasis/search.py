"""Fixed-budget search cells over (task x seed x operator-set x mode).

A cell is the atomic unit of every experiment.  It starts from the task's
frozen initial program and iteratively mutates the best-so-far program with an
active operator set (deterministic) or with an LLM repairing a deterministic
scaffold (``llm`` mode).  Admission reuses the project's ``AdmissionPolicy``
(non-regression + exact/AST/behavior dedup), so results stay comparable with the
existing evolution runs.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Protocol

from admission_policy import AdmissionPolicy, behavior_signature, source_fingerprints

from .operators import apply_operators
from .tasks import TaskSpec

CASE_PREFIX = "case_"


@dataclass(frozen=True, slots=True)
class AttemptEvent:
    iteration: int
    operator_ids: tuple[str, ...]
    parent_source_hash: str
    parent_passed: int
    source: str
    source_hash: str
    ast_hash: str
    changed: bool
    postcondition_ok: bool
    reason: str
    passed_cases: int
    total_cases: int
    accepted: bool
    improved: bool
    admission_reasons: tuple[str, ...] = ()
    holdout_passed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "operator_ids": list(self.operator_ids),
            "parent_source_hash": self.parent_source_hash,
            "parent_passed": self.parent_passed,
            "source": self.source,
            "source_hash": self.source_hash,
            "ast_hash": self.ast_hash,
            "changed": self.changed,
            "postcondition_ok": self.postcondition_ok,
            "reason": self.reason,
            "passed_cases": self.passed_cases,
            "total_cases": self.total_cases,
            "accepted": self.accepted,
            "improved": self.improved,
            "admission_reasons": list(self.admission_reasons),
            "holdout_passed": self.holdout_passed,
        }


@dataclass(frozen=True, slots=True)
class CellResult:
    task_id: str
    seed: int
    operator_ids: tuple[str, ...]
    mode: str
    budget: int
    initial_passed: int
    best_source: str
    best_passed: int
    best_holdout: int
    accepted_count: int
    improved_count: int
    no_op_count: int
    invalid_count: int
    events: tuple[AttemptEvent, ...] = field(default=())

    @property
    def yield_(self) -> float:
        return self.improved_count / max(1, self.budget)

    @property
    def accept_rate(self) -> float:
        return self.accepted_count / max(1, self.budget)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "operator_ids": list(self.operator_ids),
            "mode": self.mode,
            "budget": self.budget,
            "initial_passed": self.initial_passed,
            "best_passed": self.best_passed,
            "best_holdout": self.best_holdout,
            "accepted_count": self.accepted_count,
            "improved_count": self.improved_count,
            "no_op_count": self.no_op_count,
            "invalid_count": self.invalid_count,
            "yield": self.yield_,
            "accept_rate": self.accept_rate,
            "event_count": len(self.events),
        }

    def to_full_dict(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "best_source": self.best_source,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_full_dict(cls, payload: dict[str, Any]) -> "CellResult":
        return cls(
            task_id=str(payload["task_id"]),
            seed=int(payload["seed"]),
            operator_ids=tuple(str(item) for item in payload["operator_ids"]),
            mode=str(payload["mode"]),
            budget=int(payload["budget"]),
            initial_passed=int(payload["initial_passed"]),
            best_source=str(payload["best_source"]),
            best_passed=int(payload["best_passed"]),
            best_holdout=int(payload["best_holdout"]),
            accepted_count=int(payload["accepted_count"]),
            improved_count=int(payload["improved_count"]),
            no_op_count=int(payload["no_op_count"]),
            invalid_count=int(payload["invalid_count"]),
            events=tuple(
                AttemptEvent(
                    iteration=int(row["iteration"]),
                    operator_ids=tuple(str(item) for item in row["operator_ids"]),
                    parent_source_hash=str(row["parent_source_hash"]),
                    parent_passed=int(row["parent_passed"]),
                    source=str(row["source"]),
                    source_hash=str(row["source_hash"]),
                    ast_hash=str(row["ast_hash"]),
                    changed=bool(row["changed"]),
                    postcondition_ok=bool(row["postcondition_ok"]),
                    reason=str(row["reason"]),
                    passed_cases=int(row["passed_cases"]),
                    total_cases=int(row["total_cases"]),
                    accepted=bool(row["accepted"]),
                    improved=bool(row["improved"]),
                    admission_reasons=tuple(
                        str(item) for item in row.get("admission_reasons", [])
                    ),
                    holdout_passed=(
                        None
                        if row.get("holdout_passed") is None
                        else int(row["holdout_passed"])
                    ),
                )
                for row in payload["events"]
            ),
        )


def cell_key(
    task_id: str, seed: int, operator_ids: tuple[str, ...], mode: str, budget: int
) -> str:
    raw = json.dumps(
        [task_id, seed, list(operator_ids), mode, budget],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _admission_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "evaluator_valid": float(metrics.get("evaluator_valid", 1.0))
    }
    for case in metrics.get("case_results", []):
        out[f"{CASE_PREFIX}{case['id']}"] = 1.0 if case["passed"] else 0.0
    return out


class RepairModel(Protocol):
    def complete(self, prompt: str, *, temperature: float = 0.7) -> str: ...


def _repair_prompt(
    scaffold: str, task: TaskSpec, target_failure: str | None, rng: random.Random
) -> str:
    failing = [
        case["id"]
        for case in task.public_cases
        if target_failure is None or case["id"] == target_failure
    ]
    target = target_failure or (failing[0] if failing else "unknown")
    return (
        "[Structured repair; observer signal only]\n"
        f"Task: {task.task_id}\n"
        f"Target public failure to fix: {target}\n"
        "Repair the scaffold below so the target failure passes while preserving "
        "already-passing behavior. Do not remove the scaffold's guards. "
        "Return only the full final python program inside a ```python fence.\n\n"
        f"Scaffold:\n```python\n{scaffold}\n```\n"
    )


def _free_prompt(source: str, task: TaskSpec, target_failure: str | None) -> str:
    """Free-form generation prompt: no operator scaffold (Qwen control arm)."""
    target = target_failure or "unknown"
    return (
        "[Free generation; observer signal only]\n"
        f"Task: {task.task_id}\n"
        f"Target public failure to fix: {target}\n"
        "Produce an improved full program that fixes the target failure while "
        "preserving already-passing behavior. Return only the full final python "
        "program inside a ```python fence.\n\n"
        f"Current program:\n```python\n{source}\n```\n"
    )


def _extract_fenced_source(text: str) -> str | None:
    import re

    fence = re.compile(r"```(?:python|py)?[^\n]*\n(.*?)```", re.IGNORECASE | re.DOTALL)
    blocks = [block.strip() for block in fence.findall(text) if block.strip()]
    if blocks:
        return max(blocks, key=len)
    stripped = text.strip()
    return stripped if "def solve" in stripped else None


def run_cell(
    task: TaskSpec,
    operator_ids: tuple[str, ...],
    *,
    seed: int,
    budget: int,
    mode: str = "deterministic",
    model: RepairModel | None = None,
    target_failure: str | None = None,
    track_holdout: bool = True,
    llm_style: str = "scaffold",
) -> CellResult:
    """Run one fixed-budget search cell and return the full evidence trail.

    ``llm_style`` only applies in ``llm`` mode:

    - ``scaffold``: deterministic operators build a scaffold, then the model
      repairs it (extraction failure falls back to the scaffold).
    - ``free``: the model generates the full program from scratch with no
      operator scaffold (the control arm that historically collapses).
    """
    if mode not in {"deterministic", "llm"}:
        raise ValueError(f"unsupported cell mode: {mode}")
    if mode == "llm" and model is None:
        raise ValueError("llm mode requires a model transport")
    if mode == "llm" and llm_style not in {"scaffold", "free"}:
        raise ValueError(f"unsupported llm_style: {llm_style}")
    rng = random.Random(seed)
    initial_metrics = task.score_source(task.initial_source)
    initial_passed = int(initial_metrics.get("passed_cases", 0))
    admission = AdmissionPolicy(behavior_equivalent_limit=2)
    initial_hashes = source_fingerprints(task.initial_source)
    initial_sig = behavior_signature(_admission_metrics(initial_metrics), CASE_PREFIX)
    admission.source_hashes.add(initial_hashes[0])
    admission.ast_hashes.add(initial_hashes[1])
    admission.behavior_counts[initial_sig] = 1

    best_source = task.initial_source
    best_passed = initial_passed
    events: list[AttemptEvent] = []
    accepted_count = improved_count = no_op_count = invalid_count = 0

    for iteration in range(1, budget + 1):
        parent_source = best_source
        parent_metrics = task.score_source(parent_source)
        parent_passed = int(parent_metrics.get("passed_cases", 0))
        parent_hash = source_fingerprints(parent_source)[0]

        changed = False
        postcondition_ok = False
        reason = "no-candidate"
        candidate: str | None = None

        if mode == "deterministic":
            composed = apply_operators(parent_source, operator_ids, task.schema)
            changed = bool(composed.results and composed.results[-1].changed)
            postcondition_ok = composed.ok
            reason = composed.reason
            if composed.ok:
                candidate = composed.source
        else:
            composed = apply_operators(parent_source, operator_ids, task.schema)
            if llm_style == "free":
                changed = True
                reason = "llm-free"
                prompt = _free_prompt(parent_source, task, target_failure)
                response = model.complete(prompt, temperature=0.7)
                candidate = _extract_fenced_source(response)
                if candidate is None:
                    candidate = None
                    reason = "llm-extraction-failed"
                    changed = False
                postcondition_ok = candidate is not None
            elif not composed.ok:
                changed = False
                postcondition_ok = False
                reason = f"scaffold:{composed.reason}"
            else:
                changed = True
                postcondition_ok = True
                reason = "llm-repair"
                prompt = _repair_prompt(composed.source, task, target_failure, rng)
                response = model.complete(prompt, temperature=0.7)
                candidate = _extract_fenced_source(response)
                if candidate is None:
                    candidate = composed.source
                    reason = "llm-extraction-fallback"

        source_hash = ast_hash = ""
        if candidate is not None and changed:
            source_hash, ast_hash = source_fingerprints(candidate)
            candidate_metrics = task.score_source(candidate)
            candidate_passed = int(candidate_metrics.get("passed_cases", 0))
            decision = admission.assess_and_register(
                candidate,
                _admission_metrics(candidate_metrics),
                _admission_metrics(parent_metrics),
            )
            accepted = decision.accepted
            improved = accepted and candidate_passed > parent_passed
            if accepted and candidate_passed > best_passed:
                best_source = candidate
                best_passed = candidate_passed
            if accepted:
                accepted_count += 1
            if improved:
                improved_count += 1
            if not changed:
                no_op_count += 1
            holdout_passed = None
            if track_holdout and accepted:
                holdout_metrics = task.score_holdout(candidate)
                holdout_passed = int(holdout_metrics.get("passed_cases", 0))
        else:
            accepted = False
            improved = False
            candidate_metrics = None
            candidate_passed = parent_passed
            decision = None
            if not changed:
                no_op_count += 1
            else:
                invalid_count += 1
            holdout_passed = None

        events.append(
            AttemptEvent(
                iteration=iteration,
                operator_ids=operator_ids,
                parent_source_hash=parent_hash,
                parent_passed=parent_passed,
                source=candidate or "",
                source_hash=source_hash,
                ast_hash=ast_hash,
                changed=changed,
                postcondition_ok=postcondition_ok,
                reason=reason,
                passed_cases=candidate_passed,
                total_cases=len(task.public_cases),
                accepted=accepted,
                improved=improved,
                admission_reasons=tuple(decision.reasons) if decision else (),
                holdout_passed=holdout_passed,
            )
        )

    best_holdout = 0
    if track_holdout:
        holdout_metrics = task.score_holdout(best_source)
        best_holdout = int(holdout_metrics.get("passed_cases", 0))

    return CellResult(
        task_id=task.task_id,
        seed=seed,
        operator_ids=operator_ids,
        mode=mode,
        budget=budget,
        initial_passed=initial_passed,
        best_source=best_source,
        best_passed=best_passed,
        best_holdout=best_holdout,
        accepted_count=accepted_count,
        improved_count=improved_count,
        no_op_count=no_op_count,
        invalid_count=invalid_count,
        events=tuple(events),
    )


def run_lineage_cell(
    task: TaskSpec,
    operator_order: tuple[str, ...],
    *,
    seed: int,
    budget: int,
    track_holdout: bool = True,
) -> CellResult:
    """Sequential single-operator generations (Experiment C).

    Each iteration applies the next operator in ``operator_order`` (rotating) to
    the *chained best* program, so conjunction is reached across generations
    instead of inside one mutation.
    """
    if not operator_order:
        raise ValueError("lineage requires at least one operator")
    best_source = task.initial_source
    best_passed = int(task.score_source(best_source).get("passed_cases", 0))
    events: list[AttemptEvent] = []
    accepted_count = improved_count = no_op_count = invalid_count = 0

    for iteration in range(1, budget + 1):
        op = operator_order[(iteration - 1) % len(operator_order)]
        parent_metrics = task.score_source(best_source)
        parent_passed = int(parent_metrics.get("passed_cases", 0))
        parent_hash = source_fingerprints(best_source)[0]

        composed = apply_operators(best_source, (op,), task.schema)
        changed = composed.ok and bool(
            composed.results and composed.results[-1].changed
        )
        candidate = composed.source if changed else None

        accepted = False
        improved = False
        candidate_passed = parent_passed
        reasons: tuple[str, ...] = ()
        source_hash = ast_hash = ""
        if changed and candidate is not None:
            source_hash, ast_hash = source_fingerprints(candidate)
            candidate_metrics = task.score_source(candidate)
            candidate_passed = int(candidate_metrics.get("passed_cases", 0))
            admission_metrics_c = _admission_metrics(candidate_metrics)
            admission_metrics_p = _admission_metrics(parent_metrics)
            regressed = [
                key
                for key in admission_metrics_c
                if key.startswith(CASE_PREFIX)
                and float(admission_metrics_p.get(key, 0.0)) >= 1.0
                and float(admission_metrics_c.get(key, 0.0)) < 1.0
            ]
            if regressed:
                reasons = ("parent_regression",)
            elif candidate_passed < best_passed:
                reasons = ("score_decrease",)
            else:
                accepted = True
            if accepted:
                improved = candidate_passed > parent_passed
                if candidate_passed > best_passed:
                    best_source = candidate
                    best_passed = candidate_passed
                accepted_count += 1
            if improved:
                improved_count += 1
        else:
            no_op_count += 1

        events.append(
            AttemptEvent(
                iteration=iteration,
                operator_ids=(op,),
                parent_source_hash=parent_hash,
                parent_passed=parent_passed,
                source=candidate or "",
                source_hash=source_hash,
                ast_hash=ast_hash,
                changed=changed,
                postcondition_ok=composed.ok,
                reason=composed.reason if not changed else "ok",
                passed_cases=candidate_passed,
                total_cases=len(task.public_cases),
                accepted=accepted,
                improved=improved,
                admission_reasons=reasons,
                holdout_passed=None,
            )
        )

    best_holdout = 0
    if track_holdout:
        best_holdout = int(task.score_holdout(best_source).get("passed_cases", 0))

    return CellResult(
        task_id=task.task_id,
        seed=seed,
        operator_ids=operator_order,
        mode="lineage",
        budget=budget,
        initial_passed=int(
            task.score_source(task.initial_source).get("passed_cases", 0)
        ),
        best_source=best_source,
        best_passed=best_passed,
        best_holdout=best_holdout,
        accepted_count=accepted_count,
        improved_count=improved_count,
        no_op_count=no_op_count,
        invalid_count=invalid_count,
        events=tuple(events),
    )
