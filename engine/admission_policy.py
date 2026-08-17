"""Project-local admission guard for OpenEvolve candidates.

The upstream OpenEvolve checkout remains unmodified.  ``OpenEvolveAdmissionGuard``
wraps one database instance before a run starts, so every evaluated child remains
traceable while only non-regressive, novel candidates become selectable parents.
"""

from __future__ import annotations

import ast
import hashlib
import json
import types
from dataclasses import dataclass
from typing import Any


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_fingerprints(source: str) -> tuple[str, str]:
    """Return exact-source and formatting/comment-insensitive AST hashes."""
    exact_hash = _sha256(source)
    try:
        tree_dump = ast.dump(ast.parse(source), include_attributes=False)
    except SyntaxError:
        tree_dump = f"<invalid>:{source}"
    return exact_hash, _sha256(tree_dump)


def behavior_signature(metrics: dict[str, Any], prefix: str = "case_") -> str:
    """Hash the full ordered public-case pass vector."""
    values = [
        f"{key}={int(float(metrics[key]) >= 1.0)}"
        for key in sorted(metrics)
        if key.startswith(prefix)
    ]
    return _sha256("|".join(values))


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: bool
    reasons: tuple[str, ...]
    regressed_cases: tuple[str, ...]
    gained_cases: tuple[str, ...]
    source_hash: str
    ast_hash: str
    behavior_signature: str


class AdmissionPolicy:
    """Stateful non-regression and semantic-dedup admission policy."""

    def __init__(
        self,
        *,
        protected_metric_prefix: str = "case_",
        behavior_equivalent_limit: int = 2,
    ) -> None:
        if behavior_equivalent_limit < 1:
            raise ValueError("behavior_equivalent_limit must be at least one")
        self.protected_metric_prefix = protected_metric_prefix
        self.behavior_equivalent_limit = behavior_equivalent_limit
        self.source_hashes: set[str] = set()
        self.ast_hashes: set[str] = set()
        self.behavior_counts: dict[str, int] = {}

    def assess_and_register(
        self,
        source: str,
        metrics: dict[str, Any],
        parent_metrics: dict[str, Any] | None,
    ) -> AdmissionDecision:
        source_hash, ast_hash = source_fingerprints(source)
        signature = behavior_signature(metrics, self.protected_metric_prefix)
        reasons: list[str] = []

        protected = sorted(
            key for key in metrics if key.startswith(self.protected_metric_prefix)
        )
        parent_metrics = parent_metrics or {}
        regressed = tuple(
            key
            for key in protected
            if float(parent_metrics.get(key, 0.0)) >= 1.0
            and float(metrics.get(key, 0.0)) < 1.0
        )
        gained = tuple(
            key
            for key in protected
            if float(parent_metrics.get(key, 0.0)) < 1.0
            and float(metrics.get(key, 0.0)) >= 1.0
        )

        if float(metrics.get("evaluator_valid", 1.0)) < 1.0:
            reasons.append("evaluator_rejected")
        if regressed:
            reasons.append("parent_regression")
        if source_hash in self.source_hashes:
            reasons.append("exact_duplicate")
        elif ast_hash in self.ast_hashes:
            reasons.append("ast_duplicate")
        elif self.behavior_counts.get(signature, 0) >= self.behavior_equivalent_limit:
            reasons.append("behavior_duplicate_limit")

        accepted = not reasons
        decision = AdmissionDecision(
            accepted=accepted,
            reasons=tuple(reasons),
            regressed_cases=regressed,
            gained_cases=gained,
            source_hash=source_hash,
            ast_hash=ast_hash,
            behavior_signature=signature,
        )
        if accepted:
            self.source_hashes.add(source_hash)
            self.ast_hashes.add(ast_hash)
            self.behavior_counts[signature] = self.behavior_counts.get(signature, 0) + 1
        return decision


class OpenEvolveAdmissionGuard:
    """Install :class:`AdmissionPolicy` on a single OpenEvolve database."""

    def __init__(
        self,
        *,
        run_context: dict[str, Any],
        event_store: Any | None = None,
        behavior_equivalent_limit: int = 2,
    ) -> None:
        self.policy = AdmissionPolicy(
            behavior_equivalent_limit=behavior_equivalent_limit
        )
        self.run_context = dict(run_context)
        self.event_store = event_store
        self._indexed_program_ids: set[str] = set()

    def _sync_existing(self, database: Any) -> None:
        """Rebuild the guard index lazily after checkpoint loading."""
        for program_id, program in database.programs.items():
            if program_id in self._indexed_program_ids:
                continue
            admission = program.metadata.get("admission", {})
            if admission.get("accepted", True):
                self.policy.assess_and_register(program.code, program.metrics, None)
            self._indexed_program_ids.add(program_id)

    def _record_event(
        self,
        program: Any,
        parent: Any | None,
        decision: AdmissionDecision,
        iteration: int | None,
    ) -> None:
        if self.event_store is None:
            return
        parent_score = (
            float(parent.metrics.get("combined_score", 0.0)) if parent else 0.0
        )
        child_score = float(program.metrics.get("combined_score", 0.0))
        payload = {
            **self.run_context,
            "event_type": "candidate",
            "program_id": program.id,
            "parent_id": getattr(parent, "id", None),
            "iteration": int(iteration or 0),
            "accepted": decision.accepted,
            "admission_reasons": list(decision.reasons),
            "regressed_cases": list(decision.regressed_cases),
            "gained_cases": list(decision.gained_cases),
            "source_hash": decision.source_hash,
            "ast_hash": decision.ast_hash,
            "behavior_signature": decision.behavior_signature,
            "parent_score": parent_score,
            "child_score": child_score,
            "metrics": program.metrics,
            "code": program.code,
            "holdout_verified": False,
            "tags": [case.removeprefix("case_") for case in decision.gained_cases],
        }
        stable = json.dumps(
            {key: payload[key] for key in ("run_id", "program_id", "iteration")},
            sort_keys=True,
        )
        payload["event_id"] = _sha256(stable)
        self.event_store.append_event(payload)

    def install(self, database: Any) -> None:
        """Wrap ``database.add`` without changing the upstream checkout."""
        original_add = database.add
        guard = self

        def guarded_add(
            database_self: Any,
            program: Any,
            iteration: int | None = None,
            target_island: int | None = None,
        ) -> str:
            guard._sync_existing(database_self)
            parent = (
                database_self.programs.get(program.parent_id)
                if program.parent_id
                else None
            )
            decision = guard.policy.assess_and_register(
                program.code,
                program.metrics,
                parent.metrics if parent else None,
            )
            program.metadata["admission"] = {
                "accepted": decision.accepted,
                "reasons": list(decision.reasons),
                "regressed_cases": list(decision.regressed_cases),
                "gained_cases": list(decision.gained_cases),
                "source_hash": decision.source_hash,
                "ast_hash": decision.ast_hash,
                "behavior_signature": decision.behavior_signature,
            }
            guard._record_event(program, parent, decision, iteration)
            guard._indexed_program_ids.add(program.id)

            if decision.accepted:
                return original_add(
                    program,
                    iteration=iteration,
                    target_island=target_island,
                )

            # Keep rejected candidates in the program database for the trace and
            # checkpoint, but never place them in an island/archive or best set.
            if iteration is not None:
                program.iteration_found = iteration
                database_self.last_iteration = max(
                    database_self.last_iteration, iteration
                )
            if target_island is not None:
                program.metadata["island"] = target_island % len(database_self.islands)
            database_self.programs[program.id] = program
            return program.id

        database.add = types.MethodType(guarded_add, database)
