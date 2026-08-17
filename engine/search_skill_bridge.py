"""Compile v2.2 search results into project-local candidate skills with a cross-task transfer gate."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from skill_registry import (
    SkillCandidate,
    SkillEvidenceRef,
    SkillRegistry,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
MIN_PAIRED_TASKS = 8
SCORE_TOLERANCE = 0.01
COST_INCREASE_LIMIT = 0.10


class SearchSkillBridgeError(ValueError):
    """Raised when search evidence cannot be compiled or gated safely."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SearchSkillBridgeError(f"missing evidence file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SearchSkillBridgeError(f"missing evidence file: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _evidence_refs(paths: tuple[tuple[Path, str], ...]) -> tuple[SkillEvidenceRef, ...]:
    refs = []
    for path, role in paths:
        if path.is_file():
            refs.append(SkillEvidenceRef.from_path(path, role=role))
    return tuple(dict.fromkeys(refs))


def _find_champion_changeset(
    run_root: Path, generation: int, champion: str
) -> tuple[Path, dict[str, Any]]:
    """Locate the frozen changeset for a selected candidate (written in the prior generation)."""
    provenance_dir = run_root / f"generation-{generation - 1}"
    for possible in sorted((provenance_dir / "candidates").glob("*")):
        candidate_changeset = possible / "CHANGESET.json"
        if not candidate_changeset.is_file():
            continue
        payload = json.loads(candidate_changeset.read_text(encoding="utf-8"))
        if payload.get("candidate_agent_program_sha256") == champion:
            return candidate_changeset, payload
    raise SearchSkillBridgeError(
        f"selected candidate {champion[:16]}… lacks a frozen changeset"
    )


def extract_confirmation_paired_evals(
    *, run_root: Path, generation: int
) -> dict[str, Any]:
    """Fresh-task paired evidence from the confirmation stage of one generation."""
    generation_dir = run_root / f"generation-{generation}"
    tournament = _read_json(generation_dir / "TOURNAMENT.json")
    confirmation = tournament.get("confirmation", {})
    task_uids = set(confirmation.get("task_uids", ()))
    ledger = _read_jsonl(generation_dir / "EXECUTION-LEDGER.jsonl")
    rows = [
        row
        for row in ledger
        if row.get("task_uid") in task_uids
        and row.get("role") in ("original", "candidate")
    ]
    fields = (
        "task_uid",
        "benchmark_family",
        "role",
        "native_score",
        "safety_passed",
        "cost_units",
        "matched_contract_sha256",
        "native_evaluator_epoch",
    )
    evals = [{key: row[key] for key in fields} for row in rows]
    contracts = {row["matched_contract_sha256"] for row in rows}
    epochs = {row["native_evaluator_epoch"] for row in rows}
    return {
        "evals": evals,
        "expected_contract_sha256": next(iter(contracts))
        if len(contracts) == 1
        else None,
        "expected_evaluator_epoch": next(iter(epochs)) if len(epochs) == 1 else None,
    }


def finalize_skill_bridge(
    *,
    result: dict[str, Any],
    run_root: Path,
    registry_root: Path,
    auto_gate: bool = True,
) -> dict[str, Any]:
    """Compile selected candidates into the registry and optionally apply the transfer gate."""
    candidates = compile_candidate_skills(result=result, run_root=run_root)
    registry = SkillRegistry(registry_root)
    for candidate in candidates:
        registry.append(candidate)
    applied: list[dict[str, Any]] = []
    if auto_gate:
        for generation in result.get("generations", []):
            champion = generation.get("selected_candidate")
            if not champion:
                continue
            gen = int(generation["generation"])
            paired = extract_confirmation_paired_evals(
                run_root=run_root, generation=gen
            )
            gate = evaluate_transfer_gate(
                paired_evals=paired["evals"],
                expected_contract_sha256=paired["expected_contract_sha256"],
                expected_evaluator_epoch=paired["expected_evaluator_epoch"],
            )
            _changeset_path, changeset = _find_champion_changeset(
                run_root, gen, champion
            )
            surface = str(changeset.get("surface", "skills"))
            skill_id = f"search-candidate-{champion[:12]}-{surface}"
            gate_path = run_root / f"TRANSFER-GATE-{skill_id}.json"
            gate_path.write_text(
                json.dumps(gate.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            terminal = apply_transfer_gate(
                registry=registry,
                candidate=registry.latest(skill_id),
                gate_result=gate,
                gate_evidence_path=gate_path,
            )
            registry.render_for_review(skill_id)
            applied.append(terminal.to_dict())
    summary = {
        "compiled": [candidate.to_dict() for candidate in candidates],
        "applied_gates": applied,
    }
    (run_root / "CANDIDATE-SKILLS.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def compile_candidate_skills(
    *, result: dict[str, Any], run_root: Path
) -> tuple[SkillCandidate, ...]:
    """Compile every selected candidate into an inactive project-local SkillCandidate."""
    run_root = run_root.resolve()
    compiled: list[SkillCandidate] = []
    for generation in result.get("generations", []):
        champion = generation.get("selected_candidate")
        if not champion:
            continue
        gen = int(generation["generation"])
        generation_dir = run_root / f"generation-{gen}"
        cards = [row for row in _read_jsonl(generation_dir / "PATTERN-CARDS.jsonl")]
        by_id = {row["pattern_id"]: row for row in cards}
        tournament = _read_json(generation_dir / "TOURNAMENT.json")
        confirmation = tournament.get("confirmation", {})
        task_uids = tuple(confirmation.get("task_uids", ()))
        families = sorted(
            {
                row.get("benchmark_family", "")
                for row in _read_jsonl(generation_dir / "OBSERVER-EVIDENCE.jsonl")
                if row.get("benchmark_family")
            }
        )
        task_family = "search-" + ("-".join(families) if families else "validated")
        changeset_path, changeset = _find_champion_changeset(run_root, gen, champion)
        hypothesis_ids = tuple(changeset.get("hypothesis_ids", ()))
        surface = str(changeset.get("surface", "skills"))
        hypothesis_cards = tuple(by_id.get(hid) for hid in hypothesis_ids)
        hypothesis_cards = tuple(card for card in hypothesis_cards if card)
        conditions = sorted(
            {
                condition
                for card in hypothesis_cards
                for condition in card.get("conditions", ())
            }
        )
        failure_cards = [
            card for card in cards if card.get("pattern_kind") == "failure"
        ]
        observer = _read_jsonl(generation_dir / "OBSERVER-EVIDENCE.jsonl")
        negative_ids = sorted(
            {
                row["evidence_id"]
                for row in observer
                if row.get("outcome") != "advantage"
            }
        )
        counterexamples = tuple(
            dict.fromkeys(
                [
                    *negative_ids,
                    *[
                        item
                        for card in failure_cards
                        for item in card.get("counterexample_evidence_ids", ())
                    ],
                ]
            )
        )
        failure_modes = tuple(
            dict.fromkeys(
                card.get("observed_feature", "")
                for card in failure_cards
                if card.get("observed_feature")
            )
        )
        if not counterexamples or not failure_modes:
            raise SearchSkillBridgeError(
                f"generation {gen} lacks negative evidence; refusing to fabricate"
            )
        content = tuple(
            dict.fromkeys(
                [
                    *[
                        line
                        for card in hypothesis_cards
                        for line in (
                            f"hypothesis: {card['pattern_id']}",
                            f"behavior: {card.get('observed_feature', '')}",
                            f"condition: {'; '.join(card.get('conditions', ()))}",
                        )
                    ],
                    f"surface: {surface}",
                    *[
                        f"operation: {op.get('op')} {op.get('path')}"
                        for op in changeset.get("operations", ())
                    ],
                ]
            )
        )
        skill_id = f"search-candidate-{champion[:12]}-{surface}"
        revision_id = f"{skill_id}-{_sha({'content': content})[:12]}"
        semantics = sorted({*conditions, "search_validated"})
        applicability = {"required_semantics": semantics}
        candidate = SkillCandidate.create(
            skill_id=skill_id,
            revision_id=revision_id,
            parent_revision_id=None,
            status="candidate",
            status_reason=(
                "compiled from v2.2 search evidence; inactive project-local candidate"
            ),
            task_family=task_family,
            content=content,
            source_task_ids=task_uids,
            applicability=applicability,
            counterexamples=counterexamples,
            known_failure_modes=failure_modes,
            evidence_refs=_evidence_refs(
                (
                    (
                        generation_dir / "EXECUTION-LEDGER.jsonl",
                        "search_execution_ledger",
                    ),
                    (
                        generation_dir / "OBSERVER-EVIDENCE.jsonl",
                        "search_observer_evidence",
                    ),
                    (generation_dir / "PATTERN-CARDS.jsonl", "search_pattern_cards"),
                    (generation_dir / "TOURNAMENT.json", "search_tournament"),
                    (generation_dir / "PARENT-DECISION.json", "search_parent_decision"),
                    (generation_dir / "RESULT.zh-CN.md", "search_generation_report"),
                    (changeset_path, "changeset"),
                )
            ),
            project_local_only=True,
            auto_install=False,
            active=False,
        )
        compiled.append(candidate)
    return tuple(sorted(compiled, key=lambda row: (row.skill_id, row.revision_id)))


@dataclass(frozen=True)
class TransferGateResult:
    passed: bool
    reasons: tuple[str, ...]
    paired_tasks: int
    native_score_delta_mean: float | None
    cost_increase_fraction: float | None
    safety_regression: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "paired_tasks": self.paired_tasks,
            "native_score_delta_mean": self.native_score_delta_mean,
            "cost_increase_fraction": self.cost_increase_fraction,
            "safety_regression": self.safety_regression,
        }


def evaluate_transfer_gate(
    *,
    paired_evals: list[dict[str, Any]],
    expected_contract_sha256: str | None = None,
    expected_evaluator_epoch: str | None = None,
) -> TransferGateResult:
    """Apply the cross-task transfer gate to fresh-task paired evaluations."""
    if not paired_evals:
        return TransferGateResult(
            passed=False,
            reasons=("no_paired_evidence",),
            paired_tasks=0,
            native_score_delta_mean=None,
            cost_increase_fraction=None,
            safety_regression=False,
        )
    by_task: dict[str, dict[str, Any]] = {}
    for row in paired_evals:
        if not isinstance(row, dict) or "task_uid" not in row:
            raise SearchSkillBridgeError("paired evidence row must have task_uid")
        by_task.setdefault(row["task_uid"], {})[row["role"]] = row
    paired = [
        (task, rows["original"], rows["candidate"])
        for task, rows in by_task.items()
        if "original" in rows and "candidate" in rows
    ]
    reasons: list[str] = []
    if len(paired) < MIN_PAIRED_TASKS:
        reasons.append(f"paired_tasks_below_min:{len(paired)}")
    contracts = {
        row.get("matched_contract_sha256")
        for _task, original, candidate in paired
        for row in (original, candidate)
    }
    epochs = {
        row.get("native_evaluator_epoch")
        for _task, original, candidate in paired
        for row in (original, candidate)
    }
    if expected_contract_sha256 is not None and contracts != {expected_contract_sha256}:
        reasons.append("matched_contract_not_identical")
    if expected_evaluator_epoch is not None and epochs != {expected_evaluator_epoch}:
        reasons.append("evaluator_epoch_not_fixed")
    if paired:
        score_deltas = [
            candidate["native_score"] - original["native_score"]
            for _task, original, candidate in paired
        ]
        delta_mean = fmean(score_deltas)
        if delta_mean < -SCORE_TOLERANCE:
            reasons.append("native_score_regression_vs_original")
        families: dict[str, list[float]] = {}
        for _task, original, candidate in paired:
            families.setdefault(original.get("benchmark_family", ""), []).append(
                candidate["native_score"] - original["native_score"]
            )
        for family, deltas in sorted(families.items()):
            if fmean(deltas) < -1e-9:
                reasons.append(f"family_noninferior_failed:{family}")
        safety_regression = any(
            original.get("safety_passed") is True
            and candidate.get("safety_passed") is not True
            for _task, original, candidate in paired
        )
        if safety_regression:
            reasons.append("safety_regression")
        original_costs = [
            original["cost_units"] for _task, original, _candidate in paired
        ]
        candidate_costs = [
            candidate["cost_units"] for _task, _original, candidate in paired
        ]
        baseline_cost = fmean(original_costs)
        candidate_cost = fmean(candidate_costs)
        cost_fraction = (
            (candidate_cost - baseline_cost) / baseline_cost
            if baseline_cost > 0
            else 0.0
            if candidate_cost == 0
            else float("inf")
        )
        if cost_fraction > COST_INCREASE_LIMIT + 1e-12:
            reasons.append(f"cost_increase_over_limit:{cost_fraction:.4f}")
    else:
        delta_mean = None
        safety_regression = False
        cost_fraction = None
    return TransferGateResult(
        passed=not reasons,
        reasons=tuple(reasons),
        paired_tasks=len(paired),
        native_score_delta_mean=delta_mean,
        cost_increase_fraction=cost_fraction,
        safety_regression=safety_regression,
    )


def apply_transfer_gate(
    *,
    registry: SkillRegistry,
    candidate: SkillCandidate,
    gate_result: TransferGateResult,
    gate_evidence_path: Path,
) -> SkillCandidate:
    """Append-only transition: candidate -> transfer_verified / rejected."""
    if not gate_evidence_path.is_file():
        raise SearchSkillBridgeError("gate evidence file missing")
    evidence = SkillEvidenceRef.from_path(
        gate_evidence_path, role="cross_task_transfer_gate"
    )
    terminal = registry.transition(
        skill_id=candidate.skill_id,
        new_status="transfer_verified" if gate_result.passed else "rejected",
        reason=(
            "cross-task transfer gate passed (matched contract, native non-inferior, "
            "no safety regression, bounded cost)"
            if gate_result.passed
            else "cross-task transfer gate failed; candidate retained as rejected: "
            + ", ".join(gate_result.reasons)
        ),
        evidence_refs=(evidence,),
    )
    return terminal


def compile_local_lens_candidate(
    *,
    run_artifact: Path,
    registry_root: Path,
    skill_id: str,
    source_task_id: str,
    reusable_instruction: str,
    task_family: str = "local-lens-swe",
) -> SkillCandidate:
    """Compile a local-lens track run artifact into an inactive SkillCandidate.

    The run artifact must be the frozen RUN-ARTIFACT.json produced by
    local_lens_agent (tool_events + layer_records + patch).  layer_records are
    attached as jlens observation evidence; the candidate stays inactive.
    """
    run_artifact = run_artifact.resolve()
    if not run_artifact.is_file():
        raise SearchSkillBridgeError(f"missing run artifact: {run_artifact}")
    artifact = json.loads(run_artifact.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or "tool_events" not in artifact:
        raise SearchSkillBridgeError("run artifact must contain tool_events")
    if not reusable_instruction.strip():
        raise SearchSkillBridgeError("reusable_instruction is required")
    layer_count = sum(
        int(event.get("layer_record_count", 0) or 0)
        for event in artifact.get("tool_events", ())
    )
    if layer_count <= 0:
        raise SearchSkillBridgeError(
            "run artifact has no layer_records; refusing compile"
        )
    target_file = artifact.get("target_file", "")
    patch = artifact.get("patch", "")
    content = tuple(
        dict.fromkeys(
            [
                f"behavior: {reusable_instruction.strip()}",
                f"task: {source_task_id}",
                f"target: {target_file}" if target_file else "target: unknown",
                f"layer_records: {layer_count}",
                f"patch: {patch[:400]}" if patch else "patch: (empty)",
            ]
        )
    )
    evidence_refs = (
        SkillEvidenceRef.from_path(run_artifact, role="local_lens_run_artifact"),
    )
    candidate = SkillCandidate.create(
        skill_id=skill_id,
        revision_id=f"{skill_id}-{_sha({'content': content})[:12]}",
        parent_revision_id=None,
        status="candidate",
        status_reason=(
            "compiled from v2.5 local-lens run artifact; inactive project-local "
            "candidate (observational, jlens layer evidence)"
        ),
        task_family=task_family,
        content=content,
        source_task_ids=(source_task_id,),
        applicability={"required_semantics": [source_task_id, "local_lens_observed"]},
        counterexamples=("local_lens_negative",),
        known_failure_modes=("4b_quantized_hidden_states",),
        evidence_refs=evidence_refs,
        project_local_only=True,
        auto_install=False,
        active=False,
    )
    registry = SkillRegistry(registry_root)
    registry.append(candidate)
    registry.render_for_review(candidate.skill_id)
    return candidate
