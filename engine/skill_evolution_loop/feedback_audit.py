"""Offline, holdout-free audit of one complete Round 1 feedback request."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_json, sha256_json
from .evolution_catalog import EvolutionCatalog


def audit_feedback_request(
    *,
    request_path: Path,
    catalog: EvolutionCatalog,
    output_path: Path,
) -> dict[str, Any]:
    """Project feedback-only A/B evidence into a deduplicated next-step gate."""

    wrapper = _load_request(request_path.resolve())
    request = wrapper["request"]
    failures = request.get("failures")
    if not isinstance(failures, list) or not failures:
        raise ContractError("feedback audit requires non-empty feedback pairs")

    transitions: Counter[str] = Counter()
    mechanisms: dict[str, Counter[str]] = defaultdict(Counter)
    failure_reasons: dict[str, Counter[str]] = defaultdict(Counter)
    verified_gain_count = 0
    structural_degradation_count = 0
    for pair in failures:
        if not isinstance(pair, dict) or pair.get("mechanism") not in {
            "operator",
            "span",
        }:
            raise ContractError("feedback audit pair is invalid")
        mechanism = str(pair["mechanism"])
        baseline = _arm_state(pair.get("baseline"), "baseline")
        taught = _arm_state(pair.get("taught"), "taught")
        transition = f"{baseline['state']}->{taught['state']}"
        transitions[transition] += 1
        mechanisms[mechanism][transition] += 1
        failure_reasons["baseline"][baseline["reason"]] += 1
        failure_reasons["taught"][taught["reason"]] += 1
        if transition == "native-unresolved->native-resolved":
            verified_gain_count += 1
        if baseline["structural_valid"] and not taught["structural_valid"]:
            structural_degradation_count += 1

    recorded_gain = request.get("feedback_gain_count")
    if recorded_gain != verified_gain_count:
        raise ContractError("feedback audit gain count does not match bound pairs")
    if request.get("feedback_gain_gate_passed") is not (verified_gain_count > 0):
        raise ContractError("feedback audit gain gate does not match bound pairs")

    dedup_context = catalog.proposal_context(
        capability_tags=("localization", "patch-realization"),
        task_tags=("swe-bench",),
        failure_mode_tags=(
            "native-unresolved",
            "selector-no-match",
            "structural-invalid",
            "wrong-target",
        ),
    )
    decision = (
        {
            "next_step": "holdout-safety-evaluation",
            "parent_call_recommended": False,
            "new_skill_compilation_recommended": False,
            "reason": (
                "The strict feedback gain gate is already satisfied; preserve the "
                "frozen Skill and spend the next evidence budget on evaluator-only "
                "holdout safety coverage."
            ),
        }
        if verified_gain_count > 0
        else {
            "next_step": "feedback-strategy-dedup-review",
            "parent_call_recommended": True,
            "new_skill_compilation_recommended": False,
            "reason": (
                "No strict feedback gain is present; compare a proposed revision "
                "against the frozen catalog before compiling another inactive Skill."
            ),
        }
    )
    content = {
        "schema_version": 1,
        "request_file_sha256": _sha_file(request_path.resolve()),
        "request_sha256": wrapper["request_sha256"],
        "taskset_fingerprint": wrapper.get("taskset_fingerprint"),
        "native_summary_sha256": wrapper.get("native_summary_sha256"),
        "feedback_pair_count": len(failures),
        "feedback_gain_count_verified": verified_gain_count,
        "feedback_gain_gate_passed": verified_gain_count > 0,
        "teaching_structural_degradation_count": structural_degradation_count,
        "pair_transition_counts": dict(sorted(transitions.items())),
        "mechanism_transition_counts": {
            mechanism: dict(sorted(counts.items()))
            for mechanism, counts in sorted(mechanisms.items())
        },
        "condition_failure_counts": {
            arm: dict(sorted(counts.items()))
            for arm, counts in sorted(failure_reasons.items())
        },
        "dedup_context": dedup_context,
        "decision": decision,
        "holdout_cells_included": False,
        "source_holdout_evidence_present": False,
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    _freeze(output_path.resolve(), report)
    return report


def _arm_state(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"feedback audit {label} arm is invalid")
    structural = value.get("structural_valid")
    outcome = value.get("native_outcome")
    if not isinstance(structural, bool) or not isinstance(outcome, dict):
        raise ContractError(f"feedback audit {label} outcome is invalid")
    resolved = outcome.get("resolved")
    if not isinstance(resolved, bool):
        raise ContractError(f"feedback audit {label} resolved flag is invalid")
    if not structural:
        state = "structural-invalid"
        reason = str(value.get("failure_reason") or "structural-invalid")
    elif resolved:
        state = "native-resolved"
        reason = "native-resolved"
    else:
        state = "native-unresolved"
        reason = str(value.get("failure_reason") or "native-unresolved")
    return {"state": state, "reason": reason, "structural_valid": structural}


def _load_request(path: Path) -> dict[str, Any]:
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("feedback audit request is unreadable") from exc
    if not isinstance(wrapper, dict):
        raise ContractError("feedback audit request must be an object")
    content = {key: value for key, value in wrapper.items() if key != "evidence_sha256"}
    request = wrapper.get("request")
    if (
        wrapper.get("evidence_sha256") != sha256_json(content)
        or not isinstance(request, dict)
        or wrapper.get("request_sha256") != sha256_json(request)
        or request.get("request_type") != "round1-feedback-skill-evolution-v1"
        or wrapper.get("holdout_cells_included") is not False
        or wrapper.get("source_holdout_evidence_present") is not False
    ):
        raise ContractError("feedback audit request boundary is invalid")
    return wrapper


def _freeze(path: Path, report: dict[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError("feedback audit evidence is unreadable") from exc
        if existing != report:
            raise ContractError("frozen feedback audit does not match replay")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(report) + "\n", encoding="utf-8")


def _sha_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
