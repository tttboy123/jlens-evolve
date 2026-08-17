"""Read-only evaluator shadow cross-play with an immutable anchor epoch."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_STAGE = ROOT / "artifacts/v1.0.0/v0.6.0-evaluator-shadow"


class ShadowContractError(ValueError):
    """Raised when an evaluator shadow violates the frozen anchor contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def score_program(evaluator: dict[str, Any], program: dict[str, Any]) -> dict[str, Any]:
    weights = [float(value) for value in evaluator["weights"]]
    outcomes = list(program["outcomes"])
    if not weights or len(weights) != len(outcomes) or sum(weights) <= 0:
        raise ShadowContractError("evaluator/program dimension mismatch")
    if any(weight < 0 for weight in weights):
        raise ShadowContractError("evaluator weights must be non-negative")
    score = sum(weight * bool(outcome) for weight, outcome in zip(weights, outcomes))
    score /= sum(weights)
    return {
        "score": score,
        "admitted": score >= float(evaluator["threshold"]),
    }


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average = (position + 1 + end) / 2
        for index in order[position:end]:
            ranks[index] = average
        position = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ShadowContractError("rank correlation requires equal non-trivial inputs")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    covariance = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks)
    )
    left_variance = sum((value - left_mean) ** 2 for value in left_ranks)
    right_variance = sum((value - right_mean) ** 2 for value in right_ranks)
    if left_variance == 0 or right_variance == 0:
        return 0.0
    return covariance / (left_variance * right_variance) ** 0.5


def _evaluate_all(
    evaluator: dict[str, Any], programs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    evaluator_hash = _sha256_json(evaluator)
    return [
        {
            "evaluator_id": evaluator["evaluator_id"],
            "evaluator_hash": evaluator_hash,
            "program_id": program["program_id"],
            "program_hash": _sha256_json(program),
            **score_program(evaluator, program),
        }
        for program in programs
    ]


def run_evaluator_shadow(
    *,
    config_path: Path,
    evaluator_path: Path,
    corpus_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    evaluators = json.loads(evaluator_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if config.get("system_version") != "0.6.0":
        raise ShadowContractError("shadow experiment must use system_version 0.6.0")
    if config.get("shadow_used_for_admission") is not False:
        raise ShadowContractError("shadow evaluator cannot be admission authority")
    if config.get("auto_evaluator_promotion") is not False:
        raise ShadowContractError("automatic evaluator promotion is forbidden")
    if config.get("model_calls") != 0 or config.get("network_calls") != 0:
        raise ShadowContractError("deterministic shadow cannot call model or network")

    anchor = evaluators["anchor"]
    candidates = list(evaluators["candidates"])
    programs = list(corpus["programs"])
    if anchor["evaluator_id"] != config["active_evaluator_id"]:
        raise ShadowContractError("active evaluator does not match frozen anchor")
    case_count = len(evaluators["case_ids"])
    if any(len(program["outcomes"]) != case_count for program in programs):
        raise ShadowContractError("program corpus outcome dimension mismatch")
    if any(len(row["weights"]) != case_count for row in [anchor, *candidates]):
        raise ShadowContractError("evaluator weight dimension mismatch")
    hashes_before = {
        "config": _sha256_file(config_path),
        "evaluators": _sha256_file(evaluator_path),
        "corpus": _sha256_file(corpus_path),
    }
    active_before = {
        "evaluator_id": anchor["evaluator_id"],
        "evaluator_hash": _sha256_json(anchor),
        "epoch": config["evaluator_epoch"],
    }
    anchor_rows = _evaluate_all(anchor, programs)
    anchor_truth = {row["program_id"]: row["admitted"] for row in anchor_rows}
    if anchor_truth != corpus["truth"]["expected_admission"]:
        raise ShadowContractError("anchor truth mismatch in frozen program corpus")
    if corpus["truth"]["anchor_evaluator_id"] != anchor["evaluator_id"]:
        raise ShadowContractError("anchor truth authority mismatch")

    anchor_view = [
        {
            "program_id": row["program_id"],
            "score": row["score"],
            "admitted": row["admitted"],
        }
        for row in anchor_rows
    ]
    anchor_shadow_on_rows = _evaluate_all(anchor, programs)
    anchor_shadow_on_view = [
        {
            "program_id": row["program_id"],
            "score": row["score"],
            "admitted": row["admitted"],
        }
        for row in anchor_shadow_on_rows
    ]
    cross_play = list(anchor_rows)
    candidate_results: dict[str, Any] = {}
    proposal_candidate = None
    anchor_scores = [row["score"] for row in anchor_rows]
    anchor_max = max(anchor_scores)
    anchor_champions = {
        row["program_id"] for row in anchor_rows if row["score"] == anchor_max
    }
    for candidate in candidates:
        rows = _evaluate_all(candidate, programs)
        cross_play.extend(rows)
        false_accepts = [
            row["program_id"]
            for row in rows
            if row["admitted"] and not anchor_truth[row["program_id"]]
        ]
        false_rejects = [
            row["program_id"]
            for row in rows
            if not row["admitted"] and anchor_truth[row["program_id"]]
        ]
        scores = [row["score"] for row in rows]
        candidate_max = max(scores)
        candidate_champions = {
            row["program_id"] for row in rows if row["score"] == candidate_max
        }
        correlation = _spearman(anchor_scores, scores)
        champion_stable = candidate_champions == anchor_champions
        if false_accepts:
            status = "rejected_false_accept"
        elif false_rejects:
            status = "rejected_false_reject"
        elif champion_stable and correlation >= float(config["rank_correlation_gate"]):
            status = "review_proposed"
        else:
            status = "rejected_drift"
        candidate_results[candidate["evaluator_id"]] = {
            "status": status,
            "false_accepts": len(false_accepts),
            "false_accept_programs": false_accepts,
            "false_rejects": len(false_rejects),
            "false_reject_programs": false_rejects,
            "rank_correlation": correlation,
            "champion_stable": champion_stable,
            "champions": sorted(candidate_champions),
            "evaluator_hash": _sha256_json(candidate),
            "auto_promoted": False,
        }
        if status == "review_proposed":
            if proposal_candidate is not None:
                raise ShadowContractError("multiple review proposals are not supported")
            proposal_candidate = candidate["evaluator_id"]

    review_proposal = (
        {
            "candidate_id": proposal_candidate,
            "requires_human_review": True,
            "activation_allowed": False,
            "epoch_boundary_only": True,
            "auto_promoted": False,
        }
        if proposal_candidate
        else None
    )
    events: list[dict[str, Any]] = []
    for shadow_mode, admission_rows in (
        ("disabled", anchor_rows),
        ("enabled", anchor_shadow_on_rows),
    ):
        for row in admission_rows:
            events.append(
                {
                    "event_type": "admission",
                    "shadow_mode": shadow_mode,
                    "program_id": row["program_id"],
                    "score": row["score"],
                    "admitted": row["admitted"],
                    "authority": anchor["evaluator_id"],
                    "evaluator_epoch": config["evaluator_epoch"],
                }
            )
    for row in cross_play[len(anchor_rows) :]:
        events.append(
            {
                "event_type": "shadow_score",
                "program_id": row["program_id"],
                "evaluator_id": row["evaluator_id"],
                "score": row["score"],
                "admitted_if_active": row["admitted"],
                "used_for_admission": False,
            }
        )
    active_after = dict(active_before)
    hashes_after = {
        "config": _sha256_file(config_path),
        "evaluators": _sha256_file(evaluator_path),
        "corpus": _sha256_file(corpus_path),
    }
    contract_checks = {
        "anchor_shadow_off_on_equal": anchor_view == anchor_shadow_on_view,
        "cross_play_complete": len(cross_play) == (1 + len(candidates)) * len(programs),
        "shadow_never_used_for_admission": all(
            event["authority"] == anchor["evaluator_id"]
            for event in events
            if event["event_type"] == "admission"
        )
        and all(
            event["used_for_admission"] is False
            for event in events
            if event["event_type"] == "shadow_score"
        ),
        "active_evaluator_unchanged": active_before == active_after,
        "inputs_unchanged": hashes_before == hashes_after,
        "candidate_statuses_match_predeclaration": all(
            candidate_results[row["evaluator_id"]]["status"] == row["expected_status"]
            for row in candidates
        ),
        "review_proposal_has_no_activation_authority": review_proposal is not None
        and review_proposal["activation_allowed"] is False
        and review_proposal["auto_promoted"] is False,
    }
    accepted = all(contract_checks.values())
    stable = {
        "hashes": hashes_before,
        "active_evaluator_before": active_before,
        "active_evaluator_after": active_after,
        "anchor_shadow_off": anchor_view,
        "anchor_shadow_on": anchor_shadow_on_view,
        "candidates": candidate_results,
        "review_proposal": review_proposal,
        "contract_checks": contract_checks,
    }
    result = {
        "schema_version": 1,
        "stage": "v0.6.0-evaluator-shadow",
        **stable,
        "decision": "accepted" if accepted else "rejected",
        "experiment_fingerprint": hashlib.sha256(
            _canonical_json(stable).encode("utf-8")
        ).hexdigest(),
        "claims": {
            "auto_evaluator_promotion": False,
            "shadow_used_for_admission": False,
            "active_epoch_changed": False,
            "production_ready": False,
        },
    }
    for sequence, event in enumerate(events, 1):
        event["sequence"] = sequence
    _atomic_json(output_dir / "cross-play.json", cross_play)
    _atomic_json(output_dir / "disagreement.json", candidate_results)
    _atomic_json(output_dir / "review-proposal.json", review_proposal)
    _atomic_text(
        output_dir / "events.jsonl",
        "".join(_canonical_json(event) + "\n" for event in events),
    )
    _atomic_json(
        output_dir / "evidence.json",
        {
            "active_before": active_before,
            "active_after": active_after,
            "contract_checks": contract_checks,
            "anchor_truth": anchor_truth,
            "review_proposal": review_proposal,
        },
    )
    _atomic_json(output_dir / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_STAGE / "configs/experiment.json"
    )
    parser.add_argument(
        "--evaluators",
        type=Path,
        default=DEFAULT_STAGE / "configs/evaluators.json",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_STAGE / "configs/program-corpus.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_evaluator_shadow(
        config_path=args.config,
        evaluator_path=args.evaluators,
        corpus_path=args.corpus,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "experiment_fingerprint": result["experiment_fingerprint"],
                "candidate_statuses": {
                    key: value["status"] for key, value in result["candidates"].items()
                },
                "review_proposal": result["review_proposal"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
