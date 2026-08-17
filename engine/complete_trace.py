#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover trace rows omitted by OpenEvolve from checkpoint records."
    )
    parser.add_argument(
        "--trace", type=Path, default=EXPERIMENT_DIR / "runs/main/evolution_trace.jsonl"
    )
    parser.add_argument(
        "--checkpoints", type=Path, default=EXPERIMENT_DIR / "runs/main/checkpoints"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_DIR / "analysis/evolution_trace_complete.jsonl",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=EXPERIMENT_DIR / "analysis/trace_repair_audit.json",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def checkpoint_programs(checkpoints: Path) -> tuple[dict[str, dict[str, Any]], int]:
    programs: dict[str, dict[str, Any]] = {}
    last_iterations: list[int] = []
    for metadata_path in checkpoints.glob("checkpoint_*/metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        last_iterations.append(int(metadata["last_iteration"]))
    for path in checkpoints.glob("checkpoint_*/programs/*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        program_id = str(row["id"])
        if program_id in programs:
            for field in ("code", "parent_id", "iteration_found", "metrics"):
                if programs[program_id].get(field) != row.get(field):
                    raise ValueError(
                        f"conflicting checkpoint program {program_id} field {field}"
                    )
        else:
            programs[program_id] = row
    if not last_iterations:
        raise ValueError("no checkpoint metadata found")
    return programs, max(last_iterations)


def recover_trace(
    raw_rows: list[dict[str, Any]],
    programs: dict[str, dict[str, Any]],
    *,
    last_iteration: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    by_iteration = {int(row["iteration"]): row for row in raw_rows}
    if len(by_iteration) != len(raw_rows):
        raise ValueError("raw trace contains duplicate iterations")
    code_by_id = {
        str(row[f"{role}_id"]): str(row[f"{role}_code"])
        for row in raw_rows
        for role in ("parent", "child")
    }
    code_by_id.update(
        {program_id: str(program["code"]) for program_id, program in programs.items()}
    )
    missing = sorted(set(range(1, last_iteration + 1)) - by_iteration.keys())
    for iteration in missing:
        candidates = [
            program
            for program in programs.values()
            if int(program.get("iteration_found", -1)) == iteration
        ]
        candidate_ids = {str(program["id"]) for program in candidates}
        if len(candidate_ids) != 1:
            raise ValueError(
                f"iteration {iteration} has {len(candidate_ids)} checkpoint candidates"
            )
        child = candidates[0]
        parent_id = str(child["parent_id"])
        if parent_id not in code_by_id:
            raise ValueError(f"missing parent code for recovered iteration {iteration}")
        parent_metrics = dict(child.get("metadata", {}).get("parent_metrics", {}))
        child_metrics = dict(child["metrics"])
        if not parent_metrics:
            raise ValueError(
                f"missing parent metrics for recovered iteration {iteration}"
            )
        prompt_record = child.get("prompts", {}).get("full_rewrite_user", {})
        responses = list(prompt_record.get("responses", []))
        if not responses:
            raise ValueError(
                f"missing LLM response for recovered iteration {iteration}"
            )
        artifacts_raw = child.get("artifacts_json")
        artifacts = json.loads(artifacts_raw) if artifacts_raw else {}
        parent = programs.get(parent_id, {})
        row = {
            "iteration": iteration,
            "timestamp": float(child["timestamp"]),
            "parent_id": parent_id,
            "child_id": str(child["id"]),
            "parent_metrics": parent_metrics,
            "child_metrics": child_metrics,
            "parent_code": code_by_id[parent_id],
            "child_code": str(child["code"]),
            "parent_changes_description": str(parent.get("changes_description") or ""),
            "prompt": {
                "system": str(prompt_record.get("system", "")),
                "user": str(prompt_record.get("user", "")),
            },
            "llm_response": str(responses[0]),
            "improvement_delta": {
                metric: float(child_metrics.get(metric, 0.0))
                - float(parent_metrics.get(metric, 0.0))
                for metric in sorted(parent_metrics.keys() | child_metrics.keys())
            },
            "island_id": int(child.get("metadata", {}).get("island", -1)),
            "generation": int(child.get("generation", -1)),
            "artifacts": artifacts,
            "metadata": {
                "changes": child.get("metadata", {}).get("changes"),
                "recovered_from_checkpoint": True,
            },
        }
        by_iteration[iteration] = row
    complete = [by_iteration[iteration] for iteration in range(1, last_iteration + 1)]
    return complete, missing


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    raw_rows = read_jsonl(args.trace)
    programs, last_iteration = checkpoint_programs(args.checkpoints)
    complete, recovered = recover_trace(
        raw_rows, programs, last_iteration=last_iteration
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in complete),
        encoding="utf-8",
    )
    audit = {
        "raw_trace": str(args.trace),
        "raw_trace_sha256": sha256(args.trace),
        "raw_rows": len(raw_rows),
        "checkpoint_last_iteration": last_iteration,
        "recovered_iterations": recovered,
        "complete_rows": len(complete),
        "complete_trace": str(args.output),
        "complete_trace_sha256": sha256(args.output),
        "all_iterations_present": [row["iteration"] for row in complete]
        == list(range(1, last_iteration + 1)),
    }
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
