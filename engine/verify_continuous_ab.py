"""Fail-closed verifier for the frozen v2.1.0 continuous A/B stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark_adapters import TaskPool
from benchmark_catalog import PINNED_SOURCES
from continuous_ab import ChangeSetRegistry, PermanentBaselineAuthority
from continuous_ab_service import ContinuousABService
from freeze_benchmark_pool import (
    EXPECTED_SOURCE_COUNTS,
    PARTITION_QUOTAS,
    PREVIOUSLY_OPENED_INSTANCE_IDS,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_stage(stage_dir: Path) -> dict[str, Any]:
    stage_dir = stage_dir.resolve()
    pool_root = stage_dir / "configs/benchmark-pool"
    frozen_pool = TaskPool.load(pool_root / "TASK_POOL.json")
    summary = json.loads((pool_root / "POOL_SUMMARY.json").read_text(encoding="utf-8"))
    catalog = json.loads(
        (pool_root / "BENCHMARK_CATALOG.json").read_text(encoding="utf-8")
    )
    adapter_contracts = json.loads(
        (pool_root / "ADAPTER_CONTRACTS.json").read_text(encoding="utf-8")
    )
    source_manifest = json.loads(
        (pool_root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
    )
    source_hashes_valid = True
    for item in source_manifest["files"]:
        path = pool_root / "inputs" / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or _sha256_file(path) != item["sha256"]
        ):
            source_hashes_valid = False
            break

    runtime_root = stage_dir / "runs/ready/runtime"
    service = ContinuousABService.load(runtime_root)
    runtime_pool = TaskPool.load(runtime_root / "TASK_POOL.json")
    baseline = PermanentBaselineAuthority(
        runtime_root / "permanent-baseline.json"
    ).load()
    changesets = ChangeSetRegistry.load(runtime_root / "changesets.json")
    seed_candidate = json.loads(
        (stage_dir / "configs/SEED_CANDIDATE.json").read_text(encoding="utf-8")
    )
    pilot_plan = json.loads(
        (stage_dir / "configs/PILOT_PLAN.json").read_text(encoding="utf-8")
    )
    pilot_preflight = json.loads(
        (stage_dir / "PILOT-PREFLIGHT.json").read_text(encoding="utf-8")
    )
    integrity_pass3 = json.loads(
        (
            stage_dir
            / "runs/protocol-smoke-pass3-integrity-authoritative/PASS3_AGGREGATE.json"
        ).read_text(encoding="utf-8")
    )
    execution_bridge = json.loads(
        (stage_dir / "evidence/EXECUTION-BRIDGE-VERIFICATION.json").read_text(
            encoding="utf-8"
        )
    )
    frozen_source_root = stage_dir / "source-snapshot"
    execution_bridge_sources_match = all(
        (frozen_source_root / relative).is_file()
        and _sha256_file(frozen_source_root / relative) == expected_sha256
        for relative, expected_sha256 in execution_bridge["source_sha256"].items()
    )

    partitions = Counter(record.assigned_partition for record in frozen_pool.records)
    adapters = Counter(record.benchmark_id for record in frozen_pool.records)
    expected_adapter_counts = {
        adapter_id: sum(quotas.values())
        for adapter_id, quotas in PARTITION_QUOTAS.items()
    }
    task_contract_keys = {
        key for record in frozen_pool.records for key in record.task_contract
    }
    commands = [
        tuple(value["sample_command"])
        for value in adapter_contracts["adapters"].values()
    ]
    staged_inputs_valid = True
    for adapter_id, value in adapter_contracts["adapters"].items():
        staging = value["dataset_staging"]
        source = PINNED_SOURCES[adapter_id]
        if value["harness_checkout"]["revision"] != source.harness_revision:
            staged_inputs_valid = False
        if staging["method"] == "copy_hashed_local_jsonl":
            path = pool_root / staging["path"]
            if (
                not path.is_file()
                or path.stat().st_size != staging["bytes"]
                or _sha256_file(path) != staging["sha256"]
            ):
                staged_inputs_valid = False
        elif staging.get("revision", staging.get("dataset_revision")) != (
            source.dataset_revision
        ):
            staged_inputs_valid = False
    stream_by_partition = {
        partition: [
            record.benchmark_id
            for record in frozen_pool.records
            if record.assigned_partition == partition
        ]
        for partition in ("search", "promotion", "final_sealed")
    }
    expected_cycle = sorted(PINNED_SOURCES)
    terminal_command = adapter_contracts["adapters"]["terminal-bench-2"][
        "sample_command"
    ]
    first_search = [
        record.task_uid
        for record in frozen_pool.records
        if record.assigned_partition == "search"
    ][:10]
    checks = {
        "source_inputs_hashed": source_hashes_valid,
        "source_counts_frozen": catalog["source_counts"] == EXPECTED_SOURCE_COUNTS,
        "source_revisions_pinned": all(
            catalog["sources"][adapter_id]["dataset_revision"]
            == source.dataset_revision
            and catalog["sources"][adapter_id]["harness_revision"]
            == source.harness_revision
            for adapter_id, source in PINNED_SOURCES.items()
        ),
        "four_native_adapters": set(adapters) == set(PINNED_SOURCES),
        "stratified_75_each": dict(adapters) == expected_adapter_counts,
        "partition_counts": dict(partitions)
        == {"search": 160, "promotion": 80, "final_sealed": 60},
        "all_300_frozen_tasks_unopened": len(frozen_pool.records) == 300
        and all(record.state == "unopened" for record in frozen_pool.records),
        "all_prior_opened_tasks_retired_from_pool": {
            item["instance_id"] for item in frozen_pool.retired_exclusions
        }
        == set(PREVIOUSLY_OPENED_INSTANCE_IDS)
        and not (
            {record.instance_id for record in frozen_pool.records}
            & set(PREVIOUSLY_OPENED_INSTANCE_IDS)
        ),
        "gold_content_absent_from_scheduler_contract": not (
            {"patch", "fix_patch", "test_patch", "problem_statement"}
            & task_contract_keys
        ),
        "commands_are_argv_not_shell": all(
            command and all(";" not in token for token in command)
            for command in commands
        ),
        "harnesses_and_runtime_inputs_are_pinned": staged_inputs_valid,
        "all_streams_start_round_robin": all(
            stream[:8] == expected_cycle * 2 for stream in stream_by_partition.values()
        ),
        "runtime_pool_matches_frozen_pool": runtime_pool.to_dict()[
            "semantic_fingerprint"
        ]
        == frozen_pool.to_dict()["semantic_fingerprint"],
        "runtime_has_zero_completed_rounds": service.completed_round_count == 0,
        "service_manifest_tamper_evident": isinstance(
            service.manifest.get("integrity_sha256"), str
        )
        and len(service.manifest["integrity_sha256"]) == 64,
        "baseline_is_immutable": baseline["immutable"] is True
        and baseline["freeze_count"] == 1,
        "baseline_is_registry_parent": baseline["baseline"]["agent_program_sha256"]
        == changesets.payload["initial_agent_sha256"]
        == changesets.active_agent_sha256
        == seed_candidate["parent_agent_sha256"],
        "seed_candidate_not_promoted": seed_candidate["global_apply"] is False
        and seed_candidate["status"] == "carry_forward_candidate_not_promoted"
        and not changesets.payload["proposals"],
        "changeset_cadence_is_ten": changesets.payload["cadence_interval"] == 10,
        "changeset_registry_tamper_evident": isinstance(
            changesets.payload.get("integrity_sha256"), str
        )
        and len(changesets.payload["integrity_sha256"]) == 64,
        "terminal_prediction_boundary_wrapper": (
            "evolve_jlens_harbor:FrozenCodexAgent" in terminal_command
            and "--model" in terminal_command
            and "harness_revision="
            + PINNED_SOURCES["terminal-bench-2"].harness_revision
            in terminal_command
        ),
        "pilot_predeclared_without_opening_tasks": pilot_plan["matched_rounds"] == 10
        and pilot_plan["maximum_agent_calls"] == 20
        and [row["task_uid"] for row in pilot_plan["rounds"]] == first_search
        and all(row["partition"] == "search" for row in pilot_plan["rounds"]),
        "pilot_paid_actions_human_required": pilot_preflight["status"]
        == "HUMAN_REQUIRED"
        and pilot_preflight["paid_actions_dispatched"] is False,
        "integrity_pass3_stable": integrity_pass3["status"] == "pass3_verified"
        and integrity_pass3["stable_outcome"] is True
        and integrity_pass3["quality_claim"] == "none",
        "execution_bridge_covers_all_adapters": execution_bridge["status"] == "verified"
        and set(execution_bridge["adapters_covered"]) == set(PINNED_SOURCES)
        and all(execution_bridge["checks"].values())
        and execution_bridge["tests_passed"] == 31,
        "execution_bridge_is_zero_cost_only": execution_bridge["formal_agent_calls"]
        == 0
        and execution_bridge["native_evaluator_invocations"] == 0
        and execution_bridge["matched_quality_rounds"] == 0
        and execution_bridge["quality_claim"] == "none",
        "execution_bridge_sources_match": execution_bridge_sources_match,
        "final_sealed_not_opened": not (
            runtime_root / "FINAL_SEALED_AUDIT.json"
        ).exists(),
        "summary_matches_pool": summary["selected_task_count"]
        == len(frozen_pool.records),
    }
    facts = {
        "source_task_count": sum(EXPECTED_SOURCE_COUNTS.values()),
        "selected_task_count": len(frozen_pool.records),
        "duplicate_count": len(frozen_pool.duplicates),
        "prior_opened_exclusion_count": len(frozen_pool.retired_exclusions),
        "final_sealed_unopened": sum(
            record.assigned_partition == "final_sealed" and record.state == "unopened"
            for record in frozen_pool.records
        ),
        "completed_rounds": service.completed_round_count,
        "baseline_agent_sha256": baseline["baseline"]["agent_program_sha256"],
        "seed_candidate_sha256": seed_candidate["candidate_agent_sha256"],
        "execution_bridge_tests_passed": execution_bridge["tests_passed"],
    }
    return {
        "schema_version": "1.0",
        "status": "verified" if all(checks.values()) else "rejected",
        "checks": checks,
        "facts": facts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_stage(args.stage_dir)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
