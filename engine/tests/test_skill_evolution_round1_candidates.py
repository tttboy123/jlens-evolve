from __future__ import annotations

import json
from pathlib import Path

from skill_evolution_loop.contracts import canonical_json
from skill_evolution_loop.round1_candidates import freeze_round1_candidates


def test_round1_candidates_exclude_opened_and_frozen_partitions(tmp_path: Path) -> None:
    records = []
    for index, partition in enumerate(["search", "search", "promotion"]):
        instance_id = f"org__repo-{index}"
        contract = {
            "benchmark_id": "swe-bench-verified",
            "benchmark_revision": "b" * 40,
            "base_commit": f"{index + 1:040x}",
            "repo": "org/repo",
            "language": "python",
            "content_sha256": "c" * 64,
        }
        records.append(
            {
                "assigned_partition": partition,
                "state": "unopened",
                "instance_id": instance_id,
                "task_uid": f"{index + 1:064x}",
                "identity_fingerprint": f"{index + 4:064x}",
                "task_contract": contract,
            }
        )
    pool = tmp_path / "TASK_POOL.json"
    pool.write_text(canonical_json({"records": records}), encoding="utf-8")
    opened = tmp_path / "TASKSET.json"
    opened.write_text(
        json.dumps({"tasks": [{"instance_id": "org__repo-1"}]}), encoding="utf-8"
    )

    report = freeze_round1_candidates(
        task_pool_path=pool,
        opened_taskset_paths=(opened,),
        output_path=tmp_path / "CANDIDATES.json",
    )

    assert report["task_count"] == 1
    assert report["tasks"][0]["instance_id"] == "org__repo-0"
    assert report["promotion_partition_opened"] is False
    assert report["final_sealed_partition_opened"] is False
    assert report["answer_fields_read"] is False
