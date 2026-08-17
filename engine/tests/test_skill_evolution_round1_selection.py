from __future__ import annotations

import hashlib
import json
from pathlib import Path

from skill_evolution_loop.contracts import canonical_json, sha256_json
from skill_evolution_loop.round1_selection import freeze_round1_selection


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    records = []
    for number in range(60):
        uid = f"uid-{number:02d}"
        instance = f"repo__repo-{number}"
        records.append(
            {
                "task_uid": uid,
                "instance_id": instance,
                "benchmark_id": "swe-bench-verified",
                "identity_fingerprint": f"identity-{number}",
                "assigned_partition": "search",
                "state": "unopened",
                "task_contract": {
                    "benchmark_revision": "a" * 40,
                    "base_commit": "b" * 40,
                    "repo": "repo/repo",
                    "language": "python" if number < 30 else "rust",
                    "content_sha256": f"content-{number}",
                },
            }
        )
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"records": records}), encoding="utf-8")
    pool_sha = hashlib.sha256(pool.read_bytes()).hexdigest()
    content = {
        "status": "ready",
        "target_tasks": 60,
        "selected_task_uids": [row["task_uid"] for row in records],
        "source_task_pool_sha256": pool_sha,
        "runtime_opened_instance_ids": [],
        "promotion_partition_opened": False,
        "final_sealed_partition_opened": False,
    }
    readiness = tmp_path / "readiness.json"
    readiness.write_text(
        canonical_json({**content, "evidence_sha256": sha256_json(content)}) + "\n",
        encoding="utf-8",
    )
    return readiness, pool


def test_round1_selection_freezes_balanced_search_only_routes(tmp_path: Path) -> None:
    readiness, pool = _inputs(tmp_path)

    result = freeze_round1_selection(
        readiness_path=readiness,
        task_pool_path=pool,
        output_path=tmp_path / "selection.json",
    )

    assert result["task_count"] == 60
    assert result["cohort_counts"] == {"feedback": 30, "holdout": 30}
    assert result["mechanism_counts"] == {"operator": 30, "span": 30}
    assert result["gold_fields_included"] is False
    assert result["promotion_partition_opened"] is False
    assert result["final_sealed_partition_opened"] is False
    assert len({row["task_id"] for row in result["tasks"]}) == 60
