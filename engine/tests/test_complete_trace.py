from __future__ import annotations

from complete_trace import recover_trace


def test_recover_trace_fills_checkpoint_backed_iteration():
    raw = [
        {
            "iteration": 1,
            "parent_id": "p0",
            "child_id": "p1",
            "parent_code": "old",
            "child_code": "middle",
        }
    ]
    programs = {
        "p1": {
            "id": "p1",
            "code": "middle",
            "parent_id": "p0",
            "iteration_found": 1,
        },
        "p2": {
            "id": "p2",
            "code": "new",
            "parent_id": "p1",
            "iteration_found": 2,
            "timestamp": 2.0,
            "generation": 2,
            "metrics": {"combined_score": 0.5},
            "metadata": {
                "parent_metrics": {"combined_score": 0.25},
                "island": 0,
                "changes": "Full rewrite",
            },
            "prompts": {
                "full_rewrite_user": {
                    "system": "system",
                    "user": "user",
                    "responses": ["response"],
                }
            },
            "artifacts_json": "{}",
        },
    }

    complete, recovered = recover_trace(raw, programs, last_iteration=2)

    assert recovered == [2]
    assert [row["iteration"] for row in complete] == [1, 2]
    assert complete[1]["parent_code"] == "middle"
    assert complete[1]["child_code"] == "new"
    assert complete[1]["improvement_delta"]["combined_score"] == 0.25
    assert complete[1]["metadata"]["recovered_from_checkpoint"] is True
