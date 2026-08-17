from __future__ import annotations

from novelty_proxy import (
    ProposalController,
    StructuredMutationController,
    detect_stagnation,
)


def _response(code: str, response_id: str) -> dict:
    return {
        "id": response_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"```python\n{code}\n```",
                },
                "finish_reason": "stop",
            }
        ],
    }


def _payload() -> dict:
    return {
        "model": "local-model",
        "messages": [
            {
                "role": "user",
                "content": "Current program:\n```python\ndef solve(rows):\n    return []\n```",
            }
        ],
    }


def test_duplicate_aware_controller_selects_bounded_retry_with_fixed_calls():
    responses = iter(
        [
            _response("def solve(rows):\n    # same AST\n    return []", "first"),
            _response("def solve(rows):\n    return list(rows)", "second"),
        ]
    )
    calls: list[dict] = []

    def forward(payload: dict) -> dict:
        calls.append(payload)
        return next(responses)

    controller = ProposalController(mode="duplicate-aware")
    selected, audit = controller.process_chat(_payload(), forward)

    assert len(calls) == 2
    assert selected["id"] == "second"
    assert audit["first_duplicate"] is True
    assert audit["selected_index"] == 2
    assert audit["retry_feedback_added"] is True
    retry_prompt = calls[1]["messages"][-1]["content"].lower()
    assert "duplicate-aware retry" in retry_prompt
    assert "hidden" not in retry_prompt
    assert "holdout" not in retry_prompt


def test_duplicate_aware_controller_keeps_first_novel_candidate_but_shadows_second():
    responses = iter(
        [
            _response("def solve(rows):\n    return list(rows)", "first"),
            _response("def solve(rows):\n    return sorted(rows)", "shadow"),
        ]
    )
    calls: list[dict] = []

    def forward(payload: dict) -> dict:
        calls.append(payload)
        return next(responses)

    selected, audit = ProposalController(mode="duplicate-aware").process_chat(
        _payload(), forward
    )

    assert len(calls) == 2
    assert selected["id"] == "first"
    assert audit["first_duplicate"] is False
    assert audit["selected_index"] == 1
    assert audit["retry_feedback_added"] is False
    assert calls[1] == calls[0]


def test_shadow_control_always_returns_first_and_never_changes_prompt():
    responses = iter(
        [
            _response("def solve(rows):\n    return []", "first"),
            _response("def solve(rows):\n    return list(rows)", "shadow"),
        ]
    )
    calls: list[dict] = []

    def forward(payload: dict) -> dict:
        calls.append(payload)
        return next(responses)

    selected, audit = ProposalController(mode="shadow-control").process_chat(
        _payload(), forward
    )

    assert len(calls) == 2
    assert selected["id"] == "first"
    assert audit["first_duplicate"] is True
    assert audit["selected_index"] == 1
    assert audit["retry_feedback_added"] is False
    assert calls[1] == calls[0]


def test_stagnation_detector_uses_only_recent_public_candidate_events():
    stalled = [
        {
            "event_type": "candidate",
            "accepted": False,
            "parent_score": 0.8,
            "child_score": 0.8,
        }
        for _ in range(3)
    ]
    recovered = [
        *stalled[:2],
        {
            "event_type": "candidate",
            "accepted": True,
            "parent_score": 0.8,
            "child_score": 0.9,
        },
    ]

    assert detect_stagnation(stalled, window=3)["active"] is True
    assert detect_stagnation(recovered, window=3)["active"] is False


def test_stagnation_detector_does_not_count_recovering_to_global_best():
    events = [
        {
            "event_type": "candidate",
            "accepted": True,
            "parent_score": 0.2,
            "child_score": 0.8,
        },
        {
            "event_type": "candidate",
            "accepted": True,
            "parent_score": 0.2,
            "child_score": 0.8,
        },
        {
            "event_type": "candidate",
            "accepted": True,
            "parent_score": 0.5,
            "child_score": 0.8,
        },
        {
            "event_type": "candidate",
            "accepted": True,
            "parent_score": 0.7,
            "child_score": 0.8,
        },
    ]

    result = detect_stagnation(events, window=3)

    assert result["recent_improvements"] == 0
    assert result["active"] is True


def test_duplicate_aware_controller_selects_escape_candidate_during_stagnation():
    responses = iter(
        [
            _response("def solve(rows):\n    return list(rows)", "first"),
            _response("def solve(rows):\n    return sorted(rows)", "escape"),
        ]
    )
    calls: list[dict] = []

    def forward(payload: dict) -> dict:
        calls.append(payload)
        return next(responses)

    selected, audit = ProposalController(mode="duplicate-aware").process_chat(
        _payload(),
        forward,
        stagnation={"active": True, "window": 3, "recent_improvements": 0},
    )

    assert len(calls) == 2
    assert selected["id"] == "escape"
    assert audit["selection_trigger"] == "search_stagnation"
    assert audit["stagnation_active"] is True
    assert "search_stagnation" in calls[1]["messages"][-1]["content"]


def _structured_payload() -> dict:
    source = """def solve(records):
    return [row for row in records if row.get("status") == "paid"]
"""
    return {
        "model": "local-model",
        "messages": [
            {
                "role": "user",
                "content": (
                    "### target_failure\n"
                    "```\n{'id': 'filter_normalized_status'}\n```\n\n"
                    f"Current program:\n```python\n{source}\n```"
                ),
            }
        ],
    }


def test_structured_controller_falls_back_when_repair_removes_operator():
    old_source = """def solve(records):
    return [row for row in records if row.get("status") == "paid"]
"""
    responses = iter(
        [
            {
                "id": "plan",
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"schema_version":1,'
                                '"operator_id":"canonicalize_before_predicate",'
                                '"target_symbol":"solve",'
                                '"public_failure":"filter_normalized_status",'
                                '"preserve":[]}'
                            )
                        }
                    }
                ],
            },
            _response(old_source, "repair-without-operator"),
        ]
    )
    calls: list[dict] = []

    def forward(payload: dict) -> dict:
        calls.append(payload)
        return next(responses)

    selected, audit = StructuredMutationController(
        mode="structured-mutation"
    ).process_chat(_structured_payload(), forward)

    assert len(calls) == 2
    selected_code = selected["choices"][0]["message"]["content"]
    assert ".strip().lower()" in selected_code
    assert audit["operator_id"] == "canonicalize_before_predicate"
    assert audit["repair_postcondition_valid"] is False
    assert audit["selected_origin"] == "deterministic_scaffold"
    assert "holdout" not in calls[1]["messages"][-1]["content"].lower()
    assert "hidden" not in calls[1]["messages"][-1]["content"].lower()


def test_planner_control_uses_two_calls_and_returns_free_coder_result():
    plan = {
        "id": "plan",
        "choices": [{"message": {"content": "Focus the public failure."}}],
    }
    coder = _response("def solve(records):\n    return list(records)", "free-coder")
    responses = iter([plan, coder])
    calls: list[dict] = []

    def forward(payload: dict) -> dict:
        calls.append(payload)
        return next(responses)

    selected, audit = StructuredMutationController(mode="planner-control").process_chat(
        _structured_payload(), forward
    )

    assert len(calls) == 2
    assert selected["id"] == "free-coder"
    assert audit["selected_origin"] == "free_coder"
    assert audit["upstream_calls"] == 2
