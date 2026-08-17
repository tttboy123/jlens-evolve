from __future__ import annotations

import json
from io import BytesIO

import pytest

from proposal_controller import (
    load_policy_proposal_controller,
    load_proposal_controller,
    verify_proposal_controller_endpoint,
)


def test_policy_controller_is_project_scoped_and_observer_bounded(tmp_path):
    controller_path = tmp_path / "controller.json"
    controller_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "controller_id": "jlens-duplicate-aware-v2",
                "mode": "duplicate-aware",
                "calls_per_request": 2,
                "max_duplicate_retries": 1,
                "stagnation_detector_version": "global-best-v2",
                "causal_boundary": "jlens_observation_not_correctness",
                "admission_gate_allowed": False,
            }
        ),
        encoding="utf-8",
    )

    controller, path, digest = load_policy_proposal_controller(
        tmp_path, {"proposal_controller_file": "controller.json"}
    )

    assert path == controller_path
    assert len(digest) == 64
    assert controller["mode"] == "duplicate-aware"
    assert controller["admission_gate_allowed"] is False


def test_policy_controller_rejects_admission_authority(tmp_path):
    controller_path = tmp_path / "controller.json"
    controller_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "controller_id": "unsafe",
                "mode": "duplicate-aware",
                "calls_per_request": 2,
                "max_duplicate_retries": 1,
                "stagnation_detector_version": "global-best-v2",
                "causal_boundary": "jlens_observation_not_correctness",
                "admission_gate_allowed": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="admission"):
        load_policy_proposal_controller(
            tmp_path, {"proposal_controller_file": "controller.json"}
        )


def test_endpoint_binding_includes_proxy_implementation_hash(monkeypatch):
    served = {
        "controller_id": "controller",
        "mode": "duplicate-aware",
        "calls_per_request": 2,
        "controller_sha256": "config-hash",
        "implementation_sha256": "implementation-hash",
    }
    monkeypatch.setattr(
        "proposal_controller.urllib.request.urlopen",
        lambda *args, **kwargs: BytesIO(json.dumps(served).encode("utf-8")),
    )

    result = verify_proposal_controller_endpoint(
        "http://127.0.0.1:18081/v1",
        {
            "controller_id": "controller",
            "mode": "duplicate-aware",
            "calls_per_request": 2,
        },
        "config-hash",
        implementation_sha256="implementation-hash",
    )

    assert result["implementation_sha256"] == "implementation-hash"


def test_structured_v4_controller_requires_two_stage_protocol(tmp_path):
    controller_path = tmp_path / "controller-v4.json"
    controller_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "controller_id": "jlens-structured-mutation-v4",
                "mode": "structured-mutation",
                "calls_per_request": 2,
                "planner_protocol_version": "structured-mutation-v4",
                "operator_catalog": [
                    "canonicalize_before_predicate",
                    "finite_numeric_guard",
                ],
                "operator_enforcement": True,
                "causal_boundary": "jlens_observation_not_correctness",
                "admission_gate_allowed": False,
            }
        ),
        encoding="utf-8",
    )

    controller = load_proposal_controller(controller_path)

    assert controller["mode"] == "structured-mutation"
    assert controller["operator_enforcement"] is True


def test_structured_v4_control_rejects_operator_enforcement(tmp_path):
    controller_path = tmp_path / "control-v4.json"
    controller_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "controller_id": "planner-control-v4",
                "mode": "planner-control",
                "calls_per_request": 2,
                "planner_protocol_version": "structured-mutation-v4",
                "operator_catalog": [
                    "canonicalize_before_predicate",
                    "finite_numeric_guard",
                ],
                "operator_enforcement": True,
                "causal_boundary": "jlens_observation_not_correctness",
                "admission_gate_allowed": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="control cannot enforce"):
        load_proposal_controller(controller_path)
