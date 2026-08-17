"""Load and verify bounded proposal-controller configurations."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

_LEGACY_MODES = {"shadow-control", "duplicate-aware"}
_STRUCTURED_MODES = {"planner-control", "structured-mutation"}
_MODES = _LEGACY_MODES | _STRUCTURED_MODES
_V4_OPERATORS = {
    "canonicalize_before_predicate",
    "finite_numeric_guard",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_proposal_controller(path: Path) -> dict[str, Any]:
    """Load one fixed-budget controller while preserving JLens boundaries."""
    data = json.loads(path.read_text(encoding="utf-8"))
    common_required = {
        "schema_version",
        "controller_id",
        "mode",
        "calls_per_request",
        "causal_boundary",
        "admission_gate_allowed",
    }
    missing = sorted(common_required - data.keys())
    if missing:
        raise ValueError(f"proposal controller missing fields: {missing}")
    if data["schema_version"] not in {1, 2}:
        raise ValueError("unsupported proposal controller schema")
    if data["mode"] not in _MODES:
        raise ValueError(f"unsupported proposal controller mode: {data['mode']}")
    if int(data["calls_per_request"]) != 2:
        raise ValueError("proposal controller must use exactly two calls per request")
    if data["mode"] in _LEGACY_MODES:
        legacy_required = {"max_duplicate_retries", "stagnation_detector_version"}
        missing = sorted(legacy_required - data.keys())
        if missing:
            raise ValueError(f"proposal controller missing fields: {missing}")
        if data["schema_version"] != 1:
            raise ValueError("legacy proposal controller must use schema 1")
        if int(data["max_duplicate_retries"]) != 1:
            raise ValueError("proposal controller must use exactly one bounded retry")
        if int(data.get("stagnation_window", 3)) < 2:
            raise ValueError(
                "proposal controller stagnation window must be at least two"
            )
        if data["stagnation_detector_version"] != "global-best-v2":
            raise ValueError("unsupported proposal controller stagnation detector")
    else:
        structured_required = {
            "planner_protocol_version",
            "operator_catalog",
            "operator_enforcement",
        }
        missing = sorted(structured_required - data.keys())
        if missing:
            raise ValueError(f"proposal controller missing fields: {missing}")
        if data["schema_version"] != 2:
            raise ValueError("structured proposal controller must use schema 2")
        if data["planner_protocol_version"] != "structured-mutation-v4":
            raise ValueError("unsupported structured planner protocol")
        if set(data["operator_catalog"]) != _V4_OPERATORS:
            raise ValueError("structured controller operator catalog mismatch")
        enforcement = data["operator_enforcement"]
        if data["mode"] == "planner-control" and enforcement is not False:
            raise ValueError("planner control cannot enforce operators")
        if data["mode"] == "structured-mutation" and enforcement is not True:
            raise ValueError("structured treatment must enforce operators")
    if data["causal_boundary"] != "jlens_observation_not_correctness":
        raise ValueError("proposal controller lost the JLens observation boundary")
    if data["admission_gate_allowed"] is not False:
        raise ValueError("proposal controller cannot claim admission authority")
    if not str(data["controller_id"]).strip():
        raise ValueError("proposal controller ID cannot be empty")
    return data


def load_policy_proposal_controller(
    project_root: Path, selected_policy: dict[str, Any]
) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    """Resolve a policy-bound controller without leaving the project root."""
    configured = selected_policy.get("proposal_controller_file")
    if not configured:
        return None, None, None
    project_root = project_root.resolve()
    path = (project_root / str(configured)).resolve()
    if not path.is_relative_to(project_root):
        raise ValueError("proposal controller must stay inside the project root")
    controller = load_proposal_controller(path)
    expected_mode = selected_policy.get("proposal_controller_mode")
    if expected_mode and controller["mode"] != expected_mode:
        raise ValueError(
            "policy proposal-controller mode does not match controller file"
        )
    return controller, path, _sha256(path)


def verify_proposal_controller_endpoint(
    api_base: str,
    controller: dict[str, Any],
    controller_sha256: str,
    *,
    implementation_sha256: str | None = None,
    timeout: float = 3.0,
) -> dict[str, Any]:
    """Verify the proxy serving the configured OpenAI endpoint before a run."""
    root = api_base.rstrip("/").removesuffix("/v1")
    with urllib.request.urlopen(
        f"{root}/proposal-controller", timeout=timeout
    ) as response:
        served = json.load(response)
    expected = {
        "controller_id": controller["controller_id"],
        "mode": controller["mode"],
        "calls_per_request": controller["calls_per_request"],
        "controller_sha256": controller_sha256,
    }
    if implementation_sha256 is not None:
        expected["implementation_sha256"] = implementation_sha256
    mismatches = {
        key: {"expected": value, "served": served.get(key)}
        for key, value in expected.items()
        if served.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"proposal controller endpoint mismatch: {mismatches}")
    return served
