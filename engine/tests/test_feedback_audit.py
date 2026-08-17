from __future__ import annotations

import json

from skill_evolution_loop.contracts import canonical_json, sha256_json
from skill_evolution_loop.evolution_catalog import EvolutionCatalog, EvolutionRecord
from skill_evolution_loop.feedback_audit import audit_feedback_request


def _record() -> EvolutionRecord:
    return EvolutionRecord.create(
        record_type="mechanisms",
        record_id="r073-issue-anchored-candidate-seeding",
        title="Issue anchored candidate seeding",
        status="implemented",
        capability_tags=("localization", "patch-realization"),
        task_tags=("swe-bench",),
        failure_mode_tags=("selector-no-match", "wrong-target"),
        source_model="deepseek-v4-flash",
        source_runtime="api",
        payload={"do_not_repropose": True},
        evidence_refs=(),
        cross_model_validations=(),
    )


def _arm(*, structural: bool, resolved: bool, patch: str | None) -> dict:
    return {
        "condition_id": "unused",
        "experiment_cell_sha256": "a" * 64,
        "native_cell_sha256": "b" * 64,
        "structural_valid": structural,
        "failure_reason": None if structural else "unresolved",
        "detail": "detail",
        "raw_output": "output",
        "raw_output_sha256": "c" * 64,
        "patch_sha256": patch,
        "native_outcome": {
            "native_error": None if structural else "structural_invalid",
            "native_valid": structural,
            "regression_test_names": [],
            "resolved": resolved,
        },
        "native_report_sha256": "d" * 64 if structural else None,
    }


def test_feedback_audit_is_holdout_free_and_stops_duplicate_skill_proposal(
    tmp_path,
) -> None:
    request = {
        "schema_version": 1,
        "request_type": "round1-feedback-skill-evolution-v1",
        "feedback_gain_count": 1,
        "feedback_gain_gate_passed": True,
        "condition_failure_counts": {},
        "current_inactive_skills": {},
        "constraints": ["Use only feedback tasks."],
        "failures": [
            {
                "task_id": "feedback-gain",
                "mechanism": "span",
                "issue": "feedback issue",
                "allowed_targets": ["src/a.py"],
                "baseline": _arm(structural=True, resolved=False, patch="1" * 64),
                "taught": _arm(structural=True, resolved=True, patch="2" * 64),
            },
            {
                "task_id": "feedback-degradation",
                "mechanism": "operator",
                "issue": "another feedback issue",
                "allowed_targets": ["src/b.py"],
                "baseline": _arm(structural=True, resolved=False, patch="3" * 64),
                "taught": _arm(structural=False, resolved=False, patch=None),
            },
        ],
    }
    content = {
        "schema_version": 1,
        "request": request,
        "request_sha256": sha256_json(request),
        "taskset_fingerprint": "e" * 64,
        "native_summary_sha256": "f" * 64,
        "native_cell_evidence_fingerprint": "0" * 64,
        "feedback_task_count": 2,
        "feedback_cell_count": 4,
        "holdout_cells_included": False,
        "source_holdout_evidence_present": False,
        "current_holdout_reuse_prohibited": False,
        "network_calls_performed": False,
    }
    wrapper = {**content, "evidence_sha256": sha256_json(content)}
    request_path = tmp_path / "REQUEST.json"
    request_path.write_text(canonical_json(wrapper) + "\n")
    catalog = EvolutionCatalog(tmp_path / "catalog")
    catalog.append(_record())

    report = audit_feedback_request(
        request_path=request_path,
        catalog=catalog,
        output_path=tmp_path / "AUDIT.json",
    )

    assert report["holdout_cells_included"] is False
    assert report["source_holdout_evidence_present"] is False
    assert report["pair_transition_counts"] == {
        "native-unresolved->native-resolved": 1,
        "native-unresolved->structural-invalid": 1,
    }
    assert report["feedback_gain_count_verified"] == 1
    assert report["teaching_structural_degradation_count"] == 1
    assert report["decision"]["next_step"] == "holdout-safety-evaluation"
    assert report["decision"]["parent_call_recommended"] is False
    assert report["dedup_context"]["implemented_mechanisms"][0]["record_id"] == (
        "r073-issue-anchored-candidate-seeding"
    )
    assert "feedback issue" not in json.dumps(report)
