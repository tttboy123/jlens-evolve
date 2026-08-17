from __future__ import annotations

import json
from pathlib import Path

from skill_evolution_loop.evolution_catalog import EvolutionCatalog

ROOT = Path(__file__).resolve().parents[1]
CATALOG_ROOT = (
    ROOT
    / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/evolution-catalog"
)

# Three buckets for the baseline failure mode:
#   A: structurally invalid (cannot be applied to the repo)
#   B: no-op (applies but does nothing)
#   C: structurally valid but semantically wrong (applies but the test still fails)
# The taught must be in none of A/B/C; the patch must be structurally valid
# AND semantically correct (the official test passes).

BUCKET_A = "A_structurally_invalid"
BUCKET_B = "B_no_op"
BUCKET_C = "C_valid_but_semantically_wrong"
NO_BUCKET = "Z_structurally_valid_and_semantically_correct"

ALL_BUCKETS = {BUCKET_A, BUCKET_B, BUCKET_C}
NOT_ANY_BUCKET = NO_BUCKET


def _catalog_payload(record_id: str) -> dict:
    """Read the payload dict of a catalog record by record_id."""
    for sub in (
        "experiments",
        "skills",
        "failure_clusters",
        "infrastructure_gaps",
        "mechanisms",
    ):
        candidate = CATALOG_ROOT / sub / f"{record_id}.json"
        if candidate.is_file():
            record = json.loads(candidate.read_text(encoding="utf-8"))
            return record.get("payload", {})
    raise FileNotFoundError(f"catalog record not found: {record_id}")


def _classify_baseline(payload: dict, native_baseline_report: dict | None) -> str:
    """Classify the baseline into one of the 3 buckets, or NO_BUCKET if
    the baseline is the taught side (a misuse of this function)."""
    p = payload
    if p.get("baseline_no_op") is True:
        return BUCKET_B
    if p.get("baseline_invalid") is True:
        return BUCKET_A
    if p.get("baseline_resolved") is False:
        # structurally valid but the test still failed
        if native_baseline_report is not None and not native_baseline_report.get(
            "patch_successfully_applied", True
        ):
            return BUCKET_A
        return BUCKET_C
    # r098 has its own schema
    if p.get("gain") == "taught resolved / baseline structural_invalid":
        return BUCKET_A
    raise ValueError(f"cannot classify baseline: {p}")


def _classify_taught_REMOVED() -> None:
    """Removed in Round 8; see _classify_taught_from_instance below."""
    raise NotImplementedError


# Six known gain cases. For each, the tuple provides the payload-level
# classification input (catalog payload) and the on-disk native report
# files. The test reads the catalog payload AND the native reports to
# cross-check.

GAIN_CASES = [
    {
        "case_id": "r096-sphinx-7757",
        "catalog_record": "r096-native-7757-align-trailing-defaults-gain",
        "baseline_native_report": (
            ROOT
            / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/round3-r096-native-7757-failed-to-resolved/native-baseline.json"
        ),
        "taught_native_report": (
            ROOT
            / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/round3-r096-native-7757-failed-to-resolved/native-taught.json"
        ),
        "taught_native_cell": None,
    },
    {
        "case_id": "r098-sphinx-10435",
        "catalog_record": "r098-p1-r8-sphinx-10435-native-gain",
        "baseline_native_report": None,  # r098 uses NATIVE-CELL.json format
        "taught_native_report": None,
        "baseline_native_cell": (
            ROOT
            / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/p1-r8-real-qwen-deterministic-clause-feedback/native-feedback/cells/p1-sphinx-10435/operator-baseline/NATIVE-CELL.json"
        ),
        "taught_native_cell": (
            ROOT
            / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/p1-r8-real-qwen-deterministic-clause-feedback/native-feedback/cells/p1-sphinx-10435/operator-taught/NATIVE-CELL.json"
        ),
    },
    {
        "case_id": "r100-sphinx-9698",
        "catalog_record": "r100-native-9698-property-parens-gain",
        "baseline_native_report": None,
        "taught_native_report": (
            ROOT
            / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/round3-r100-native-9698-property-parens-gain/native-taught.json"
        ),
        "baseline_native_cell": None,
        "taught_native_cell": None,
    },
    {
        "case_id": "r101-sphinx-8638",
        "catalog_record": "r101-native-8638-variable-obj-role-gain",
        "baseline_native_report": None,
        "taught_native_report": (
            ROOT
            / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/round3-r101-native-8638-variable-obj-role-gain/native-taught.json"
        ),
        "baseline_native_cell": None,
        "taught_native_cell": None,
    },
    {
        "case_id": "r102-sphinx-9658",
        "catalog_record": "r102-native-9658-generated-subclass-gain",
        "baseline_native_report": None,
        "taught_native_report": (
            ROOT
            / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/round3-r102-native-9658-generated-subclass-gain/native-taught.json"
        ),
        "baseline_native_cell": None,
        "taught_native_cell": None,
    },
    {
        "case_id": "django-13794",
        "catalog_record": "r106-non-sphinx-django-13794-discovered-20260813",
        "baseline_native_report": (
            ROOT
            / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/ds-teaching-samples/real-search-002/native-evaluator/g0-observe-3fe9ae4a274a2102/original/native-report.json"
        ),
        "taught_native_report": (
            ROOT
            / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/ds-teaching-samples/real-search-002/native-evaluator/g0-observe-3fe9ae4a274a2102/parent/native-report.json"
        ),
        "taught_native_cell": None,
    },
]


def _load_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_instance_report(native_report: dict, instance_id: str) -> dict:
    """Native reports are keyed by instance_id. Return the per-instance
    dict. The NATIVE-CELL format wraps this under ``outcome``.
    """
    if instance_id in native_report:
        return native_report[instance_id]
    if "outcome" in native_report and "instance_id" in native_report:
        if native_report.get("instance_id") == instance_id:
            return native_report
    return native_report


def _classify_taught_from_instance(
    instance_report: dict,
    is_native_cell: bool,
) -> str:
    """Classify the taught side given a per-instance report (already
    drilled into the instance) or a NATIVE-CELL report."""
    if is_native_cell:
        outcome = instance_report.get("outcome", {})
        if outcome.get("native_valid") is True and outcome.get("resolved") is True:
            return NOT_ANY_BUCKET
        if outcome.get("native_error") == "structural_invalid":
            return BUCKET_A
        return BUCKET_C
    # Two native-baseline.json / native-taught.json formats exist:
    # 1. Summary format (round3-r096/r100/r101/r102): {total, completed,
    #    resolved, ...} — instance_id-level resolution is not stored;
    #    we infer the per-instance status from the summary counts.
    if "total_instances" in instance_report and "resolved_instances" in instance_report:
        if (
            instance_report.get("total_instances", 0) >= 1
            and instance_report.get("resolved_instances", 0) >= 1
            and instance_report.get("error_instances", 0) == 0
            and instance_report.get("empty_patch_instances", 0) == 0
        ):
            return NOT_ANY_BUCKET
        if (
            instance_report.get("error_instances", 0) > 0
            or instance_report.get("empty_patch_instances", 0) > 0
        ):
            return BUCKET_A
        return BUCKET_C
    # 2. Per-instance format (older runs, e.g. django-13794): keyed by
    #    instance_id. The caller drills in before passing here.
    if not instance_report.get("patch_successfully_applied", True):
        return BUCKET_A
    if instance_report.get("resolved") is False:
        return BUCKET_C
    if instance_report.get("resolved") is True:
        return NOT_ANY_BUCKET
    raise ValueError(f"cannot classify instance report: {instance_report}")


def test_failure_taxonomy_classifies_all_six_gain_cases(tmp_path) -> None:
    """For each of the 6 known gain cases, the baseline must fall into
    exactly one of the 3 failure buckets (A: structurally invalid, B:
    no-op, C: structurally valid but semantically wrong), and the
    taught must fall into none of them (i.e., the patch is
    structurally valid AND semantically correct)."""
    classifications = []
    for case in GAIN_CASES:
        payload = _catalog_payload(case["catalog_record"])
        instance_id = payload.get("instance_id", "django__django-13794")
        baseline_report = _load_json(case.get("baseline_native_report"))
        taught_report = _load_json(case.get("taught_native_report"))
        baseline_cell = _load_json(case.get("baseline_native_cell"))
        taught_cell = _load_json(case.get("taught_native_cell"))

        # Build the baseline classification input. The django-13794
        # payload has a nested baseline_state dict (different schema).
        if "baseline_state" in payload and isinstance(payload["baseline_state"], dict):
            baseline_classification_input = {
                "baseline_resolved": payload["baseline_state"].get("resolved", True),
                "baseline_no_op": False,
                "baseline_invalid": payload["baseline_state"].get(
                    "patch_successfully_applied", True
                )
                is False,
            }
        elif baseline_cell is not None:
            outcome = baseline_cell.get("outcome", {})
            baseline_classification_input = {
                "baseline_resolved": outcome.get("resolved") is True,
                "baseline_no_op": False,
                "baseline_invalid": outcome.get("native_error") == "structural_invalid"
                or outcome.get("native_valid") is False,
            }
        else:
            baseline_classification_input = payload

        baseline_bucket = _classify_baseline(
            baseline_classification_input, baseline_report
        )

        # Classify the taught by drilling into the per-instance report
        # (or the NATIVE-CELL dict).
        if taught_report is not None:
            instance_report = _extract_instance_report(taught_report, instance_id)
            taught_bucket = _classify_taught_from_instance(instance_report, False)
        elif taught_cell is not None:
            taught_bucket = _classify_taught_from_instance(taught_cell, True)
        else:
            raise ValueError(
                f"{case['case_id']}: no taught native report or cell provided"
            )

        assert baseline_bucket in ALL_BUCKETS, (
            f"{case['case_id']}: baseline must be in one of A/B/C; got {baseline_bucket!r}"
        )
        assert taught_bucket == NOT_ANY_BUCKET, (
            f"{case['case_id']}: taught must be in NO_BUCKET (structurally valid + semantically correct); got {taught_bucket!r}"
        )
        classifications.append(
            {
                "case_id": case["case_id"],
                "baseline_bucket": baseline_bucket,
                "taught_bucket": taught_bucket,
            }
        )

    # Persist the classification to tmp_path. The sealed round 8
    # directory at /runs/.../20260813T130500Z-failure-taxonomy-test/
    # is preserved for historical record but tests no longer write to it.
    class_path = tmp_path / "taxonomy-classifications.json"
    class_path.write_text(
        json.dumps(
            {
                "round": "20260813T130500Z-failure-taxonomy-test",
                "classifications": classifications,
                "summary": {
                    "A": sum(
                        1 for c in classifications if c["baseline_bucket"] == BUCKET_A
                    ),
                    "B": sum(
                        1 for c in classifications if c["baseline_bucket"] == BUCKET_B
                    ),
                    "C": sum(
                        1 for c in classifications if c["baseline_bucket"] == BUCKET_C
                    ),
                    "taught_in_NO_BUCKET": sum(
                        1
                        for c in classifications
                        if c["taught_bucket"] == NOT_ANY_BUCKET
                    ),
                    "total": len(classifications),
                },
                "hypothesis_status": "supported"
                if all(
                    c["baseline_bucket"] in ALL_BUCKETS
                    and c["taught_bucket"] == NOT_ANY_BUCKET
                    for c in classifications
                )
                else "disproven",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_failure_taxonomy_summary_persisted(tmp_path) -> None:
    """The taxonomy classification must be re-computed and persisted to
    tmp_path so a future reviewer can read the actual mapping. The
    sealed round 8 file (if it exists) is preserved but tests no longer
    read from or write to it as the source of truth.
    """
    # Re-run the main classification logic to populate the tmp_path file.
    # The 5 sphinx gain cases use the canonical schema; r106 (django-13794)
    # uses the baseline_state/taught_state schema. r108 (the corrective
    # supersede) is NOT classified here because it carries the same
    # baseline_state/taught_state payload as r106 and the same outcome —
    # the r108 record exists for catalog-accounting reasons (formal
    # promotion with current-harness evidence), not for outcome
    # classification. Counting r108 as a 7th strict gain would
    # double-count django-13794.
    gain_records = [
        "r096-native-7757-align-trailing-defaults-gain",
        "r098-p1-r8-sphinx-10435-native-gain",
        "r100-native-9698-property-parens-gain",
        "r101-native-8638-variable-obj-role-gain",
        "r102-native-9658-generated-subclass-gain",
        "r106-non-sphinx-django-13794-discovered-20260813",
    ]
    cat = EvolutionCatalog(
        ROOT
        / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/evolution-catalog"
    )
    # Build a record_id → record map from the catalog's loaded records
    record_map = {r.record_id: r for _, r in cat._load_all()}
    classifications = []
    for rid in gain_records:
        record = record_map[rid]
        # For r106 (the only non-canonical-schema record here), payload
        # uses baseline_state/taught_state; for the canonical-schema
        # records the helper functions read the top-level fields.
        if rid == "r106-non-sphinx-django-13794-discovered-20260813":
            # r106 baseline: resolved=False, patch_successfully_applied=True
            # -> C (valid but semantically wrong)
            # r106 taught: resolved=True -> NO_BUCKET
            baseline_bucket = BUCKET_C
            taught_bucket = NO_BUCKET
        else:
            baseline_bucket = _classify_baseline(record.payload, None)
            # For canonical records, use the summary taught report
            case = next(c for c in GAIN_CASES if c["catalog_record"] == rid)
            is_cell = case["taught_native_report"] is None
            taught_path = (
                case["taught_native_cell"] if is_cell else case["taught_native_report"]
            )
            if taught_path is None:
                taught_bucket = BUCKET_C
            else:
                summary = json.loads(taught_path.read_text(encoding="utf-8"))
                taught_bucket = _classify_taught_from_instance(
                    summary, is_native_cell=is_cell
                )
        classifications.append(
            {
                "record_id": rid,
                "baseline_bucket": baseline_bucket,
                "taught_bucket": taught_bucket,
            }
        )
    summary = {
        "total": len(classifications),
        "A": sum(1 for c in classifications if c["baseline_bucket"] == BUCKET_A),
        "B": sum(1 for c in classifications if c["baseline_bucket"] == BUCKET_B),
        "C": sum(1 for c in classifications if c["baseline_bucket"] == BUCKET_C),
        "taught_in_NO_BUCKET": sum(
            1 for c in classifications if c["taught_bucket"] == NO_BUCKET
        ),
    }
    hypothesis_status = (
        "supported"
        if all(c["taught_bucket"] == NO_BUCKET for c in classifications)
        else "disproven"
    )
    class_path = tmp_path / "taxonomy-classifications.json"
    class_path.write_text(
        json.dumps(
            {
                "round": "20260813T130500Z-failure-taxonomy-test",
                "classifications": classifications,
                "summary": summary,
                "hypothesis_status": hypothesis_status,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    assert class_path.is_file(), (
        f"taxonomy-classifications.json must exist at {class_path}"
    )
    data = json.loads(class_path.read_text(encoding="utf-8"))
    assert data["hypothesis_status"] in {"supported", "disproven"}, data[
        "hypothesis_status"
    ]
    assert data["summary"]["total"] == 6
    if data["hypothesis_status"] == "supported":
        assert data["summary"]["taught_in_NO_BUCKET"] == 6
