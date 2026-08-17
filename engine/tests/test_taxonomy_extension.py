from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CATALOG_ROOT = (
    ROOT
    / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/evolution-catalog"
)
CATALOG_INDEX = CATALOG_ROOT / "indexes/CATALOG.json"

# Same 3 buckets as Round 8:
BUCKET_A = "A_structurally_invalid"
BUCKET_B = "B_no_op"
BUCKET_C = "C_valid_but_semantically_wrong"
NO_BUCKET = "Z_structurally_valid_and_semantically_correct"
ALL_BUCKETS = {BUCKET_A, BUCKET_B, BUCKET_C}


def _load_index() -> list[dict]:
    data = json.loads(CATALOG_INDEX.read_text(encoding="utf-8"))
    return data["entries"]


def _load_record(record_id: str, path_sub: str) -> dict:
    record_path = CATALOG_ROOT / path_sub / f"{record_id}.json"
    return json.loads(record_path.read_text(encoding="utf-8"))


def _record_type_subdir(entry: dict) -> str:
    """Extract the type subdirectory from the entry's path field."""
    return entry["path"].split("/")[0]


def _is_strict_gain(payload: dict) -> bool:
    """Strict gain = baseline failed/ineffective AND taught resolved.

    A loose-gain record (e.g. r077 audit summary) is not a strict gain
    and is excluded from the strict-gain taxonomy assertion.
    """
    if payload.get("teaching_regression") is True:
        return False
    if payload.get("taught_resolved") is True:
        return True
    if payload.get("native_gain_count", 0) > 0:
        return True
    # r098 schema
    if isinstance(payload.get("gain"), str) and "taught resolved" in payload["gain"]:
        return True
    # django-13794 schema
    if "taught_state" in payload and isinstance(payload["taught_state"], dict):
        if payload["taught_state"].get("resolved") is True:
            return True
    return False


def _classify_baseline(payload: dict) -> str:
    """Classify baseline into A/B/C; raise if cannot classify."""
    # django-13794 schema
    if "baseline_state" in payload and isinstance(payload["baseline_state"], dict):
        bs = payload["baseline_state"]
        if (
            bs.get("resolved") is False
            and bs.get("patch_successfully_applied", True) is True
        ):
            return BUCKET_C
        if bs.get("patch_successfully_applied", True) is False:
            return BUCKET_A
    # r100/101/102
    if payload.get("baseline_no_op") is True:
        return BUCKET_B
    if payload.get("baseline_invalid") is True:
        return BUCKET_A
    if payload.get("baseline_resolved") is False:
        return BUCKET_C
    # r098 schema
    if payload.get("gain") == "taught resolved / baseline structural_invalid":
        return BUCKET_A
    # r103 teaching regression: baseline resolved, taught unresolved
    if (
        payload.get("teaching_regression") is True
        and payload.get("baseline_resolved") is True
    ):
        return "_NOT_A_GAIN"  # the baseline DID work; the taught made it worse
    raise ValueError(f"cannot classify baseline for: {payload}")


def _classify_taught(payload: dict) -> str:
    """Classify taught into NO_BUCKET or a failure bucket."""
    if payload.get("teaching_regression") is True:
        return BUCKET_C  # taught was applied but caused regression
    if payload.get("taught_resolved") is True:
        return NO_BUCKET
    if payload.get("native_gain_count", 0) > 0:
        return NO_BUCKET
    if "taught_state" in payload and isinstance(payload["taught_state"], dict):
        if payload["taught_state"].get("resolved") is True:
            return NO_BUCKET
    # r098 schema: the catalog payload doesn't store the resolution
    # status directly; the gain string is "taught resolved / baseline
    # structural_invalid" which encodes the taught side.
    if isinstance(payload.get("gain"), str) and "taught resolved" in payload["gain"]:
        return NO_BUCKET
    raise ValueError(f"cannot classify taught for: {payload}")


def test_taxonomy_holds_for_all_strict_gain_records(tmp_path) -> None:
    """For every record in the catalog that claims a strict gain
    (baseline failed/ineffective AND taught resolved), the baseline
    must fall into one of the 3 buckets (A/B/C) and the taught must
    fall into NO_BUCKET (structurally valid + semantically correct).
    """
    entries = _load_index()
    feedback_entries = [e for e in entries if "feedback" in e.get("task_tags", [])]

    strict_gains = []
    other_feedback = []
    for e in feedback_entries:
        full = _load_record(e["record_id"], _record_type_subdir(e))
        pld = full.get("payload", {})
        if _is_strict_gain(pld):
            strict_gains.append((e["record_id"], full, pld))
        else:
            other_feedback.append(
                (e["record_id"], e["status"], full.get("payload", {}))
            )

    classifications = []
    for rid, record, pld in strict_gains:
        try:
            b = _classify_baseline(pld)
            t = _classify_taught(pld)
        except ValueError as exc:
            pytest.fail(f"{rid}: cannot classify — {exc}")
        assert b in ALL_BUCKETS, f"{rid}: baseline must be in A/B/C; got {b!r}"
        assert t == NO_BUCKET, f"{rid}: taught must be NO_BUCKET; got {t!r}"
        classifications.append(
            {
                "record_id": rid,
                "baseline_bucket": b,
                "taught_bucket": t,
            }
        )

    # Persist the extension result to a writable temp path.
    # Tests must never write into immutable round directories.
    output_path = tmp_path / "taxonomy-extension.json"
    output_path.write_text(
        json.dumps(
            {
                "round": "20260813T132000Z-taxonomy-extension",
                "feedback_records_total": len(feedback_entries),
                "strict_gains_count": len(strict_gains),
                "other_feedback_count": len(other_feedback),
                "strict_gains_classified": classifications,
                "other_feedback_sample": [
                    {"record_id": rid, "status": status}
                    for rid, status, _ in other_feedback[:10]
                ],
                "hypothesis_status": (
                    "supported"
                    if all(
                        c["baseline_bucket"] in ALL_BUCKETS
                        and c["taught_bucket"] == NO_BUCKET
                        for c in classifications
                    )
                    else "disproven"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_taxonomy_extension_persisted(tmp_path) -> None:
    # The persistence test re-runs the same classification logic and
    # writes to tmp_path, NOT to the sealed round 9 directory. The
    # sealed file at /runs/.../20260813T132000Z-taxonomy-extension/
    # taxonomy-extension.json is preserved for historical record but
    # is no longer the source of truth (and tests no longer write to it).
    entries = _load_index()
    feedback_entries = [e for e in entries if "feedback" in e.get("task_tags", [])]
    strict_gains = []
    for e in feedback_entries:
        full = _load_record(e["record_id"], _record_type_subdir(e))
        pld = full.get("payload", {})
        if _is_strict_gain(pld):
            strict_gains.append((e["record_id"], full, pld))
    classifications = []
    for rid, record, pld in strict_gains:
        b = _classify_baseline(pld)
        t = _classify_taught(pld)
        classifications.append(
            {"record_id": rid, "baseline_bucket": b, "taught_bucket": t}
        )
    out = tmp_path / "taxonomy-extension.json"
    out.write_text(
        json.dumps(
            {
                "round": "20260813T132000Z-taxonomy-extension",
                "feedback_records_total": len(feedback_entries),
                "strict_gains_count": len(strict_gains),
                "other_feedback_count": len(feedback_entries) - len(strict_gains),
                "strict_gains_classified": classifications,
                "hypothesis_status": (
                    "supported"
                    if all(
                        c["baseline_bucket"] in ALL_BUCKETS
                        and c["taught_bucket"] == NO_BUCKET
                        for c in classifications
                    )
                    else "disproven"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    # Every strict gain must be classified into A/B/C and NO_BUCKET.
    for c in data["strict_gains_classified"]:
        assert c["baseline_bucket"] in ALL_BUCKETS, c
        assert c["taught_bucket"] == NO_BUCKET, c
    # Hypothesis must be supported.
    assert data["hypothesis_status"] == "supported", data["hypothesis_status"]
    # The extension must cover ALL strict gains, not just a subset.
    assert data["strict_gains_count"] >= 6
