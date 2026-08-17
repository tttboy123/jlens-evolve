"""Round 10: Full feedback regression scan + cross-project gain census.

Hypotheses (falsifiable):

H1 (regression scan): For all 33 feedback records in the catalog, when
classified by (baseline_resolved, taught_resolved, evaluator_failure_count),
there is at most 1 regression (r103) and 0 evaluator failures.

H2 (cross-project gain census): For all (orig, parent) pairs across
7 real-search native-evaluator directories, there is exactly 1
non-sphinx gain (django-13794, status=pending). No 3rd-project gain
exists in current on-disk data.

H3 (loss census): For all (orig, parent) pairs, there is exactly 1 loss
(caddyserver__caddy-5404 from v2.1.0). This loss is from an older
evolution run and is not in the current v2.5.0 catalog.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path("/Users/lune/Documents/Codex/2026-07-18/bang/work/evolve-jlens-cluster")
CATALOG_INDEX = (
    ROOT
    / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/evolution-catalog/indexes/CATALOG.json"
)
CATALOG_DIR = (
    ROOT
    / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/evolution-catalog"
)
ROUND_DIR = (
    ROOT
    / "runs/skill-evolution-loop/autonomous/rounds/20260813T134000Z-feedback-regression-scan"
)
REAL_SEARCH_DIRS = [
    "artifacts/v2.1.0/v2.1.1-jlens-evolution/runs/real-search-001-deepseek",
    "artifacts/v2.1.0/v2.1.1-jlens-evolution/runs/real-search-001-deepseek-v2",
    "artifacts/v2.1.0/v2.1.1-jlens-evolution/runs/real-search-001-deepseek-v3",
    "artifacts/v2.1.0/v2.1.1-jlens-evolution/runs/real-search-001-resume",
    "artifacts/v2.1.0/v2.1.1-jlens-evolution/runs/real-search-001-resume-QUOTA-INVALID",
    "artifacts/v2.1.0/v2.1.1-jlens-evolution/runs/real-search-002-deepseek-nointernet",
    "artifacts/v2.5.0/v2.5.0-local-jlens/runs/ds-teaching-samples/real-search-002",
]


def _load_catalog() -> list[dict]:
    with open(CATALOG_INDEX) as f:
        cat = json.load(f)
    return cat["entries"]


def _load_records() -> list[dict]:
    """Walk all 71 catalog records from disk."""
    records = []
    for subdir in [
        "experiments",
        "failure_clusters",
        "mechanisms",
        "skills",
        "infrastructure_gaps",
        "audits",
        "qualifications",
    ]:
        p = CATALOG_DIR / subdir
        if not p.is_dir():
            continue
        for fn in os.listdir(p):
            if not fn.endswith(".json"):
                continue
            try:
                with open(p / fn) as f:
                    records.append(json.load(f))
            except Exception:
                pass
    return records


def _classify_feedback_record(record: dict) -> tuple[str, dict]:
    """Classify a feedback record into gain / loss / both_resolved / both_unresolved / unclassified.

    Returns (classification, evidence_dict) where evidence_dict has the
    fields used to make the classification.
    """
    p = record.get("payload", {}) or {}
    rid = record.get("record_id", "?")

    # Schema 1: canonical (baseline_resolved, taught_resolved) booleans
    if "baseline_resolved" in p and "taught_resolved" in p:
        br = p["baseline_resolved"]
        tr = p["taught_resolved"]
        ef = p.get("evaluator_failure_count", 0)
        return _classify_pair(br, tr, ef, rid)

    # Schema 2: negative-baseline flag (baseline_no_op or baseline_invalid)
    # Used by r100 (no_op), r101/r102 (invalid). baseline_no_op=True or
    # baseline_invalid=True means baseline FAILED (did not resolve).
    if "taught_resolved" in p and ("baseline_no_op" in p or "baseline_invalid" in p):
        baseline_failed = bool(p.get("baseline_no_op")) or bool(
            p.get("baseline_invalid")
        )
        # baseline_resolved is the inverse of baseline_failed
        br = not baseline_failed
        tr = p["taught_resolved"]
        ef = p.get("evaluator_failure_count", 0)
        return _classify_pair(br, tr, ef, rid)

    # Schema 3: nested-state (r106 django-13794)
    if "baseline_state" in p and "taught_state" in p:
        bs = p["baseline_state"]
        ts = p["taught_state"]
        if isinstance(bs, dict) and isinstance(ts, dict):
            br = bs.get("resolved")
            tr = ts.get("resolved")
            return _classify_pair(br, tr, 0, rid)

    # Schema 4: textual gain claim (r098) — parse "taught resolved / baseline ..."
    if "gain" in p and isinstance(p["gain"], str):
        g = p["gain"].lower()
        if "taught resolved" in g and "baseline" in g:
            # Parse baseline state from string
            baseline_failed = (
                "unresolved" in g or "invalid" in g or "no_op" in g or "no-op" in g
            )
            br = not baseline_failed
            tr = "taught resolved" in g
            return _classify_pair(br, tr, 0, rid)

    # Schema 5: nothing resolvable
    return "unclassified", {"record_id": rid, "reason": "no resolved booleans"}


def _classify_pair(br, tr, ef, rid) -> tuple[str, dict]:
    """Classify (baseline_resolved, taught_resolved) pair.

    Convention:
    - baseline_resolved=True means baseline produced a working patch
    - baseline_resolved=False means baseline failed
    - taught_resolved=True means taught produced a working patch
    - taught_resolved=False means taught failed
    """
    if ef and ef > 0:
        return "infra_failure", {"record_id": rid, "evaluator_failure_count": ef}
    pair = (br, tr)
    if pair == (False, True):
        return "gain", {
            "record_id": rid,
            "baseline_resolved": br,
            "taught_resolved": tr,
        }
    elif pair == (True, False):
        return "loss", {
            "record_id": rid,
            "baseline_resolved": br,
            "taught_resolved": tr,
        }
    elif pair == (True, True):
        return "both_resolved", {"record_id": rid}
    elif pair == (False, False):
        return "both_unresolved", {"record_id": rid}
    return "unclassified", {
        "record_id": rid,
        "baseline_resolved": br,
        "taught_resolved": tr,
    }


def _walk_real_search_native_evaluator() -> dict:
    """Walk all (orig, parent) pairs in real-search native-evaluator dirs.

    Returns a summary dict with totals and the list of gains/losses/both_resolved.
    """

    def get_state(path):
        if not os.path.exists(path):
            return (None, None)
        try:
            with open(path) as f:
                r = json.load(f)
        except Exception:
            return (None, None)
        if not isinstance(r, dict) or len(r) != 1:
            return (None, None)
        key = list(r.keys())[0]
        val = r[key]
        if not isinstance(val, dict) or "resolved" not in val:
            return (None, None)
        return (key, val.get("resolved"))

    gains = []
    losses = []
    both_resolved = []
    both_unresolved = []
    incomplete = []
    projects = {}

    for run_base in REAL_SEARCH_DIRS:
        ne = ROOT / run_base / "native-evaluator"
        if not ne.is_dir():
            continue
        run_name = run_base.split("/")[-1]
        for d in sorted(os.listdir(ne)):
            op = ne / d / "original" / "native-report.json"
            pp = ne / d / "parent" / "native-report.json"
            o_key, o_resolved = get_state(op)
            p_key, p_resolved = get_state(pp)
            if o_key is None or p_key is None or o_key != p_key:
                incomplete.append(
                    {"run": run_name, "dir": d, "reason": "missing_or_mismatched"}
                )
                continue
            project = o_key.split("__")[0]
            projects.setdefault(
                project,
                {
                    "gains": 0,
                    "losses": 0,
                    "both_resolved": 0,
                    "both_unresolved": 0,
                    "runs": set(),
                },
            )
            projects[project]["runs"].add(run_name)
            pair = (o_resolved, p_resolved)
            entry = {"run": run_name, "dir": d, "task_id": o_key}
            if pair == (False, True):
                gains.append(entry)
                projects[project]["gains"] += 1
            elif pair == (True, False):
                losses.append(entry)
                projects[project]["losses"] += 1
            elif pair == (True, True):
                both_resolved.append(entry)
                projects[project]["both_resolved"] += 1
            else:
                both_unresolved.append(entry)
                projects[project]["both_unresolved"] += 1

    return {
        "totals": {
            "gains": len(gains),
            "losses": len(losses),
            "both_resolved": len(both_resolved),
            "both_unresolved": len(both_unresolved),
            "incomplete": len(incomplete),
        },
        "gains": gains,
        "losses": losses,
        "both_resolved_count": len(both_resolved),
        "both_unresolved_count": len(both_unresolved),
        "projects": {
            k: {
                "gains": v["gains"],
                "losses": v["losses"],
                "both_resolved": v["both_resolved"],
                "both_unresolved": v["both_unresolved"],
                "runs": sorted(v["runs"]),
            }
            for k, v in sorted(projects.items())
        },
    }


def test_feedback_regression_scan_h1(tmp_path) -> None:
    """H1: feedback regression scan finds 1 regression (r103) and 0 evaluator failures."""
    records = _load_records()
    feedback = [r for r in records if "feedback" in (r.get("task_tags") or [])]
    # As of Round 13 the catalog has 34 feedback records (was 33; r108
    # is a corrective record about django-13794 with 'feedback' tag).
    assert len(feedback) == 34, f"Expected 34 feedback records, got {len(feedback)}"

    classified = {}
    infra_failures = []
    regressions = []
    gains = []
    for r in feedback:
        cls, ev = _classify_feedback_record(r)
        classified.setdefault(cls, []).append(ev)
        if cls == "infra_failure":
            infra_failures.append(ev)
        elif cls == "loss":
            regressions.append(ev)
        elif cls == "gain":
            gains.append(ev)

    # Apply unique-task counting: a record with payload.supersedes is
    # a supersede of a previous record and should not double-count.
    # In the current catalog, r108 supersedes r106 (both for django-13794).
    # Per the corrected long-term accounting, that task counts once.
    supersede_pairs = set()
    for r in records:
        for sup in (r.get("payload") or {}).get("supersedes", []) or []:
            supersede_pairs.add((sup, r.get("record_id")))

    def _is_superseded(rec_id: str) -> bool:
        for sup, sup_of in supersede_pairs:
            if rec_id == sup and sup_of != rec_id:
                return True
        return False

    unique_gains = [g for g in gains if not _is_superseded(g["record_id"])]

    # Persist the scan result for audit to a writable temp path.
    # Tests must never write into immutable round directories.
    scan = {
        "round": "20260813T134000Z-feedback-regression-scan",
        "feedback_records_total": len(feedback),
        "classification_counts": {k: len(v) for k, v in classified.items()},
        "infra_failures": infra_failures,
        "regressions": regressions,
        "gains": gains,
        "unique_gain_task_count": len(unique_gains),
        "unclassified": classified.get("unclassified", []),
    }
    scan_path = tmp_path / "feedback-regression-scan.json"
    with open(scan_path, "w") as f:
        json.dump(scan, f, indent=2)

    assert len(infra_failures) == 0, (
        f"H1 fails: expected 0 infra failures, got {len(infra_failures)}: {infra_failures}"
    )
    assert len(regressions) == 1, (
        f"H1 fails: expected exactly 1 regression (r103), got {len(regressions)}: {regressions}"
    )
    assert regressions[0]["record_id"] == "r103-8595-teaching-regression", (
        f"H1 fails: expected the regression to be r103, got {regressions[0]['record_id']}"
    )
    # Unique-task count: 6 unique tasks (5 sphinx + 1 django; r108 and r106
    # both claim django-13794 but only one unique task is counted).
    assert len(unique_gains) == 6, (
        f"H1: expected 6 unique gain tasks (5 sphinx + 1 django); got {len(unique_gains)}: "
        f"{[g['record_id'] for g in unique_gains]}"
    )


def test_cross_project_gain_census_h2(tmp_path) -> None:
    """H2: cross-project gain census finds exactly 1 non-sphinx gain (django-13794).

    No 3rd-project gain exists in current on-disk data.
    """
    census = _walk_real_search_native_evaluator()

    # Persist the census to a writable temp path.
    census_path = tmp_path / "cross-project-gain-census.json"
    with open(census_path, "w") as f:
        json.dump(census, f, indent=2)

    gains = census["gains"]
    assert len(gains) == 1, f"H2 fails: expected 1 gain, got {len(gains)}: {gains}"
    assert gains[0]["task_id"] == "django__django-13794", (
        f"H2 fails: expected django-13794 gain, got {gains[0]['task_id']}"
    )

    # The single gain should be from a non-sphinx project
    assert "sphinx" not in gains[0]["task_id"], (
        f"H2 fails: gain should be non-sphinx, got {gains[0]['task_id']}"
    )


def test_loss_census_h3() -> None:
    """H3: loss census finds exactly 1 (orig=T, parent=F) loss from v2.1.0 data.

    The caddy-5404 loss is from real-search-001-deepseek-v3 (v2.1.0) and
    is not in the current v2.5.0 catalog. This documents the existence
    of a known pre-catalog loss and confirms the current catalog is
    regression-free (only r103 in v2.5.0).
    """
    census = _walk_real_search_native_evaluator()
    losses = census["losses"]
    assert len(losses) == 1, f"H3 fails: expected 1 loss, got {len(losses)}: {losses}"
    assert losses[0]["task_id"] == "caddyserver__caddy-5404", (
        f"H3 fails: expected caddy-5404 loss, got {losses[0]['task_id']}"
    )
    assert losses[0]["run"] == "real-search-001-deepseek-v3", (
        f"H3 fails: expected v2.1.0 run, got {losses[0]['run']}"
    )

    # Verify caddy-5404 is NOT in the current catalog (it's a pre-catalog loss)
    records = _load_records()
    caddy_in_catalog = [r for r in records if "caddy" in json.dumps(r).lower()]
    assert len(caddy_in_catalog) == 0, (
        f"H3: caddy-5404 should not be in v2.5.0 catalog, found: {caddy_in_catalog}"
    )


def test_regression_scan_persisted(tmp_path) -> None:
    """The scan result files should exist in tmp_path (NOT in a sealed round dir).

    The sealed round directory was used historically but is now immutable.
    Tests that want a persisted scan must write to tmp_path and the
    persistence test verifies the tmp_path copy, not the round dir.
    """
    # Run a fresh scan to ensure the tmp_path file is regenerated
    records = _load_records()
    feedback = [r for r in records if "feedback" in (r.get("task_tags") or [])]
    classified = {}
    for r in feedback:
        cls, ev = _classify_feedback_record(r)
        classified.setdefault(cls, []).append(ev)
    # Apply unique-task counting (same as the main test)
    supersede_pairs = set()
    for r in records:
        for sup in (r.get("payload") or {}).get("supersedes", []) or []:
            supersede_pairs.add((sup, r.get("record_id")))

    def _is_superseded(rec_id: str) -> bool:
        for sup, sup_of in supersede_pairs:
            if rec_id == sup and sup_of != rec_id:
                return True
        return False

    unique_gains = [
        g for g in classified.get("gain", []) if not _is_superseded(g["record_id"])
    ]
    scan = {
        "round": "20260813T134000Z-feedback-regression-scan",
        "feedback_records_total": len(feedback),
        "classification_counts": {k: len(v) for k, v in classified.items()},
        "unique_gain_task_count": len(unique_gains),
    }
    scan_path = tmp_path / "feedback-regression-scan.json"
    with open(scan_path, "w") as f:
        json.dump(scan, f, indent=2)
    census = _walk_real_search_native_evaluator()
    census_path = tmp_path / "cross-project-gain-census.json"
    with open(census_path, "w") as f:
        json.dump(census, f, indent=2)
    assert scan_path.exists(), f"Missing {scan_path}"
    assert census_path.exists(), f"Missing {census_path}"
    scan = json.load(open(scan_path))
    census = json.load(open(census_path))
    assert scan["feedback_records_total"] == 34
    assert scan["unique_gain_task_count"] == 6
    assert census["totals"]["gains"] == 1
    assert census["totals"]["losses"] == 1
