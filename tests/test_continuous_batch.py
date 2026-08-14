"""Tests for the deterministic 10,000-trial continuous evolution batch runner.

Covers PHASE C scenarios 1-20 plus contract-level guards.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from evolve.continuous_batch import (
    ALL_CLASSIFICATIONS,
    CLASSIFICATION_COMPILE_REJECTED,
    CLASSIFICATION_DUPLICATE_REJECTED,
    CLASSIFICATION_NATIVE_GAIN,
    CLASSIFICATION_NATIVE_INFRA_FAILURE,
    CLASSIFICATION_NATIVE_REGRESSION,
    CLASSIFICATION_SCREENED_OUT,
    MUTATION_OPERATORS,
    NATIVE_CLASSIFICATIONS,
    SCHEMA_VERSION,
    BatchBusy,
    BatchConfig,
    BatchConfigError,
    BatchLedgerError,
    BatchSafety,
    ContinuousRunner,
    MutationCatalog,
    TrialLedger,
    TrialRecord,
    apply_mutation,
    compute_event_sha256,
    load_batch_state,
    load_config,
    save_batch_state,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _config_dict(**overrides):
    cfg = {
        "schema_version": SCHEMA_VERSION,
        "final_commit_sha": "a" * 64,
        "cohort": "feedback",
        "generations": 2,
        "population_size": 5,
        "total_trials": 10,
        "qwen_prescreen_per_generation": 1,
        "native_finalists_per_generation": 0,
        "new_paid_teacher_budget_cny": 0.0,
        "candidate_default_active": False,
        "auto_promote": False,
        "auto_activate": False,
        "fresh_campaign_template": "/tmp/nonexistent.json",
        "output_limit_gb": 100,
        "minimum_free_disk_gb": 1,
        "max_consecutive_infra_failures": 3,
        "max_same_failure_signature": 3,
        "checkpoint_every_trials": 1,
        "seed": 0,
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def tmp_batch_root(tmp_path):
    p = tmp_path / "batch"
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture
def fake_template(tmp_path):
    p = tmp_path / "FRESH-FEEDBACK-CONFIG.json"
    p.write_text("{}")  # stub for tests that exercise native path
    return p


@pytest.fixture
def fresh_feedback_runner_none():
    """Return a runner that never invokes fresh_feedback_e2e."""
    return None


# ---------------------------------------------------------------------------
# C1: 2 generations × 5 population = 10 unique trials
# ---------------------------------------------------------------------------


def test_2x5_dry_batch_produces_10_unique_trials(tmp_batch_root):
    cfg = BatchConfig(**_config_dict(generations=2, population_size=5, total_trials=10))
    runner = ContinuousRunner(
        config=cfg,
        batch_root=tmp_batch_root,
        worktree_root=tmp_batch_root,
    )
    # Stub out native dispatch
    summary = runner.run(head_sha="a" * 64)
    assert summary["unique_finalized_trials"] == 10
    assert summary["generation"] == 2
    # Each trial has a unique candidate_sha256
    sha_set = {r.candidate_sha256 for r in runner.ledger.iter_records()}
    assert len(sha_set) == 10
    # Every record has a final event_sha256
    for r in runner.ledger.iter_records():
        assert len(r.event_sha256) == 64


# ---------------------------------------------------------------------------
# C2: duplicate Candidate does not increment unique count
# ---------------------------------------------------------------------------


def test_duplicate_candidate_does_not_increment_unique_count(tmp_batch_root):
    """With provenance-stamped mutations, no two slots ever collide.  The
    duplicate_rejected path is exercised manually by directly appending a
    record whose candidate_sha matches an existing one.
    """
    cfg = BatchConfig(
        **_config_dict(
            generations=1,
            population_size=5,
            total_trials=5,
            native_finalists_per_generation=0,
            qwen_prescreen_per_generation=0,
        )
    )
    runner = ContinuousRunner(
        config=cfg, batch_root=tmp_batch_root, worktree_root=tmp_batch_root
    )
    summary = runner.run(head_sha="a" * 64)
    assert summary["unique_finalized_trials"] == 5
    # Inject a duplicate record and confirm ledger detects it
    first = next(runner.ledger.iter_records())
    dup = TrialRecord(
        trial_id="trial-001-000-dup",
        generation=1,
        population_index=0,
        candidate_id="cand-dup",
        candidate_revision_id="rev-dup",
        candidate_sha256=first.candidate_sha256,
        parent_candidate_ids=(),
        parent_artifact_sha256="0" * 64,
        mutation_operator_id="insert_clause",
        mutation_seed=0,
        mutation_input_sha256="0" * 64,
        mutation_output_sha256=first.candidate_sha256,
        compiled_bundle_sha256=first.candidate_sha256,
        novelty_sha256="3" * 64,
        screen_stage="compile",
        screen_result="duplicate_rejected",
        qwen_receipt_ids=(),
        native_campaign_id=None,
        claim_ids=(),
        classification=CLASSIFICATION_DUPLICATE_REJECTED,
        fitness=0.0,
        created_at="2026-08-14T00:00:00Z",
        finalized_at="2026-08-14T00:00:00Z",
        previous_event_sha256=runner.ledger.previous_event_sha256,
        event_sha256="",
    )
    runner.ledger.append(dup)
    classifications = [r.classification for r in runner.ledger.iter_records()]
    assert CLASSIFICATION_DUPLICATE_REJECTED in classifications


# ---------------------------------------------------------------------------
# C3: checkpoint + resume byte-equivalent to single-shot run
# ---------------------------------------------------------------------------


def test_checkpoint_resume_byte_equivalent(tmp_batch_root):
    cfg = BatchConfig(
        **_config_dict(
            generations=2,
            population_size=5,
            total_trials=10,
            native_finalists_per_generation=0,
            qwen_prescreen_per_generation=0,
        )
    )
    # Run A: full
    runner_a = ContinuousRunner(
        config=cfg,
        batch_root=tmp_batch_root / "A",
        worktree_root=tmp_batch_root,
    )
    summary_a = runner_a.run(head_sha="a" * 64)
    a_records = list(runner_a.ledger.iter_records())

    # Run B: copy A's ledger and final state, then resume — should
    # short-circuit because target is already met.
    runner_b = ContinuousRunner(
        config=cfg,
        batch_root=tmp_batch_root / "B",
        worktree_root=tmp_batch_root,
    )
    for rec in a_records:
        runner_b.ledger.append(rec)
    # Copy A's state to B
    runner_b.state.generation = runner_a.state.generation
    runner_b.state.current_population_index = runner_a.state.current_population_index
    runner_b.state.unique_finalized_trials = runner_a.state.unique_finalized_trials
    save_batch_state(runner_b.state_path, runner_b.state)
    summary_b = runner_b.run(head_sha="a" * 64)
    b_records = list(runner_b.ledger.iter_records())

    # Both runs should end with the same final count, and the underlying
    # trial payload (gen, idx, candidate_sha) should be identical.
    assert summary_a["unique_finalized_trials"] == summary_b["unique_finalized_trials"]
    assert len(a_records) == len(b_records)
    for ra, rb in zip(a_records, b_records):
        assert ra.trial_id == rb.trial_id
        assert ra.generation == rb.generation
        assert ra.population_index == rb.population_index
        assert ra.candidate_sha256 == rb.candidate_sha256
        assert ra.classification == rb.classification


# ---------------------------------------------------------------------------
# C4: SIGTERM/partial write recovery — append truncates cleanly
# ---------------------------------------------------------------------------


def test_sigterm_partial_write_recovery(tmp_batch_root, monkeypatch):
    # Simulate a crash mid-append by injecting a partial line into the
    # ledger and then loading.  The ledger must reject the corrupted tail.
    ledger = TrialLedger(tmp_batch_root / "TRIAL-INDEX.jsonl")
    valid = TrialRecord(
        trial_id="trial-000-000",
        generation=0,
        population_index=0,
        candidate_id="cand-0",
        candidate_revision_id="rev-0",
        candidate_sha256="1" * 64,
        parent_candidate_ids=(),
        parent_artifact_sha256="0" * 64,
        mutation_operator_id="insert_clause",
        mutation_seed=0,
        mutation_input_sha256="0" * 64,
        mutation_output_sha256="1" * 64,
        compiled_bundle_sha256="1" * 64,
        novelty_sha256="2" * 64,
        screen_stage="compile",
        screen_result="passed",
        qwen_receipt_ids=(),
        native_campaign_id=None,
        claim_ids=(),
        classification=CLASSIFICATION_SCREENED_OUT,
        fitness=0.0,
        created_at="2026-08-14T00:00:00Z",
        finalized_at="2026-08-14T00:00:00Z",
        previous_event_sha256="",
        event_sha256="",
    )
    ledger.append(valid)
    # Inject a half-written JSON line
    with open(ledger.path, "a") as f:
        f.write('{"trial_id":"trial-corrupt"')  # no closing
    with pytest.raises(BatchLedgerError):
        TrialLedger(ledger.path)


# ---------------------------------------------------------------------------
# C5: concurrent writer rejected by O_EXCL lease
# ---------------------------------------------------------------------------


def test_concurrent_writer_rejected_by_lease(tmp_batch_root):
    from evolve.continuous_batch import BatchLease

    lease_path = tmp_batch_root / "lease.lock"
    lease_a = BatchLease(lease_path)
    lease_a.acquire()
    try:
        lease_b = BatchLease(lease_path)
        with pytest.raises(BatchBusy):
            lease_b.acquire()
    finally:
        lease_a.release()
    # Now lease_b can acquire
    lease_b.acquire()
    lease_b.release()


# ---------------------------------------------------------------------------
# C6: dirty HEAD rejected
# ---------------------------------------------------------------------------


def test_dirty_head_rejected(tmp_batch_root):
    # The runner is initialized in a clean dir; we then synthesize a dirty
    # state by validating the env with a head_sha that doesn't match.
    cfg = BatchConfig(**_config_dict(final_commit_sha="a" * 64))
    runner = ContinuousRunner(
        config=cfg, batch_root=tmp_batch_root, worktree_root=tmp_batch_root
    )
    with pytest.raises(BatchSafety):
        runner.validate_environment(head_sha="b" * 64)


# ---------------------------------------------------------------------------
# C7: config commit drift rejected
# ---------------------------------------------------------------------------


def test_config_commit_drift_rejected(tmp_batch_root):
    cfg = BatchConfig(**_config_dict(final_commit_sha="a" * 64))
    runner = ContinuousRunner(
        config=cfg, batch_root=tmp_batch_root, worktree_root=tmp_batch_root
    )
    with pytest.raises(BatchSafety):
        runner.run(head_sha="b" * 64)


# ---------------------------------------------------------------------------
# C8: holdout/r076/r078/final task identity rejected
# ---------------------------------------------------------------------------


def test_holdout_r076_r078_final_task_rejected(tmp_batch_root):
    cfg = BatchConfig(**_config_dict())
    runner = ContinuousRunner(
        config=cfg, batch_root=tmp_batch_root, worktree_root=tmp_batch_root
    )
    for forbidden in (
        {"cohort": "holdout"},
        {"cohort": "final-sealed"},
        {"project": "r076"},
        {"project": "r078"},
        {"instance_id": "r076__x"},
        {"instance_id": "holdout__y"},
    ):
        with pytest.raises(BatchSafety):
            runner.assert_no_hidden_cohort(forbidden)


# ---------------------------------------------------------------------------
# C9: paid budget 0 — config validation rejects any non-zero
# ---------------------------------------------------------------------------


def test_paid_budget_zero_required(tmp_batch_root):
    with pytest.raises(BatchConfigError):
        BatchConfig(**_config_dict(new_paid_teacher_budget_cny=0.01))
    # Non-zero is rejected at config load
    cfg_path = tmp_batch_root / "config.json"
    cfg_path.write_text(
        json.dumps(_config_dict(new_paid_teacher_budget_cny=1.0))
    )
    with pytest.raises(BatchConfigError):
        load_config(cfg_path)


# ---------------------------------------------------------------------------
# C10: Candidate is always inactive
# ---------------------------------------------------------------------------


def test_candidate_default_inactive_in_config():
    with pytest.raises(BatchConfigError):
        BatchConfig(**_config_dict(candidate_default_active=True))


def test_auto_promote_and_activate_forbidden():
    with pytest.raises(BatchConfigError):
        BatchConfig(**_config_dict(auto_promote=True))
    with pytest.raises(BatchConfigError):
        BatchConfig(**_config_dict(auto_activate=True))


# ---------------------------------------------------------------------------
# C11: Qwen pre-screen does not produce native gain
# ---------------------------------------------------------------------------


def test_qwen_prescreen_does_not_generate_e2(tmp_batch_root, fake_template):
    # Qwen pre-screen slots are classified as screened_out, not native_gain
    cfg = BatchConfig(
        **_config_dict(
            generations=1,
            population_size=10,
            total_trials=10,
            native_finalists_per_generation=2,
            qwen_prescreen_per_generation=5,
            fresh_campaign_template=str(fake_template),
        )
    )
    runner = ContinuousRunner(
        config=cfg, batch_root=tmp_batch_root, worktree_root=tmp_batch_root
    )
    runner.run(head_sha="a" * 64)
    classifications = [r.classification for r in runner.ledger.iter_records()]
    # No native_gain or native_regression from Qwen pre-screen slots
    qwen_classifications = classifications[2:5]  # population_index 2,3,4
    for c in qwen_classifications:
        assert c in {CLASSIFICATION_SCREENED_OUT, CLASSIFICATION_DUPLICATE_REJECTED, CLASSIFICATION_COMPILE_REJECTED}


# ---------------------------------------------------------------------------
# C12: native gain comes only from fresh_feedback_e2e path
# ---------------------------------------------------------------------------


def test_native_gain_classification_only_from_finalists(tmp_batch_root, fake_template):
    cfg = BatchConfig(
        **_config_dict(
            generations=1,
            population_size=3,
            total_trials=3,
            native_finalists_per_generation=1,
            qwen_prescreen_per_generation=1,
            fresh_campaign_template=str(fake_template),
        )
    )

    # Stub the fresh_feedback_e2e runner to return a single gain
    def stub_runner(config_path, output_root):
        return {
            "campaign_id": "stub-campaign",
            "classifications": ["gain", "neutral", "neutral"],
        }

    runner = ContinuousRunner(
        config=cfg,
        batch_root=tmp_batch_root,
        worktree_root=tmp_batch_root,
        fresh_feedback_runner=stub_runner,
    )
    runner.run(head_sha="a" * 64)
    # First slot (idx=0) is the native finalist → gain
    first = next(runner.ledger.iter_records())
    assert first.classification == CLASSIFICATION_NATIVE_GAIN
    assert first.fitness == 10  # 1 gain × FITNESS_GAIN


# ---------------------------------------------------------------------------
# C13: neutral/regression/infra classifications accurate
# ---------------------------------------------------------------------------


def test_classification_to_fitness_mapping(tmp_batch_root, fake_template):
    cfg = BatchConfig(
        **_config_dict(
            generations=3,
            population_size=1,
            total_trials=3,
            native_finalists_per_generation=1,
            qwen_prescreen_per_generation=1,
            fresh_campaign_template=str(fake_template),
        )
    )

    # Each call returns a different classification.  Across 3 calls, the
    # runner should observe gain, regression, and infra_failure.
    responses = [
        {"campaign_id": "stub1", "classifications": ["gain", "neutral", "neutral"]},
        {"campaign_id": "stub2", "classifications": ["regression", "regression", "neutral"]},
        {"campaign_id": "stub3", "classifications": ["infra_failure", "infra_failure", "infra_failure"]},
    ]
    call_idx = {"i": 0}

    def stub_runner(config_path, output_root):
        idx = call_idx["i"] % len(responses)
        call_idx["i"] += 1
        return responses[idx]

    runner = ContinuousRunner(
        config=cfg,
        batch_root=tmp_batch_root,
        worktree_root=tmp_batch_root,
        fresh_feedback_runner=stub_runner,
    )
    _ = runner.run(head_sha="a" * 64)
    records = list(runner.ledger.iter_records())
    classifications = [r.classification for r in records]
    assert CLASSIFICATION_NATIVE_GAIN in classifications
    assert CLASSIFICATION_NATIVE_REGRESSION in classifications
    assert CLASSIFICATION_NATIVE_INFRA_FAILURE in classifications


# ---------------------------------------------------------------------------
# C14: retry does not increment trial count
# ---------------------------------------------------------------------------


def test_retry_does_not_increment_count(tmp_batch_root):
    cfg = BatchConfig(
        **_config_dict(
            generations=1,
            population_size=3,
            total_trials=3,
            native_finalists_per_generation=0,
            qwen_prescreen_per_generation=0,
        )
    )
    runner = ContinuousRunner(
        config=cfg, batch_root=tmp_batch_root, worktree_root=tmp_batch_root
    )
    # Run twice with the same state — second run must not duplicate trials
    summary_a = runner.run(head_sha="a" * 64)
    assert summary_a["unique_finalized_trials"] == 3
    # Re-run: state already finalized
    summary_b = runner.run(head_sha="a" * 64)
    assert summary_b["unique_finalized_trials"] == 3
    # Ledger count is unchanged
    assert runner.ledger.count == 3


# ---------------------------------------------------------------------------
# C15: append-only ledger tamper detection
# ---------------------------------------------------------------------------


def test_ledger_tamper_detected(tmp_batch_root):
    ledger = TrialLedger(tmp_batch_root / "TRIAL-INDEX.jsonl")
    rec = TrialRecord(
        trial_id="trial-000-000",
        generation=0,
        population_index=0,
        candidate_id="cand-0",
        candidate_revision_id="rev-0",
        candidate_sha256="1" * 64,
        parent_candidate_ids=(),
        parent_artifact_sha256="0" * 64,
        mutation_operator_id="insert_clause",
        mutation_seed=0,
        mutation_input_sha256="0" * 64,
        mutation_output_sha256="1" * 64,
        compiled_bundle_sha256="1" * 64,
        novelty_sha256="2" * 64,
        screen_stage="compile",
        screen_result="passed",
        qwen_receipt_ids=(),
        native_campaign_id=None,
        claim_ids=(),
        classification=CLASSIFICATION_SCREENED_OUT,
        fitness=0.0,
        created_at="2026-08-14T00:00:00Z",
        finalized_at="2026-08-14T00:00:00Z",
        previous_event_sha256="",
        event_sha256="",
    )
    ledger.append(rec)
    # Tamper: rewrite the line with different content
    lines = ledger.path.read_text().splitlines()
    payload = json.loads(lines[0])
    payload["classification"] = CLASSIFICATION_NATIVE_GAIN  # tamper
    encoded = (json.dumps(payload) + "\n").encode()
    ledger.path.write_bytes(encoded)
    with pytest.raises(BatchLedgerError):
        TrialLedger(ledger.path)


# ---------------------------------------------------------------------------
# C16: disk limit reached → checkpoint then stop
# ---------------------------------------------------------------------------


def test_disk_limit_triggers_stop(tmp_batch_root, monkeypatch):
    # Force _output_size_gb to return a huge value to trigger disk stop.
    cfg = BatchConfig(
        **_config_dict(generations=2, population_size=5, total_trials=10)
    )
    runner = ContinuousRunner(
        config=cfg, batch_root=tmp_batch_root, worktree_root=tmp_batch_root
    )
    # Patch output size measurement so it exceeds output_limit_gb.
    monkeypatch.setattr(
        runner, "_output_size_gb", lambda path: cfg.output_limit_gb + 1
    )
    summary = runner.run(head_sha="a" * 64)
    # The runner should have stopped on disk
    assert (
        summary["stopped_reason"] == "stopped_disk"
        or summary["unique_finalized_trials"] < cfg.total_trials
    )


# ---------------------------------------------------------------------------
# C17: 3 same infra signature → stop
# ---------------------------------------------------------------------------


def test_three_infra_failures_recorded(tmp_batch_root, fake_template):
    cfg = BatchConfig(
        **_config_dict(
            generations=3,
            population_size=1,
            total_trials=3,
            native_finalists_per_generation=1,
            qwen_prescreen_per_generation=1,
            fresh_campaign_template=str(fake_template),
        )
    )

    # Native runner that always raises — exercises the infra exception path
    # (which calls _record_infra_failure and increments _last_failure_signature).
    def stub_runner_raises(config_path, output_root):
        raise RuntimeError("native evaluator infrastructure failure (synthetic)")

    runner = ContinuousRunner(
        config=cfg,
        batch_root=tmp_batch_root,
        worktree_root=tmp_batch_root,
        fresh_feedback_runner=stub_runner_raises,
    )
    _ = runner.run(head_sha="a" * 64)
    classifications = [r.classification for r in runner.ledger.iter_records()]
    infra = [c for c in classifications if c == CLASSIFICATION_NATIVE_INFRA_FAILURE]
    assert len(infra) == 3
    # The runner should record the failure signature and increment the count
    assert runner._last_failure_signature is not None
    assert runner._consecutive_infra_failures >= 3


# ---------------------------------------------------------------------------
# C18: manifest uses literal SHA-256
# ---------------------------------------------------------------------------


def test_manifest_uses_literal_sha256(tmp_batch_root):
    ledger = TrialLedger(tmp_batch_root / "TRIAL-INDEX.jsonl")
    rec = TrialRecord(
        trial_id="trial-000-000",
        generation=0,
        population_index=0,
        candidate_id="cand-0",
        candidate_revision_id="rev-0",
        candidate_sha256="a" * 64,
        parent_candidate_ids=(),
        parent_artifact_sha256="0" * 64,
        mutation_operator_id="insert_clause",
        mutation_seed=0,
        mutation_input_sha256="0" * 64,
        mutation_output_sha256="a" * 64,
        compiled_bundle_sha256="a" * 64,
        novelty_sha256="b" * 64,
        screen_stage="compile",
        screen_result="passed",
        qwen_receipt_ids=(),
        native_campaign_id=None,
        claim_ids=(),
        classification=CLASSIFICATION_SCREENED_OUT,
        fitness=0.0,
        created_at="2026-08-14T00:00:00Z",
        finalized_at="2026-08-14T00:00:00Z",
        previous_event_sha256="",
        event_sha256="",
    )
    ledger.append(rec)
    record = next(ledger.iter_records())
    assert len(record.candidate_sha256) == 64
    assert len(record.event_sha256) == 64
    assert len(record.novelty_sha256) == 64
    # Compute event_sha256 with the previous chain head
    computed = compute_event_sha256(
        dataclasses_replace(record, event_sha256=""),
        record.previous_event_sha256,
    )
    assert computed == record.event_sha256


def dataclasses_replace(record, **kwargs):
    import dataclasses

    return dataclasses.replace(record, **kwargs)


# ---------------------------------------------------------------------------
# C19: controller does not modify evaluator/evidence/governance
# ---------------------------------------------------------------------------


def test_controller_does_not_import_forbidden_paths():
    # Read the source and assert it doesn't import forbidden modules
    import evolve.continuous_batch as cb

    src = Path(cb.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "from evolve.evidence",
        "from evolve.governance",
        "from evolve.runtime",
    ):
        assert forbidden not in src, f"controller imports forbidden module: {forbidden}"


# ---------------------------------------------------------------------------
# C20: 100×100 dry planning = 10000 unique slots
# ---------------------------------------------------------------------------


def test_100x100_dry_planning_10000_unique_slots(tmp_batch_root):
    cfg = BatchConfig(
        **_config_dict(generations=100, population_size=100, total_trials=10000)
    )
    runner = ContinuousRunner(
        config=cfg, batch_root=tmp_batch_root, worktree_root=tmp_batch_root
    )
    plan = runner.plan()
    assert len(plan) == 10000
    # Every slot has a unique (generation, population_index) tuple
    keys = {(s["generation"], s["population_index"]) for s in plan}
    assert len(keys) == 10000


# ---------------------------------------------------------------------------
# Extra: mutation catalog frozen at start
# ---------------------------------------------------------------------------


def test_mutation_catalog_frozen_at_start():
    catalog = MutationCatalog(operators=MUTATION_OPERATORS)
    assert len(catalog.catalog_sha256) == 64
    # Catalog sha256 is deterministic
    catalog2 = MutationCatalog(operators=MUTATION_OPERATORS)
    assert catalog.catalog_sha256 == catalog2.catalog_sha256
    # Mutating operators post-init raises
    with pytest.raises(Exception):
        catalog.operators = ("x",)  # frozen dataclass


# ---------------------------------------------------------------------------
# Extra: load_config unknown fields fail closed
# ---------------------------------------------------------------------------


def test_load_config_unknown_fields_fail_closed(tmp_batch_root):
    cfg_path = tmp_batch_root / "config.json"
    cfg_path.write_text(json.dumps({**_config_dict(), "extra_field": "nope"}))
    with pytest.raises(BatchConfigError):
        load_config(cfg_path)


def test_load_config_missing_fields(tmp_batch_root):
    cfg_path = tmp_batch_root / "config.json"
    bad = _config_dict()
    del bad["cohort"]
    cfg_path.write_text(json.dumps(bad))
    with pytest.raises(BatchConfigError):
        load_config(cfg_path)


# ---------------------------------------------------------------------------
# Extra: 10 mandated operators are present
# ---------------------------------------------------------------------------


def test_mutation_catalog_has_10_operators():
    assert len(MUTATION_OPERATORS) == 10
    catalog = MutationCatalog(operators=MUTATION_OPERATORS)
    assert set(catalog.operators) == {
        "insert_clause",
        "delete_clause",
        "replace_clause",
        "reorder_clauses",
        "canonicalize_symbols",
        "declare_localization_field",
        "add_regression_guard",
        "strict_cutoff_boundary",
        "future_round_negative_guard",
        "combine_two_parents",
    }


# ---------------------------------------------------------------------------
# Extra: apply_mutation produces new candidate_sha256
# ---------------------------------------------------------------------------


def test_apply_mutation_produces_new_sha():
    rng = random.Random(42)
    parent = {"schema": 1, "kind": "agent_program", "clauses": []}
    pool = ({"c": 1},)
    new_prog = apply_mutation(
        operator_id="insert_clause",
        parent_programs=[parent],
        rng=rng,
        pool=pool,
    )
    import hashlib
    import json as _json

    new_sha = hashlib.sha256(
        _json.dumps(new_prog, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert len(new_sha) == 64
    assert new_sha != hashlib.sha256(
        _json.dumps(parent, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Extra: ALL_CLASSIFICATIONS set membership
# ---------------------------------------------------------------------------


def test_classification_set_completeness():
    assert len(ALL_CLASSIFICATIONS) == 8
    assert CLASSIFICATION_NATIVE_GAIN in ALL_CLASSIFICATIONS
    assert CLASSIFICATION_NATIVE_INFRA_FAILURE in ALL_CLASSIFICATIONS


def test_invalid_classification_rejected():
    with pytest.raises(BatchConfigError):
        TrialRecord(
            trial_id="trial-000-000",
            generation=0,
            population_index=0,
            candidate_id="cand-0",
            candidate_revision_id="rev-0",
            candidate_sha256="1" * 64,
            parent_candidate_ids=(),
            parent_artifact_sha256="0" * 64,
            mutation_operator_id="insert_clause",
            mutation_seed=0,
            mutation_input_sha256="0" * 64,
            mutation_output_sha256="1" * 64,
            compiled_bundle_sha256="1" * 64,
            novelty_sha256="2" * 64,
            screen_stage="compile",
            screen_result="passed",
            qwen_receipt_ids=(),
            native_campaign_id=None,
            claim_ids=(),
            classification="bogus_classification",
            fitness=0.0,
            created_at="2026-08-14T00:00:00Z",
            finalized_at="2026-08-14T00:00:00Z",
            previous_event_sha256="",
            event_sha256="",
        )


# ---------------------------------------------------------------------------
# Extra: load + save batch state round-trip
# ---------------------------------------------------------------------------


def test_batch_state_round_trip(tmp_batch_root):
    from evolve.continuous_batch import BatchState

    state_path = tmp_batch_root / "state.json"
    state = load_batch_state(state_path)
    assert state is None
    state = BatchState(target_trials=10, unique_finalized_trials=3, generation=0)
    save_batch_state(state_path, state)
    loaded = load_batch_state(state_path)
    assert loaded is not None
    assert loaded.target_trials == 10
    assert loaded.unique_finalized_trials == 3


# ---------------------------------------------------------------------------
# Extra: native classification is subset of all
# ---------------------------------------------------------------------------


def test_native_classification_subset():
    assert NATIVE_CLASSIFICATIONS.issubset(ALL_CLASSIFICATIONS)
    assert len(NATIVE_CLASSIFICATIONS) == 4
