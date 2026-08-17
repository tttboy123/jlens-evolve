"""Tests for the v2.3 search->candidate-skill bridge and cross-task transfer gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evolution_fixture import run_fixture
from search_skill_bridge import (
    apply_transfer_gate,
    compile_candidate_skills,
    evaluate_transfer_gate,
)
from skill_registry import SkillRegistry


def _fixture_result(tmp_path: Path) -> dict:
    run_root = tmp_path / "run"
    result = run_fixture(run_root)
    assert result["status"] == "completed"
    return result


def _paired_evals(
    *,
    tasks: int = 10,
    candidate_score: float = 0.55,
    candidate_safety: bool = True,
    candidate_cost: float = 1.05,
) -> list[dict]:
    rows = []
    contract = hashlib.sha256(b"contract").hexdigest()
    epoch = "native-adapters-v2.1.0-frozen"
    for index in range(tasks):
        task = f"fresh-task-{index:03d}"
        rows.append(
            {
                "task_uid": task,
                "benchmark_family": "fixture-family",
                "role": "original",
                "native_score": 0.5,
                "safety_passed": True,
                "cost_units": 1.0,
                "matched_contract_sha256": contract,
                "native_evaluator_epoch": epoch,
            }
        )
        rows.append(
            {
                "task_uid": task,
                "benchmark_family": "fixture-family",
                "role": "candidate",
                "native_score": candidate_score,
                "safety_passed": candidate_safety,
                "cost_units": candidate_cost,
                "matched_contract_sha256": contract,
                "native_evaluator_epoch": epoch,
            }
        )
    return rows


def test_compile_from_fixture_is_deterministic_and_inactive(tmp_path):
    run_root = tmp_path / "run"
    result = run_fixture(run_root)

    first = compile_candidate_skills(result=result, run_root=run_root)
    second = compile_candidate_skills(result=result, run_root=run_root)

    assert first
    assert [row.candidate_fingerprint for row in first] == [
        row.candidate_fingerprint for row in second
    ]
    for candidate in first:
        assert candidate.status == "candidate"
        assert candidate.project_local_only is True
        assert candidate.auto_install is False
        assert candidate.active is False
        assert candidate.counterexamples
        assert candidate.known_failure_modes
        assert candidate.evidence_refs
        assert candidate.applicability["required_semantics"]


def test_registry_roundtrip_and_retrieve(tmp_path):
    run_root = tmp_path / "run"
    result = run_fixture(run_root)
    candidates = compile_candidate_skills(result=result, run_root=run_root)
    registry = SkillRegistry(tmp_path / "registry")
    for candidate in candidates:
        assert registry.append(candidate) is True
    assert registry.read_revisions() == list(candidates)
    latest = registry.latest(candidates[0].skill_id)
    assert latest is not None
    assert latest.status == "candidate"


def test_transfer_gate_pass_and_fail(tmp_path):
    run_root = tmp_path / "run"
    result = run_fixture(run_root)
    candidates = compile_candidate_skills(result=result, run_root=run_root)
    candidate = candidates[0]
    contract = hashlib.sha256(b"contract").hexdigest()
    epoch = "native-adapters-v2.1.0-frozen"

    passed = evaluate_transfer_gate(
        paired_evals=_paired_evals(),
        expected_contract_sha256=contract,
        expected_evaluator_epoch=epoch,
    )
    assert passed.passed is True
    assert passed.paired_tasks == 10

    registry = SkillRegistry(tmp_path / "registry")
    registry.append(candidate)
    gate_path = tmp_path / "TRANSFER-GATE-pass.json"
    gate_path.write_text(json.dumps(passed.to_dict()), encoding="utf-8")
    terminal = apply_transfer_gate(
        registry=registry,
        candidate=candidate,
        gate_result=passed,
        gate_evidence_path=gate_path,
    )
    assert terminal.status == "transfer_verified"
    assert terminal.active is False and terminal.auto_install is False

    # safety regression -> rejected
    unsafe = evaluate_transfer_gate(
        paired_evals=_paired_evals(candidate_safety=False),
        expected_contract_sha256=contract,
        expected_evaluator_epoch=epoch,
    )
    assert unsafe.passed is False
    assert "safety_regression" in unsafe.reasons

    # score regression -> rejected
    regressed = evaluate_transfer_gate(
        paired_evals=_paired_evals(candidate_score=0.40),
        expected_contract_sha256=contract,
        expected_evaluator_epoch=epoch,
    )
    assert regressed.passed is False
    assert any("native_score_regression" in reason for reason in regressed.reasons)

    # cost over limit -> rejected
    costly = evaluate_transfer_gate(
        paired_evals=_paired_evals(candidate_cost=1.20),
        expected_contract_sha256=contract,
        expected_evaluator_epoch=epoch,
    )
    assert costly.passed is False
    assert any("cost_increase_over_limit" in reason for reason in costly.reasons)

    # too few paired tasks -> rejected
    few = evaluate_transfer_gate(
        paired_evals=_paired_evals(tasks=3),
        expected_contract_sha256=contract,
        expected_evaluator_epoch=epoch,
    )
    assert few.passed is False
    assert any("paired_tasks_below_min" in reason for reason in few.reasons)


def test_cli_skill_candidates_smoke(tmp_path):
    from evolve_service import run_cli

    run_root = tmp_path / "run"
    run_fixture(run_root)
    registry_root = tmp_path / "registry"
    candidates = compile_candidate_skills(
        result=json.loads((run_root / "RESULT.json").read_text(encoding="utf-8")),
        run_root=run_root,
    )
    assert candidates
    payload = {
        candidates[0].skill_id: {
            "evals": _paired_evals(),
            "expected_contract_sha256": hashlib.sha256(b"contract").hexdigest(),
            "expected_evaluator_epoch": "native-adapters-v2.1.0-frozen",
        }
    }
    paired_path = tmp_path / "paired.json"
    paired_path.write_text(json.dumps(payload), encoding="utf-8")
    code = run_cli(
        [
            "skill-candidates",
            "--run-root",
            str(run_root),
            "--registry",
            str(registry_root),
            "--paired-evidences",
            str(paired_path),
        ]
    )
    assert code == 0
    registry = SkillRegistry(registry_root)
    assert registry.read_revisions()
    assert (run_root / "CANDIDATE-SKILLS.json").is_file()
    assert (run_root / f"TRANSFER-GATE-{candidates[0].skill_id}.json").is_file()


def test_extract_confirmation_paired_evals_from_fixture(tmp_path):
    from search_skill_bridge import extract_confirmation_paired_evals

    run_root = tmp_path / "run"
    run_fixture(run_root)
    for generation in (1, 2, 3):
        paired = extract_confirmation_paired_evals(
            run_root=run_root, generation=generation
        )
        assert len(paired["evals"]) >= 8 * 2
        # fixture uses per-task matched contracts, so uniform contract may be None;
        # the real run uses one frozen baseline contract (uniform).
        assert paired["expected_evaluator_epoch"]
        roles = {row["role"] for row in paired["evals"]}
        assert roles == {"original", "candidate"}


def test_finalize_skill_bridge_auto_gate(tmp_path):
    from search_skill_bridge import finalize_skill_bridge
    from skill_registry import SkillRegistry

    run_root = tmp_path / "run"
    result = run_fixture(run_root)
    summary = finalize_skill_bridge(
        result=result,
        run_root=run_root,
        registry_root=tmp_path / "registry",
        auto_gate=True,
    )
    assert summary["compiled"]
    assert len(summary["applied_gates"]) == len(summary["compiled"])
    registry = SkillRegistry(tmp_path / "registry")
    revisions = registry.read_revisions()
    assert len(revisions) >= len(summary["compiled"])
    for revision in revisions:
        assert revision.active is False and revision.auto_install is False
    assert (run_root / "CANDIDATE-SKILLS.json").is_file()
    for entry in summary["applied_gates"]:
        assert entry["status"] in {"transfer_verified", "rejected"}


def test_real_evolution_run_parser_accepts_v22_v23_flags():
    source = Path(__file__).resolve().parents[1] / "real_evolution_run.py"
    text = source.read_text(encoding="utf-8")
    for flag in ("--worker-count", "--skill-registry", "--auto-gate"):
        assert flag in text
