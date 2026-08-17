from __future__ import annotations

import json
from pathlib import Path

from supervised_evolution_poc import build_supervision_plan, run_poc

ROOT = Path(__file__).resolve().parents[1]


def test_supervisor_builds_bounded_plan_from_observational_evidence():
    observation = json.loads(
        (ROOT / "analysis/agent-baseline/agent_strategy.json").read_text()
    )

    plan = build_supervision_plan(observation)

    assert plan["evidence_boundary"] == "observational_not_causal"
    assert plan["strategy"] == "bounded_structured_mutation"
    assert plan["public_failure_sequence"] == [
        "filter_normalized_status",
        "reject_invalid_amounts",
    ]
    assert plan["model_calls"] == 0


def test_real_poc_improves_public_score_without_hidden_regression(tmp_path):
    result = run_poc(
        program_path=ROOT / "initial_program.py",
        observation_path=ROOT / "analysis/agent-baseline/agent_strategy.json",
        output_dir=tmp_path,
    )

    assert result["baseline"]["public_passed"] == 3
    assert result["final"]["public_passed"] == 6
    assert result["final"]["hidden_passed"] >= result["baseline"]["hidden_passed"]
    assert len(result["steps"]) == 2
    assert all(step["decision"] == "accepted" for step in result["steps"])
    assert all(step["lost_public_cases"] == [] for step in result["steps"])
    assert result["decision"] == "poc_candidate_accepted"
    assert result["production_ready"] is False


def test_poc_writes_human_and_machine_readable_outputs(tmp_path):
    run_poc(
        program_path=ROOT / "initial_program.py",
        observation_path=ROOT / "analysis/agent-baseline/agent_strategy.json",
        output_dir=tmp_path,
    )

    assert (tmp_path / "result.json").is_file()
    assert (tmp_path / "candidate.py").is_file()
    markdown = (tmp_path / "report.md").read_text()
    html = (tmp_path / "report.html").read_text()
    assert "监督演化 POC" in markdown
    assert "3/13" in markdown
    assert "6/13" in markdown
    assert "监督演化 POC" in html
    assert "production_ready=false" in html
