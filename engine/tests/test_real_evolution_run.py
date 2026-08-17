from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agent_arm_runner import profile_tree_hash
from real_evolution_run import assemble_real_evolution_run


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _profile(root: Path, text: str) -> str:
    root.mkdir(parents=True)
    (root / "AGENTS.md").write_text(text, encoding="utf-8")
    return profile_tree_hash(root)


def test_real_run_assembly_freezes_profiles_schedule_budget_and_observer_boundary(
    tmp_path: Path,
):
    original_root = tmp_path / "original"
    seed_root = tmp_path / "seed"
    original = _profile(original_root, "Original.\n")
    seed = _profile(seed_root, "Seed.\n")
    tasks = [
        {
            "ordinal": index + 1,
            "generation": index // 25,
            "stage": "observe" if index < 25 else "scout",
            "task_uid": f"task-{index:03d}",
            "benchmark_id": "swe-bench-verified",
            "instance_id": f"repo__project-{index}",
            "task_contract_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
        }
        for index in range(100)
    ]
    unsigned_schedule = {
        "schema_version": "1.0",
        "status": "frozen_before_task_materialization",
        "tasks": tasks,
        "promotion_tasks_opened": 0,
        "final_sealed_tasks_opened": 0,
        "task_content_included": False,
        "gold_fields_included": False,
    }
    schedule = {
        **unsigned_schedule,
        "semantic_fingerprint": hashlib.sha256(
            _canonical(unsigned_schedule).encode()
        ).hexdigest(),
    }
    schedule_path = tmp_path / "SEARCH-TASK-SCHEDULE.json"
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    baseline_path = tmp_path / "permanent-baseline.json"
    baseline = {
        "schema_version": "1.0",
        "experiment_id": "real-run-test",
        "agent_program_sha256": original,
        "model": "gpt-test",
        "reasoning": "low",
        "token_budget": 4096,
        "timeout_seconds": 30,
        "tools": ["shell", "apply_patch"],
        "retries": 0,
        "evaluator_epoch": "native-test-epoch",
    }
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "baseline": baseline,
                "baseline_contract_sha256": hashlib.sha256(
                    _canonical(baseline).encode()
                ).hexdigest(),
                "freeze_count": 1,
                "immutable": True,
            }
        ),
        encoding="utf-8",
    )
    authorization_path = tmp_path / "AUTHORIZATION.json"
    authorization_path.write_text(
        json.dumps(
            {
                "status": "AUTHORIZED_WITH_HARD_CAPS",
                "scope": {
                    "maximum_unique_search_tasks": 100,
                    "maximum_real_codex_calls": 2000,
                    "maximum_temporary_cloud_instances": 1,
                    "maximum_elapsed_hours": 24,
                    "maximum_cloud_cost_cny": 30,
                },
            }
        ),
        encoding="utf-8",
    )
    executable = tmp_path / "executable"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    swe_python = tmp_path / "swe-python"
    multi_python = tmp_path / "multi-python"
    swe_python.symlink_to(executable)
    multi_python.symlink_to(executable)
    for directory in ("pool", "swe", "multi"):
        (tmp_path / directory).mkdir()

    assembly = assemble_real_evolution_run(
        run_root=tmp_path / "run",
        schedule_path=schedule_path,
        pool_root=tmp_path / "pool",
        baseline_authority_path=baseline_path,
        authorization_path=authorization_path,
        original_profile_root=original_root,
        seed_profile_root=seed_root,
        codex_executable=executable,
        swe_python=swe_python,
        multi_python=multi_python,
        swe_harness_root=tmp_path / "swe",
        multi_harness_root=tmp_path / "multi",
        project_root=tmp_path,
    )

    contract = json.loads((tmp_path / "run/RUN-CONTRACT.json").read_text())
    assert assembly.plan.unique_search_tasks == 100
    assert assembly.authorization.maximum_real_codex_calls == 2000
    assert assembly.baseline.agent_program_sha256 == original
    assert assembly.seed_parent_agent_program_sha256 == seed
    assert assembly.adapters.controller is assembly.controller
    assert assembly.adapters.native_evaluator.swe_python == swe_python.absolute()
    assert assembly.adapters.native_evaluator.multi_python == multi_python.absolute()
    assert contract["observer"] == {
        "admission_gate_allowed": False,
        "hidden_state_access": False,
        "kind": "jlens_trajectory_sidecar",
    }
    assert contract["final_sealed_opened"] is False
    assert contract["production_promotion_allowed"] is False
    assert contract["model_weights_frozen"] is True


def test_write_partial_result_records_frozen_state(tmp_path):
    from evolution_fixture import run_fixture
    from real_evolution_run import _write_partial_result

    run_root = tmp_path / "run"
    run_fixture(run_root)
    json_path = _write_partial_result(
        output_dir=run_root, run_root=run_root, reason="signal_15"
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "partial"
    assert payload["terminal_state"] == "partial"
    assert payload["stop_reason"]["code"] == "signal"
    assert (
        payload["controller_state_sha256"]
        == (run_root / "controller/STATE.sha256").read_text().strip()
    )
    assert payload["usage"] is not None
    assert payload["final_sealed_opened"] is False
    assert (run_root / "RESULT.partial.zh-CN.md").is_file()


def test_sigterm_writer_terminates_and_writes_partial(tmp_path):
    import select
    import subprocess
    import sys

    from evolution_fixture import run_fixture

    run_root = tmp_path / "run"
    run_fixture(run_root)
    root = Path(__file__).resolve().parents[1]
    script = (
        "import signal, sys, time; "
        "sys.path.insert(0, sys.argv[2]); "
        "from real_evolution_run import install_signal_partial_writer; "
        "install_signal_partial_writer(run_root=sys.argv[1]); "
        "print('signal-handler-ready', flush=True); "
        "time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(run_root), str(root)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready, _, _ = select.select([process.stdout], [], [], 15)
        assert ready, "signal handler subprocess did not become ready"
        assert process.stdout.readline().strip() == "signal-handler-ready"
        process.send_signal(15)  # SIGTERM
        process.wait(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert process.returncode == 128 + 15
    assert (run_root / "RESULT.partial.json").is_file()
    payload = json.loads((run_root / "RESULT.partial.json").read_text(encoding="utf-8"))
    assert payload["status"] == "partial"
    assert payload["stop_reason"]["signal"] == "signal_15"


def test_cli_installs_sigterm_writer_before_assembly(tmp_path):
    import select
    import subprocess
    import sys

    run_root = tmp_path / "early-run"
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
import time
sys.path.insert(0, sys.argv[1])
import real_evolution_run as run

def block_assembly(**_kwargs):
    print("assembly-started", flush=True)
    time.sleep(60)

run.assemble_real_evolution_run = block_assembly
dummy = sys.argv[3]
run.main([
    "--run-root", sys.argv[2],
    "--schedule", dummy,
    "--pool-root", dummy,
    "--baseline-authority", dummy,
    "--authorization", dummy,
    "--original-profile", dummy,
    "--seed-profile", dummy,
    "--codex-executable", dummy,
    "--swe-python", dummy,
    "--multi-python", dummy,
    "--swe-harness-root", dummy,
    "--multi-harness-root", dummy,
    "--project-root", dummy,
])
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(root), str(run_root), str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready, _, _ = select.select([process.stdout], [], [], 15)
        assert ready, "assembly subprocess did not become ready"
        assert process.stdout.readline().strip() == "assembly-started"
        process.send_signal(15)
        process.wait(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert process.returncode == 128 + 15
    payload = json.loads((run_root / "RESULT.partial.json").read_text())
    assert payload["status"] == "partial"
    assert payload["controller_state_sha256"] is None
