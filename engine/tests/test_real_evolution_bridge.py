from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from agent_arm_runner import (
    AgentExecutionError,
    ArmProgram,
    build_multi_arm_invocations,
)
from benchmark_adapters import BenchmarkRegistry, StaticBenchmarkAdapter, TaskPool
from benchmark_catalog import PINNED_SOURCES, normalize_swe_task
from continuous_ab import BaselineContract
from evolution_controller import (
    EvolutionAuthorization,
    EvolutionController,
    EvolutionPlan,
)
from evolution_runtime import ExecutionRequest
from real_evolution_bridge import (
    BridgeContractError,
    ObservationFeatures,
    RealEvolutionAdapters,
    freeze_search_task_schedule,
    load_search_task_schedule,
    materialize_evolution_claimed_task,
)

ORIGINAL = "a" * 64
PARENT = "b" * 64
EPOCH = "native-adapters-v2.1.0-frozen"


def _tree_hash(root: Path) -> str:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _profile(root: Path, text: str, *, harness: bool = False) -> str:
    root.mkdir(parents=True)
    (root / "AGENTS.md").write_text(text, encoding="utf-8")
    if harness:
        target = root / ".codex/harness/runner.py"
        target.parent.mkdir(parents=True)
        target.write_text("def run():\n    return 1\n", encoding="utf-8")
    return _tree_hash(root)


def _baseline(agent_sha256: str) -> BaselineContract:
    return BaselineContract(
        experiment_id="real-evolution-bridge-test",
        agent_program_sha256=agent_sha256,
        model="gpt-test",
        reasoning="low",
        token_budget=4096,
        timeout_seconds=30,
        tools=("shell", "apply_patch"),
        retries=0,
        evaluator_epoch=EPOCH,
    )


def _authorization() -> EvolutionAuthorization:
    return EvolutionAuthorization(
        maximum_unique_search_tasks=100,
        maximum_real_codex_calls=2000,
        maximum_temporary_cloud_instances=1,
        maximum_elapsed_hours=24.0,
        maximum_cloud_cost_cny=30.0,
    )


def _git_workspace(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    (path / "source.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        cwd=path,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _pool_with_100_search_tasks(root: Path, *, base_commit: str) -> Path:
    source = PINNED_SOURCES["swe-bench-verified"]
    rows = []
    tasks = []
    for index in range(100):
        row = {
            "repo": "example/repo",
            "instance_id": f"example__repo-{index + 1}",
            "base_commit": base_commit,
            "problem_statement": f"Fix public behavior {index + 1}.",
            "patch": f"SECRET GOLD {index + 1}",
            "test_patch": f"SECRET TEST {index + 1}",
        }
        rows.append(row)
        tasks.append(normalize_swe_task(source, row, row_index=index))
    registry = BenchmarkRegistry()
    registry.register(
        StaticBenchmarkAdapter(
            adapter_id=source.adapter_id,
            revision=source.dataset_revision,
            executable=True,
            tasks=tuple(tasks),
        )
    )
    pool_root = root / "benchmark-pool"
    pool_root.mkdir(parents=True)
    pool_path = pool_root / "TASK_POOL.json"
    TaskPool.build(
        registry=registry,
        seed_material="real-evolution-bridge-test",
        target_count=100,
        promotion_count=0,
        final_sealed_count=0,
    ).save(pool_path)
    harness_input = pool_root / "harness-inputs/swe-bench-verified.jsonl"
    harness_input.parent.mkdir(parents=True)
    harness_input.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return pool_path


def _canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _resume_schedule(root: Path, *, task_count: int = 91) -> Path:
    tasks = []
    for index in range(task_count):
        uid = hashlib.sha256(f"resume-{index}".encode()).hexdigest()
        tasks.append(
            {
                "benchmark_id": "swe-bench-verified",
                "generation": 0,
                "instance_id": f"example__repo-{index + 1}",
                "ordinal": index + 1,
                "stage": "observe",
                "task_contract_sha256": hashlib.sha256(uid.encode()).hexdigest(),
                "task_uid": uid,
            }
        )
    payload = {
        "schema_version": "1.1-resume",
        "status": "frozen_resume_before_task_materialization",
        "resume": {
            "of_semantic_fingerprint": "f" * 64,
            "lost_task_uids": [
                hashlib.sha256(f"lost-{i}".encode()).hexdigest() for i in range(9)
            ],
            "incident_reference": "cloud-control/INCIDENT-016-EVIDENCE-LOSS.md",
        },
        "tasks": tasks,
        "unique_search_tasks": task_count,
        "promotion_tasks_opened": 0,
        "final_sealed_tasks_opened": 0,
        "task_content_included": False,
        "gold_fields_included": False,
    }
    unsigned = {
        key: value for key, value in payload.items() if key != "semantic_fingerprint"
    }
    payload["semantic_fingerprint"] = hashlib.sha256(
        _canonical_json(unsigned).encode()
    ).hexdigest()
    path = root / "SEARCH-TASK-SCHEDULE-RESUME.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_resume_schedule_loader_accepts_91_tasks_and_rejects_malformed(tmp_path):
    valid = _resume_schedule(tmp_path)
    loaded = load_search_task_schedule(valid)
    assert loaded["schema_version"] == "1.1-resume"
    assert len(loaded["tasks"]) == 91

    malformed = _resume_schedule(tmp_path)
    payload = json.loads(malformed.read_text(encoding="utf-8"))
    payload["schema_version"] = "1.0"
    del payload["resume"]
    unsigned = {
        key: value for key, value in payload.items() if key != "semantic_fingerprint"
    }
    payload["semantic_fingerprint"] = hashlib.sha256(
        _canonical_json(unsigned).encode()
    ).hexdigest()
    malformed.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(BridgeContractError, match="100 unique tasks"):
        load_search_task_schedule(malformed)

    too_many = _resume_schedule(tmp_path, task_count=100)
    with pytest.raises(BridgeContractError, match="91 unique tasks"):
        load_search_task_schedule(too_many)


def test_schedule_freezes_100_search_identities_without_opening_content(tmp_path: Path):
    frozen_pool = Path(
        "artifacts/v2.1.0/v2.1.0-continuous-ab/configs/benchmark-pool/TASK_POOL.json"
    ).resolve()
    before = hashlib.sha256(frozen_pool.read_bytes()).hexdigest()
    schedule_path = tmp_path / "SEARCH-TASK-SCHEDULE.json"

    schedule = freeze_search_task_schedule(frozen_pool, schedule_path)

    assert len(schedule["tasks"]) == 100
    assert len({row["task_uid"] for row in schedule["tasks"]}) == 100
    assert [
        sum(row["generation"] == generation for row in schedule["tasks"])
        for generation in range(4)
    ] == [25, 25, 25, 25]
    assert {row["stage"] for row in schedule["tasks"]} == {
        "observe",
        "scout",
        "semifinal",
        "confirmation",
    }
    assert (
        {
            benchmark: sum(
                row["benchmark_id"] == benchmark for row in schedule["tasks"]
            )
            for benchmark in schedule["benchmark_allocation"]
        }
        == schedule["benchmark_allocation"]
        == {
            "multi-swe-bench-flash": 34,
            "swe-bench-multilingual": 33,
            "swe-bench-verified": 33,
        }
    )
    assert "terminal-bench-2" not in schedule["benchmark_allocation"]
    assert schedule["excluded_search_benchmark_families"] == ["terminal-bench-2"]
    forbidden = {"instruction", "problem_statement", "patch", "test_patch", "gold"}
    assert all(not (set(row) & forbidden) for row in schedule["tasks"])
    assert schedule["final_sealed_tasks_opened"] == 0
    assert schedule["promotion_tasks_opened"] == 0
    assert hashlib.sha256(frozen_pool.read_bytes()).hexdigest() == before
    assert load_search_task_schedule(schedule_path) == schedule


def test_tool_event_freezer_keeps_nested_codex_command_and_file_change(tmp_path: Path):
    trajectory = tmp_path / "events.jsonl"
    trajectory.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "command_execution",
                            "command": "pytest -q",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "file_change", "path": "source.py"},
                    }
                ),
                json.dumps({"type": "message", "text": "done"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "tool-events.json"

    RealEvolutionAdapters._write_tool_events(trajectory, output)

    frozen = json.loads(output.read_text())
    assert len(frozen["events"]) == 2
    assert [event["item"]["type"] for event in frozen["events"]] == [
        "command_execution",
        "file_change",
    ]


def test_multi_arm_contract_is_matched_and_supports_constrained_harness_code(
    tmp_path: Path,
):
    original_root = tmp_path / "original"
    parent_root = tmp_path / "parent"
    candidate_root = tmp_path / "candidate"
    original_hash = _profile(original_root, "Original.\n")
    parent_hash = _profile(parent_root, "Parent.\n", harness=True)
    candidate_hash = _profile(candidate_root, "Candidate.\n")
    materialized = {
        "round_id": "g1-scout-task-1",
        "task_uid": "task-1",
        "benchmark_id": "swe-bench-verified",
        "instance_id": "example__repo-1",
        "repo": "example/repo",
        "base_commit": "d" * 40,
        "instruction": "Fix it.",
        "instruction_sha256": hashlib.sha256(b"Fix it.").hexdigest(),
    }
    arms = (
        ArmProgram("original", original_hash, original_root, tmp_path / "ws-original"),
        ArmProgram("parent", parent_hash, parent_root, tmp_path / "ws-parent"),
        ArmProgram(
            "candidate", candidate_hash, candidate_root, tmp_path / "ws-candidate"
        ),
    )

    invocations = build_multi_arm_invocations(
        baseline=_baseline(original_hash),
        materialized_task=materialized,
        arms=arms,
        evidence_root=tmp_path / "evidence",
    )

    assert set(invocations) == {"original", "parent", "candidate"}
    assert len({item.matched_contract_sha256 for item in invocations.values()}) == 1
    assert len({item.workspace for item in invocations.values()}) == 3
    assert len({item.evidence_dir for item in invocations.values()}) == 3
    assert invocations["parent"].agent_program_sha256 == parent_hash

    (candidate_root / "forbidden.txt").write_text("no\n", encoding="utf-8")
    with pytest.raises(AgentExecutionError, match="not allowed"):
        build_multi_arm_invocations(
            baseline=_baseline(original_hash),
            materialized_task=materialized,
            arms=arms,
            evidence_root=tmp_path / "second-evidence",
        )


def test_materialization_requires_new_controller_claim_and_excludes_gold(
    tmp_path: Path,
):
    repo = tmp_path / "source-repo"
    base_commit = _git_workspace(repo)
    pool_path = _pool_with_100_search_tasks(tmp_path, base_commit=base_commit)
    schedule_path = tmp_path / "SEARCH-TASK-SCHEDULE.json"
    schedule = freeze_search_task_schedule(pool_path, schedule_path)
    task_uids = tuple(row["task_uid"] for row in schedule["tasks"])
    controller = EvolutionController.initialize(
        tmp_path / "controller",
        plan=EvolutionPlan.build(task_uids),
        authorization=_authorization(),
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
        native_evaluator_epoch=EPOCH,
    )
    first = task_uids[0]
    with pytest.raises(BridgeContractError, match="claimed"):
        materialize_evolution_claimed_task(
            controller=controller,
            schedule_path=schedule_path,
            task_uid=first,
            pool_root=pool_path.parent,
            output_path=tmp_path / "task-input.json",
        )

    controller.claim_stage(0, "observe", candidate_sha256s=())
    materialized = materialize_evolution_claimed_task(
        controller=controller,
        schedule_path=schedule_path,
        task_uid=first,
        pool_root=pool_path.parent,
        output_path=tmp_path / "task-input.json",
    )

    rendered = json.dumps(materialized)
    assert materialized["gold_fields_included"] is False
    assert "SECRET GOLD" not in rendered
    assert "SECRET TEST" not in rendered
    assert materialized["task_uid"] == first


def test_real_adapter_freezes_native_and_observer_evidence_but_ranks_native_only(
    tmp_path: Path,
):
    source_repo = tmp_path / "source-repo"
    base_commit = _git_workspace(source_repo)
    pool_path = _pool_with_100_search_tasks(tmp_path, base_commit=base_commit)
    schedule_path = tmp_path / "SEARCH-TASK-SCHEDULE.json"
    schedule = freeze_search_task_schedule(pool_path, schedule_path)
    task_uids = tuple(row["task_uid"] for row in schedule["tasks"])
    controller = EvolutionController.initialize(
        tmp_path / "controller",
        plan=EvolutionPlan.build(task_uids),
        authorization=_authorization(),
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
        native_evaluator_epoch=EPOCH,
    )
    controller.claim_stage(0, "observe", candidate_sha256s=())
    original_profile = tmp_path / "original-profile"
    parent_profile = tmp_path / "parent-profile"
    original_hash = _profile(original_profile, "Original.\n")
    parent_hash = _profile(parent_profile, "Parent.\n")
    assert original_hash != parent_hash
    # The controller uses the authoritative profile hashes for real execution.
    controller = EvolutionController.initialize(
        tmp_path / "real-controller",
        plan=EvolutionPlan.build(task_uids),
        authorization=_authorization(),
        original_agent_program_sha256=original_hash,
        seed_parent_agent_program_sha256=parent_hash,
        native_evaluator_epoch=EPOCH,
    )
    controller.claim_stage(0, "observe", candidate_sha256s=())
    first = schedule["tasks"][0]

    workspaces = {}

    def workspace_factory(_materialized, arm):
        target = tmp_path / "workspaces" / arm.name
        subprocess.run(
            ["git", "clone", "-q", str(source_repo), str(target)], check=True
        )
        workspaces[arm.name] = target
        return target

    fake_codex = tmp_path / "fake-codex"
    fake_codex.write_text(
        "#!/bin/sh\n"
        "while [ $# -gt 0 ]; do\n"
        '  if [ "$1" = "-C" ]; then shift; work=$1; fi\n'
        "  shift\n"
        "done\n"
        'printf \'%s\\n\' \'{"type":"tool","name":"shell"}\'\n'
        'printf \'%s\\n\' \'{"usage":{"input_tokens":20,"cached_input_tokens":0,"output_tokens":5}}\'\n'
        "printf '%s\\n' 'value = 2' > \"$work/source.py\"\n",
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)

    def native_evaluator(invocation, _materialized, _receipt):
        report = Path(invocation.evidence_dir) / "native-report.json"
        report.write_text(
            json.dumps(
                {
                    invocation.instance_id: {
                        "resolved": True,
                        "patch_successfully_applied": True,
                        "tests_status": {
                            "PASS_TO_PASS": {"success": ["stable"], "failure": []},
                            "PASS_TO_FAIL": {"success": [], "failure": []},
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        return report

    adapter = RealEvolutionAdapters(
        controller=controller,
        schedule_path=schedule_path,
        pool_root=pool_path.parent,
        baseline=_baseline(original_hash),
        profile_roots={original_hash: original_profile, parent_hash: parent_profile},
        run_root=tmp_path / "real-run",
        codex_executable=fake_codex,
        authorization={
            "status": "AUTHORIZED_WITH_HARD_CAPS",
            "scope": {
                "maximum_unique_search_tasks": 100,
                "maximum_real_codex_calls": 2000,
                "maximum_temporary_cloud_instances": 1,
                "maximum_elapsed_hours": 24,
                "maximum_cloud_cost_cny": 30,
            },
        },
        workspace_factory=workspace_factory,
        native_evaluator=native_evaluator,
        observer=lambda _context: ObservationFeatures(
            observed_features=("tests_before_edit",),
            conditions=("python", "observe"),
            expected_surfaces=("prompt", "skills"),
        ),
        proposer=None,
    )
    request = ExecutionRequest(
        generation=0,
        stage="observe",
        task_uid=first["task_uid"],
        role="original",
        arm_sha256=original_hash,
        original_sha256=original_hash,
        parent_sha256=parent_hash,
        candidate_ordinal=None,
    )

    artifact = adapter.execute(request)
    resumed = adapter.execute(request)

    assert artifact.arm.native_score == 1.0
    assert artifact.arm.safety_passed is True
    assert artifact.arm.cost_units == 25.0
    assert artifact.arm.role == "original"
    assert set(artifact.arm.to_dict()) == artifact.arm._FIELDS
    assert artifact.observation is not None
    assert artifact.observation.admission_gate_allowed is False
    assert set(artifact.observation.evidence) == {
        "trajectory",
        "tool_events",
        "native_evaluator",
        "cost",
        "safety",
    }
    assert all(
        not Path(item.path).is_absolute()
        for item in artifact.observation.evidence.values()
    )
    assert resumed == artifact
    assert controller.inspect()["usage"]["real_codex_calls"] == 1
    assert (
        tmp_path
        / "real-run/evidence/generation-0/observe"
        / first["task_uid"]
        / "original/execution-artifact.json"
    ).is_file()


def test_real_adapter_fails_closed_when_native_evaluator_report_is_missing(
    tmp_path: Path,
):
    with pytest.raises(BridgeContractError, match="native evaluator"):
        RealEvolutionAdapters.require_native_report(tmp_path / "missing.json")
