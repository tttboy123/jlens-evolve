from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from evolve.contracts import Cohort, ContractViolation, canonical_json
from evolve.evidence import ReceiptStore
from evolve.fresh_feedback import (
    _build_tasks,
    _launcher,
    _load_config,
    _require_clean_head,
    _trusted_jlens_runtime,
    run_fresh_feedback_e2e,
    seal_run,
)
from evolve.proposals import CandidateCompiler, CompiledRevision, CompileSpec
from evolve.reporting import AuditVerifier


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source(tmp_path: Path, name: str) -> tuple[Path, str]:
    root = tmp_path / name
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "Tests")
    _git(root, "config", "user.email", "tests@example.invalid")
    (root / "source.txt").write_text(name, encoding="utf-8")
    _git(root, "add", "source.txt")
    _git(root, "commit", "-m", "source")
    return root, _git(root, "rev-parse", "HEAD")


class _FixtureQwenCellRunner:
    calls: list[str] = []

    def __init__(self, **kwargs: Any) -> None:
        self.compiled = CompiledRevision.load(kwargs["compiled_revision_root"])

    def run(self, plan, workspace, output_root: Path):
        self.calls.append(plan.plan_id)
        artifact_root = output_root / plan.plan_id
        artifact_root.mkdir(parents=True)
        raw = b"fixture local Qwen output\n"
        raw_path = artifact_root / "raw-output.txt"
        raw_path.write_bytes(raw)
        candidate_prompt = None
        compiled_artifacts: dict[str, str] = {}
        prompt_text = "SYSTEM: deterministic baseline fixture"
        if plan.arm == "taught":
            candidate_prompt = (
                f"revision={self.compiled.change_set.revision_id}\n"
                f"bundle={self.compiled.bundle_sha256}"
            )
            compiled_artifacts = dict(self.compiled.artifact_sha256)
            prompt_text = f"COMPILED-CANDIDATE:\n{candidate_prompt}"
        prompt_path = artifact_root / "prompt-000.txt"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        patch = "diff --git a/fixture.txt b/fixture.txt\n"
        return {
            "patch": patch,
            "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "raw_output_path": str(raw_path),
            "raw_output_sha256": hashlib.sha256(raw).hexdigest(),
            "prompt_paths": [str(prompt_path)],
            "prompt_texts": [prompt_text],
            "prompt_sha256": [hashlib.sha256(prompt_text.encode()).hexdigest()],
            "candidate_consumed": plan.arm == "taught",
            "candidate_revision_id": (
                self.compiled.change_set.revision_id
                if plan.arm == "taught"
                else None
            ),
            "candidate_bundle_sha256": (
                self.compiled.bundle_sha256 if plan.arm == "taught" else None
            ),
            "candidate_prompt": candidate_prompt,
            "candidate_prompt_sha256": (
                hashlib.sha256(candidate_prompt.encode()).hexdigest()
                if candidate_prompt is not None
                else None
            ),
            "compiled_artifact_sha256": compiled_artifacts,
            "parent_harness_revision_id": None,
            "parent_harness_bundle_sha256": None,
            "parent_harness_prompt": None,
            "parent_harness_prompt_sha256": None,
            "structural_valid": True,
            "failure_reason": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_cny": 0,
        }


class _FixtureNativeEvaluator:
    calls: list[str] = []

    def __init__(self, *, evaluator_id: str, **_kwargs: Any) -> None:
        self.evaluator_id = evaluator_id

    def evaluate(self, plan, workspace, model_output):
        self.calls.append(plan.plan_id)
        return {
            "resolved": plan.arm == "taught",
            "native_valid": True,
            "native_error": None,
            "regressions": [],
            "prediction_sha256": model_output["patch_sha256"],
        }


def _compiled_fixture(tmp_path: Path, task_ids: tuple[str, ...]) -> CompiledRevision:
    request = {
        "request_id": "teacher-fixture",
        "provider": "fixture",
        "model": "fixture-teacher",
        "failure_package": {"feedback_only": True},
    }
    request_path = tmp_path / "teacher-request.json"
    request_path.write_text(canonical_json(request) + "\n", encoding="utf-8")
    response = {
        "schema_version": 1,
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "provider": "fixture",
        "model": "fixture-teacher",
        "candidate": {
            "protocol": "skill-v1",
            "prompt_template": "Repair {task}",
            "skill_text": "Use the deterministic fixture route.",
            "eval_note": "Feedback-only fixture evaluation.",
        },
        "candidate_status": "inactive",
        "auto_activate": False,
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "network_calls": 1,
        "pricing_cny_per_million": {"input": 0, "output": 0},
        "estimated_cost_cny": 0,
    }
    response_path = tmp_path / "teacher-response.json"
    response_path.write_text(canonical_json(response) + "\n", encoding="utf-8")
    return CandidateCompiler().compile(
        request_path=request_path,
        response_path=response_path,
        compile_spec=CompileSpec(
            candidate_id="candidate-fixture",
            revision_id="candidate-fixture-r1",
            parent_revision_id="qwen-zero-teaching-v1",
            cohort=Cohort.FEEDBACK,
            operator_id="operator-fixture",
            operator_instruction="Apply fixture teaching.",
            routes=tuple((task_id, "operator-fixture") for task_id in task_ids),
        ),
        output_root=tmp_path / "compiled",
    )


def _fresh_fixture_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    source_root, final_commit = _source(tmp_path, "product-source")
    swe_harness, _ = _source(tmp_path, "swe-harness")
    multi_harness, _ = _source(tmp_path, "multi-harness")
    task_ids = ("fixture-a", "fixture-b", "fixture-c")
    tasks = []
    for task_id, project in zip(
        task_ids, ("project-a", "project-b", "project-c"), strict=True
    ):
        source, revision = _source(tmp_path, f"source-{task_id}")
        tasks.append(
            {
                "source_uri": str(source),
                "base_revision": revision,
                "instance_id": task_id,
                "project": project,
                "benchmark_id": "swe-bench-verified",
                "catalog_fingerprint": hashlib.sha256(task_id.encode()).hexdigest(),
                "cohort": "feedback",
            }
        )
    compiled = _compiled_fixture(tmp_path, task_ids)
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    (legacy_root / "official_patch_evaluator.py").write_text(
        "# frozen fixture evaluator identity\n", encoding="utf-8"
    )
    model_path = tmp_path / "model"
    model_path.mkdir()
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
    ):
        (model_path / name).write_text("{}\n", encoding="utf-8")
    pool_root = tmp_path / "pool"
    dataset = pool_root / "harness-inputs/swe-bench-verified.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}\n", encoding="utf-8")
    taskset = tmp_path / "TASKSET.json"
    taskset.write_text('{"tasks":[]}\n', encoding="utf-8")
    routes = tmp_path / "ROUTES.json"
    routes.write_text('{"routes":{}}\n', encoding="utf-8")
    launcher = tmp_path / "python"
    launcher.write_text("fixture launcher\n", encoding="utf-8")
    native_assets = tmp_path / "native-assets.json"
    native_assets.write_text("{}\n", encoding="utf-8")
    harness_receipt = tmp_path / "HARNESS-RUNTIME.json"
    harness_receipt.write_text('{"fixture":true}\n', encoding="utf-8")
    monkeypatch.setattr(
        "evolve.fresh_feedback.LegacyQwenCellRunner", _FixtureQwenCellRunner
    )
    monkeypatch.setattr(
        "evolve.fresh_feedback.LegacyOfficialNativeEvaluator",
        _FixtureNativeEvaluator,
    )
    monkeypatch.setattr(
        "evolve.fresh_feedback._freeze_harness",
        lambda _config, _output_root: harness_receipt,
    )
    config = {
        "schema_version": 1,
        "campaign_id": "fresh-idempotence-fixture",
        "source_root": str(source_root),
        "final_commit_sha": final_commit,
        "legacy_root": str(legacy_root),
        "model_path": str(model_path),
        "pool_root": str(pool_root),
        "swe_harness_root": str(swe_harness),
        "multi_harness_root": str(multi_harness),
        "swe_python": str(launcher),
        "multi_python": str(launcher),
        "native_assets_path": str(native_assets),
        "taskset_path": str(taskset),
        "routes_path": str(routes),
        "compiled_revision_root": str(compiled.root),
        "candidate_id": compiled.change_set.candidate_id,
        "candidate_revision_id": compiled.change_set.revision_id,
        "parent_revision_id": compiled.change_set.parent_revision_id,
        "inline_api_key": "must-not-be-copied-into-run-artifacts",
        "tasks": tasks,
    }
    config_path = tmp_path / "fresh-config.json"
    config_path.write_text(canonical_json(config) + "\n", encoding="utf-8")
    return config_path, tmp_path / "run"


def test_completed_fresh_campaign_replay_preserves_every_sealed_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FixtureQwenCellRunner.calls = []
    _FixtureNativeEvaluator.calls = []
    config_path, output_root = _fresh_fixture_config(tmp_path, monkeypatch)

    first = run_fresh_feedback_e2e(
        config_path=config_path, output_root=output_root
    )
    manifest_path = output_root / "EVIDENCE-MANIFEST.json"
    manifest_before = manifest_path.read_bytes()
    manifest = json.loads(manifest_before)
    sealed_before = {
        row["path"]: (output_root / row["path"]).read_bytes()
        for row in manifest["entries"]
    }
    receipt_store = ReceiptStore(output_root / "receipt-store")
    receipt_ids_before = tuple(
        receipt.receipt_id for receipt in receipt_store.list_receipts()
    )
    qwen_calls = tuple(_FixtureQwenCellRunner.calls)
    native_calls = tuple(_FixtureNativeEvaluator.calls)

    replayed = run_fresh_feedback_e2e(
        config_path=config_path, output_root=output_root
    )

    assert replayed == first
    assert manifest_path.read_bytes() == manifest_before
    assert {
        row["path"]: (output_root / row["path"]).read_bytes()
        for row in manifest["entries"]
    } == sealed_before
    assert tuple(
        receipt.receipt_id for receipt in receipt_store.list_receipts()
    ) == receipt_ids_before
    assert tuple(_FixtureQwenCellRunner.calls) == qwen_calls
    assert tuple(_FixtureNativeEvaluator.calls) == native_calls
    assert AuditVerifier().verify_manifest(manifest_path, root=output_root) == len(
        manifest["entries"]
    )
    assert not (output_root / "RUN-CONFIG.json").exists()
    assert all(
        b"must-not-be-copied-into-run-artifacts" not in content
        for content in sealed_before.values()
    )


def test_completed_fresh_campaign_replay_rejects_config_literal_drift_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, output_root = _fresh_fixture_config(tmp_path, monkeypatch)
    run_fresh_feedback_e2e(config_path=config_path, output_root=output_root)
    manifest_path = output_root / "EVIDENCE-MANIFEST.json"
    manifest_before = manifest_path.read_bytes()
    result_before = (output_root / "CAMPAIGN-RESULT.json").read_bytes()
    config_path.write_bytes(config_path.read_bytes() + b"\n")

    with pytest.raises(ContractViolation, match="replay identity mismatch"):
        run_fresh_feedback_e2e(config_path=config_path, output_root=output_root)

    assert manifest_path.read_bytes() == manifest_before
    assert (output_root / "CAMPAIGN-RESULT.json").read_bytes() == result_before


def test_completed_fresh_campaign_replay_rejects_manifest_tamper_without_reseal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, output_root = _fresh_fixture_config(tmp_path, monkeypatch)
    run_fresh_feedback_e2e(config_path=config_path, output_root=output_root)
    manifest_path = output_root / "EVIDENCE-MANIFEST.json"
    manifest_before = manifest_path.read_bytes()
    report_path = output_root / "FINAL-REPORT.json"
    report_path.write_bytes(report_path.read_bytes() + b" ")
    tampered_report = report_path.read_bytes()

    with pytest.raises(ContractViolation, match="manifest hash mismatch"):
        run_fresh_feedback_e2e(config_path=config_path, output_root=output_root)

    assert manifest_path.read_bytes() == manifest_before
    assert report_path.read_bytes() == tampered_report


def test_completed_legacy_seal_without_config_hash_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, output_root = _fresh_fixture_config(tmp_path, monkeypatch)
    run_fresh_feedback_e2e(config_path=config_path, output_root=output_root)
    result_path = output_root / "CAMPAIGN-RESULT.json"
    legacy_result = json.loads(result_path.read_text(encoding="utf-8"))
    legacy_result.pop("run_config_sha256")
    result_path.write_text(
        canonical_json(legacy_result) + "\n", encoding="utf-8"
    )
    seal_run(output_root)
    manifest_before = (output_root / "EVIDENCE-MANIFEST.json").read_bytes()
    result_before = result_path.read_bytes()

    with pytest.raises(ContractViolation, match="replay identity mismatch"):
        run_fresh_feedback_e2e(config_path=config_path, output_root=output_root)

    assert (output_root / "EVIDENCE-MANIFEST.json").read_bytes() == manifest_before
    assert result_path.read_bytes() == result_before


def test_fresh_config_denies_any_non_feedback_task(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tasks": [
                    {"instance_id": "a", "cohort": "feedback"},
                    {"instance_id": "b", "cohort": "holdout"},
                    {"instance_id": "c", "cohort": "feedback"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ContractViolation, match="holdout"):
        _load_config(path)


@pytest.mark.parametrize("field", ["operator_skill_path", "span_skill_path"])
def test_fresh_config_denies_legacy_frozen_skill_fallbacks(
    tmp_path: Path, field: str
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tasks": [
                    {"instance_id": "a", "cohort": "feedback"},
                    {"instance_id": "b", "cohort": "feedback"},
                    {"instance_id": "c", "cohort": "feedback"},
                ],
                field: "/legacy/frozen-skill.json",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ContractViolation, match="fallback"):
        _load_config(path)


def test_trusted_jlens_requires_a_process_local_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("JLENS_TEST_SECRET", raising=False)
    config = {
        "trusted_jlens": {
            "secret_env": "JLENS_TEST_SECRET",
        }
    }

    with pytest.raises(ContractViolation, match="environment variable is missing"):
        _trusted_jlens_runtime(
            config=config,
            compiled=cast(CompiledRevision, object()),
            receipt_store=ReceiptStore(tmp_path / "receipts"),
        )


def test_build_tasks_binds_exact_clean_git_tree_and_feedback_identity(
    tmp_path: Path,
) -> None:
    rows = []
    for index, project in enumerate(("sphinx", "phpoffice", "laravel"), 1):
        source, revision = _source(tmp_path, project)
        rows.append(
            {
                "source_uri": str(source),
                "base_revision": revision,
                "instance_id": f"task-{index}",
                "project": project,
                "benchmark_id": "swe-bench-verified",
                "catalog_fingerprint": hashlib.sha256(project.encode()).hexdigest(),
            }
        )

    tasks, metadata, inventory = _build_tasks(rows, "native-v1")

    assert len(tasks) == len(metadata) == len(inventory) == 3
    assert all(str(task.cohort) == "feedback" for task in tasks)
    assert all(task.evaluator_id == "native-v1" for task in tasks)
    assert metadata[tasks[0].revision_id]["base_revision"] == rows[0]["base_revision"]


def test_seal_run_hashes_every_non_source_artifact_and_verifies(tmp_path: Path) -> None:
    (tmp_path / "artifact.txt").write_text("sealed", encoding="utf-8")
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources/large-cache.txt").write_text("cache", encoding="utf-8")

    assert seal_run(tmp_path) == 1
    manifest = json.loads((tmp_path / "EVIDENCE-MANIFEST.json").read_text())
    assert manifest["entries"] == [
        {
            "path": "artifact.txt",
            "sha256": hashlib.sha256(b"sealed").hexdigest(),
        }
    ]
    assert manifest["excluded_prefixes"] == ["sources/"]


def test_release_identity_rejects_a_dirty_worktree(tmp_path: Path) -> None:
    source, revision = _source(tmp_path, "release")
    assert _require_clean_head(source) == revision
    (source / "source.txt").write_text("dirty", encoding="utf-8")

    with pytest.raises(ContractViolation, match="clean committed HEAD"):
        _require_clean_head(source)


def test_python_launcher_preserves_venv_symlink(tmp_path: Path) -> None:
    target = tmp_path / "python-real"
    target.write_text("binary", encoding="utf-8")
    launcher = tmp_path / "venv-python"
    launcher.symlink_to(target)

    assert _launcher({"python": str(launcher)}, "python") == launcher.absolute()


def test_fresh_campaign_consumes_precompiled_candidate_with_parent_lineage(
    tmp_path: Path,
) -> None:
    request = {
        "request_id": "teacher-r1",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "failure_package": {"feedback_only": True},
    }
    request_path = tmp_path / "TEACHER-REQUEST.json"
    request_path.write_text(canonical_json(request) + "\n", encoding="utf-8")
    response = {
        "schema_version": 1,
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "candidate": {
            "protocol": "skill-v1",
            "prompt_template": "Repair {task}",
            "skill_text": "Localize the declared symbol before editing.",
            "eval_note": "Native feedback paired evaluation.",
        },
        "candidate_status": "inactive",
        "auto_activate": False,
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        "network_calls": 1,
        "pricing_cny_per_million": {"input": 2.0, "output": 8.0},
        "estimated_cost_cny": 0.00006,
    }
    response_path = tmp_path / "TEACHER-RESPONSE.json"
    response_path.write_text(canonical_json(response) + "\n", encoding="utf-8")
    compiled = CandidateCompiler().compile(
        request_path=request_path,
        response_path=response_path,
        compile_spec=CompileSpec(
            candidate_id="candidate-r1",
            revision_id="candidate-r1-revision",
            parent_revision_id="best-parent-r0",
            cohort=Cohort.FEEDBACK,
            operator_id="operator-r1",
            operator_instruction="Apply skill.",
            routes=(("task-a", "operator-r1"),),
        ),
        output_root=tmp_path / "compiled",
    )

    from evolve.fresh_feedback import _compile_teacher_candidate

    loaded = _compile_teacher_candidate(
        config={
            "compiled_revision_root": str(compiled.root),
            "candidate_id": compiled.change_set.candidate_id,
            "candidate_revision_id": compiled.change_set.revision_id,
            "parent_revision_id": "best-parent-r0",
            "tasks": [{"instance_id": "task-a"}],
        },
        output_root=tmp_path / "run",
    )

    assert loaded.bundle_sha256 == compiled.bundle_sha256
    assert loaded.change_set.parent_revision_id == "best-parent-r0"
