from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolve.contracts import (
    Cohort,
    ContractViolation,
    ExecutionLimits,
    ExecutionPlan,
    ModelIdentity,
    TaskRevision,
)
from evolve.runtime import EvaluatorInfrastructureError
from evolve.runtime.live_adapters import (
    FrozenSourceWorkspaceManager,
    LegacyOfficialNativeEvaluator,
)

SHA = "a" * 64


def _git(*args: str, cwd: Path) -> str:
    completed = subprocess.run(
        ("git", *args), cwd=cwd, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _checkout(tmp_path: Path) -> tuple[Path, str]:
    checkout = tmp_path / "source"
    checkout.mkdir()
    _git("init", cwd=checkout)
    _git("config", "user.email", "tests@example.invalid", cwd=checkout)
    _git("config", "user.name", "Tests", cwd=checkout)
    (checkout / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git("add", "module.py", cwd=checkout)
    _git("commit", "-m", "fixture", cwd=checkout)
    return checkout, _git("rev-parse", "HEAD", cwd=checkout)


def _plan(
    checkout: Path,
    revision: str,
    *,
    cohort: Cohort = Cohort.FEEDBACK,
    metadata: dict[str, object] | None = None,
) -> ExecutionPlan:
    task = TaskRevision(
        task_id="round1-sphinx-doc__sphinx-7757",
        revision_id="feedback-sphinx-7757-v1",
        project="sphinx-doc/sphinx",
        cohort=cohort,
        source_sha256=SHA,
        evaluator_id="official-native-v1",
        source_uri=str(checkout),
    )
    return ExecutionPlan(
        plan_id="plan-baseline",
        campaign_id="campaign-live",
        strategy_id="skill-paired-v3",
        task=task,
        candidate_revision_id="candidate-baseline",
        arm="baseline",
        model=ModelIdentity(
            provider="local-mlx", model="Qwen3.5-4B", revision="frozen"
        ),
        context_policy_id="operator-context-v1",
        tool_policy_id="deterministic-operator-v1",
        observer_policy_ids=("native",),
        native_evaluator_id=task.evaluator_id,
        limits=ExecutionLimits(max_tokens=1536, max_seconds=900, max_cost_cny=0),
        holdout_scope="feedback-only" if cohort is Cohort.FEEDBACK else "holdout",
        metadata=metadata
        or {
            "base_revision": revision,
            "benchmark_id": "swe-bench-verified",
            "instance_id": "sphinx-doc__sphinx-7757",
        },
    )


def test_frozen_source_workspace_materializes_exact_feedback_checkout(
    tmp_path: Path,
) -> None:
    checkout, revision = _checkout(tmp_path)

    workspace = FrozenSourceWorkspaceManager().materialize(_plan(checkout, revision))

    assert workspace == {
        "benchmark_id": "swe-bench-verified",
        "checkout": str(checkout.resolve()),
        "git_tree": _git("rev-parse", "HEAD^{tree}", cwd=checkout),
        "instance_id": "sphinx-doc__sphinx-7757",
        "project": "sphinx-doc/sphinx",
        "source_revision": revision,
        "task_id": "round1-sphinx-doc__sphinx-7757",
        "task_revision_id": "feedback-sphinx-7757-v1",
        "task_source_sha256": SHA,
    }


def test_frozen_source_workspace_rejects_non_feedback_and_revision_drift(
    tmp_path: Path,
) -> None:
    checkout, revision = _checkout(tmp_path)
    manager = FrozenSourceWorkspaceManager()

    with pytest.raises(ContractViolation, match="feedback"):
        manager.materialize(_plan(checkout, revision, cohort=Cohort.HOLDOUT))
    with pytest.raises(ContractViolation, match="HEAD"):
        manager.materialize(_plan(checkout, "b" * 40))


def test_legacy_official_evaluator_freezes_patch_and_normalizes_receipts(
    tmp_path: Path,
) -> None:
    checkout, revision = _checkout(tmp_path)
    plan = _plan(checkout, revision)
    workspace = FrozenSourceWorkspaceManager().materialize(plan)
    patch = "diff --git a/module.py b/module.py\n"
    patch_sha256 = hashlib.sha256(patch.encode()).hexdigest()
    observed: list[tuple[object, dict[str, object], dict[str, object]]] = []

    def official_call(invocation, materialized, receipt):
        observed.append((invocation, materialized, receipt))
        report_root = tmp_path / "official" / plan.plan_id / plan.arm
        report_root.mkdir(parents=True)
        report = report_root / "native-report.json"
        report.write_text(
            json.dumps(
                {
                    plan.metadata["instance_id"]: {
                        "resolved": True,
                        "patch_successfully_applied": True,
                        "tests_status": {
                            "PASS_TO_PASS": {"failure": []},
                            "PASS_TO_FAIL": {"failure": ["test_regression"]},
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        report_sha256 = hashlib.sha256(report.read_bytes()).hexdigest()
        (report_root / "NATIVE-EVALUATOR-RECEIPT.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "native_report": {
                        "path": report.name,
                        "sha256": report_sha256,
                    },
                }
            ),
            encoding="utf-8",
        )
        return report

    def normalize(report, *, benchmark_id, instance_id):
        row = report[instance_id]
        return SimpleNamespace(
            resolved=row["resolved"],
            native_valid=row["patch_successfully_applied"],
            native_error=None,
            regression_test_names=("test_regression",),
        )

    adapter = LegacyOfficialNativeEvaluator(
        evaluator_id=plan.native_evaluator_id,
        legacy_root=tmp_path / "legacy",
        swe_python=tmp_path / "venv/bin/python",
        multi_python=tmp_path / "venv/bin/python",
        swe_harness_root=tmp_path / "swe-harness",
        multi_harness_root=tmp_path / "multi-harness",
        pool_root=tmp_path / "pool",
        output_root=tmp_path / "official",
        evaluator_call=official_call,
        normalizer=normalize,
    )

    result = adapter.evaluate(
        plan,
        workspace,
        {
            "arm": plan.arm,
            "patch": patch,
            "patch_sha256": patch_sha256,
            "plan_id": plan.plan_id,
            "task_revision_id": plan.task.revision_id,
            "task_source_sha256": plan.task.source_sha256,
        },
    )

    assert result["resolved"] is True
    assert result["native_valid"] is True
    assert result["regressions"] == ["test_regression"]
    assert result["native_report_sha256"] == hashlib.sha256(
        Path(result["native_report_path"]).read_bytes()
    ).hexdigest()
    assert result["official_receipt_sha256"] == hashlib.sha256(
        Path(result["official_receipt_path"]).read_bytes()
    ).hexdigest()
    invocation, materialized, receipt = observed[0]
    assert invocation.instance_id == "sphinx-doc__sphinx-7757"
    assert materialized["task_revision_id"] == plan.task.revision_id
    assert Path(receipt["prediction"]["path"]).read_text() == patch


def test_legacy_official_evaluator_rejects_patch_identity_drift(tmp_path: Path) -> None:
    checkout, revision = _checkout(tmp_path)
    plan = _plan(checkout, revision)
    workspace = FrozenSourceWorkspaceManager().materialize(plan)
    called = False

    def must_not_run(*args):
        nonlocal called
        called = True
        raise AssertionError("identity drift reached the harness")

    adapter = LegacyOfficialNativeEvaluator(
        evaluator_id=plan.native_evaluator_id,
        legacy_root=tmp_path / "legacy",
        swe_python=tmp_path / "python",
        multi_python=tmp_path / "python",
        swe_harness_root=tmp_path / "swe",
        multi_harness_root=tmp_path / "multi",
        pool_root=tmp_path / "pool",
        output_root=tmp_path / "official",
        evaluator_call=must_not_run,
        normalizer=lambda *args, **kwargs: None,
    )

    with pytest.raises(ContractViolation, match="patch identity"):
        adapter.evaluate(
            plan,
            workspace,
            {
                "arm": plan.arm,
                "patch": "literal patch",
                "patch_sha256": "0" * 64,
                "plan_id": plan.plan_id,
                "task_revision_id": plan.task.revision_id,
                "task_source_sha256": plan.task.source_sha256,
            },
        )
    assert called is False


def test_legacy_official_evaluator_translates_harness_failure(tmp_path: Path) -> None:
    checkout, revision = _checkout(tmp_path)
    plan = _plan(checkout, revision)
    workspace = FrozenSourceWorkspaceManager().materialize(plan)
    patch = ""

    def failed_harness(*args):
        raise RuntimeError("official harness crashed")

    adapter = LegacyOfficialNativeEvaluator(
        evaluator_id=plan.native_evaluator_id,
        legacy_root=tmp_path / "legacy",
        swe_python=tmp_path / "python",
        multi_python=tmp_path / "python",
        swe_harness_root=tmp_path / "swe",
        multi_harness_root=tmp_path / "multi",
        pool_root=tmp_path / "pool",
        output_root=tmp_path / "official",
        evaluator_call=failed_harness,
        normalizer=lambda *args, **kwargs: None,
    )

    with pytest.raises(EvaluatorInfrastructureError, match="official harness crashed"):
        adapter.evaluate(
            plan,
            workspace,
            {
                "arm": plan.arm,
                "patch": patch,
                "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                "plan_id": plan.plan_id,
                "task_revision_id": plan.task.revision_id,
                "task_source_sha256": plan.task.source_sha256,
            },
        )


def test_legacy_official_evaluator_rejects_non_feedback_and_unsupported_task(
    tmp_path: Path,
) -> None:
    checkout, revision = _checkout(tmp_path)
    valid_plan = _plan(checkout, revision)
    workspace = FrozenSourceWorkspaceManager().materialize(valid_plan)

    def must_not_run(*args):
        raise AssertionError("invalid task reached the harness")

    adapter = LegacyOfficialNativeEvaluator(
        evaluator_id=valid_plan.native_evaluator_id,
        legacy_root=tmp_path / "legacy",
        swe_python=tmp_path / "python",
        multi_python=tmp_path / "python",
        swe_harness_root=tmp_path / "swe",
        multi_harness_root=tmp_path / "multi",
        pool_root=tmp_path / "pool",
        output_root=tmp_path / "official",
        evaluator_call=must_not_run,
        normalizer=lambda *args, **kwargs: None,
    )
    patch_sha256 = hashlib.sha256(b"").hexdigest()
    output = {
        "arm": valid_plan.arm,
        "patch": "",
        "patch_sha256": patch_sha256,
        "plan_id": valid_plan.plan_id,
        "task_revision_id": valid_plan.task.revision_id,
        "task_source_sha256": valid_plan.task.source_sha256,
    }

    holdout_plan = _plan(checkout, revision, cohort=Cohort.HOLDOUT)
    with pytest.raises(ContractViolation, match="feedback"):
        adapter.evaluate(holdout_plan, workspace, output)

    unsupported = _plan(
        checkout,
        revision,
        metadata={
            "base_revision": revision,
            "benchmark_id": "not-a-native-benchmark",
            "instance_id": "sphinx-doc__sphinx-7757",
        },
    )
    with pytest.raises(ContractViolation, match="unsupported"):
        adapter.evaluate(unsupported, workspace, output)


def test_legacy_official_evaluator_rejects_receipt_hash_drift(tmp_path: Path) -> None:
    checkout, revision = _checkout(tmp_path)
    plan = _plan(checkout, revision)
    workspace = FrozenSourceWorkspaceManager().materialize(plan)

    def drifted_receipt(invocation, materialized, receipt):
        root = tmp_path / "official-result"
        root.mkdir()
        report = root / "native-report.json"
        report.write_text("{}", encoding="utf-8")
        (root / "NATIVE-EVALUATOR-RECEIPT.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "native_report": {
                        "path": report.name,
                        "sha256": "0" * 64,
                    },
                }
            ),
            encoding="utf-8",
        )
        return report

    adapter = LegacyOfficialNativeEvaluator(
        evaluator_id=plan.native_evaluator_id,
        legacy_root=tmp_path / "legacy",
        swe_python=tmp_path / "python",
        multi_python=tmp_path / "python",
        swe_harness_root=tmp_path / "swe",
        multi_harness_root=tmp_path / "multi",
        pool_root=tmp_path / "pool",
        output_root=tmp_path / "official",
        evaluator_call=drifted_receipt,
        normalizer=lambda *args, **kwargs: None,
    )
    patch = ""

    with pytest.raises(EvaluatorInfrastructureError, match="receipt drift"):
        adapter.evaluate(
            plan,
            workspace,
            {
                "arm": plan.arm,
                "patch": patch,
                "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                "plan_id": plan.plan_id,
                "task_revision_id": plan.task.revision_id,
                "task_source_sha256": plan.task.source_sha256,
            },
        )
