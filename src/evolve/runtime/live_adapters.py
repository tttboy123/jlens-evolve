"""Fail-closed adapters for live feedback execution against frozen legacy assets."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping

from evolve.contracts import Cohort, ContractViolation, ExecutionPlan

from .execution_runtime import EvaluatorInfrastructureError

_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SUPPORTED_BENCHMARKS = {
    "swe-bench-verified",
    "swe-bench-multilingual",
    "multi-swe-bench-flash",
}


class FrozenSourceWorkspaceManager:
    """Admit only a clean, exact Git checkout for a feedback plan."""

    def materialize(self, plan: ExecutionPlan) -> Mapping[str, Any]:
        if plan.task.cohort is not Cohort.FEEDBACK:
            raise ContractViolation("frozen workspace accepts feedback plans only")
        if plan.holdout_scope != "feedback-only":
            raise ContractViolation("feedback workspace requires feedback-only scope")

        source_uri = plan.task.source_uri
        if not isinstance(source_uri, str) or not source_uri:
            raise ContractViolation("task source_uri must identify a Git checkout")
        checkout = Path(source_uri)
        if not checkout.is_absolute():
            raise ContractViolation("task source_uri must be absolute")
        checkout = checkout.resolve()
        if not checkout.is_dir():
            raise ContractViolation("task source_uri Git checkout is missing")

        base_revision = plan.metadata.get("base_revision")
        if not isinstance(base_revision, str) or not _GIT_REVISION.fullmatch(
            base_revision
        ):
            raise ContractViolation("plan metadata base_revision must be a Git SHA")
        benchmark_id = plan.metadata.get("benchmark_id")
        instance_id = plan.metadata.get("instance_id")
        if benchmark_id not in _SUPPORTED_BENCHMARKS:
            raise ContractViolation("plan metadata benchmark_id is unsupported")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ContractViolation("plan metadata instance_id is missing")

        top_level = self._git(checkout, "rev-parse", "--show-toplevel")
        if Path(top_level).resolve() != checkout:
            raise ContractViolation("task source_uri must be the Git checkout root")
        head = self._git(checkout, "rev-parse", "HEAD")
        if head != base_revision:
            raise ContractViolation("Git HEAD does not match plan base_revision")
        if self._git(checkout, "status", "--porcelain", "--untracked-files=all"):
            raise ContractViolation("frozen Git checkout must be clean")
        git_tree = self._git(checkout, "rev-parse", "HEAD^{tree}")

        return {
            "benchmark_id": benchmark_id,
            "checkout": str(checkout),
            "git_tree": git_tree,
            "instance_id": instance_id,
            "project": plan.task.project,
            "source_revision": head,
            "task_id": plan.task.task_id,
            "task_revision_id": plan.task.revision_id,
            "task_source_sha256": plan.task.source_sha256,
        }

    @staticmethod
    def _git(checkout: Path, *args: str) -> str:
        completed = subprocess.run(
            ("git", *args),
            cwd=checkout,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise ContractViolation("task source_uri is not a readable Git checkout")
        return completed.stdout.strip()


class LegacyOfficialNativeEvaluator:
    """Adapt a literal v3 patch to the pinned legacy official evaluator."""

    def __init__(
        self,
        *,
        evaluator_id: str,
        legacy_root: Path,
        swe_python: Path,
        multi_python: Path,
        swe_harness_root: Path,
        multi_harness_root: Path,
        pool_root: Path,
        output_root: Path,
        timeout_seconds: int = 7200,
        evaluator_call: Callable[[Any, dict[str, Any], dict[str, Any]], Path]
        | None = None,
        normalizer: Callable[..., Any] | None = None,
    ) -> None:
        if not evaluator_id.strip():
            raise ContractViolation("native evaluator_id must be non-empty")
        if timeout_seconds < 1:
            raise ContractViolation("native evaluator timeout must be positive")
        self.evaluator_id = evaluator_id
        self.legacy_root = legacy_root.resolve()
        self.swe_python = swe_python.absolute()
        self.multi_python = multi_python.absolute()
        self.swe_harness_root = swe_harness_root.resolve()
        self.multi_harness_root = multi_harness_root.resolve()
        self.pool_root = pool_root.resolve()
        self.output_root = output_root.resolve()
        self.timeout_seconds = timeout_seconds
        self._evaluator_call = evaluator_call
        self._normalizer = normalizer

    def evaluate(
        self,
        plan: ExecutionPlan,
        workspace: Mapping[str, Any],
        model_output: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        benchmark_id, instance_id = self._validate_identity(
            plan, workspace, model_output
        )
        patch = model_output["patch"]
        patch_sha256 = model_output["patch_sha256"]
        patch_path = self._freeze_patch(plan.plan_id, patch, patch_sha256)

        invocation = SimpleNamespace(
            round_id=plan.plan_id,
            arm=plan.arm,
            benchmark_id=benchmark_id,
            instance_id=instance_id,
            agent_program_sha256=hashlib.sha256(
                plan.candidate_revision_id.encode("utf-8")
            ).hexdigest(),
        )
        try:
            evaluator = self._evaluator_call or self._load_evaluator()
            report_path = Path(
                evaluator(
                    invocation,
                    dict(workspace),
                    {"prediction": {"path": str(patch_path)}},
                )
            ).resolve()
            report, report_sha256, receipt_path, receipt_sha256 = (
                self._read_official_evidence(report_path)
            )
            normalizer = self._normalizer or self._load_normalizer()
            outcome = normalizer(
                report, benchmark_id=benchmark_id, instance_id=instance_id
            )
            normalized = self._normalized_outcome(outcome)
        except EvaluatorInfrastructureError:
            raise
        except Exception as error:
            raise EvaluatorInfrastructureError(
                f"legacy official evaluator failed: {type(error).__name__}: {error}"
            ) from error

        return {
            "resolved": normalized["resolved"],
            "native_valid": normalized["native_valid"],
            "native_error": normalized["native_error"],
            "regressions": normalized["regressions"],
            "native_report_path": str(report_path),
            "native_report_sha256": report_sha256,
            "official_receipt_path": str(receipt_path),
            "official_receipt_sha256": receipt_sha256,
            "patch_sha256": patch_sha256,
            "benchmark_id": benchmark_id,
            "instance_id": instance_id,
        }

    def _validate_identity(
        self,
        plan: ExecutionPlan,
        workspace: Mapping[str, Any],
        model_output: Mapping[str, Any],
    ) -> tuple[str, str]:
        if plan.task.cohort is not Cohort.FEEDBACK:
            raise ContractViolation("native evaluator accepts feedback plans only")
        if plan.holdout_scope != "feedback-only":
            raise ContractViolation("native evaluator requires feedback-only scope")
        if plan.native_evaluator_id != self.evaluator_id:
            raise ContractViolation("native evaluator identity drift")

        benchmark_id = plan.metadata.get("benchmark_id")
        instance_id = plan.metadata.get("instance_id")
        if benchmark_id not in _SUPPORTED_BENCHMARKS:
            raise ContractViolation("native task benchmark is unsupported")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ContractViolation("native task instance identity is missing")

        expected_workspace = {
            "benchmark_id": benchmark_id,
            "instance_id": instance_id,
            "task_id": plan.task.task_id,
            "task_revision_id": plan.task.revision_id,
            "task_source_sha256": plan.task.source_sha256,
            "source_revision": plan.metadata.get("base_revision"),
        }
        for field, expected in expected_workspace.items():
            if workspace.get(field) != expected:
                raise ContractViolation(f"workspace {field} identity drift")
        if plan.task.source_uri is None or workspace.get("checkout") != str(
            Path(plan.task.source_uri).resolve()
        ):
            raise ContractViolation("workspace checkout identity drift")

        expected_output = {
            "arm": plan.arm,
            "plan_id": plan.plan_id,
            "task_revision_id": plan.task.revision_id,
            "task_source_sha256": plan.task.source_sha256,
        }
        for field, expected in expected_output.items():
            if model_output.get(field) != expected:
                raise ContractViolation(f"model output {field} identity drift")
        patch = model_output.get("patch")
        patch_sha256 = model_output.get("patch_sha256")
        if not isinstance(patch, str):
            raise ContractViolation("model output patch must be literal text")
        if (
            not isinstance(patch_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", patch_sha256) is None
            or hashlib.sha256(patch.encode("utf-8")).hexdigest() != patch_sha256
        ):
            raise ContractViolation("model output patch identity drift")
        return benchmark_id, instance_id

    def _freeze_patch(self, plan_id: str, patch: str, expected_sha256: str) -> Path:
        identity = hashlib.sha256(plan_id.encode("utf-8")).hexdigest()
        root = self.output_root / "v3-predictions" / identity
        root.mkdir(parents=True, exist_ok=True)
        path = root / "patch.diff"
        encoded = patch.encode("utf-8")
        try:
            with path.open("xb") as handle:
                handle.write(encoded)
        except FileExistsError:
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
                raise ContractViolation("frozen patch input drift")
        return path

    @staticmethod
    def _read_official_evidence(
        report_path: Path,
    ) -> tuple[dict[str, Any], str, Path, str]:
        if not report_path.is_file():
            raise EvaluatorInfrastructureError("native report is missing")
        report_bytes = report_path.read_bytes()
        report_sha256 = hashlib.sha256(report_bytes).hexdigest()
        report = json.loads(report_bytes)
        if not isinstance(report, dict):
            raise EvaluatorInfrastructureError("native report must be an object")

        receipt_path = report_path.parent / "NATIVE-EVALUATOR-RECEIPT.json"
        if not receipt_path.is_file():
            raise EvaluatorInfrastructureError("official evaluator receipt is missing")
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        native_ref = receipt.get("native_report") if isinstance(receipt, dict) else None
        if (
            not isinstance(receipt, dict)
            or receipt.get("status") != "completed"
            or not isinstance(native_ref, dict)
            or native_ref.get("path") != report_path.name
            or native_ref.get("sha256") != report_sha256
        ):
            raise EvaluatorInfrastructureError("official evaluator receipt drift")
        return (
            report,
            report_sha256,
            receipt_path,
            hashlib.sha256(receipt_bytes).hexdigest(),
        )

    @staticmethod
    def _normalized_outcome(outcome: Any) -> dict[str, Any]:
        if isinstance(outcome, Mapping):
            resolved = outcome.get("resolved")
            native_valid = outcome.get("native_valid")
            native_error = outcome.get("native_error")
            regressions = outcome.get(
                "regression_test_names", outcome.get("regressions")
            )
        else:
            resolved = getattr(outcome, "resolved", None)
            native_valid = getattr(outcome, "native_valid", None)
            native_error = getattr(outcome, "native_error", None)
            regressions = getattr(outcome, "regression_test_names", None)
        if not isinstance(resolved, bool) or not isinstance(native_valid, bool):
            raise EvaluatorInfrastructureError("normalized native booleans are invalid")
        if native_error is not None and not isinstance(native_error, str):
            raise EvaluatorInfrastructureError("normalized native error is invalid")
        if not isinstance(regressions, (tuple, list)) or any(
            not isinstance(item, str) for item in regressions
        ):
            raise EvaluatorInfrastructureError("normalized regressions are invalid")
        return {
            "resolved": resolved,
            "native_valid": native_valid,
            "native_error": native_error,
            "regressions": sorted(set(regressions)),
        }

    def _load_evaluator(self) -> Callable[[Any, dict[str, Any], dict[str, Any]], Path]:
        module = self._legacy_module("official_patch_evaluator")
        evaluator_type = getattr(module, "OfficialPatchEvaluator")
        return evaluator_type(
            swe_python=self.swe_python,
            multi_python=self.multi_python,
            swe_harness_root=self.swe_harness_root,
            multi_harness_root=self.multi_harness_root,
            pool_root=self.pool_root,
            output_root=self.output_root,
            timeout_seconds=self.timeout_seconds,
        )

    def _load_normalizer(self) -> Callable[..., Any]:
        return getattr(
            self._legacy_module("skill_evolution_loop.p1_native"),
            "normalize_native_report",
        )

    def _legacy_module(self, name: str) -> Any:
        if not self.legacy_root.is_dir():
            raise EvaluatorInfrastructureError("legacy source root is missing")
        with _prepend_import_path(self.legacy_root):
            module = importlib.import_module(name)
        raw_module_path = getattr(module, "__file__", None)
        if not isinstance(raw_module_path, str):
            raise EvaluatorInfrastructureError("legacy module path is missing")
        module_path = Path(raw_module_path).resolve()
        if not module_path.is_relative_to(self.legacy_root):
            raise EvaluatorInfrastructureError("legacy module identity drift")
        return module


@contextmanager
def _prepend_import_path(root: Path) -> Iterator[None]:
    rendered = str(root)
    sys.path.insert(0, rendered)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == rendered:
            sys.path.pop(0)
        else:
            sys.path.remove(rendered)
