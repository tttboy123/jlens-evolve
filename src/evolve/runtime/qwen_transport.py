"""Frozen local-Qwen transport and the legacy paired-generation bridge."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Protocol

from evolve.contracts import Cohort, ContractViolation, ExecutionPlan, canonical_json


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


class QwenCellRunner(Protocol):
    def run(
        self,
        plan: ExecutionPlan,
        workspace: Mapping[str, Any],
        output_root: Path,
    ) -> Mapping[str, Any]: ...


class LegacyQwenPairTransport:
    """Dispatch one fresh feedback arm through a frozen local-Qwen runner."""

    remote = False

    def __init__(self, *, cell_runner: QwenCellRunner, output_root: Path) -> None:
        self._cell_runner = cell_runner
        self._output_root = output_root.resolve()

    def infer(
        self, plan: ExecutionPlan, workspace: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if plan.task.cohort is not Cohort.FEEDBACK:
            raise ContractViolation("local Qwen transport is feedback-only")
        if plan.arm not in {"baseline", "taught"}:
            raise ContractViolation("local Qwen transport requires a paired arm")
        if workspace.get("task_revision_id") != plan.task.revision_id:
            raise ContractViolation("workspace task identity does not match plan")
        result = dict(self._cell_runner.run(plan, workspace, self._output_root))
        result.update(
            {
                "arm": plan.arm,
                "plan_id": plan.plan_id,
                "task_revision_id": plan.task.revision_id,
                "task_source_sha256": plan.task.source_sha256,
            }
        )
        required = {
            "patch",
            "patch_sha256",
            "raw_output_path",
            "raw_output_sha256",
            "prompt_paths",
            "prompt_sha256",
            "structural_valid",
            "failure_reason",
            "input_tokens",
            "output_tokens",
            "cost_cny",
        }
        if not required <= result.keys():
            raise ContractViolation("Qwen cell result is incomplete")
        patch = result["patch"]
        if not isinstance(patch, str) or _sha256_bytes(patch.encode()) != result[
            "patch_sha256"
        ]:
            raise ContractViolation("Qwen patch identity mismatch")
        raw_path = Path(str(result["raw_output_path"])).resolve()
        if not raw_path.is_file() or _sha256_file(raw_path) != result[
            "raw_output_sha256"
        ]:
            raise ContractViolation("Qwen raw output identity mismatch")
        prompt_paths = result["prompt_paths"]
        prompt_hashes = result["prompt_sha256"]
        if (
            not isinstance(prompt_paths, list)
            or not isinstance(prompt_hashes, list)
            or len(prompt_paths) != len(prompt_hashes)
        ):
            raise ContractViolation("Qwen prompt evidence is invalid")
        for raw_prompt, expected in zip(prompt_paths, prompt_hashes, strict=True):
            prompt = Path(str(raw_prompt)).resolve()
            if not prompt.is_file() or _sha256_file(prompt) != expected:
                raise ContractViolation("Qwen prompt identity mismatch")
        if not isinstance(result["structural_valid"], bool):
            raise ContractViolation("Qwen structural outcome is invalid")
        return result


class LegacyQwenCellRunner:
    """Run the existing typed operator/span Student without replaying old cells.

    This adapter is intentionally loaded only at execution time.  The v3 platform
    stays stdlib-only, while the authorized live command runs under the legacy
    virtualenv that owns MLX and the frozen Student implementation.
    """

    def __init__(
        self,
        *,
        legacy_root: Path,
        model_path: Path,
        taskset_path: Path,
        routes_path: Path,
        operator_skill_path: Path,
        span_skill_path: Path,
    ) -> None:
        self.legacy_root = legacy_root.resolve()
        self.model_path = model_path.resolve()
        self.taskset_path = taskset_path.resolve()
        self.routes_path = routes_path.resolve()
        self.operator_skill_path = operator_skill_path.resolve()
        self.span_skill_path = span_skill_path.resolve()
        for path in (
            self.legacy_root,
            self.model_path,
            self.taskset_path,
            self.routes_path,
            self.operator_skill_path,
            self.span_skill_path,
        ):
            if not path.exists():
                raise ContractViolation(f"legacy Qwen input is missing: {path}")
        self._input_sha256 = {
            "taskset": _sha256_file(self.taskset_path),
            "routes": _sha256_file(self.routes_path),
            "operator_skill": _sha256_file(self.operator_skill_path),
            "span_skill": _sha256_file(self.span_skill_path),
        }
        taskset = json.loads(self.taskset_path.read_text(encoding="utf-8"))
        routes = json.loads(self.routes_path.read_text(encoding="utf-8"))
        self._tasks = {
            str(row["instance_id"]): row
            for row in taskset.get("tasks", [])
            if row.get("cohort") == "feedback"
        }
        self._routes = {
            str(task_id): str(mechanism)
            for task_id, mechanism in routes.get("routes", {}).items()
        }
        self._runtime: dict[str, Any] | None = None

    def _load_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        root = str(self.legacy_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from mlx_lm import generate, load
            from skill_evolution_loop.operator_student import (
                MlxOperatorPlanGenerator,
                OperatorPlanAdapter,
                build_operator_conditions,
            )
            from skill_evolution_loop.p1_operator import (
                load_frozen_operator_skill_revision,
            )
            from skill_evolution_loop.span_student import (
                MlxSpanPlanGenerator,
                SpanPlanAdapter,
                build_span_conditions,
            )
        except ImportError as exc:
            raise ContractViolation(
                "legacy Qwen runtime dependencies are unavailable"
            ) from exc

        loaded: list[tuple[Any, Any]] = []

        def shared_loader(path: str) -> tuple[Any, Any]:
            if not loaded:
                loaded.append(load(path))
            return loaded[0]

        operator_generator = MlxOperatorPlanGenerator(
            model_path=self.model_path,
            loader=shared_loader,
            text_generator=generate,
            max_plan_repairs=1,
        )
        span_generator = MlxSpanPlanGenerator(
            model_path=self.model_path,
            loader=shared_loader,
            text_generator=generate,
            max_plan_repairs=1,
        )
        adapters = {
            "operator": OperatorPlanAdapter(generator=operator_generator),
            "span": SpanPlanAdapter(generator=span_generator),
        }
        operator_revision = load_frozen_operator_skill_revision(
            self.operator_skill_path
        )
        span_wrapper = json.loads(self.span_skill_path.read_text(encoding="utf-8"))
        if (
            span_wrapper.get("candidate_status") != "inactive"
            or span_wrapper.get("auto_activate") is not False
        ):
            raise ContractViolation("legacy span Skill is not inactive")
        conditions = [
            *build_operator_conditions(
                taught_skill=operator_revision.skill_text,
                parent_revision_id=operator_revision.revision_id,
                source_round=operator_revision.source_round,
                generation_config=adapters["operator"].experiment_config(),
            ),
            *build_span_conditions(
                taught_skill=str(span_wrapper["skill_text"]),
                parent_revision_id=operator_revision.revision_id,
                source_round=operator_revision.source_round,
                generation_config=adapters["span"].experiment_config(),
            ),
        ]
        condition_index = {
            (row.mechanism, row.teaching): row for row in conditions
        }
        self._runtime = {
            "adapters": adapters,
            "conditions": condition_index,
            "generators": {
                "operator": operator_generator,
                "span": span_generator,
            },
        }
        return self._runtime

    def run(
        self,
        plan: ExecutionPlan,
        workspace: Mapping[str, Any],
        output_root: Path,
    ) -> Mapping[str, Any]:
        current_input_sha256 = {
            "taskset": _sha256_file(self.taskset_path),
            "routes": _sha256_file(self.routes_path),
            "operator_skill": _sha256_file(self.operator_skill_path),
            "span_skill": _sha256_file(self.span_skill_path),
        }
        if current_input_sha256 != self._input_sha256:
            raise ContractViolation("frozen Qwen input artifact drift")
        candidate_bundle_sha256 = plan.metadata.get("candidate_bundle_sha256")
        if (
            not isinstance(candidate_bundle_sha256, str)
            or len(candidate_bundle_sha256) != 64
        ):
            raise ContractViolation("Qwen plan candidate bundle identity is missing")
        task_row = self._tasks.get(plan.task.task_id)
        if task_row is None:
            raise ContractViolation("Qwen task is not in the frozen feedback catalog")
        if (
            task_row.get("fingerprint") != plan.metadata.get("catalog_fingerprint")
            or task_row.get("benchmark_base_commit")
            != plan.metadata.get("base_revision")
            or task_row.get("benchmark_id") != plan.metadata.get("benchmark_id")
        ):
            raise ContractViolation("Qwen task catalog identity drift")
        legacy_task_id = str(task_row["task_id"])
        mechanism = self._routes.get(legacy_task_id)
        if mechanism not in {"operator", "span"}:
            raise ContractViolation("Qwen task has no supported mechanism route")
        checkout = Path(str(workspace.get("checkout", ""))).resolve()
        if not checkout.is_dir():
            raise ContractViolation("Qwen workspace checkout is missing")
        identity = {
            "plan_sha256": plan.content_sha256,
            "workspace_source_sha256": workspace.get("task_source_sha256"),
            "legacy_task_id": legacy_task_id,
            "mechanism": mechanism,
            "candidate_bundle_sha256": candidate_bundle_sha256,
            "qwen_input_sha256": self._input_sha256,
        }
        target = output_root.resolve() / plan.plan_id
        if target.exists():
            return self._load_frozen_result(target, identity)
        target.parent.mkdir(parents=True, exist_ok=True)
        runtime = self._load_runtime()
        from skill_evolution_loop.student_adapter import StudentTask

        student_task = StudentTask.create(
            task_id=legacy_task_id,
            checkout=checkout,
            instruction=str(task_row["instruction"]),
            allowed_targets=[str(item) for item in task_row["allowed_targets"]],
            cohort="feedback",
        )
        condition = runtime["conditions"][(mechanism, plan.arm)]
        adapter = runtime["adapters"][mechanism]
        started = time.monotonic()
        attempt = adapter.run(student_task, condition.revision)
        elapsed = time.monotonic() - started
        generator = runtime["generators"][mechanism]
        prompts = tuple(generator.generation_prompt_trace())
        traces = tuple(generator.generation_trace())
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{plan.plan_id}-", dir=target.parent)
        )
        try:
            (temporary / "REQUEST.json").write_text(
                canonical_json(identity) + "\n", encoding="utf-8"
            )
            (temporary / "raw-output.txt").write_text(
                attempt.raw_output, encoding="utf-8"
            )
            (temporary / "patch.diff").write_text(attempt.patch, encoding="utf-8")
            for index, prompt in enumerate(prompts):
                if prompt is not None:
                    (temporary / f"prompt-{index:03d}.txt").write_text(
                        prompt, encoding="utf-8"
                    )
            for index, trace in enumerate(traces):
                (temporary / f"generation-{index:03d}.txt").write_text(
                    trace, encoding="utf-8"
                )
            frozen = {
                "schema_version": 1,
                "request": identity,
                "patch": attempt.patch,
                "patch_sha256": _sha256_bytes(attempt.patch.encode()),
                "raw_output_file": "raw-output.txt",
                "raw_output_sha256": _sha256_bytes(attempt.raw_output.encode()),
                "prompt_files": [
                    f"prompt-{index:03d}.txt"
                    for index, prompt in enumerate(prompts)
                    if prompt is not None
                ],
                "structural_valid": attempt.structural_valid,
                "failure_reason": attempt.failure_reason,
                "elapsed_seconds": round(elapsed, 6),
                "mechanism": mechanism,
                "condition_id": condition.condition_id,
                "model_identity_sha256": self._model_identity(),
                "input_tokens": 0,
                "output_tokens": 0,
                "token_usage_available": False,
                "cost_cny": 0,
            }
            frozen["prompt_sha256"] = [
                _sha256_file(temporary / name) for name in frozen["prompt_files"]
            ]
            (temporary / "RESULT.json").write_text(
                canonical_json(frozen) + "\n", encoding="utf-8"
            )
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                for path in sorted(temporary.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
                temporary.rmdir()
        return self._load_frozen_result(target, identity)

    def _model_identity(self) -> dict[str, str]:
        names = (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer_config.json",
        )
        result = {}
        for name in names:
            path = self.model_path / name
            if not path.is_file():
                raise ContractViolation(f"frozen Qwen identity file is missing: {name}")
            result[name] = _sha256_file(path)
        return result

    @staticmethod
    def _load_frozen_result(
        target: Path, expected_identity: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        try:
            frozen = json.loads((target / "RESULT.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractViolation("frozen Qwen cell is unreadable") from exc
        if frozen.get("request") != dict(expected_identity):
            raise ContractViolation("frozen Qwen cell request identity drifted")
        raw = target / str(frozen["raw_output_file"])
        patch = str(frozen["patch"])
        if (
            not raw.is_file()
            or _sha256_file(raw) != frozen.get("raw_output_sha256")
            or _sha256_bytes(patch.encode()) != frozen.get("patch_sha256")
        ):
            raise ContractViolation("frozen Qwen cell artifact was tampered")
        prompt_files = [target / str(name) for name in frozen.get("prompt_files", [])]
        prompt_hashes = list(frozen.get("prompt_sha256", []))
        if len(prompt_files) != len(prompt_hashes) or any(
            not path.is_file() or _sha256_file(path) != digest
            for path, digest in zip(prompt_files, prompt_hashes, strict=True)
        ):
            raise ContractViolation("frozen Qwen prompt evidence was tampered")
        return {
            "patch": patch,
            "patch_sha256": frozen["patch_sha256"],
            "raw_output_path": str(raw),
            "raw_output_sha256": frozen["raw_output_sha256"],
            "prompt_paths": [str(path) for path in prompt_files],
            "prompt_sha256": prompt_hashes,
            "structural_valid": bool(frozen["structural_valid"]),
            "failure_reason": frozen.get("failure_reason"),
            "mechanism": frozen["mechanism"],
            "condition_id": frozen["condition_id"],
            "model_identity_sha256": frozen["model_identity_sha256"],
            "input_tokens": int(frozen["input_tokens"]),
            "output_tokens": int(frozen["output_tokens"]),
            "token_usage_available": bool(frozen["token_usage_available"]),
            "cost_cny": float(frozen["cost_cny"]),
            "elapsed_seconds": float(frozen["elapsed_seconds"]),
        }


__all__ = ["LegacyQwenCellRunner", "LegacyQwenPairTransport", "QwenCellRunner"]
