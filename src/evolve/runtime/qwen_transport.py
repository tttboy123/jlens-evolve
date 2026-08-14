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
from evolve.proposals import CompiledRevision

from .candidate_prompt import compiled_candidate_prompt


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
            "prompt_texts",
            "prompt_sha256",
            "candidate_prompt",
            "candidate_prompt_sha256",
            "compiled_artifact_sha256",
            "structural_valid",
            "failure_reason",
            "input_tokens",
            "output_tokens",
            "cost_cny",
        }
        if not required <= result.keys():
            raise ContractViolation("Qwen cell result is incomplete")
        patch = result["patch"]
        if (
            not isinstance(patch, str)
            or _sha256_bytes(patch.encode()) != result["patch_sha256"]
        ):
            raise ContractViolation("Qwen patch identity mismatch")
        raw_path = Path(str(result["raw_output_path"])).resolve()
        if (
            not raw_path.is_file()
            or _sha256_file(raw_path) != result["raw_output_sha256"]
        ):
            raise ContractViolation("Qwen raw output identity mismatch")
        prompt_paths = result["prompt_paths"]
        prompt_texts = result["prompt_texts"]
        prompt_hashes = result["prompt_sha256"]
        if (
            not isinstance(prompt_paths, list)
            or not isinstance(prompt_texts, list)
            or not isinstance(prompt_hashes, list)
            or len(prompt_paths) != len(prompt_texts)
            or len(prompt_paths) != len(prompt_hashes)
        ):
            raise ContractViolation("Qwen prompt evidence is invalid")
        for raw_prompt, prompt_text, expected in zip(
            prompt_paths, prompt_texts, prompt_hashes, strict=True
        ):
            prompt = Path(str(raw_prompt)).resolve()
            if (
                not isinstance(prompt_text, str)
                or not prompt.is_file()
                or prompt.read_text(encoding="utf-8") != prompt_text
                or _sha256_file(prompt) != expected
            ):
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
        compiled_revision_root: Path,
        baseline_compiled_revision_root: Path | None = None,
    ) -> None:
        self.legacy_root = legacy_root.resolve()
        self.model_path = model_path.resolve()
        self.taskset_path = taskset_path.resolve()
        self.routes_path = routes_path.resolve()
        # Store only the path here. Baseline execution must not read, hash, or
        # deserialize candidate content; taught execution loads it fail-closed.
        self.compiled_revision_root = compiled_revision_root.resolve()
        self.baseline_compiled_revision_root = (
            baseline_compiled_revision_root.resolve()
            if baseline_compiled_revision_root is not None
            else None
        )
        if self.baseline_compiled_revision_root == self.compiled_revision_root:
            raise ContractViolation(
                "baseline and proposed candidate must use separate roots"
            )
        for path in (
            self.legacy_root,
            self.model_path,
            self.taskset_path,
            self.routes_path,
        ):
            if not path.exists():
                raise ContractViolation(f"legacy Qwen input is missing: {path}")
        self._input_sha256 = {
            "taskset": _sha256_file(self.taskset_path),
            "routes": _sha256_file(self.routes_path),
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
            from mlx_lm import generate, load  # type: ignore[import-not-found]
            from skill_evolution_loop.operator_student import (  # type: ignore[import-not-found]
                MlxOperatorPlanGenerator,
                OperatorPlanAdapter,
                build_operator_conditions,
            )
            from skill_evolution_loop.span_student import (  # type: ignore[import-not-found]
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
        self._runtime = {
            "adapters": adapters,
            "condition_builders": {
                "operator": build_operator_conditions,
                "span": build_span_conditions,
            },
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
        }
        if current_input_sha256 != self._input_sha256:
            raise ContractViolation("frozen Qwen input artifact drift")
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
        compiled = self._compiled_for_plan(plan)
        parent_lineage = self._parent_harness_lineage(plan, compiled)
        candidate_prompt = (
            compiled_candidate_prompt(compiled, plan.task.task_id)
            if compiled is not None and plan.arm == "taught"
            else None
        )
        identity = {
            "plan_sha256": plan.content_sha256,
            "workspace_source_sha256": workspace.get("task_source_sha256"),
            "legacy_task_id": legacy_task_id,
            "mechanism": mechanism,
            "qwen_input_sha256": self._input_sha256,
            "candidate_consumed": plan.arm == "taught",
            "candidate_bundle_sha256": (
                compiled.bundle_sha256
                if compiled is not None and plan.arm == "taught"
                else None
            ),
            "candidate_revision_id": (
                compiled.change_set.revision_id
                if compiled is not None and plan.arm == "taught"
                else None
            ),
            "compiled_artifact_sha256": (
                dict(compiled.artifact_sha256)
                if compiled is not None and plan.arm == "taught"
                else {}
            ),
            "candidate_prompt_sha256": (
                _sha256_bytes(candidate_prompt.encode())
                if candidate_prompt is not None
                else None
            ),
            "parent_harness_revision_id": parent_lineage[
                "parent_harness_revision_id"
            ],
            "parent_harness_bundle_sha256": parent_lineage[
                "parent_harness_bundle_sha256"
            ],
            "parent_harness_prompt_sha256": parent_lineage[
                "parent_harness_prompt_sha256"
            ],
        }
        target = output_root.resolve() / plan.plan_id
        if target.exists():
            return self._load_frozen_result(target, identity)
        target.parent.mkdir(parents=True, exist_ok=True)
        runtime = self._load_runtime()
        from skill_evolution_loop.student_adapter import (  # type: ignore[import-not-found]
            StudentTask,
        )

        student_task = StudentTask.create(
            task_id=legacy_task_id,
            checkout=checkout,
            instruction=str(task_row["instruction"]),
            allowed_targets=[str(item) for item in task_row["allowed_targets"]],
            cohort="feedback",
        )
        adapter = runtime["adapters"][mechanism]
        condition = self._condition_for_plan(
            plan=plan,
            mechanism=mechanism,
            adapter=adapter,
            compiled=compiled,
            builder=runtime["condition_builders"][mechanism],
        )
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
                "candidate_consumed": plan.arm == "taught",
                "candidate_bundle_sha256": (
                    compiled.bundle_sha256
                    if compiled is not None and plan.arm == "taught"
                    else None
                ),
                "candidate_revision_id": (
                    compiled.change_set.revision_id
                    if compiled is not None and plan.arm == "taught"
                    else None
                ),
                "compiled_artifact_sha256": (
                    dict(compiled.artifact_sha256)
                    if compiled is not None and plan.arm == "taught"
                    else {}
                ),
                "candidate_prompt": candidate_prompt,
                "candidate_prompt_sha256": (
                    _sha256_bytes(candidate_prompt.encode())
                    if candidate_prompt is not None
                    else None
                ),
                **parent_lineage,
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

    def _compiled_for_plan(self, plan: ExecutionPlan) -> CompiledRevision | None:
        if plan.arm == "baseline":
            if self.baseline_compiled_revision_root is None:
                return None
            compiled = CompiledRevision.load(self.baseline_compiled_revision_root)
            if plan.candidate_revision_id != compiled.change_set.revision_id:
                raise ContractViolation("baseline plan parent harness revision mismatch")
            return compiled
        if plan.arm != "taught":
            raise ContractViolation("local Qwen runner requires paired arms")
        compiled = CompiledRevision.load(self.compiled_revision_root)
        if plan.candidate_revision_id != compiled.change_set.revision_id:
            raise ContractViolation("taught plan candidate revision mismatch")
        route = dict(compiled.router.routes).get(plan.task.task_id)
        if route is None:
            raise ContractViolation("compiled Router has no route for taught task")
        if route != compiled.operator.operator_id:
            raise ContractViolation("compiled Router selected another Operator")
        if self.baseline_compiled_revision_root is not None:
            parent = CompiledRevision.load(self.baseline_compiled_revision_root)
            if compiled.change_set.parent_revision_id != parent.change_set.revision_id:
                raise ContractViolation("proposed candidate parent harness mismatch")
        return compiled

    def _parent_harness_lineage(
        self,
        plan: ExecutionPlan,
        compiled: CompiledRevision | None,
    ) -> dict[str, str | None]:
        parent = compiled if plan.arm == "baseline" else None
        if plan.arm == "taught" and self.baseline_compiled_revision_root is not None:
            parent = CompiledRevision.load(self.baseline_compiled_revision_root)
        if parent is None:
            return {
                "parent_harness_revision_id": None,
                "parent_harness_bundle_sha256": None,
                "parent_harness_prompt": None,
                "parent_harness_prompt_sha256": None,
            }
        prompt = compiled_candidate_prompt(parent, plan.task.task_id)
        return {
            "parent_harness_revision_id": parent.change_set.revision_id,
            "parent_harness_bundle_sha256": parent.bundle_sha256,
            "parent_harness_prompt": prompt,
            "parent_harness_prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        }

    @staticmethod
    def _condition_for_plan(
        *,
        plan: ExecutionPlan,
        mechanism: str,
        adapter: Any,
        compiled: CompiledRevision | None,
        builder: Any,
    ) -> Any:
        if plan.arm == "baseline":
            # The builder requires a taught value even when selecting the
            # baseline row. This fixed sentinel is not candidate-derived.
            if compiled is None:
                teaching = "BASELINE-SENTINEL-NOT-CANDIDATE-DERIVED"
                parent_revision_id = "baseline-no-candidate"
            else:
                teaching = (
                    "BASELINE-HARNESS:\n"
                    + compiled_candidate_prompt(compiled, plan.task.task_id)
                )
                parent_revision_id = compiled.change_set.revision_id
        else:
            if compiled is None:
                raise ContractViolation("taught execution requires compiled candidate")
            teaching = compiled_candidate_prompt(compiled, plan.task.task_id)
            parent_revision_id = compiled.change_set.parent_revision_id
        conditions = builder(
            taught_skill=teaching,
            parent_revision_id=parent_revision_id,
            source_round=0,
            generation_config=adapter.experiment_config(),
        )
        try:
            return next(
                row
                for row in conditions
                if row.mechanism == mechanism and row.teaching == plan.arm
            )
        except StopIteration as error:
            raise ContractViolation(
                "legacy condition builder omitted paired arm"
            ) from error

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
        for name in (
            "candidate_consumed",
            "candidate_bundle_sha256",
            "candidate_revision_id",
            "compiled_artifact_sha256",
            "candidate_prompt_sha256",
            "parent_harness_revision_id",
            "parent_harness_bundle_sha256",
            "parent_harness_prompt_sha256",
        ):
            if frozen.get(name) != expected_identity.get(name):
                raise ContractViolation("frozen Qwen cell harness lineage drifted")
        parent_prompt = frozen.get("parent_harness_prompt")
        parent_prompt_sha256 = frozen.get("parent_harness_prompt_sha256")
        if parent_prompt_sha256 is None:
            if parent_prompt is not None:
                raise ContractViolation("frozen Qwen parent harness prompt drifted")
        elif (
            not isinstance(parent_prompt, str)
            or _sha256_bytes(parent_prompt.encode("utf-8"))
            != parent_prompt_sha256
        ):
            raise ContractViolation("frozen Qwen parent harness prompt was tampered")
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
            "prompt_texts": [
                path.read_bytes().decode("utf-8") for path in prompt_files
            ],
            "prompt_sha256": prompt_hashes,
            "structural_valid": bool(frozen["structural_valid"]),
            "failure_reason": frozen.get("failure_reason"),
            "mechanism": frozen["mechanism"],
            "condition_id": frozen["condition_id"],
            "candidate_consumed": bool(frozen["candidate_consumed"]),
            "candidate_bundle_sha256": frozen.get("candidate_bundle_sha256"),
            "candidate_revision_id": frozen.get("candidate_revision_id"),
            "compiled_artifact_sha256": frozen.get("compiled_artifact_sha256", {}),
            "candidate_prompt": frozen.get("candidate_prompt"),
            "candidate_prompt_sha256": frozen.get("candidate_prompt_sha256"),
            "parent_harness_revision_id": frozen.get(
                "parent_harness_revision_id"
            ),
            "parent_harness_bundle_sha256": frozen.get(
                "parent_harness_bundle_sha256"
            ),
            "parent_harness_prompt": frozen.get("parent_harness_prompt"),
            "parent_harness_prompt_sha256": frozen.get(
                "parent_harness_prompt_sha256"
            ),
            "model_identity_sha256": frozen["model_identity_sha256"],
            "input_tokens": int(frozen["input_tokens"]),
            "output_tokens": int(frozen["output_tokens"]),
            "token_usage_available": bool(frozen["token_usage_available"]),
            "cost_cny": float(frozen["cost_cny"]),
            "elapsed_seconds": float(frozen["elapsed_seconds"]),
        }


__all__ = ["LegacyQwenCellRunner", "LegacyQwenPairTransport", "QwenCellRunner"]
