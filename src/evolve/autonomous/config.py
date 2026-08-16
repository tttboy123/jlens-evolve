"""Strict, user-facing configuration for autonomous evolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from evolve.contracts import ContractViolation


class AutonomousEvolutionError(ContractViolation):
    """An autonomous product input or invariant is invalid."""


def _object(name: str, value: object, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise AutonomousEvolutionError(f"{name} fields are invalid")
    return value


def _object_with_optional(
    name: str,
    value: object,
    *,
    required: set[str],
    optional: set[str],
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or not required.issubset(value)
        or not set(value).issubset(required | optional)
    ):
        raise AutonomousEvolutionError(f"{name} fields are invalid")
    return value


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AutonomousEvolutionError(f"{name} must be non-empty text")
    return value


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise AutonomousEvolutionError(f"{name} must be a positive number")
    return float(value)


def _temperature_schedule(name: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise AutonomousEvolutionError(f"{name} must be a non-empty list")
    out: list[float] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not (0.0 <= float(item) <= 2.0)
        ):
            raise AutonomousEvolutionError(
                f"{name} entries must be numbers in [0, 2]"
            )
        out.append(float(item))
    return tuple(out)


def _temperature(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 2.0):
        raise AutonomousEvolutionError(f"{name} must be a number in [0, 2]")
    return float(value)


def _positive_int(name: str, value: object, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise AutonomousEvolutionError(f"{name} must be a {qualifier} integer")
    return value


def _path(name: str, value: object, *, file: bool) -> Path:
    result = Path(_text(name, value)).expanduser().resolve()
    valid = result.is_file() if file else result.is_dir()
    if not valid:
        kind = "file" if file else "directory"
        raise AutonomousEvolutionError(f"{name} must be an existing {kind}")
    return result


@dataclass(frozen=True, slots=True)
class GoalConfig:
    goal_id: str
    description: str
    target_native_gains: int
    max_rounds: int
    no_progress_patience: int
    max_consecutive_infra_failures: int
    max_same_failure_signature: int
    disk_limit_bytes: int


@dataclass(frozen=True, slots=True)
class ModelConfig:
    provider: str
    model_path: Path
    model_identity_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SweBenchConfig:
    task_pool: Path
    source_pool: Path
    official_harness: Path
    official_evaluator: Path
    cohort: str


@dataclass(frozen=True, slots=True)
class TeacherConfig:
    provider: str
    model: str
    endpoint: str
    api_key_env: str
    budget_cny: float
    max_output_tokens: int
    timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_base_delay: float = 1.0
    temperature: float = 0.0
    temperature_schedule: tuple[float, ...] = (0.0,)


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    tasks_per_campaign: int
    qwen_prescreen_count: int
    native_finalist_count: int
    seed: int


@dataclass(frozen=True, slots=True)
class AutonomousEvolutionConfig:
    schema_version: int
    goal: GoalConfig
    model: ModelConfig
    swe_bench: SweBenchConfig
    teacher: TeacherConfig
    execution: ExecutionConfig

    @classmethod
    def load(cls, path: str | Path) -> AutonomousEvolutionConfig:
        source = Path(path).expanduser().resolve()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise AutonomousEvolutionError(
                "autonomous evolution config is unreadable"
            ) from error
        root = _object(
            "autonomous evolution config",
            payload,
            {"schema_version", "goal", "model", "swe_bench", "teacher", "execution"},
        )
        if root["schema_version"] != 1:
            raise AutonomousEvolutionError(
                "autonomous evolution config schema is unsupported"
            )
        goal_fields = {
            "goal_id",
            "description",
            "target_native_gains",
            "max_rounds",
            "no_progress_patience",
        }
        goal_row = _object_with_optional(
            "goal",
            root["goal"],
            required=goal_fields,
            optional={
                "max_consecutive_infra_failures",
                "max_same_failure_signature",
                "disk_limit_bytes",
            },
        )
        model_row = _object(
            "model",
            root["model"],
            {"provider", "model_path", "model_identity_files"},
        )
        swe_row = _object(
            "swe_bench",
            root["swe_bench"],
            {
                "task_pool",
                "source_pool",
                "official_harness",
                "official_evaluator",
                "cohort",
            },
        )
        teacher_row = _object_with_optional(
            "teacher",
            root["teacher"],
            required={
                "provider",
                "model",
                "endpoint",
                "api_key_env",
                "budget_cny",
                "max_output_tokens",
            },
            optional={
                "timeout_seconds",
                "max_retries",
                "retry_base_delay",
                "temperature",
                "temperature_schedule",
            },
        )
        execution_row = _object(
            "execution",
            root["execution"],
            {
                "tasks_per_campaign",
                "qwen_prescreen_count",
                "native_finalist_count",
                "seed",
            },
        )

        identity_files = model_row["model_identity_files"]
        if (
            not isinstance(identity_files, list)
            or not identity_files
            or any(
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                for name in identity_files
            )
            or len(set(identity_files)) != len(identity_files)
        ):
            raise AutonomousEvolutionError(
                "model_identity_files must be unique file names"
            )
        model_path = _path("model.model_path", model_row["model_path"], file=False)
        missing = [name for name in identity_files if not (model_path / name).is_file()]
        if missing:
            raise AutonomousEvolutionError(
                "model identity file is missing: " + ", ".join(missing)
            )
        cohort = _text("swe_bench.cohort", swe_row["cohort"])
        if cohort != "feedback":
            raise AutonomousEvolutionError("autonomous evolution is feedback-only")
        budget = teacher_row["budget_cny"]
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget < 0:
            raise AutonomousEvolutionError("teacher.budget_cny must be non-negative")
        tasks_per_campaign = _positive_int(
            "execution.tasks_per_campaign", execution_row["tasks_per_campaign"]
        )
        if tasks_per_campaign != 3:
            raise AutonomousEvolutionError(
                "paired autonomous campaigns require exactly three tasks"
            )
        prescreen_count = _positive_int(
            "execution.qwen_prescreen_count",
            execution_row["qwen_prescreen_count"],
            allow_zero=True,
        )
        finalist_count = _positive_int(
            "execution.native_finalist_count",
            execution_row["native_finalist_count"],
        )
        if prescreen_count not in {0, 1} or finalist_count != 1:
            raise AutonomousEvolutionError(
                "single-candidate search requires qwen_prescreen_count 0 or 1 "
                "and native_finalist_count 1"
            )
        return cls(
            schema_version=1,
            goal=GoalConfig(
                goal_id=_text("goal.goal_id", goal_row["goal_id"]),
                description=_text("goal.description", goal_row["description"]),
                target_native_gains=_positive_int(
                    "goal.target_native_gains", goal_row["target_native_gains"]
                ),
                max_rounds=_positive_int("goal.max_rounds", goal_row["max_rounds"]),
                no_progress_patience=_positive_int(
                    "goal.no_progress_patience", goal_row["no_progress_patience"]
                ),
                max_consecutive_infra_failures=_positive_int(
                    "goal.max_consecutive_infra_failures",
                    goal_row.get("max_consecutive_infra_failures", 3),
                ),
                max_same_failure_signature=_positive_int(
                    "goal.max_same_failure_signature",
                    goal_row.get("max_same_failure_signature", 5),
                ),
                disk_limit_bytes=_positive_int(
                    "goal.disk_limit_bytes",
                    goal_row.get("disk_limit_bytes", 10 * 1024 * 1024 * 1024),
                ),
            ),
            model=ModelConfig(
                provider=_text("model.provider", model_row["provider"]),
                model_path=model_path,
                model_identity_files=tuple(identity_files),
            ),
            swe_bench=SweBenchConfig(
                task_pool=_path("swe_bench.task_pool", swe_row["task_pool"], file=True),
                source_pool=_path(
                    "swe_bench.source_pool", swe_row["source_pool"], file=False
                ),
                official_harness=_path(
                    "swe_bench.official_harness",
                    swe_row["official_harness"],
                    file=False,
                ),
                official_evaluator=_path(
                    "swe_bench.official_evaluator",
                    swe_row["official_evaluator"],
                    file=True,
                ),
                cohort=cohort,
            ),
            teacher=TeacherConfig(
                provider=_text("teacher.provider", teacher_row["provider"]),
                model=_text("teacher.model", teacher_row["model"]),
                endpoint=_text("teacher.endpoint", teacher_row["endpoint"]),
                api_key_env=_text("teacher.api_key_env", teacher_row["api_key_env"]),
                budget_cny=float(budget),
                max_output_tokens=_positive_int(
                    "teacher.max_output_tokens", teacher_row["max_output_tokens"]
                ),
                timeout_seconds=_positive_float(
                    "teacher.timeout_seconds",
                    teacher_row.get("timeout_seconds", 60.0),
                ),
                max_retries=_positive_int(
                    "teacher.max_retries",
                    teacher_row.get("max_retries", 3),
                ),
                retry_base_delay=_positive_float(
                    "teacher.retry_base_delay",
                    teacher_row.get("retry_base_delay", 1.0),
                ),
                temperature=_temperature(
                    "teacher.temperature", teacher_row.get("temperature", 0.0)
                ),
                temperature_schedule=_temperature_schedule(
                    "teacher.temperature_schedule",
                    teacher_row.get("temperature_schedule", [0.0]),
                ),
            ),
            execution=ExecutionConfig(
                tasks_per_campaign=tasks_per_campaign,
                qwen_prescreen_count=prescreen_count,
                native_finalist_count=finalist_count,
                seed=_positive_int(
                    "execution.seed", execution_row["seed"], allow_zero=True
                ),
            ),
        )
