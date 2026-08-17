"""Real Codex history adapter and deterministic project-profile replay.

The adapter intentionally does not invoke a language model.  It binds a real local
Codex CLI identity and real Codex Desktop user history to project-native Codex
surfaces, then evaluates the delivery contract compiled from those surfaces.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


class CodexHistoryError(ValueError):
    """Raised when selected Codex history no longer matches the frozen contract."""


class CodexProfileError(ValueError):
    """Raised when a project-local Codex profile violates the target contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _default_sessions_root() -> Path:
    """Resolve the real Codex session root.

    The default is the live ``~/.codex/sessions`` tree. Tests can override it
    by passing ``sessions_root`` to :meth:`CodexHistoryContract.from_path` so
    the contract validator can be exercised against a synthetic source that
    lives outside the user's real Codex history.
    """

    return (Path.home() / ".codex/sessions").resolve()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tree_hash(root: Path) -> str:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return _sha256_text(_canonical_json(rows))


@dataclass(frozen=True)
class CodexHistoryTask:
    ordinal: int
    timestamp: str
    text: str
    text_sha256: str
    task_family: str
    required_sections: tuple[str, ...]
    partition: str
    source_role: str = "user"


@dataclass(frozen=True)
class CodexHistoryContract:
    path: Path
    schema_version: int
    thread_id: str
    source_path: Path
    source_snapshot: dict[str, Any]
    partitions: dict[str, tuple[dict[str, Any], ...]]

    @classmethod
    def from_path(
        cls, path: Path, *, sessions_root: Path | None = None
    ) -> CodexHistoryContract:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise CodexHistoryError("unsupported history contract schema")
        partitions = payload.get("partitions")
        if not isinstance(partitions, dict):
            raise CodexHistoryError("history partitions must be a mapping")
        root = (sessions_root or _default_sessions_root()).resolve()
        configured_source = Path(payload["source_path"])
        source_path = (
            configured_source.resolve()
            if configured_source.is_absolute()
            else (root / configured_source).resolve()
        )
        if source_path.suffix != ".jsonl" or not source_path.is_relative_to(root):
            raise CodexHistoryError(
                f"history source is outside Codex sessions: {source_path}"
            )
        return cls(
            path=path.resolve(),
            schema_version=1,
            thread_id=str(payload["thread_id"]),
            source_path=source_path,
            source_snapshot=dict(payload.get("source_snapshot", {})),
            partitions={
                name: tuple(dict(item) for item in items)
                for name, items in partitions.items()
            },
        )

    def _read_selected_user_messages(
        self, ordinals: set[int]
    ) -> tuple[dict[int, tuple[str, str]], str | None]:
        if not self.source_path.is_file():
            raise CodexHistoryError(f"history source missing: {self.source_path}")
        selected: dict[int, tuple[str, str]] = {}
        thread_id: str | None = None
        user_ordinal = -1
        with self.source_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                payload = row.get("payload", {})
                if row.get("type") == "session_meta" and thread_id is None:
                    thread_id = payload.get("id")
                if not (
                    row.get("type") == "response_item"
                    and payload.get("type") == "message"
                    and payload.get("role") == "user"
                ):
                    continue
                user_ordinal += 1
                if user_ordinal not in ordinals:
                    continue
                text = "\n".join(
                    item.get("text", "")
                    for item in payload.get("content", [])
                    if item.get("type") == "input_text"
                ).strip()
                selected[user_ordinal] = (str(row.get("timestamp", "")), text)
                if len(selected) == len(ordinals):
                    break
        return selected, thread_id

    def load_partition(self, partition: str) -> tuple[CodexHistoryTask, ...]:
        if partition not in self.partitions:
            raise CodexHistoryError(f"unknown history partition: {partition}")
        specs = self.partitions[partition]
        ordinals = {int(item["ordinal"]) for item in specs}
        if len(ordinals) != len(specs):
            raise CodexHistoryError(f"duplicate ordinals in partition: {partition}")
        selected, actual_thread_id = self._read_selected_user_messages(ordinals)
        if actual_thread_id != self.thread_id:
            raise CodexHistoryError(
                f"thread id mismatch: expected {self.thread_id}, got {actual_thread_id}"
            )
        tasks = []
        for spec in specs:
            ordinal = int(spec["ordinal"])
            if ordinal not in selected:
                raise CodexHistoryError(f"selected user message missing: {ordinal}")
            timestamp, text = selected[ordinal]
            actual_sha256 = _sha256_text(text)
            expected_sha256 = str(spec["sha256"])
            if actual_sha256 != expected_sha256:
                raise CodexHistoryError(
                    "message hash mismatch "
                    f"for ordinal {ordinal}: expected {expected_sha256}, got {actual_sha256}"
                )
            tasks.append(
                CodexHistoryTask(
                    ordinal=ordinal,
                    timestamp=timestamp,
                    text=text,
                    text_sha256=actual_sha256,
                    task_family=str(spec["task_family"]),
                    required_sections=tuple(str(x) for x in spec["required_sections"]),
                    partition=partition,
                )
            )
        return tuple(tasks)


@dataclass(frozen=True)
class CodexRuntimeIdentity:
    binary_path: str
    cli_version: str
    execution_mode: str = "offline_history_replay"

    @classmethod
    def for_source_snapshot(
        cls, source_snapshot: dict[str, Any]
    ) -> CodexRuntimeIdentity:
        if source_snapshot.get("synthetic") is True:
            return cls(
                binary_path="synthetic-fixture",
                cli_version="synthetic-fixture",
                execution_mode="synthetic_offline_history_replay",
            )
        return cls.discover()

    @classmethod
    def discover(cls) -> CodexRuntimeIdentity:
        binary = shutil.which("codex")
        if binary is None:
            raise CodexProfileError("local codex binary is unavailable")
        completed = subprocess.run(
            [binary, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return cls(binary_path=str(Path(binary)), cli_version=completed.stdout.strip())


_CAPABILITY_MARKER = re.compile(r"<!--\s*evolve-capabilities:\s*(\[[^\n]*\])\s*-->")


@dataclass(frozen=True)
class CodexProjectProfile:
    root: Path
    capabilities: frozenset[str]
    policy: dict[str, Any]
    tree_hash: str

    @classmethod
    def from_path(cls, root: Path) -> CodexProjectProfile:
        root = root.resolve()
        agents = root / "AGENTS.md"
        policy_path = root / ".codex/evolution-policy.json"
        if not agents.is_file() or not policy_path.is_file():
            raise CodexProfileError(
                "profile requires AGENTS.md and evolution-policy.json"
            )
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if policy.get("schema_version") != 1:
            raise CodexProfileError("unsupported evolution policy schema")
        if policy.get("auto_apply") is not False:
            raise CodexProfileError("profile must keep auto_apply false")
        capabilities: set[str] = set()
        surfaces = [agents]
        skill_root = root / ".agents/skills"
        if skill_root.is_dir():
            surfaces.extend(sorted(skill_root.glob("*/SKILL.md")))
        for surface in surfaces:
            for match in _CAPABILITY_MARKER.finditer(
                surface.read_text(encoding="utf-8")
            ):
                values = json.loads(match.group(1))
                if not isinstance(values, list) or not all(
                    isinstance(value, str) for value in values
                ):
                    raise CodexProfileError(f"invalid capability marker: {surface}")
                capabilities.update(values)
        policy_capabilities = policy.get("capabilities", [])
        if not isinstance(policy_capabilities, list) or not all(
            isinstance(value, str) for value in policy_capabilities
        ):
            raise CodexProfileError("policy capabilities must be a string list")
        capabilities.update(policy_capabilities)
        return cls(
            root=root,
            capabilities=frozenset(capabilities),
            policy=policy,
            tree_hash=_tree_hash(root),
        )


_SECTION_CAPABILITY = {
    "outcome": "outcome_contract",
    "evidence": "outcome_contract",
    "limitations": "outcome_contract",
    "next_action": "outcome_contract",
    "report_path": "outcome_contract",
    "decision": "outcome_contract",
    "observed_pattern": "causal_change",
    "causal_boundary": "causal_change",
    "surface_mapping": "causal_change",
    "verification": "causal_change",
    "failure_hypothesis": "causal_change",
    "changeset": "causal_change",
    "dataset_provenance": "causal_change",
    "partition": "causal_change",
    "current_level": "causal_change",
    "target_level": "causal_change",
    "missing_capability": "causal_change",
    "validation_boundary": "causal_change",
    "entrypoint": "operation_contract",
    "process": "operation_contract",
    "artifacts": "operation_contract",
    "decision_gate": "operation_contract",
    "commands": "operation_contract",
    "failure_modes": "operation_contract",
    "complexity_cause": "complexity_control",
    "minimal_scope": "complexity_control",
    "removal_list": "complexity_control",
    "acceptance": "complexity_control",
    "module_boundaries": "complexity_control",
    "interfaces": "complexity_control",
    "migration": "complexity_control",
    "visible_effect": "complexity_control",
    "milestones": "complexity_control",
    "observer_boundary": "plugin_governance",
    "plugin_contract": "plugin_governance",
    "default_state": "plugin_governance",
    "integration_test": "plugin_governance",
    "rollback": "rollback_safety",
    "comparators": "research_transfer",
    "differences": "research_transfer",
    "transfer_limit": "research_transfer",
    "control_definition": "coevolution_safety",
    "coevolution_definition": "coevolution_safety",
    "evaluator_risk": "coevolution_safety",
    "supervisor_role": "supervisor_governance",
    "authority_boundary": "supervisor_governance",
    "feedback_artifact": "supervisor_governance",
    "admission_gate": "supervisor_governance",
    "provider_interface": "supervisor_governance",
    "trust_boundary": "supervisor_governance",
    "cost_policy": "supervisor_governance",
    "components": "structured_system_delivery",
    "data_flow": "structured_system_delivery",
    "process_evidence": "structured_system_delivery",
    "convergence": "structured_system_delivery",
    "failure_cause": "structured_system_delivery",
}


@dataclass(frozen=True)
class CodexDelivery:
    task_ordinal: int
    requested_sections: tuple[str, ...]
    delivered_sections: tuple[str, ...]

    @property
    def score(self) -> float:
        if not self.requested_sections:
            return 1.0
        return len(self.delivered_sections) / len(self.requested_sections)


class CodexTargetAgentAdapter:
    """Compile project-local Codex surfaces into an offline delivery contract."""

    def __init__(
        self, profile: CodexProjectProfile, identity: CodexRuntimeIdentity
    ) -> None:
        self.profile = profile
        self.identity = identity

    @classmethod
    def from_profile(
        cls, root: Path, *, identity: CodexRuntimeIdentity | None = None
    ) -> CodexTargetAgentAdapter:
        return cls(
            CodexProjectProfile.from_path(root),
            identity or CodexRuntimeIdentity.discover(),
        )

    def execute(self, task: CodexHistoryTask) -> CodexDelivery:
        delivered = tuple(
            section
            for section in task.required_sections
            if _SECTION_CAPABILITY.get(section) in self.profile.capabilities
        )
        return CodexDelivery(
            task_ordinal=task.ordinal,
            requested_sections=task.required_sections,
            delivered_sections=delivered,
        )


def evaluate_profile(
    adapter: CodexTargetAgentAdapter, tasks: tuple[CodexHistoryTask, ...]
) -> dict[str, Any]:
    deliveries = [adapter.execute(task) for task in tasks]
    scores = [delivery.score for delivery in deliveries]
    stable = {
        "profile_tree_hash": adapter.profile.tree_hash,
        "task_hashes": [task.text_sha256 for task in tasks],
        "scores": scores,
    }
    return {
        "profile_tree_hash": adapter.profile.tree_hash,
        "mean_score": mean(scores) if scores else 0.0,
        "tasks_total": len(tasks),
        "tasks_nonzero": sum(score > 0 for score in scores),
        "tasks_full": sum(score == 1 for score in scores),
        "model_calls": 0,
        "global_writes": 0,
        "task_results": [
            {
                "ordinal": task.ordinal,
                "task_family": task.task_family,
                "score": delivery.score,
                "requested_sections": list(delivery.requested_sections),
                "delivered_sections": list(delivery.delivered_sections),
            }
            for task, delivery in zip(tasks, deliveries, strict=True)
        ],
        "evaluation_fingerprint": _sha256_text(_canonical_json(stable)),
    }
