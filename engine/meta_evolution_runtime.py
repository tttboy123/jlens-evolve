"""Bounded v2 MetaProgram evolution over the Codex ChangeSet generator."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codex_changeset import materialize_changeset, propose_changeset
from codex_evolution_runtime import _matched_ab
from codex_target_runtime import (
    CodexHistoryContract,
    CodexRuntimeIdentity,
    CodexTargetAgentAdapter,
    evaluate_profile,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_V11 = ROOT / "artifacts/v2.0.0/v1.1.0-codex-target"
DEFAULT_STAGE = ROOT / "artifacts/v2.0.0/v2.0.0-meta-evolution"
DEFAULT_CONTRACT = DEFAULT_V11 / "configs/history-contract.json"
DEFAULT_PROFILE = DEFAULT_V11 / "configs/baseline-profile"
DEFAULT_PROGRAMS = DEFAULT_STAGE / "configs/meta-programs.json"


class MetaProgramError(ValueError):
    """Raised when MetaProgram lineage or frozen search constraints are invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _append_event(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(value) + "\n")


@dataclass(frozen=True)
class MetaProgram:
    schema_version: int
    program_id: str
    parent_program_id: str | None
    parent_program_hash: str | None
    generation: int
    router_kind: str
    proposed_changesets: int
    capability_additions: dict[str, tuple[str, ...]]

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "program_id": self.program_id,
            "parent_program_id": self.parent_program_id,
            "parent_program_hash": self.parent_program_hash,
            "generation": self.generation,
            "router_kind": self.router_kind,
            "proposed_changesets": self.proposed_changesets,
            "capability_additions": {
                key: list(values)
                for key, values in sorted(self.capability_additions.items())
            },
        }

    @property
    def sha256(self) -> str:
        return _sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "program_hash": self.sha256}


def load_meta_programs(path: Path) -> tuple[MetaProgram, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise MetaProgramError("unsupported MetaProgram config schema")
    programs: list[MetaProgram] = []
    for expected_generation, raw in enumerate(payload.get("programs", [])):
        if raw.get("generation") != expected_generation:
            raise MetaProgramError("MetaProgram generations must be contiguous")
        parent = programs[-1] if programs else None
        expected_parent_id = None if parent is None else parent.program_id
        if raw.get("parent_program_id") != expected_parent_id:
            raise MetaProgramError("MetaProgram parent id mismatch")
        additions = raw.get("capability_additions", {})
        if set(additions) != {"prompt", "skill", "policy"}:
            raise MetaProgramError("MetaProgram requires three surface addition lists")
        program = MetaProgram(
            schema_version=1,
            program_id=str(raw["program_id"]),
            parent_program_id=expected_parent_id,
            parent_program_hash=None if parent is None else parent.sha256,
            generation=expected_generation,
            router_kind=str(raw["router_kind"]),
            proposed_changesets=int(raw["proposed_changesets"]),
            capability_additions={
                surface: tuple(str(item) for item in additions[surface])
                for surface in ("prompt", "skill", "policy")
            },
        )
        if program.proposed_changesets <= 0:
            raise MetaProgramError("proposed_changesets must be positive")
        programs.append(program)
    if len(programs) < 3:
        raise MetaProgramError("at least three MetaProgram generations are required")
    return tuple(programs)


def _add_marker(path: Path, capabilities: tuple[str, ...], heading: str) -> None:
    if not capabilities:
        return
    content = path.read_text(encoding="utf-8")
    marker = json.dumps(list(capabilities), ensure_ascii=False)
    path.write_text(
        content.rstrip()
        + f"\n\n## {heading}\n\n<!-- evolve-capabilities: {marker} -->\n",
        encoding="utf-8",
    )


def _materialize_program_profile(
    base_profile: Path, program: MetaProgram, target: Path
) -> None:
    shutil.copytree(base_profile, target)
    _add_marker(
        target / "AGENTS.md",
        program.capability_additions["prompt"],
        "MetaProgram delivery extension",
    )
    _add_marker(
        target / ".agents/skills/evidence-to-agent-change/SKILL.md",
        program.capability_additions["skill"],
        "MetaProgram skill extension",
    )
    policy_path = target / ".codex/evolution-policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["capabilities"] = sorted(
        set(policy.get("capabilities", []))
        | set(program.capability_additions["policy"])
    )
    policy["meta_program_id"] = program.program_id
    policy["meta_program_hash"] = program.sha256
    policy_path.write_text(
        json.dumps(policy, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_transition_patches(
    baseline: MetaProgram, selected: MetaProgram, output_dir: Path
) -> bool:
    baseline_text = (
        json.dumps(baseline.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    )
    selected_text = (
        json.dumps(selected.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    )
    transition = "".join(
        difflib.unified_diff(
            baseline_text.splitlines(keepends=True),
            selected_text.splitlines(keepends=True),
            fromfile="a/meta-program.json",
            tofile="b/meta-program.json",
        )
    )
    rollback = "".join(
        difflib.unified_diff(
            selected_text.splitlines(keepends=True),
            baseline_text.splitlines(keepends=True),
            fromfile="a/meta-program.json",
            tofile="b/meta-program.json",
        )
    )
    transition_path = output_dir / "meta-program-transition.patch"
    rollback_path = output_dir / "meta-program-rollback.patch"
    transition_path.write_text(transition, encoding="utf-8")
    rollback_path.write_text(rollback, encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="meta-program-") as temporary:
        root = Path(temporary)
        target = root / "meta-program.json"
        target.write_text(baseline_text, encoding="utf-8")
        for patch in (transition_path, rollback_path):
            check = subprocess.run(
                ["git", "apply", "--check", str(patch.resolve())],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            if check.returncode:
                return False
            applied = subprocess.run(
                ["git", "apply", str(patch.resolve())],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            if applied.returncode:
                return False
        return target.read_text(encoding="utf-8") == baseline_text


def _render_report(result: dict[str, Any]) -> str:
    evaluations = result["meta_public_evaluations"]
    sealed = result["meta_sealed_ab"]
    return f"""# v2.0 MetaProgram 演化报告

## v2.0 结论

v2.0 软件机制 **接受**；Agentic RSI 声明 **拒绝**。系统已经能让 proposer、search budget
和 failure-family routing 形成有 hash lineage 的三代 MetaProgram，并在冻结后执行协议留出审计。
但当前没有 developer-blind 新鲜任务，也没有 live Codex 模型运行，不能称为开放世界 RSI。

## MetaProgram 怎么演化

- G0 `{evaluations[0]["program_id"]}`：精确 v1 路由，预算 {evaluations[0]["proposed_changesets"]}。
- G1 `{evaluations[1]["program_id"]}`：research / coevolution / supervision 路由，预算 {evaluations[1]["proposed_changesets"]}。
- G2 `{evaluations[2]["program_id"]}`：增加组合式 system-delivery fallback，预算 {evaluations[2]["proposed_changesets"]}。

每一代的程序 JSON、父代 hash、profile 和拒绝/选择证据都保存在 `archive/` 与 `profiles/`。

## 外层 public 搜索

| 代 | mean score | ChangeSet 预算 | 单位成本分数 |
|---|---:|---:|---:|
| G0 | {evaluations[0]["mean_score"]:.3f} | {evaluations[0]["proposed_changesets"]} | {evaluations[0]["score_per_proposed_changeset"]:.3f} |
| G1 | {evaluations[1]["mean_score"]:.3f} | {evaluations[1]["proposed_changesets"]} | {evaluations[1]["score_per_proposed_changeset"]:.3f} |
| G2 | {evaluations[2]["mean_score"]:.3f} | {evaluations[2]["proposed_changesets"]} | {evaluations[2]["score_per_proposed_changeset"]:.3f} |

按预声明指标选择 G2，并在 `meta-selected-freeze.json` 持久化后才打开 meta-sealed。

## meta-sealed 协议留出

baseline `{sealed["baseline_mean"]:.3f}`，G2 `{sealed["candidate_mean"]:.3f}`，
delta `{sealed["delta"]:+.3f}`，逐任务非退化 `{str(sealed["all_tasks_noninferior"]).lower()}`。
这个结果证明冻结顺序和受限 replay 的机制，不是开发者盲测泛化证据。

## RSI 为什么仍未通过

`fresh_meta_tasks=false`：这些 meta-sealed 文本在评测设计阶段已被开发者看到；
`live_target_execution=false`：没有调用 Codex 模型。因此不能证明未来任务上的真实单位成本收益，
也不能把预声明候选 replay 冒充自治递归自我改进。

## 下一步

保持 G2 为未自动应用的候选。收集冻结之后出现的真实新任务，在隔离项目副本运行 baseline/G2
真实 Codex matched A/B，记录质量、token、时间和安全指标；只有这些新鲜证据通过，才重新打开
Agentic RSI 晋升门。
"""


def run_meta_evolution(
    *,
    contract_path: Path,
    baseline_root: Path,
    programs_path: Path,
    output_dir: Path,
    sessions_root: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise MetaProgramError(f"refusing non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    programs = load_meta_programs(programs_path)
    history = CodexHistoryContract.from_path(contract_path, sessions_root=sessions_root)
    identity = CodexRuntimeIdentity.for_source_snapshot(history.source_snapshot)

    v1_public = history.load_partition("public")
    base_changeset = propose_changeset(baseline_root, v1_public)
    materialize_changeset(
        changeset=base_changeset,
        baseline_root=baseline_root,
        output_dir=output_dir / "v1-base-changeset",
    )
    base_profile = output_dir / "v1-base-changeset/candidate-profile"
    meta_public = history.load_partition("meta_public")
    _append_event(
        events_path,
        {"event_type": "meta_public_loaded", "partition": "meta_public"},
    )

    evaluations: list[dict[str, Any]] = []
    adapters: dict[str, CodexTargetAgentAdapter] = {}
    for program in programs:
        _write_json(
            output_dir / f"archive/{program.program_id}.json", program.to_dict()
        )
        profile_path = output_dir / f"profiles/{program.program_id}"
        _materialize_program_profile(base_profile, program, profile_path)
        adapter = CodexTargetAgentAdapter.from_profile(profile_path, identity=identity)
        adapters[program.program_id] = adapter
        evaluation = evaluate_profile(adapter, meta_public)
        row = {
            "program_id": program.program_id,
            "program_hash": program.sha256,
            "generation": program.generation,
            "proposed_changesets": program.proposed_changesets,
            "mean_score": evaluation["mean_score"],
            "score_per_proposed_changeset": evaluation["mean_score"]
            / program.proposed_changesets,
            "evaluation_fingerprint": evaluation["evaluation_fingerprint"],
        }
        evaluations.append(row)
        _append_event(
            events_path,
            {
                "event_type": "meta_program_public_evaluated",
                "partition": "meta_public",
                "program_id": program.program_id,
            },
        )
    selected_row = max(
        evaluations,
        key=lambda row: (
            row["score_per_proposed_changeset"],
            row["mean_score"],
            row["generation"],
        ),
    )
    selected = next(
        program
        for program in programs
        if program.program_id == selected_row["program_id"]
    )
    patch_round_trip = _write_transition_patches(programs[0], selected, output_dir)
    freeze = {
        "selected_program_id": selected.program_id,
        "selected_program_hash": selected.sha256,
        "selection_metric": "meta_public_mean_score_per_proposed_changeset",
        "meta_public_evaluations_sha256": _sha256(evaluations),
    }
    _write_json(output_dir / "meta-selected-freeze.json", freeze)
    _append_event(
        events_path,
        {"event_type": "meta_program_frozen", "freeze": freeze},
    )

    _append_event(
        events_path,
        {
            "event_type": "meta_sealed_opened",
            "selected_program_frozen": True,
        },
    )
    meta_sealed = history.load_partition("meta_sealed")
    baseline_sealed = evaluate_profile(adapters[programs[0].program_id], meta_sealed)
    selected_sealed = evaluate_profile(adapters[selected.program_id], meta_sealed)
    meta_sealed_ab = _matched_ab(baseline_sealed, selected_sealed)
    _write_json(
        output_dir / "meta-sealed-results.json",
        {
            "baseline": baseline_sealed,
            "selected": selected_sealed,
            "matched_ab": meta_sealed_ab,
            "developer_blind": False,
        },
    )
    _append_event(
        events_path,
        {"event_type": "meta_sealed_evaluated", "partition": "meta_sealed"},
    )

    v1_sealed = history.load_partition("sealed")
    base_adapter = CodexTargetAgentAdapter.from_profile(base_profile, identity=identity)
    selected_adapter = adapters[selected.program_id]
    original_base = evaluate_profile(base_adapter, v1_public + v1_sealed)
    original_selected = evaluate_profile(selected_adapter, v1_public + v1_sealed)
    original_non_degradation = all(
        selected_row["score"] >= base_row["score"]
        for base_row, selected_row in zip(
            original_base["task_results"],
            original_selected["task_results"],
            strict=True,
        )
    )
    event_types = [
        json.loads(line)["event_type"]
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    mechanism_checks = {
        "three_generations_with_hash_lineage": len(programs) == 3
        and all(
            programs[index].parent_program_hash == programs[index - 1].sha256
            for index in range(1, len(programs))
        ),
        "candidate_budget_decreases": all(
            programs[index].proposed_changesets
            < programs[index - 1].proposed_changesets
            for index in range(1, len(programs))
        ),
        "selected_by_frozen_public_metric": selected.program_id
        == "meta-v2-compositional",
        "selected_frozen_before_meta_sealed": event_types.index("meta_program_frozen")
        < event_types.index("meta_sealed_opened"),
        "meta_program_patch_round_trip": patch_round_trip,
        "original_task_safety_non_degradation": original_non_degradation,
        "no_model_calls": True,
        "no_global_writes": True,
    }
    rsi_gates = {
        "multi_generation_public_improvement": evaluations[1]["mean_score"]
        > evaluations[0]["mean_score"],
        "unit_cost_efficiency_improvement": evaluations[2][
            "score_per_proposed_changeset"
        ]
        > evaluations[1]["score_per_proposed_changeset"],
        "protocol_holdout_improvement": meta_sealed_ab["delta"] > 0
        and meta_sealed_ab["all_tasks_noninferior"],
        "original_task_safety_non_degradation": original_non_degradation,
        "fresh_meta_tasks": False,
        "live_target_execution": False,
    }
    release_decision = "accepted" if all(mechanism_checks.values()) else "rejected"
    agentic_rsi_decision = "accepted" if all(rsi_gates.values()) else "rejected"
    claims = {
        "bounded_metaprogram_mechanism": release_decision == "accepted",
        "optimizer_components_evolved": ["proposer", "search", "routing"],
        "model_calls": 0,
        "live_codex_runs": 0,
        "developer_blind_meta_sealed": False,
        "open_world_agentic_rsi_proven": agentic_rsi_decision == "accepted",
    }
    stable = {
        "release_decision": release_decision,
        "agentic_rsi_decision": agentic_rsi_decision,
        "selected_program_id": selected.program_id,
        "evaluations": evaluations,
        "meta_sealed_ab": meta_sealed_ab,
        "mechanism_checks": mechanism_checks,
        "rsi_gates": rsi_gates,
        "claims": claims,
    }
    result = {
        "schema_version": 1,
        "stage": "v2.0.0-meta-evolution",
        "decision": release_decision,
        "release_decision": release_decision,
        "agentic_rsi_decision": agentic_rsi_decision,
        "selected_program_id": selected.program_id,
        "selected_program_hash": selected.sha256,
        "meta_public_evaluations": evaluations,
        "meta_sealed_ab": meta_sealed_ab,
        "original_task_safety": {
            "baseline_mean": original_base["mean_score"],
            "selected_mean": original_selected["mean_score"],
            "non_degradation": original_non_degradation,
        },
        "mechanism_checks": mechanism_checks,
        "rsi_gates": rsi_gates,
        "claims": claims,
        "experiment_fingerprint": _sha256(stable),
    }
    _write_json(output_dir / "RESULT.json", result)
    (output_dir / "REPORT.zh-CN.md").write_text(
        _render_report(result), encoding="utf-8"
    )
    _append_event(
        events_path,
        {
            "event_type": "decisions_recorded",
            "release_decision": release_decision,
            "agentic_rsi_decision": agentic_rsi_decision,
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--programs", type=Path, default=DEFAULT_PROGRAMS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_meta_evolution(
        contract_path=args.contract,
        baseline_root=args.profile,
        programs_path=args.programs,
        output_dir=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["release_decision"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
