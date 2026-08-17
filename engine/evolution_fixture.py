"""Run a deterministic, zero-network four-generation evolution smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from candidate_tournament import ArmEvaluation
from evolution_controller import EvolutionAuthorization, EvolutionPlan
from evolution_runtime import ExecutionArtifact, ExecutionRequest, run_evolution_search
from mutation_proposer import MutationProposer
from pattern_miner import FrozenObservationEvidence

ORIGINAL = "a" * 64
SEED_PARENT = "b" * 64
EPOCH = "native-adapters-v2.1.0-frozen"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class DeterministicEvolutionAdapters:
    """Exercise every engine boundary without consuming real model calls."""

    real_codex_calls = False

    def __init__(self):
        self.proposer = MutationProposer()

    def execute(self, request: ExecutionRequest) -> ExecutionArtifact:
        if request.role == "original":
            score = 0.5
            feature = "stable_shadow"
            delta = 0.0
        elif request.role == "parent":
            score = 0.6
            feature = "tests_before_edit"
            delta = 0.1
        else:
            ordinal = request.candidate_ordinal or 0
            score = 0.9 - ordinal * 0.1
            feature = "tests_before_edit" if ordinal < 2 else "broad_rewrite"
            delta = -0.1 if ordinal == 3 else score - 0.6
        evidence_id = (
            f"g{request.generation}-{request.stage}-{request.task_uid}-"
            f"{request.arm_sha256[:8]}"
        )
        arm = ArmEvaluation.from_dict(
            {
                "schema_version": 1,
                "task_uid": request.task_uid,
                "benchmark_family": "fixture-family",
                "role": request.role,
                "agent_program_sha256": request.arm_sha256,
                "matched_contract_sha256": _sha(request.task_uid),
                "native_evaluator_epoch": EPOCH,
                "native_score": score,
                "safety_passed": True,
                "cost_units": 1.0,
                "evidence_sha256": _sha(evidence_id),
            }
        )
        observation = None
        if request.role == "candidate" or (
            request.generation == 0 and request.role == "parent"
        ):
            parent_hash = (
                request.original_sha256
                if request.generation == 0
                else request.parent_sha256
            )
            observation = FrozenObservationEvidence.from_dict(
                {
                    "schema_version": 1,
                    "evidence_id": evidence_id,
                    "task_uid": request.task_uid,
                    "benchmark_family": "fixture-family",
                    "agent_program_sha256": request.arm_sha256,
                    "parent_agent_program_sha256": parent_hash,
                    "native_evaluator_epoch": EPOCH,
                    "native_score_delta": delta,
                    "safety_passed": True,
                    "observed_features": [feature],
                    "conditions": ["python", request.stage],
                    "expected_surfaces": [
                        "prompt",
                        "skills",
                        "policy",
                        "constrained_harness_code",
                    ],
                    "evidence": {
                        name: {
                            "path": f"raw/{evidence_id}/{name}.json",
                            "sha256": _sha(f"{evidence_id}-{name}"),
                        }
                        for name in (
                            "trajectory",
                            "tool_events",
                            "native_evaluator",
                            "cost",
                            "safety",
                        )
                    },
                    "causal_boundary": "observational_not_causal",
                    "admission_gate_allowed": False,
                }
            )
        return ExecutionArtifact(arm=arm, observation=observation)

    def propose(self, request, generation: int):
        candidate_hash = _sha(f"g{generation}-{request.request_id}")
        path = {
            "prompt": "AGENTS.md",
            "skills": f".agents/skills/g{generation}/SKILL.md",
            "policy": ".codex/evolution-policy.json",
            "router": ".codex/evolution-policy.json",
            "memory_policy": ".codex/evolution-policy.json",
            "constrained_harness_code": ".codex/harness/runner.py",
        }[request.surface]
        response = {
            "schema_version": 1,
            "changeset_id": f"g{generation}-{request.request_id}",
            "status": "inactive",
            "parent_agent_program_sha256": request.parent_agent_program_sha256,
            "candidate_agent_program_sha256": candidate_hash,
            "hypothesis_ids": list(request.hypothesis_ids),
            "surface": request.surface,
            "operations": [{"op": "replace", "path": path, "after": "new"}],
            "rollback_operations": [{"op": "replace", "path": path, "after": "old"}],
            "proposer": {"platform": "fixture", "model": "deterministic"},
            "native_evaluator_epoch": EPOCH,
            "native_evaluator_authority": "external_fixed",
            "auto_apply": False,
            "production_promotion_allowed": False,
        }
        return self.proposer.propose(
            request=request,
            provider={"platform": "fixture", "model": "deterministic"},
            propose=lambda _prompt: json.dumps(response),
            repair=None,
        )

    def rollback(self, changeset):
        return {
            "forward_patch_sha256": _sha(changeset.changeset_id + "-forward"),
            "rollback_patch_sha256": _sha(changeset.changeset_id + "-rollback"),
            "verified": True,
        }


def run_fixture(output_dir: Path):
    tasks = tuple(f"fixture-task-{index:03d}" for index in range(100))
    return run_evolution_search(
        output_dir=output_dir,
        plan=EvolutionPlan.build(tasks),
        authorization=EvolutionAuthorization(
            maximum_unique_search_tasks=100,
            maximum_real_codex_calls=2000,
            maximum_temporary_cloud_instances=1,
            maximum_elapsed_hours=24.0,
            maximum_cloud_cost_cny=30.0,
        ),
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=SEED_PARENT,
        native_evaluator_epoch=EPOCH,
        adapters=DeterministicEvolutionAdapters(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_fixture(args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "experiment_fingerprint": result["experiment_fingerprint"],
                "real_codex_calls": result["usage"]["real_codex_calls"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
