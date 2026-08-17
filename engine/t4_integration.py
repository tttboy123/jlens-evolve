"""T4 integration: local-lens run artifact -> candidate skill -> transfer gate -> ladder.

Orchestrates the v2.5 Skills-mode landing path:
  1. Build an observational PatternCard from the run artifact (layer_records + trace).
  2. Compile an inactive project-local SkillCandidate via search_skill_bridge.
  3. If fresh-task paired evidence is provided, run the cross-task transfer gate
     (>=8 pairs, native non-inferior, no safety regression, cost<=10%) and apply.
  4. Record candidate/reviewed status in the append-only PromotionLadder.

CLI:
  python t4_integration.py --artifact <RUN-ARTIFACT.json> \
      --registry-root <dir> --skill-id <id> --source-task-id <id> \
      --instruction "<text>" [--paired-evals <json>] [--out <dir>]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pattern_card import build_pattern_card
from promotion_ladder import PromotionLadder, ReviewDecision
from search_skill_bridge import (
    SearchSkillBridgeError,
    apply_transfer_gate,
    compile_local_lens_candidate,
    evaluate_transfer_gate,
)
from skill_registry import SkillRegistry


@dataclass(frozen=True)
class IntegrationResult:
    pattern_card: dict[str, Any]
    candidate_skill_id: str
    candidate_revision_id: str
    gate: dict[str, Any]
    ladder_status: str


def integrate(
    *,
    run_artifact: Path,
    registry_root: Path,
    skill_id: str,
    source_task_id: str,
    instruction: str,
    paired_evals: list[dict[str, Any]] | None = None,
    expected_contract_sha256: str | None = None,
    expected_evaluator_epoch: str | None = None,
) -> IntegrationResult:
    run_artifact = run_artifact.resolve()
    if not run_artifact.is_file():
        raise SearchSkillBridgeError(f"missing run artifact: {run_artifact}")
    artifact = json.loads(run_artifact.read_text(encoding="utf-8"))
    tool_events = artifact.get("tool_events", [])
    layer_records: list[dict[str, Any]] = [
        record
        for event in tool_events
        for record in (event.get("layer_records", []) or ())
    ]
    # If an older artifact only carries counts (no embedded records), surface it.
    embedded = sum(
        len(event.get("layer_records", []) or ())
        for event in tool_events
        if isinstance(event, dict)
    )
    if embedded == 0:
        raise SearchSkillBridgeError(
            "run artifact has layer_record_count but no embedded layer_records; "
            "regenerate with fixed local_lens_agent (LensStep.to_dict embeds records)"
        )

    pattern_card = build_pattern_card(
        layer_records=layer_records,
        tool_events=tool_events,
        instruction=instruction,
        pattern_id=skill_id,
        evidence_refs=[str(run_artifact)],
    )
    registry_root.mkdir(parents=True, exist_ok=True)
    candidate = compile_local_lens_candidate(
        run_artifact=run_artifact,
        registry_root=registry_root,
        skill_id=skill_id,
        source_task_id=source_task_id,
        reusable_instruction=instruction,
    )

    gate = evaluate_transfer_gate(
        paired_evals=paired_evals or [],
        expected_contract_sha256=expected_contract_sha256,
        expected_evaluator_epoch=expected_evaluator_epoch,
    )
    if paired_evals:
        gate_path = registry_root / "gate-evidence.json"
        gate_path.write_text(
            json.dumps(paired_evals, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        candidate = apply_transfer_gate(
            registry=SkillRegistry(registry_root),
            candidate=candidate,
            gate_result=gate,
            gate_evidence_path=gate_path,
        )
    ladder = PromotionLadder(registry_root / "ladder")
    if paired_evals:
        # Only human-style decisions are ladder-valid; automation records
        # reviewed/rejected after a gate run, and never auto-activates.
        decision = ReviewDecision.create(
            skill_id=candidate.skill_id,
            revision_id=candidate.revision_id,
            decision="reviewed" if gate.passed else "rejected",
            reviewer="automation-t4-integration",
            notes=f"gate passed={gate.passed}; paired={gate.paired_tasks}",
        )
        ladder.record(decision)
    return IntegrationResult(
        pattern_card=pattern_card,
        candidate_skill_id=candidate.skill_id,
        candidate_revision_id=candidate.revision_id,
        gate=gate.to_dict(),
        ladder_status=ladder.effective_status(
            candidate.skill_id, candidate.revision_id
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--source-task-id", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--paired-evals", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    paired = None
    if args.paired_evals:
        paired = json.loads(args.paired_evals.read_text(encoding="utf-8"))
    result = integrate(
        run_artifact=args.artifact,
        registry_root=args.registry_root,
        skill_id=args.skill_id,
        source_task_id=args.source_task_id,
        instruction=args.instruction,
        paired_evals=paired,
    )
    summary = {
        "pattern_card": result.pattern_card,
        "candidate_skill_id": result.candidate_skill_id,
        "candidate_revision_id": result.candidate_revision_id,
        "gate": result.gate,
        "ladder_status": result.ladder_status,
    }
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "INTEGRATION.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
