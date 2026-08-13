"""Single v3 product entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from evolve.alignment import align_native_pair
from evolve.contracts import (
    Claim,
    Cohort,
    EvidenceEnvelope,
    Receipt,
    TaskRevision,
    canonical_json,
)
from evolve.evidence import ClaimEngine, EvidenceGraph, ReceiptStore
from evolve.fresh_feedback import run_fresh_feedback_e2e, seal_run
from evolve.observers import NativeOutcomeObserver, ObserverHub
from evolve.registry import CapabilityRecord, CapabilityRegistry
from evolve.reporting import AuditVerifier, CampaignReportProjector


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def _resolved(path: Path, instance_id: str) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "resolved_instances" in data:
        return instance_id in data.get("resolved_ids", [])
    row = data.get(instance_id)
    if not isinstance(row, dict) or not isinstance(row.get("resolved"), bool):
        raise ValueError(f"unsupported native report: {path}")
    return row["resolved"]


def run_legacy_feedback_e2e(
    *,
    root: Path,
    legacy_root: Path,
    output: Path,
    teacher_receipt: Path,
    qwen_receipt: Path,
) -> dict[str, object]:
    """Import three real feedback pairs through the v3 Evidence/Claim path."""

    output.mkdir(parents=True, exist_ok=False)
    campaign = "v3-feedback-vertical-slice"
    native_pairs = (
        (
            "sphinx-doc__sphinx-7757",
            "sphinx",
            legacy_root
            / "runs/skill-evolution-loop/autonomous/rounds/20260813T140000Z-sphinx-regression-scan/harness-results/round3-r096-baseline.round11-sphinx-doc__sphinx-7757-baseline-20260813212518.json",
            legacy_root
            / "runs/skill-evolution-loop/autonomous/rounds/20260813T140000Z-sphinx-regression-scan/harness-results/round3-r096-taught.round11-sphinx-doc__sphinx-7757-taught-20260813212334.json",
        ),
        (
            "phpoffice__phpspreadsheet-3463",
            "phpspreadsheet",
            legacy_root
            / ".runtime/swebench-f7bbbb2/logs/run_evaluation/round1-phpoffice__phpspreadsheet-3463-span-baseline-8c349736b075/evolve-span-baseline-a30c7b63a81c/phpoffice__phpspreadsheet-3463/report.json",
            legacy_root
            / ".runtime/swebench-f7bbbb2/logs/run_evaluation/round1-phpoffice__phpspreadsheet-3463-span-taught-776280fa5f1e/evolve-span-taught-afcda7e19ee0/phpoffice__phpspreadsheet-3463/report.json",
        ),
        (
            "laravel__framework-52684",
            "laravel",
            legacy_root
            / ".runtime/swebench-f7bbbb2/logs/run_evaluation/round1-laravel__framework-52684-span-baseline-d3d84cefc9cc/evolve-span-baseline-16c47f69b90e/laravel__framework-52684/report.json",
            legacy_root
            / ".runtime/swebench-f7bbbb2/logs/run_evaluation/round1-laravel__framework-52684-span-taught-2d1be75282a9/evolve-span-taught-afcda7e19ee0/laravel__framework-52684/report.json",
        ),
    )
    store = ReceiptStore(output / "receipts")
    graph = EvidenceGraph(output / "evidence-graph")
    hub = ObserverHub((NativeOutcomeObserver(),), graph=graph)
    claims: list[Claim] = []
    receipts: list[Receipt] = []
    tasks: list[dict[str, object]] = []
    source_paths: list[Path] = [teacher_receipt, qwen_receipt]
    for task_index, (task_id, project, baseline_path, taught_path) in enumerate(
        native_pairs, 1
    ):
        source_paths.extend((baseline_path, taught_path))
        task_source_sha = hashlib.sha256(task_id.encode()).hexdigest()
        task = TaskRevision(
            task_id=task_id,
            revision_id=f"legacy-feedback-{task_index}",
            project=project,
            cohort=Cohort.FEEDBACK,
            source_sha256=task_source_sha,
            evaluator_id="legacy-native@sha256:" + _sha(baseline_path),
            source_uri=str(baseline_path),
        )
        envelopes: list[EvidenceEnvelope] = []
        for arm, path in (("baseline", baseline_path), ("taught", taught_path)):
            artifact = path.read_bytes()
            payload = {
                "arm": arm,
                "task_revision_id": task.revision_id,
                "task_source_sha256": task.source_sha256,
                "model_identity": "local-mlx/Qwen3.5-4B@4bit",
                "native_evaluator_id": "official-legacy-replay-v1",
                "execution_config_sha256": "c" * 64,
                "resolved": _resolved(path, task_id),
                "evaluator_error": None,
                "legacy_source_path": str(path.relative_to(legacy_root)),
                "legacy_source_sha256": _sha(path),
            }
            receipt = Receipt(
                receipt_id=f"receipt-{task_index}-{arm}",
                campaign_id=campaign,
                plan_id=f"plan-{task_index}-{arm}",
                sequence=1,
                kind="native_evaluation",
                created_at="2026-08-14T00:00:00Z",
                payload=payload,
                artifact_sha256=hashlib.sha256(artifact).hexdigest(),
            )
            store.append(receipt, artifact)
            receipts.append(receipt)
            envelopes.append(hub.observe(receipt)[0])
        claim = ClaimEngine(graph).classify_pair(
            "candidate-v3-teacher-001", align_native_pair(*envelopes)
        )
        claims.append(claim)
        tasks.append(
            {
                "task_id": task_id,
                "project": project,
                "cohort": "feedback",
                "classification": str(claim.classification),
            }
        )
    capability = CapabilityRecord(
        capability_id="capability-deterministic-operator",
        revision_id="v3-candidate-001",
        capability_kind="operator-skill",
        evidence_claim_ids=tuple(claim.claim_id for claim in claims),
        artifact_sha256=_sha(teacher_receipt),
        active=False,
    )
    registry = CapabilityRegistry(output / "capability-registry.jsonl")
    registry.append(capability)
    teacher_payload = json.loads(teacher_receipt.read_text(encoding="utf-8"))
    metadata = {
        "outcome": "MVP_closed_loop",
        "task_count": len(tasks),
        "project_count": len({row["project"] for row in tasks}),
        "feedback_tasks": tasks,
        "holdout_opened": False,
        "burned_holdout_opened": False,
        "skill_auto_activated": False,
        "capability_active": False,
        "teacher_receipt_path": str(teacher_receipt),
        "teacher_receipt_sha256": _sha(teacher_receipt),
        "qwen_receipt_path": str(qwen_receipt),
        "qwen_receipt_sha256": _sha(qwen_receipt),
        "actual_api_spend_cny": teacher_payload["estimated_cost_cny"],
        "api_budget_limit_cny": 10.0,
        "api_budget_remaining_cny": round(
            10.0 - float(teacher_payload["estimated_cost_cny"]), 8
        ),
    }
    report = CampaignReportProjector().project(
        campaign_id=campaign,
        receipts=receipts,
        claims=claims,
        final_commit_sha=_git_sha(root),
        metadata=metadata,
    )
    paths = CampaignReportProjector().write(report, output)
    sources = output / "source-evidence"
    sources.mkdir()
    for index, path in enumerate(source_paths, 1):
        shutil.copyfile(path, sources / f"{index:02d}-{path.name}")
    manifest = {"schema_version": 1, "entries": []}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path != paths.manifest_path:
            manifest["entries"].append(
                {"path": str(path.relative_to(output)), "sha256": _sha(path)}
            )
    paths.manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evolve-v3")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("legacy-feedback-e2e")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--legacy-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--teacher-receipt", type=Path, required=True)
    run.add_argument("--qwen-receipt", type=Path, required=True)
    fresh = sub.add_parser("fresh-feedback-e2e")
    fresh.add_argument("--config", type=Path, required=True)
    fresh.add_argument("--output", type=Path, required=True)
    seal = sub.add_parser("seal-run")
    seal.add_argument("--root", type=Path, required=True)
    verify = sub.add_parser("verify-manifest")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "legacy-feedback-e2e":
        report = run_legacy_feedback_e2e(
            root=args.root.resolve(),
            legacy_root=args.legacy_root.resolve(),
            output=args.output.resolve(),
            teacher_receipt=args.teacher_receipt.resolve(),
            qwen_receipt=args.qwen_receipt.resolve(),
        )
        print(canonical_json(report))
        return 0
    if args.command == "fresh-feedback-e2e":
        result = run_fresh_feedback_e2e(
            config_path=args.config.resolve(), output_root=args.output.resolve()
        )
        print(canonical_json(result))
        return 0
    if args.command == "seal-run":
        count = seal_run(args.root.resolve())
        print(f"sealed and verified {count} manifest entries")
        return 0
    count = AuditVerifier().verify_manifest(args.manifest.resolve(), root=args.root)
    print(f"verified {count} manifest entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
