#!/usr/bin/env python3
"""Bootstrap the r073-r078 evolution catalog from immutable evidence."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skill_evolution_loop.evolution_catalog import (
    CatalogConflict,
    EvolutionCatalog,
    EvolutionRecord,
)

ROOT = Path("artifacts/v2.5.0/v2.5.0-local-jlens")
RUNS = ROOT / "runs/skill-evolution-loop"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path: Path) -> dict[str, str]:
    return {"path": path.as_posix(), "sha256": _sha(path)}


def records() -> tuple[EvolutionRecord, ...]:
    r073 = RUNS / "round3-r073-feedback-strategy/RESPONSE.json"
    operator = RUNS / "round3-r074-inactive-skills/OPERATOR-SKILL.json"
    span = RUNS / "round3-r074-inactive-skills/SPAN-SKILL.json"
    feedback = RUNS / "round3-r074-locale-role-feedback-smoke-native/SUMMARY.json"
    holdout_checkpoint = RUNS / "aws-migration-r075/LOCAL-HOLDOUT-CHECKPOINT.json"
    aws_attempt = RUNS / "aws-migration-r075/AWS-DEPLOYMENT-ATTEMPT.json"
    aws_attempt_2 = RUNS / "aws-migration-r075/AWS-DEPLOYMENT-ATTEMPT-002.json"
    blocker_audit = RUNS / "aws-migration-r075/AWS-ACCOUNT-BLOCKER-AUDIT.json"
    r076 = RUNS / "tencent-bounded-holdout-r076"
    r076_report = r076 / "ROUND-REPORT.json"
    r076_report_r002 = r076 / "ROUND-REPORT.r002.json"
    r076_progress = r076 / "experiment/PROGRESS.json"
    r076_checkpoint = r076 / "tencent/checkpoints/20260812T110417Z/CHECKPOINT.json"
    r077 = RUNS / "round3-r077-feedback-only-audit"
    r077_request = r077 / "REQUEST.json"
    r077_audit = r077 / "AUDIT.json"
    r078 = RUNS / "tencent-bounded-holdout-r078"
    r078_report = r078 / "ROUND-REPORT.json"
    r078_holdout = r078 / "experiment/HOLDOUT-SUMMARY.json"
    r078_native = r078 / "native-full-r002/SUMMARY.json"
    r078_residual = r078 / "TENCENT-RESIDUAL-AUDIT.json"
    r078_checkpoint = r078 / "tencent/checkpoints/20260812T120905Z/CHECKPOINT.json"
    r079 = RUNS / "round3-r079-control-plane-hardening/ROUND-REPORT.json"
    r080_root = RUNS / "round3-r080-pattern-card-fpr"
    r080_report = r080_root / "ROUND-REPORT.json"
    r080_calibration = r080_root / "PATTERN-CARD-FPR.refined-r002.json"
    r080_candidate = r080_root / "TRIGGER-REFINEMENT.r002.json"
    r081_root = RUNS / "round3-r081-convergence-safety"
    r081_report = r081_root / "ROUND-REPORT.json"
    r081_convergence = r081_root / "CONVERGENCE-QUALIFICATION.json"
    r081_safety = r081_root / "SAFETY-SUITE-STATUS.json"
    r082_root = RUNS / "round3-r082-executable-control-gates"
    r082_capability = r082_root / "STATISTICAL-CAPABILITY-GATE.json"
    r083_root = RUNS / "tencent-feedback-r083"
    r083_preflight = r083_root / "PREFLIGHT.json"
    r083_closeout = r083_root / "CLOSEOUT.json"
    r083_correction = r083_root / "BOOTSTRAP-RACE-CORRECTION.json"
    r083_correction_r002 = r083_root / "BOOTSTRAP-CAUSAL-CORRECTION.r002.json"
    return (
        EvolutionRecord.create(
            record_type="mechanisms",
            record_id="r073-issue-anchored-candidate-seeding",
            title="Issue-anchored candidate seeding and boundary gating",
            status="implemented",
            capability_tags=("localization", "patch-realization"),
            task_tags=("swe-bench",),
            failure_mode_tags=("wrong-target", "selector-no-match"),
            source_model="deepseek-v4-flash",
            source_runtime="deepseek-api",
            payload={"source_round": 73, "auto_apply": False},
            evidence_refs=(_ref(r073),),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="mechanisms",
            record_id="r083-explicit-sha-bound-accelerator-bootstrap",
            title="Explicit SHA-bound accelerator bootstrap before readiness admission",
            status="implemented",
            capability_tags=("cuda-execution-plane", "cost-guard", "replayability"),
            task_tags=("infrastructure",),
            failure_mode_tags=("missing-driver", "provider-state-drift"),
            source_model="none",
            source_runtime="local-control-plane",
            payload={
                "supersedes_interpretation": "passive-provider-bootstrap-race",
                "validated_interpretation": "control-plane-triggered-sha-bound-installer",
                "bootstrap_trigger": "explicit-installer-or-custom-image",
                "installer_timeout_seconds": 900,
                "paid_model_work_requires": "accelerator-runtime-preflight-v1",
                "provider_special_case": False,
            },
            evidence_refs=(_ref(r083_correction_r002),),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="skills",
            record_id="r074-typed-operator-realization-r010",
            title="Typed operator realization r010",
            status="candidate",
            capability_tags=("patch-realization", "locale-role-substitution"),
            task_tags=("swe-bench",),
            failure_mode_tags=("selector-no-match", "native-unresolved"),
            source_model="deepseek-v4-flash",
            source_runtime="deepseek-api",
            payload={"skill_sha256": _sha(operator), "active": False},
            evidence_refs=(_ref(operator), _ref(feedback)),
            cross_model_validations=(
                {
                    "target_model": "Qwen/Qwen3.5-4B",
                    "target_runtime": "mlx-4bit",
                    "outcome": "validated",
                    "evidence_sha256": _sha(feedback),
                },
                {
                    "target_model": "Qwen/Qwen3.5-4B",
                    "target_runtime": "cuda-vllm",
                    "outcome": "pending",
                    "evidence_sha256": None,
                },
            ),
        ),
        EvolutionRecord.create(
            record_type="skills",
            record_id="r074-exact-span-realization-r011",
            title="Exact-span realization r011",
            status="candidate",
            capability_tags=("patch-realization", "exact-span"),
            task_tags=("swe-bench",),
            failure_mode_tags=("selector-no-match", "native-unresolved"),
            source_model="deepseek-v4-flash",
            source_runtime="deepseek-api",
            payload={"skill_sha256": _sha(span), "active": False},
            evidence_refs=(_ref(span), _ref(feedback)),
            cross_model_validations=(
                {
                    "target_model": "Qwen/Qwen3.5-4B",
                    "target_runtime": "mlx-4bit",
                    "outcome": "validated",
                    "evidence_sha256": _sha(feedback),
                },
                {
                    "target_model": "Qwen/Qwen3.5-4B",
                    "target_runtime": "cuda-vllm",
                    "outcome": "pending",
                    "evidence_sha256": None,
                },
            ),
        ),
        EvolutionRecord.create(
            record_type="experiments",
            record_id="r074-feedback-native-60-cell",
            title="r074 full feedback native A/B",
            status="validated",
            capability_tags=("agent-self-evolution", "patch-realization"),
            task_tags=("swe-bench",),
            failure_mode_tags=("native-unresolved",),
            source_model="Qwen/Qwen3.5-4B",
            source_runtime="mlx-4bit",
            payload={
                "planned_cells": 60,
                "completed_cells": 60,
                "feedback_gain_count": 1,
                "evaluator_failure_count": 0,
                "resolved_to_unresolved_regression_count": 0,
                "embedded_summary_sha256": "0996c9618b8125ff912f5fe015de5d811777a3e4d5921b8c20dc6cf524716082",
            },
            evidence_refs=(_ref(feedback),),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="experiments",
            record_id="r077-feedback-only-next-step-audit",
            title="Feedback-only gain and dedup audit before further evolution",
            status="validated",
            capability_tags=("agent-self-evolution", "evaluation-integrity"),
            task_tags=("swe-bench", "feedback"),
            failure_mode_tags=("native-unresolved", "structural-invalid"),
            source_model="Qwen/Qwen3.5-4B",
            source_runtime="local-control-plane",
            payload={
                "feedback_pairs": 30,
                "strict_gain_count": 1,
                "teaching_structural_degradation_count": 2,
                "parent_call_recommended": False,
                "new_skill_compilation_recommended": False,
                "next_step": "holdout-safety-evaluation",
                "holdout_cells_included": False,
            },
            evidence_refs=(_ref(r077_request), _ref(r077_audit)),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="experiments",
            record_id="r078-tencent-cuda-full-holdout-safety",
            title="Complete 30-pair holdout safety evaluation on Tencent CUDA",
            status="validated",
            capability_tags=("agent-self-evolution", "evaluation-integrity"),
            task_tags=("swe-bench", "holdout"),
            failure_mode_tags=("structural-invalid", "native-unresolved"),
            source_model="Qwen/Qwen3.5-4B",
            source_runtime="tencent-t4-vllm-awq",
            payload={
                "completed_cells": 60,
                "complete_pairs": 30,
                "feedback_gain_count": 1,
                "structural_valid_cells": 2,
                "holdout_regression_count": 0,
                "native_evaluator_failure_count": 0,
                "holdout_safety_qualified_pair_count": 29,
                "holdout_evaluable_pair_count": 0,
                "capability_gate_passed": True,
                "capability_claim": "safety-only; no cross-task gain claimed",
                "deepseek_calls": 0,
            },
            evidence_refs=(
                _ref(r078_report),
                _ref(r078_holdout),
                _ref(r078_native),
                _ref(r078_checkpoint),
            ),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="mechanisms",
            record_id="r078-explicit-shard-native-evaluation",
            title="Explicit task shards for append-only native holdout evaluation",
            status="implemented",
            capability_tags=("evaluation-integrity", "native-validation"),
            task_tags=("swe-bench", "holdout", "infrastructure"),
            failure_mode_tags=("eval-infra", "partial-run"),
            source_model="none",
            source_runtime="local-control-plane",
            payload={
                "interface": "round1-holdout-native --task",
                "partial_scope": "round1-holdout-shard",
                "full_scope_preserved": "round1-full-capability",
                "cloud_or_model_special_case": False,
            },
            evidence_refs=(_ref(r078_report),),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="infrastructure_gaps",
            record_id="r078-tencent-api-termination-protection-drift",
            title="Tencent instance termination protection can drift from create request",
            status="validated",
            capability_tags=("cuda-execution-plane", "cost-guard"),
            task_tags=("infrastructure",),
            failure_mode_tags=("resource-leak", "provider-state-drift"),
            source_model="none",
            source_runtime="tencent-cvm-api",
            payload={
                "requested_disable_api_termination": False,
                "observed_disable_api_termination": True,
                "validated_recovery": "explicitly disable protection, terminate, poll inventory to zero",
                "core_framework_changed": False,
            },
            evidence_refs=(
                _ref(r078_report),
                _ref(r078_residual),
                _ref(r078_checkpoint),
            ),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="mechanisms",
            record_id="r083-accelerator-bootstrap-readiness-state-machine",
            title="Provider-neutral accelerator bootstrap readiness state machine",
            status="implemented",
            capability_tags=("cuda-execution-plane", "cost-guard", "replayability"),
            task_tags=("infrastructure",),
            failure_mode_tags=("runtime-mismatch", "provider-state-drift"),
            source_model="none",
            source_runtime="local-control-plane",
            payload={
                "supersedes_interpretation": "permanent-missing-driver",
                "validated_interpretation": "asynchronous-provider-bootstrap-race",
                "states": ("wait", "verify-gpu-smoke", "admit", "terminate"),
                "timeout_seconds": 900,
                "paid_model_work_requires": "accelerator-runtime-preflight-v1",
                "provider_special_case": False,
            },
            evidence_refs=(_ref(r083_correction),),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="infrastructure_gaps",
            record_id="r078-experiment-single-writer-lease",
            title="Append-only experiment roots need a single-writer lease",
            status="pending",
            capability_tags=("evaluation-integrity", "orchestration"),
            task_tags=("infrastructure",),
            failure_mode_tags=("concurrent-writer", "duplicate-execution"),
            source_model="none",
            source_runtime="local-control-plane",
            payload={
                "observed_guard": "append-only duplicate cell rejection",
                "missing_guard": "pre-run experiment-root writer lease",
                "data_overwritten": False,
            },
            evidence_refs=(_ref(r078_report),),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="infrastructure_gaps",
            record_id="r078-aggregate-model-transport-accounting",
            title="Aggregate remote ModelTransport calls across persistent cache misses",
            status="pending",
            capability_tags=("cost-guard", "observability"),
            task_tags=("infrastructure", "holdout"),
            failure_mode_tags=("metric-undercount",),
            source_model="Qwen/Qwen3.5-4B",
            source_runtime="generic-openai-compatible-transport",
            payload={
                "cell_network_flags_true": 0,
                "cached_transport_responses": 15,
                "observed_total_tokens": 93474,
                "required_metric": "aggregate transport cache misses and remote calls",
            },
            evidence_refs=(_ref(r078_report),),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="mechanisms",
            record_id="r079-experiment-root-single-writer-lease",
            title="OS-backed single-writer lease for append-only experiment roots",
            status="implemented",
            capability_tags=("evaluation-integrity", "orchestration"),
            task_tags=("infrastructure",),
            failure_mode_tags=("concurrent-writer", "duplicate-execution"),
            source_model="none",
            source_runtime="local-control-plane",
            payload={
                "supersedes_gap": "r078-experiment-single-writer-lease",
                "lock_policy": "non-blocking OS file lock scoped to runner.run",
                "automatic_release": True,
                "cloud_or_model_special_case": False,
            },
            evidence_refs=(_ref(r079),),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="mechanisms",
            record_id="r079-persistent-model-transport-metrics",
            title="Reconstruct remote calls and token usage from transport cache",
            status="implemented",
            capability_tags=("cost-guard", "observability"),
            task_tags=("infrastructure", "holdout"),
            failure_mode_tags=("metric-undercount",),
            source_model="none",
            source_runtime="generic-openai-compatible-transport",
            payload={
                "supersedes_gap": "r078-aggregate-model-transport-accounting",
                "metrics": (
                    "remote_calls",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                ),
                "invalid_usage_policy": "fail-closed",
                "cloud_or_model_special_case": False,
            },
            evidence_refs=(_ref(r079),),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="mechanisms",
            record_id="r080-explicit-pattern-card-fpr-gate",
            title="Explicit-label PatternCard router FPR calibration gate",
            status="implemented",
            capability_tags=("skill-retrieval", "evaluation-integrity"),
            task_tags=("swe-bench", "feedback"),
            failure_mode_tags=("false-positive-routing", "holdout-leakage"),
            source_model="none",
            source_runtime="local-control-plane",
            payload={
                "negative_pairs": 87,
                "false_positive_rate": 0.0,
                "wilson_95_upper": 0.042288,
                "explicit_labels_only": True,
                "holdout_task_ids_included": False,
                "positive_sample_gate_satisfied": False,
            },
            evidence_refs=(_ref(r080_report), _ref(r080_calibration)),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="skills",
            record_id="r080-mocked-inheritance-trigger-anchor",
            title="Feedback-derived mocked-inheritance PatternCard trigger anchor",
            status="candidate",
            capability_tags=("skill-retrieval", "patch-realization"),
            task_tags=("swe-bench", "python", "documentation"),
            failure_mode_tags=("false-negative-routing",),
            source_model="Qwen/Qwen3.5-4B",
            source_runtime="model-agnostic-skill-text",
            payload={
                "parent_revision_id": "p1-local-qwen-operator-skill-r007-trigger-anchors",
                "revision_id": "p1-local-qwen-operator-skill-r008-trigger-anchors",
                "feedback_true_positives": 3,
                "feedback_false_positives": 0,
                "transformations_changed": False,
                "validations_changed": False,
                "auto_activate": False,
            },
            evidence_refs=(_ref(r080_report), _ref(r080_candidate)),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="mechanisms",
            record_id="r074-unrepairable-semantic-recipe-early-stop",
            title="Early stop for invalid semantic recipe fields and candidate IDs",
            status="implemented",
            capability_tags=("generation-efficiency", "patch-realization"),
            task_tags=("swe-bench",),
            failure_mode_tags=("schema-invalid", "repeated-generation"),
            source_model="none",
            source_runtime="local-control-plane",
            payload={
                "historical_evidence_round": 74,
                "current_implementation": "span_student._span_failure_is_repairable",
                "dedup_policy": "do not reimplement from stale run evidence",
            },
            evidence_refs=(_ref(r080_report),),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="mechanisms",
            record_id="r081-parent-relative-convergence-normalization",
            title="Reachable parent-relative convergence normalization",
            status="implemented",
            capability_tags=("convergence", "evaluation-integrity"),
            task_tags=("infrastructure",),
            failure_mode_tags=("unreachable-gate", "pairing-drift"),
            source_model="none",
            source_runtime="local-control-plane",
            payload={
                "normalization": "parent-relative-delta-of-deltas-v1",
                "epsilon": 0.05,
                "k_consecutive": 2,
                "core_engine_wired": False,
                "pair_count_mismatch_policy": "fail-closed",
            },
            evidence_refs=(_ref(r081_report), _ref(r081_convergence)),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="mechanisms",
            record_id="r081-independent-agent-safety-suite",
            title="Four-category independent Agent safety evidence contract",
            status="implemented",
            capability_tags=("safety", "evaluation-integrity"),
            task_tags=("infrastructure",),
            failure_mode_tags=(
                "dangerous-command",
                "http-5xx",
                "private-data-exposure",
                "unauthorized-side-effect",
            ),
            source_model="none",
            source_runtime="local-control-plane",
            payload={
                "native_admission_reusable": False,
                "evaluator_failure_policy": "fail-closed",
                "candidate_probe_receipts_collected": False,
                "candidate_safety_claim_allowed": False,
            },
            evidence_refs=(_ref(r081_report), _ref(r081_safety)),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="mechanisms",
            record_id="r082-executable-control-gate-projections",
            title="Append-only executable projections for control gates",
            status="implemented",
            capability_tags=("evaluation-integrity", "replayability"),
            task_tags=("infrastructure",),
            failure_mode_tags=("evidence-drift", "unreachable-gate"),
            source_model="none",
            source_runtime="local-control-plane",
            payload={
                "projections": (
                    "parent-relative-convergence",
                    "independent-agent-safety",
                    "statistical-skill-transfer-capability-v2",
                    "replayable-runtime-identity",
                    "accelerator-runtime-preflight",
                ),
                "append_only_replay": True,
                "core_engine_wired": False,
                "cloud_or_model_special_case": False,
            },
            evidence_refs=(_ref(r082_capability),),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="mechanisms",
            record_id="r082-strict-statistical-skill-transfer-gate-v2",
            title="Strict statistical Skill-transfer completion gate",
            status="implemented",
            capability_tags=("agent-self-evolution", "evaluation-integrity"),
            task_tags=("swe-bench", "holdout"),
            failure_mode_tags=("safety-only-evidence", "evaluator-invalid-pair"),
            source_model="none",
            source_runtime="local-control-plane",
            payload={
                "minimum_evaluator_valid_holdout_pairs": 3,
                "safety_only_pairs_count_as_evaluator_valid": False,
                "current_gate_passed": False,
                "current_failed_requirements": (
                    "minimum_evaluator_valid_holdout_pairs",
                    "independent_safety_passed",
                    "runtime_identity_complete",
                ),
            },
            evidence_refs=(_ref(r082_capability),),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="infrastructure_gaps",
            record_id="r083-gpu-driver-bootstrap-identity-not-frozen",
            title="Accelerator bootstrap identity is missing from cloud admission",
            status="validated",
            capability_tags=("cuda-execution-plane", "cost-guard", "replayability"),
            task_tags=("infrastructure", "feedback"),
            failure_mode_tags=("runtime-mismatch", "missing-driver"),
            source_model="none",
            source_runtime="tencent-cvm-api",
            payload={
                "feedback_cells_started": 0,
                "estimated_compute_cost_upper_bound_cny": 0.61,
                "residual_hourly_cost_cny": 0.0,
                "observed_missing": ("nvidia-smi", "docker"),
                "required_generic_gate": "accelerator-runtime-preflight-v1",
                "model_download_before_gate": False,
            },
            evidence_refs=(_ref(r083_preflight), _ref(r083_closeout)),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="experiments",
            record_id="r076-tencent-cuda-holdout-two-pair",
            title="Tencent CUDA bounded holdout shard for Docusaurus and Nushell",
            status="validated",
            capability_tags=("agent-self-evolution", "patch-realization"),
            task_tags=("swe-bench", "holdout"),
            failure_mode_tags=("structural-invalid", "native-unresolved"),
            source_model="Qwen/Qwen3.5-4B",
            source_runtime="tencent-t4-vllm-awq",
            payload={
                "completed_cells": 4,
                "complete_pairs": 2,
                "structural_gain_count": 0,
                "structural_regression_count": 0,
                "native_skipped_structural_invalid": 4,
                "estimated_total_cost_cny_upper_bound": 5.1878,
                "deepseek_calls": 0,
            },
            evidence_refs=(
                _ref(r076_report),
                _ref(r076_progress),
                _ref(r076_checkpoint),
            ),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="infrastructure_gaps",
            record_id="r076-tencent-container-gpu-first-start",
            title="Tencent T4 container GPU first-start enumeration race",
            status="validated",
            capability_tags=("cuda-execution-plane", "remote-inference"),
            task_tags=("infrastructure",),
            failure_mode_tags=("runtime-mismatch", "container-gpu-unavailable"),
            source_model="Qwen/Qwen3.5-4B",
            source_runtime="tencent-t4-vllm-awq",
            payload={
                "symptom": "first vLLM service start saw zero CUDA devices",
                "validated_recovery": "run a minimal --gpus all probe then restart the service once",
                "skill_policy": "do not mutate Skills for container runtime readiness",
                "framework_special_case_added": False,
            },
            evidence_refs=(_ref(r076_report), _ref(r076_checkpoint)),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="experiments",
            record_id="r076-tencent-cuda-holdout-two-pair-boundary-r002",
            title="Evaluator-only boundary correction for the r076 holdout shard",
            status="validated",
            capability_tags=("evaluation-integrity", "holdout-leakage-control"),
            task_tags=("swe-bench", "holdout"),
            failure_mode_tags=("holdout-leakage",),
            source_model="none",
            source_runtime="local-control-plane",
            payload={
                "supersedes_report": "ROUND-REPORT.json",
                "holdout_used_for_skill_training": False,
                "holdout_task_content_exported_to_teacher": False,
                "next_proposal_evidence_scope": "feedback-only",
            },
            evidence_refs=(_ref(r076_report_r002),),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="experiments",
            record_id="r075-aws-g4dn-launch-attempt-002",
            title="AWS CUDA worker retry after completed deployment bundle",
            status="disproven",
            capability_tags=("cuda-execution-plane",),
            task_tags=("infrastructure",),
            failure_mode_tags=("eval-infra",),
            source_model="none",
            source_runtime="aws-ec2-api",
            payload={
                "dry_run": "allowed",
                "real_run": "free-tier-instance-restriction",
                "instances_created": 0,
                "local_inference_resumed": False,
            },
            evidence_refs=(_ref(aws_attempt_2),),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="failure_clusters",
            record_id="r074-structural-valid-native-unresolved",
            title="Structurally valid patches that remain native-unresolved",
            status="validated",
            capability_tags=("patch-realization", "native-validation"),
            task_tags=("swe-bench",),
            failure_mode_tags=("native-unresolved",),
            source_model="Qwen/Qwen3.5-4B",
            source_runtime="mlx-4bit",
            payload={
                "source_round": 74,
                "decision_policy": "prefer Skill; change mechanism only after stable cross-task recurrence",
            },
            evidence_refs=(_ref(feedback),),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="experiments",
            record_id="r075-aws-g4dn-launch-attempt-001",
            title="AWS CUDA worker launch via on-demand and Spot",
            status="disproven",
            capability_tags=("cuda-execution-plane",),
            task_tags=("infrastructure",),
            failure_mode_tags=("eval-infra",),
            source_model="none",
            source_runtime="aws-ec2-api",
            payload={
                "on_demand_attempts": 1,
                "spot_attempts": 1,
                "instances_created": 0,
                "failure_class": "account-free-tier-instance-restriction",
            },
            evidence_refs=(_ref(aws_attempt),),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="infrastructure_gaps",
            record_id="r075-aws-paid-instance-policy",
            title="AWS account policy rejects non-Free-Tier CUDA instances",
            status="pending",
            capability_tags=("cuda-execution-plane",),
            task_tags=("infrastructure",),
            failure_mode_tags=("eval-infra",),
            source_model="none",
            source_runtime="aws-ec2-api",
            payload={
                "region": "ap-southeast-1",
                "requested_instance_type": "g4dn.xlarge",
                "on_demand_result": "InvalidParameterCombination: not Free Tier eligible",
                "spot_result": "InvalidParameterCombination: not Free Tier eligible",
                "instances_created": 0,
            },
            evidence_refs=(_ref(holdout_checkpoint),),
            cross_model_validations=(),
        ),
        EvolutionRecord.create(
            record_type="experiments",
            record_id="r075-aws-cross-region-blocker-audit",
            title="Cross-region EC2 CUDA account-policy audit",
            status="validated",
            capability_tags=("cuda-execution-plane",),
            task_tags=("infrastructure",),
            failure_mode_tags=("eval-infra",),
            source_model="none",
            source_runtime="aws-ec2-api",
            payload={
                "paths_checked": 3,
                "regions_with_real_launch_attempts": 2,
                "free_tier_gpu_types": 0,
                "reusable_accelerator_instances": 0,
                "blocking_external_state": "permit non-Free-Tier GPU instances",
            },
            evidence_refs=(_ref(blocker_audit),),
            cross_model_validations=(),
        ),
    )


def main() -> None:
    catalog = EvolutionCatalog(RUNS / "evolution-catalog")
    results = []
    preserved_conflicts = []
    source_records = records()
    for record in source_records:
        try:
            results.append(catalog.append(record))
        except CatalogConflict:
            # The canonical append-only catalog may already carry a richer
            # historical record under this immutable ID. Preserve it and
            # report the divergence instead of overwriting evidence.
            preserved_conflicts.append(record.record_id)
    print(
        json.dumps(
            {
                "created": sum(row.created for row in results),
                "records_considered": len(source_records),
                "preserved_conflicts": preserved_conflicts,
            }
        )
    )


if __name__ == "__main__":
    main()
