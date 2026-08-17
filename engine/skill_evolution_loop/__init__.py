"""Fail-closed infrastructure for the project-local Skill evolution loop."""

from .block_rewrite import (
    LineBlockRewrite,
    LineBlockRewriteAdapter,
    MlxLineBlockRewriteGenerator,
    build_block_conditions,
)
from .composition import ExperimentEvidenceSource, compose_experiment_evidence
from .contracts import (
    ContractError,
    FailureEvidence,
    FeedbackPackage,
    LoopAuthorization,
    LoopRevision,
    ParentModelRequest,
    ParentModelResponse,
)
from .eval_manifest import (
    EvaluationTask,
    EvaluationTaskSet,
    TaskSetPreflight,
    materialize_evaluation_task,
)
from .evaluator import (
    EvaluationBaseline,
    EvaluationPolicy,
    LoopEvaluator,
    NativeOutcome,
    RoundEvaluation,
)
from .experiment import ExperimentCondition, PairedExperimentRunner
from .ledger import ParentCallLedger, ParentCallRecord
from .loop import LoopConfig, LoopDriver, LoopResult
from .mlx_student import MlxHunkGenerator, MlxStructuredGenerator
from .operator_rewrite import (
    EditIntent,
    MaterializationResult,
    OperatorOperation,
    OperatorPlan,
    materialize_operator_plan,
    run_operator_renderer_qualification,
)
from .operator_student import (
    MlxOperatorPlanGenerator,
    OperatorPlanAdapter,
    build_operator_conditions,
    parse_operator_plan_output,
)
from .p1_block import run_local_qwen_block_p1
from .p1_feedback import freeze_p1_parent_request, freeze_p1_round_parent_request
from .p1_local_feedback import (
    freeze_p1_local_feedback_revision,
    load_frozen_p1_local_feedback_revision,
)
from .p1_native import (
    P1NativeOutcome,
    evaluate_p1_experiment_native,
    evaluate_p1_holdout_cell_native,
    multi_materialized_identity,
    normalize_native_report,
    summarize_native_cells,
)
from .p1_operator import (
    freeze_operator_skill_revision,
    load_frozen_operator_skill_revision,
    run_local_qwen_operator_p1,
)
from .p1_parent import (
    P1DeepSeekTransport,
    P1ParentCallAuthorization,
    audit_p1_parent_call_budget,
    dispatch_p1_parent_call,
    freeze_p1_parent_preflight,
    load_frozen_p1_parent_request,
    load_frozen_p1_parent_revision,
    p1_parent_preflight,
)
from .p1_strategy import (
    dispatch_p1_realization_strategy_call,
    freeze_p1_realization_strategy_request,
)
from .p1_symbol import run_local_qwen_symbol_p1
from .parent_model import ParentModelAdapter
from .realization_adapter import DiagnosisFrozenRealizationAdapter
from .realization_candidates import (
    FrozenDiagnosis,
    RealizationCandidate,
    RealizationSelection,
    select_realization_candidate,
)
from .registry import LoopRevisionRegistry
from .round1_feedback import (
    compile_round1_feedback_skills,
    create_round1_feedback_authorization,
    dispatch_round1_feedback_strategy,
    freeze_round1_feedback_request,
)
from .round1_native import (
    retry_round1_holdout_native_failures,
    run_round1_holdout_native,
)
from .round1_run import load_round1_feedback_gain_gate, run_round1_holdout
from .scale_readiness import freeze_round1_scale_readiness
from .span_rewrite import (
    SpanEditIntent,
    SpanMaterializationResult,
    SpanOperation,
    SpanPlan,
    materialize_span_plan,
    run_span_renderer_qualification,
)
from .span_student import (
    MlxSpanPlanGenerator,
    SpanPlanAdapter,
    build_span_conditions,
    parse_span_plan_output,
)
from .student_adapter import (
    HunkStudentAdapter,
    StructuredEdit,
    StudentAdapter,
    StudentAttempt,
    StudentTask,
)
from .symbol_rewrite import (
    MlxSymbolRewriteGenerator,
    SymbolRewrite,
    SymbolRewriteAdapter,
    build_symbol_conditions,
)
from .target_audit import (
    GoldPatchReference,
    MechanismCapacityAudit,
    MechanismCapacityPolicy,
    MechanismCapacityRow,
    TargetCoverageAudit,
    TargetCoverageRow,
    audit_mechanism_capacity,
    audit_target_coverage,
)
from .target_selection import TargetSelectionManifest, TargetSelectionRecord

__all__ = [
    "ContractError",
    "LineBlockRewrite",
    "LineBlockRewriteAdapter",
    "MlxLineBlockRewriteGenerator",
    "build_block_conditions",
    "EvaluationBaseline",
    "EvaluationPolicy",
    "EvaluationTask",
    "EvaluationTaskSet",
    "ExperimentEvidenceSource",
    "ExperimentCondition",
    "FailureEvidence",
    "FeedbackPackage",
    "HunkStudentAdapter",
    "LoopAuthorization",
    "LoopConfig",
    "LoopDriver",
    "LoopEvaluator",
    "LoopResult",
    "MlxStructuredGenerator",
    "MlxHunkGenerator",
    "MlxOperatorPlanGenerator",
    "OperatorPlanAdapter",
    "build_operator_conditions",
    "parse_operator_plan_output",
    "freeze_operator_skill_revision",
    "load_frozen_operator_skill_revision",
    "run_local_qwen_operator_p1",
    "freeze_round1_scale_readiness",
    "freeze_round1_feedback_request",
    "compile_round1_feedback_skills",
    "create_round1_feedback_authorization",
    "dispatch_round1_feedback_strategy",
    "load_round1_feedback_gain_gate",
    "run_round1_holdout",
    "run_round1_holdout_native",
    "retry_round1_holdout_native_failures",
    "SpanEditIntent",
    "SpanMaterializationResult",
    "SpanOperation",
    "SpanPlan",
    "materialize_span_plan",
    "run_span_renderer_qualification",
    "MlxSpanPlanGenerator",
    "SpanPlanAdapter",
    "build_span_conditions",
    "parse_span_plan_output",
    "EditIntent",
    "MaterializationResult",
    "OperatorOperation",
    "OperatorPlan",
    "materialize_operator_plan",
    "run_operator_renderer_qualification",
    "LoopRevision",
    "LoopRevisionRegistry",
    "ParentCallLedger",
    "ParentCallRecord",
    "ParentModelAdapter",
    "ParentModelRequest",
    "ParentModelResponse",
    "FrozenDiagnosis",
    "DiagnosisFrozenRealizationAdapter",
    "RealizationCandidate",
    "RealizationSelection",
    "select_realization_candidate",
    "PairedExperimentRunner",
    "NativeOutcome",
    "RoundEvaluation",
    "StudentAdapter",
    "StudentAttempt",
    "StudentTask",
    "SymbolRewrite",
    "SymbolRewriteAdapter",
    "MlxSymbolRewriteGenerator",
    "build_symbol_conditions",
    "GoldPatchReference",
    "MechanismCapacityAudit",
    "MechanismCapacityPolicy",
    "MechanismCapacityRow",
    "TargetCoverageAudit",
    "TargetCoverageRow",
    "audit_mechanism_capacity",
    "audit_target_coverage",
    "TargetSelectionManifest",
    "TargetSelectionRecord",
    "StructuredEdit",
    "TaskSetPreflight",
    "materialize_evaluation_task",
    "compose_experiment_evidence",
    "freeze_p1_parent_request",
    "freeze_p1_round_parent_request",
    "run_local_qwen_block_p1",
    "freeze_p1_local_feedback_revision",
    "load_frozen_p1_local_feedback_revision",
    "P1DeepSeekTransport",
    "P1ParentCallAuthorization",
    "audit_p1_parent_call_budget",
    "dispatch_p1_parent_call",
    "freeze_p1_parent_preflight",
    "load_frozen_p1_parent_request",
    "load_frozen_p1_parent_revision",
    "p1_parent_preflight",
    "run_local_qwen_symbol_p1",
    "freeze_p1_realization_strategy_request",
    "dispatch_p1_realization_strategy_call",
    "P1NativeOutcome",
    "evaluate_p1_experiment_native",
    "evaluate_p1_holdout_cell_native",
    "multi_materialized_identity",
    "normalize_native_report",
    "summarize_native_cells",
]
