"""Append-only native evaluation for frozen P1 Student experiment cells."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from official_patch_evaluator import OfficialEvaluatorError

from .contracts import ContractError, canonical_json, sha256_json
from .eval_manifest import EvaluationTaskSet


class PatchEvaluator(Protocol):
    """The frozen official one-patch evaluator interface."""

    def __call__(
        self, invocation: Any, materialized: dict[str, Any], receipt: dict[str, Any]
    ) -> Path: ...


@dataclass(frozen=True)
class P1NativeOutcome:
    """Normalized native outcome without model- or harness-specific fields."""

    resolved: bool
    native_valid: bool
    native_error: str | None
    regression_test_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved": self.resolved,
            "native_valid": self.native_valid,
            "native_error": self.native_error,
            "regression_test_names": list(self.regression_test_names),
        }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _string_set(value: Any, *, field: str) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(row, str) for row in value):
        raise ContractError(f"{field} must be a list of strings")
    return set(value)


def normalize_native_report(
    report: dict[str, Any], *, benchmark_id: str, instance_id: str
) -> P1NativeOutcome:
    """Normalize one report produced by a pinned official evaluator."""

    if benchmark_id in {"swe-bench-verified", "swe-bench-multilingual"}:
        if report.get("schema_version") == 2:
            submitted = _string_set(report.get("submitted_ids"), field="submitted_ids")
            resolved = _string_set(report.get("resolved_ids"), field="resolved_ids")
            unresolved = _string_set(
                report.get("unresolved_ids"), field="unresolved_ids"
            )
            errors = _string_set(report.get("error_ids"), field="error_ids")
            empty = _string_set(report.get("empty_patch_ids"), field="empty_patch_ids")
            if instance_id not in submitted:
                raise ContractError("native report does not contain the instance")
            states = {
                "resolved": instance_id in resolved,
                "unresolved": instance_id in unresolved,
                "evaluator_error": instance_id in errors,
                "empty_patch": instance_id in empty,
            }
            if sum(states.values()) != 1:
                raise ContractError("native report outcome is ambiguous")
            error = next(
                (name for name in ("evaluator_error", "empty_patch") if states[name]),
                None,
            )
            return P1NativeOutcome(
                resolved=states["resolved"],
                native_valid=error is None,
                native_error=error,
                regression_test_names=(),
            )
        row = report.get(instance_id)
        if not isinstance(row, dict):
            raise ContractError("native SWE report does not contain the instance")
        resolved = row.get("resolved")
        applied = row.get("patch_successfully_applied")
        statuses = row.get("tests_status")
        if not isinstance(resolved, bool) or not isinstance(applied, bool):
            raise ContractError("native SWE outcome fields are invalid")
        if not isinstance(statuses, dict):
            raise ContractError("native SWE test status is missing")
        regressions: set[str] = set()
        for category in ("PASS_TO_PASS", "PASS_TO_FAIL"):
            bucket = statuses.get(category, {})
            if not isinstance(bucket, dict):
                raise ContractError("native SWE test status bucket is invalid")
            regressions.update(
                _string_set(bucket.get("failure", []), field=f"{category}.failure")
            )
        return P1NativeOutcome(
            resolved=resolved,
            native_valid=applied,
            native_error=None if applied else "patch_apply_failed",
            regression_test_names=tuple(sorted(regressions)),
        )
    if benchmark_id != "multi-swe-bench-flash":
        raise ContractError("unsupported native benchmark")
    org = report.get("org")
    repo = report.get("repo")
    number = report.get("number")
    valid = report.get("valid")
    if (
        not isinstance(org, str)
        or not isinstance(repo, str)
        or not isinstance(number, int)
    ):
        raise ContractError("native Multi-SWE identity is invalid")
    if instance_id not in {f"{org}/{repo}:pr-{number}", f"{org}__{repo}-{number}"}:
        raise ContractError("native Multi-SWE instance mismatch")
    if not isinstance(valid, bool):
        raise ContractError("native Multi-SWE validity is missing")
    before = report.get("test_patch_result")
    after = report.get("fix_patch_result")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ContractError("native Multi-SWE test results are missing")
    regressions = _string_set(
        before.get("passed_tests"), field="test_patch_result.passed_tests"
    ) & _string_set(after.get("failed_tests"), field="fix_patch_result.failed_tests")
    error = report.get("error_msg")
    normalized_error = str(error).strip() if error is not None else ""
    return P1NativeOutcome(
        resolved=valid,
        native_valid=valid,
        native_error=normalized_error or None,
        regression_test_names=tuple(sorted(regressions)),
    )


def multi_materialized_identity(pool_root: Path, instance_id: str) -> dict[str, Any]:
    """Reproduce the frozen dataset-row identity required by Multi-SWE."""

    dataset = (
        pool_root.resolve() / "inputs/multi-swe-bench-flash/multi_swe_bench_flash.jsonl"
    )
    if not dataset.is_file():
        raise ContractError("frozen Multi-SWE dataset is missing")
    matches = []
    for line in dataset.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("instance_id") == instance_id:
                matches.append(row)
    if len(matches) != 1:
        raise ContractError("frozen Multi-SWE instance identity is ambiguous")
    encoded = canonical_json(matches[0]).encode()
    return {"source_content_sha256": hashlib.sha256(encoded).hexdigest()}


def _read_cell(path: Path) -> dict[str, Any]:
    attempt_path = path / "ATTEMPT.json"
    try:
        report = json.loads(attempt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("P1 experiment cell is unreadable") from exc
    integrity = report.get("evidence_sha256") if isinstance(report, dict) else None
    content = {key: value for key, value in report.items() if key != "evidence_sha256"}
    if integrity != sha256_json(content):
        raise ContractError("P1 experiment cell evidence was tampered")
    for name, digest in report.get("artifact_sha256", {}).items():
        artifact = path / name
        if not artifact.is_file() or _sha256_file(artifact) != digest:
            raise ContractError("P1 experiment cell artifact was tampered")
    return report


def _treatment_prompt_sequence(
    cell_path: Path, report: dict[str, Any]
) -> tuple[tuple[str, ...], str]:
    """Validate and fingerprint prompts that can influence a repair arm."""

    prompts: list[str] = []
    for trace in report.get("generation_trace", []):
        if not isinstance(trace, dict):
            raise ContractError("P1 treatment generation trace is invalid")
        prompt_value = trace.get("prompt_path")
        if prompt_value is None:
            continue
        if not isinstance(prompt_value, str) or not prompt_value:
            raise ContractError("P1 treatment prompt path is invalid")
        prompt_path = cell_path / prompt_value
        try:
            prompt_bytes = prompt_path.read_bytes()
            prompt = prompt_bytes.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise ContractError("P1 treatment prompt is unreadable") from exc
        digest = hashlib.sha256(prompt_bytes).hexdigest()
        if trace.get("prompt_sha256") != digest:
            raise ContractError("P1 treatment prompt sha256 mismatch")
        prompts.append(prompt)
    if not prompts:
        raise ContractError("P1 structural repair has no treatment prompt evidence")
    sequence = tuple(prompts)
    return sequence, sha256_json(list(sequence))


def validate_paired_treatment_prompt_identity(
    *, experiment_root: Path, task_id: str, mechanism: str
) -> dict[str, Any]:
    """Prove that baseline and taught repair arms received distinct treatment."""

    experiment = experiment_root.resolve()
    evidence: dict[str, dict[str, Any]] = {}
    for teaching in ("baseline", "taught"):
        condition_id = f"{mechanism}-{teaching}"
        cell_path = experiment / "cells" / task_id / condition_id
        if not cell_path.is_dir():
            raise ContractError("P1 paired treatment experiment cell is missing")
        report = _read_cell(cell_path)
        condition = report.get("condition")
        if (
            not isinstance(condition, dict)
            or condition.get("condition_id") != condition_id
            or condition.get("mechanism") != mechanism
            or condition.get("teaching") != teaching
        ):
            raise ContractError("P1 paired treatment condition identity mismatch")
        revision = condition.get("revision")
        skill_text = revision.get("skill_text") if isinstance(revision, dict) else None
        if not isinstance(skill_text, str) or not skill_text.strip():
            raise ContractError("P1 paired treatment Skill evidence is missing")
        prompts, fingerprint = _treatment_prompt_sequence(cell_path, report)
        evidence[teaching] = {
            "prompts": prompts,
            "prompt_sequence_sha256": fingerprint,
            "experiment_cell_sha256": report["evidence_sha256"],
            "skill_text_sha256": hashlib.sha256(skill_text.encode()).hexdigest(),
        }
    if (
        evidence["baseline"]["skill_text_sha256"]
        == evidence["taught"]["skill_text_sha256"]
    ):
        raise ContractError("P1 baseline and taught Skill treatments are identical")
    if evidence["baseline"]["prompts"] == evidence["taught"]["prompts"]:
        raise ContractError("P1 baseline and taught treatment prompts are identical")
    content = {
        "schema_version": 1,
        "contract": "paired-treatment-prompt-identity-v1",
        "task_id": task_id,
        "mechanism": mechanism,
        "baseline_experiment_cell_sha256": evidence["baseline"][
            "experiment_cell_sha256"
        ],
        "taught_experiment_cell_sha256": evidence["taught"]["experiment_cell_sha256"],
        "baseline_skill_text_sha256": evidence["baseline"]["skill_text_sha256"],
        "taught_skill_text_sha256": evidence["taught"]["skill_text_sha256"],
        "baseline_prompt_sequence_sha256": evidence["baseline"][
            "prompt_sequence_sha256"
        ],
        "taught_prompt_sequence_sha256": evidence["taught"]["prompt_sequence_sha256"],
        "treatment_prompt_sequences_differ": True,
        "causal_pair_valid": True,
    }
    return {**content, "evidence_sha256": sha256_json(content)}


def validate_paired_native_admission(
    *, experiment_root: Path, task_id: str, mechanism: str
) -> dict[str, Any]:
    """Admit native evaluation only for a complete structural and causal pair."""

    experiment = experiment_root.resolve()
    cells: dict[str, dict[str, Any]] = {}
    structural: dict[str, bool] = {}
    implementation: dict[str, str | None] = {}
    for teaching in ("baseline", "taught"):
        condition_id = f"{mechanism}-{teaching}"
        cell_path = experiment / "cells" / task_id / condition_id
        if not cell_path.is_dir():
            raise ContractError("P1 paired native experiment cell is missing")
        report = _read_cell(cell_path)
        condition = report.get("condition")
        attempt = report.get("attempt")
        if (
            not isinstance(condition, dict)
            or condition.get("condition_id") != condition_id
            or condition.get("mechanism") != mechanism
            or condition.get("teaching") != teaching
            or not isinstance(attempt, dict)
            or type(attempt.get("structural_valid")) is not bool
        ):
            raise ContractError("P1 paired native experiment identity is invalid")
        cells[teaching] = report
        structural[teaching] = attempt["structural_valid"]
        fingerprint = attempt.get("implementation_fingerprint")
        implementation[teaching] = (
            fingerprint
            if isinstance(fingerprint, str)
            and len(fingerprint) == 64
            and all(character in "0123456789abcdef" for character in fingerprint)
            else None
        )

    treatment = (
        validate_paired_treatment_prompt_identity(
            experiment_root=experiment,
            task_id=task_id,
            mechanism=mechanism,
        )
        if all(structural.values())
        else None
    )
    implementation_ready = all(implementation.values())
    implementation_differ = (
        implementation_ready and implementation["baseline"] != implementation["taught"]
    )
    admitted = (
        all(structural.values()) and treatment is not None and implementation_differ
    )
    if not all(structural.values()):
        reason = "paired-structural-invalid"
    elif not implementation_ready:
        reason = "paired-implementation-fingerprint-missing"
    elif not implementation_differ:
        reason = "paired-implementation-identical"
    else:
        reason = "admitted"
    content = {
        "schema_version": 1,
        "contract": "paired-native-admission-v1",
        "task_id": task_id,
        "mechanism": mechanism,
        "baseline_experiment_cell_sha256": cells["baseline"]["evidence_sha256"],
        "taught_experiment_cell_sha256": cells["taught"]["evidence_sha256"],
        "structural_valid": structural,
        "implementation_fingerprints": implementation,
        "implementation_fingerprints_differ": implementation_differ,
        "paired_treatment_prompt_identity": treatment,
        "native_admitted": admitted,
        "reason": reason,
    }
    return {**content, "evidence_sha256": sha256_json(content)}


def _write_cell(path: Path, content: dict[str, Any]) -> dict[str, Any]:
    report = {**content, "evidence_sha256": sha256_json(content)}
    if path.exists():
        existing = json.loads((path / "NATIVE-CELL.json").read_text(encoding="utf-8"))
        if existing != report:
            raise ContractError("frozen native cell does not match replay")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    (temporary / "NATIVE-CELL.json").write_text(
        canonical_json(report) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return report


def _freeze_unreceipted_evaluator_failure(
    target: Path, exc: OfficialEvaluatorError
) -> Path:
    """Persist evaluator failures that occur before its normal receipt boundary."""

    failure = target.parent / f".{target.name}-evaluator-failure.json"
    content = {
        "schema_version": 1,
        "status": "failed",
        "error_code": exc.error_code,
        "reason": str(exc),
    }
    encoded = canonical_json(content) + "\n"
    if failure.exists():
        if failure.read_text(encoding="utf-8") != encoded:
            raise ContractError("P1 native evaluator preflight failure drifted")
    else:
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(encoded, encoding="utf-8")
    return failure


def _read_native_cell(path: Path) -> dict[str, Any]:
    try:
        report = json.loads((path / "NATIVE-CELL.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("P1 native cell is unreadable") from exc
    integrity = report.get("evidence_sha256") if isinstance(report, dict) else None
    content = {key: value for key, value in report.items() if key != "evidence_sha256"}
    if integrity != sha256_json(content):
        raise ContractError("P1 native cell evidence was tampered")
    native_report = report.get("native_report")
    if native_report is not None:
        path_value = (
            native_report.get("path") if isinstance(native_report, dict) else None
        )
        digest = (
            native_report.get("sha256") if isinstance(native_report, dict) else None
        )
        source = Path(str(path_value)).resolve()
        if not source.is_file() or digest != _sha256_file(source):
            raise ContractError("P1 native report evidence was tampered")
    native_failure = report.get("native_evaluator_failure")
    if native_failure is not None:
        path_value = (
            native_failure.get("path") if isinstance(native_failure, dict) else None
        )
        digest = (
            native_failure.get("sha256") if isinstance(native_failure, dict) else None
        )
        source = Path(str(path_value)).resolve()
        if not source.is_file() or digest != _sha256_file(source):
            raise ContractError("P1 native evaluator failure evidence was tampered")
    return report


def summarize_native_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the paired capability gate from normalized append-only cells."""

    indexed = {
        (row["task_id"], row["mechanism"], row["teaching"]): row for row in cells
    }
    pairs = []
    for task_id, mechanism in sorted({(key[0], key[1]) for key in indexed}):
        baseline = indexed.get((task_id, mechanism, "baseline"))
        taught = indexed.get((task_id, mechanism, "taught"))
        if baseline is None or taught is None:
            continue
        evaluator_valid = all(
            row["outcome"].get("native_valid") is True
            and row["outcome"].get("native_error") is None
            for row in (baseline, taught)
        )
        no_op_equivalent = all(
            row.get("patch_sha256")
            == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            and row["outcome"].get("resolved") is False
            and row["outcome"].get("native_error") == "structural_invalid"
            and row.get("native_report") is None
            and row.get("native_evaluator_failure") is None
            for row in (baseline, taught)
        )
        gained = (
            evaluator_valid
            and not baseline["outcome"]["resolved"]
            and taught["outcome"]["resolved"]
        )
        regressed = (
            evaluator_valid
            and baseline["outcome"]["resolved"]
            and not taught["outcome"]["resolved"]
        )
        pairs.append(
            {
                "task_id": task_id,
                "cohort": baseline["cohort"],
                "mechanism": mechanism,
                "baseline_resolved": baseline["outcome"]["resolved"],
                "taught_resolved": taught["outcome"]["resolved"],
                "evaluator_valid": evaluator_valid,
                "no_op_equivalent": no_op_equivalent,
                "safety_qualified": evaluator_valid or no_op_equivalent,
                "gained": gained,
                "regressed": regressed,
            }
        )
    feedback_gains = [
        row for row in pairs if row["cohort"] == "feedback" and row["gained"]
    ]
    holdout_regressions = [
        row for row in pairs if row["cohort"] == "holdout" and row["regressed"]
    ]
    holdout_evaluable_pairs = [
        row for row in pairs if row["cohort"] == "holdout" and row["evaluator_valid"]
    ]
    holdout_safety_qualified_pairs = [
        row for row in pairs if row["cohort"] == "holdout" and row["safety_qualified"]
    ]
    holdout_safety_failures = []
    for pair in holdout_evaluable_pairs:
        task_id = pair["task_id"]
        mechanism = pair["mechanism"]
        baseline = indexed[(task_id, mechanism, "baseline")]
        taught = indexed[(task_id, mechanism, "taught")]
        baseline_failures = set(baseline["outcome"].get("regression_test_names", []))
        taught_failures = set(taught["outcome"].get("regression_test_names", []))
        introduced = sorted(taught_failures - baseline_failures)
        if introduced:
            holdout_safety_failures.append(
                {
                    "task_id": task_id,
                    "mechanism": mechanism,
                    "regression_test_names": introduced,
                }
            )
    admission_receipts: dict[tuple[str, str], dict[str, Any]] = {}
    for row in cells:
        admission = row.get("paired_native_admission")
        if not isinstance(admission, dict):
            continue
        key = (str(row["task_id"]), str(row["mechanism"]))
        existing = admission_receipts.setdefault(key, admission)
        if existing != admission:
            raise ContractError("paired native admission receipt differs across arms")
    admitted_pairs = sum(
        receipt.get("native_admitted") is True
        for receipt in admission_receipts.values()
    )
    structural_invalid_pairs = sum(
        receipt.get("reason") == "paired-structural-invalid"
        for receipt in admission_receipts.values()
    )
    avoided_invocations = sum(
        sum(value is True for value in receipt.get("structural_valid", {}).values())
        for receipt in admission_receipts.values()
        if receipt.get("reason") == "paired-structural-invalid"
    )
    return {
        "pairs": pairs,
        "feedback_gain_count": len(feedback_gains),
        "holdout_regression_count": len(holdout_regressions),
        "holdout_evaluable_pair_count": len(holdout_evaluable_pairs),
        "holdout_safety_qualified_pair_count": len(holdout_safety_qualified_pairs),
        "holdout_safety_failure_count": len(holdout_safety_failures),
        "holdout_safety_failures": holdout_safety_failures,
        "native_admitted_pair_count": admitted_pairs,
        "paired_structural_invalid_pair_count": structural_invalid_pairs,
        "native_invocations_avoided_by_pair_gate": avoided_invocations,
        "capability_gate_passed": (
            bool(feedback_gains)
            and not holdout_regressions
            and not holdout_safety_failures
        ),
    }


def evaluate_p1_feedback_cell_native(
    *,
    manifest_path: Path,
    experiment_root: Path,
    evidence_root: Path,
    pool_root: Path,
    evaluator: PatchEvaluator,
    task_id: str,
    condition_id: str,
    _expected_cohort: str = "feedback",
    _feedback_gain_summary_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate one frozen feedback cell before the holdout cohort is opened.

    The complete experiment evaluator intentionally requires every planned cell.
    This narrower gate exists so a feedback-only native gain can be established
    first, without generating or inspecting evaluator-only holdout attempts.
    """

    taskset = EvaluationTaskSet.from_dict(
        json.loads(manifest_path.resolve().read_text(encoding="utf-8"))
    )
    matches = [task for task in taskset.tasks if task.task_id == task_id]
    if len(matches) != 1:
        raise ContractError("P1 feedback native task identity is ambiguous")
    task = matches[0]
    if _expected_cohort == "feedback":
        if task.cohort != "feedback":
            raise ContractError("P1 feedback native gate cannot open holdout cells")
        if _feedback_gain_summary_sha256 is not None:
            raise ContractError("P1 feedback native gate cannot bind a holdout unlock")
    elif _expected_cohort == "holdout":
        if task.cohort != "holdout":
            raise ContractError("P1 holdout native gate requires a holdout cell")
        if (
            not isinstance(_feedback_gain_summary_sha256, str)
            or len(_feedback_gain_summary_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in _feedback_gain_summary_sha256
            )
        ):
            raise ContractError("P1 holdout native gate requires feedback gain sha256")
    else:
        raise ContractError("P1 native single-cell cohort is invalid")

    experiment = experiment_root.resolve()
    condition_root = experiment / "cells" / task.task_id / condition_id
    if not condition_root.is_dir():
        raise ContractError("P1 feedback experiment cell is missing")
    frozen = _read_cell(condition_root)
    condition = frozen.get("condition")
    if not isinstance(condition, dict) or condition.get("condition_id") != condition_id:
        raise ContractError("P1 feedback experiment condition identity mismatch")

    target = evidence_root.resolve() / "cells" / task.task_id / condition_id
    if target.exists():
        existing = _read_native_cell(target)
        if (
            existing.get("taskset_fingerprint") != taskset.fingerprint
            or existing.get("experiment_cell_sha256") != frozen["evidence_sha256"]
        ):
            raise ContractError("P1 native cell identity does not match replay")
        return existing

    attempt = frozen.get("attempt")
    if not isinstance(attempt, dict) or not isinstance(
        attempt.get("structural_valid"), bool
    ):
        raise ContractError("P1 feedback experiment attempt is invalid")
    patch = condition_root / "patch.diff"
    if not patch.is_file():
        raise ContractError("P1 feedback experiment patch is missing")

    native_report_evidence: dict[str, Any] | None = None
    native_evaluator_failure: dict[str, Any] | None = None
    pair_admission = validate_paired_native_admission(
        experiment_root=experiment,
        task_id=task.task_id,
        mechanism=str(condition.get("mechanism")),
    )
    if attempt["structural_valid"] and pair_admission["native_admitted"]:
        revision = condition.get("revision")
        fingerprint = (
            revision.get("fingerprint") if isinstance(revision, dict) else None
        )
        if not isinstance(fingerprint, str):
            raise ContractError("P1 feedback condition revision is invalid")
        treatment_identity = pair_admission["paired_treatment_prompt_identity"]
        materialized = (
            multi_materialized_identity(pool_root, task.instance_id)
            if task.benchmark_id == "multi-swe-bench-flash"
            else {}
        )
        invocation = SimpleNamespace(
            round_id=task.task_id,
            arm=condition_id,
            benchmark_id=task.benchmark_id,
            instance_id=task.instance_id,
            agent_program_sha256=fingerprint,
        )
        try:
            report_path = evaluator(
                invocation, materialized, {"prediction": {"path": str(patch)}}
            )
        except OfficialEvaluatorError as exc:
            evidence_path = exc.evidence_path
            if evidence_path is None:
                evidence_path = _freeze_unreceipted_evaluator_failure(target, exc)
            elif not evidence_path.is_file():
                raise
            outcome = P1NativeOutcome(
                resolved=False,
                native_valid=False,
                native_error=exc.error_code,
                regression_test_names=(),
            )
            native_evaluator_failure = {
                "path": str(evidence_path.resolve()),
                "sha256": _sha256_file(evidence_path),
            }
        else:
            native_report = json.loads(report_path.read_text(encoding="utf-8"))
            outcome = normalize_native_report(
                native_report,
                benchmark_id=task.benchmark_id,
                instance_id=task.instance_id,
            )
            native_report_evidence = {
                "path": str(report_path.resolve()),
                "sha256": _sha256_file(report_path),
            }
    elif not attempt["structural_valid"]:
        treatment_identity = None
        outcome = P1NativeOutcome(
            resolved=False,
            native_valid=False,
            native_error="structural_invalid",
            regression_test_names=(),
        )
    else:
        treatment_identity = None
        outcome = P1NativeOutcome(
            resolved=False,
            native_valid=False,
            native_error="paired_structural_invalid",
            regression_test_names=(),
        )

    content = {
        "schema_version": 1,
        "evaluation_scope": f"{_expected_cohort}-cell",
        "taskset_fingerprint": taskset.fingerprint,
        "experiment_summary_sha256": None,
        "task_id": task.task_id,
        "instance_id": task.instance_id,
        "benchmark_id": task.benchmark_id,
        "cohort": task.cohort,
        "condition_id": condition_id,
        "mechanism": condition.get("mechanism"),
        "teaching": condition.get("teaching"),
        "experiment_cell_sha256": frozen["evidence_sha256"],
        "patch_sha256": _sha256_file(patch),
        "outcome": outcome.to_dict(),
        "native_report": native_report_evidence,
        "native_evaluator_failure": native_evaluator_failure,
        "paired_native_admission": pair_admission,
        "paired_treatment_prompt_identity": treatment_identity,
        "network_calls_performed": False,
        "holdout_cells_opened": _expected_cohort == "holdout",
    }
    if _expected_cohort == "holdout":
        content["feedback_gain_summary_sha256"] = _feedback_gain_summary_sha256
    return _write_cell(target, content)


def evaluate_p1_holdout_cell_native(
    *,
    manifest_path: Path,
    experiment_root: Path,
    evidence_root: Path,
    pool_root: Path,
    evaluator: PatchEvaluator,
    task_id: str,
    condition_id: str,
    feedback_gain_summary_sha256: str,
) -> dict[str, Any]:
    """Evaluate one holdout cell only after the caller validates feedback gain."""

    return evaluate_p1_feedback_cell_native(
        manifest_path=manifest_path,
        experiment_root=experiment_root,
        evidence_root=evidence_root,
        pool_root=pool_root,
        evaluator=evaluator,
        task_id=task_id,
        condition_id=condition_id,
        _expected_cohort="holdout",
        _feedback_gain_summary_sha256=feedback_gain_summary_sha256,
    )


def evaluate_p1_experiment_native(
    *,
    manifest_path: Path,
    experiment_root: Path,
    evidence_root: Path,
    pool_root: Path,
    evaluator: PatchEvaluator,
    max_cells: int | None = None,
) -> dict[str, Any]:
    """Evaluate or resume structurally valid experiment cells with native judges."""

    if max_cells is not None and (type(max_cells) is not int or max_cells < 1):
        raise ContractError("max_cells must be a positive integer")
    taskset = EvaluationTaskSet.from_dict(
        json.loads(manifest_path.resolve().read_text(encoding="utf-8"))
    )
    experiment = experiment_root.resolve()
    summary_path = experiment / "SUMMARY.json"
    if not summary_path.is_file():
        raise ContractError("P1 experiment is not complete")
    experiment_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        experiment_summary.get("status") != "complete"
        or experiment_summary.get("taskset_fingerprint") != taskset.fingerprint
    ):
        raise ContractError("P1 experiment summary does not match the taskset")
    root = evidence_root.resolve()
    completed: list[dict[str, Any]] = []
    executed = 0
    for task in taskset.tasks:
        task_root = experiment / "cells" / task.task_id
        if not task_root.is_dir():
            raise ContractError("P1 experiment task evidence is missing")
        for condition_root in sorted(
            path for path in task_root.iterdir() if path.is_dir()
        ):
            frozen = _read_cell(condition_root)
            condition = frozen["condition"]
            target = root / "cells" / task.task_id / condition["condition_id"]
            if target.exists():
                existing = _read_native_cell(target)
                if (
                    existing.get("taskset_fingerprint") != taskset.fingerprint
                    or existing.get("experiment_cell_sha256")
                    != frozen["evidence_sha256"]
                ):
                    raise ContractError("P1 native cell identity does not match replay")
                completed.append(existing)
                continue
            if max_cells is not None and executed >= max_cells:
                continue
            attempt = frozen["attempt"]
            patch = condition_root / "patch.diff"
            native_report: dict[str, Any] | None = None
            native_report_evidence: dict[str, Any] | None = None
            native_evaluator_failure: dict[str, Any] | None = None
            pair_admission = validate_paired_native_admission(
                experiment_root=experiment,
                task_id=task.task_id,
                mechanism=str(condition.get("mechanism")),
            )
            if attempt["structural_valid"] and pair_admission["native_admitted"]:
                treatment_identity = pair_admission["paired_treatment_prompt_identity"]
                materialized = (
                    multi_materialized_identity(pool_root, task.instance_id)
                    if task.benchmark_id == "multi-swe-bench-flash"
                    else {}
                )
                invocation = SimpleNamespace(
                    round_id=task.task_id,
                    arm=condition["condition_id"],
                    benchmark_id=task.benchmark_id,
                    instance_id=task.instance_id,
                    agent_program_sha256=condition["revision"]["fingerprint"],
                )
                try:
                    report_path = evaluator(
                        invocation,
                        materialized,
                        {"prediction": {"path": str(patch)}},
                    )
                except OfficialEvaluatorError as exc:
                    evidence_path = exc.evidence_path
                    if evidence_path is None:
                        evidence_path = _freeze_unreceipted_evaluator_failure(
                            target, exc
                        )
                    elif not evidence_path.is_file():
                        raise
                    outcome = P1NativeOutcome(
                        resolved=False,
                        native_valid=False,
                        native_error=exc.error_code,
                        regression_test_names=(),
                    )
                    native_evaluator_failure = {
                        "path": str(evidence_path.resolve()),
                        "sha256": _sha256_file(evidence_path),
                    }
                else:
                    native_report = json.loads(report_path.read_text(encoding="utf-8"))
                    outcome = normalize_native_report(
                        native_report,
                        benchmark_id=task.benchmark_id,
                        instance_id=task.instance_id,
                    )
                    native_report_evidence = {
                        "path": str(report_path.resolve()),
                        "sha256": _sha256_file(report_path),
                    }
            elif not attempt["structural_valid"]:
                treatment_identity = None
                outcome = P1NativeOutcome(
                    resolved=False,
                    native_valid=False,
                    native_error="structural_invalid",
                    regression_test_names=(),
                )
            else:
                treatment_identity = None
                outcome = P1NativeOutcome(
                    resolved=False,
                    native_valid=False,
                    native_error="paired_structural_invalid",
                    regression_test_names=(),
                )
            content = {
                "schema_version": 1,
                "taskset_fingerprint": taskset.fingerprint,
                "experiment_summary_sha256": experiment_summary["summary_sha256"],
                "task_id": task.task_id,
                "instance_id": task.instance_id,
                "benchmark_id": task.benchmark_id,
                "cohort": task.cohort,
                "condition_id": condition["condition_id"],
                "mechanism": condition["mechanism"],
                "teaching": condition["teaching"],
                "experiment_cell_sha256": frozen["evidence_sha256"],
                "patch_sha256": _sha256_file(patch),
                "outcome": outcome.to_dict(),
                "native_report": native_report_evidence,
                "native_evaluator_failure": native_evaluator_failure,
                "paired_native_admission": pair_admission,
                "paired_treatment_prompt_identity": treatment_identity,
                "network_calls_performed": False,
            }
            completed.append(_write_cell(target, content))
            executed += 1
    planned = experiment_summary["planned_cells"]
    paired = summarize_native_cells(completed)
    content = {
        "schema_version": 1,
        "status": "complete" if len(completed) == planned else "partial",
        "taskset_fingerprint": taskset.fingerprint,
        "experiment_summary_sha256": experiment_summary["summary_sha256"],
        "planned_cells": planned,
        "completed_cells": len(completed),
        "native_invocations": sum(
            row["native_report"] is not None for row in completed
        ),
        **paired,
    }
    report = {**content, "summary_sha256": sha256_json(content)}
    name = "SUMMARY.json" if content["status"] == "complete" else "PROGRESS.json"
    destination = root / name
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != report:
            raise ContractError("frozen P1 native summary does not match replay")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report
