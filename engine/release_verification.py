"""Reconstruct every v1.0 release gate from direct staged evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from artifact_verifier import verify_manifest

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "artifacts/v1.0.0/v1.0.0-release/configs/release.json"


class ReleaseVerificationError(ValueError):
    """Raised when the final release verification contract is malformed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _run_json(command: list[str], *, output_file: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONHASHSEED": "0",
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(completed.stdout, encoding="utf-8")
    return json.loads(completed.stdout)


def _stage_evidence(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    root = ROOT / "artifacts/v1.0.0"
    kernel = _json(root / "v0.1.0-kernel/evaluations/sqlite-readonly-audit.json")
    agent = _json(root / "v0.2.0-agent-program/evaluations/pass3-summary.json")
    observer = _json(root / "v0.3.0-jlens-observer/evaluations/matched-summary.json")
    psi = _json(root / "v0.4.0-psi-skill-library/evaluations/pass3-summary.json")
    code = _json(root / "v0.5.0-agent-code-mutation/evaluations/pass3-summary.json")
    shadow = _json(root / "v0.6.0-evaluator-shadow/evaluations/pass3-summary.json")
    integration = _json(root / "v0.7.0-integration/evaluations/pass3-summary.json")
    hardening = _json(root / "v0.8.0-hardening/evaluations/pass3-summary.json")
    rc = _json(root / "v0.9.0-release-candidate/evaluations/pass3-summary.json")
    unsafe = {
        name: verdict
        for name, verdict in code["candidate_verdicts"].items()
        if name.startswith("unsafe-")
    }
    stages = {
        "v0.1.0-kernel": {
            "accepted": kernel["integrity_check"] == "ok"
            and kernel["distinct_outcome_fingerprints"] == 1
            and len(kernel["runs"]) == 3
            and {row["status"] for row in kernel["runs"]} == {"completed"},
            "integrity": kernel["integrity_check"],
            "runs": len(kernel["runs"]),
            "outcome_fingerprints": kernel["distinct_outcome_fingerprints"],
        },
        "v0.2.0-agent-program": {
            "accepted": agent["baseline"]["public_passed_by_seed"]
            == {"11": 3, "22": 3, "33": 3}
            and agent["final"]["public_passed_by_seed"]
            == {"11": 13, "22": 13, "33": 13}
            and agent["final"]["sealed_passed_by_seed"] == {"11": 6, "22": 6, "33": 6}
            and agent["final"]["program_hash"] == config["expected_agent_program_hash"],
            "baseline_public": [3, 3, 3],
            "final_public": [13, 13, 13],
            "final_sealed": [6, 6, 6],
            "program_hash": agent["final"]["program_hash"],
        },
        "v0.3.0-jlens-observer": {
            "accepted": observer["stage_decision"] == "accepted"
            and observer["failure_injection"]["isolated"]
            and observer["jlens_incremental"]["conclusion"]
            == config["expected_jlens_conclusion"],
            "conclusion": observer["jlens_incremental"]["conclusion"],
            "advantage": observer["jlens_incremental"]["advantage"],
            "failure_isolated": observer["failure_injection"]["isolated"],
        },
        "v0.4.0-psi-skill-library": {
            "accepted": psi["decision"] == "accepted"
            and psi["candidate_status"] == "transfer_verified"
            and psi["legacy_candidate_status"] == "rejected"
            and set(psi["targets"])
            == {"payout-record-cleaning-v1", "refund-record-cleaning-v1"}
            and all(
                target["transfer_sealed_mean"] == 1.0
                and target["sealed_noninferior_seeds"] == 3
                for target in psi["targets"].values()
            ),
            "candidate_status": psi["candidate_status"],
            "legacy_status": psi["legacy_candidate_status"],
            "targets": sorted(psi["targets"]),
            "active": False,
        },
        "v0.5.0-agent-code-mutation": {
            "accepted": code["decision"] == "accepted"
            and code["rollback"]["performed"]
            and len(unsafe) == 3
            and all(not verdict["executed"] for verdict in unsafe.values()),
            "unsafe_not_executed": sorted(unsafe),
            "rollback_performed": code["rollback"]["performed"],
            "verified_candidate": code["verified_candidate_id"],
        },
        "v0.6.0-evaluator-shadow": {
            "accepted": shadow["decision"] == "accepted"
            and not shadow["active_evaluator_changed"]
            and not shadow["review_proposal"]["activation_allowed"]
            and not shadow["review_proposal"]["auto_promoted"],
            "active_changed": shadow["active_evaluator_changed"],
            "review_candidate": shadow["review_proposal"]["candidate_id"],
            "auto_promoted": shadow["review_proposal"]["auto_promoted"],
        },
        "v0.7.0-integration": {
            "accepted": integration["all_contract_gates_passed"]
            and integration["formal_runs"] == 3
            and len(set(integration["experiment_fingerprints"])) == 1,
            "formal_runs": integration["formal_runs"],
            "all_contract_gates": integration["all_contract_gates_passed"],
            "historical_fingerprint": integration["experiment_fingerprints"][0],
        },
        "v0.8.0-hardening": {
            "accepted": hardening["all_contract_checks_passed"]
            and hardening["formal_runs"] == 3
            and len(set(hardening["experiment_fingerprints"])) == 1,
            "formal_runs": hardening["formal_runs"],
            "all_contract_checks": hardening["all_contract_checks_passed"],
            "fingerprint": hardening["experiment_fingerprints"][0],
        },
        "v0.9.0-release-candidate": {
            "accepted": rc["all_contract_checks_passed"]
            and rc["formal_matrices"] == 3
            and rc["independent_durable_operations"] == 9
            and rc["integration_fingerprint"]
            == config["expected_integration_fingerprint"]
            and set(rc["experiment_fingerprints"])
            == {config["expected_rc_evidence_fingerprint"]},
            "formal_matrices": rc["formal_matrices"],
            "operations": rc["independent_durable_operations"],
            "integration_fingerprint": rc["integration_fingerprint"],
            "rc_fingerprint": rc["experiment_fingerprints"][0],
            "clean_room_core_tests": rc["clean_room_core_tests"],
        },
    }
    return stages


def _documents(config: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for relative in config["required_documents"]:
        path = (ROOT / relative).resolve()
        content = path.read_text(encoding="utf-8") if path.is_file() else ""
        rows[relative] = {
            "exists": path.is_file(),
            "nonempty": len(content) > 200,
            "sha256": _sha256_file(path) if path.is_file() else None,
        }
    combined = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in config["required_documents"]
        if (ROOT / relative).is_file()
    )
    consistency = {
        "v1_current": "v1.0.0-release" in combined,
        "weights_frozen": "模型权重" in combined and "冻结" in combined,
        "jlens_observer": "JLens" in combined and "Observer" in combined,
        "no_auto_promotion": "自动晋升" in combined or "auto promotion" in combined,
        "production_limit": "production" in combined and "不是" in combined,
    }
    return {
        "documents": rows,
        "consistency": consistency,
        "valid": all(row["exists"] and row["nonempty"] for row in rows.values())
        and all(consistency.values()),
    }


def run_release_verification(*, config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _json(config_path)
    if config.get("system_version") != "1.0.0" or config.get("schema_version") != 1:
        raise ReleaseVerificationError("unsupported release config")
    if any(
        config.get(field) != 0
        for field in ("model_calls", "network_calls", "global_skill_installs")
    ):
        raise ReleaseVerificationError(
            "release cannot use model network or global install"
        )
    manifest_path = (ROOT / config["artifact_manifest"]).resolve()
    protected_paths = [manifest_path]
    protected_paths.extend(
        (ROOT / path).resolve() for path in config["required_documents"]
    )
    protected_before = {str(path): _sha256_file(path) for path in protected_paths}
    manifest = verify_manifest(manifest_path)
    stages = _stage_evidence(config)
    documents = _documents(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    cli_output = output_dir / "cli"
    service = str(ROOT / "evolve_service.py")
    rc_config = str(
        ROOT / "artifacts/v1.0.0/v0.9.0-release-candidate/configs/experiment.json"
    )
    cli_run_dir = output_dir / "cli-run"
    cli_run = _run_json(
        [
            sys.executable,
            service,
            "run",
            "--config",
            rc_config,
            "--output",
            str(cli_run_dir),
        ],
        output_file=cli_output / "run.stdout.json",
    )
    cli_inspect = _run_json(
        [
            sys.executable,
            service,
            "inspect",
            "--result",
            str(cli_run_dir / "result.json"),
        ],
        output_file=cli_output / "inspect.stdout.json",
    )
    cli_verify = _run_json(
        [
            sys.executable,
            service,
            "verify",
            "--manifest",
            str(manifest_path),
        ],
        output_file=cli_output / "verify.stdout.json",
    )
    rollback_plans = {}
    for kind in ("agent-program", "agent-code", "skill", "evaluator"):
        rollback_plans[kind] = _run_json(
            [
                sys.executable,
                service,
                "rollback-plan",
                "--kind",
                kind,
                "--evidence-root",
                str(ROOT / "artifacts/v1.0.0"),
            ],
            output_file=cli_output / f"rollback-{kind}.stdout.json",
        )
    protected_after = {str(path): _sha256_file(path) for path in protected_paths}
    cli = {
        "run": cli_run,
        "inspect": cli_inspect,
        "verify": cli_verify,
        "rollback_plans": rollback_plans,
    }
    contract_checks = {
        "all_stage_evidence_accepted": set(stages) == set(config["required_stages"])
        and all(stage["accepted"] for stage in stages.values()),
        "cli_inspect_passed": cli_inspect["decision"] == "accepted"
        and cli_inspect["checks_passed"] == cli_inspect["checks_total"],
        "cli_rollback_plans_non_mutating": all(
            not plan["applied"] and plan["requires_new_operation"]
            for plan in rollback_plans.values()
        ),
        "cli_run_passed": cli_run["decision"] == "accepted"
        and cli_run["experiment_fingerprint"]
        == config["expected_current_rc_fingerprint"],
        "cli_verify_passed": cli_verify["valid"] and not cli_verify["failures"],
        "documentation_consistent": documents["valid"],
        "manifest_read_only_and_valid": manifest["valid"]
        and not manifest["failures"]
        and protected_before == protected_after,
        "model_network_and_global_install_zero": config["model_calls"] == 0
        and config["network_calls"] == 0
        and config["global_skill_installs"] == 0,
        "recovery_rollback_and_clean_room_evidence_present": stages[
            "v0.5.0-agent-code-mutation"
        ]["rollback_performed"]
        and stages["v0.8.0-hardening"]["all_contract_checks"]
        and stages["v0.9.0-release-candidate"]["clean_room_core_tests"].endswith(
            "passed"
        ),
    }
    claims = {
        "direct_llm_generalization_proven": False,
        "distributed_exactly_once": False,
        "global_skill_installs": 0,
        "jlens_incremental_gain_proven": False,
        "model_calls": 0,
        "model_weights_frozen": True,
        "network_calls": 0,
        "production_deployed": False,
    }
    stable = {
        "contract_checks": contract_checks,
        "stage_evidence": stages,
        "manifest_fingerprint": manifest["verification_fingerprint"],
        "cli": {
            "run": {
                key: cli_run[key] for key in ("decision", "experiment_fingerprint")
            },
            "inspect": {
                key: cli_inspect[key]
                for key in ("decision", "checks_passed", "checks_total")
            },
            "verify": cli_verify["verification_fingerprint"],
            "rollback": {
                kind: {
                    "applied": plan["applied"],
                    "requires_new_operation": plan["requires_new_operation"],
                }
                for kind, plan in rollback_plans.items()
            },
        },
        "documents": documents["consistency"],
        "claims": claims,
    }
    result = {
        "schema_version": 1,
        "stage": "v1.0.0-release",
        "decision": "accepted" if all(contract_checks.values()) else "rejected",
        "contract_checks": contract_checks,
        "stage_evidence": stages,
        "manifest": manifest,
        "cli": cli,
        "documentation": documents,
        "protected_hashes_unchanged": protected_before == protected_after,
        "claims": claims,
        "experiment_fingerprint": hashlib.sha256(
            _canonical_json(stable).encode("utf-8")
        ).hexdigest(),
    }
    _atomic_json(
        output_dir / "evidence.json",
        {
            "contract_checks": contract_checks,
            "stage_evidence": stages,
            "protected_hashes_unchanged": protected_before == protected_after,
        },
    )
    _atomic_json(output_dir / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_release_verification(config_path=args.config, output_dir=args.output)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "experiment_fingerprint": result["experiment_fingerprint"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0 if result["decision"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
