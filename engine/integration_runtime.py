"""Unified CLI vertical slice over the already-verified v0.2-v0.6 components."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from agent_code_runtime import run_agent_code_experiment
from agent_program import AgentProgram
from agent_program_runtime import run_agent_program_experiment
from evaluator_shadow import run_evaluator_shadow
from observer_runtime import run_observer_matrix
from plugin_envelope import ArtifactRef, EnvelopeLog, PluginEnvelope
from psi_runtime import run_psi_experiment

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "artifacts/v1.0.0/v0.7.0-integration/configs/experiment.json"


class IntegrationContractError(ValueError):
    """Raised when an operation id or component contract conflicts."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _component_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        name: (ROOT / value).resolve() for name, value in config["components"].items()
    }


def _path_hashes(path: Path) -> dict[str, str]:
    def portable_name(source: Path) -> str:
        try:
            return source.resolve().relative_to(ROOT).as_posix()
        except ValueError as error:
            raise IntegrationContractError(
                f"component input is outside project root: {source}"
            ) from error

    if path.is_file():
        return {portable_name(path): _sha256_file(path)}
    if path.is_dir():
        return {
            portable_name(child): _sha256_file(child)
            for child in sorted(item for item in path.rglob("*") if item.is_file())
        }
    raise IntegrationContractError(f"component input does not exist: {path}")


def _operation_contract(
    config_path: Path, config: dict[str, Any], components: dict[str, Path]
) -> tuple[str, dict[str, str]]:
    hashes = {"integration_config": _sha256_file(config_path)}
    for name, path in components.items():
        for source_path, digest in _path_hashes(path).items():
            hashes[f"component:{name}:{source_path}"] = digest
    for source in (
        "agent_program_runtime.py",
        "observer_runtime.py",
        "psi_runtime.py",
        "agent_code_runtime.py",
        "evaluator_shadow.py",
        "plugin_envelope.py",
        "integration_runtime.py",
    ):
        hashes[f"source:{source}"] = _sha256_file(ROOT / source)
    contract = {
        "operation_id": config["operation_id"],
        "system_version": config["system_version"],
        "hashes": hashes,
    }
    return hashlib.sha256(_canonical_json(contract).encode("utf-8")).hexdigest(), hashes


def _ref(path: Path, role: str) -> ArtifactRef:
    return ArtifactRef.from_path(path, role=role)


def _envelope(
    *,
    operation_id: str,
    plugin_id: str,
    plugin_version: str,
    authority: str,
    semantic_hash: str,
    config_hash: str,
    result_path: Path,
    evidence_path: Path,
    candidate_path: Path | None = None,
    active_path: Path | None = None,
    used_for_admission: bool = False,
) -> PluginEnvelope:
    return PluginEnvelope.create(
        operation_id=operation_id,
        plugin_id=plugin_id,
        plugin_version=plugin_version,
        authority=authority,
        status="completed",
        input_hashes={"semantic_result": semantic_hash},
        config_hashes={"component_config": config_hash},
        candidate_ref=(
            _ref(candidate_path, f"{plugin_id}_candidate") if candidate_path else None
        ),
        active_ref=_ref(active_path, f"{plugin_id}_active") if active_path else None,
        result_refs=(_ref(result_path, f"{plugin_id}_result"),),
        evidence_refs=(_ref(evidence_path, f"{plugin_id}_evidence"),),
        used_for_admission=used_for_admission,
        error=None,
    )


def run_integration(*, config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("system_version") != "0.7.0":
        raise IntegrationContractError("integration must use system_version 0.7.0")
    if config.get("model_calls") != 0 or config.get("network_calls") != 0:
        raise IntegrationContractError("integration POC cannot call model or network")
    if config.get("global_skill_installs") != 0:
        raise IntegrationContractError("integration cannot install global Skills")
    components = _component_paths(config)
    contract_hash, contract_inputs = _operation_contract(
        config_path, config, components
    )
    state_path = output_dir / "integration-state.json"
    result_path = output_dir / "result.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("operation_id") != config["operation_id"]
            or state.get("contract_hash") != contract_hash
        ):
            raise IntegrationContractError("operation contract conflict")
        if (
            not result_path.is_file()
            or _sha256_file(result_path) != state["result_sha256"]
        ):
            raise IntegrationContractError(
                "idempotent operation result is missing or changed"
            )
        return {
            **json.loads(result_path.read_text(encoding="utf-8")),
            "idempotent_replay": True,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    component_root = output_dir / "components"
    agent_configs = components["agent_program_configs"]
    agent_output = component_root / "agent-program"
    agent_result = run_agent_program_experiment(
        baseline_path=agent_configs / "baseline_agent_program.json",
        registry_path=agent_configs / "component_registry.json",
        proposals_path=agent_configs / "replay_proposals.json",
        experiment_path=agent_configs / "experiment.json",
        output_dir=agent_output,
    )
    observer_formal_output = component_root / "observer-formal"
    observer_formal = run_observer_matrix(
        observer_config_path=components["observer_config"],
        agent_config_dir=agent_configs,
        output_dir=observer_formal_output,
        replays_per_mode=int(config["observer_replays"]),
    )
    observer_failure_output = component_root / "observer-failure"
    observer_failure = run_observer_matrix(
        observer_config_path=components["observer_config"],
        agent_config_dir=agent_configs,
        output_dir=observer_failure_output,
        replays_per_mode=1,
        inject_failure_mode=str(config["observer_failure_injection"]),
    )
    psi_output = component_root / "psi"
    psi_result = run_psi_experiment(
        config_path=components["psi_config"],
        candidate_path=components["psi_candidates"],
        output_dir=psi_output,
    )
    code_output = component_root / "agent-code"
    code_result = run_agent_code_experiment(
        config_path=components["code_config"],
        evaluator_path=components["code_evaluator"],
        candidate_manifest_path=components["code_candidates"],
        output_dir=code_output,
    )
    shadow_output = component_root / "evaluator-shadow"
    shadow_result = run_evaluator_shadow(
        config_path=components["shadow_config"],
        evaluator_path=components["shadow_evaluators"],
        corpus_path=components["shadow_corpus"],
        output_dir=shadow_output,
    )

    operation_id = str(config["operation_id"])
    agent_config_hash = _sha256_file(agent_configs / "experiment.json")
    candidate_archive = agent_output / "candidate_archive.jsonl"
    active_program = agent_output / "active_agent_program.json"
    agent_evidence = agent_output / "evidence.json"
    envelopes = [
        _envelope(
            operation_id=operation_id,
            plugin_id="runtime",
            plugin_version="0.2.0",
            authority="execute",
            semantic_hash=agent_result["outcome_fingerprint"],
            config_hash=agent_config_hash,
            result_path=agent_output / "result.json",
            evidence_path=agent_evidence,
        ),
        _envelope(
            operation_id=operation_id,
            plugin_id="optimizer",
            plugin_version="0.2.0",
            authority="propose",
            semantic_hash=agent_result["outcome_fingerprint"],
            config_hash=agent_config_hash,
            candidate_path=candidate_archive,
            result_path=agent_output / "result.json",
            evidence_path=agent_evidence,
        ),
        _envelope(
            operation_id=operation_id,
            plugin_id="observer-formal",
            plugin_version="0.3.0",
            authority="observe",
            semantic_hash=observer_formal["matrix_fingerprint"],
            config_hash=_sha256_file(components["observer_config"]),
            result_path=observer_formal_output / "matrix-result.json",
            evidence_path=observer_formal_output / "evidence.json",
        ),
        _envelope(
            operation_id=operation_id,
            plugin_id="observer-failure",
            plugin_version="0.3.0",
            authority="observe",
            semantic_hash=observer_failure["matrix_fingerprint"],
            config_hash=_sha256_file(components["observer_config"]),
            result_path=observer_failure_output / "matrix-result.json",
            evidence_path=observer_failure_output / "evidence.json",
        ),
        _envelope(
            operation_id=operation_id,
            plugin_id="skill-registry",
            plugin_version="0.4.0",
            authority="persist",
            semantic_hash=psi_result["experiment_fingerprint"],
            config_hash=_sha256_file(components["psi_config"]),
            candidate_path=psi_output
            / "registry/skills/record-cleaning-invariants-v2/SKILL.md",
            result_path=psi_output / "result.json",
            evidence_path=psi_output / "evidence.json",
        ),
        _envelope(
            operation_id=operation_id,
            plugin_id="mutation-archive",
            plugin_version="0.5.0",
            authority="persist",
            semantic_hash=code_result["experiment_fingerprint"],
            config_hash=_sha256_file(components["code_config"]),
            candidate_path=code_output / "archive/records.jsonl",
            result_path=code_output / "result.json",
            evidence_path=code_output / "evidence.json",
        ),
        _envelope(
            operation_id=operation_id,
            plugin_id="evaluator-shadow",
            plugin_version="0.6.0",
            authority="observe",
            semantic_hash=shadow_result["experiment_fingerprint"],
            config_hash=_sha256_file(components["shadow_config"]),
            candidate_path=shadow_output / "review-proposal.json",
            result_path=shadow_output / "result.json",
            evidence_path=shadow_output / "evidence.json",
        ),
        _envelope(
            operation_id=operation_id,
            plugin_id="admission-gate",
            plugin_version="0.7.0",
            authority="admit",
            semantic_hash=agent_result["outcome_fingerprint"],
            config_hash=agent_config_hash,
            candidate_path=candidate_archive,
            active_path=active_program,
            result_path=agent_output / "result.json",
            evidence_path=agent_evidence,
            used_for_admission=True,
        ),
    ]
    log = EnvelopeLog(output_dir / "operation-log.jsonl")
    for envelope in envelopes:
        log.append(envelope)
    serialized_envelopes = [envelope.to_dict() for envelope in log.read()]
    component_decisions = {
        "agent_program": agent_result["decision"],
        "observer_formal": observer_formal["decision"],
        "observer_failure": observer_failure["decision"],
        "psi": psi_result["decision"],
        "agent_code": code_result["decision"],
        "evaluator_shadow": shadow_result["decision"],
    }
    active_envelopes = [
        row for row in serialized_envelopes if row["active_ref"] is not None
    ]
    contract_checks = {
        "all_envelopes_round_trip": all(
            PluginEnvelope.from_dict(row).to_dict() == row
            for row in serialized_envelopes
        ),
        "all_authorities_present": {row["authority"] for row in serialized_envelopes}
        == {"execute", "observe", "propose", "persist", "admit"},
        "only_admission_publishes_active": len(active_envelopes) == 1
        and active_envelopes[0]["authority"] == "admit",
        "all_components_accepted": set(component_decisions.values()) == {"accepted"},
        "observer_runtime_equivalent": all(
            observer_formal["mechanism_checks"].values()
        ),
        "observer_failure_isolated": observer_failure["failure_injection"]
        == {"mode": config["observer_failure_injection"], "isolated": True},
        "observer_never_admission": all(
            row["used_for_admission"] is False
            for row in serialized_envelopes
            if row["authority"] == "observe"
        ),
        "psi_inactive_and_local": psi_result["candidate_status"] == "transfer_verified"
        and psi_result["claims"]["global_skill_installs"] == 0,
        "code_rollback_to_parent": code_result["rollback"]["performed"]
        and code_result["rollback"]["final_active_sha256"]
        == code_result["parent"]["source_sha256"],
        "shadow_governance_preserved": shadow_result["active_evaluator_before"]
        == shadow_result["active_evaluator_after"]
        and shadow_result["review_proposal"]["activation_allowed"] is False,
        "active_program_matches_agent_admission": AgentProgram.from_path(
            active_program
        ).sha256
        == agent_result["final"]["program_hash"],
    }
    accepted = all(contract_checks.values())
    semantic_components = {
        "agent_program": agent_result["outcome_fingerprint"],
        "observer_formal": observer_formal["matrix_fingerprint"],
        "observer_failure": observer_failure["matrix_fingerprint"],
        "psi": psi_result["experiment_fingerprint"],
        "agent_code": code_result["experiment_fingerprint"],
        "evaluator_shadow": shadow_result["experiment_fingerprint"],
    }
    stable = {
        "contract_hash": contract_hash,
        "semantic_components": semantic_components,
        "component_decisions": component_decisions,
        "contract_checks": contract_checks,
        "active_agent_program_hash": agent_result["final"]["program_hash"],
        "observer_incremental": observer_formal["jlens_incremental"]["conclusion"],
        "authorities_present": sorted(
            {row["authority"] for row in serialized_envelopes}
        ),
    }
    result = {
        "schema_version": 1,
        "stage": "v0.7.0-integration",
        "operation_id": operation_id,
        **stable,
        "decision": "accepted" if accepted else "rejected",
        "envelopes": serialized_envelopes,
        "observer": {
            "incremental": observer_formal["jlens_incremental"]["conclusion"],
            "failure_isolated": observer_failure["failure_injection"]["isolated"],
            "used_for_admission": False,
        },
        "psi": {
            "candidate_status": psi_result["candidate_status"],
            "active": False,
        },
        "agent_code": {
            "verified_candidate": code_result["verified_candidate_id"],
            "rollback_to_parent": contract_checks["code_rollback_to_parent"],
        },
        "evaluator_shadow": {
            "active_changed": False,
            "auto_promoted": False,
            "review_candidate": shadow_result["review_proposal"]["candidate_id"],
        },
        "experiment_fingerprint": hashlib.sha256(
            _canonical_json(stable).encode("utf-8")
        ).hexdigest(),
        "idempotent_replay": False,
        "claims": {
            "global_skill_installs": 0,
            "model_calls": 0,
            "network_calls": 0,
            "model_weights_frozen": True,
            "production_ready": False,
        },
    }
    _atomic_json(output_dir / "contract-inputs.json", contract_inputs)
    _atomic_json(
        output_dir / "evidence.json",
        {
            "contract_checks": contract_checks,
            "semantic_components": semantic_components,
        },
    )
    _atomic_json(result_path, result)
    _atomic_json(
        state_path,
        {
            "operation_id": operation_id,
            "contract_hash": contract_hash,
            "result_sha256": _sha256_file(result_path),
            "status": result["decision"],
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_integration(config_path=args.config, output_dir=args.output)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "operation_id": result["operation_id"],
                "idempotent_replay": result["idempotent_replay"],
                "experiment_fingerprint": result["experiment_fingerprint"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
