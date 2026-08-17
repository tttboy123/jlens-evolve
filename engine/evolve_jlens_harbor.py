"""Harbor Codex wrapper that seals an ATIF prediction before verification."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from continuous_ab import ABContractError


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(value: str, *, length: int, field: str) -> None:
    if len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise ABContractError(f"{field} must be a lowercase {length}-character hash")


def write_prediction_receipt(
    *,
    receipt_path: Path,
    trajectory_path: Path,
    round_id: str,
    arm: str,
    agent_program_sha256: str,
    baseline_contract_sha256: str,
    harness_revision: str,
) -> dict[str, Any]:
    """Seal the downloaded ATIF trajectory during Harbor's pre-verifier sync."""

    if arm not in {"baseline", "evolved"} or not round_id:
        raise ABContractError("Harbor prediction receipt identity is invalid")
    _require_hash(agent_program_sha256, length=64, field="agent_program_sha256")
    _require_hash(baseline_contract_sha256, length=64, field="baseline_contract_sha256")
    _require_hash(harness_revision, length=40, field="harness_revision")
    trajectory_path = trajectory_path.resolve()
    if not trajectory_path.is_file():
        raise ABContractError("Harbor ATIF trajectory is missing before verification")
    payload = {
        "schema_version": "1.0",
        "boundary": "harbor_agent_sync_before_verifier",
        "prediction_kind": "atif_trajectory_and_agent_end_environment_state",
        "prediction_frozen_before_evaluator": True,
        "round_id": round_id,
        "arm": arm,
        "agent_program_sha256": agent_program_sha256,
        "baseline_contract_sha256": baseline_contract_sha256,
        "harness_revision": harness_revision,
        "trajectory": {
            "path": str(trajectory_path),
            "bytes": trajectory_path.stat().st_size,
            "sha256": _sha256_file(trajectory_path),
        },
    }
    payload["integrity_sha256"] = _sha256_json(payload)
    receipt_path = receipt_path.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(receipt_path)
    return payload


def _parse_time(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ABContractError(f"Harbor {field} timestamp is missing")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ABContractError(f"Harbor {field} timestamp is invalid") from error


def validate_prediction_receipt(
    *,
    receipt_path: Path,
    trajectory_path: Path,
    result_path: Path,
    expected_round_id: str,
    expected_arm: str,
    expected_agent_program_sha256: str,
    expected_baseline_contract_sha256: str,
    expected_harness_revision: str,
) -> dict[str, Any]:
    """Verify the frozen trajectory and Harbor agent-before-verifier ordering."""

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    integrity = receipt.pop("integrity_sha256", None)
    if integrity != _sha256_json(receipt):
        raise ABContractError("Harbor prediction receipt was tampered")
    receipt["integrity_sha256"] = integrity
    expected = {
        "round_id": expected_round_id,
        "arm": expected_arm,
        "agent_program_sha256": expected_agent_program_sha256,
        "baseline_contract_sha256": expected_baseline_contract_sha256,
        "harness_revision": expected_harness_revision,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ABContractError("Harbor prediction receipt contract mismatch")
    trajectory_path = trajectory_path.resolve()
    trajectory = receipt.get("trajectory", {})
    if (
        not trajectory_path.is_file()
        or trajectory.get("bytes") != trajectory_path.stat().st_size
        or trajectory.get("sha256") != _sha256_file(trajectory_path)
    ):
        raise ABContractError("Harbor trajectory was tampered after prediction freeze")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    agent_finished = _parse_time(
        result.get("agent_execution", {}).get("finished_at"), field="agent finished"
    )
    verifier_started = _parse_time(
        result.get("verifier", {}).get("started_at"), field="verifier started"
    )
    if agent_finished > verifier_started:
        raise ABContractError("Harbor agent/verifier phase order is invalid")
    if receipt.get("boundary") != "harbor_agent_sync_before_verifier":
        raise ABContractError("Harbor prediction boundary is invalid")
    return receipt


try:
    from harbor.agents.installed.codex import Codex
    from harbor.models.agent.context import AgentContext
except ModuleNotFoundError:  # Local protocol tests do not install Harbor.
    Codex = None  # type: ignore[assignment,misc]
    AgentContext = Any  # type: ignore[assignment,misc]


if Codex is not None:

    class FrozenCodexAgent(Codex):  # type: ignore[misc,valid-type]
        """Pinned Harbor Codex agent with a pre-verifier prediction receipt."""

        def __init__(
            self,
            *args: Any,
            round_id: str,
            arm: str,
            agent_program_sha256: str,
            baseline_contract_sha256: str,
            harness_revision: str,
            **kwargs: Any,
        ) -> None:
            self._jlens_round_id = round_id
            self._jlens_arm = arm
            self._jlens_agent_program_sha256 = agent_program_sha256
            self._jlens_baseline_contract_sha256 = baseline_contract_sha256
            self._jlens_harness_revision = harness_revision
            super().__init__(*args, **kwargs)

        @staticmethod
        def name() -> str:
            return "evolve-jlens-frozen-codex"

        def populate_context_post_run(self, context: AgentContext) -> None:
            super().populate_context_post_run(context)
            write_prediction_receipt(
                receipt_path=self.logs_dir / "frozen-prediction.json",
                trajectory_path=self.logs_dir / "trajectory.json",
                round_id=self._jlens_round_id,
                arm=self._jlens_arm,
                agent_program_sha256=self._jlens_agent_program_sha256,
                baseline_contract_sha256=self._jlens_baseline_contract_sha256,
                harness_revision=self._jlens_harness_revision,
            )

else:

    class FrozenCodexAgent:  # pragma: no cover - fail-closed local placeholder
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("FrozenCodexAgent requires the pinned Harbor runtime")
