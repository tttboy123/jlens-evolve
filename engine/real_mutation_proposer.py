"""Real provider callback, profile materialization, and rollback verification."""

from __future__ import annotations

import difflib
import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_arm_runner import profile_tree_hash
from mutation_proposer import (
    InactiveChangeSet,
    MutationContractError,
    MutationProposer,
    MutationRequest,
    ProposalResult,
)

_USE_MODEL_CALL = object()

_PATH_RULE_PREFIX = {
    "prompt": "AGENTS.md",
    "skills": ".agents/skills/",
    "policy": ".codex/evolution-policy.json",
    "router": ".codex/evolution-policy.json",
    "memory_policy": ".codex/evolution-policy.json",
    "constrained_harness_code": ".codex/harness/",
}


class RealProposalError(MutationContractError):
    """Raised when provider output cannot become a reversible AgentProgram."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _safe_target(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise RealProposalError("mutation path escapes candidate profile")
    return target


def _diff(path: str, before: str | None, after: str | None) -> str:
    return "".join(
        difflib.unified_diff(
            [] if before is None else before.splitlines(keepends=True),
            [] if after is None else after.splitlines(keepends=True),
            fromfile="/dev/null" if before is None else f"a/{path}",
            tofile="/dev/null" if after is None else f"b/{path}",
        )
    )


class RealMutationProposerAdapter:
    """Turn one real LLM JSON response into an inactive, executable profile."""

    def __init__(
        self,
        *,
        profile_roots: dict[str, Path],
        output_root: Path,
        provider: dict[str, str],
        model_call: Callable[[str], str],
        repair_call: Callable[[str], str] | None | object = _USE_MODEL_CALL,
        reserve_call: Callable[[str], bool] | None = None,
        complete_call: Callable[[str, str], None] | None = None,
    ) -> None:
        if set(provider) != {"platform", "model"} or not all(provider.values()):
            raise RealProposalError("provider identity is invalid")
        self.profile_roots = {
            digest: root.resolve() for digest, root in profile_roots.items()
        }
        self.output_root = output_root.resolve()
        self.provider = dict(provider)
        self.model_call = model_call
        self.repair_call = model_call if repair_call is _USE_MODEL_CALL else repair_call
        self.reserve_call = reserve_call or (lambda _reservation_id: True)
        self.complete_call = complete_call or (
            lambda _reservation_id, _response_sha256: None
        )
        self.validator = MutationProposer()
        self._candidate_roots: dict[str, Path] = {}
        self._rollback_receipts: dict[str, dict[str, Any]] = {}

    def _parent_profile_payload(self, parent_sha256: str) -> dict[str, Any]:
        root = self.profile_root(parent_sha256)
        files = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _sha256_file(path),
                    "content": path.read_text(encoding="utf-8"),
                }
            )
        return {"tree_sha256": parent_sha256, "files": files}

    def profile_root(self, agent_program_sha256: str) -> Path:
        root = self._candidate_roots.get(
            agent_program_sha256
        ) or self.profile_roots.get(agent_program_sha256)
        if root is None or not root.is_dir():
            raise RealProposalError("AgentProgram profile root is missing")
        if profile_tree_hash(root) != agent_program_sha256:
            raise RealProposalError("AgentProgram profile tree hash mismatch")
        return root

    @staticmethod
    def _current_text(target: Path) -> str | None:
        if not target.exists():
            return None
        if not target.is_file() or target.is_symlink():
            raise RealProposalError("mutation target must be a regular profile file")
        return target.read_text(encoding="utf-8")

    def _materialize_candidate(
        self,
        *,
        request: MutationRequest,
        proposed: InactiveChangeSet,
        proposal_root: Path,
    ) -> InactiveChangeSet:
        parent_root = self.profile_root(request.parent_agent_program_sha256)
        with tempfile.TemporaryDirectory(prefix="evolve-real-proposal-") as temporary:
            temporary_root = Path(temporary)
            candidate_root = temporary_root / "candidate-profile"
            shutil.copytree(parent_root, candidate_root)
            forward_diff = []
            rollback_diff = []
            for forward, reverse in zip(
                proposed.operations, proposed.rollback_operations, strict=True
            ):
                target = _safe_target(candidate_root, forward["path"])
                before = self._current_text(target)
                if forward["op"] == "create":
                    if before is not None or reverse["op"] != "delete":
                        raise RealProposalError(
                            "create mutation does not have a valid delete rollback"
                        )
                else:
                    if before is None or reverse["after"] != before:
                        raise RealProposalError(
                            "replace rollback does not match the parent profile"
                        )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(forward["after"], encoding="utf-8")
                forward_diff.append(_diff(forward["path"], before, forward["after"]))
                rollback_diff.append(_diff(forward["path"], forward["after"], before))
            candidate_hash = profile_tree_hash(candidate_root)
            if candidate_hash == request.parent_agent_program_sha256:
                raise RealProposalError("mutation produced a no-op AgentProgram")
            admitted = replace(proposed, candidate_agent_program_sha256=candidate_hash)

            rollback_probe = temporary_root / "rollback-probe"
            shutil.copytree(candidate_root, rollback_probe)
            for forward, reverse in zip(
                reversed(admitted.operations),
                reversed(admitted.rollback_operations),
                strict=True,
            ):
                target = _safe_target(rollback_probe, reverse["path"])
                if self._current_text(target) != forward["after"]:
                    raise RealProposalError("candidate changed before rollback probe")
                if reverse["op"] == "delete":
                    target.unlink()
                else:
                    target.write_text(reverse["after"], encoding="utf-8")
            rollback_verified = (
                profile_tree_hash(rollback_probe) == request.parent_agent_program_sha256
            )
            if not rollback_verified:
                raise RealProposalError("rollback did not restore parent tree hash")

            destination = proposal_root / "candidate-profile"
            if destination.exists():
                if profile_tree_hash(destination) != candidate_hash:
                    raise RealProposalError("persisted candidate profile is immutable")
            else:
                shutil.copytree(candidate_root, destination)
            forward_path = proposal_root / "forward.patch"
            rollback_path = proposal_root / "rollback.patch"
            forward_path.write_text("".join(forward_diff), encoding="utf-8")
            rollback_path.write_text("".join(reversed(rollback_diff)), encoding="utf-8")
            receipt = {
                "schema_version": 1,
                "parent_agent_program_sha256": request.parent_agent_program_sha256,
                "candidate_agent_program_sha256": candidate_hash,
                "forward_patch_sha256": _sha256_file(forward_path),
                "rollback_patch_sha256": _sha256_file(rollback_path),
                "verified": True,
                "production_applied": False,
                "global_skill_installed": False,
            }
            _write_json(proposal_root / "ROLLBACK-VERIFICATION.json", receipt)
            _write_json(proposal_root / "ADMITTED-CHANGESET.json", admitted.to_dict())
            self._candidate_roots[candidate_hash] = destination
            self.profile_roots[candidate_hash] = destination
            self._rollback_receipts[candidate_hash] = receipt
            return admitted

    def _validate_and_materialize(
        self,
        *,
        raw: str,
        request: MutationRequest,
        proposal_root: Path,
    ) -> InactiveChangeSet:
        proposed = self.validator.validate_response(
            raw,
            expected_parent_sha256=request.parent_agent_program_sha256,
            allowed_hypothesis_ids=set(request.hypothesis_ids),
            frozen_native_evaluator_epoch=request.native_evaluator_epoch,
        )
        if proposed.proposer != self.provider:
            raise RealProposalError("response proposer identity mismatch")
        if proposed.surface != request.surface:
            raise RealProposalError("response mutation surface mismatch")
        return self._materialize_candidate(
            request=request, proposed=proposed, proposal_root=proposal_root
        )

    def _resume(
        self, request: MutationRequest, proposal_root: Path
    ) -> ProposalResult | None:
        admitted_path = proposal_root / "ADMITTED-CHANGESET.json"
        if not admitted_path.is_file():
            return None
        admitted = self.validator.validate_response(
            admitted_path.read_text(encoding="utf-8"),
            expected_parent_sha256=request.parent_agent_program_sha256,
            allowed_hypothesis_ids=set(request.hypothesis_ids),
            frozen_native_evaluator_epoch=request.native_evaluator_epoch,
        )
        candidate_root = proposal_root / "candidate-profile"
        if profile_tree_hash(candidate_root) != admitted.candidate_agent_program_sha256:
            raise RealProposalError("persisted candidate profile was tampered")
        receipt = json.loads(
            (proposal_root / "ROLLBACK-VERIFICATION.json").read_text(encoding="utf-8")
        )
        if (
            receipt.get("candidate_agent_program_sha256")
            != admitted.candidate_agent_program_sha256
        ):
            raise RealProposalError("persisted rollback receipt is mismatched")
        raw_paths = sorted(proposal_root.glob("raw-*.txt"))
        if not raw_paths or len(raw_paths) > 2:
            raise RealProposalError("persisted raw proposer evidence is incomplete")
        self._candidate_roots[admitted.candidate_agent_program_sha256] = candidate_root
        self.profile_roots[admitted.candidate_agent_program_sha256] = candidate_root
        self._rollback_receipts[admitted.candidate_agent_program_sha256] = receipt
        return ProposalResult(
            changeset=admitted,
            repairs_used=len(raw_paths) - 1,
            raw_responses=tuple(path.read_text(encoding="utf-8") for path in raw_paths),
        )

    def propose(self, request: MutationRequest, generation: int) -> ProposalResult:
        proposal_root = (
            self.output_root / f"generation-{generation}" / request.request_id
        )
        proposal_root.mkdir(parents=True, exist_ok=True)
        resumed = self._resume(request, proposal_root)
        if resumed is not None:
            return resumed
        path_example = {
            "prompt": ("AGENTS.md", "create/replace"),
            "skills": (".agents/skills/<skill-id>/SKILL.md", "create/replace"),
            "policy": (".codex/evolution-policy.json", "create/replace"),
            "router": (".codex/evolution-policy.json", "create/replace"),
            "memory_policy": (".codex/evolution-policy.json", "create/replace"),
            "constrained_harness_code": (
                ".codex/harness/<file>.py",
                "create/replace",
            ),
        }[request.surface]
        prompt = json.dumps(
            {
                "generation": generation,
                "mutation_contract": json.loads(request.prompt(self.provider)),
                "materialization_rule": (
                    "candidate_agent_program_sha256 is a 64-zero placeholder; the "
                    "controller recomputes it from the materialized profile"
                ),
                "parent_profile": self._parent_profile_payload(
                    request.parent_agent_program_sha256
                ),
                "response_schema": {
                    "schema_version": 1,
                    "changeset_id": request.request_id,
                    "status": "inactive",
                    "parent_agent_program_sha256": request.parent_agent_program_sha256,
                    "candidate_agent_program_sha256": "0" * 64,
                    "hypothesis_ids": list(request.hypothesis_ids),
                    "surface": request.surface,
                    "operations": [
                        {
                            "op": "replace",
                            "path": path_example[0],
                            "after": "new file content",
                        }
                    ],
                    "rollback_operations": [
                        {
                            "op": "replace",
                            "path": path_example[0],
                            "after": "original file content",
                        }
                    ],
                    "path_rule": (
                        f"operation path must start with "
                        f"{_PATH_RULE_PREFIX[request.surface]!r}; "
                        f"forward ops={path_example[1]}; "
                        "rollback ops=replace/delete; every operation object "
                        "must contain exactly op, path and after; when op is "
                        "delete, after must be null"
                    ),
                    "proposer": dict(self.provider),
                    "native_evaluator_epoch": request.native_evaluator_epoch,
                    "native_evaluator_authority": "external_fixed",
                    "auto_apply": False,
                    "production_promotion_allowed": False,
                },
                "response_rule": "Return one JSON object only.",
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        (proposal_root / "PROMPT.txt").write_text(prompt, encoding="utf-8")
        _write_json(
            proposal_root / "REQUEST.json",
            {
                "generation": generation,
                "request_id": request.request_id,
                "parent_agent_program_sha256": request.parent_agent_program_sha256,
                "surface": request.surface,
                "hypothesis_ids": list(request.hypothesis_ids),
            },
        )
        raw_responses = []
        call = self.model_call
        current_prompt = prompt
        for attempt in (1, 2):
            raw_path = proposal_root / f"raw-{attempt}.txt"
            reservation_id = (
                f"proposer|g{generation}|{request.request_id}|attempt-{attempt}"
            )
            if raw_path.is_file():
                raw = raw_path.read_text(encoding="utf-8")
            else:
                if self.reserve_call(reservation_id) is not True:
                    raise RealProposalError(
                        "real proposer call is already reserved; reconcile its raw "
                        "response before retry"
                    )
                raw = call(current_prompt)
                if not isinstance(raw, str):
                    raise RealProposalError("provider response must be text")
                raw_path.write_text(raw, encoding="utf-8")
                self.complete_call(
                    reservation_id, hashlib.sha256(raw.encode()).hexdigest()
                )
            if not isinstance(raw, str) or not raw.strip():
                error: Exception = RealProposalError("provider response is empty")
            else:
                error = RealProposalError("provider response was not admitted")
            raw_responses.append(raw)
            try:
                admitted = self._validate_and_materialize(
                    raw=raw, request=request, proposal_root=proposal_root
                )
                return ProposalResult(
                    changeset=admitted,
                    repairs_used=attempt - 1,
                    raw_responses=tuple(raw_responses),
                )
            except (
                MutationContractError,
                OSError,
                UnicodeError,
                json.JSONDecodeError,
            ) as caught:
                error = caught
            if attempt == 2 or self.repair_call is None:
                raise error
            current_prompt = json.dumps(
                {
                    "original_request": json.loads(prompt),
                    "invalid_response": raw,
                    "validation_or_materialization_error": str(error),
                    "instruction": (
                        "Return one corrected JSON object only. The rollback must "
                        "restore exact parent content; create must pair with delete."
                    ),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            call = self.repair_call
        raise RealProposalError("unreachable proposer state")

    def rollback(self, changeset: InactiveChangeSet) -> dict[str, Any]:
        receipt = self._rollback_receipts.get(changeset.candidate_agent_program_sha256)
        if receipt is None:
            self.profile_root(changeset.candidate_agent_program_sha256)
            matches = list(self.output_root.rglob("*/ROLLBACK-VERIFICATION.json"))
            for path in matches:
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if (
                    candidate.get("candidate_agent_program_sha256")
                    == changeset.candidate_agent_program_sha256
                ):
                    receipt = candidate
                    break
        if receipt is None or receipt.get("verified") is not True:
            raise RealProposalError("verified rollback receipt is missing")
        return {
            "forward_patch_sha256": receipt["forward_patch_sha256"],
            "rollback_patch_sha256": receipt["rollback_patch_sha256"],
            "verified": True,
        }
