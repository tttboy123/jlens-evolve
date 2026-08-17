"""Capability validation and append-only lineage for bounded harness mutations."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ALLOWED_NODES = {
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.If,
    ast.Return,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.In,
    ast.NotIn,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Tuple,
}


class MutationContractError(ValueError):
    """Raised when candidate code or lineage violates the mutation contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_source(source: str, *, limits: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic capability report without executing source."""
    reasons: list[str] = []
    encoded = source.encode("utf-8")
    if len(encoded) > int(limits["max_source_bytes"]):
        reasons.append("source_bytes_exceeded")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {
            "allowed": False,
            "source_sha256": _sha256(encoded),
            "source_bytes": len(encoded),
            "ast_nodes": 0,
            "function_name": None,
            "function_args": [],
            "reasons": [f"SyntaxError:{exc.msg}"],
            "capabilities": [],
        }
    nodes = list(ast.walk(tree))
    if len(nodes) > int(limits["max_ast_nodes"]):
        reasons.append("ast_nodes_exceeded")
    forbidden = sorted(
        {type(node).__name__ for node in nodes if type(node) not in _ALLOWED_NODES}
    )
    reasons.extend(f"forbidden_node:{name}" for name in forbidden)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(tree.body) != 1 or len(functions) != 1:
        reasons.append("module_must_contain_exactly_one_function")
    function = functions[0] if functions else None
    function_name = function.name if function else None
    function_args = (
        [argument.arg for argument in function.args.args] if function else []
    )
    if function_name != "select_route":
        reasons.append("function_name_must_be_select_route")
    if function_args != ["attempt", "last_error"]:
        reasons.append("function_args_must_be_attempt_last_error")
    if function and (
        function.decorator_list
        or function.returns is not None
        or function.args.defaults
        or function.args.kw_defaults
        or function.args.vararg is not None
        or function.args.kwarg is not None
        or function.args.posonlyargs
        or function.args.kwonlyargs
    ):
        reasons.append("function_signature_extensions_forbidden")
    allowed_names = {"attempt", "last_error"}
    invalid_names = sorted(
        {
            node.id
            for node in nodes
            if isinstance(node, ast.Name) and node.id not in allowed_names
        }
    )
    reasons.extend(f"forbidden_name:{name}" for name in invalid_names)
    if "__" in source:
        reasons.append("dunder_forbidden")
    return {
        "allowed": not reasons,
        "source_sha256": _sha256(encoded),
        "source_bytes": len(encoded),
        "ast_nodes": len(nodes),
        "function_name": function_name,
        "function_args": function_args,
        "reasons": sorted(set(reasons)),
        "capabilities": [
            "pure_argument_read",
            "conditional_branch",
            "constant_route_return",
        ]
        if not reasons
        else [],
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


class MutationArchive:
    """Append-only mutation records plus an atomic active pointer."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.records_path = self.root / "records.jsonl"
        self.transitions_path = self.root / "active-transitions.jsonl"
        self.active_path = self.root / "active.json"

    @staticmethod
    def _validate_pointer(candidate_id: str, source_sha256: str) -> None:
        if not candidate_id or _SHA256.fullmatch(source_sha256) is None:
            raise MutationContractError("invalid active pointer")

    @staticmethod
    def _append_unique(path: Path, row: dict[str, Any], fingerprint: str) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            seen = {
                json.loads(line).get("record_fingerprint")
                for line in handle
                if line.strip()
            }
            if fingerprint in seen:
                return False
            handle.seek(0, os.SEEK_END)
            handle.write(
                _canonical_json({**row, "record_fingerprint": fingerprint}) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
            return True

    def append_record(self, record: dict[str, Any]) -> bool:
        required = {
            "candidate_id",
            "source_sha256",
            "parent_source_sha256",
            "status",
        }
        if not required.issubset(record):
            raise MutationContractError("mutation record missing required fields")
        if _SHA256.fullmatch(str(record["source_sha256"])) is None:
            raise MutationContractError("invalid mutation source hash")
        fingerprint = _sha256(_canonical_json(record).encode("utf-8"))
        return self._append_unique(self.records_path, record, fingerprint)

    def read_records(self) -> list[dict[str, Any]]:
        if not self.records_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def initialize_active(self, *, candidate_id: str, source_sha256: str) -> None:
        self._validate_pointer(candidate_id, source_sha256)
        if self.active_path.exists():
            raise MutationContractError("active pointer already initialized")
        pointer = {
            "candidate_id": candidate_id,
            "source_sha256": source_sha256,
            "sequence": 0,
        }
        _atomic_json(self.active_path, pointer)
        self._append_transition(
            {
                "event_type": "initialize",
                "previous": None,
                "active": pointer,
                "reason": "frozen parent",
            }
        )

    def read_active(self) -> dict[str, Any]:
        if not self.active_path.exists():
            raise MutationContractError("active pointer is not initialized")
        return json.loads(self.active_path.read_text(encoding="utf-8"))

    def _append_transition(self, event: dict[str, Any]) -> None:
        fingerprint = _sha256(_canonical_json(event).encode("utf-8"))
        self._append_unique(self.transitions_path, event, fingerprint)

    def activate(self, *, candidate_id: str, source_sha256: str) -> dict[str, Any]:
        self._validate_pointer(candidate_id, source_sha256)
        verified = any(
            row.get("candidate_id") == candidate_id
            and row.get("source_sha256") == source_sha256
            and row.get("status") == "verified"
            for row in self.read_records()
        )
        if not verified:
            raise MutationContractError("only verified candidate can be activated")
        previous = self.read_active()
        active = {
            "candidate_id": candidate_id,
            "source_sha256": source_sha256,
            "sequence": int(previous["sequence"]) + 1,
        }
        _atomic_json(self.active_path, active)
        self._append_transition(
            {
                "event_type": "activate",
                "previous": previous,
                "active": active,
                "reason": "verified admission",
            }
        )
        return active

    def rollback(self, *, reason: str) -> dict[str, Any]:
        if not reason.strip():
            raise MutationContractError("rollback reason is required")
        current = self.read_active()
        transitions = [
            json.loads(line)
            for line in self.transitions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        activation = next(
            (
                row
                for row in reversed(transitions)
                if row["event_type"] == "activate"
                and row["active"]["source_sha256"] == current["source_sha256"]
            ),
            None,
        )
        if activation is None or activation["previous"] is None:
            raise MutationContractError("no activation parent available for rollback")
        parent = {
            **activation["previous"],
            "sequence": int(current["sequence"]) + 1,
        }
        _atomic_json(self.active_path, parent)
        self._append_transition(
            {
                "event_type": "rollback",
                "previous": current,
                "active": parent,
                "reason": reason,
            }
        )
        return parent
