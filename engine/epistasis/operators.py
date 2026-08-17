"""Schema-parameterized deterministic AST operators and sequential composition.

The operator universe mirrors the record-cleaning failure mechanisms:

- ``canonicalize_status``      -> normalize the status field before filtering
- ``reject_invalid_amounts``   -> skip bool / non-numeric / non-finite / non-positive values
- ``normalize_identity``       -> strip + casefold the identity field in outputs
- ``drop_empty_identity``      -> skip rows whose identity is blank

Each operator is a pure ``source -> source`` transform.  Its postcondition is a
**behavioral probe**: a minimal case that only that mechanism can satisfy, so
postconditions are precise and order-independent (a structural substring check
cannot distinguish "identity normalized" from "status normalized").
"""

from __future__ import annotations

import ast
import copy
import math
from dataclasses import dataclass
from typing import Any, Callable

from .schema import TaskSchema

MECHANISMS: tuple[str, ...] = ("status", "amount", "identity", "empty")

OPERATOR_IDS: tuple[str, ...] = (
    "canonicalize_status",
    "reject_invalid_amounts",
    "normalize_identity",
    "drop_empty_identity",
)

_OPERATOR_BY_MECHANISM = dict(zip(MECHANISMS, OPERATOR_IDS, strict=True))


@dataclass(frozen=True, slots=True)
class OpResult:
    operator_id: str
    source: str
    changed: bool
    postcondition_ok: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ComposeResult:
    source: str
    ok: bool
    reason: str
    operator_ids: tuple[str, ...]
    results: tuple[OpResult, ...]


# --------------------------------------------------------------------------
# Probe evaluation (behavioral postconditions)
# --------------------------------------------------------------------------


def _safe_execute(source: str, records: list[Any]) -> list[tuple[str, float]] | None:
    """Execute candidate source in the restricted experiment sandbox."""

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "math" and level == 0:
            return math
        raise ImportError(f"import blocked by experiment sandbox: {name}")

    safe_builtins = {
        "__import__": safe_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    namespace: dict[str, Any] = {"__builtins__": safe_builtins, "math": math}
    try:
        exec(compile(source, "<candidate>", "exec"), namespace, namespace)  # noqa: S102
        solve = namespace.get("solve")
        if not callable(solve):
            return None
        value = solve(records)
        if not isinstance(value, list):
            return None
        return [(str(user), float(total)) for user, total in value]
    except Exception:  # noqa: BLE001 - candidate failures are evidence
        return None


def _probe_rows(
    schema: TaskSchema,
) -> dict[str, tuple[list[Any], list[tuple[str, float]]]]:
    status = schema.status_field
    accepted = schema.accepted_status
    identity = schema.identity_field
    value = schema.value_field

    def row(i: str, v: Any, st: Any = None) -> dict[str, Any]:
        out = {identity: i, value: v}
        if st is not None:
            out[status] = st
        if schema.currency_field is not None and schema.accepted_currency is not None:
            out[schema.currency_field] = schema.accepted_currency
        return out

    return {
        "canonicalize_status": (
            [
                row("alice", 2, f" {accepted.upper()} "),
                row("bob", 1, accepted.capitalize()),
            ],
            [("alice", 2.0), ("bob", 1.0)],
        ),
        "reject_invalid_amounts": (
            [
                row("ok", 3, accepted),
                row("bool", True, accepted),
                row("nan", float("nan"), accepted),
                row("zero", 0, accepted),
            ],
            [("ok", 3.0)],
        ),
        "normalize_identity": (
            [
                row(" Alice ", 2, accepted),
                row("bob", 3, accepted),
            ],
            [("alice", 2.0), ("bob", 3.0)],
        ),
        "drop_empty_identity": (
            [
                row(" ", 30, accepted),
                row("bob", 4, accepted),
            ],
            [("bob", 4.0)],
        ),
    }


def _probe_passes(source: str, operator_id: str, schema: TaskSchema) -> bool:
    probes = _probe_rows(schema)
    records, expected = probes[operator_id]
    actual = _safe_execute(source, records)
    return actual == expected


# --------------------------------------------------------------------------
# AST transforms
# --------------------------------------------------------------------------


def _solve_function(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "solve"
        ):
            return node
    return None


def _first_record_loop(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.For | None:
    for node in ast.walk(function):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            return node
    return None


def _ensure_math_import(tree: ast.Module) -> None:
    for node in tree.body:
        if isinstance(node, ast.Import) and any(
            alias.name == "math" for alias in node.names
        ):
            return
        if isinstance(node, ast.ImportFrom) and node.module == "math":
            return
    tree.body.insert(0, ast.Import(names=[ast.alias(name="math")]))


def _field_access(node: ast.AST, field: str) -> bool:
    """True when ``node`` reads ``row[field]`` or ``row.get(field, ...)``."""
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        return node.slice.value == field
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return bool(
            node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == field
        )
    return False


def _normalized_expr(node: ast.expr) -> ast.expr:
    as_text = ast.Call(
        func=ast.Name(id="str", ctx=ast.Load()),
        args=[copy.deepcopy(node)],
        keywords=[],
    )
    stripped = ast.Call(
        func=ast.Attribute(value=as_text, attr="strip", ctx=ast.Load()),
        args=[],
        keywords=[],
    )
    return ast.Call(
        func=ast.Attribute(value=stripped, attr="lower", ctx=ast.Load()),
        args=[],
        keywords=[],
    )


def _already_normalized(node: ast.AST) -> bool:
    return any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == "lower"
        for item in ast.walk(node)
    )


class _StatusPredicateTransformer(ast.NodeTransformer):
    def __init__(self, schema: TaskSchema) -> None:
        self.schema = schema
        self.changed = 0

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        expressions = [node.left, *node.comparators]
        has_accepted = any(
            isinstance(item, ast.Constant)
            and isinstance(item.value, str)
            and item.value.lower() == self.schema.accepted_status.lower()
            for item in expressions
        )
        if not has_accepted:
            return node
        if _already_normalized(node):
            return node
        if _field_access(node.left, self.schema.status_field):
            node.left = _normalized_expr(node.left)
            self.changed += 1
        for index, comparator in enumerate(node.comparators):
            if _field_access(comparator, self.schema.status_field):
                node.comparators[index] = _normalized_expr(comparator)
                self.changed += 1
        return node


class _IdentityOutputTransformer(ast.NodeTransformer):
    """Wrap identity-field reads that feed output tuples with str().strip().lower()."""

    def __init__(self, schema: TaskSchema) -> None:
        self.schema = schema
        self.changed = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "append":
            return node
        if not node.args:
            return node
        arg = node.args[0]
        if not isinstance(arg, ast.Tuple) or not arg.elts:
            return node
        first = arg.elts[0]
        if _field_access(first, self.schema.identity_field) and not _already_normalized(
            first
        ):
            arg.elts[0] = _normalized_expr(first)
            self.changed += 1
        return node


def _insert_loop_guard(tree: ast.Module, guard_source: str) -> int:
    function = _solve_function(tree)
    if function is None:
        return 0
    loop = _first_record_loop(function)
    if loop is None or not isinstance(loop.target, ast.Name):
        return 0
    guard = ast.parse(guard_source).body
    loop.body[0:0] = guard
    return 1


def _canonicalize_status(source: str, schema: TaskSchema) -> OpResult:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return OpResult("canonicalize_status", source, False, False, str(exc))
    transformer = _StatusPredicateTransformer(schema)
    transformer.visit(tree)
    ast.fix_missing_locations(tree)
    out = ast.unparse(tree).strip() + "\n"
    changed = transformer.changed > 0 and out != source.strip() + "\n"
    return OpResult(
        "canonicalize_status",
        out,
        changed=changed,
        postcondition_ok=_probe_passes(out, "canonicalize_status", schema),
    )


def _reject_invalid_amounts(source: str, schema: TaskSchema) -> OpResult:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return OpResult("reject_invalid_amounts", source, False, False, str(exc))
    if _probe_passes(source, "reject_invalid_amounts", schema):
        return OpResult("reject_invalid_amounts", source, False, True)
    function = _solve_function(tree)
    loop = _first_record_loop(function) if function is not None else None
    if loop is None or not isinstance(loop.target, ast.Name):
        return OpResult("reject_invalid_amounts", source, False, False, "no loop")
    row_name = loop.target.id
    guard_source = f"""if not isinstance({row_name}, dict):
    continue
_ep_amount = {row_name}.get({schema.value_field!r})
if (isinstance(_ep_amount, bool) or
        not isinstance(_ep_amount, (int, float)) or
        not math.isfinite(float(_ep_amount)) or
        _ep_amount <= 0):
    continue
"""
    changed = _insert_loop_guard(tree, guard_source)
    _ensure_math_import(tree)
    ast.fix_missing_locations(tree)
    out = ast.unparse(tree).strip() + "\n"
    return OpResult(
        "reject_invalid_amounts",
        out,
        changed=changed > 0 and out != source.strip() + "\n",
        postcondition_ok=_probe_passes(out, "reject_invalid_amounts", schema),
    )


def _normalize_identity(source: str, schema: TaskSchema) -> OpResult:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return OpResult("normalize_identity", source, False, False, str(exc))
    if _probe_passes(source, "normalize_identity", schema):
        return OpResult("normalize_identity", source, False, True)
    transformer = _IdentityOutputTransformer(schema)
    transformer.visit(tree)
    ast.fix_missing_locations(tree)
    out = ast.unparse(tree).strip() + "\n"
    changed = transformer.changed > 0 and out != source.strip() + "\n"
    return OpResult(
        "normalize_identity",
        out,
        changed=changed,
        postcondition_ok=_probe_passes(out, "normalize_identity", schema),
    )


def _drop_empty_identity(source: str, schema: TaskSchema) -> OpResult:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return OpResult("drop_empty_identity", source, False, False, str(exc))
    if _probe_passes(source, "drop_empty_identity", schema):
        return OpResult("drop_empty_identity", source, False, True)
    guard_source = (
        f"if not str(row.get({schema.identity_field!r}, '')).strip():\n    continue\n"
    )
    changed = _insert_loop_guard(tree, guard_source)
    ast.fix_missing_locations(tree)
    out = ast.unparse(tree).strip() + "\n"
    return OpResult(
        "drop_empty_identity",
        out,
        changed=changed > 0 and out != source.strip() + "\n",
        postcondition_ok=_probe_passes(out, "drop_empty_identity", schema),
    )


_APPLY: dict[str, Callable[[str, TaskSchema], OpResult]] = {
    "canonicalize_status": _canonicalize_status,
    "reject_invalid_amounts": _reject_invalid_amounts,
    "normalize_identity": _normalize_identity,
    "drop_empty_identity": _drop_empty_identity,
}


def postcondition_ok(source: str, operator_id: str, schema: TaskSchema) -> bool:
    if operator_id not in _APPLY:
        raise ValueError(f"unknown operator: {operator_id}")
    return _probe_passes(source, operator_id, schema)


def apply_operator(source: str, operator_id: str, schema: TaskSchema) -> OpResult:
    """Apply one deterministic operator, returning a verifiable result."""
    if operator_id not in _APPLY:
        raise ValueError(f"unknown operator: {operator_id}")
    return _APPLY[operator_id](source, schema)


def apply_operators(
    source: str, operator_ids: tuple[str, ...] | list[str], schema: TaskSchema
) -> ComposeResult:
    """Apply operators in sequence; abort on first no-op or probe failure."""
    current = source
    results: list[OpResult] = []
    for operator_id in operator_ids:
        result = apply_operator(current, operator_id, schema)
        results.append(result)
        if not result.changed:
            return ComposeResult(
                source=current,
                ok=False,
                reason=f"{operator_id}:no-op",
                operator_ids=tuple(operator_ids),
                results=tuple(results),
            )
        if not result.postcondition_ok:
            return ComposeResult(
                source=current,
                ok=False,
                reason=f"{operator_id}:postcondition",
                operator_ids=tuple(operator_ids),
                results=tuple(results),
            )
        current = result.source
    return ComposeResult(
        source=current,
        ok=True,
        reason="ok",
        operator_ids=tuple(operator_ids),
        results=tuple(results),
    )


def coverage_of(operator_ids: tuple[str, ...], mechanisms: tuple[str, ...]) -> float:
    """Fraction of task mechanisms covered by an active operator set."""
    if not mechanisms:
        return 0.0
    active = set(operator_ids)
    covered = sum(1 for m in mechanisms if _OPERATOR_BY_MECHANISM[m] in active)
    return covered / len(mechanisms)
