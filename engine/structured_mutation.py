"""Structured mutation plans and deterministic AST operators for proposer v4."""

from __future__ import annotations

import ast
import copy
import json
import re
from dataclasses import dataclass
from typing import Any

_CURRENT_PROGRAM = re.compile(
    r"Current program:\s*```(?:python|py)?[^\n]*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_TARGET_FAILURE = re.compile(
    r"###\s*target_failure\s*```.*?['\"]id['\"]\s*:\s*['\"]([^'\"]+)",
    re.IGNORECASE | re.DOTALL,
)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_FAILURE_OPERATOR = {
    "filter_normalized_status": "canonicalize_before_predicate",
    "reject_invalid_amounts": "finite_numeric_guard",
}
_OPERATORS = {*_FAILURE_OPERATOR.values(), "free_form_rewrite"}


@dataclass(frozen=True)
class MutationPlan:
    schema_version: int
    operator_id: str
    target_symbol: str
    public_failure: str
    preserve: tuple[str, ...] = ()

    @property
    def structured(self) -> bool:
        return self.operator_id != "free_form_rewrite"


@dataclass(frozen=True)
class MutationResult:
    source: str
    changed: bool
    operator_id: str
    postcondition_valid: bool
    error: str | None = None


def _message_text(payload: dict[str, Any]) -> str:
    return "\n".join(
        str(message.get("content", ""))
        for message in payload.get("messages", [])
        if isinstance(message, dict)
    )


def extract_current_program(payload: dict[str, Any]) -> str | None:
    """Extract only the explicitly labelled current program, not references."""
    matches = _CURRENT_PROGRAM.findall(_message_text(payload))
    return matches[-1].strip() if matches else None


def extract_public_target_failure(payload: dict[str, Any]) -> str | None:
    """Read the prompt-visible target failure without inspecting holdout data."""
    match = _TARGET_FAILURE.search(_message_text(payload))
    return match.group(1) if match else None


def derive_fallback_plan(public_failure: str | None) -> MutationPlan:
    """Map one public failure to a bounded operator or a free-form fallback."""
    failure = str(public_failure or "unknown_public_failure")
    return MutationPlan(
        schema_version=1,
        operator_id=_FAILURE_OPERATOR.get(failure, "free_form_rewrite"),
        target_symbol="solve",
        public_failure=failure,
        preserve=(),
    )


def parse_mutation_plan(content: str, public_failure: str | None) -> MutationPlan:
    """Parse and validate a model plan against the prompt-visible failure."""
    match = _JSON_OBJECT.search(content)
    if not match:
        raise ValueError("planner response has no JSON object")
    data = json.loads(match.group(0))
    required = {
        "schema_version",
        "operator_id",
        "target_symbol",
        "public_failure",
        "preserve",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"mutation plan missing fields: {missing}")
    if data["schema_version"] != 1:
        raise ValueError("unsupported mutation plan schema")
    if data["operator_id"] not in _OPERATORS:
        raise ValueError("mutation plan operator is not allowlisted")
    if data["target_symbol"] != "solve":
        raise ValueError("mutation plan may only target solve")
    if data["public_failure"] != public_failure:
        raise ValueError("mutation plan changed the public target failure")
    expected = _FAILURE_OPERATOR.get(str(public_failure))
    if expected and data["operator_id"] != expected:
        raise ValueError("mutation plan operator does not match public failure")
    if not expected and data["operator_id"] != "free_form_rewrite":
        raise ValueError("unsupported public failure requires free-form fallback")
    preserve = data["preserve"]
    if not isinstance(preserve, list) or not all(
        isinstance(item, str) for item in preserve
    ):
        raise ValueError("mutation plan preserve must be a string list")
    if any(
        token in item.lower() for item in preserve for token in ("hidden", "holdout")
    ):
        raise ValueError("mutation plan cannot reference hidden evaluation")
    return MutationPlan(
        schema_version=1,
        operator_id=str(data["operator_id"]),
        target_symbol="solve",
        public_failure=str(data["public_failure"]),
        preserve=tuple(preserve),
    )


def _is_status_access(node: ast.AST) -> bool:
    if isinstance(node, ast.Subscript):
        return isinstance(node.slice, ast.Constant) and node.slice.value == "status"
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    return bool(
        node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "status"
    )


def _normalized_status(node: ast.expr) -> ast.expr:
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


class _StatusPredicateTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = 0

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        expressions = [node.left, *node.comparators]
        has_paid = any(
            isinstance(item, ast.Constant)
            and isinstance(item.value, str)
            and item.value.lower() == "paid"
            for item in expressions
        )
        if not has_paid:
            return node
        if any(
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "lower"
            for item in ast.walk(node)
        ):
            return node
        if _is_status_access(node.left):
            node.left = _normalized_status(node.left)
            self.changed += 1
        for index, comparator in enumerate(node.comparators):
            if _is_status_access(comparator):
                node.comparators[index] = _normalized_status(comparator)
                self.changed += 1
        return node


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


def _apply_canonicalize(tree: ast.Module) -> int:
    transformer = _StatusPredicateTransformer()
    transformer.visit(tree)
    return transformer.changed


def _apply_numeric_guard(tree: ast.Module) -> int:
    function = _solve_function(tree)
    if function is None:
        return 0
    loop = _first_record_loop(function)
    if loop is None or not isinstance(loop.target, ast.Name):
        return 0
    if postcondition_satisfied(
        ast.unparse(tree), derive_fallback_plan("reject_invalid_amounts")
    ):
        return 0
    row_name = loop.target.id
    guard = ast.parse(
        f"""if not isinstance({row_name}, dict):
    continue
_v4_amount = {row_name}.get("amount")
if (isinstance(_v4_amount, bool) or
        not isinstance(_v4_amount, (int, float)) or
        not math.isfinite(float(_v4_amount)) or
        _v4_amount <= 0):
    continue
"""
    ).body
    loop.body[0:0] = guard
    _ensure_math_import(tree)
    return 1


def postcondition_satisfied(source: str, plan: MutationPlan) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    dump = ast.dump(tree, include_attributes=False)
    if plan.operator_id == "free_form_rewrite":
        return True
    if plan.operator_id == "canonicalize_before_predicate":
        return bool(
            "attr='strip'" in dump
            and "attr='lower'" in dump
            and "value='status'" in dump
        )
    if plan.operator_id == "finite_numeric_guard":
        return bool(
            "attr='isfinite'" in dump
            and "id='math'" in dump
            and "id='bool'" in dump
            and "id='int'" in dump
            and "id='float'" in dump
        )
    return False


def apply_mutation_plan(source: str, plan: MutationPlan) -> MutationResult:
    """Apply one deterministic operator and report its verifiable outcome."""
    if not plan.structured:
        return MutationResult(
            source=source,
            changed=False,
            operator_id=plan.operator_id,
            postcondition_valid=True,
        )
    try:
        tree = ast.parse(source)
        changes = (
            _apply_canonicalize(tree)
            if plan.operator_id == "canonicalize_before_predicate"
            else _apply_numeric_guard(tree)
        )
        ast.fix_missing_locations(tree)
        mutated = ast.unparse(tree).strip() + "\n"
        valid = postcondition_satisfied(mutated, plan)
        return MutationResult(
            source=mutated,
            changed=bool(changes and mutated.strip() != source.strip()),
            operator_id=plan.operator_id,
            postcondition_valid=valid,
        )
    except (SyntaxError, ValueError, TypeError) as exc:
        return MutationResult(
            source=source,
            changed=False,
            operator_id=plan.operator_id,
            postcondition_valid=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _append_user_instruction(
    payload: dict[str, Any], instruction: str
) -> dict[str, Any]:
    updated = copy.deepcopy(payload)
    messages = updated.setdefault("messages", [])
    if messages and isinstance(messages[-1], dict):
        messages[-1]["content"] = str(messages[-1].get("content", "")) + instruction
    else:
        messages.append({"role": "user", "content": instruction.strip()})
    return updated


def build_planner_payload(
    payload: dict[str, Any], *, structured: bool
) -> dict[str, Any]:
    if structured:
        instruction = """

[Mutation planner stage]
Do not output code in this first stage. Use only the prompt-visible target_failure.
Return exactly one compact JSON object with schema_version=1, operator_id,
target_symbol="solve", public_failure, and preserve. The only structured mappings
are filter_normalized_status -> canonicalize_before_predicate and
reject_invalid_amounts -> finite_numeric_guard. For any other public failure use
free_form_rewrite. Never mention private evaluation data.
"""
    else:
        instruction = """

[Planner control stage]
Do not output code in this first stage. Briefly plan one change for the
prompt-visible target_failure while preserving listed passing behavior.
"""
    return _append_user_instruction(payload, instruction)


def build_coder_payload(
    payload: dict[str, Any],
    *,
    plan: MutationPlan,
    planner_content: str,
    scaffold: str | None,
    structured: bool,
) -> dict[str, Any]:
    if structured and scaffold is not None:
        plan_json = json.dumps(
            {
                "schema_version": plan.schema_version,
                "operator_id": plan.operator_id,
                "target_symbol": plan.target_symbol,
                "public_failure": plan.public_failure,
                "preserve": list(plan.preserve),
            },
            sort_keys=True,
        )
        instruction = f"""

[Bounded mutation repair stage]
MutationPlan: {plan_json}
The deterministic scaffold below already applies the required structural
operator. Return one complete fenced Python program. You may repair local syntax
or data flow, but must retain the operator postcondition and listed passing
behavior. Do not broaden the edit beyond solve(records).
```python
{scaffold.strip()}
```
"""
    else:
        instruction = f"""

[Two-stage coder]
Planner result:
{planner_content[:1200]}
Implement one complete program for the prompt-visible target failure. Return only
one fenced Python program preserving solve(records).
"""
    return _append_user_instruction(payload, instruction)
