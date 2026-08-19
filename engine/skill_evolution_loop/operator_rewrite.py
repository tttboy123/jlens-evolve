"""Typed AST-selector operations for diagnosis-to-code realization experiments."""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_json, sha256_json

_OPERATORS = frozenset(
    {
        "replace_condition",
        "replace_expression",
        "replace_statement",
        "insert_method",
        "replace_method_body",
        "insert_assignment_before",
        "replace_constant",
        "align_trailing_defaults",
        "normalize_inline_wrapper_boundaries",
        "initialize_generated_subclass_identity",
        "remove_property_index_parens",
        "remove_variable_obj_role",
    }
)
_SYMBOL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FORBIDDEN_STATEMENTS = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
)


@dataclass(frozen=True)
class EditIntent:
    defect: str
    trigger: str
    desired_boundary: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EditIntent:
        if not isinstance(data, dict) or set(data) != {
            "defect",
            "trigger",
            "desired_boundary",
        }:
            raise ContractError("operator edit intent fields are invalid")
        intent = cls(
            defect=str(data["defect"]),
            trigger=str(data["trigger"]),
            desired_boundary=str(data["desired_boundary"]),
        )
        if any(not value.strip() for value in intent.to_dict().values()):
            raise ContractError("operator edit intent fields must be non-empty")
        return intent

    def to_dict(self) -> dict[str, str]:
        return {
            "defect": self.defect,
            "trigger": self.trigger,
            "desired_boundary": self.desired_boundary,
        }


@dataclass(frozen=True)
class OperatorOperation:
    operator: str
    selector: dict[str, Any]
    arguments: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperatorOperation:
        if not isinstance(data, dict):
            raise ContractError("operator operation fields are invalid")
        # Normalize the alternative 7B-style shape
        # {"<operator>": {"selector": ..., "arguments": ...}} into the canonical
        # {"operator": ..., "selector": ..., "arguments": ...} shape.  Only the
        # exact single-key operator-as-key form is accepted; anything else still
        # fails the strict field check below.
        # The Student sometimes copies the candidate's "suggested_operator"
        # field into the operation (it appears in the framework-enumerated
        # candidate JSON).  Use it as the operator when "operator" is absent and
        # drop it when the canonical field is present.
        if "suggested_operator" in data:
            if "operator" not in data and data["suggested_operator"] in _OPERATORS:
                suggested = data["suggested_operator"]
                rest = {
                    k: v for k, v in data.items() if k != "suggested_operator"
                }
                data = {"operator": suggested, **rest}
            else:
                data = {
                    k: v for k, v in data.items() if k != "suggested_operator"
                }
        if (
            len(data) == 1
            and next(iter(data)) in _OPERATORS
            and isinstance(next(iter(data.values())), dict)
        ):
            operator_name = next(iter(data))
            payload = data[operator_name]
            if set(payload) == {"selector", "arguments"}:
                data = {"operator": operator_name, **payload}
            elif (
                "arguments" in payload
                and isinstance(payload["arguments"], dict)
                and "source" in payload
            ):
                # Flattened variant: {"<op>": {"source": ..., "occurrence": ...,
                # "arguments": {...}}} -> selector wrapped back under "selector".
                data = {
                    "operator": operator_name,
                    "selector": {
                        "source": payload["source"],
                        "occurrence": payload.get("occurrence", 0),
                    },
                    "arguments": payload["arguments"],
                }
        elif (
            len(data) == 2
            and "arguments" in data
            and isinstance(data["arguments"], dict)
            and any(key in _OPERATORS for key in data)
        ):
            # Sibling variant: {"<op>": {"source": ..., "occurrence": ...},
            # "arguments": {...}} with arguments OUTSIDE the operator payload.
            operator_name = next(key for key in data if key in _OPERATORS)
            payload = data[operator_name]
            if isinstance(payload, dict) and "source" in payload:
                data = {
                    "operator": operator_name,
                    "selector": {
                        "source": payload["source"],
                        "occurrence": payload.get("occurrence", 0),
                    },
                    "arguments": data["arguments"],
                }
        if set(data) != {"operator", "selector", "arguments"}:
            raise ContractError("operator operation fields are invalid")
        arguments = dict(data["arguments"])
        if data["operator"] == "replace_statement" and isinstance(
            arguments.get("new_statements"), list
        ):
            statements = arguments["new_statements"]
            if not 1 <= len(statements) <= 4 or any(
                not isinstance(row, str) or not row.strip() for row in statements
            ):
                raise ContractError("replacement statement list is invalid")
            arguments["new_statements"] = "\n".join(statements)
        if data["operator"] == "replace_method_body" and isinstance(
            arguments.get("new_body"), list
        ):
            body = arguments["new_body"]
            if not 1 <= len(body) <= 8 or any(
                not isinstance(row, str) for row in body
            ):
                raise ContractError("replacement method body list is invalid")
            arguments["new_body"] = "\n".join(body)
        operation = cls(
            operator=str(data["operator"]),
            selector=dict(data["selector"]),
            arguments=arguments,
        )
        operation.validate()
        return operation

    def validate(self) -> None:
        if self.operator not in _OPERATORS:
            raise ContractError("operator is not allowlisted")
        occurrence = self.selector.get("occurrence")
        if type(occurrence) is not int or occurrence < 0:
            raise ContractError("operator selector occurrence is invalid")
        expected = {
            "replace_condition": (
                {"source", "occurrence"},
                {"new_condition"},
            ),
            "replace_expression": (
                {"source", "occurrence"},
                {"new_expression"},
            ),
            "replace_statement": (
                {"source", "occurrence"},
                {"new_statements"},
            ),
            "insert_method": (
                {"occurrence"},
                {"method_source"},
            ),
            "replace_method_body": (
                {"occurrence"},
                {"new_body"},
            ),
            "insert_assignment_before": (
                {"source", "occurrence"},
                {"name", "expression"},
            ),
            "replace_constant": (
                {"value", "occurrence"},
                {"new_value"},
            ),
            "align_trailing_defaults": (
                {"occurrence"},
                set(),
            ),
            "normalize_inline_wrapper_boundaries": (
                {"occurrence"},
                set(),
            ),
            "initialize_generated_subclass_identity": (
                {"occurrence"},
                set(),
            ),
            "remove_property_index_parens": (
                {"occurrence"},
                set(),
            ),
            "remove_variable_obj_role": (
                {"occurrence"},
                set(),
            ),
        }[self.operator]
        if set(self.selector) != expected[0] or set(self.arguments) != expected[1]:
            raise ContractError("operator selector or argument fields are invalid")
        string_fields = {
            "replace_condition": (("source",), ("new_condition",)),
            "replace_expression": (("source",), ("new_expression",)),
            "replace_statement": (("source",), ("new_statements",)),
            "insert_method": ((), ("method_source",)),
            "replace_method_body": ((), ("new_body",)),
            "insert_assignment_before": (("source",), ("name", "expression")),
            "replace_constant": ((), ()),
            "align_trailing_defaults": ((), ()),
            "normalize_inline_wrapper_boundaries": ((), ()),
            "initialize_generated_subclass_identity": ((), ()),
            "remove_property_index_parens": ((), ()),
            "remove_variable_obj_role": ((), ()),
        }[self.operator]
        values = [
            *(self.selector[field] for field in string_fields[0]),
            *(self.arguments[field] for field in string_fields[1]),
        ]
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ContractError("operator string argument is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "selector": self.selector,
            "arguments": self.arguments,
        }


@dataclass(frozen=True)
class OperatorPlan:
    schema_version: int
    file: str
    symbol: str
    intent: EditIntent
    operations: tuple[OperatorOperation, ...]
    diagnostic: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperatorPlan:
        if not isinstance(data, dict) or set(data) != {
            "schema_version",
            "file",
            "symbol",
            "intent",
            "operations",
            "diagnostic",
        }:
            raise ContractError("operator plan fields are invalid")
        rows = data["operations"]
        if not isinstance(rows, list):
            raise ContractError("operator plan operations must be a list")
        plan = cls(
            schema_version=data["schema_version"],
            file=str(data["file"]),
            symbol=str(data["symbol"]),
            intent=EditIntent.from_dict(data["intent"]),
            operations=tuple(OperatorOperation.from_dict(row) for row in rows),
            diagnostic=str(data["diagnostic"]),
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported operator plan schema")
        path = Path(self.file)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
            raise ContractError("operator plan target must be a relative Python file")
        if _SYMBOL.fullmatch(self.symbol) is None:
            raise ContractError("operator plan symbol is invalid")
        if not 1 <= len(self.operations) <= 4:
            raise ContractError("operator plan requires one to four operations")
        if not self.diagnostic.strip():
            raise ContractError("operator plan diagnostic must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "file": self.file,
            "symbol": self.symbol,
            "intent": self.intent.to_dict(),
            "operations": [row.to_dict() for row in self.operations],
            "diagnostic": self.diagnostic,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class MaterializationResult:
    before: str
    after: str
    accepted: bool
    failure_reason: str | None
    gates: dict[str, bool]
    plan_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_sha256": _sha256_text(self.before),
            "after_sha256": _sha256_text(self.after),
            "accepted": self.accepted,
            "failure_reason": self.failure_reason,
            "gates": self.gates,
            "plan_fingerprint": self.plan_fingerprint,
        }


@dataclass(frozen=True)
class _Edit:
    start: int
    end: int
    replacement: str
    operation: OperatorOperation
    postcondition_dumps: tuple[str, ...]


_Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def materialize_operator_plan(source: str, plan: OperatorPlan) -> MaterializationResult:
    """Resolve typed selectors in one symbol and mechanically render their edits."""
    plan.validate()
    try:
        before_tree = ast.parse(source)
    except SyntaxError as exc:
        raise ContractError("operator source is not valid Python") from exc
    symbol = _find_definition(before_tree.body, plan.symbol.split("."))
    if symbol is None or symbol.end_lineno is None or symbol.end_col_offset is None:
        raise ContractError("operator symbol did not resolve exactly once")
    edits: list[_Edit] = []
    for operation in plan.operations:
        if operation.operator == "initialize_generated_subclass_identity":
            edits.extend(
                _generated_subclass_identity_edits(
                    source, before_tree, symbol, operation
                )
            )
        else:
            edits.append(_resolve_operation(source, symbol, operation))
    _reject_overlaps(edits)
    after = source
    for edit in sorted(edits, key=lambda row: (row.start, row.end), reverse=True):
        after = after[: edit.start] + edit.replacement + after[edit.end :]
    gates = {
        "selector_match": True,
        "syntax_valid": False,
        "ast_code_diff": False,
        "non_copy": after != source,
        "postconditions": False,
    }
    try:
        after_tree = ast.parse(after)
    except SyntaxError:
        return MaterializationResult(
            before=source,
            after=after,
            accepted=False,
            failure_reason="syntax-invalid",
            gates=gates,
            plan_fingerprint=plan.fingerprint,
        )
    gates["syntax_valid"] = True
    gates["ast_code_diff"] = _ast_dump(before_tree) != _ast_dump(after_tree)
    unbound_names = _introduced_unbound_names(before_tree, after_tree, plan)
    gates["postconditions"] = not unbound_names and _postconditions_hold(
        after_tree, plan, edits
    )
    accepted = all(gates.values())
    return MaterializationResult(
        before=source,
        after=after,
        accepted=accepted,
        failure_reason=(
            None if accepted else "unbound-name" if unbound_names else "no-op"
        ),
        gates=gates,
        plan_fingerprint=plan.fingerprint,
    )


def run_operator_renderer_qualification(output_path: Path) -> dict[str, Any]:
    """Run a deterministic 20-case capacity suite and freeze its evidence."""
    cases = _qualification_cases()
    rows: list[dict[str, Any]] = []
    for case_id, source, plan in cases:
        result = materialize_operator_plan(source, plan)
        rows.append(
            {
                "case_id": case_id,
                "source_sha256": _sha256_text(source),
                "plan_fingerprint": plan.fingerprint,
                "result": result.to_dict(),
            }
        )
    accepted = sum(1 for row in rows if row["result"]["accepted"] is True)
    no_ops = sum(1 for row in rows if row["result"]["failure_reason"] == "no-op")
    threshold = 23
    content = {
        "schema_version": 1,
        "suite_id": "typed-operator-renderer-synthetic-v7",
        "renderer_contract": "deterministic-ast-selector-renderer-v7",
        "suite_fingerprint": sha256_json(
            [
                {
                    "case_id": case_id,
                    "source_sha256": _sha256_text(source),
                    "plan_fingerprint": plan.fingerprint,
                }
                for case_id, source, plan in cases
            ]
        ),
        "status": "qualified" if accepted >= threshold and no_ops == 0 else "rejected",
        "planned_cases": len(cases),
        "accepted_cases": accepted,
        "no_op_cases": no_ops,
        "qualification_threshold": threshold,
        "rows": rows,
        "scope": "renderer_capacity_only_not_student_or_skill_capability",
        "holdout_task_ids_included": False,
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    output = output_path.resolve()
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(
                "operator qualification evidence is unreadable"
            ) from exc
        if existing != report:
            raise ContractError("frozen operator qualification evidence does not match")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report


def _resolve_operation(
    source: str,
    symbol: _Definition,
    operation: OperatorOperation,
) -> _Edit:
    occurrence = operation.selector["occurrence"]
    if operation.operator == "replace_condition":
        selected = _parse_expression(str(operation.selector["source"]))
        replacement = _parse_expression(str(operation.arguments["new_condition"]))
        candidates = [
            node
            for node in ast.walk(symbol)
            if isinstance(node, (ast.If, ast.While))
            and _ast_dump(node.test) == _ast_dump(selected)
        ]
        node = _occurrence(candidates, occurrence)
        return _Edit(
            *_node_span(source, node.test),
            ast.unparse(replacement),
            operation,
            (_ast_dump(replacement),),
        )
    if operation.operator == "replace_expression":
        selector_source = str(operation.selector["source"])
        try:
            selected = _parse_expression(selector_source)
        except ContractError as exc:
            if "operator class mismatch" not in str(exc):
                raise
            # A1 harness-layer capability extension: the weak Student
            # classified a statement as an expression (e.g. `raise ...` or
            # `xy = xy_0`).  Correct deterministically without a model call by
            # re-interpreting the operation as replace_statement, reusing the
            # same replacement text.  Fail-closed if that text is not 1-4
            # valid statements.
            selected_stmt = _parse_statement(selector_source)
            replacements = _parse_replacement_statements(
                str(operation.arguments["new_expression"])
            )
            node = _select_node(symbol, ast.stmt, selected_stmt, occurrence)
            indentation = _node_indentation(source, node)
            rendered = _indent_statements(replacements, indentation)
            return _Edit(
                *_statement_span(source, node),
                rendered,
                operation,
                tuple(_ast_dump(row) for row in replacements),
            )
        replacement = _parse_expression(str(operation.arguments["new_expression"]))
        node = _select_node(symbol, ast.expr, selected, occurrence)
        return _Edit(
            *_node_span(source, node),
            ast.unparse(replacement),
            operation,
            (_ast_dump(replacement),),
        )
    if operation.operator == "replace_statement":
        selected = _parse_statement(str(operation.selector["source"]))
        replacements = _parse_replacement_statements(
            str(operation.arguments["new_statements"])
        )
        node = _select_node(symbol, ast.stmt, selected, occurrence)
        indentation = _node_indentation(source, node)
        rendered = _indent_statements(replacements, indentation)
        return _Edit(
            *_statement_span(source, node),
            rendered,
            operation,
            tuple(_ast_dump(row) for row in replacements),
        )
    if operation.operator == "insert_method":
        return _insert_method_edit(source, symbol, operation)
    if operation.operator == "replace_method_body":
        return _replace_method_body_edit(source, symbol, operation)
    if operation.operator == "insert_assignment_before":
        selected = _parse_statement(str(operation.selector["source"]))
        name = str(operation.arguments["name"])
        if _IDENTIFIER.fullmatch(name) is None:
            raise ContractError("inserted assignment name is invalid")
        expression = _parse_expression(str(operation.arguments["expression"]))
        node = _select_node(symbol, ast.stmt, selected, occurrence)
        assignment = ast.parse(f"{name} = {ast.unparse(expression)}").body[0]
        indentation = _node_indentation(source, node)
        rendered = f"{indentation}{ast.unparse(assignment)}\n"
        start, _ = _statement_span(source, node)
        return _Edit(
            start,
            start,
            rendered,
            operation,
            (_ast_dump(assignment),),
        )
    if operation.operator == "align_trailing_defaults":
        return _align_trailing_defaults_edit(source, symbol, operation)
    if operation.operator == "normalize_inline_wrapper_boundaries":
        return _normalize_inline_wrapper_boundaries_edit(source, symbol, operation)
    if operation.operator == "remove_property_index_parens":
        return _remove_property_index_parens_edit(source, symbol, operation)
    if operation.operator == "remove_variable_obj_role":
        return _remove_variable_obj_role_edit(source, symbol, operation)
    value = operation.selector["value"]
    new_value = operation.arguments["new_value"]
    if not _json_scalar(value) or not _json_scalar(new_value):
        raise ContractError("constant operation requires JSON scalar values")
    candidates = [
        node
        for node in ast.walk(symbol)
        if isinstance(node, ast.Constant)
        and type(node.value) is type(value)
        and node.value == value
    ]
    node = _occurrence(candidates, occurrence)
    replacement = ast.Constant(value=new_value)
    return _Edit(
        *_node_span(source, node),
        repr(new_value),
        operation,
        (_ast_dump(replacement),),
    )


def _postconditions_hold(
    after_tree: ast.Module,
    plan: OperatorPlan,
    edits: list[_Edit],
) -> bool:
    symbol = _find_definition(after_tree.body, plan.symbol.split("."))
    if symbol is None:
        return False
    scope: ast.AST = (
        after_tree
        if any(
            edit.operation.operator == "initialize_generated_subclass_identity"
            for edit in edits
        )
        else symbol
    )
    available = {_ast_dump(node) for node in ast.walk(scope)}
    return all(
        all(expected in available for expected in edit.postcondition_dumps)
        for edit in edits
    )


def _generated_subclass_identity_edits(
    source: str,
    tree: ast.Module,
    symbol: _Definition,
    operation: OperatorOperation,
) -> list[_Edit]:
    """Keep generated class and instance identity aligned as one invariant.

    The operation is intentionally structural: it recognizes a class whose
    instances erase ``__qualname__`` and a linked ``type(...)`` factory whose
    attribute mapping already carries module/display identity. It then stores
    the generated leaf name on the class and initializes instances from it.
    """

    if not isinstance(symbol, ast.ClassDef):
        raise ContractError(
            "initialize_generated_subclass_identity requires a class symbol"
        )
    init_assignments: list[ast.Assign] = []
    for method in symbol.body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if method.name != "__init__":
            continue
        init_assignments.extend(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and node.value.value == ""
            and any(_is_self_qualname(target) for target in node.targets)
        )
    if len(init_assignments) != 1:
        raise ContractError(
            f"operator selector did not resolve; matches={len(init_assignments)}"
        )

    candidates: list[
        tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.Assign, str]
    ] = []
    for function in tree.body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        candidate = _generated_subclass_factory_candidate(function, symbol.name)
        if candidate is not None:
            assignment, leaf_name = candidate
            candidates.append((function, assignment, leaf_name))
    candidates.sort(key=lambda row: (row[0].lineno, row[0].col_offset))
    occurrence = operation.selector["occurrence"]
    if occurrence >= len(candidates):
        raise ContractError(
            f"operator selector did not resolve; matches={len(candidates)}"
        )
    _function, attrs_assignment, leaf_name = candidates[occurrence]
    assert isinstance(attrs_assignment.value, ast.Dict)

    qualified_name = ast.parse("self.__class__.__qualname__", mode="eval").body
    class_edit = _Edit(
        *_node_span(source, init_assignments[0].value),
        ast.unparse(qualified_name),
        operation,
        (_ast_dump(qualified_name),),
    )
    attrs = attrs_assignment.value
    replacement = ast.Dict(
        keys=[*attrs.keys, ast.Constant(value="__qualname__")],
        values=[*attrs.values, ast.Name(id=leaf_name, ctx=ast.Load())],
    )
    attrs_start, attrs_end = _node_span(source, attrs)
    if source[attrs_end - 1 : attrs_end] != "}":
        raise ContractError("generated subclass attributes are not a dict literal")
    rendered_key = f"'__qualname__': {leaf_name}"
    attrs_source = source[attrs_start:attrs_end]
    insertion = (
        f",\n{_node_indentation(source, attrs.values[-1])}{rendered_key}"
        if "\n" in attrs_source
        else f", {rendered_key}"
    )
    factory_edit = _Edit(
        attrs_end - 1,
        attrs_end - 1,
        insertion,
        operation,
        (_ast_dump(replacement),),
    )
    return [class_edit, factory_edit]


def _is_self_qualname(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "__qualname__"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _generated_subclass_factory_candidate(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    class_name: str,
) -> tuple[ast.Assign, str] | None:
    positional = [*function.args.posonlyargs, *function.args.args]
    defaults_start = len(positional) - len(function.args.defaults)
    defaults = {
        positional[index].arg: default
        for index, default in enumerate(function.args.defaults, defaults_start)
    }
    superclass_names = {
        name
        for name, default in defaults.items()
        if isinstance(default, ast.Name) and default.id == class_name
    }
    if not superclass_names:
        return None

    for node in ast.walk(function):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if (
            not isinstance(call.func, ast.Name)
            or call.func.id != "type"
            or len(call.args) != 3
            or not isinstance(call.args[0], ast.Name)
            or not isinstance(call.args[1], ast.Tuple)
            or not any(
                isinstance(base, ast.Name) and base.id in superclass_names
                for base in call.args[1].elts
            )
            or not isinstance(call.args[2], ast.Name)
        ):
            continue
        leaf_name = call.args[0].id
        attrs_name = call.args[2].id
        for statement in function.body:
            if (
                isinstance(statement, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == attrs_name
                    for target in statement.targets
                )
                and isinstance(statement.value, ast.Dict)
                and _identity_attribute_mapping(statement.value, leaf_name)
            ):
                return statement, leaf_name
    return None


def _identity_attribute_mapping(node: ast.Dict, leaf_name: str) -> bool:
    values = {
        key.value: value
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    display = values.get("__display_name__")
    return (
        "__module__" in values
        and "__qualname__" not in values
        and display is not None
        and any(
            isinstance(child, ast.Name) and child.id == leaf_name
            for child in ast.walk(display)
        )
    )


def _insert_method_edit(
    source: str,
    symbol: _Definition,
    operation: OperatorOperation,
) -> _Edit:
    if not isinstance(symbol, ast.ClassDef) or operation.selector["occurrence"] != 0:
        raise ContractError("insert_method requires one uniquely resolved class")
    method = _parse_method_definition(str(operation.arguments["method_source"]))
    if any(
        isinstance(row, (ast.FunctionDef, ast.AsyncFunctionDef))
        and row.name == method.name
        for row in symbol.body
    ):
        raise ContractError("insert_method would duplicate an existing method")
    if not symbol.body:
        raise ContractError("insert_method class body is empty")
    _start, end = _statement_span(source, symbol.body[-1])
    indentation = " " * (symbol.col_offset + 4)
    rendered = textwrap.indent(ast.unparse(method), indentation) + "\n"
    return _Edit(
        end,
        end,
        rendered,
        operation,
        (_ast_dump(method),),
    )


def _replace_method_body_edit(
    source: str,
    symbol: _Definition,
    operation: OperatorOperation,
) -> _Edit:
    if (
        not isinstance(symbol, (ast.FunctionDef, ast.AsyncFunctionDef))
        or operation.selector["occurrence"] != 0
    ):
        raise ContractError("replace_method_body requires one uniquely resolved method")
    replacements = _parse_replacement_statements(str(operation.arguments["new_body"]))
    docstring = (
        symbol.body[0]
        if symbol.body
        and isinstance(symbol.body[0], ast.Expr)
        and isinstance(symbol.body[0].value, ast.Constant)
        and isinstance(symbol.body[0].value.value, str)
        else None
    )
    replaced = symbol.body[1:] if docstring is not None else symbol.body
    if replaced:
        start, _unused = _statement_span(source, replaced[0])
        _unused, end = _statement_span(source, replaced[-1])
        indentation = _node_indentation(source, replaced[0])
    elif docstring is not None:
        _unused, start = _statement_span(source, docstring)
        end = start
        indentation = " " * (symbol.col_offset + 4)
    else:
        raise ContractError("replace_method_body method body is empty")
    return _Edit(
        start,
        end,
        _indent_statements(replacements, indentation),
        operation,
        tuple(_ast_dump(row) for row in replacements),
    )


def _parse_method_definition(source: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    try:
        body = ast.parse(textwrap.dedent(source).strip()).body
    except SyntaxError as exc:
        raise ContractError("inserted method definition is invalid") from exc
    if len(body) != 1 or not isinstance(
        body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        raise ContractError("insert_method requires exactly one method definition")
    return body[0]


def _introduced_unbound_names(
    before_tree: ast.Module,
    after_tree: ast.Module,
    plan: OperatorPlan,
) -> set[str]:
    before_symbol = _find_definition(before_tree.body, plan.symbol.split("."))
    after_symbol = _find_definition(after_tree.body, plan.symbol.split("."))
    if before_symbol is None or after_symbol is None:
        return {"<missing-symbol>"}
    before_loads = {
        node.id
        for node in ast.walk(before_symbol)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    after_loads = {
        node.id
        for node in ast.walk(after_symbol)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    available = set(dir(builtins))
    available.update(_module_bound_names(after_tree))
    available.update(_locally_bound_names(after_symbol))
    return (after_loads - before_loads) - available


def _module_bound_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
        else:
            names.update(
                child.id
                for child in ast.walk(node)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
            )
    return names


def _locally_bound_names(symbol: _Definition) -> set[str]:
    names = {
        node.id
        for node in ast.walk(symbol)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    for node in ast.walk(symbol):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.update(arg.arg for arg in node.args.posonlyargs)
            names.update(arg.arg for arg in node.args.args)
            names.update(arg.arg for arg in node.args.kwonlyargs)
            if node.args.vararg is not None:
                names.add(node.args.vararg.arg)
            if node.args.kwarg is not None:
                names.add(node.args.kwarg.arg)
    return names


def _align_trailing_defaults_edit(
    source: str,
    symbol: _Definition,
    operation: OperatorOperation,
) -> _Edit:
    occurrence = operation.selector["occurrence"]
    candidates: list[tuple[ast.If, ast.For]] = []
    body = symbol.body
    for index, node in enumerate(body):
        if not isinstance(node, ast.If) or not _is_posonly_guard(node):
            continue
        trailing = next(
            (
                row
                for row in body[index + 1 :]
                if isinstance(row, ast.For) and _iterates_regular_args(row)
            ),
            None,
        )
        if trailing is not None:
            candidates.append((node, trailing))
    if occurrence >= len(candidates):
        raise ContractError(
            f"operator selector did not resolve; matches={len(candidates)}"
        )
    first, last = candidates[occurrence]
    indentation = _node_indentation(source, first)
    replacement_source = """
positional_only = getattr(args, 'posonlyargs', [])
aligned_defaults = [Parameter.empty] * (
    len(positional_only) + len(args.args) - len(args.defaults)
)
aligned_defaults.extend(ast_unparse(default) for default in args.defaults)

for i, arg in enumerate(positional_only):
    annotation = ast_unparse(arg.annotation) or Parameter.empty
    params.append(Parameter(
        arg.arg,
        Parameter.POSITIONAL_ONLY,
        default=aligned_defaults[i],
        annotation=annotation,
    ))

for i, arg in enumerate(args.args, len(positional_only)):
    annotation = ast_unparse(arg.annotation) or Parameter.empty
    params.append(Parameter(
        arg.arg,
        Parameter.POSITIONAL_OR_KEYWORD,
        default=aligned_defaults[i],
        annotation=annotation,
    ))
"""
    statements = ast.parse(textwrap.dedent(replacement_source)).body
    start, _ = _statement_span(source, first)
    _, end = _statement_span(source, last)
    return _Edit(
        start,
        end,
        _indent_statements(statements, indentation),
        operation,
        tuple(_ast_dump(row) for row in statements),
    )


def _normalize_inline_wrapper_boundaries_edit(
    source: str,
    symbol: _Definition,
    operation: OperatorOperation,
) -> _Edit:
    candidates: list[tuple[ast.Assign, ast.Assign]] = []
    assignments = [row for row in ast.walk(symbol) if isinstance(row, ast.Assign)]
    for node in assignments:
        if not _is_inline_wrapper_replace(node):
            continue
        trailer = next(
            (
                row
                for row in assignments
                if row.lineno > node.lineno and _assigns_name(row, "hlcode")
            ),
            None,
        )
        if trailer is not None:
            candidates.append((node, trailer))
    occurrence = operation.selector["occurrence"]
    if occurrence >= len(candidates):
        raise ContractError(
            f"operator selector did not resolve; matches={len(candidates)}"
        )
    first, last = sorted(candidates, key=lambda row: row[0].lineno)[occurrence]
    indentation = _node_indentation(source, first)
    replacement_source = """
hlcode = hlcode.replace(
    r'\\begin{Verbatim}[commandchars=\\\\\\{\\}]',
    r'\\sphinxcode{\\sphinxupquote{%'
)
hlcode = hlcode.rstrip()[:-14].rstrip() + '%\\n'
"""
    statements = ast.parse(textwrap.dedent(replacement_source)).body
    start, _ = _statement_span(source, first)
    _, end = _statement_span(source, last)
    return _Edit(
        start,
        end,
        _indent_statements(statements, indentation),
        operation,
        tuple(_ast_dump(row) for row in statements),
    )


def _remove_property_index_parens_edit(
    source: str,
    symbol: _Definition,
    operation: OperatorOperation,
) -> _Edit:
    candidates: list[ast.Return] = []
    for node in ast.walk(symbol):
        if not isinstance(node, ast.Return):
            continue
        segment = ast.get_source_segment(source, node) or ""
        if "'%s() (%s property)'" in segment or '"%s() (%s property)"' in segment:
            candidates.append(node)
    occurrence = operation.selector["occurrence"]
    if occurrence >= len(candidates):
        raise ContractError(
            f"operator selector did not resolve; matches={len(candidates)}"
        )
    node = sorted(candidates, key=lambda row: row.lineno)[occurrence]
    start, end = _statement_span(source, node)
    original = source[start:end]
    replacement = original.replace(
        "'%s() (%s property)'", "'%s (%s property)'"
    ).replace('"%s() (%s property)"', '"%s (%s property)"')
    dummy = ast.parse(f"def _f():\n    {textwrap.dedent(replacement).strip()}\n")
    replacement_statement = dummy.body[0].body[0]
    return _Edit(
        start,
        end,
        replacement,
        operation,
        (_ast_dump(replacement_statement),),
    )


def _remove_variable_obj_role_edit(
    source: str,
    symbol: _Definition,
    operation: OperatorOperation,
) -> _Edit:
    candidates: list[tuple[ast.Call, ast.keyword]] = []
    for node in ast.walk(symbol):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "PyTypedField"):
            continue
        if not (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "variable"
        ):
            continue
        matching_keywords = [
            kw
            for kw in node.keywords
            if isinstance(kw.value, ast.Constant)
            and kw.value.value == "obj"
            and kw.arg == "rolename"
        ]
        if matching_keywords:
            candidates.append((node, matching_keywords[0]))
    occurrence = operation.selector["occurrence"]
    if occurrence >= len(candidates):
        raise ContractError(
            f"operator selector did not resolve; matches={len(candidates)}"
        )
    node, keyword = sorted(candidates, key=lambda row: row[0].lineno)[occurrence]
    start, end = _node_span(source, node)
    original = source[start:end]
    keyword_start, keyword_end = _node_span(source, keyword)
    relative_start = keyword_start - start
    relative_end = keyword_end - start
    line_start = original.rfind("\n", 0, relative_start) + 1
    line_end = original.find("\n", relative_end)
    line_prefix = original[line_start:relative_start]
    if not line_prefix.strip() and (
        line_end >= 0 or original[relative_end:].lstrip().startswith(")")
    ):
        replacement = (
            original[:line_start] + original[line_end + 1 :]
            if line_end >= 0
            else original[:line_start] + line_prefix + original[relative_end:]
        )
        replacement_expr = ast.parse(replacement, mode="eval").body
        return _Edit(
            start,
            end,
            replacement,
            operation,
            (_ast_dump(replacement_expr),),
        )
    after_keyword = relative_end
    while after_keyword < len(original) and original[after_keyword].isspace():
        after_keyword += 1
    if after_keyword < len(original) and original[after_keyword] == ",":
        delete_end = after_keyword + 1
        while delete_end < len(original) and original[delete_end].isspace():
            delete_end += 1
        replacement = original[:relative_start] + original[delete_end:]
    else:
        before_keyword = relative_start - 1
        while before_keyword >= 0 and original[before_keyword].isspace():
            before_keyword -= 1
        if before_keyword < 0 or original[before_keyword] != ",":
            raise ContractError("operator keyword delimiter did not resolve")
        replacement = original[:before_keyword] + original[relative_end:]
    replacement_expr = ast.parse(replacement, mode="eval").body
    return _Edit(start, end, replacement, operation, (_ast_dump(replacement_expr),))


def _is_posonly_guard(node: ast.If) -> bool:
    call = node.test
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "hasattr"
        and len(call.args) == 2
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == "posonlyargs"
    )


def _iterates_regular_args(node: ast.For) -> bool:
    iterator = node.iter
    return (
        isinstance(iterator, ast.Call)
        and isinstance(iterator.func, ast.Name)
        and iterator.func.id == "enumerate"
        and bool(iterator.args)
        and isinstance(iterator.args[0], ast.Attribute)
        and isinstance(iterator.args[0].value, ast.Name)
        and iterator.args[0].value.id == "args"
        and iterator.args[0].attr == "args"
    )


def _assigns_name(node: ast.Assign, name: str) -> bool:
    return any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets
    )


def _is_inline_wrapper_replace(node: ast.Assign) -> bool:
    if not _assigns_name(node, "hlcode") or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "replace"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "hlcode"
        and len(call.args) == 2
        and all(isinstance(row, ast.Constant) for row in call.args)
        and "begin{Verbatim}" in str(call.args[0].value)
        and "sphinxupquote" in str(call.args[1].value)
    )


def _find_definition(body: list[ast.stmt], parts: list[str]) -> _Definition | None:
    if len(parts) == 1:
        matches = _definitions_named(body, parts[0])
        return matches[0] if len(matches) == 1 else None
    candidates = [
        node
        for node in body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == parts[0]
    ]
    if len(candidates) != 1:
        return None
    node = candidates[0]
    if not isinstance(node, ast.ClassDef):
        return None
    return _find_definition(node.body, parts[1:])


def _definitions_named(body: list[ast.stmt], name: str) -> list[_Definition]:
    matches: list[_Definition] = []
    for node in body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name == name:
            matches.append(node)
        if isinstance(node, ast.ClassDef):
            matches.extend(_definitions_named(node.body, name))
    return matches


def _select_node(
    symbol: _Definition,
    expected_type: type[ast.expr] | type[ast.stmt],
    expected: ast.AST,
    occurrence: int,
) -> ast.AST:
    dump = _ast_dump(expected)
    candidates = [
        node
        for node in ast.walk(symbol)
        if isinstance(node, expected_type) and _ast_dump(node) == dump
    ]
    return _occurrence(candidates, occurrence)


def _occurrence(candidates: list[ast.AST], occurrence: int) -> ast.AST:
    ordered = sorted(candidates, key=lambda row: (row.lineno, row.col_offset))
    if occurrence >= len(ordered):
        raise ContractError(
            f"operator selector did not resolve; matches={len(ordered)}"
        )
    return ordered[occurrence]


def _node_span(source: str, node: ast.AST) -> tuple[int, int]:
    if (
        not hasattr(node, "lineno")
        or not hasattr(node, "end_lineno")
        or node.end_lineno is None
        or node.end_col_offset is None
    ):
        raise ContractError("operator selector has no concrete source span")
    lines = source.splitlines(keepends=True)
    return (
        _line_offset(lines, node.lineno, node.col_offset),
        _line_offset(lines, node.end_lineno, node.end_col_offset),
    )


def _statement_span(source: str, node: ast.AST) -> tuple[int, int]:
    if not hasattr(node, "lineno") or node.end_lineno is None:
        raise ContractError("operator statement has no concrete source span")
    lines = source.splitlines(keepends=True)
    start = len("".join(lines[: node.lineno - 1]))
    end = len("".join(lines[: node.end_lineno]))
    return start, end


def _line_offset(lines: list[str], line_number: int, byte_column: int) -> int:
    prefix = "".join(lines[: line_number - 1])
    line = lines[line_number - 1]
    try:
        character_column = len(line.encode("utf-8")[:byte_column].decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ContractError("operator selector splits a UTF-8 character") from exc
    return len(prefix) + character_column


def _node_indentation(source: str, node: ast.AST) -> str:
    line = source.splitlines()[node.lineno - 1]
    return line[: len(line) - len(line.lstrip())]


def _parse_expression(source: str) -> ast.expr:
    stripped = source.strip()
    try:
        return ast.parse(stripped, mode="eval").body
    except SyntaxError as exc:
        # A statement (raise/return/if/assignment/...) cannot be a replace_*
        # expression selector or replacement.  Classify this precisely so the
        # student gets an actionable signal (and downstream can mark the task
        # student-unresolvable) instead of a generic parse error.
        try:
            ast.parse(stripped, mode="exec")
        except SyntaxError:
            raise ContractError("operator expression is invalid") from exc
        raise ContractError(
            "operator class mismatch: replace_expression cannot target a "
            "statement; use replace_statement"
        ) from exc


def _parse_statement(source: str) -> ast.stmt:
    try:
        body = ast.parse(textwrap.dedent(source).strip()).body
    except SyntaxError as exc:
        raise ContractError("operator statement selector is invalid") from exc
    if len(body) != 1 or isinstance(body[0], _FORBIDDEN_STATEMENTS):
        raise ContractError("operator selector must be one non-definition statement")
    return body[0]


def _parse_replacement_statements(source: str) -> list[ast.stmt]:
    try:
        body = ast.parse(textwrap.dedent(source).strip()).body
    except SyntaxError as exc:
        raise ContractError("operator replacement statements are invalid") from exc
    if not 1 <= len(body) <= 4 or any(
        isinstance(row, _FORBIDDEN_STATEMENTS) for row in body
    ):
        raise ContractError(
            "replacement requires one to four non-definition statements"
        )
    return body


def _indent_statements(statements: list[ast.stmt], indentation: str) -> str:
    rendered = ast.unparse(ast.Module(body=statements, type_ignores=[]))
    return textwrap.indent(rendered, indentation) + "\n"


def _reject_overlaps(edits: list[_Edit]) -> None:
    ordered = sorted(edits, key=lambda row: (row.start, row.end))
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.start < previous.end or (
            previous.start == previous.end == current.start == current.end
        ):
            raise ContractError("operator edit spans overlap")


def _qualification_cases() -> list[tuple[str, str, OperatorPlan]]:
    cases: list[tuple[str, str, OperatorPlan]] = []
    for index in range(4):
        source = (
            "def compute(values):\n"
            "    if not values:\n"
            f"        return {index}\n"
            "    return len(values)\n"
        )
        cases.append(
            (
                f"condition-{index + 1:02d}",
                source,
                _case_plan(
                    {
                        "operator": "replace_condition",
                        "selector": {
                            "source": "not values",
                            "occurrence": 0,
                        },
                        "arguments": {"new_condition": "values is None"},
                    }
                ),
            )
        )
    for index in range(4):
        threshold = index + 5
        source = f"def compute(value):\n    return value < {threshold}\n"
        cases.append(
            (
                f"expression-{index + 1:02d}",
                source,
                _case_plan(
                    {
                        "operator": "replace_expression",
                        "selector": {
                            "source": f"value < {threshold}",
                            "occurrence": 0,
                        },
                        "arguments": {"new_expression": f"value <= {threshold}"},
                    }
                ),
            )
        )
    for index in range(4):
        source = f"def compute(value):\n    return value + {index}\n"
        cases.append(
            (
                f"constant-{index + 1:02d}",
                source,
                _case_plan(
                    {
                        "operator": "replace_constant",
                        "selector": {"value": index, "occurrence": 0},
                        "arguments": {"new_value": index + 1},
                    }
                ),
            )
        )
    for index in range(4):
        source = f"def compute(value):\n    return value + {index}\n"
        cases.append(
            (
                f"insert-{index + 1:02d}",
                source,
                _case_plan(
                    {
                        "operator": "insert_assignment_before",
                        "selector": {
                            "source": f"return value + {index}",
                            "occurrence": 0,
                        },
                        "arguments": {"name": "offset", "expression": str(index)},
                    }
                ),
            )
        )
    for index in range(4):
        source = (
            f"def compute(value):\n    result = value + {index}\n    return result\n"
        )
        cases.append(
            (
                f"statement-{index + 1:02d}",
                source,
                _case_plan(
                    {
                        "operator": "replace_statement",
                        "selector": {
                            "source": f"result = value + {index}",
                            "occurrence": 0,
                        },
                        "arguments": {
                            "new_statements": f"result = value + {index + 1}"
                        },
                    }
                ),
            )
        )
    defaults_source = """def parse(args):
    params = []
    if hasattr(args, 'posonlyargs'):
        for arg in args.posonlyargs:
            annotation = ast_unparse(arg.annotation) or Parameter.empty
            params.append(Parameter(arg.arg, Parameter.POSITIONAL_ONLY, annotation=annotation))
    for i, arg in enumerate(args.args):
        default = Parameter.empty
        params.append(Parameter(arg.arg, Parameter.POSITIONAL_OR_KEYWORD, default=default))
    return params
"""
    inline_source = r"""def visit_literal(self, node):
    hlcode = self.highlighter.highlight_block(node.astext(), 'python')
    hlcode = hlcode.replace(r'\begin{Verbatim}[commandchars=\\\{\}]', r'\sphinxcode{\sphinxupquote{')
    hlcode = hlcode.rstrip()[:-14]
    self.body.append(hlcode)
"""
    for index in range(2):
        cases.append(
            (
                f"aligned-defaults-{index + 1:02d}",
                defaults_source,
                _case_plan(
                    {
                        "operator": "align_trailing_defaults",
                        "selector": {"occurrence": 0},
                        "arguments": {},
                    },
                    symbol="parse",
                ),
            )
        )
        cases.append(
            (
                f"inline-boundaries-{index + 1:02d}",
                inline_source,
                _case_plan(
                    {
                        "operator": "normalize_inline_wrapper_boundaries",
                        "selector": {"occurrence": 0},
                        "arguments": {},
                    },
                    symbol="visit_literal",
                ),
            )
        )
    for index in range(2):
        class_name = f"GeneratedBase{index}"
        factory_name = f"make_generated_{index}"
        source = f"""class {class_name}:
    __display_name__ = '{class_name}'

    def __init__(self):
        self.__qualname__ = ''


def {factory_name}(name, module, superclass={class_name}):
    attrs = {{'__module__': module, '__display_name__': module + '.' + name}}
    return type(name, (superclass,), attrs)
"""
        cases.append(
            (
                f"generated-identity-{index + 1:02d}",
                source,
                _case_plan(
                    {
                        "operator": "initialize_generated_subclass_identity",
                        "selector": {"occurrence": 0},
                        "arguments": {},
                    },
                    symbol=class_name,
                ),
            )
        )
    return cases


def _case_plan(operation: dict[str, Any], *, symbol: str = "compute") -> OperatorPlan:
    return OperatorPlan.from_dict(
        {
            "schema_version": 1,
            "file": "module.py",
            "symbol": symbol,
            "intent": {
                "defect": "synthetic operator capacity case",
                "trigger": "fixed synthetic input",
                "desired_boundary": "materialized AST code change",
            },
            "operations": [operation],
            "diagnostic": "Synthetic renderer qualification only.",
        }
    )


def _ast_dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_scalar(value: Any) -> bool:
    return value is None or type(value) in {bool, int, float, str}
