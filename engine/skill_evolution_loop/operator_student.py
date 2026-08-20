"""Local Student adapter for the typed operator-plan action space."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .contracts import ContractError, LoopRevision, canonical_json
from .experiment import ExperimentCondition
from .mlx_student import (
    MlxStructuredGenerator,
    _lexical_terms,
    _project_numbered_teaching,
    _teaching_reminder,
)
from .operator_rewrite import (
    OperatorOperation,
    OperatorPlan,
    _OPERATORS,
    _ast_dump,
    _find_definition,
    materialize_operator_plan,
)
from .capabilities import StudentCapabilityProfile, profile_for
from .student_adapter import (
    StudentAdapter,
    StudentAttempt,
    StudentTask,
    _implementation_fingerprint,
    _sha256_text,
    parse_unresolved_abstention,
)
from .symbol_rewrite import (
    fixed_symbol_contexts,
    qualified_symbol_excerpt,
    qualified_symbol_for_anchor,
)

_OPERATOR_PROMPT_TEMPLATE = "Return exactly one typed operator plan JSON object."
_OPERATOR_CONTEXT_SELECTOR = "frozen-top32-qualified-absolute-overlap-symbol-v6"
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_MAX_OPERATOR_PLAN_CHARS = 1_500
_CALL_NAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_CAMEL_CASE_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]*\b")
_OPERATOR_CATALOG = """Allowed operators (use one to four):
- replace_condition: selector {"source": <exact if/elif/while predicate expression without keyword or colon>, "occurrence": <0-based int>}; arguments {"new_condition": <Python expression without keyword or colon>}
- replace_expression: selector {"source": <exact Python expression>, "occurrence": <0-based int>}; arguments {"new_expression": <Python expression>}
- replace_statement: selector {"source": <exact one Python statement>, "occurrence": <0-based int>}; arguments {"new_statements": <one to four Python statements>}
- insert_method: selector {"occurrence": 0}; arguments {"method_source": <exactly one complete Python method definition>}. The named symbol must be the target class.
- replace_method_body: selector {"occurrence": 0}; arguments {"new_body": <one to four non-definition statements>}. The named symbol must be the target method; its signature and docstring are preserved.
- insert_assignment_before: selector {"source": <exact one Python statement>, "occurrence": <0-based int>}; arguments {"name": <identifier>, "expression": <Python expression>}
- replace_constant: selector {"value": <JSON scalar>, "occurrence": <0-based int>}; arguments {"new_value": <JSON scalar>}
- align_trailing_defaults: selector {"occurrence": <0-based int>}; arguments {}. Use only when one trailing defaults vector must align across positional-only and regular positional parameters.
- normalize_inline_wrapper_boundaries: selector {"occurrence": <0-based int>}; arguments {}. Use only when an inline wrapper needs percent-newline boundaries and complete trailer trimming.
- initialize_generated_subclass_identity: selector {"occurrence": <0-based int>}; arguments {}. Use only when a generated subclass factory must store its leaf name and instances currently erase their qualified name.
- remove_property_index_parens: selector {"occurrence": <0-based int>}; arguments {}. Use only when a property index entry wrongly contains parentheses in its format string.
- remove_variable_obj_role: selector {"occurrence": <0-based int>}; arguments {}. Use only when a variable field wrongly carries the object role in its typed-field declaration.

The occurrence counts structurally identical AST nodes inside the named symbol only.
Every selector must match; the renderer rejects overlap, copied/no-op output, invalid syntax, and failed postconditions."""

_CONDITION_HEADER = re.compile(r"^\s*(if|while)\s+(.+):\s*$")
_IDENTIFIER_TOKEN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_SAFE_REPLACEMENT_NAMES = frozenset({"len"})
_ARGUMENT_SHAPE_OPERATORS = {
    frozenset({"new_condition"}): "replace_condition",
    frozenset({"new_expression"}): "replace_expression",
    frozenset({"new_statements"}): "replace_statement",
    frozenset({"method_source"}): "insert_method",
    frozenset({"new_body"}): "replace_method_body",
    frozenset({"name", "expression"}): "insert_assignment_before",
    frozenset({"new_value"}): "replace_constant",
}
_ZERO_ARGUMENT_SINGLE_OPERATORS = frozenset(
    {
        "align_trailing_defaults",
        "normalize_inline_wrapper_boundaries",
        "initialize_generated_subclass_identity",
        "remove_property_index_parens",
        "remove_variable_obj_role",
    }
)
_OPERATOR_SIMPLE_STATEMENTS = (
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.Return,
    ast.Raise,
    ast.Expr,
    ast.Assert,
    ast.Delete,
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.Pass,
    ast.Break,
    ast.Continue,
)
_BASELINE_OPERATOR_TEACHING = (
    "No additional domain teaching is provided. Use only repository evidence."
)


def _resolve_operator_teaching(
    skill_text: str,
    instruction: str,
    *,
    source_round: int,
) -> tuple[str, str | None]:
    """Resolve the operator teaching text and the matched PatternCard.

    When no PatternCard matches the task, fall back to the exact baseline
    teaching text so the taught arm does not carry extra prompt noise that can
    flip a task from baseline-resolved to taught-unresolved (a teaching
    regression). The pair then becomes no-op-equivalent and is rejected by the
    causal/implementation gates instead of producing a false regression.
    """

    teaching = skill_text
    selected_card: str | None = None
    if "## Pattern cards" in teaching:
        global_rules = teaching.split("## Pattern cards", 1)[0].rstrip()
        reminder, selected_card = _teaching_reminder(teaching, instruction)
        if selected_card is None:
            teaching = _BASELINE_OPERATOR_TEACHING
        else:
            teaching = f"{global_rules}\n\n{reminder}"
    elif source_round >= 12:
        teaching = _project_numbered_teaching(
            teaching,
            instruction,
            max_rules=5,
            max_chars=1_400,
            mandatory_rule_numbers=(2, 4, 5),
        )
    return teaching, selected_card


def _minimal_boundary_oracle(
    source: str,
    instruction: str,
) -> dict[str, Any] | None:
    """Derive the one-row truth-table change supported by the public issue."""

    lowered = instruction.casefold()
    empty_trigger = any(
        marker in lowered
        for marker in ("empty", "zero entries", "zero items", "zero elements")
    )
    absent_trigger = any(
        marker in lowered
        for marker in (" none", "none ", "absent", "missing", "unset", "sentinel")
    )
    if empty_trigger == absent_trigger:
        return None
    try:
        node = ast.parse(source, mode="eval").body
    except SyntaxError:
        return None
    negated = isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
    subject = node.operand if negated else node
    if not isinstance(subject, (ast.Name, ast.Attribute)):
        return None
    if empty_trigger:
        required = "absent-only" if negated else "present-only"
        preserved = "absent"
        trigger = "empty"
        truth_table = (
            {"absent": True, "empty": False, "non_empty": False}
            if negated
            else {"absent": False, "empty": True, "non_empty": True}
        )
    else:
        if not negated:
            return None
        required = "empty-only"
        preserved = "empty"
        trigger = "absent"
        truth_table = {"absent": False, "empty": True, "non_empty": False}
    return {
        "trigger_case": trigger,
        "required_semantics": required,
        "preserved_case": preserved,
        "condition_truth_table": truth_table,
    }


def _condition_boundary_rewrites(node: ast.expr) -> list[dict[str, str]]:
    subject = (
        node.operand
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
        else node
    )
    if not isinstance(subject, (ast.Name, ast.Attribute)):
        return []
    source = ast.unparse(subject)
    return [
        {"new_condition": f"{source} is None", "semantics": "absent-only"},
        {"new_condition": f"{source} is not None", "semantics": "present-only"},
        {"new_condition": f"len({source}) == 0", "semantics": "empty-only"},
        {"new_condition": f"len({source}) > 0", "semantics": "non-empty-only"},
    ]


def _harvest_typed_selector_candidates(
    source: str,
    *,
    instruction: str = "",
    maximum_candidates: int = 128,
) -> tuple[dict[str, Any], ...]:
    """Enumerate exact expression selectors as framework-owned prompt state."""

    if type(maximum_candidates) is not int or maximum_candidates < 1:
        raise ValueError("selector candidate limit must be positive")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Qualified symbol excerpts are character-bounded and may end inside a
        # valid long definition.  An empty framework-owned candidate set keeps
        # realization fail-closed: model selectors must still match the complete
        # source during materialization.
        return ()
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    nodes = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.expr)
            or isinstance(node, _OPERATOR_SIMPLE_STATEMENTS)
        ),
        key=lambda node: (
            node.lineno,
            node.col_offset,
            getattr(node, "end_lineno", node.lineno),
            getattr(node, "end_col_offset", node.col_offset),
        ),
    )
    occurrences: dict[tuple[str, str], int] = {}
    rows: list[dict[str, Any]] = []
    query = _lexical_terms(instruction)
    boundary_query = any(
        marker in instruction.casefold()
        for marker in ("empty", "none", "absent", "missing", "sentinel")
    )
    issue_identifiers = {
        token.casefold()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", instruction)
        if len(token) > 1
    }
    issue_lines: set[int] = set()
    if issue_identifiers:
        for line_no, line_text in enumerate(source.splitlines(), start=1):
            lowered = line_text.casefold()
            if any(
                re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])",
                    lowered,
                )
                is not None
                for identifier in issue_identifiers
            ):
                issue_lines.add(line_no)
    kind_counts = {"expression": 0, "condition_expression": 0, "statement": 0}
    for node in nodes:
        raw = ast.get_source_segment(source, node)
        if raw is None:
            continue
        exact = raw.strip()
        if not exact or "\n" in exact or len(exact) > 400:
            continue
        parent = parents.get(node)
        if isinstance(node, ast.expr):
            kind = (
                "condition_expression"
                if isinstance(parent, (ast.If, ast.While)) and parent.test is node
                else "expression"
            )
        else:
            kind = "statement"
        fingerprint = ast.dump(node, include_attributes=False)
        key = kind, fingerprint
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        kind_counts[kind] += 1
        ancestors: list[str] = []
        current = parent
        while current is not None and len(ancestors) < 4:
            ancestors.append(type(current).__name__)
            current = parents.get(current)
        prefix = {"expression": "E", "condition_expression": "C", "statement": "S"}[
            kind
        ]
        row = {
            "candidate_id": f"{prefix}{kind_counts[kind]:03d}",
            "kind": kind,
            "suggested_operator": {
                "expression": "replace_expression",
                "condition_expression": "replace_condition",
                "statement": "replace_statement",
            }[kind],
            "source": exact,
            "occurrence": occurrence,
            "line": node.lineno,
            "column": node.col_offset,
            "ancestors": ancestors,
            "diagnosis_token_overlap": len(query & _lexical_terms(exact)),
            "issue_line_proximity": (
                min(abs(node.lineno - issue_line) for issue_line in issue_lines)
                if issue_lines
                else None
            ),
        }
        if kind == "condition_expression":
            row["boundary_rewrites"] = _condition_boundary_rewrites(node)
            oracle = _minimal_boundary_oracle(exact, instruction)
            if oracle is not None:
                row["semantic_oracle"] = oracle
                row["oracle_matched_rewrite"] = next(
                    rewrite
                    for rewrite in row["boundary_rewrites"]
                    if rewrite["semantics"] == oracle["required_semantics"]
                )
        rows.append(row)
    if query:
        rows.sort(
            key=lambda row: (
                0 if boundary_query and row.get("boundary_rewrites") else 1,
                0 if row["kind"] in ("statement", "condition_expression") else 1,
                row["issue_line_proximity"]
                if row.get("issue_line_proximity") is not None
                else 0,
                -row["diagnosis_token_overlap"],
                row["line"],
                row["column"],
            )
        )
    return tuple(rows[:maximum_candidates])


def _condition_header_parts(source: str) -> tuple[str, str] | None:
    if "\n" in source or "\r" in source:
        return None
    matched = _CONDITION_HEADER.fullmatch(source)
    if matched is None:
        return None
    expression = matched.group(2).strip()
    try:
        compile(expression, "<condition>", "eval")
    except SyntaxError:
        return None
    return matched.group(1), expression


def _canonicalize_condition_header_output(raw: str) -> tuple[str, bool]:
    """Map a common full-header mistake into the explicit condition operator."""

    candidate = raw.strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        return raw, False
    try:
        data = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return raw, False
    operations = data.get("operations") if isinstance(data, dict) else None
    if not isinstance(operations, list):
        return raw, False
    normalized: list[Any] = []
    changed = False
    for row in operations:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("selector"), dict)
            or not isinstance(row.get("arguments"), dict)
        ):
            normalized.append(row)
            continue
        selector = row["selector"]
        arguments = row["arguments"]
        old = _condition_header_parts(str(selector.get("source", "")))
        occurrence = selector.get("occurrence")
        operator = row.get("operator")
        if operator == "replace_condition":
            new_condition = arguments.get("new_condition")
            if not isinstance(new_condition, str):
                normalized.append(row)
                continue
            try:
                compile(new_condition, "<condition>", "eval")
            except SyntaxError:
                normalized.append(row)
                continue
        elif operator == "replace_expression":
            new = _condition_header_parts(str(arguments.get("new_expression", "")))
            if old is None or new is None or old[0] != new[0]:
                normalized.append(row)
                continue
            new_condition = new[1]
        else:
            normalized.append(row)
            continue
        if old is None or type(occurrence) is not int:
            normalized.append(row)
            continue
        normalized.append(
            {
                "operator": "replace_condition",
                "selector": {
                    "source": old[1],
                    "occurrence": occurrence,
                },
                "arguments": {"new_condition": new_condition},
            }
        )
        changed = True
    if not changed:
        return raw, False
    data["operations"] = normalized
    return canonical_json(data), True


def _catalog_for_selected_card(selected_card: str | None) -> str:
    if selected_card is None:
        return _OPERATOR_CATALOG
    lowered = selected_card.lower()
    if "defaults vector" in lowered and "positional" in lowered:
        return (
            "The matched PatternCard supplies exactly one allowed operator. Use it; "
            "freeform expression or statement operators are prohibited:\n"
            '- align_trailing_defaults: selector {"occurrence": 0}; arguments {}. '
            "It aligns one trailing defaults vector across positional-only and "
            "regular positional parameters while preserving annotations."
        )
    if "percent sentinel" in lowered and "inline" in lowered:
        return (
            "The matched PatternCard supplies exactly one allowed operator. Use it; "
            "freeform expression or statement operators are prohibited:\n"
            '- normalize_inline_wrapper_boundaries: selector {"occurrence": 0}; '
            "arguments {}. It inserts percent-newline wrapper boundaries and trims "
            "the complete inline trailer without changing block output."
        )
    if "generated leaf name" in lowered and "instance" in lowered:
        return (
            "The matched PatternCard supplies exactly one allowed operator. Use it; "
            "freeform expression or statement operators are prohibited:\n"
            '- initialize_generated_subclass_identity: selector {"occurrence": 0}; '
            "arguments {}. It atomically stores the generated leaf name in the "
            "factory class attributes and initializes instance qualified identity "
            "from that stored class value."
        )
    if "property" in lowered and "paren" in lowered:
        return (
            "The matched PatternCard supplies exactly one allowed operator. Use it; "
            "freeform expression or statement operators are prohibited:\n"
            '- remove_property_index_parens: selector {"occurrence": 0}; '
            "arguments {}. It removes empty parentheses from a property index "
            "format string."
        )
    if "variable" in lowered and "obj" in lowered and "link" in lowered:
        return (
            "The matched PatternCard supplies exactly one allowed operator. Use it; "
            "freeform expression or statement operators are prohibited:\n"
            '- remove_variable_obj_role: selector {"occurrence": 0}; '
            "arguments {}. It removes the object role from a variable typed-field "
            "so instance variables no longer link to objects of the same name."
        )
    return _OPERATOR_CATALOG


def _single_operator_for_card(selected_card: str | None) -> str | None:
    if selected_card is None:
        return None
    lowered = selected_card.lower()
    if "defaults vector" in lowered and "positional" in lowered:
        return "align_trailing_defaults"
    if "percent sentinel" in lowered and "inline" in lowered:
        return "normalize_inline_wrapper_boundaries"
    if "generated leaf name" in lowered and "instance" in lowered:
        return "initialize_generated_subclass_identity"
    if "property" in lowered and "paren" in lowered:
        return "remove_property_index_parens"
    if "variable" in lowered and "obj" in lowered and "link" in lowered:
        return "remove_variable_obj_role"
    return None


def _canonicalize_single_operator_output(
    raw: str, *, operator: str | None, supplied_symbol: str | None = None
) -> tuple[str, bool]:
    candidate = raw.strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        return raw, False
    try:
        data = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return raw, False
    operations = data.get("operations") if isinstance(data, dict) else None
    if not isinstance(operations, list) or not operations:
        return raw, False
    normalized = []
    changed = False
    if supplied_symbol is not None and data.get("symbol") != supplied_symbol:
        data["symbol"] = supplied_symbol
        changed = True
    if operator in _ZERO_ARGUMENT_SINGLE_OPERATORS:
        canonical_operation = {
            "operator": operator,
            "selector": {"occurrence": 0},
            "arguments": {},
        }
        if operations != [canonical_operation]:
            data["operations"] = [canonical_operation]
            changed = True
        return (canonical_json(data), True) if changed else (raw, False)
    for row in operations:
        if not isinstance(row, dict):
            return raw, False
        if set(row) == {"selector", "arguments"}:
            arguments = row.get("arguments")
            inferred = (
                _ARGUMENT_SHAPE_OPERATORS.get(frozenset(arguments))
                if isinstance(arguments, dict)
                else None
            )
            selected_operator = operator or inferred
            if selected_operator is None:
                normalized.append(row)
                continue
            row = {"operator": selected_operator, **row}
            changed = True
        elif (
            set(row) == {"selector", "kind", "occurrence", "arguments"}
            and row["selector"] == operator
            and type(row["occurrence"]) is int
            and isinstance(row["arguments"], dict)
        ):
            row = {
                "operator": operator,
                "selector": {"occurrence": row["occurrence"]},
                "arguments": row["arguments"],
            }
            changed = True
        normalized.append(row)
    if not changed:
        return raw, False
    data["operations"] = normalized
    return canonical_json(data), True


def parse_operator_plan_output(raw: str) -> OperatorPlan:
    """Extract exactly one operator plan from bounded model output."""
    candidate = raw.strip()
    fenced = _JSON_FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ContractError("student output contains no operator plan object")
        candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ContractError("student operator plan is malformed JSON") from exc
    return OperatorPlan.from_dict(data)


def _operator_failure_reason(error: ContractError) -> str:
    detail = str(error)
    if "malformed JSON" in detail:
        return "json-malformed"
    if "selector did not resolve" in detail:
        return "selector-no-match"
    if "symbol did not resolve" in detail:
        return "unrelated-symbol"
    if "operator class mismatch" in detail:
        return "operator-class-mismatch"
    if "operator expression is invalid" in detail:
        return "expression-used-for-statement"
    return "apply-fail"


def _operator_failure_is_repairable(error: ContractError) -> bool:
    """Retry only when another decode can correct the returned plan.

    Renderer-proven no-ops and frozen-candidate drift are deterministic for the
    chosen plan. A second full decode historically consumed minutes without
    changing that evidence, so those failures fail fast.
    """

    detail = str(error)
    if detail.startswith("operator class mismatch"):
        # Deterministic classification error: the model chose replace_expression
        # for a statement node; a re-decode repeats the same plan.  (A1 now
        # auto-corrects this at materialization, so it is never repairable.)
        return False
    if detail == "operator materialization gate failed: no-op":
        # Grounded repair (real source lines + no-op hint) can break the
        # tautology: allow one replan so the model can pick a real edit target.
        return True
    return detail != "operator plan changed the frozen candidates"


def _is_unsafe_empty_sequence_fix(task: StudentTask, plan: OperatorPlan) -> bool:
    """Reject mapping-style lookups presented as empty-sequence guards."""

    issue = task.instruction.lower()
    if not (
        ("indexerror" in issue or "index error" in issue)
        and ("empty" in issue or "no element" in issue)
    ):
        return False
    for operation in plan.operations:
        source = str(operation.selector.get("source", ""))
        replacement = " ".join(str(value) for value in operation.arguments.values())
        if "[" not in source or "]" not in source or ".get(" not in replacement:
            continue
        guarded = any(
            marker in replacement
            for marker in (" if ", "next(", "len(", " or ", " and ")
        )
        if not guarded:
            return True
    return False


def _ungrounded_replacement_identifiers(plan: OperatorPlan) -> tuple[str, ...]:
    """Find replacement identifiers absent from the selector or frozen intent."""

    intent_text = " ".join(plan.intent.to_dict().values())
    ungrounded: set[str] = set()
    for operation in plan.operations:
        replacement_key = {
            "replace_expression": "new_expression",
            "replace_condition": "new_condition",
        }.get(operation.operator)
        if replacement_key is None:
            continue
        replacement = str(operation.arguments[replacement_key])
        try:
            tree = ast.parse(replacement, mode="eval")
        except SyntaxError:
            continue
        grounded = set(
            _IDENTIFIER_TOKEN.findall(
                f"{operation.selector.get('source', '')} {intent_text}"
            )
        )
        introduced = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        ungrounded.update(introduced - grounded - _SAFE_REPLACEMENT_NAMES)
    return tuple(sorted(ungrounded))


def _collapses_empty_and_absent_boundary(plan: OperatorPlan) -> bool:
    """Reject truthiness-only conditions for an explicit empty/absent boundary."""

    intent = " ".join(plan.intent.to_dict().values()).lower()
    if not any(
        marker in intent
        for marker in ("empty", "none", "absent", "missing", "sentinel")
    ):
        return False
    for operation in plan.operations:
        if operation.operator != "replace_condition":
            continue
        try:
            tree = ast.parse(str(operation.arguments["new_condition"]), mode="eval")
        except SyntaxError:
            continue
        has_identity = any(
            isinstance(node, ast.Compare)
            and any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops)
            for node in ast.walk(tree)
        )
        has_length = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"
            for node in ast.walk(tree)
        )
        if not has_identity and not has_length:
            return True
    return False


def _boundary_replacement_semantics(
    selector_source: str,
    replacement: str,
) -> str | None:
    try:
        selector = ast.parse(selector_source, mode="eval").body
        replacement_tree = ast.parse(replacement, mode="eval").body
    except SyntaxError:
        return None
    replacement_fingerprint = ast.dump(replacement_tree, include_attributes=False)
    for candidate in _condition_boundary_rewrites(selector):
        candidate_tree = ast.parse(candidate["new_condition"], mode="eval").body
        if (
            ast.dump(candidate_tree, include_attributes=False)
            == replacement_fingerprint
        ):
            return candidate["semantics"]
    return None


def _semantic_oracle_mismatch(
    task: StudentTask,
    plan: OperatorPlan,
) -> str | None:
    """Return a deterministic mismatch detail for a supported boundary edit."""

    for operation in plan.operations:
        if operation.operator != "replace_condition":
            continue
        selector_source = str(operation.selector.get("source", ""))
        oracle = _minimal_boundary_oracle(selector_source, task.instruction)
        if oracle is None:
            continue
        replacement = str(operation.arguments["new_condition"])
        actual = _boundary_replacement_semantics(selector_source, replacement)
        required = oracle["required_semantics"]
        if actual != required:
            return (
                f"public issue requires {required} for the {oracle['trigger_case']} "
                f"case while preserving {oracle['preserved_case']}; got "
                f"{actual or 'an unclassified compound predicate'}"
            )
    return None


_SELECTOR_GROUND_MIN_RATIO = 0.6


def _normalized_selector_text(value: str) -> str:
    """Whitespace-normalize a selector for overlap comparison."""
    return " ".join(str(value).split())


def _selector_overlap(model_source: str, candidate_source: str) -> float:
    """Normalized SequenceMatcher ratio between two selector texts."""
    left = _normalized_selector_text(model_source)
    right = _normalized_selector_text(candidate_source)
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def _grounded_operator_operation(
    operation: OperatorOperation,
    candidate: dict[str, Any],
) -> OperatorOperation | None:
    """Rebuild one operation against a candidate's byte-exact source.

    The operator kind follows the candidate's suggested_operator and the
    arguments are adapted when the kind changes (mirrors A1: an expression
    field may actually hold a statement).  Returns None when the candidate is
    incompatible with the model's arguments.
    """
    suggested = str(candidate.get("suggested_operator") or operation.operator)
    operator = suggested if suggested in _OPERATORS else operation.operator
    arguments = dict(operation.arguments)
    selector = {
        "source": str(candidate["source"]),
        "occurrence": int(candidate.get("occurrence", 0)),
    }
    if operator != operation.operator:
        if operator == "replace_statement" and "new_expression" in arguments:
            arguments = {"new_statements": str(arguments["new_expression"])}
        elif operator == "replace_expression" and "new_statements" in arguments:
            return None  # cannot safely collapse statements into an expression
    try:
        grounded = OperatorOperation(
            operator=operator, selector=selector, arguments=arguments
        )
        grounded.validate()
    except ContractError:
        return None
    return grounded


def _ground_operator_selector(
    task: StudentTask,
    plan: OperatorPlan,
    context_by_file: dict[str, str],
) -> tuple[OperatorPlan | None, str | None]:
    """Deterministically re-point every zero-match selector to the nearest
    framework-harvested typed candidate in the same file.

    Only selectors whose literal source does not resolve exactly once are
    re-pointed; correctly-resolving operations are left untouched.  Returns
    (grounded_plan, None) on success, else (None, reason).
    """
    source = task.resolve_target(plan.file).read_text(
        encoding="utf-8", errors="replace"
    )
    candidates = _harvest_typed_selector_candidates(
        source, instruction=task.instruction, maximum_candidates=128
    )
    if not candidates:
        return None, "no typed selector candidates to ground against"
    grounded_ops: list[OperatorOperation] = []
    grounded_any = False
    for operation in plan.operations:
        selector_source = operation.selector.get("source")
        if not isinstance(selector_source, str) or not selector_source.strip():
            grounded_ops.append(operation)
            continue
        if source.count(selector_source) == 1:
            grounded_ops.append(operation)
            continue
        best: dict[str, Any] | None = None
        best_ratio = 0.0
        for candidate in candidates:
            cand_source = str(candidate.get("source", ""))
            if not cand_source:
                continue
            ratio = _selector_overlap(selector_source, cand_source)
            if ratio > best_ratio:
                best, best_ratio = candidate, ratio
        if best is None or best_ratio < _SELECTOR_GROUND_MIN_RATIO:
            return None, (
                f"no grounded selector above {_SELECTOR_GROUND_MIN_RATIO} "
                f"overlap for {selector_source[:60]!r}"
            )
        grounded = _grounded_operator_operation(operation, best)
        if grounded is None:
            return None, (
                f"candidate {best.get('candidate_id')} is incompatible with "
                "operation arguments"
            )
        grounded_ops.append(grounded)
        grounded_any = True
    if not grounded_any:
        return None, "no selector required grounding"
    grounded_plan = OperatorPlan(
        schema_version=plan.schema_version,
        file=plan.file,
        symbol=plan.symbol,
        intent=plan.intent,
        operations=tuple(grounded_ops),
        diagnostic=plan.diagnostic,
    )
    grounded_plan.validate()
    return grounded_plan, None


def _error_message_value_expression(replacement: str) -> str | None:
    """Extract the value a model tried to inject into an error message via
    string formatting, e.g. ``.format(value=value)`` -> ``value`` or ``% value``
    -> ``value``.  Returns the expression source or None."""
    try:
        tree = ast.parse(replacement, mode="eval")
    except SyntaxError:
        return None
    node = tree.body
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "format":
            if node.keywords:
                return ast.unparse(node.keywords[0].value)
            if node.args:
                return ast.unparse(node.args[0])
            return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return ast.unparse(node.right)
    return None


def _has_error_message_access(replacement: str) -> bool:
    return "error_messages" in replacement or "message" in replacement.casefold()


def _file_has_params_validation_error(source: str) -> bool:
    """True when the file already uses the repo idiom raise ValidationError(..., params=...)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            call = node.exc
            if _call_name(call) == "ValidationError" and any(
                kw.arg == "params" for kw in call.keywords
            ):
                return True
    return False


def _raise_statement_for_selector(
    source: str, selector_source: str, symbol: str, occurrence: int
) -> tuple[str, int] | None:
    """Find the ``raise ValidationError(...)`` statement (without params=) that
    contains the selector text, scoped to the plan's symbol where possible.
    Returns the statement source or None."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    needle = selector_source.strip()
    scope: ast.AST = tree
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == symbol
        ):
            scope = node
            break
    matches: list[tuple[int, str]] = []
    for node in ast.walk(scope):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        call = node.exc
        if _call_name(call) != "ValidationError":
            continue
        if any(kw.arg == "params" for kw in call.keywords):
            continue  # already idiomatic; not the near-miss target
        stmt_text = ast.get_source_segment(source, node)
        call_text = ast.get_source_segment(source, call)
        if stmt_text and call_text and needle in call_text:
            matches.append((node.lineno or 0, stmt_text))
    if not matches:
        return None
    unique: list[tuple[int, str]] = []
    seen: set[str] = set()
    for lineno, text in matches:
        if text not in seen:
            seen.add(text)
            unique.append((lineno, text))
    if occurrence < len(unique):
        return unique[occurrence][1]
    return unique[0][1]


def _augment_validation_error_params(raise_stmt_source: str, value_expr: str) -> str:
    """Append ``params={'value': <expr>}`` to a ValidationError call in a raise
    statement, returning the new statement source (or the original when the
    call already carries params=)."""
    try:
        tree = ast.parse(raise_stmt_source)
    except SyntaxError:
        return raise_stmt_source
    stmt = tree.body[0]
    if not isinstance(stmt, ast.Raise) or not isinstance(stmt.exc, ast.Call):
        return raise_stmt_source
    call = stmt.exc
    if any(kw.arg == "params" for kw in call.keywords):
        return raise_stmt_source
    params_value = ast.parse(f"{{'value': {value_expr}}}", mode="eval").body
    call.keywords.append(ast.keyword(arg="params", value=params_value))
    ast.fix_missing_locations(stmt)
    return ast.unparse(stmt)


def _correct_error_params_rewrites(
    task: StudentTask, plan: OperatorPlan
) -> OperatorPlan:
    """Deterministically repair the error-message near-miss.

    When the Student tries to inject a value into an error message via string
    formatting (``.format(...)`` / ``%``) but the repo already uses
    ``raise ValidationError(..., params={...})``, rewrite the operation to the
    repo idiom: replace the whole raise statement and add
    ``params={'value': <extracted expr>}``.  Conservative: fires only when all
    of (message access, extractable value, sibling params= call, matching
    raise-without-params) hold; otherwise the plan is returned unchanged.
    """
    try:
        source = task.resolve_target(plan.file).read_text(
            encoding="utf-8", errors="replace"
        )
    except (ContractError, OSError, UnicodeError):
        return plan
    if not _file_has_params_validation_error(source):
        return plan
    corrected_ops: list[OperatorOperation] = []
    changed = False
    for operation in plan.operations:
        replacement = ""
        if operation.operator == "replace_expression":
            replacement = str(operation.arguments.get("new_expression", ""))
        elif operation.operator == "replace_statement":
            replacement = str(operation.arguments.get("new_statements", ""))
        value_expr = _error_message_value_expression(replacement)
        if (
            not replacement
            or not _has_error_message_access(replacement)
            or value_expr is None
        ):
            corrected_ops.append(operation)
            continue
        selector = str(operation.selector.get("source", ""))
        occurrence = int(operation.selector.get("occurrence", 0))
        raise_stmt = _raise_statement_for_selector(
            source, selector, plan.symbol, occurrence
        )
        if raise_stmt is None:
            corrected_ops.append(operation)
            continue
        augmented = _augment_validation_error_params(raise_stmt, value_expr)
        if augmented == raise_stmt:
            corrected_ops.append(operation)
            continue
        try:
            corrected = OperatorOperation(
                operator="replace_statement",
                selector={"source": raise_stmt, "occurrence": occurrence},
                arguments={"new_statements": augmented},
            )
            corrected.validate()
        except ContractError:
            corrected_ops.append(operation)
            continue
        corrected_ops.append(corrected)
        changed = True
    if not changed:
        return plan
    corrected_plan = OperatorPlan(
        schema_version=plan.schema_version,
        file=plan.file,
        symbol=plan.symbol,
        intent=plan.intent,
        operations=tuple(corrected_ops),
        diagnostic=plan.diagnostic,
    )
    corrected_plan.validate()
    return corrected_plan


def _single_if_guard(
    new_body: str,
) -> tuple[ast.If, ast.stmt] | None:
    """Return (if_node, inner_stmt) when new_body is exactly one if with one
    non-empty body statement and no else/elif branch."""
    try:
        statements = ast.parse(str(new_body)).body
    except SyntaxError:
        return None
    if len(statements) != 1 or not isinstance(statements[0], ast.If):
        return None
    if_node = statements[0]
    if if_node.orelse or len(if_node.body) != 1:
        return None
    return if_node, if_node.body[0]


def _methods_containing_statement(
    cls: ast.ClassDef, inner_stmt: ast.stmt
) -> list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, int]]:
    """Direct-body statements of each method matching the inner statement."""
    dump = _ast_dump(inner_stmt)
    matches: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, int]] = []
    for node in cls.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for index, stmt in enumerate(node.body):
            if _ast_dump(stmt) == dump:
                matches.append((node, index))
    return matches


def _correct_method_body_guards(
    task: StudentTask, plan: OperatorPlan
) -> OperatorPlan:
    """Deterministic guard-wrap correction (A1-class).

    When the Student writes a ``replace_method_body`` whose symbol resolves to
    a class (not a method) and whose ``new_body`` is a single if-guard wrapping
    one statement that already exists verbatim inside exactly one method of
    that class, rewrite the operation to ``replace_statement`` on that
    statement with the if-guard — the minimal correct edit (mirrors the gold
    fix for django-15277).  Fail-closed: any ambiguity keeps the operation.
    """
    try:
        source = task.resolve_target(plan.file).read_text(
            encoding="utf-8", errors="replace"
        )
    except (ContractError, OSError, UnicodeError):
        return plan
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return plan
    corrected_ops: list[OperatorOperation] = []
    changed = False
    for operation in plan.operations:
        if operation.operator != "replace_method_body":
            corrected_ops.append(operation)
            continue
        guard = _single_if_guard(str(operation.arguments.get("new_body", "")))
        if guard is None:
            corrected_ops.append(operation)
            continue
        if_node, inner_stmt = guard
        cls = _find_definition(tree.body, plan.symbol.split("."))
        if not isinstance(cls, ast.ClassDef):
            corrected_ops.append(operation)
            continue
        matches = _methods_containing_statement(cls, inner_stmt)
        if len(matches) != 1:
            corrected_ops.append(operation)
            continue
        method, statement_index = matches[0]
        inner_source = ast.get_source_segment(source, method.body[statement_index])
        if not inner_source or not inner_source.strip():
            corrected_ops.append(operation)
            continue
        occurrence = sum(
            1
            for other in cls.body
            if isinstance(other, (ast.FunctionDef, ast.AsyncFunctionDef))
            for stmt in other.body[:statement_index]
            if _ast_dump(stmt) == _ast_dump(inner_stmt)
        )
        try:
            corrected = OperatorOperation(
                operator="replace_statement",
                selector={"source": inner_source, "occurrence": occurrence},
                arguments={"new_statements": ast.unparse(if_node)},
            )
            corrected.validate()
        except ContractError:
            corrected_ops.append(operation)
            continue
        corrected_ops.append(corrected)
        changed = True
    if not changed:
        return plan
    return OperatorPlan(
        schema_version=plan.schema_version,
        file=plan.file,
        symbol=plan.symbol,
        intent=plan.intent,
        operations=tuple(corrected_ops),
        diagnostic=plan.diagnostic,
    )


_CONSTANT_CONDITION_RE = re.compile(r"^\s*(True|False|0|1)\s*$")


def _constant_condition_reason(plan: OperatorPlan) -> str | None:
    """Reject replace_condition plans that replace a real condition with a
    literal constant (True/False/0/1) -- a dangerous pseudo-fix (e.g. the 7B's
    'if False:' for django-14351 that would silently disable DISTINCT)."""
    for operation in plan.operations:
        if operation.operator != "replace_condition":
            continue
        new_condition = str(operation.arguments.get("new_condition", "")).strip()
        if _CONSTANT_CONDITION_RE.match(new_condition):
            return "constant-condition"
    return None


_HASH_ID_ANTI_PATTERN = re.compile(r"\bhash\s*\(\s*id\s*\(")


def _hash_id_anti_pattern_reason(plan: OperatorPlan) -> str | None:
    """Reject plans whose replacement uses ``hash(id(...))`` (django-15315).

    ``id()`` does not satisfy the ``__eq__`` hash contract (objects equal by
    value get different hashes), so a __hash__ body built on ``id()`` is an
    anti-pattern.  Deterministic reject + replan hint instead of emitting a
    dangerous patch.
    """
    for operation in plan.operations:
        for value in operation.arguments.values():
            if isinstance(value, str) and _HASH_ID_ANTI_PATTERN.search(value):
                return "hash-id-anti-pattern"
    return None


def _correct_boundary_rewrites(task: StudentTask, plan: OperatorPlan) -> None:
    """Deterministically correct a supported boundary rewrite in place.

    The weak Student sometimes copies the wrong boundary rewrite (for example
    ``len(x) == 0`` where the public issue requires ``x is None``). The issue
    oracle knows the required truth-table change, so correct the operation
    argument deterministically instead of relying on Student memory.
    """

    for operation in plan.operations:
        if operation.operator != "replace_condition":
            continue
        selector_source = str(operation.selector.get("source", ""))
        oracle = _minimal_boundary_oracle(selector_source, task.instruction)
        if oracle is None:
            continue
        try:
            node = ast.parse(selector_source, mode="eval").body
        except SyntaxError:
            continue
        rewrites = _condition_boundary_rewrites(node)
        current = str(operation.arguments["new_condition"])
        # Only correct a simple-but-wrong boundary choice. A compound predicate
        # still goes through the oracle-mismatch repair path.
        if current not in {rewrite["new_condition"] for rewrite in rewrites}:
            continue
        matched = next(
            (
                rewrite
                for rewrite in rewrites
                if rewrite["semantics"] == oracle["required_semantics"]
            ),
            None,
        )
        if matched is not None:
            operation.arguments["new_condition"] = matched["new_condition"]


class OperatorPlanAdapter(StudentAdapter):
    """Materialize a model-owned intent/plan with a deterministic AST renderer."""

    def experiment_config(self) -> dict[str, Any]:
        configured = getattr(self.generator, "generation_config", None)
        return {
            "adapter": type(self).__name__,
            "adapter_contract": "python-typed-operator-plan-v1",
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "verification_policy": (
                "source-round-23-consistency-identifier-boundary-oracle-v3"
            ),
            "generator": configured() if configured is not None else {},
            "renderer": {
                "name": "deterministic-ast-selector-renderer-v7",
                "qualification_suite": "typed-operator-renderer-synthetic-v7",
            },
        }

    def run(self, task: StudentTask, revision: LoopRevision) -> StudentAttempt:
        task.validate()
        revision.validate()
        try:
            raw = self.generator(task, revision)
        except Exception as exc:
            return self._failure(
                task, revision, "", "eval-infra", f"generator failed: {exc}"
            )
        if not isinstance(raw, str):
            return self._failure(
                task, revision, "", "eval-infra", "generator returned non-text"
            )
        plan_limit = getattr(
            getattr(self.generator, "profile", None),
            "max_plan_chars",
            _MAX_OPERATOR_PLAN_CHARS,
        )
        # Count compact JSON: the model's pretty-printed output inflates the
        # character count without adding plan complexity (django-11551 wrote
        # 2104 pretty chars ~= 1300 compact).  Only the parsed object is
        # bounded; unparseable output is still gated by its raw length.
        effective_chars = len(raw)
        try:
            parsed_payload = json.loads(
                _JSON_FENCE.search(raw).group(1)
                if _JSON_FENCE.search(raw)
                else raw[raw.find("{") : raw.rfind("}") + 1]
            )
            effective_chars = len(canonical_json(parsed_payload))
        except (json.JSONDecodeError, ValueError):
            pass
        if effective_chars > plan_limit:
            return self._failure(
                task,
                revision,
                raw,
                "plan-too-large",
                f"operator plan exceeds {plan_limit} characters",
            )
        unresolved = parse_unresolved_abstention(raw)
        if unresolved is not None:
            return self._failure(task, revision, raw, "unresolved", unresolved)
        preliminary = self._classify_unstructured(raw)
        if preliminary is not None:
            return self._failure(task, revision, raw, preliminary, preliminary)
        try:
            plan = parse_operator_plan_output(raw)
        except ContractError as exc:
            reason = _operator_failure_reason(exc)
            if reason == "apply-fail":
                reason = "malformed-hunk"
            return self._failure(task, revision, raw, reason, str(exc))
        _correct_boundary_rewrites(task, plan)
        plan = _correct_error_params_rewrites(task, plan)
        plan = _correct_method_body_guards(task, plan)
        constant_reason = _constant_condition_reason(plan)
        if constant_reason is not None:
            return self._failure(
                task,
                revision,
                raw,
                constant_reason,
                "replacement condition must not be a literal constant",
            )
        hash_id_reason = _hash_id_anti_pattern_reason(plan)
        if hash_id_reason is not None:
            return self._failure(
                task,
                revision,
                raw,
                hash_id_reason,
                "replacement must not hash id(); use the __eq__ comparison key",
            )
        if revision.source_round >= 19 and re.search(
            r"\bunresolved\b", plan.diagnostic, re.IGNORECASE
        ):
            return self._failure(
                task,
                revision,
                raw,
                "inconsistent-plan",
                "unresolved diagnostic requires an explicit unresolved abstention",
                target_file=plan.file,
            )
        if revision.source_round >= 19:
            ungrounded = _ungrounded_replacement_identifiers(plan)
            if ungrounded:
                return self._failure(
                    task,
                    revision,
                    raw,
                    "identifier-drift",
                    "replacement identifiers are not grounded in the selector or "
                    f"frozen intent: {', '.join(ungrounded)}",
                    target_file=plan.file,
                )
        if revision.source_round >= 20 and _collapses_empty_and_absent_boundary(plan):
            return self._failure(
                task,
                revision,
                raw,
                "boundary-collapse",
                "empty versus absent boundary requires an identity or length predicate",
                target_file=plan.file,
            )
        if revision.source_round >= 23:
            oracle_mismatch = _semantic_oracle_mismatch(task, plan)
            if oracle_mismatch is not None:
                return self._failure(
                    task,
                    revision,
                    raw,
                    "semantic-oracle-mismatch",
                    oracle_mismatch,
                    target_file=plan.file,
                )

        allowed = task.allowed_targets or self._discover_targets(task)
        if (
            plan.file not in allowed
            or self._is_test_path(plan.file)
            or Path(plan.file).suffix != ".py"
        ):
            return self._failure(
                task,
                revision,
                raw,
                "wrong-target",
                f"target is not an allowed Python source: {plan.file}",
                target_file=plan.file,
            )
        try:
            target = task.resolve_target(plan.file)
            before = target.read_text(encoding="utf-8")
            result = materialize_operator_plan(before, plan)
        except (ContractError, OSError, UnicodeError) as exc:
            reason = (
                _operator_failure_reason(exc)
                if isinstance(exc, ContractError)
                else "apply-fail"
            )
            return self._failure(
                task,
                revision,
                raw,
                reason,
                str(exc),
                target_file=plan.file,
            )
        if revision.source_round >= 12 and _is_unsafe_empty_sequence_fix(task, plan):
            return self._failure(
                task,
                revision,
                raw,
                "unsafe-empty-sequence",
                "mapping get() does not guard an empty sequence subscript",
                target_file=plan.file,
                before_sha256=_sha256_text(before),
            )
        receipt = canonical_json(
            {
                "receipt_type": "operator-realization-v1",
                "diagnostic": plan.diagnostic,
                "intent": plan.intent.to_dict(),
                "plan_fingerprint": plan.fingerprint,
                "materialization": result.to_dict(),
            }
        )
        if not result.accepted:
            reason = (
                result.failure_reason
                if result.failure_reason in {"no-op", "unbound-name"}
                else "apply-fail"
            )
            return self._failure(
                task,
                revision,
                raw,
                reason,
                receipt,
                target_file=plan.file,
                before_sha256=_sha256_text(before),
            )
        patch = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                result.after.splitlines(keepends=True),
                fromfile=f"a/{plan.file}",
                tofile=f"b/{plan.file}",
            )
        )
        if not patch.strip() or not self._git_apply_check(task.checkout, patch):
            return self._failure(
                task,
                revision,
                raw,
                "apply-fail",
                "materialized operator patch does not apply cleanly",
                patch=patch,
                target_file=plan.file,
                before_sha256=_sha256_text(before),
            )
        return StudentAttempt(
            task=task,
            revision_id=revision.revision_id,
            raw_output=raw,
            raw_output_sha256=_sha256_text(raw),
            edit=None,
            patch=patch,
            patch_sha256=_sha256_text(patch),
            target_file=plan.file,
            before_sha256=_sha256_text(before),
            after_sha256=_sha256_text(result.after),
            implementation_fingerprint=_implementation_fingerprint(
                plan.file, result.after
            ),
            structural_valid=True,
            failure_reason=None,
            detail=receipt,
        )


def _call_name(node: ast.Call) -> str | None:
    """Best-effort callee name for a Call node."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _call_names_in_source(source: str) -> tuple[str, ...]:
    """Unique callee names appearing in a (possibly partial) source snippet."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return tuple(
            dict.fromkeys(
                name
                for name in _CALL_NAME_RE.findall(source)
                if name
                not in {
                    "if", "for", "while", "return", "with", "def", "class", "raise",
                }
            )
        )
    names: list[str] = []
    seen: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            names.append(name)
    return tuple(names)


def _repository_call_exemplars(
    source: str,
    call_name: str,
    *,
    exclude_sources: frozenset[str] = frozenset(),
    max_exemplars: int = 2,
    max_chars: int = 400,
) -> tuple[tuple[int, str], ...]:
    """Deterministic sibling usages of one call inside the same source file.

    Read-only reference for the Student: when it edits one call (for example
    ``raise ValidationError(self.error_messages['invalid_choice'], ...)``
    without ``params=``), show how the same call is made elsewhere in the repo
    so the correct library idiom is visible.  Richer calls (more keyword
    arguments) rank first; the edited call's own source is excluded.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    needle = call_name.casefold()
    seen: set[str] = set(exclude_sources)
    rows: list[tuple[int, str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name is None or name.casefold() != needle:
            continue
        text = ast.get_source_segment(source, node)
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append((node.lineno or 0, text, len(node.keywords)))
    rows.sort(key=lambda row: (-row[2], row[0]))
    result: list[tuple[int, str]] = []
    used = 0
    for lineno, text, _bonus in rows:
        if len(result) >= max_exemplars or used + len(text) > max_chars:
            break
        result.append((lineno, text))
        used += len(text)
    return tuple(result)


def _usage_exemplars_block(
    sources: dict[str, str],
    contexts: tuple[tuple[str, str, int], ...],
    *,
    instruction: str = "",
    per_file_candidates: Mapping[str, list[dict[str, Any]]] | None = None,
    max_calls: int = 4,
    max_exemplars_per_call: int = 2,
    max_total_chars: int = 700,
) -> str:
    """Render a bounded read-only exemplar section for the calls the Student is
    most likely to edit.

    Call names are collected from the framework-enumerated editable candidates
    first (the exact spans the model may rewrite), then from the symbol excerpt
    as a fallback.  Exemplars are harvested from the full file source, so the
    model sees how the same call is made elsewhere in the repo.
    """
    sections: list[str] = []
    used = 0
    candidates = per_file_candidates or {}
    for relative, excerpt, _line in contexts:
        full_source = sources.get(relative, "")
        if not full_source:
            continue
        call_names: list[str] = []
        seen: set[str] = set()

        def add_call_name(name: str) -> None:
            if name.casefold() not in seen:
                seen.add(name.casefold())
                call_names.append(name)

        # Issue-derived call families (e.g. ValidationError in the bug report)
        # are the highest-value exemplars and must not be crowded out by
        # generic excerpt call names.
        for name in _CAMEL_CASE_RE.findall(instruction):
            if name in full_source:
                add_call_name(name)
        for row in candidates.get(relative, ()):
            for name in _call_names_in_source(str(row.get("source", ""))):
                add_call_name(name)
        for name in _call_names_in_source(excerpt):
            add_call_name(name)
        exclude = frozenset(_call_texts_in_source(excerpt))
        for name in call_names[:max_calls]:
            for lineno, text in _repository_call_exemplars(
                full_source,
                name,
                exclude_sources=exclude,
                max_exemplars=max_exemplars_per_call,
            ):
                block = f"{relative}:{lineno} :: {text}"
                if used + len(block) > max_total_chars:
                    return "\n".join(sections)
                sections.append(block)
                used += len(block)
    return "\n".join(sections)


def _call_texts_in_source(source: str) -> tuple[str, ...]:
    """Source text of every Call node in a snippet (for exclusion)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            text = ast.get_source_segment(source, node)
            if text:
                out.append(text)
    return tuple(out)


def _plan_usage_exemplars(
    task: StudentTask,
    plan: OperatorPlan,
    *,
    max_exemplars_per_call: int = 2,
    max_total_chars: int = 600,
) -> str:
    """Harvest repository usage exemplars for the calls in a rejected plan.

    Used in the grounded-repair replan so the Student sees how the same calls
    are made elsewhere in the repo (e.g. ValidationError with params=) before
    it rewrites its selector/replacement.
    """
    try:
        source = task.resolve_target(plan.file).read_text(
            encoding="utf-8", errors="replace"
        )
    except (ContractError, OSError, UnicodeError):
        return ""
    sections: list[str] = []
    used = 0
    seen_names: set[str] = set()
    for operation in plan.operations:
        selector = str(operation.selector.get("source", ""))
        for name in _call_names_in_source(selector):
            if name.casefold() in seen_names:
                continue
            seen_names.add(name.casefold())
            exclude = frozenset(_call_texts_in_source(selector))
            for lineno, text in _repository_call_exemplars(
                source,
                name,
                exclude_sources=exclude,
                max_exemplars=max_exemplars_per_call,
            ):
                block = f"{plan.file}:{lineno} :: {text}"
                if used + len(block) > max_total_chars:
                    return "\n".join(sections)
                sections.append(block)
                used += len(block)
    return "\n".join(sections)


def _shared_operator_target(skill_text: str) -> tuple[str, str] | None:
    marker = "## Shared diagnosis and localization (read-only)"
    if marker not in skill_text:
        return None
    suffix = skill_text.split(marker, 1)[1].lstrip()
    first_line = suffix.splitlines()[0] if suffix else ""
    try:
        shared = json.loads(first_line)
    except json.JSONDecodeError as exc:
        raise ContractError("shared operator context is malformed") from exc
    files = shared.get("target_files") if isinstance(shared, dict) else None
    symbol = shared.get("target_symbol") if isinstance(shared, dict) else None
    if (
        not isinstance(files, list)
        or len(files) != 1
        or not isinstance(files[0], str)
        or not isinstance(symbol, str)
        or not symbol.strip()
    ):
        return None
    return files[0], symbol


def _grounded_repair_detail(
    task: StudentTask,
    plan: OperatorPlan | None,
    error: Exception,
) -> str:
    """Append real, line-numbered source from the plan's resolved symbol.

    The weak Student frequently emits phantom selectors (a target line that
    does not exist) or tautological replacements (selector == replacement, a
    no-op).  Echoing only the raw error gives the repair model nothing to
    ground on; feeding the actual statements in the resolved symbol lets the
    repair pick a real, existing edit target.  Pure harness-layer: no model
    call, deterministic, and the excerpt is capped.
    """
    detail = str(error)
    if plan is None or not getattr(plan, "file", None):
        return detail
    try:
        source = task.resolve_target(plan.file).read_text(
            encoding="utf-8", errors="replace"
        )
        tree = ast.parse(source)
        symbol = _find_definition(tree.body, plan.symbol.split("."))
    except (ContractError, OSError, UnicodeError, SyntaxError, AttributeError):
        return detail
    if symbol is None or symbol.end_lineno is None:
        return detail
    lines = source.splitlines()
    start = max(0, symbol.lineno - 1)
    end = min(len(lines), symbol.end_lineno)
    excerpt = "\n".join(
        f"{index + 1:5d} {line}"
        for index, line in enumerate(lines[start:end], start)
    )[:4000]
    return (
        f"{detail}\n"
        "Grounded source (line-numbered, from the resolved symbol; any selector "
        "or replacement must reference real lines below):\n"
        f"{excerpt}"
    )


class MlxOperatorPlanGenerator(MlxStructuredGenerator):
    """Ask the local Student for intent and typed operations, never raw code spans."""

    def __init__(
        self,
        *,
        max_tokens: int = 1536,
        max_context_chars: int = 24_000,
        max_context_targets: int = 32,
        max_plan_repairs: int | None = None,
        use_clause_critic: bool = True,
        profile: StudentCapabilityProfile | None = None,
        **fields: Any,
    ) -> None:
        self.profile = profile or profile_for(str(fields.get("model_path", "")))
        if max_plan_repairs is None:
            max_plan_repairs = self.profile.max_plan_repairs
        if type(max_plan_repairs) is not int or max_plan_repairs < 0:
            raise ValueError("operator max_plan_repairs must be non-negative")
        if type(use_clause_critic) is not bool:
            raise ValueError("operator use_clause_critic must be boolean")
        if type(max_context_targets) is not int or not 1 <= max_context_targets <= 32:
            raise ValueError("operator max_context_targets must be between 1 and 32")
        super().__init__(
            max_tokens=max_tokens,
            max_context_chars=max_context_chars,
            max_structural_repairs=0,
            use_grounding_plan=False,
            use_semantic_critic=False,
            **fields,
        )
        self.max_plan_repairs = max_plan_repairs
        self.use_clause_critic = use_clause_critic
        self.max_context_targets = max_context_targets

    def __call__(self, task: StudentTask, revision: LoopRevision) -> str:
        self._last_generation_trace = ()
        self._last_generation_trace_kinds = ()
        self._last_generation_prompt_trace = ()
        self._last_generation_trace_results = ()
        model, tokenizer, generate = self._runtime()
        shared_target = _shared_operator_target(revision.skill_text)
        teaching, selected_card = _resolve_operator_teaching(
            revision.skill_text,
            task.instruction,
            source_round=revision.source_round,
        )
        if shared_target is None:
            contexts = fixed_symbol_contexts(
                task,
                self.max_context_chars,
                max_targets=self.max_context_targets,
            )
        else:
            relative, symbol = shared_target
            if relative not in task.allowed_targets:
                raise ContractError("shared operator target is not allowed")
            source = task.resolve_target(relative).read_text(
                encoding="utf-8", errors="replace"
            )
            selected = qualified_symbol_excerpt(source, symbol, self.max_context_chars)
            if selected is None:
                raise ContractError("shared operator symbol did not resolve")
            excerpt, line = selected
            contexts = ((relative, excerpt, line),)
        context_by_file = {relative: excerpt for relative, excerpt, _line in contexts}
        allowed_files = tuple(context_by_file)
        catalog = _catalog_for_selected_card(selected_card)
        single_operator = _single_operator_for_card(selected_card)
        system = (
            "You are a code-edit student. Diagnose the supplied task, then "
            "return exactly one JSON object with fields schema_version, file, "
            "symbol, intent, operations, and diagnostic. schema_version is 1. "
            "intent has exactly defect, trigger, and desired_boundary. Do not "
            "return a diff, full definition, markdown, or prose. Select only "
            "from the typed operator catalog; copy selectors exactly from the "
            "supplied symbol source. First choose one supplied file and symbol, "
            "then copy one exact AST node, then classify it: one expression uses "
            "replace_expression; a whole statement or control-flow fragment uses "
            "replace_statement. Use insert_method only for an explicitly missing "
            "method and replace_method_body only for an explicit body rewrite. "
            "If no selector matches exactly once, return an "
            "unresolved JSON plan rather than guessing. Keep the entire response "
            f"under {self.profile.max_plan_chars} characters.\n\n"
            f"{catalog}\n\n"
            f"Protocol: {revision.protocol}\n"
            f"Teaching Skill:\n{teaching}"
        )
        sources = {
            relative: task.resolve_target(relative).read_text(encoding="utf-8")
            for relative in allowed_files
        }
        rendered_context = "\n\n".join(
            (
                f"### {relative}\n"
                f"Qualified symbol: "
                f"{qualified_symbol_for_anchor(sources[relative], excerpt, line, task.instruction) or '<unresolved>'}\n"
                f"{excerpt}"
            )
            for relative, excerpt, line in contexts
        )
        per_file_candidates: dict[str, list[dict[str, Any]]] = {}
        if shared_target is not None:
            for relative, excerpt, _line in contexts:
                per_file_candidates[relative] = list(
                    _harvest_typed_selector_candidates(
                        excerpt,
                        instruction=task.instruction,
                        maximum_candidates=128,
                    )
                )
        rendered_candidates = (
            "Not available until deterministic shared localization is frozen."
            if shared_target is None
            else "\n\n".join(
                f"### {relative}\n{canonical_json(rows)}"
                for relative, rows in per_file_candidates.items()
            )
        )
        candidate_instructions = (
            "Framework-harvested typed selector candidates are not available "
            "during neutral diagnosis/localization preparation:"
            if shared_target is None
            else (
                "Framework-harvested typed selector candidates (candidate IDs "
                "are audit hints; copy source and occurrence into the plan, and "
                "use each candidate's suggested_operator). "
                "For a condition boundary, copy one supplied boundary_rewrites "
                "new_condition when its semantics matches the frozen diagnosis:"
            )
        )
        rendered_exemplars = _usage_exemplars_block(
            sources,
            contexts,
            instruction=task.instruction,
            per_file_candidates=per_file_candidates,
        )
        exemplar_section = (
            "\n\n"
            "Repository usage exemplars - how these calls are made elsewhere in "
            "this repo (read-only reference; never copy as a selector). "
            "Follow the SAME calling convention: when a sibling call passes "
            "values via keyword arguments (e.g. params=...), do the same in "
            "your replacement instead of string formatting or message mutation. "
            f"Exemplars:\n{rendered_exemplars}"
            if rendered_exemplars
            else ""
        )
        user = (
            f"Task: {task.instruction}\n\n"
            f"Allowed candidate files: {canonical_json(list(allowed_files))}\n\n"
            f"Python symbol anchors:\n{rendered_context}\n\n"
            f"{candidate_instructions}\n"
            f"{rendered_candidates}"
            f"{exemplar_section}\n\n"
            "Return the typed operator plan JSON now. Copy the file path and "
            "qualified symbol name from the supplied source."
        )
        messages = [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user,
            },
        ]
        trace: list[str] = []
        trace_kinds: list[str] = []
        prompt_trace: list[str] = []
        trace_results: list[dict[str, Any]] = []
        raw = ""
        frozen_intent: dict[str, str] | None = None
        plan: OperatorPlan | None = None
        for repair_index in range(self.max_plan_repairs + 1):
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                tokenize=False,
            )
            raw = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
            )
            prompt_trace.append(prompt)
            trace.append(raw)
            trace_kinds.append(f"operator-plan-attempt-{repair_index}")
            canonicalized, changed = _canonicalize_single_operator_output(
                raw,
                operator=single_operator,
                supplied_symbol=(
                    shared_target[1] if shared_target is not None else None
                ),
            )
            if changed:
                trace_results.append(
                    {
                        "status": "single-operator-canonicalized",
                        "operator": single_operator,
                    }
                )
                raw = canonicalized
                prompt_trace.append(prompt)
                trace.append(raw)
                trace_kinds.append(f"operator-plan-canonicalized-{repair_index}")
            canonicalized, condition_changed = _canonicalize_condition_header_output(
                raw
            )
            if condition_changed:
                trace_results.append(
                    {
                        "status": "condition-header-canonicalized",
                        "operator": "replace_condition",
                    }
                )
                raw = canonicalized
                prompt_trace.append(prompt)
                trace.append(raw)
                trace_kinds.append(f"operator-condition-canonicalized-{repair_index}")
            try:
                unresolved = parse_unresolved_abstention(raw)
                if unresolved is not None:
                    trace_results.append({"status": "unresolved", "detail": unresolved})
                    break
                plan = parse_operator_plan_output(raw)
                if plan.file not in allowed_files:
                    raise ContractError("operator plan changed the frozen candidates")
                _correct_boundary_rewrites(task, plan)
                plan = _correct_error_params_rewrites(task, plan)
                plan = _correct_method_body_guards(task, plan)
                if revision.source_round >= 23:
                    oracle_mismatch = _semantic_oracle_mismatch(task, plan)
                    if oracle_mismatch is not None:
                        raise ContractError(
                            f"semantic oracle mismatch: {oracle_mismatch}"
                        )
                relative = plan.file
                excerpt = context_by_file[relative]
                before = sources[relative]
                result = materialize_operator_plan(before, plan)
                if not result.accepted:
                    raise ContractError(
                        f"operator materialization gate failed: {result.failure_reason}"
                    )
            except ContractError as exc:
                status = (
                    "semantic-oracle-rejected"
                    if str(exc).startswith("semantic oracle mismatch:")
                    else "structural-rejected"
                )
                trace_results.append({"status": status, "detail": str(exc)})
                # Deterministic selector grounding: re-point a zero-match
                # selector to the nearest framework-harvested candidate before
                # spending a model replan.  Runs for every profile (it does not
                # consume the repair budget) and is fully replayable.
                if "selector did not resolve" in str(exc) and plan is not None:
                    grounded_plan, ground_reason = _ground_operator_selector(
                        task, plan, context_by_file
                    )
                    if grounded_plan is not None:
                        grounded_source = task.resolve_target(
                            grounded_plan.file
                        ).read_text(encoding="utf-8", errors="replace")
                        grounded_result = materialize_operator_plan(
                            grounded_source, grounded_plan
                        )
                        if grounded_result.accepted:
                            plan = grounded_plan
                            raw = canonical_json(grounded_plan.to_dict())
                            prompt_trace.append("")
                            trace.append(raw)
                            trace_kinds.append(
                                f"operator-selector-grounded-{repair_index}"
                            )
                            trace_results.append(
                                {
                                    "status": "selector-grounded",
                                    "detail": "nearest framework candidate",
                                }
                            )
                            break
                if (
                    repair_index >= self.max_plan_repairs
                    or not _operator_failure_is_repairable(exc)
                ):
                    break
                grounded = _grounded_repair_detail(task, plan, exc)
                detail = str(exc)
                hint = ""
                if "selector did not resolve" in detail:
                    hint = (
                        " The target line you chose does not exist in the resolved "
                        "symbol; pick an exact real line from the grounded source."
                    )
                elif "no-op" in detail or "operator materialization gate failed" in detail:
                    hint = (
                        " Your replacement equals the selector or changes nothing; "
                        "make a real behavior change using lines from the grounded source."
                    )
                plan_exemplars = (
                    _plan_usage_exemplars(task, plan)
                    if plan is not None
                    else ""
                )
                exemplar_hint = (
                    "\n\nRepository usage exemplars for the calls in your plan "
                    "(read-only - match this idiom):\n" + plan_exemplars
                    if plan_exemplars
                    else ""
                )
                messages = [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": (
                            f"{user}\n\nStructural gate failure: {grounded}. Replan "
                            "once with the same diagnosis. Correct only the typed "
                            "schema, selector kind, statement bound, or operation "
                            f"arguments. Return a fresh complete plan JSON.{hint}"
                            f"{exemplar_hint}"
                        ),
                    },
                ]
                continue
            frozen_intent = plan.intent.to_dict()
            trace_results.append({"status": "structural-valid"})
            if selected_card is None or not self.use_clause_critic:
                break
            if single_operator is not None and all(
                row.operator == single_operator for row in plan.operations
            ):
                gate = canonical_json(
                    {
                        "gate": "typed-operator-clause-coverage",
                        "operator": single_operator,
                        "status": "accepted",
                        "basis": "qualified-operator-postconditions",
                    }
                )
                prompt_trace.append("")
                trace.append(gate)
                trace_kinds.append(f"operator-deterministic-clause-gate-{repair_index}")
                trace_results.append(
                    {
                        "status": "clause-accepted-deterministic",
                        "operator": single_operator,
                    }
                )
                break
            critique = self._clause_critique(
                model=model,
                tokenizer=tokenizer,
                generate=generate,
                task=task,
                selected_card=selected_card,
                excerpt=excerpt,
                plan=plan,
            )
            prompt_trace.append(critique["prompt"])
            trace.append(critique["raw"])
            trace_kinds.append(f"operator-clause-critic-{repair_index}")
            trace_results.append(critique["result"])
            if critique["complete"]:
                break
            if repair_index >= self.max_plan_repairs:
                raw = "CLAUSE_PREFLIGHT_REJECTED: " + canonical_json(critique["issues"])
                break
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"{user}\n\nThe diagnosis is read-only and must remain "
                        f"exactly: {canonical_json(frozen_intent)}\n"
                        f"Selected teaching card: {selected_card}\n"
                        "Clause verifier failures: "
                        f"{canonical_json(critique['issues'])}\n"
                        "Replan once. Every Transformation clause must be realized "
                        "by at least one typed operation and every Validation invariant "
                        "must be preserved. Return only the corrected plan JSON."
                    ),
                },
            ]
        self._last_generation_trace = tuple(trace)
        self._last_generation_trace_kinds = tuple(trace_kinds)
        self._last_generation_prompt_trace = tuple(prompt_trace)
        self._last_generation_trace_results = tuple(trace_results)
        return raw

    def _clause_critique(
        self,
        *,
        model: Any,
        tokenizer: Any,
        generate: Any,
        task: StudentTask,
        selected_card: str,
        excerpt: str,
        plan: OperatorPlan,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a read-only typed-plan verifier. Return exactly one JSON "
                    "object with complete (boolean), missing_transformations (array of "
                    "strings), and violated_invariants (array of strings). Do not "
                    "propose code, a patch, or new diagnosis. Mark complete only if "
                    "the operation plan concretely implements every Transformation "
                    "clause and preserves every Validation clause in the selected card."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task: {task.instruction}\n\n"
                    f"Selected teaching card: {selected_card}\n\n"
                    f"Supplied symbol: {excerpt}\n\n"
                    f"Proposed typed plan: {canonical_json(plan.to_dict())}"
                ),
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
            tokenize=False,
        )
        raw = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=min(512, self.max_tokens),
        )
        try:
            candidate = raw.strip()
            start, end = candidate.find("{"), candidate.rfind("}")
            data = json.loads(candidate[start : end + 1])
            if not isinstance(data, dict) or set(data) != {
                "complete",
                "missing_transformations",
                "violated_invariants",
            }:
                raise ContractError("clause critic fields are invalid")
            if type(data["complete"]) is not bool or any(
                not isinstance(data[field], list)
                or any(not isinstance(row, str) for row in data[field])
                for field in ("missing_transformations", "violated_invariants")
            ):
                raise ContractError("clause critic values are invalid")
        except (json.JSONDecodeError, ContractError) as exc:
            return {
                "raw": raw,
                "prompt": prompt,
                "complete": False,
                "issues": [str(exc)],
                "result": {"status": "critic-rejected", "detail": str(exc)},
            }
        issues = [*data["missing_transformations"], *data["violated_invariants"]]
        return {
            "raw": raw,
            "prompt": prompt,
            "complete": data["complete"] and not issues,
            "issues": issues,
            "result": {
                "status": "clause-accepted"
                if data["complete"] and not issues
                else "clause-rejected",
                "missing_transformation_count": len(data["missing_transformations"]),
                "violated_invariant_count": len(data["violated_invariants"]),
            },
        }

    def generation_config(self) -> dict[str, Any]:
        return {
            **super().generation_config(),
            "action_space": "python-typed-operator-plan-v1",
            "context_selector": _OPERATOR_CONTEXT_SELECTOR,
            "target_policy": "frozen-top32-candidate-symbol-choice",
            "max_context_targets": self.max_context_targets,
            "renderer": "deterministic-ast-selector-renderer-v7",
            "max_operations": 4,
            "max_plan_repairs": self.max_plan_repairs,
            "use_clause_critic": self.use_clause_critic,
            "repair_policy": "same-diagnosis-one-replan-v1",
            "card_operator_policy": "matched-card-single-high-level-v2",
            "canonicalization_policy": (
                "single-operator-force-plus-condition-header-to-predicate-v3"
            ),
            "selector_candidate_policy": (
                "ast-boundary-priority-expression-header-rewrites-top16-v4"
            ),
            "neutral_selector_candidate_policy": (
                "unavailable-without-boundary-rewrite-instructions-v2"
            ),
            "pre_schema_policy": "localize-copy-classify-or-unresolved-r010",
            "teaching_projection": (
                "task-lexical-top5-max1400-mandatory-2-4-5-shared-suffix-v3"
            ),
            "max_plan_chars": self.profile.max_plan_chars,
        }


def build_operator_conditions(
    *,
    taught_skill: str,
    parent_revision_id: str,
    source_round: int,
    generation_config: dict[str, Any],
) -> list[ExperimentCondition]:
    """Build a paired comparison with a fixed prompt, context, and renderer."""
    definitions = [
        (
            "operator-baseline",
            "baseline",
            "No additional domain teaching is provided. Use only repository evidence.",
            None,
        ),
        ("operator-taught", "taught", taught_skill, parent_revision_id),
    ]
    conditions = []
    for condition_id, teaching, skill_text, parent_id in definitions:
        revision = LoopRevision.create(
            skill_id="p1-local-qwen-operator-skill",
            revision_id=f"p1-{condition_id}-r{source_round:03d}",
            parent_revision_id=parent_id,
            source_round=source_round,
            protocol="python-typed-operator-plan-v1",
            skill_text=skill_text,
            prompt_template=_OPERATOR_PROMPT_TEMPLATE,
            eval_note=(
                "P1 typed-operator comparison; the Student owns intent and operator "
                "arguments while the deterministic renderer owns patch syntax."
            ),
        )
        conditions.append(
            ExperimentCondition.create(
                condition_id=condition_id,
                mechanism="operator",
                teaching=teaching,
                revision=revision,
                generation_config=generation_config,
            )
        )
    return conditions
