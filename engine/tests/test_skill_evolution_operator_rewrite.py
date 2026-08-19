"""Tests for typed operator patch realization and its qualification gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skill_evolution_loop import ContractError
from skill_evolution_loop.contracts import sha256_json
from skill_evolution_loop.operator_rewrite import (
    OperatorPlan,
    materialize_operator_plan,
    run_operator_renderer_qualification,
)


def _plan(operation: dict, *, symbol: str = "compute") -> OperatorPlan:
    return OperatorPlan.from_dict(
        {
            "schema_version": 1,
            "file": "module.py",
            "symbol": symbol,
            "intent": {
                "defect": "boundary is wrong",
                "trigger": "value equals threshold",
                "desired_boundary": "include threshold",
            },
            "operations": [operation],
            "diagnostic": "Use one typed operation.",
        }
    )


def test_replace_expression_materializes_a_non_noop_ast_change() -> None:
    source = "def compute(value):\n    return value < 10\n"
    result = materialize_operator_plan(
        source,
        _plan(
            {
                "operator": "replace_expression",
                "selector": {"source": "value < 10", "occurrence": 0},
                "arguments": {"new_expression": "value <= 10"},
            }
        ),
    )

    assert result.accepted is True
    assert result.after == "def compute(value):\n    return value <= 10\n"
    assert result.gates == {
        "selector_match": True,
        "syntax_valid": True,
        "ast_code_diff": True,
        "non_copy": True,
        "postconditions": True,
    }


def test_replace_condition_materializes_only_the_control_flow_test() -> None:
    source = (
        "def compute(values):\n"
        "    if not values:\n"
        "        return 'implicit'\n"
        "    return list(values)\n"
    )

    result = materialize_operator_plan(
        source,
        _plan(
            {
                "operator": "replace_condition",
                "selector": {"source": "not values", "occurrence": 0},
                "arguments": {"new_condition": "values is None"},
            }
        ),
    )

    assert result.accepted is True
    assert result.after == (
        "def compute(values):\n"
        "    if values is None:\n"
        "        return 'implicit'\n"
        "    return list(values)\n"
    )


def test_statement_insertion_and_constant_replacement_are_deterministic() -> None:
    source = "def compute(value):\n    return value + 1\n"
    inserted = materialize_operator_plan(
        source,
        _plan(
            {
                "operator": "insert_assignment_before",
                "selector": {"source": "return value + 1", "occurrence": 0},
                "arguments": {"name": "offset", "expression": "2"},
            }
        ),
    )
    assert inserted.accepted is True
    assert inserted.after == (
        "def compute(value):\n    offset = 2\n    return value + 1\n"
    )

    constant = materialize_operator_plan(
        source,
        _plan(
            {
                "operator": "replace_constant",
                "selector": {"value": 1, "occurrence": 0},
                "arguments": {"new_value": 2},
            }
        ),
    )
    assert constant.accepted is True
    assert constant.after == "def compute(value):\n    return value + 2\n"


def test_replace_statement_accepts_bounded_statements_not_a_definition() -> None:
    source = "def compute(value):\n    result = value + 1\n    return result\n"
    result = materialize_operator_plan(
        source,
        _plan(
            {
                "operator": "replace_statement",
                "selector": {"source": "result = value + 1", "occurrence": 0},
                "arguments": {
                    "new_statements": "result = value + 2\nresult = max(result, 0)"
                },
            }
        ),
    )
    assert result.accepted is True
    assert "result = max(result, 0)" in result.after

    listed = materialize_operator_plan(
        source,
        _plan(
            {
                "operator": "replace_statement",
                "selector": {"source": "result = value + 1", "occurrence": 0},
                "arguments": {
                    "new_statements": [
                        "result = value + 2",
                        "result = max(result, 0)",
                    ]
                },
            }
        ),
    )
    assert listed.accepted is True
    assert "['result = value" not in listed.after
    assert "result = max(result, 0)" in listed.after

    with pytest.raises(ContractError, match="one to four non-definition statements"):
        materialize_operator_plan(
            source,
            _plan(
                {
                    "operator": "replace_statement",
                    "selector": {
                        "source": "result = value + 1",
                        "occurrence": 0,
                    },
                    "arguments": {"new_statements": "def hidden():\n    pass"},
                }
            ),
        )


def test_noop_copy_and_out_of_symbol_selector_fail_closed() -> None:
    source = (
        "LIMIT = 10\n\n"
        "def other():\n    return LIMIT\n\n"
        "def compute(value):\n    return value < 10\n"
    )
    noop = materialize_operator_plan(
        source,
        _plan(
            {
                "operator": "replace_expression",
                "selector": {"source": "value < 10", "occurrence": 0},
                "arguments": {"new_expression": "value < 10"},
            }
        ),
    )
    assert noop.accepted is False
    assert noop.failure_reason == "no-op"
    assert noop.gates["ast_code_diff"] is False

    with pytest.raises(ContractError, match="selector did not resolve"):
        materialize_operator_plan(
            source,
            _plan(
                {
                    "operator": "replace_constant",
                    "selector": {"value": "LIMIT", "occurrence": 0},
                    "arguments": {"new_value": "OTHER"},
                }
            ),
        )


def test_unique_unqualified_method_name_resolves_inside_class() -> None:
    source = (
        "class Writer:\n    def visit_literal(self, value):\n        return value + 1\n"
    )
    plan = _plan(
        {
            "operator": "replace_constant",
            "selector": {"value": 1, "occurrence": 0},
            "arguments": {"new_value": 2},
        },
        symbol="visit_literal",
    )

    result = materialize_operator_plan(source, plan)

    assert result.accepted is True
    assert "return value + 2" in result.after


def test_high_level_defaults_operator_renders_aligned_loops() -> None:
    source = """def parse(args):
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
    result = materialize_operator_plan(
        source,
        _plan(
            {
                "operator": "align_trailing_defaults",
                "selector": {"occurrence": 0},
                "arguments": {},
            },
            symbol="parse",
        ),
    )

    assert result.accepted is True
    assert "aligned_defaults = [Parameter.empty]" in result.after
    assert "for i, arg in enumerate(positional_only):" in result.after
    assert "for i, arg in enumerate(args.args, len(positional_only)):" in result.after


def test_high_level_inline_operator_renders_percent_newline_boundaries() -> None:
    source = r"""class Writer:
    def visit_literal(self, node):
        hlcode = self.highlighter.highlight_block(node.astext(), 'python')
        hlcode = hlcode.replace(r'\begin{Verbatim}[commandchars=\\\{\}]', r'\sphinxcode{\sphinxupquote{')
        hlcode = hlcode.rstrip()[:-14]
        self.body.append(hlcode)
"""
    result = materialize_operator_plan(
        source,
        _plan(
            {
                "operator": "normalize_inline_wrapper_boundaries",
                "selector": {"occurrence": 0},
                "arguments": {},
            },
            symbol="visit_literal",
        ),
    )

    assert result.accepted is True
    assert "sphinxupquote{%" in result.after
    assert ".rstrip() + '%\\n'" in result.after


def test_generated_subclass_identity_operator_updates_class_and_factory() -> None:
    source = """class MockObject:
    __display_name__ = 'MockObject'

    def __init__(self):
        self.__qualname__ = ''

    def __repr__(self):
        return self.__display_name__


def make_subclass(name, module, superclass=MockObject, attributes=None):
    attrs = {'__module__': module,
             '__display_name__': module + '.' + name}
    attrs.update(attributes or {})
    return type(name, (superclass,), attrs)
"""
    result = materialize_operator_plan(
        source,
        _plan(
            {
                "operator": "initialize_generated_subclass_identity",
                "selector": {"occurrence": 0},
                "arguments": {},
            },
            symbol="MockObject",
        ),
    )

    assert result.accepted is True
    assert "self.__qualname__ = self.__class__.__qualname__" in result.after
    assert "'__qualname__': name" in result.after
    assert "return self.__display_name__" in result.after
    assert "attrs.update(attributes or {})" in result.after


def test_generated_subclass_identity_operator_requires_linked_factory() -> None:
    source = """class MockObject:
    __display_name__ = 'MockObject'

    def __init__(self):
        self.__qualname__ = ''
"""

    with pytest.raises(ContractError, match="selector did not resolve; matches=0"):
        materialize_operator_plan(
            source,
            _plan(
                {
                    "operator": "initialize_generated_subclass_identity",
                    "selector": {"occurrence": 0},
                    "arguments": {},
                },
                symbol="MockObject",
            ),
        )


@pytest.mark.parametrize(
    "field",
    (
        "PyTypedField('variable', names=('var',), rolename=\"obj\")",
        "PyTypedField('variable', rolename = 'obj', names=('var',))",
    ),
)
def test_remove_variable_obj_role_is_independent_of_source_format(field: str) -> None:
    source = f"class DocFieldTransformer:\n    field = {field}\n"

    result = materialize_operator_plan(
        source,
        _plan(
            {
                "operator": "remove_variable_obj_role",
                "selector": {"occurrence": 0},
                "arguments": {},
            },
            symbol="DocFieldTransformer",
        ),
    )

    assert result.accepted is True
    assert "rolename" not in result.after
    assert "names=('var',)" in result.after


def test_remove_variable_obj_role_handles_multiline_comments() -> None:
    source = """class DocFieldTransformer:
    field = PyTypedField(
        'variable',
        names=('var',),  # documented field names
        rolename='obj',  # remove the object role
    )
"""

    result = materialize_operator_plan(
        source,
        _plan(
            {
                "operator": "remove_variable_obj_role",
                "selector": {"occurrence": 0},
                "arguments": {},
            },
            symbol="DocFieldTransformer",
        ),
    )

    assert result.accepted is True
    assert "rolename" not in result.after
    assert "remove the object role" not in result.after
    assert "documented field names" in result.after


def test_remove_variable_obj_role_handles_last_keyword_after_comment() -> None:
    source = """class DocFieldTransformer:
    field = PyTypedField(
        'variable',
        names=('var',),  # documented field names
        rolename='obj')
"""

    result = materialize_operator_plan(
        source,
        _plan(
            {
                "operator": "remove_variable_obj_role",
                "selector": {"occurrence": 0},
                "arguments": {},
            },
            symbol="DocFieldTransformer",
        ),
    )

    assert result.accepted is True
    assert "rolename" not in result.after
    assert "documented field names" in result.after


def test_insert_method_adds_one_bounded_method_to_a_class() -> None:
    source = (
        "class Bucket:\n    def add(self, value):\n        self.values.append(value)\n"
    )
    result = materialize_operator_plan(
        source,
        _plan(
            {
                "operator": "insert_method",
                "selector": {"occurrence": 0},
                "arguments": {
                    "method_source": "def __len__(self):\n    return len(self.values)"
                },
            },
            symbol="Bucket",
        ),
    )

    assert result.accepted is True
    assert "    def __len__(self):\n        return len(self.values)\n" in result.after


def test_replace_method_body_preserves_signature_and_docstring() -> None:
    source = (
        "class Bucket:\n"
        "    def size(self, scale=1):\n"
        '        """Return the scaled size."""\n'
        "        return 0\n"
    )
    result = materialize_operator_plan(
        source,
        _plan(
            {
                "operator": "replace_method_body",
                "selector": {"occurrence": 0},
                "arguments": {"new_body": "return len(self.values) * scale"},
            },
            symbol="Bucket.size",
        ),
    )

    assert result.accepted is True
    assert "def size(self, scale=1):" in result.after
    assert '        """Return the scaled size."""' in result.after
    assert "        return len(self.values) * scale" in result.after


def test_operator_scope_gate_rejects_new_unbound_names() -> None:
    source = "def compute(value):\n    return value + 1\n"
    result = materialize_operator_plan(
        source,
        _plan(
            {
                "operator": "replace_expression",
                "selector": {"source": "value + 1", "occurrence": 0},
                "arguments": {"new_expression": "report + 1"},
            }
        ),
    )

    assert result.accepted is False
    assert result.failure_reason == "unbound-name"


def test_synthetic_renderer_qualification_freezes_26_case_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "qualification.json"
    report = run_operator_renderer_qualification(output)

    assert report["status"] == "qualified"
    assert report["planned_cases"] == 26
    assert report["accepted_cases"] == 26
    assert report["no_op_cases"] == 0
    assert report["qualification_threshold"] == 23
    assert report["network_calls_performed"] is False
    assert report["holdout_task_ids_included"] is False
    content = {key: value for key, value in report.items() if key != "evidence_sha256"}
    assert report["evidence_sha256"] == sha256_json(content)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert run_operator_renderer_qualification(output) == report


def test_replace_expression_on_statement_autocorrects_to_replace_statement() -> None:
    """A1: a statement selector with a real replacement is auto-corrected.

    The weak Student classifies an assignment/raise as replace_expression;
    the harness deterministically downgrades it to replace_statement and
    materializes a non-empty patch (no model call).
    """
    source = "def compute():\n    xy = xy_0\n    return xy\n"
    result = materialize_operator_plan(
        source,
        _plan(
            {
                "operator": "replace_expression",
                "selector": {"source": "xy = xy_0", "occurrence": 0},
                "arguments": {"new_expression": "xy = xy_0.copy()"},
            }
        ),
    )

    assert result.accepted is True
    assert "xy = xy_0.copy()" in result.after
    assert result.after != source


def test_replace_expression_on_statement_noop_still_rejected() -> None:
    """A1 is fail-closed on tautological plans: selector == replacement."""
    source = "def compute():\n    raise ValueError('x')\n"
    result = materialize_operator_plan(
        source,
        _plan(
            {
                "operator": "replace_expression",
                "selector": {"source": "raise ValueError('x')", "occurrence": 0},
                "arguments": {"new_expression": "raise ValueError('x')"},
            }
        ),
    )

    assert result.accepted is False
    assert result.failure_reason in {"no-op", "apply-fail"}


def test_operator_operation_normalizes_operator_as_key_shape() -> None:
    """7B-style {"<operator>": {"selector","arguments"}} maps to the contract shape."""
    from skill_evolution_loop.operator_rewrite import OperatorOperation

    op = OperatorOperation.from_dict(
        {
            "replace_expression": {
                "selector": {"source": "value + 1", "occurrence": 0},
                "arguments": {"new_expression": "value + 2"},
            }
        }
    )
    assert op.operator == "replace_expression"
    assert op.selector == {"source": "value + 1", "occurrence": 0}
    assert op.arguments == {"new_expression": "value + 2"}


def test_operator_operation_rejects_non_operator_as_key_shape() -> None:
    from skill_evolution_loop.contracts import ContractError
    from skill_evolution_loop.operator_rewrite import OperatorOperation

    try:
        OperatorOperation.from_dict(
            {
                "not_an_operator": {
                    "selector": {"source": "x", "occurrence": 0},
                    "arguments": {"new_expression": "y"},
                }
            }
        )
    except ContractError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown operator-as-key must be rejected")


def test_operator_operation_normalizes_flattened_operator_as_key_shape() -> None:
    """Flattened variant {"<op>": {"source","occurrence","arguments"}} normalizes too."""
    from skill_evolution_loop.operator_rewrite import OperatorOperation

    op = OperatorOperation.from_dict(
        {
            "replace_condition": {
                "source": "len(x) == 0",
                "occurrence": 1,
                "arguments": {"new_condition": "x is None"},
            }
        }
    )
    assert op.operator == "replace_condition"
    assert op.selector == {"source": "len(x) == 0", "occurrence": 1}
    assert op.arguments == {"new_condition": "x is None"}


def test_operator_operation_normalizes_sibling_arguments_shape() -> None:
    """Sibling variant {"<op>": {"source","occurrence"}, "arguments": {...}} normalizes too."""
    from skill_evolution_loop.operator_rewrite import OperatorOperation

    op = OperatorOperation.from_dict(
        {
            "replace_condition": {
                "source": "len(x) == 0",
                "occurrence": 1,
            },
            "arguments": {"new_condition": "x is None"},
        }
    )
    assert op.operator == "replace_condition"
    assert op.selector == {"source": "len(x) == 0", "occurrence": 1}
    assert op.arguments == {"new_condition": "x is None"}


def test_operator_operation_uses_suggested_operator_when_operator_absent() -> None:
    from skill_evolution_loop.operator_rewrite import OperatorOperation

    op = OperatorOperation.from_dict(
        {
            "suggested_operator": "replace_expression",
            "selector": {"source": "value + 1", "occurrence": 0},
            "arguments": {"new_expression": "value + 2"},
        }
    )
    assert op.operator == "replace_expression"


def test_operator_operation_drops_redundant_suggested_operator() -> None:
    from skill_evolution_loop.operator_rewrite import OperatorOperation

    op = OperatorOperation.from_dict(
        {
            "operator": "replace_expression",
            "suggested_operator": "replace_statement",
            "selector": {"source": "value + 1", "occurrence": 0},
            "arguments": {"new_expression": "value + 2"},
        }
    )
    assert op.operator == "replace_expression"


def test_replace_method_body_normalizes_list_new_body() -> None:
    """new_body as a list of statements joins into a string (django-15277 7B shape)."""
    from skill_evolution_loop.operator_rewrite import OperatorOperation

    op = OperatorOperation.from_dict(
        {
            "operator": "replace_method_body",
            "selector": {"occurrence": 0},
            "arguments": {
                "new_body": [
                    "if self.max_length is not None:",
                    "    self.validators.append(validators.MaxLengthValidator(self.max_length))",
                ]
            },
        }
    )
    assert op.arguments["new_body"] == (
        "if self.max_length is not None:\n"
        "    self.validators.append(validators.MaxLengthValidator(self.max_length))"
    )
