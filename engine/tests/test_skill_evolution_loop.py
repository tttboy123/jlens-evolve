"""Behavior tests for the offline Skill feedback loop P0."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from skill_evolution_loop import (
    EvaluationBaseline,
    EvaluationPolicy,
    HunkStudentAdapter,
    LoopAuthorization,
    LoopConfig,
    LoopDriver,
    LoopEvaluator,
    LoopRevision,
    LoopRevisionRegistry,
    MlxHunkGenerator,
    MlxStructuredGenerator,
    NativeOutcome,
    ParentCallLedger,
    ParentModelAdapter,
    StudentAdapter,
    StudentTask,
)
from skill_evolution_loop.mlx_student import _select_pattern_card


def _git_checkout(path: Path, files: dict[str, str]) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    for relative, content in files.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Loop Test",
            "-c",
            "user.email=loop@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=path,
        check=True,
    )
    return path


def _revision(
    revision_id: str = "rev-001", parent_revision_id: str | None = None, round_: int = 0
) -> LoopRevision:
    return LoopRevision.create(
        skill_id="local-edit-skill",
        revision_id=revision_id,
        parent_revision_id=parent_revision_id,
        source_round=round_,
        protocol="structured-search-replace-v1",
        skill_text="Return one bounded edit.",
        prompt_template="Edit {target_path} as JSON.",
        eval_note="offline fixture",
    )


def _task(
    checkout: Path,
    *,
    task_id: str = "task-001",
    cohort: str = "feedback",
) -> StudentTask:
    return StudentTask.create(
        task_id=task_id,
        checkout=checkout,
        instruction="Make answer return two.",
        allowed_targets=["src/example.py"],
        cohort=cohort,
    )


def test_student_adapter_builds_a_fail_closed_unique_structured_edit(
    tmp_path: Path,
) -> None:
    checkout = _git_checkout(
        tmp_path / "repo", {"src/example.py": "def answer():\n    return 1\n"}
    )

    def generator(_task: StudentTask, _revision: LoopRevision) -> str:
        return json.dumps(
            {
                "file": "src/example.py",
                "search": "def answer():\n    return 1",
                "replace": "def answer():\n    return 2",
                "diagnostic": "The return value is stale.",
            }
        )

    adapter = StudentAdapter(generator=generator)
    attempt = adapter.run(_task(checkout), _revision())

    assert attempt.structural_valid is True
    assert attempt.failure_reason is None
    assert attempt.implementation_fingerprint is not None
    assert "-    return 1" in attempt.patch
    assert "+    return 2" in attempt.patch
    assert (checkout / "src/example.py").read_text() == "def answer():\n    return 1\n"

    adapter.apply(attempt)
    assert (checkout / "src/example.py").read_text() == "def answer():\n    return 2\n"


def test_student_adapter_implementation_fingerprint_ignores_python_comments(
    tmp_path: Path,
) -> None:
    checkout = _git_checkout(
        tmp_path / "repo", {"src/example.py": "def answer():\n    return 1\n"}
    )

    def run(replacement: str):
        return StudentAdapter(
            generator=lambda _task, _revision: json.dumps(
                {
                    "file": "src/example.py",
                    "search": "def answer():\n    return 1",
                    "replace": replacement,
                    "diagnostic": "candidate",
                }
            )
        ).run(_task(checkout), _revision())

    plain = run("def answer():\n    return 2")
    commented = run("def answer():\n    # formatting-only difference\n    return 2")

    assert plain.patch_sha256 != commented.patch_sha256
    assert plain.implementation_fingerprint == commented.implementation_fingerprint


def test_student_adapter_classifies_wrong_target_and_ambiguous_search(
    tmp_path: Path,
) -> None:
    checkout = _git_checkout(
        tmp_path / "repo",
        {
            "src/example.py": "value = 1\nvalue = 1\n",
            "tests/test_example.py": "def test_value(): pass\n",
        },
    )
    responses = iter(
        [
            {
                "file": "tests/test_example.py",
                "search": "pass",
                "replace": "assert True",
                "diagnostic": "edit the test",
            },
            {
                "file": "src/example.py",
                "search": "value = 1",
                "replace": "value = 2",
                "diagnostic": "ambiguous search",
            },
        ]
    )
    adapter = StudentAdapter(
        generator=lambda _task, _revision: json.dumps(next(responses))
    )

    wrong = adapter.run(_task(checkout), _revision())
    ambiguous = adapter.run(_task(checkout), _revision())

    assert wrong.failure_reason == "wrong-target"
    assert ambiguous.failure_reason == "apply-fail"
    assert ambiguous.structural_valid is False


def test_evaluator_requires_structural_native_gain_and_holdout_non_regression(
    tmp_path: Path,
) -> None:
    feedback_repo = _git_checkout(tmp_path / "feedback", {"src/example.py": "x = 1\n"})
    holdout_repo = _git_checkout(tmp_path / "holdout", {"src/example.py": "x = 1\n"})
    adapter = StudentAdapter(
        generator=lambda _task, _revision: json.dumps(
            {
                "file": "src/example.py",
                "search": "x = 1",
                "replace": "x = 2",
                "diagnostic": "bounded edit",
            }
        )
    )
    attempts = [
        adapter.run(_task(feedback_repo), _revision()),
        adapter.run(
            _task(holdout_repo, task_id="task-holdout", cohort="holdout"),
            _revision(),
        ),
    ]
    outcomes = {
        "task-001": NativeOutcome(resolved=True, safe=True),
        "task-holdout": NativeOutcome(resolved=True, safe=True),
    }
    evaluator = LoopEvaluator(EvaluationPolicy.strict())
    result = evaluator.evaluate(
        attempts,
        native_evaluator=lambda attempt: outcomes[attempt.task.task_id],
        baseline=EvaluationBaseline(feedback_native_rate=0.0, holdout_native_rate=1.0),
    )

    assert result.structural_rate == 1.0
    assert result.feedback_native_gain == 1.0
    assert result.holdout_native_gain == 0.0
    assert result.converged is True


def test_loop_driver_runs_failure_feedback_regeneration_and_convergence(
    tmp_path: Path,
) -> None:
    feedback_repo = _git_checkout(tmp_path / "feedback", {"src/example.py": "x = 1\n"})
    holdout_repo = _git_checkout(tmp_path / "holdout", {"src/example.py": "x = 1\n"})
    parent_requests = []

    def student_generator(task: StudentTask, revision: LoopRevision) -> str:
        if revision.revision_id == "rev-001":
            return "The value should probably be changed, but here is no edit."
        return json.dumps(
            {
                "file": "src/example.py",
                "search": "x = 1",
                "replace": "x = 2",
                "diagnostic": f"bounded edit for {task.task_id}",
            }
        )

    def parent_transport(request):
        parent_requests.append(request)
        return {
            "schema_version": 1,
            "protocol": "structured-search-replace-v1",
            "skill_text": "Use an exact non-test search span and one replacement.",
            "prompt_template": "Return one JSON edit for {target_path}.",
            "eval_note": "Switched from prose to a bounded edit contract.",
            "usage": {"total_tokens": 50},
        }

    authorization = LoopAuthorization.create(
        authorization_id="loop-p0-local",
        approved_by="user",
        maximum_parent_calls=2,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    registry = LoopRevisionRegistry(tmp_path / "registry")
    parent = ParentModelAdapter(
        ledger=ParentCallLedger(tmp_path / "parent-calls.json", authorization),
        transport=parent_transport,
    )
    driver = LoopDriver(
        student=StudentAdapter(generator=student_generator),
        evaluator=LoopEvaluator(EvaluationPolicy.strict()),
        parent=parent,
        registry=registry,
        evidence_root=tmp_path / "evidence",
        authorization=authorization,
        config=LoopConfig(max_rounds=3, no_progress_patience=2),
    )
    tasks = [
        _task(feedback_repo),
        _task(holdout_repo, task_id="task-holdout", cohort="holdout"),
    ]

    result = driver.run(
        initial_revision=_revision(),
        tasks=tasks,
        native_evaluator=lambda attempt: NativeOutcome(
            resolved=attempt.structural_valid, safe=True
        ),
        baseline=EvaluationBaseline(feedback_native_rate=0.0, holdout_native_rate=0.0),
    )

    assert result.status == "converged"
    assert result.rounds_completed == 2
    assert result.final_revision.parent_revision_id == "rev-001"
    assert len(parent_requests) == 1
    assert [row.task_id for row in parent_requests[0].feedback.arm_evidence] == [
        "task-001"
    ]
    assert (tmp_path / "evidence/round-000/ROUND.json").is_file()
    assert (tmp_path / "evidence/round-001/ROUND.json").is_file()


def test_loop_driver_stops_and_rolls_back_on_equivalent_parent_mechanism(
    tmp_path: Path,
) -> None:
    feedback_repo = _git_checkout(tmp_path / "feedback", {"src/example.py": "x = 1\n"})
    holdout_repo = _git_checkout(tmp_path / "holdout", {"src/example.py": "x = 1\n"})
    authorization = LoopAuthorization.create(
        authorization_id="loop-no-progress",
        approved_by="user",
        maximum_parent_calls=1,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    parent = ParentModelAdapter(
        ledger=ParentCallLedger(tmp_path / "calls.json", authorization),
        transport=lambda request: {
            "schema_version": 1,
            "protocol": request.current_revision.protocol,
            "skill_text": request.current_revision.skill_text,
            "prompt_template": request.current_revision.prompt_template,
            "eval_note": "No mechanism change.",
            "usage": {"total_tokens": 10},
        },
    )
    driver = LoopDriver(
        student=StudentAdapter(generator=lambda _task, _revision: "reasoning only"),
        evaluator=LoopEvaluator(EvaluationPolicy.strict()),
        parent=parent,
        registry=LoopRevisionRegistry(tmp_path / "registry"),
        evidence_root=tmp_path / "evidence",
        authorization=authorization,
        config=LoopConfig(max_rounds=3, no_progress_patience=1),
    )

    result = driver.run(
        initial_revision=_revision(),
        tasks=[
            _task(feedback_repo),
            _task(holdout_repo, task_id="task-holdout", cohort="holdout"),
        ],
        native_evaluator=lambda _attempt: NativeOutcome(resolved=False, safe=True),
        baseline=EvaluationBaseline(feedback_native_rate=0.0, holdout_native_rate=0.0),
    )

    assert result.status == "no-progress"
    assert result.final_revision.revision_id == "rev-001"
    round_report = json.loads((tmp_path / "evidence/round-000/ROUND.json").read_text())
    assert round_report["terminal_status"] == "no-progress"


def test_mlx_structured_generator_is_lazy_cached_and_uses_revision_skill(
    tmp_path: Path,
) -> None:
    checkout = _git_checkout(
        tmp_path / "repo", {"src/example.py": "def answer():\n    return 1\n"}
    )
    loads: list[str] = []
    prompts: list[str] = []

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            assert add_generation_prompt is True
            assert enable_thinking is False
            return "\n".join(row["content"] for row in messages)

    def load(model_path: str):
        loads.append(model_path)
        return object(), Tokenizer()

    def generate(_model, _tokenizer, *, prompt: str, max_tokens: int):
        prompts.append(prompt)
        assert max_tokens == 100
        return "{}"

    generator = MlxStructuredGenerator(
        model_path="fixture-model",
        max_tokens=100,
        loader=load,
        text_generator=generate,
    )
    task = _task(checkout)
    generator(task, _revision())
    generator(task, _revision())

    assert loads == ["fixture-model"]
    assert "Return one bounded edit." in prompts[0]
    assert "src/example.py" in prompts[0]
    assert "return 1" in prompts[0]
    assert generator.generation_trace_kinds() == ("edit-attempt-0",)
    assert generator.generation_prompt_trace() == (prompts[-1],)


def test_mlx_structured_generator_retries_one_rejected_noop_with_exact_feedback(
    tmp_path: Path,
) -> None:
    checkout = _git_checkout(
        tmp_path / "repo", {"src/example.py": "def answer():\n    return 1\n"}
    )
    outputs = iter(
        [
            json.dumps(
                {
                    "file": "src/example.py",
                    "search": "return 1",
                    "replace": "return 1",
                    "diagnostic": "noop",
                }
            ),
            json.dumps(
                {
                    "file": "src/example.py",
                    "search": "return 1",
                    "replace": "return 2",
                    "diagnostic": "corrected",
                }
            ),
        ]
    )
    prompts: list[str] = []

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            return "\n".join(row["content"] for row in messages)

    generator = MlxStructuredGenerator(
        model_path="fixture-model",
        max_tokens=100,
        max_structural_repairs=1,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda _model, _tokenizer, *, prompt, max_tokens: (
            prompts.append(prompt) or next(outputs)
        ),
    )

    raw = generator(_task(checkout), _revision())

    assert json.loads(raw)["replace"] == "return 2"
    assert len(generator.generation_trace()) == 2
    assert "student search and replacement are identical" in prompts[1]
    assert generator.generation_config()["max_structural_repairs"] == 1


def test_mlx_structured_generator_grounds_edit_in_auditable_local_plan(
    tmp_path: Path,
) -> None:
    checkout = _git_checkout(
        tmp_path / "repo", {"src/example.py": "def answer():\n    return 1\n"}
    )
    plan = json.dumps(
        {
            "target_file": "src/example.py",
            "target_symbol": "answer",
            "source_identifiers": ["answer", "return"],
            "transform_steps": ["change the returned value"],
            "invariants": ["keep the function signature"],
        }
    )
    edit = json.dumps(
        {
            "file": "src/example.py",
            "search": "return 1",
            "replace": "return 2",
            "diagnostic": "answer and return match the plan",
        }
    )
    outputs = iter([plan, edit])
    prompts: list[str] = []

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            return "\n".join(row["content"] for row in messages)

    generator = MlxStructuredGenerator(
        model_path="fixture-model",
        max_tokens=512,
        use_grounding_plan=True,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda _model, _tokenizer, *, prompt, max_tokens: (
            prompts.append(prompt) or next(outputs)
        ),
    )

    raw = generator(_task(checkout), _revision())

    assert json.loads(raw)["replace"] == "return 2"
    assert generator.generation_trace() == (plan, edit)
    assert generator.generation_trace_kinds() == (
        "grounding-plan",
        "edit-attempt-0",
    )
    assert generator.generation_prompt_trace() == tuple(prompts)
    assert "code-localization planner" in prompts[0]
    assert plan in prompts[1]
    assert generator.generation_config()["use_grounding_plan"] is True


def test_mlx_structured_generator_repairs_a_duplicate_loop_outside_search_span(
    tmp_path: Path,
) -> None:
    checkout = _git_checkout(
        tmp_path / "repo",
        {
            "src/example.py": (
                "for item in left:\n"
                "    consume(item)\n"
                "for item in right:\n"
                "    consume(item)\n"
            )
        },
    )
    duplicate = json.dumps(
        {
            "file": "src/example.py",
            "search": "for item in right:\n    consume(item)",
            "replace": (
                "for item in left:\n    consume(item)\n"
                "for item in right:\n    consume_fixed(item)"
            ),
            "diagnostic": "accidentally duplicates left loop",
        }
    )
    corrected = json.dumps(
        {
            "file": "src/example.py",
            "search": "for item in right:\n    consume(item)",
            "replace": "for item in right:\n    consume_fixed(item)",
            "diagnostic": "changes only the existing right loop",
        }
    )
    outputs = iter([duplicate, corrected])
    prompts: list[str] = []

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            return "\n".join(row["content"] for row in messages)

    generator = MlxStructuredGenerator(
        model_path="fixture-model",
        max_tokens=256,
        max_structural_repairs=1,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda _model, _tokenizer, *, prompt, max_tokens: (
            prompts.append(prompt) or next(outputs)
        ),
    )

    raw = generator(_task(checkout), _revision())

    assert json.loads(raw)["replace"] == ("for item in right:\n    consume_fixed(item)")
    assert "duplicates structural lines" in prompts[1]
    assert "expand the search span" in prompts[1]
    assert duplicate not in prompts[1]
    assert "Do not copy any prior response" in prompts[1]
    stage_results = generator.generation_trace_results()
    assert [row["status"] for row in stage_results] == [
        "structural-rejected",
        "structural-valid",
    ]
    assert "duplicates structural lines" in stage_results[0]["detail"]


def test_mlx_structured_generator_repairs_a_semantically_incomplete_edit(
    tmp_path: Path,
) -> None:
    checkout = _git_checkout(
        tmp_path / "repo", {"src/example.py": "def answer():\n    return 1\n"}
    )
    incomplete = json.dumps(
        {
            "file": "src/example.py",
            "search": "return 1",
            "replace": "return 2",
            "diagnostic": "changes only one required path",
        }
    )
    critique = json.dumps(
        {
            "complete": False,
            "missing_clauses": ["ordinary path must remain explicit"],
            "violated_invariants": [],
            "evidence": ["replacement covers only the edge value"],
        }
    )
    corrected = json.dumps(
        {
            "file": "src/example.py",
            "search": "def answer():\n    return 1",
            "replace": "def answer():\n    return 2",
            "diagnostic": "covers the full symbol while preserving its signature",
        }
    )
    accepted = json.dumps(
        {
            "complete": True,
            "missing_clauses": [],
            "violated_invariants": [],
            "evidence": ["answer and return are both covered"],
        }
    )
    outputs = iter([incomplete, critique, corrected, accepted])
    prompts: list[str] = []
    token_budgets: list[int] = []

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            return "\n".join(row["content"] for row in messages)

    def generate(_model, _tokenizer, *, prompt, max_tokens):
        prompts.append(prompt)
        token_budgets.append(max_tokens)
        return next(outputs)

    generator = MlxStructuredGenerator(
        model_path="fixture-model",
        max_tokens=1024,
        max_structural_repairs=1,
        use_semantic_critic=True,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=generate,
    )

    raw = generator(_task(checkout), _revision())

    assert json.loads(raw)["search"].startswith("def answer")
    assert generator.generation_trace_kinds() == (
        "edit-attempt-0",
        "semantic-critic-0",
        "edit-attempt-1",
        "semantic-critic-1",
    )
    assert "ordinary path must remain explicit" in prompts[2]
    assert incomplete not in prompts[2]
    assert "Do not copy any prior response" in prompts[2]
    assert token_budgets == [1024, 768, 1024, 768]
    assert generator.generation_trace_results() == (
        {"status": "structural-valid"},
        {
            "status": "semantic-rejected",
            "missing_clause_count": 1,
            "violated_invariant_count": 0,
        },
        {"status": "structural-valid"},
        {"status": "semantic-accepted"},
    )
    assert generator.generation_config()["use_semantic_critic"] is True


def test_mlx_structured_generator_classifies_a_truncated_semantic_critic(
    tmp_path: Path,
) -> None:
    checkout = _git_checkout(
        tmp_path / "repo", {"src/example.py": "def answer():\n    return 1\n"}
    )
    edit = json.dumps(
        {
            "file": "src/example.py",
            "search": "return 1",
            "replace": "return 2",
            "diagnostic": "bounded fixture edit",
        }
    )
    outputs = iter([edit, '{"complete": false, "missing_clauses": ['])

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            return "\n".join(row["content"] for row in messages)

    generator = MlxStructuredGenerator(
        model_path="fixture-model",
        max_tokens=128,
        max_structural_repairs=0,
        use_semantic_critic=True,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda _model, _tokenizer, *, prompt, max_tokens: next(outputs),
    )

    raw = generator(_task(checkout), _revision())

    assert raw.startswith("SEMANTIC_PREFLIGHT_REJECTED:")
    assert generator.generation_trace_results() == (
        {"status": "structural-valid"},
        {
            "status": "contract-rejected",
            "detail": "semantic critic returned no JSON object",
        },
    )


def test_mlx_structured_generator_retrieves_issue_relevant_tail_within_budget(
    tmp_path: Path,
) -> None:
    content = (
        "import os\n"
        + ("irrelevant_value = 1\n" * 1000)
        + "def visit_literal(node):\n"
        + "    return r'\\\\sphinxcode{\\\\sphinxupquote{'\n"
        + ("trailing_value = 2\n" * 1000)
    )
    checkout = _git_checkout(tmp_path / "repo", {"latex.py": content})
    prompts: list[str] = []

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            return "\n".join(row["content"] for row in messages)

    def generate(_model, _tokenizer, *, prompt: str, max_tokens: int):
        prompts.append(prompt)
        return "{}"

    generator = MlxStructuredGenerator(
        model_path="fixture-model",
        max_tokens=100,
        max_context_chars=6_000,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=generate,
    )
    task = StudentTask(
        task_id="tail-context",
        instruction="LaTeX inline sphinxcode sphinxupquote adds whitespace",
        checkout=checkout,
        allowed_targets=("latex.py",),
        cohort="feedback",
    )

    generator(task, _revision())

    assert "def visit_literal" in prompts[0]
    assert "sphinxupquote" in prompts[0]
    assert generator.generation_config()["context_selector"] == (
        "skill-ranked-python-symbol-v14-complete-critic-clean-repair"
    )


def test_pattern_card_router_is_deterministic_and_abstains_without_task_overlap() -> (
    None
):
    skill = """## Pattern cards
1. Symptom: a default disappears when positional arguments share one defaults vector. Transformation: align one defaults vector across both argument loops. Validation: cover each category.
2. Symptom: physical newlines become visible spaces inside an inline TeX wrapper. Transformation: put percent sentinels at both wrapper boundaries. Validation: block output stays unchanged.
3. Symptom: a dynamically generated subclass displays a truncated base name. Transformation: preserve the generated leaf name. Validation: nested generated chains retain the leaf class name.

## Commit gate
Copy an exact unique search span. Make one minimal replacement.
"""

    positional = _select_pattern_card(
        skill, "The default value for positional only argument has vanished"
    )
    latex = _select_pattern_card(
        skill, "Inline TeX code highlighting adds whitespace at both ends"
    )
    unrelated = _select_pattern_card(
        skill, "Callable FileField storage is omitted during deconstruction"
    )
    same_name_variable = _select_pattern_card(
        skill, "A class variable links to another variable of the same name"
    )

    assert positional is not None
    assert positional.startswith("1. Symptom:")
    assert latex is not None
    assert latex.startswith("2. Symptom:")
    assert unrelated is None
    assert same_name_variable is None


def test_mlx_structured_generator_repeats_selected_card_at_output_boundary(
    tmp_path: Path,
) -> None:
    checkout = _git_checkout(
        tmp_path / "repo",
        {
            "src/example.py": (
                "def render_inline():\n"
                "    wrapper = 'inline TeX wrapper'\n"
                "    return wrapper\n"
            )
        },
    )
    prompts: list[str] = []

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            return "\n".join(row["content"] for row in messages)

    generator = MlxStructuredGenerator(
        model_path="fixture-model",
        max_tokens=100,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda _model, _tokenizer, *, prompt, max_tokens: (
            prompts.append(prompt) or "{}"
        ),
    )
    revision = LoopRevision.create(
        skill_id="skill",
        revision_id="rev-patterns",
        parent_revision_id=None,
        source_round=5,
        protocol="structured-search-replace-v1",
        skill_text=(
            "## Pattern cards\n"
            "1. Symptom: physical newlines become visible spaces inside an inline "
            "TeX wrapper. Transformation: add percent sentinels at both wrapper "
            "boundaries. Validation: block output stays unchanged.\n\n"
            "## Commit gate\n"
            "Copy an exact unique search span. Make one minimal replacement."
        ),
        prompt_template="Return exactly one JSON object.",
        eval_note="fixture",
    )
    task = StudentTask.create(
        task_id="latex",
        instruction="Inline TeX newlines add visible spaces at wrapper boundaries",
        checkout=checkout,
        allowed_targets=["src/example.py"],
        cohort="feedback",
    )

    generator(task, revision)

    reminder = prompts[0].rindex("Selected teaching card:")
    output_boundary = prompts[0].rindex("Return the JSON edit object now.")
    assert reminder < output_boundary
    assert "percent sentinels" in prompts[0][reminder:output_boundary]
    assert "Copy an exact unique search span" in prompts[0][reminder:output_boundary]
    assert (
        "implement every transformation clause" in prompts[0][reminder:output_boundary]
    )
    assert (
        "never edit a path the card says must stay unchanged"
        in (prompts[0][reminder:output_boundary])
    )
    assert "one contiguous exact search span" in prompts[0][reminder:output_boundary]
    assert "descriptive no-op is invalid" in prompts[0][reminder:output_boundary]


def test_skill_aware_excerpt_normalizes_issue_and_source_word_forms(
    tmp_path: Path,
) -> None:
    content = (
        "module_header = True\n"
        + ("head_value = 0\n" * 100)
        + (
            "def visit_desc_inline(node):\n"
            "    return 'inline LaTeX wrapper start end whitespace'\n"
            + ("unrelated_value = 1\n" * 120)
        )
        * 3
        + "def visit_literal(node):\n"
        + "    # LaTeX code highlighting path for sphinxcode and sphinxupquote\n"
        + "    hlcode = self.highlighter.highlight_block(node.astext())\n"
        + "    hlcode = hlcode.rstrip()[:-14]\n"
        + "    return hlcode\n"
        + "def tail():\n"
        + "    return 2\n"
        + ("tail_value = 2\n" * 900)
    )
    checkout = _git_checkout(tmp_path / "repo", {"latex.py": content})
    prompts: list[str] = []

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            return "\n".join(row["content"] for row in messages)

    generator = MlxStructuredGenerator(
        model_path="fixture-model",
        max_tokens=100,
        max_context_chars=2_000,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda _model, _tokenizer, *, prompt, max_tokens: (
            prompts.append(prompt) or "{}"
        ),
    )
    revision = LoopRevision.create(
        skill_id="skill",
        revision_id="rev-highlight",
        parent_revision_id=None,
        source_round=5,
        protocol="structured-search-replace-v1",
        skill_text=(
            "## Pattern cards\n"
            "1. Symptom: physical newlines become visible spaces inside an inline "
            "TeX wrapper. Transformation: trim the complete newline-plus-trailer "
            "sequence and put percent sentinels at both wrapper boundaries."
        ),
        prompt_template="Return exactly one JSON object.",
        eval_note="fixture",
    )
    task = StudentTask.create(
        task_id="highlight",
        instruction=(
            "Inline LaTeX code highlighting adds whitespace at both ends\n"
            "```tex\n\\sphinxcode{\\sphinxupquote{%\n...\n}}\n```"
        ),
        checkout=checkout,
        allowed_targets=["latex.py"],
        cohort="feedback",
    )

    generator(task, revision)

    assert "hlcode = self.highlighter.highlight_block" in prompts[0]
    assert "hlcode = hlcode.rstrip()[:-14]" in prompts[0]


def test_hunk_student_adapter_accepts_only_an_allowed_applicable_diff(
    tmp_path: Path,
) -> None:
    checkout = _git_checkout(
        tmp_path / "repo", {"src/example.py": "def answer():\n    return 1\n"}
    )
    responses = iter(
        [
            "Reasoning without a patch.",
            "--- a/src/example.py\n"
            "+++ b/src/example.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def answer():\n"
            "-    return 1\n"
            "+    return 2\n",
        ]
    )
    adapter = HunkStudentAdapter(generator=lambda _task, _revision: next(responses))

    rejected = adapter.run(_task(checkout), _revision())
    accepted = adapter.run(_task(checkout), _revision())

    assert rejected.failure_reason == "reasoning-only"
    assert accepted.structural_valid is True
    assert accepted.target_file == "src/example.py"
    assert (checkout / "src/example.py").read_text() == "def answer():\n    return 1\n"
    adapter.apply(accepted)
    assert (checkout / "src/example.py").read_text() == "def answer():\n    return 2\n"


def test_hunk_adapter_preserves_an_extracted_apply_failure_as_negative_evidence(
    tmp_path: Path,
) -> None:
    checkout = _git_checkout(
        tmp_path / "repo", {"src/example.py": "def answer():\n    return 1\n"}
    )
    raw = (
        "--- a/src/example.py\n"
        "+++ b/src/example.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def invented():\n"
        "-    return 10\n"
        "+    return 20\n"
    )
    adapter = HunkStudentAdapter(generator=lambda _task, _revision: raw)

    attempt = adapter.run(_task(checkout), _revision())

    assert attempt.failure_reason == "apply-fail"
    assert attempt.patch == raw
    assert attempt.patch_sha256 is not None
    assert attempt.target_file == "src/example.py"


def test_mlx_hunk_generator_reuses_model_and_injects_the_same_revision_skill(
    tmp_path: Path,
) -> None:
    checkout = _git_checkout(
        tmp_path / "repo", {"src/example.py": "def answer():\n    return 1\n"}
    )
    loads: list[str] = []
    prompts: list[str] = []

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            assert add_generation_prompt is True
            assert enable_thinking is False
            return "\n".join(row["content"] for row in messages)

    def load(model_path: str):
        loads.append(model_path)
        return object(), Tokenizer()

    def generate(_model, _tokenizer, *, prompt: str, max_tokens: int):
        prompts.append(prompt)
        assert max_tokens == 120
        return "--- a/src/example.py\n+++ b/src/example.py\n"

    generator = MlxHunkGenerator(
        model_path="fixture-model",
        max_tokens=120,
        loader=load,
        text_generator=generate,
    )
    task = _task(checkout)
    generator(task, _revision())
    generator(task, _revision())

    assert loads == ["fixture-model"]
    assert "Return one bounded edit." in prompts[0]
    assert "src/example.py" in prompts[0]
    assert "return 1" in prompts[0]
    assert generator.generation_trace_kinds() == ("hunk-attempt-0",)
    assert generator.generation_prompt_trace() == (prompts[-1],)
