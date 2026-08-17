from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from pathlib import Path

from skill_evolution_loop import LoopRevision, StudentTask
from skill_evolution_loop.capabilities import StudentCapabilityProfile
from skill_evolution_loop.span_student import (
    MlxSpanPlanGenerator,
    SpanPlanAdapter,
    _canonicalize_operation_only_output,
    _canonicalize_semantic_recipe_output,
    _canonicalize_span_bundle_output,
    _editable_span_task,
    _frozen_causal_candidates,
    build_span_conditions,
    fixed_causal_state_contexts,
    fixed_editable_candidate_contexts,
    fixed_exact_span_candidates,
    fixed_repository_api_contexts,
    fixed_repository_state_value_contexts,
    fixed_supporting_symbol_contexts,
    fixed_typed_actions,
    fixed_typed_collection_actions,
    fixed_typed_endpoint_actions,
    fixed_typed_state_actions,
)


def _checkout(path: Path) -> Path:
    path.mkdir()
    source = path / "src/example.ts"
    source.parent.mkdir()
    source.write_text(
        "export function answer(value: number) {\n  return value + 1;\n}\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=path,
        check=True,
    )
    return path


def _task(checkout: Path) -> StudentTask:
    return StudentTask.create(
        task_id="span-fixture",
        checkout=checkout,
        instruction="The answer function should add two instead of one.",
        allowed_targets=["src/example.ts"],
        cohort="feedback",
    )


def _revision(taught: bool, *, source_round: int = 8) -> LoopRevision:
    return LoopRevision.create(
        skill_id="span-fixture-skill",
        revision_id="span-taught" if taught else "span-baseline",
        parent_revision_id=None,
        source_round=source_round,
        protocol="multilanguage-exact-span-plan-v1",
        skill_text=(
            "Change the smallest exact source span implementing the boundary."
            if taught
            else "No additional domain teaching is provided."
        ),
        prompt_template="Return exactly one exact-span plan JSON object.",
        eval_note="fixture",
    )


def _raw_plan() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "file": "src/example.ts",
            "intent": {
                "defect": "answer adds one",
                "trigger": "any value",
                "desired_boundary": "answer adds two",
            },
            "operations": [
                {"before": "return value + 1;", "after": "return value + 2;"}
            ],
            "diagnostic": "replace the offset constant",
        }
    )


def _raw_operation_only_plan() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "edits": [
                {
                    "file": "src/example.ts",
                    "before": "return value + 1;",
                    "after": "return value + 2;",
                }
            ],
        }
    )


def test_span_generator_does_not_retry_invalid_semantic_recipe(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    calls = 0
    invalid = json.dumps(
        {"schema_version": 1, "recipes": [{"candidate_id": "missing"}]}
    )

    class Tokenizer:
        @staticmethod
        def apply_chat_template(*_args, **_kwargs):
            return "prompt"

    def generate(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return invalid if calls == 1 else _raw_plan()

    generator = MlxSpanPlanGenerator(
        model_path="fixture-model",
        max_plan_repairs=1,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=generate,
    )

    generator(_task(checkout), _revision(True, source_round=70))

    assert calls == 1
    assert generator.generation_trace_results() == (
        {
            "status": "structural-rejected",
            "detail": "semantic recipe fields or candidate_id are invalid",
        },
    )


def test_span_bundle_canonicalizer_fills_unambiguous_file_plan_schema() -> None:
    data = {
        "schema_version": 1,
        "plans": [
            {
                "file": "src/example.ts",
                "intent": {
                    "defect": "answer adds one",
                    "trigger": "any value",
                    "desired_boundary": "answer adds two",
                },
                "operations": [
                    {"before": "return value + 1;", "after": "return value + 2;"}
                ],
            }
        ],
        "diagnostic": "replace the offset constant",
    }

    normalized, changed = _canonicalize_span_bundle_output(json.dumps(data))

    assert changed is True
    assert json.loads(normalized)["plans"][0]["schema_version"] == 1


def test_span_adapter_materializes_and_records_receipt(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "repo")
    adapter = SpanPlanAdapter(generator=lambda _task, _revision: _raw_plan())

    config = adapter.experiment_config()

    attempt = adapter.run(_task(checkout), _revision(True))

    assert (
        config["implementation_sha256"]
        == hashlib.sha256(
            Path(inspect.getfile(SpanPlanAdapter)).read_bytes()
        ).hexdigest()
    )
    assert attempt.structural_valid is True
    assert "+  return value + 2;" in attempt.patch
    receipt = json.loads(attempt.detail)
    assert receipt["receipt_type"] == "span-bundle-realization-v2"
    file_receipt = receipt["materialization"]["files"]["src/example.ts"]
    assert all(file_receipt["gates"].values())


def test_span_adapter_merges_framework_intent_into_operation_only_output(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    adapter = SpanPlanAdapter(
        generator=lambda _task, _revision: _raw_operation_only_plan()
    )

    attempt = adapter.run(_task(checkout), _revision(True))

    assert attempt.structural_valid is True
    assert "+  return value + 2;" in attempt.patch
    normalized = json.loads(attempt.raw_output)
    assert normalized["plans"][0]["intent"]["defect"]
    assert normalized["plans"][0]["operations"][0]["after"] == ("return value + 2;")


def test_exact_span_candidates_rank_executable_boundary_before_header(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")

    candidates = fixed_exact_span_candidates(_task(checkout))

    assert candidates[0].before == "return value + 1;"


def test_exact_span_candidates_prioritize_issue_compound_identifier(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "src/example.ts"
    source.write_text(
        "function required_usage() {\n"
        "  const required = compute_required_arguments()\n"
        "  return required\n"
        "}\n\n"
        "function flattened_usage() {\n"
        "  if (cmd.is_flatten_help_set()) {\n"
        "    return cmd.visible_subcommands()\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="compound-identifier-localization-fixture",
        checkout=checkout,
        instruction="flatten_help must hide hidden subcommands in flattened usage",
        allowed_targets=["src/example.ts"],
        cohort="feedback",
    )

    candidates = fixed_exact_span_candidates(task, max_candidates=8)

    assert candidates[0].before == (
        "if (cmd.is_flatten_help_set()) {\n    return cmd.visible_subcommands()"
    )


def test_exact_span_candidates_keep_verbatim_issue_before_inside_small_budget(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "src/Iterator.java"
    source.write_text(
        "public final class Iterator {\n"
        "  private CharacterIterator text;\n"
        "  private int currentSentence;\n"
        "  private int[] sentenceStarts;\n"
        "  public Iterator(Detector detector) { this.detector = detector; }\n"
        "  public int current() { return text.getIndex(); }\n"
        "  public int preceding(int pos) {\n"
        "    if (pos < text.getBeginIndex() || pos > text.getEndIndex()) {\n"
        "      throw new IllegalArgumentException();\n"
        "    } else {\n"
        "      currentSentence = sentenceStarts.length / 2; // start search from the middle\n"
        "      moveToSentenceAt(pos, 0, sentenceStarts.length - 1);\n"
        "      return current();\n"
        "    }\n"
        "  }\n"
        "  public void setText(CharacterIterator newText) {\n"
        "    sentenceStarts = new int[2];\n"
        "    for (int i = 0; i < sentenceStarts.length; ++i) {\n"
        "      sentenceStarts[i] = i + text.getBeginIndex();\n"
        "    }\n"
        "  }\n"
        "  private String characterIteratorToString() {\n"
        "    if (text instanceof ArrayIterator) {\n"
        "      return text.toString();\n"
        "    }\n"
        '    return "";\n'
        "  }\n"
        "  void moveToSentenceAt(int pos, int min, int max) {}\n"
        "}\n",
        encoding="utf-8",
    )
    dummy = checkout / "src/Dummy.java"
    dummy.write_text("class Dummy {}\n", encoding="utf-8")
    task = StudentTask.create(
        task_id="verbatim-issue-localization-fixture",
        checkout=checkout,
        instruction=(
            "Critical CharacterIterator parser return boundary regression in "
            "characterIteratorToString and current. The exact broken statement is:\n"
            "```java\n"
            "currentSentence = sentenceStarts.length / 2; // start search from the middle\n"
            "```\n"
            "It should be instead like following:\n"
            "```java\n"
            "currentSentence = (sentenceStarts.length - 1) / 2; // start search from the middle\n"
            "```\n"
            "Otherwise moveToSentenceAt can access beyond the array."
        ),
        allowed_targets=["src/Iterator.java", "src/Dummy.java"],
        cohort="feedback",
    )

    candidates = fixed_exact_span_candidates(task, max_candidates=1)

    assert any(
        candidate.before
        == "currentSentence = sentenceStarts.length / 2; // start search from the middle"
        for candidate in candidates
    )


def test_public_issue_before_after_pair_becomes_renderer_owned_typed_action(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "src/example.ts"
    source.write_text(
        "export function middle(length: number) {\n  return length / 2;\n}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="public-verbatim-recipe-fixture",
        checkout=checkout,
        instruction=(
            "The current implementation is:\n"
            "```ts\nreturn length / 2;\n```\n"
            "It should preserve an in-range index:\n"
            "```ts\nreturn (length - 1) / 2;\n```"
        ),
        allowed_targets=["src/example.ts"],
        cohort="feedback",
    )
    revision = _revision(True, source_round=70)
    candidates = fixed_exact_span_candidates(task, max_candidates=4)

    actions = fixed_typed_actions(task, candidates)

    assert actions == (
        {
            "candidate_id": candidates[0].candidate_id,
            "operation": "apply-issue-verbatim-replacement",
        },
    )
    raw = json.dumps({"schema_version": 1, "actions": [actions[0]]})
    normalized, changed = _canonicalize_operation_only_output(raw, task, revision)
    parsed = json.loads(normalized)
    assert changed is True
    assert parsed["plans"][0]["operations"] == [
        {
            "before": "return length / 2;",
            "after": "return (length - 1) / 2;",
        }
    ]


def test_locale_separator_roles_become_renderer_owned_typed_action(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "src/Formatter.php"
    source.write_text(
        "<?php\n"
        "function adjustSeparators(string $value): string {\n"
        "  $thousandsSeparator = Locale::getThousandsSeparator();\n"
        "  $decimalSeparator = Locale::getDecimalSeparator();\n"
        "  if ($thousandsSeparator !== ',' || $decimalSeparator !== '.') {\n"
        "    $value = str_replace(['.', ',', \"\\u{fffd}\"], "
        "[\"\\u{fffd}\", '.', ','], $value);\n"
        "  }\n"
        "  return $value;\n"
        "}\n",
        encoding="utf-8",
    )
    (checkout / "src/Other.php").write_text("<?php\n", encoding="utf-8")
    task = StudentTask.create(
        task_id="locale-role-substitution-fixture",
        checkout=checkout,
        instruction=(
            "The decimal separator and thousands separator are ignored because "
            "the replacement values are hard coded."
        ),
        allowed_targets=["src/Formatter.php", "src/Other.php"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="span-fixture-skill",
        revision_id="span-taught-r074",
        parent_revision_id=None,
        source_round=74,
        protocol="multilanguage-exact-span-plan-v1",
        skill_text=(
            "Use only framework-owned typed actions.\n\n"
            "## Shared diagnosis and localization (read-only)\n"
            + json.dumps(
                {
                    "diagnosis": {
                        "defect": "locale separators are ignored",
                        "trigger": "non-default separators",
                        "desired_boundary": "preserve locale separator roles",
                    },
                    "target_files": ["src/Formatter.php"],
                }
            )
        ),
        prompt_template="Return one typed action.",
        eval_note="fixture",
    )
    candidates = _frozen_causal_candidates(task, revision, max_candidates=8)

    actions = fixed_typed_actions(task, candidates)
    locale = next(
        row
        for row in actions
        if row["operation"] == "substitute-locale-separator-roles"
    )
    raw = json.dumps({"schema_version": 1, "actions": [locale]})
    normalized, changed = _canonicalize_operation_only_output(raw, task, revision)
    parsed = json.loads(normalized)

    assert changed is True
    assert parsed["plans"][0]["operations"] == [
        {
            "before": (
                "$value = str_replace(['.', ',', \"\\u{fffd}\"], "
                "[\"\\u{fffd}\", '.', ','], $value);"
            ),
            "after": (
                "$value = str_replace(['.', ',', \"\\u{fffd}\"], "
                '["\\u{fffd}", $thousandsSeparator, $decimalSeparator], $value);'
            ),
        }
    ]

    source.write_text(
        "<?php\n"
        "function unrelatedLocaleGuard(string $value): string {\n"
        "  $thousandsSeparator = Locale::getThousandsSeparator();\n"
        "  $decimalSeparator = Locale::getDecimalSeparator();\n"
        "  if ($thousandsSeparator !== ',' || $decimalSeparator !== '.') {\n"
        "    return $value;\n"
        "  }\n"
        "}\n"
        "function adjustSeparators(string $value): string {\n"
        "  $value = str_replace(['.', ',', \"\\u{fffd}\"], "
        "[\"\\u{fffd}\", '.', ','], $value);\n"
        "  return $value;\n"
        "}\n",
        encoding="utf-8",
    )
    unrelated_candidates = _frozen_causal_candidates(task, revision, max_candidates=8)

    assert all(
        row["operation"] != "substitute-locale-separator-roles"
        for row in fixed_typed_actions(task, unrelated_candidates)
    )


def test_semantic_recipe_maps_candidate_id_to_renderer_owned_selector(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    task = _task(checkout)
    revision = _revision(True, source_round=70)
    candidates = fixed_exact_span_candidates(task, max_candidates=4)
    raw = json.dumps(
        {
            "schema_version": 1,
            "recipes": [
                {
                    "candidate_id": candidates[0].candidate_id,
                    "after": "return value + 2;",
                }
            ],
        }
    )

    projected, changed = _canonicalize_semantic_recipe_output(raw, candidates)
    normalized, operation_changed = _canonicalize_operation_only_output(
        projected, task, revision
    )
    parsed = json.loads(normalized)

    assert changed is True
    assert operation_changed is True
    assert parsed["plans"][0]["operations"] == [
        {"before": "return value + 1;", "after": "return value + 2;"}
    ]


def test_semantic_recipe_rejects_unknown_or_duplicate_candidate_ids(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    candidates = fixed_exact_span_candidates(_task(checkout), max_candidates=4)
    unknown = json.dumps(
        {
            "schema_version": 1,
            "recipes": [{"candidate_id": "span-999", "after": "return 2;"}],
        }
    )
    duplicate = json.dumps(
        {
            "schema_version": 1,
            "recipes": [
                {
                    "candidate_id": candidates[0].candidate_id,
                    "after": "return value + 2;",
                },
                {
                    "candidate_id": candidates[0].candidate_id,
                    "after": "return value + 3;",
                },
            ],
        }
    )

    assert _canonicalize_semantic_recipe_output(unknown, candidates) == (
        unknown,
        False,
    )
    assert _canonicalize_semantic_recipe_output(duplicate, candidates) == (
        duplicate,
        False,
    )


def test_exact_span_candidates_use_php_method_scope_for_issue_api(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "src/Period.php"
    source.write_text(
        "<?php\n"
        "class Period {\n"
        "  public function unrelated() {\n"
        "    return $this->timezone;\n"
        "  }\n"
        "  public function first() {\n"
        "    return $this->rewind()->valid() ? $this->current() : null;\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="php-method-scope-fixture",
        checkout=checkout,
        instruction="Calling ->first()->timezone must preserve the period timezone.",
        allowed_targets=["src/Period.php"],
        cohort="feedback",
    )

    candidates = fixed_exact_span_candidates(task, max_candidates=8)

    assert candidates[0].before == (
        "return $this->rewind()->valid() ? $this->current() : null;"
    )


def test_exact_span_candidates_track_go_function_scope_for_issue_terms(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "command/pr.go"
    source.parent.mkdir(parents=True)
    source.write_text(
        "package command\n\n"
        'var prCmd = &cobra.Command{Use: "pr", Short: "PR status commands"}\n'
        'var prListCmd = &cobra.Command{Use: "list", Short: "List PR status"}\n\n'
        "func unrelated() error {\n"
        "\treturn renderStatus()\n"
        "}\n\n"
        "func prStatus(ctx context.Context) error {\n"
        "\tprNumber, prHeadRef, err := prSelectorForCurrentBranch(ctx)\n"
        "\tif err != nil {\n"
        "\t\treturn err\n"
        "\t}\n"
        "\treturn renderPR(prNumber, prHeadRef)\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="go-function-scope-fixture",
        checkout=checkout,
        instruction=(
            "Fix PR status with -R flag to work outside git repo. gh pr status "
            "should ignore the current branch selector and relevant branch PRs "
            "outside a repository directory."
        ),
        allowed_targets=["command/pr.go", "src/example.ts"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="go-function-scope",
        revision_id="go-function-scope-r071",
        parent_revision_id=None,
        source_round=71,
        protocol="multilanguage-typed-state-action-v13",
        skill_text=(
            "Prefer call sites on the backward slice.\n\n"
            "## Shared diagnosis and localization (read-only)\n"
            '{"target_files":["command/pr.go"]}'
        ),
        prompt_template="Return recipes.",
        eval_note="fixture",
    )

    candidates = _frozen_causal_candidates(task, revision, max_candidates=4)

    assert any(
        "prSelectorForCurrentBranch(ctx)" in candidate.before
        for candidate in candidates
    )


def test_semantic_recipe_rejects_php_variable_outside_enclosing_method(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "src/Blueprint.php"
    source.write_text(
        "<?php\n"
        "class Blueprint {\n"
        "  public function unsignedBigInteger($column, $autoIncrement = false) {\n"
        "    return $this->bigInteger($column, $autoIncrement, true);\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="php-free-variable-fixture",
        checkout=checkout,
        instruction="unsignedBigInteger must preserve its incrementing boundary.",
        allowed_targets=["src/Blueprint.php"],
        cohort="feedback",
    )
    candidates = fixed_exact_span_candidates(task, max_candidates=8)
    selected = next(
        candidate
        for candidate in candidates
        if candidate.before.startswith("return $this->bigInteger")
    )
    raw = json.dumps(
        {
            "schema_version": 1,
            "recipes": [
                {
                    "candidate_id": selected.candidate_id,
                    "after": (
                        "return $this->bigInteger($column, $autoIncrement, "
                        "$model->getIncrementing());"
                    ),
                }
            ],
        }
    )

    assert _canonicalize_semantic_recipe_output(raw, candidates, task=task) == (
        raw,
        False,
    )


def test_frozen_candidates_prefer_explicit_php_api_guard_over_related_helper(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "src/Blueprint.php"
    source.write_text(
        "<?php\n"
        "class Blueprint {\n"
        "  public function dropConstrainedForeignIdFor($model, $column = null) {\n"
        "    return $this->dropConstrainedForeignId($column ?: "
        "$model->getForeignKey());\n"
        "  }\n"
        "  public function unsignedBigInteger($column, $autoIncrement = false) {\n"
        "    return $this->bigInteger($column, $autoIncrement, true);\n"
        "  }\n"
        "  public function foreignIdFor($model, $column = null) {\n"
        "    if ($model->getKeyType() === 'int' && $model->getIncrementing()) {\n"
        "      return $this->foreignId($column);\n"
        "    }\n"
        "    return $this->foreignUuid($column);\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="php-explicit-api-guard-fixture",
        checkout=checkout,
        instruction=(
            "foreignIdFor fails with a non-incrementing bigint key. "
            "$table->foreignIdFor(MyTable::class)->constrained();"
        ),
        allowed_targets=["src/Blueprint.php", "src/example.ts"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="php-explicit-api-guard",
        revision_id="php-explicit-api-guard-r071",
        parent_revision_id=None,
        source_round=71,
        protocol="multilanguage-typed-state-action-v13",
        skill_text=(
            "Prefer the named API guard.\n\n"
            "## Shared diagnosis and localization (read-only)\n"
            '{"target_files":["src/Blueprint.php"]}'
        ),
        prompt_template="Return recipes.",
        eval_note="fixture",
    )

    candidates = _frozen_causal_candidates(task, revision, max_candidates=4)

    assert "if ($model->getKeyType()" in candidates[0].before


def test_frozen_candidates_keep_named_api_companion_method_family(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "src/Str.php"
    source.write_text(
        "<?php\n"
        "class Str {\n"
        "  public static function unrelated($value) {\n"
        "    try { return transform($value); } catch (Throwable $e) { return ''; }\n"
        "  }\n"
        "  public static function trim($value) {\n"
        "    return preg_replace('~^[x]+|[x]+$~', '', $value);\n"
        "  }\n"
        "  public static function ltrim($value) {\n"
        "    return preg_replace('~^[x]+~', '', $value);\n"
        "  }\n"
        "  public static function rtrim($value) {\n"
        "    return preg_replace('~[x]+$~', '', $value);\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="php-api-family-fixture",
        checkout=checkout,
        instruction="Str::trim() must remove the same boundary character.",
        allowed_targets=["src/Str.php", "src/example.ts"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="php-api-family",
        revision_id="php-api-family-r072",
        parent_revision_id=None,
        source_round=72,
        protocol="multilanguage-typed-state-action-v13",
        skill_text=(
            "Preserve companion method boundaries.\n\n"
            "## Shared diagnosis and localization (read-only)\n"
            '{"target_files":["src/Str.php"]}'
        ),
        prompt_template="Return recipes.",
        eval_note="fixture",
    )

    candidates = _frozen_causal_candidates(task, revision, max_candidates=4)

    editable = "\n".join(candidate.before for candidate in candidates)
    assert "~^[x]+|[x]+$~" in editable
    assert "~^[x]+~" in editable
    assert "~[x]+$~" in editable


def test_semantic_recipe_accepts_three_companion_edits(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "src/Str.php"
    source.write_text(
        "<?php\n"
        "function trim_value($value) { return trim($value); }\n"
        "function ltrim_value($value) { return ltrim($value); }\n"
        "function rtrim_value($value) { return rtrim($value); }\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="three-recipe-fixture",
        checkout=checkout,
        instruction="Keep trim_value, ltrim_value, and rtrim_value symmetric.",
        allowed_targets=["src/Str.php"],
        cohort="feedback",
    )
    candidates = fixed_exact_span_candidates(task, max_candidates=32)
    selected = [
        candidate
        for candidate in candidates
        if candidate.before.strip().startswith("function ")
    ]
    assert len(selected) >= 3
    raw = json.dumps(
        {
            "schema_version": 1,
            "recipes": [
                {
                    "candidate_id": candidate.candidate_id,
                    "after": candidate.before.replace(
                        "($value);", '($value, "\\x00");'
                    ),
                }
                for candidate in selected[:3]
            ],
        }
    )

    normalized, changed = _canonicalize_semantic_recipe_output(
        raw, candidates, task=task
    )

    assert changed is True
    assert len(json.loads(normalized)["edits"]) == 3


def test_frozen_localization_emits_role_diverse_control_flow_candidates(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    main = checkout / "lib/main.rb"
    main.parent.mkdir(parents=True)
    main.write_text(
        "def process(result, node)\n"
        "  return if result.nil?\n"
        "  case result\n"
        "  when :failed\n"
        "    node.disable!\n"
        "    rollback(result)\n"
        "  end\n"
        "end\n",
        encoding="utf-8",
    )
    reader = checkout / "lib/reader.rb"
    reader.write_text(
        "def read(socket)\n"
        "  data = IO.select([socket], nil, nil, 1)\n"
        "  return data\n"
        "rescue IOError\n"
        "  return nil\n"
        "end\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="role-diverse-public-source-fixture",
        checkout=checkout,
        instruction=(
            "Shutdown can raise IOError while reading; guard nil results and disable "
            "the failed node safely."
        ),
        allowed_targets=["lib/main.rb", "lib/reader.rb", "src/example.ts"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="role-diverse-public-source",
        revision_id="role-diverse-public-source-r064",
        parent_revision_id=None,
        source_round=64,
        protocol="multilanguage-typed-edit-action-v17",
        skill_text=(
            "Use the smallest supported control-flow boundary.\n\n"
            "## Shared diagnosis and localization (read-only)\n"
            '{"target_files":["lib/main.rb","lib/reader.rb"]}'
        ),
        prompt_template="Return edits.",
        eval_note="fixture",
    )

    candidates = _frozen_causal_candidates(task, revision)
    prompted = [candidate.to_prompt_dict() for candidate in candidates]
    roles = {role for candidate in prompted for role in candidate["roles"]}

    assert len(candidates) <= 8
    assert {candidate.file for candidate in candidates} == {
        "lib/main.rb",
        "lib/reader.rb",
    }
    assert "exception-boundary" in roles
    assert "side-effect-boundary" in roles
    assert "guard-boundary" in roles
    assert "dataflow-boundary" in roles
    assert any("node.disable!" in candidate.before for candidate in candidates)
    assert any(
        "IO.select" in candidate.before or "rescue IOError" in candidate.before
        for candidate in candidates
    )
    assert all(candidate.before != "rescue IOError" for candidate in candidates)


def test_frozen_candidates_follow_cross_file_constructor_to_initializer_dataflow(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    caller = checkout / "lib/service.rb"
    caller.parent.mkdir(parents=True)
    caller.write_text(
        "class Service\n"
        "  def start(endpoint)\n"
        "    @server = RPC::Server.new(endpoint, @log)\n"
        "  end\n"
        "end\n",
        encoding="utf-8",
    )
    server = checkout / "lib/rpc.rb"
    server.write_text(
        "module RPC\n"
        "  class Server\n"
        "    def initialize(endpoint, log)\n"
        "      bind, port = endpoint.split(':')\n"
        "      @log = log\n"
        "    end\n"
        "    def start\n"
        "      @thread = Thread.new { run }\n"
        "    end\n"
        "  end\n"
        "end\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="constructor-edge-fixture",
        checkout=checkout,
        instruction=(
            "IPv6 rpc_endpoint parsing fails for [::]:24444.\n"
            "Error log repeatedly reports a server log failure."
        ),
        allowed_targets=["lib/service.rb", "lib/rpc.rb", "src/example.ts"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="constructor-edge",
        revision_id="constructor-edge-r067",
        parent_revision_id=None,
        source_round=67,
        protocol="multilanguage-typed-edit-action-v18",
        skill_text=(
            "Follow public repository call edges.\n\n"
            "## Shared diagnosis and localization (read-only)\n"
            '{"target_files":["lib/service.rb","lib/rpc.rb"]}'
        ),
        prompt_template="Return edits.",
        eval_note="fixture",
    )

    candidates = _frozen_causal_candidates(task, revision)

    assert any(
        candidate.file == "lib/rpc.rb"
        and candidate.before == "bind, port = endpoint.split(':')"
        for candidate in candidates
    )


def test_span_generator_keeps_user_prompt_fixed_between_arms(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    rendered: list[list[dict[str, str]]] = []

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            assert tokenize is False
            rendered.append(messages)
            return "\n".join(row["content"] for row in messages)

    generator = MlxSpanPlanGenerator(
        model_path="fixture-model",
        max_tokens=512,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda *_args, **_kwargs: _raw_plan(),
    )

    generator(_task(checkout), _revision(False))
    generator(_task(checkout), _revision(True))

    assert rendered[0][1] == rendered[1][1]
    assert rendered[0][0] != rendered[1][0]
    assert "return value + 1" in rendered[0][1]["content"]
    assert "READ-ONLY causal and repository evidence" in rendered[0][1]["content"]
    assert "ONLY EDITABLE exact-span candidates" in rendered[0][1]["content"]
    assert '"edits"' in rendered[0][0]["content"]
    assert rendered[0][1]["content"].index("READ-ONLY") < rendered[0][1][
        "content"
    ].index("ONLY EDITABLE")
    assert generator.generation_config()["action_space"] == (
        "multilanguage-typed-edit-action-v20"
    )
    assert generator.generation_config()["bundle_diagnostic_fallback"] is True


def test_r070_span_generator_uses_semantic_recipe_and_renderer_owned_selector(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    rendered: list[list[dict[str, str]]] = []

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            assert tokenize is False
            rendered.append(messages)
            return "\n".join(row["content"] for row in messages)

    raw = json.dumps(
        {
            "schema_version": 1,
            "recipes": [{"candidate_id": "span-001", "after": "return value + 2;"}],
        }
    )
    generator = MlxSpanPlanGenerator(
        model_path="fixture-model",
        max_tokens=512,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda *_args, **_kwargs: raw,
    )
    adapter = SpanPlanAdapter(generator=generator)

    attempt = adapter.run(_task(checkout), _revision(True, source_round=70))

    assert attempt.structural_valid is True
    assert "+  return value + 2;" in attempt.patch
    assert '"recipes"' in rendered[0][0]["content"]
    assert "Do not copy file or before" in rendered[0][0]["content"]
    assert "Active renderer contract overrides" in rendered[0][0]["content"]
    assert "Copy each selected file path" not in rendered[0][1]["content"]
    assert "Select one to three candidate_id values" in rendered[0][1]["content"]
    assert generator.generation_trace_results() == (
        {"status": "structural-valid", "detail": "accepted"},
    )


def test_span_generator_projects_first_three_compiled_rules(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "repo")
    rendered: list[list[dict[str, str]]] = []

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            rendered.append(messages)
            return "prompt"

    revision = LoopRevision.create(
        skill_id="projection-fixture",
        revision_id="projection-r034",
        parent_revision_id=None,
        source_round=34,
        protocol="multilanguage-operation-only-span-bundle-v10",
        skill_text=(
            "1. Detect state overwrite.\n"
            "2. Emit a conditional default guard.\n"
            "3. Do not return unresolved for the overwrite.\n"
            "4. Optional unrelated rule."
        ),
        prompt_template="Return edits.",
        eval_note="fixture",
    )
    generator = MlxSpanPlanGenerator(
        model_path="fixture-model",
        max_tokens=256,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda *_args, **_kwargs: json.dumps(
            {
                "schema_version": 1,
                "status": "unresolved",
                "diagnostic": "fixture",
            }
        ),
    )

    generator(_task(checkout), revision)

    system = rendered[0][0]["content"]
    assert "1. Detect state overwrite." in system
    assert "2. Emit a conditional default guard." in system
    assert "3. Do not return unresolved for the overwrite." in system


def test_span_generator_replans_false_unresolved_state_overwrite(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "include/fmt/printf.h"
    source.parent.mkdir(parents=True)
    source.write_text(
        "void parse_flags(format_specs& specs, char flag) {\n"
        "  if (flag == '-') specs.align = align::left;\n"
        "  if (flag == '0') specs.fill[0] = '0';\n"
        "}\n"
        "void format_char() {\n"
        "  fmt_specs.align = align::right;\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="fmt-state-replan-fixture",
        checkout=checkout,
        instruction="The minus flag for char must preserve left alignment.",
        allowed_targets=["include/fmt/printf.h"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="fmt-state-replan",
        revision_id="fmt-state-replan-r034",
        parent_revision_id=None,
        source_round=34,
        protocol="multilanguage-operation-only-span-bundle-v10",
        skill_text=(
            "1. Detect flag state overwrite.\n"
            "2. Emit a conditional default guard.\n"
            "3. Do not return unresolved for an overwrite."
        ),
        prompt_template="Return edits.",
        eval_note="fixture",
    )
    outputs = iter(
        [
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "unresolved",
                    "diagnostic": "no overwrite candidate",
                }
            ),
            json.dumps(
                {
                    "schema_version": 1,
                    "edits": [
                        {
                            "file": "include/fmt/printf.h",
                            "before": "fmt_specs.align = align::right;",
                            "after": (
                                "if (fmt_specs.align == align::none)\n"
                                "  fmt_specs.align = align::right;"
                            ),
                        }
                    ],
                }
            ),
        ]
    )

    class Tokenizer:
        @staticmethod
        def apply_chat_template(*_args, **_kwargs):
            return "prompt"

    generator = MlxSpanPlanGenerator(
        model_path="fixture-model",
        max_tokens=256,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda *_args, **_kwargs: next(outputs),
    )

    raw = generator(task, revision)

    assert "fmt_specs.align == align::none" in raw
    assert [row["status"] for row in generator.generation_trace_results()] == [
        "structural-rejected",
        "structural-valid",
    ]


def test_exact_span_candidates_include_complete_vue_boundary_statements(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "src/example.ts"
    source.write_text(
        "export function transformFor(node: ForNode) {\n"
        "  const keyProp = findProp(node, `key`)\n"
        "  const keyExp =\n"
        "    keyProp &&\n"
        "    (keyProp.type === NodeTypes.ATTRIBUTE\n"
        "      ? createSimpleExpression(keyProp.value!.content, true)\n"
        "      : keyProp.exp!)\n"
        "  const fragmentFlag = isStableFragment\n"
        "    ? PatchFlags.STABLE_FRAGMENT\n"
        "    : keyProp\n"
        "      ? PatchFlags.KEYED_FRAGMENT\n"
        "      : PatchFlags.UNKEYED_FRAGMENT\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="vue-span-fixture",
        checkout=checkout,
        instruction=(
            'A shorthand :key on v-for must not be treated like :key="key". '
            "Preserve KEYED_FRAGMENT behavior for an explicit value."
        ),
        allowed_targets=["src/example.ts"],
        cohort="feedback",
    )

    candidates = fixed_exact_span_candidates(task, max_candidates=12)
    before = {candidate.before for candidate in candidates}

    assert (
        "const keyExp =\n"
        "    keyProp &&\n"
        "    (keyProp.type === NodeTypes.ATTRIBUTE\n"
        "      ? createSimpleExpression(keyProp.value!.content, true)\n"
        "      : keyProp.exp!)"
    ) in before
    assert (
        "const fragmentFlag = isStableFragment\n"
        "    ? PatchFlags.STABLE_FRAGMENT\n"
        "    : keyProp\n"
        "      ? PatchFlags.KEYED_FRAGMENT\n"
        "      : PatchFlags.UNKEYED_FRAGMENT"
    ) in before
    ordered = [candidate.before for candidate in candidates]
    assert ordered.index("const keyProp = findProp(node, `key`)") < ordered.index(
        "const fragmentFlag = isStableFragment\n"
        "    ? PatchFlags.STABLE_FRAGMENT\n"
        "    : keyProp\n"
        "      ? PatchFlags.KEYED_FRAGMENT\n"
        "      : PatchFlags.UNKEYED_FRAGMENT"
    )
    assert all(candidate.occurrence == 0 for candidate in candidates)


def test_supporting_context_resolves_local_imported_function_contract(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "src/example.ts"
    source.write_text(
        "import { findProp } from './utils'\n"
        "export function transform(node: Node) {\n"
        "  const keyProp = findProp(node, 'key')\n"
        "  return keyProp ? 'KEYED_FRAGMENT' : 'UNKEYED_FRAGMENT'\n"
        "}\n",
        encoding="utf-8",
    )
    (checkout / "src/utils.ts").write_text(
        "export function findProp(\n"
        "  node: Node,\n"
        "  name: string,\n"
        "  allowEmpty: boolean = false,\n"
        ") {\n"
        "  return node.props.find(prop => prop.name === name && "
        "(prop.value || allowEmpty))\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="support-context-fixture",
        checkout=checkout,
        instruction="A shorthand key must produce KEYED_FRAGMENT.",
        allowed_targets=["src/example.ts"],
        cohort="feedback",
    )
    candidates = fixed_exact_span_candidates(task, max_candidates=8)

    contexts = fixed_supporting_symbol_contexts(task, candidates)

    assert contexts[0].symbol == "findProp"
    assert contexts[0].file == "src/utils.ts"
    assert "allowEmpty: boolean = false" in contexts[0].source


def test_exact_span_candidates_trace_environment_sink_to_process_constructor(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "src/run_external.rs"
    source.write_text(
        "pub fn create_process(env_vars: &Env) {\n"
        '    let mut process = std::process::Command::new("nu");\n'
        "    process.envs(env_vars);\n"
        "    process.spawn();\n"
        "}\n\n"
        "mod test {\n"
        "    fn checks_test_variable() {}\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="rust-process-span-fixture",
        checkout=checkout,
        instruction=(
            "Child processes inherit the TEST environment variable. Clear the "
            "environment before spawning the external process."
        ),
        allowed_targets=["src/run_external.rs"],
        cohort="feedback",
    )

    candidates = fixed_exact_span_candidates(task, max_candidates=8)

    assert candidates[0].before == (
        'let mut process = std::process::Command::new("nu");'
    )
    assert candidates[0].line == 2


def test_repository_api_context_finds_issue_aligned_method_exemplar(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    target = checkout / "src/run_external.rs"
    target.write_text(
        "pub fn spawn_external(env_vars: &Env) {\n"
        '    let mut process = std::process::Command::new("nu");\n'
        "    process.envs(env_vars);\n"
        "}\n",
        encoding="utf-8",
    )
    exemplar = checkout / "src/process_fixture.rs"
    exemplar.write_text(
        "pub fn isolated_command() {\n"
        '    let mut command = std::process::Command::new("nu");\n'
        "    command.env_clear();\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="rust-api-evidence-fixture",
        checkout=checkout,
        instruction=(
            "Child processes inherit environment variables; clear the environment "
            "before spawning."
        ),
        allowed_targets=["src/run_external.rs"],
        cohort="feedback",
    )
    candidates = fixed_exact_span_candidates(task, max_candidates=8)[:1]

    contexts = fixed_repository_api_contexts(task, candidates)

    assert contexts[0].symbol == "env_clear"
    assert contexts[0].file == "src/process_fixture.rs"
    assert "command.env_clear();" in contexts[0].source


def test_exact_span_candidates_rank_minus_flag_alignment_overwrite(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "include/fmt/printf.h"
    source.parent.mkdir(parents=True)
    source.write_text(
        "void parse_flags(format_specs& specs) {\n"
        "  specs.align = align::left;\n"
        "}\n"
        "template <typename T> void format_char(T value) {\n"
        "  if (std::is_same<T, char>::value) {\n"
        "    fmt_specs.align = align::right;\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="fmt-minus-alignment-fixture",
        checkout=checkout,
        instruction=(
            "fmt::sprintf ignores the minus flag for char: %-5c should keep "
            "left alignment instead of right alignment."
        ),
        allowed_targets=["include/fmt/printf.h"],
        cohort="feedback",
    )

    candidates = fixed_exact_span_candidates(task, max_candidates=8)

    assert candidates[0].before == "fmt_specs.align = align::right;"


def test_causal_state_context_exposes_flag_parser_and_default(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "include/fmt/printf.h"
    source.parent.mkdir(parents=True)
    source.write_text(
        "void parse_flags(format_specs& specs, char flag) {\n"
        "  switch (flag) {\n"
        "  case '-':\n"
        "    specs.align = align::left;\n"
        "  }\n"
        "}\n"
        "void format() {\n"
        "  format_specs specs;\n"
        "  specs.align = align::right;\n"
        "}\n"
        "template <typename T> void format_char(T value) {\n"
        "  if (std::is_same<T, char>::value) {\n"
        "    fmt_specs.align = align::right;\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="fmt-causal-state-fixture",
        checkout=checkout,
        instruction=(
            "fmt::sprintf ignores the minus flag for char: %-5c should keep "
            "left alignment while the default remains right aligned."
        ),
        allowed_targets=["include/fmt/printf.h"],
        cohort="feedback",
    )
    candidates = fixed_exact_span_candidates(task, max_candidates=8)[:1]

    contexts = fixed_causal_state_contexts(task, candidates, max_contexts=2)
    evidence = "\n".join(context.source for context in contexts)

    assert "case '-':" in evidence
    assert "specs.align = align::left;" in evidence
    assert "specs.align = align::right;" in evidence
    assert "fmt_specs.align = align::right;" not in evidence


def test_repository_state_value_context_exposes_neutral_enum_value(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    printf = checkout / "include/fmt/printf.h"
    printf.parent.mkdir(parents=True)
    printf.write_text(
        "void format_char() {\n  fmt_specs.align = align::right;\n}\n",
        encoding="utf-8",
    )
    format_header = checkout / "include/fmt/format.h"
    format_header.write_text(
        "struct specs {\n  align_t align = align::none;\n};\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="fmt-state-value-fixture",
        checkout=checkout,
        instruction="The minus flag for char must preserve left alignment.",
        allowed_targets=["include/fmt/printf.h", "include/fmt/format.h"],
        cohort="feedback",
    )
    candidates = fixed_exact_span_candidates(task, max_candidates=8)[:1]

    contexts = fixed_repository_state_value_contexts(task, candidates)

    assert contexts[0].symbol == "state-value:align:none"
    assert contexts[0].file == "include/fmt/format.h"
    assert "align::none" in contexts[0].source


def test_editable_candidate_context_exposes_enclosing_branch(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "include/fmt/printf.h"
    source.parent.mkdir(parents=True)
    source.write_text(
        "template <typename T> void format_value(T value) {\n"
        "  if (is_bool<T>()) {\n"
        "    write_bool(value);\n"
        "  } else if (is_char<T>()) {\n"
        "    fmt_specs.sign = sign::none;\n"
        "    fmt_specs.align = align::right;\n"
        "    write(value);\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="fmt-editable-context-fixture",
        checkout=checkout,
        instruction="The minus flag for char must preserve left alignment.",
        allowed_targets=["include/fmt/printf.h"],
        cohort="feedback",
    )
    candidates = fixed_exact_span_candidates(task, max_candidates=8)[:1]

    contexts = fixed_editable_candidate_contexts(task, candidates)

    assert contexts[0].symbol == "editable-context:span-001"
    assert "else if (is_char<T>())" in contexts[0].source
    assert "fmt_specs.align = align::right;" in contexts[0].source
    assert "write(value);" in contexts[0].source


def test_shared_editable_scope_preserves_cross_file_state_evidence(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    printf = checkout / "include/fmt/printf.h"
    printf.parent.mkdir(parents=True)
    printf.write_text(
        "void format_char() {\n  fmt_specs.align = align::right;\n}\n",
        encoding="utf-8",
    )
    (checkout / "include/fmt/format.h").write_text(
        "struct specs { align_t align = align::none; };\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="fmt-dual-scope-state-evidence-fixture",
        checkout=checkout,
        instruction="The minus flag for char must preserve left alignment.",
        allowed_targets=["include/fmt/printf.h", "include/fmt/format.h"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="fmt-dual-scope",
        revision_id="fmt-dual-scope-r054",
        parent_revision_id=None,
        source_round=54,
        protocol="multilanguage-typed-state-action-v13",
        skill_text=(
            "Use one typed action.\n\n"
            "## Shared diagnosis and localization (read-only)\n"
            '{"target_files":["include/fmt/printf.h"]}'
        ),
        prompt_template="Return one typed action.",
        eval_note="fixture",
    )

    editable = _editable_span_task(task, revision)
    candidates = fixed_exact_span_candidates(editable, max_candidates=8)[:1]
    actions = fixed_typed_state_actions(task, candidates)

    assert editable.allowed_targets == ("include/fmt/printf.h",)
    assert any(
        action["operation"] == "set-neutral-state"
        and action["state_value"] == "align::none"
        for action in actions
    )


def test_typed_collection_action_derives_hidden_item_filter_from_repository(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    usage = checkout / "src/usage.rs"
    usage.write_text(
        "fn write_help_usage(cmd: &Command) {\n"
        "  if cmd.is_flatten_help_set() {\n"
        "    for (i, sub) in cmd.get_subcommands().enumerate() {\n"
        "      if i != 0 { separate(); }\n"
        "      write_usage(sub);\n"
        "    }\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    command = checkout / "src/command.rs"
    command.write_text(
        "impl Command {\n"
        "  pub fn is_hide_set(&self) -> bool { self.hide }\n"
        "  pub fn is_hide_long_help_set(&self) -> bool { self.hide_long }\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="hidden-collection-filter-fixture",
        checkout=checkout,
        instruction=("Hide subcommands marked hide(true) from flattened help usage."),
        allowed_targets=["src/usage.rs", "src/command.rs"],
        cohort="feedback",
    )
    candidates = fixed_exact_span_candidates(task, max_candidates=8)[:1]

    actions = fixed_typed_collection_actions(task, candidates)

    assert actions == (
        {
            "candidate_id": "span-001",
            "operation": "filter-iteration-item",
            "predicate_method": "is_hide_set",
        },
    )


def test_typed_collection_action_materializes_filter_before_enumeration(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    usage = checkout / "src/usage.rs"
    usage.write_text(
        "fn write_help_usage(cmd: &Command) {\n"
        "  if cmd.is_flatten_help_set() {\n"
        "    for (i, sub) in cmd.get_subcommands().enumerate() {\n"
        "      if i != 0 { separate(); }\n"
        "      write_usage(sub);\n"
        "    }\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    command = checkout / "src/command.rs"
    command.write_text(
        "impl Command {\n  pub fn is_hide_set(&self) -> bool { self.hide }\n}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="hidden-collection-render-fixture",
        checkout=checkout,
        instruction="Hide subcommands from flattened help usage.",
        allowed_targets=["src/usage.rs", "src/command.rs"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="typed-collection-filter",
        revision_id="typed-collection-filter-r055",
        parent_revision_id=None,
        source_round=55,
        protocol="multilanguage-typed-edit-action-v14",
        skill_text="Filter excluded iteration items before enumeration.",
        prompt_template="Return one typed action.",
        eval_note="fixture",
    )
    candidates = fixed_exact_span_candidates(task, max_candidates=8)[:1]
    action = fixed_typed_collection_actions(task, candidates)[0]
    assert action in fixed_typed_actions(task, candidates)

    attempt = SpanPlanAdapter(
        generator=lambda _task, _revision: json.dumps(
            {"schema_version": 1, "actions": [action]}
        )
    ).run(task, revision)

    assert attempt.structural_valid is True
    assert (
        "+    for (i, sub) in cmd.get_subcommands()"
        ".filter(|sub| !sub.is_hide_set()).enumerate() {"
    ) in attempt.patch


def test_typed_collection_action_abstains_without_source_derived_predicate(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    usage = checkout / "src/usage.rs"
    usage.write_text(
        "fn write_help_usage(cmd: &Command) {\n"
        "  for (i, sub) in cmd.get_subcommands().enumerate() {\n"
        "    write_usage(sub);\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="missing-filter-predicate-fixture",
        checkout=checkout,
        instruction="Hide subcommands from help usage.",
        allowed_targets=["src/usage.rs"],
        cohort="feedback",
    )
    candidates = fixed_exact_span_candidates(task, max_candidates=8)[:1]

    assert fixed_typed_collection_actions(task, candidates) == ()


def test_typed_endpoint_action_materializes_bracket_aware_ipv6_parser(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "lib/rpc.rb"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class Server\n"
        "  def initialize(endpoint)\n"
        "    bind, port = endpoint.split(':')\n"
        "    @bind = bind\n"
        "    @port = port\n"
        "  end\n"
        "end\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="typed-ipv6-endpoint-fixture",
        checkout=checkout,
        instruction="IPv6 rpc_endpoint [::]:24444 must parse host and port.",
        allowed_targets=["lib/rpc.rb"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="typed-ipv6-endpoint",
        revision_id="typed-ipv6-endpoint-r068",
        parent_revision_id=None,
        source_round=68,
        protocol="multilanguage-typed-edit-action-v19",
        skill_text=(
            "Use bracket-aware endpoint parsing for bracketed host:port inputs."
        ),
        prompt_template="Return one typed action.",
        eval_note="fixture",
    )
    candidates = fixed_exact_span_candidates(task, max_candidates=8)[:1]
    actions = fixed_typed_endpoint_actions(task, candidates)
    bracketed = next(
        action
        for action in actions
        if action["operation"] == "parse-bracketed-endpoint"
    )
    last_colon = next(
        action
        for action in actions
        if action["operation"] == "split-endpoint-at-last-colon"
    )
    validated = next(
        action
        for action in actions
        if action["operation"] == "validate-and-parse-endpoint"
    )

    rejected = SpanPlanAdapter(
        generator=lambda _task, _revision: json.dumps(
            {"schema_version": 1, "actions": [last_colon]}
        )
    ).run(task, revision)
    accepted = SpanPlanAdapter(
        generator=lambda _task, _revision: json.dumps(
            {"schema_version": 1, "actions": [bracketed]}
        )
    ).run(task, revision)

    assert rejected.structural_valid is False
    assert rejected.failure_reason == "semantic-overbroad"
    assert accepted.structural_valid is True
    assert "match(/\\A\\[(.*)\\]:(\\d+)\\z/)" in accepted.patch
    assert "&.captures || endpoint.split(':')" in accepted.patch

    validation_revision = LoopRevision.create(
        skill_id="typed-ipv6-endpoint-validation",
        revision_id="typed-ipv6-endpoint-validation-r069",
        parent_revision_id=None,
        source_round=69,
        protocol="multilanguage-typed-edit-action-v20",
        skill_text=(
            "Use validation-preserving endpoint parsing and preserve invalid-input "
            "rejection while adding bracketed address support."
        ),
        prompt_template="Return one typed action.",
        eval_note="fixture",
    )
    shallow = SpanPlanAdapter(
        generator=lambda _task, _revision: json.dumps(
            {"schema_version": 1, "actions": [bracketed]}
        )
    ).run(task, validation_revision)
    strict = SpanPlanAdapter(
        generator=lambda _task, _revision: json.dumps(
            {"schema_version": 1, "actions": [validated]}
        )
    ).run(task, validation_revision)

    assert shallow.structural_valid is False
    assert shallow.failure_reason == "semantic-overbroad"
    assert strict.structural_valid is True
    assert "raise Fluent::ConfigError" in strict.patch
    assert "match[1] || match[2], match[3]" in strict.patch


def test_typed_state_action_preserves_existing_state_by_exact_deletion(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    printf = checkout / "include/fmt/printf.h"
    printf.parent.mkdir(parents=True)
    printf.write_text(
        "void format_char() {\n  fmt_specs.align = align::right;\n}\n",
        encoding="utf-8",
    )
    (checkout / "include/fmt/format.h").write_text(
        "struct specs { align_t align = align::none; };\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="fmt-typed-state-action-fixture",
        checkout=checkout,
        instruction="The minus flag for char must preserve left alignment.",
        allowed_targets=["include/fmt/printf.h", "include/fmt/format.h"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="fmt-typed-state-action",
        revision_id="fmt-typed-state-action-r039",
        parent_revision_id=None,
        source_round=39,
        protocol="multilanguage-typed-state-action-v12",
        skill_text="Preserve parser-owned state.",
        prompt_template="Return one typed action.",
        eval_note="fixture",
    )
    candidates = fixed_exact_span_candidates(task, max_candidates=8)[:1]
    actions = fixed_typed_state_actions(task, candidates)
    raw = json.dumps(
        {
            "schema_version": 1,
            "actions": [
                next(
                    action
                    for action in actions
                    if action["operation"] == "preserve-existing-state"
                )
            ],
        }
    )

    attempt = SpanPlanAdapter(generator=lambda _task, _revision: raw).run(
        task, revision
    )

    assert attempt.structural_valid is True
    assert "-  fmt_specs.align = align::right;" in attempt.patch
    added_source = [
        line
        for line in attempt.patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    assert all(line.strip() == "+" for line in added_source)


def test_typed_state_action_revision_rejects_legacy_plan_shape(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "include/fmt/printf.h"
    source.parent.mkdir(parents=True)
    source.write_text(
        "void format_char() {\n  fmt_specs.align = align::right;\n}\n",
        encoding="utf-8",
    )
    (checkout / "include/fmt/format.h").write_text(
        "struct specs { align_t align = align::none; };\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="fmt-typed-state-legacy-fixture",
        checkout=checkout,
        instruction="The minus flag for char must preserve left alignment.",
        allowed_targets=["include/fmt/printf.h", "include/fmt/format.h"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="fmt-typed-state-legacy",
        revision_id="fmt-typed-state-legacy-r039",
        parent_revision_id=None,
        source_round=39,
        protocol="multilanguage-typed-state-action-v12",
        skill_text="Preserve parser-owned state.",
        prompt_template="Return one typed action.",
        eval_note="fixture",
    )
    legacy = json.dumps(
        {
            "schema_version": 1,
            "edits": [
                {
                    "file": "include/fmt/printf.h",
                    "before": "fmt_specs.align = align::right;",
                    "after": "fmt_specs.align = align::none;",
                }
            ],
        }
    )

    attempt = SpanPlanAdapter(generator=lambda _task, _revision: legacy).run(
        task, revision
    )

    assert attempt.structural_valid is False
    assert attempt.failure_reason == "schema-invalid"


def test_typed_state_actions_derive_and_materialize_transient_guard(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "include/fmt/printf.h"
    source.parent.mkdir(parents=True)
    source.write_text(
        "void parse_flags(format_specs& specs, char flag) {\n"
        "  specs.fill[0] = ' ';\n"
        "  switch (flag) {\n"
        "  case '-':\n"
        "    specs.align = align::left;\n"
        "    break;\n"
        "  }\n"
        "}\n"
        "void prepare(format_specs& specs, bool zero) {\n"
        "  if (zero) {\n"
        "    specs.fill[0] = '0';\n"
        "    specs.align = align::numeric;\n"
        "  }\n"
        "}\n"
        "void format_char() {\n"
        "  fmt_specs.align = align::right;\n"
        "}\n",
        encoding="utf-8",
    )
    (checkout / "include/fmt/format.h").write_text(
        "struct specs { align_t align = align::none; };\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="fmt-transient-state-action-fixture",
        checkout=checkout,
        instruction="The minus flag for char must preserve left alignment.",
        allowed_targets=["include/fmt/printf.h", "include/fmt/format.h"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="fmt-transient-state-action",
        revision_id="fmt-transient-state-action-r050",
        parent_revision_id=None,
        source_round=50,
        protocol="multilanguage-typed-state-action-v13",
        skill_text=(
            "Enumerate every writer. Never select set-neutral-state for alignment "
            "state that participates in width/fill behavior."
        ),
        prompt_template="Return one typed action.",
        eval_note="fixture",
    )
    candidates = fixed_exact_span_candidates(task, max_candidates=8)
    candidate = next(
        row for row in candidates if row.before == "fmt_specs.align = align::right;"
    )
    actions = fixed_typed_state_actions(task, (candidate,))
    transient = next(
        row
        for row in actions
        if row["operation"] == "guard-neutral-or-transient-default"
        and row["state_value"] == "align::numeric"
    )
    assert not any(
        row["operation"] == "guard-neutral-or-transient-default"
        and row["state_value"] == "align::left"
        for row in actions
    )

    attempt = SpanPlanAdapter(
        generator=lambda _task, _revision: json.dumps(
            {"schema_version": 1, "actions": [transient]}
        )
    ).run(task, revision)

    assert attempt.structural_valid is True
    assert (
        "if (fmt_specs.align == align::none || fmt_specs.align == align::numeric)"
        in attempt.patch
    )


def test_inactive_skill_preference_rejects_neutral_and_accepts_preserve(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "include/fmt/printf.h"
    source.parent.mkdir(parents=True)
    source.write_text(
        "void format_char() {\n  fmt_specs.align = align::right;\n}\n",
        encoding="utf-8",
    )
    (checkout / "include/fmt/format.h").write_text(
        "struct specs { align_t align = align::none; };\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="fmt-skill-action-preference-fixture",
        checkout=checkout,
        instruction="The minus flag for char must preserve left alignment.",
        allowed_targets=["include/fmt/printf.h", "include/fmt/format.h"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="fmt-skill-action-preference",
        revision_id="fmt-skill-action-preference-r044",
        parent_revision_id=None,
        source_round=44,
        protocol="multilanguage-typed-state-action-v12",
        skill_text=(
            "When one operation preserves parser-derived state, prefer it over "
            "neutral assignment."
        ),
        prompt_template="Return one typed action.",
        eval_note="fixture",
    )

    def action(operation: str) -> str:
        return json.dumps(
            {
                "schema_version": 1,
                "actions": [
                    {
                        "candidate_id": "span-001",
                        "operation": operation,
                        "state_value": "align::none",
                    }
                ],
            }
        )

    rejected = SpanPlanAdapter(
        generator=lambda _task, _revision: action("set-neutral-state")
    ).run(task, revision)
    accepted = SpanPlanAdapter(
        generator=lambda _task, _revision: action("preserve-existing-state")
    ).run(task, revision)

    assert rejected.structural_valid is False
    assert rejected.failure_reason == "semantic-overbroad"
    assert "requires preserve-existing-state" in rejected.detail
    assert accepted.structural_valid is True


def test_span_generator_replans_to_inactive_skill_preferred_action(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "include/fmt/printf.h"
    source.parent.mkdir(parents=True)
    source.write_text(
        "void format_char() {\n  fmt_specs.align = align::right;\n}\n",
        encoding="utf-8",
    )
    (checkout / "include/fmt/format.h").write_text(
        "struct specs { align_t align = align::none; };\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="fmt-skill-action-replan-fixture",
        checkout=checkout,
        instruction="The minus flag for char must preserve left alignment.",
        allowed_targets=["include/fmt/printf.h", "include/fmt/format.h"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="fmt-skill-action-replan",
        revision_id="fmt-skill-action-replan-r044",
        parent_revision_id=None,
        source_round=44,
        protocol="multilanguage-typed-state-action-v12",
        skill_text=(
            "When one operation preserves parser-derived state, prefer it over "
            "neutral assignment."
        ),
        prompt_template="Return one typed action.",
        eval_note="fixture",
    )

    def action(operation: str) -> str:
        return json.dumps(
            {
                "schema_version": 1,
                "actions": [
                    {
                        "candidate_id": "span-001",
                        "operation": operation,
                        "state_value": "align::none",
                    }
                ],
            }
        )

    outputs = iter([action("set-neutral-state"), action("preserve-existing-state")])

    class Tokenizer:
        @staticmethod
        def apply_chat_template(*_args, **_kwargs):
            return "prompt"

    generator = MlxSpanPlanGenerator(
        model_path="fixture-model",
        max_tokens=256,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda *_args, **_kwargs: next(outputs),
    )

    raw = generator(task, revision)

    assert json.loads(raw)["actions"][0]["operation"] == "preserve-existing-state"
    assert [row["status"] for row in generator.generation_trace_results()] == [
        "structural-rejected",
        "structural-valid",
    ]
    assert (
        "requires preserve-existing-state"
        in generator.generation_trace_results()[0]["detail"]
    )


def test_inactive_skill_guard_preference_rejects_preserve_and_accepts_guard(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "include/fmt/printf.h"
    source.parent.mkdir(parents=True)
    source.write_text(
        "void format_char() {\n  fmt_specs.align = align::right;\n}\n",
        encoding="utf-8",
    )
    (checkout / "include/fmt/format.h").write_text(
        "struct specs { align_t align = align::none; };\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="fmt-skill-guard-preference-fixture",
        checkout=checkout,
        instruction="The minus flag for char must preserve left alignment.",
        allowed_targets=["include/fmt/printf.h", "include/fmt/format.h"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="fmt-skill-guard-preference",
        revision_id="fmt-skill-guard-preference-r047",
        parent_revision_id=None,
        source_round=47,
        protocol="multilanguage-typed-state-action-v12",
        skill_text=(
            "guard-neutral-default is preferred when an earlier write and a "
            "later state-dependent branch exist."
        ),
        prompt_template="Return one typed action.",
        eval_note="fixture",
    )

    def action(operation: str) -> str:
        return json.dumps(
            {
                "schema_version": 1,
                "actions": [
                    {
                        "candidate_id": "span-001",
                        "operation": operation,
                        "state_value": "align::none",
                    }
                ],
            }
        )

    rejected = SpanPlanAdapter(
        generator=lambda _task, _revision: action("preserve-existing-state")
    ).run(task, revision)
    accepted = SpanPlanAdapter(
        generator=lambda _task, _revision: action("guard-neutral-default")
    ).run(task, revision)

    assert rejected.structural_valid is False
    assert rejected.failure_reason == "semantic-overbroad"
    assert "requires guard-neutral-default" in rejected.detail
    assert accepted.structural_valid is True
    assert "if (fmt_specs.align == align::none)" in accepted.patch


def test_flag_state_overwrite_gate_rejects_foreign_scope_and_accepts_guard(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    source = checkout / "include/fmt/printf.h"
    source.parent.mkdir(parents=True)
    source.write_text(
        "void parse_flags(format_specs& specs, char flag) {\n"
        "  if (flag == '-') specs.align = align::left;\n"
        "}\n"
        "void format_char() {\n"
        "  fmt_specs.align = align::right;\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="fmt-state-gate-fixture",
        checkout=checkout,
        instruction="The minus flag for char must preserve left alignment.",
        allowed_targets=["include/fmt/printf.h"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="fmt-state-gate",
        revision_id="fmt-state-gate-r034",
        parent_revision_id=None,
        source_round=34,
        protocol="multilanguage-operation-only-span-bundle-v10",
        skill_text="Preserve parsed flag state with a conditional default guard.",
        prompt_template="Return operation-only edits.",
        eval_note="fixture",
    )

    def raw(after: str) -> str:
        return json.dumps(
            {
                "schema_version": 1,
                "edits": [
                    {
                        "file": "include/fmt/printf.h",
                        "before": "fmt_specs.align = align::right;",
                        "after": after,
                    }
                ],
            }
        )

    rejected = SpanPlanAdapter(
        generator=lambda _task, _revision: raw(
            "fmt_specs.align = align::right;\n"
            "if (specs.fill[0] == '0' && arg.type()) specs.fill[0] = ' ';"
        )
    ).run(task, revision)
    accepted = SpanPlanAdapter(
        generator=lambda _task, _revision: raw(
            "if (fmt_specs.align == align::none || "
            "fmt_specs.align == align::numeric)\n"
            "  fmt_specs.align = align::right;"
        )
    ).run(task, revision)
    neutral = SpanPlanAdapter(
        generator=lambda _task, _revision: raw("fmt_specs.align = align::none;")
    ).run(task, revision)
    removed = SpanPlanAdapter(
        generator=lambda _task, _revision: raw(
            "// Preserve alignment selected by parse_flags."
        )
    ).run(task, revision)

    assert rejected.structural_valid is False
    assert rejected.failure_reason == "semantic-overbroad"
    assert accepted.structural_valid is True
    assert neutral.structural_valid is True
    assert removed.structural_valid is True


def test_span_adapter_materializes_two_file_bundle_atomically(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "repo")
    (checkout / "src/second.ts").write_text(
        "export const offset = 1;\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    raw = json.dumps(
        {
            "schema_version": 1,
            "plans": [
                json.loads(_raw_plan()),
                {
                    "schema_version": 1,
                    "file": "src/second.ts",
                    "intent": {
                        "defect": "offset is stale",
                        "trigger": "answer uses two",
                        "desired_boundary": "offset is two",
                    },
                    "operations": [{"before": "offset = 1", "after": "offset = 2"}],
                    "diagnostic": "align the second file",
                },
            ],
            "diagnostic": "preserve the two-file invariant",
        }
    )
    task = StudentTask.create(
        task_id="span-bundle-fixture",
        checkout=checkout,
        instruction="Keep both answer offsets aligned at two.",
        allowed_targets=["src/example.ts", "src/second.ts"],
        cohort="feedback",
    )

    attempt = SpanPlanAdapter(generator=lambda _task, _revision: raw).run(
        task, _revision(True)
    )

    assert attempt.structural_valid is True
    assert "+++ b/src/example.ts" in attempt.patch
    assert "+++ b/src/second.ts" in attempt.patch


def test_span_adapter_classifies_malformed_json(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "repo")
    raw = '{"schema_version": 1, "plans": [}'

    attempt = SpanPlanAdapter(generator=lambda _task, _revision: raw).run(
        _task(checkout), _revision(True)
    )

    assert attempt.structural_valid is False
    assert attempt.failure_reason == "json-malformed"


def test_span_adapter_classifies_duplicate_file_bundle(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "repo")
    plan = json.loads(_raw_plan())
    raw = json.dumps(
        {
            "schema_version": 1,
            "plans": [plan, plan],
            "diagnostic": "duplicate plan",
        }
    )

    attempt = SpanPlanAdapter(generator=lambda _task, _revision: raw).run(
        _task(checkout), _revision(True)
    )

    assert attempt.structural_valid is False
    assert attempt.failure_reason == "duplicate-file"


def test_span_adapter_rejects_bundle_over_r010_budget(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "repo")
    oversized = _raw_plan() + (" " * 1200)

    attempt = SpanPlanAdapter(generator=lambda _task, _revision: oversized).run(
        _task(checkout), _revision(True)
    )

    assert attempt.structural_valid is False
    assert attempt.failure_reason == "plan-too-large"
    assert "1200" in attempt.detail


def test_span_adapter_accepts_explicit_unresolved_abstention(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "repo")
    raw = json.dumps(
        {
            "schema_version": 1,
            "status": "unresolved",
            "diagnostic": "No supplied span is unique.",
        }
    )

    attempt = SpanPlanAdapter(generator=lambda _task, _revision: raw).run(
        _task(checkout), _revision(True)
    )

    assert attempt.structural_valid is False
    assert attempt.failure_reason == "unresolved"
    assert "unique" in attempt.detail


def test_span_adapter_rejects_comment_only_pseudo_fix(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "repo")
    data = json.loads(_raw_plan())
    data["operations"] = [
        {
            "before": "  return value + 1;",
            "after": "  // Add two instead of one.\n  return value + 1;",
        }
    ]

    attempt = SpanPlanAdapter(generator=lambda _task, _revision: json.dumps(data)).run(
        _task(checkout), _revision(True, source_round=12)
    )

    assert attempt.structural_valid is False
    assert attempt.failure_reason == "non-executable-insertion"


def test_span_adapter_rejects_unrequested_definition_only_pseudo_fix(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    data = json.loads(_raw_plan())
    data["operations"] = [
        {
            "before": "export function answer(value: number) {",
            "after": (
                "export function unusedTelemetry() {}\n"
                "export function answer(value: number) {"
            ),
        }
    ]

    attempt = SpanPlanAdapter(generator=lambda _task, _revision: json.dumps(data)).run(
        _task(checkout), _revision(True, source_round=12)
    )

    assert attempt.structural_valid is False
    assert attempt.failure_reason == "non-executable-insertion"


def test_span_conditions_vary_only_teaching_content() -> None:
    baseline, taught = build_span_conditions(
        taught_skill="Use a unique exact source span.",
        parent_revision_id="r8",
        source_round=8,
        generation_config={"generator": "fixture"},
    )

    assert baseline.mechanism == taught.mechanism == "span"
    assert baseline.revision.protocol == taught.revision.protocol
    assert baseline.revision.protocol == "multilanguage-typed-state-action-v13"
    assert baseline.revision.prompt_template == taught.revision.prompt_template
    assert baseline.revision.skill_text != taught.revision.skill_text
    assert baseline.generation_config == taught.generation_config


def test_baseline_emits_bounded_candidate_set_with_definition_reachable(
    tmp_path: Path,
) -> None:
    """C1: the unpinned baseline exposes a bounded role-diverse candidate set.

    Before C1 the baseline returned only ``[:1]``; a wrong top-1 span (test-file
    line) made the student abstain -> empty patch.  The Catch2-style fixture
    below must yield the bounded set (min(16, 4*2, 8) == 8) and include the
    header's ``operator()`` definition line.
    """
    from skill_evolution_loop.span_student import _candidate_control_flow_roles

    checkout = tmp_path / "repo"
    checkout.mkdir()
    (checkout / "include").mkdir()
    (checkout / "test").mkdir()
    (checkout / "include/catch_approx.h").write_text(
        "namespace Catch {\n"
        "class Approx {\n"
        "public:\n"
        "  template <typename T> Approx operator()( T const& value ) {\n"
        "    return *this;\n"
        "  }\n"
        "  bool operator == ( const T& lhs ) const;\n"
        "};\n"
        "}\n",
        encoding="utf-8",
    )
    (checkout / "test/approx_tests.cpp").write_text(
        "const double dZero = 0;\n"
        "const double dSmall = 0.00001;\n"
        "REQUIRE( 1 == Approx( 1 ) );\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "fixture"],
        cwd=checkout,
        check=True,
    )
    task = StudentTask.create(
        task_id="c2-baseline-fixture",
        checkout=checkout,
        instruction="Make Approx::operator() const to fix the bug.",
        allowed_targets=["include/catch_approx.h", "test/approx_tests.cpp"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="baseline-no-skill",
        revision_id="baseline-v0",
        parent_revision_id=None,
        source_round=0,
        protocol="multilanguage-typed-state-action-v13",
        skill_text="No additional domain teaching is provided.",
        prompt_template="Return exactly one JSON object.",
        eval_note="baseline span probe",
    )
    candidates = _frozen_causal_candidates(task, revision, max_candidates=16)
    assert len(candidates) == 8  # min(16, 4 * 2 targets, 8)
    definition = next(
        (
            candidate
            for candidate in candidates
            if "operator()" in candidate.before
            and candidate.file.endswith("catch_approx.h")
        ),
        None,
    )
    assert definition is not None, [
        (c.file, c.line, c.before) for c in candidates
    ]
    assert "lexical-boundary" in _candidate_control_flow_roles(
        definition.before
    )


def test_definition_line_role_is_lexical_boundary() -> None:
    from skill_evolution_loop.span_student import _candidate_control_flow_roles

    roles = _candidate_control_flow_roles(
        "Approx operator()( T const& value ) {"
    )
    assert "lexical-boundary" in roles
    assert "call-site-boundary" in roles
    assert _candidate_control_flow_roles("const double dZero = 0;") == (
        "dataflow-boundary",
    )
    assert _candidate_control_flow_roles("foo(1);") == ("call-site-boundary",)


# --------------------------------------------------------------------------
# B: model capability profile wiring (B1 role hygiene + B2 length contracts)
# --------------------------------------------------------------------------


def test_exact_span_candidate_hides_roles_when_profile_disables() -> None:
    checkout = _checkout(Path(__import__("tempfile").mkdtemp()) / "repo")
    (candidate,) = _frozen_causal_candidates(_task(checkout), _revision(False))[:1]
    assert "roles" in candidate.to_prompt_dict(include_roles=True)
    assert "roles" not in candidate.to_prompt_dict(include_roles=False)


def test_span_generator_hides_roles_for_weak_4b_profile(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "repo")
    rendered: list[list[dict[str, str]]] = []

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            assert tokenize is False
            rendered.append(messages)
            return "\n".join(row["content"] for row in messages)

    generator = MlxSpanPlanGenerator(
        model_path="models/Qwen3.5-4B-mlx-4bit",
        max_tokens=512,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda *_args, **_kwargs: _raw_plan(),
    )
    assert generator.profile.model_id == "qwen3.5-4b-mlx-4bit"
    assert generator.profile.show_roles_in_prompt is False
    generator(_task(checkout), _revision(False))
    user_content = rendered[0][1]["content"]
    assert "ONLY EDITABLE exact-span candidates" in user_content
    assert '"roles"' not in user_content


def test_span_generator_shows_roles_when_profile_enables(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "repo")
    rendered: list[list[dict[str, str]]] = []

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            assert tokenize is False
            rendered.append(messages)
            return "\n".join(row["content"] for row in messages)

    generator = MlxSpanPlanGenerator(
        model_path="models/Qwen3.5-4B-mlx-4bit",
        max_tokens=512,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda *_args, **_kwargs: _raw_plan(),
        profile=StudentCapabilityProfile(
            model_id="custom-strong",
            show_roles_in_prompt=True,
            max_span_repairs=0,
        ),
    )
    generator(_task(checkout), _revision(False))
    user_content = rendered[0][1]["content"]
    assert '"roles"' in user_content
    assert '"lexical-boundary"' in user_content


def test_span_generator_bundle_chars_follow_profile(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "repo")
    rendered: list[list[dict[str, str]]] = []

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            assert tokenize is False
            rendered.append(messages)
            return "\n".join(row["content"] for row in messages)

    generator = MlxSpanPlanGenerator(
        model_path="models/Qwen2.5-Coder-7B-Instruct-4bit",
        max_tokens=512,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda *_args, **_kwargs: _raw_plan(),
    )
    # 7B profile: recipe per-op 800, bundle 1400.
    system_content = rendered[0][0]["content"] if rendered else ""
    generator(_task(checkout), _revision(True, source_round=70))
    system_content = rendered[0][0]["content"]
    assert "under 800 characters" in system_content
    assert "under 1400 characters" in system_content
    assert generator.generation_config()["max_bundle_chars"] == 1400


def test_span_adapter_plan_too_large_uses_profile_bundle_chars(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            return "\n".join(row["content"] for row in messages)

    generator = MlxSpanPlanGenerator(
        model_path="fixture-model",
        max_tokens=512,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda *_args, **_kwargs: _raw_plan(),
        profile=StudentCapabilityProfile(
            model_id="tiny", bundle_output_chars=100
        ),
    )
    adapter = SpanPlanAdapter(generator=generator)
    attempt = adapter.run(_task(checkout), _revision(False))
    assert attempt.structural_valid is False
    assert attempt.failure_reason == "plan-too-large"
    assert "100 characters" in attempt.detail


def test_semantic_recipe_after_gate_follows_profile_recipe_chars(
    tmp_path: Path,
) -> None:
    """P0-1: recipe after-limit comes from the profile, not a hard 600."""
    checkout = _checkout(tmp_path / "repo")
    task = _task(checkout)
    revision = _revision(True, source_round=70)
    candidates = _frozen_causal_candidates(task, revision)
    assert candidates
    candidate = candidates[0]
    long_after = "x" * 700  # > default 600, < profile 800
    raw = json.dumps(
        {
            "schema_version": 1,
            "recipes": [
                {"candidate_id": candidate.candidate_id, "after": long_after}
            ],
        }
    )
    projected, changed = _canonicalize_semantic_recipe_output(
        raw, candidates, task=task
    )
    assert changed is False  # default 600 rejects a 700-char after
    projected, changed = _canonicalize_semantic_recipe_output(
        raw,
        candidates,
        task=task,
        recipe_output_chars=800,
    )
    assert changed is True  # profile 800 accepts it
    assert "x" * 700 in projected
