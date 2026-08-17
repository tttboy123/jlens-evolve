"""AST-anchored Python symbol rewrite action space for local Student experiments."""

from __future__ import annotations

import ast
import difflib
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ContractError, LoopRevision
from .experiment import ExperimentCondition
from .mlx_student import (
    MlxStructuredGenerator,
    _issue_code_evidence,
    _issue_overlap_excerpt,
    _lexical_terms,
    _python_symbol_excerpt,
)
from .student_adapter import (
    StudentAdapter,
    StudentAttempt,
    StudentTask,
    _implementation_fingerprint,
    _sha256_text,
)

_SYMBOL_FIELDS = frozenset({"file", "symbol", "replacement", "diagnostic"})
_SYMBOL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_SYMBOL_PROMPT_TEMPLATE = (
    "Return exactly one JSON object with file, symbol, replacement, and diagnostic."
)
_SYMBOL_CONTEXT_SELECTOR = "frozen-first-target-issue-title-code-symbol-v3"


@dataclass(frozen=True)
class SymbolRewrite:
    """One complete Python definition replacement located by qualified symbol."""

    file: str
    symbol: str
    replacement: str
    diagnostic: str

    @classmethod
    def from_model_output(cls, raw: str) -> SymbolRewrite:
        candidate = raw.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
        if fenced:
            candidate = fenced.group(1)
        else:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start < 0 or end <= start:
                raise ContractError("student output contains no symbol rewrite object")
            candidate = candidate[start : end + 1]
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ContractError("student symbol rewrite is malformed JSON") from exc
        if not isinstance(data, dict) or set(data) != _SYMBOL_FIELDS:
            raise ContractError("student symbol rewrite fields are invalid")
        rewrite = cls(
            file=str(data["file"]),
            symbol=str(data["symbol"]),
            replacement=str(data["replacement"]),
            diagnostic=str(data["diagnostic"]),
        )
        rewrite.validate()
        return rewrite

    def validate(self) -> None:
        if (
            not self.file.strip()
            or not self.replacement.strip()
            or not self.diagnostic.strip()
        ):
            raise ContractError("student symbol rewrite fields must be non-empty")
        if _SYMBOL_NAME.fullmatch(self.symbol) is None:
            raise ContractError("student symbol name is invalid")

    def to_dict(self) -> dict[str, str]:
        return {
            "file": self.file,
            "symbol": self.symbol,
            "replacement": self.replacement,
            "diagnostic": self.diagnostic,
        }


_Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


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


def _definition(body: list[ast.stmt], parts: list[str]) -> _Definition | None:
    if len(parts) == 1:
        matches = _definitions_named(body, parts[0])
        return matches[0] if len(matches) == 1 else None
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name != parts[0]:
                continue
            if isinstance(node, ast.ClassDef):
                return _definition(node.body, parts[1:])
            return None
    return None


def qualified_symbol_at_line(source: str, line: int) -> str | None:
    """Return the deepest qualified definition enclosing one source line."""

    if type(line) is not int or line < 1:
        raise ContractError("qualified symbol line must be positive")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    matches: list[tuple[int, int, str]] = []

    def visit(body: list[ast.stmt], prefix: tuple[str, ...]) -> None:
        for node in body:
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            if node.end_lineno is None or not node.lineno <= line <= node.end_lineno:
                continue
            qualified = (*prefix, node.name)
            matches.append(
                (len(qualified), node.end_lineno - node.lineno, ".".join(qualified))
            )
            visit(node.body, qualified)

    visit(tree.body, ())
    return max(matches, key=lambda row: (row[0], -row[1]))[2] if matches else None


def qualified_symbol_for_anchor(
    source: str, excerpt: str, start_line: int, instruction: str
) -> str | None:
    """Choose the issue-relevant qualified definition inside one visible anchor."""

    if type(start_line) is not int or start_line < 1:
        raise ContractError("qualified symbol anchor line must be positive")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    end_line = start_line + max(0, len(excerpt.splitlines()) - 1)
    query = _lexical_terms(instruction)
    ranked: list[tuple[int, int, int, int, str]] = []

    def visit(body: list[ast.stmt], prefix: tuple[str, ...]) -> None:
        for node in body:
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            if node.end_lineno is None:
                continue
            qualified = (*prefix, node.name)
            if node.end_lineno >= start_line and node.lineno <= end_line:
                segment = ast.get_source_segment(source, node) or node.name
                terms = _lexical_terms(f"{'.'.join(qualified)}\n{segment}")
                ranked.append(
                    (
                        len(query & terms),
                        len(qualified),
                        -(node.end_lineno - node.lineno),
                        -node.lineno,
                        ".".join(qualified),
                    )
                )
            visit(node.body, qualified)

    visit(tree.body, ())
    if ranked:
        return max(ranked)[4]
    return qualified_symbol_at_line(source, start_line)


_QUALIFIED_SIGNAL_INDICES = (0, 2, 3, 4, 5, 7, 8)


def _qualified_symbol_rankings(
    source: str, instruction: str
) -> list[tuple[tuple[int, ...], str]]:
    """Rank definitions by issue signals, best first.

    Shared ranking core for both the legacy single-symbol API and the new
    top-N candidate API. The ranking is deterministic and uses only the public
    issue text plus the source file, never golden patch data.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    query = _lexical_terms(instruction)
    normalized_query = {
        {
            "classes": "class",
            "inherited": "inherit",
            "inherits": "inherit",
            "mocked": "mock",
            "mocking": "mock",
        }.get(token, token)
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", instruction.lower())
    }
    class_issue = bool(normalized_query.intersection({"class", "inherit", "subclass"}))
    raw_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", instruction)
        if "_" in token
    }
    boundary_issue = any(
        marker in instruction.lower()
        for marker in ("empty", "ignored", "ignore", "missing", "none")
    )
    ranked: list[
        tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, str]
    ] = []

    def visit(body: list[ast.stmt], prefix: tuple[str, ...]) -> None:
        for node in body:
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            qualified = (*prefix, node.name)
            segment = ast.get_source_segment(source, node) or node.name
            lowered = segment.lower()
            terms = _lexical_terms(segment)
            exact_hits = sum(lowered.count(token) for token in raw_tokens)
            span = getattr(node, "end_lineno", node.lineno) - node.lineno
            name_terms = {
                piece.casefold()
                for token in re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", node.name).split(
                    "_"
                )
                for piece in (token,)
                if len(piece) > 1
            }
            name_overlap = len(
                normalized_query & name_terms & {"class", "inherit", "mock", "subclass"}
            )
            # General name-term overlap over the full qualified path (not just the
            # leaf name). A definition whose qualified name contains distinctive issue
            # tokens (e.g. "method" + "index" in ``PyMethod.get_index_text``) is a much
            # stronger localization signal than a large body that merely mentions the
            # token incidentally. This closes the gap where ``resolve_xref`` /
            # ``PythonModuleIndex`` beat the real fix site on body-token volume alone.
            qualified_name_terms = {
                piece.casefold()
                for part in qualified
                for token in re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", part).split("_")
                for piece in (token,)
                if len(piece) > 1
            }
            name_overlap_general = len(query & qualified_name_terms)
            identity_hits = (
                sum(
                    marker in lowered
                    for marker in (
                        "__bases__",
                        "__display_name__",
                        "__module__",
                        "__mro__",
                        "__mro_entries__",
                        "__name__",
                        "__qualname__",
                    )
                )
                if class_issue
                else 0
            )
            boundary_hits = 0
            if boundary_issue and raw_tokens:
                for child in ast.walk(node):
                    if not isinstance(child, ast.If):
                        continue
                    test = (ast.get_source_segment(source, child.test) or "").lower()
                    if (
                        any(token in test for token in raw_tokens)
                        and isinstance(child.test, ast.UnaryOp)
                        and isinstance(child.test.op, ast.Not)
                    ):
                        boundary_hits += 1
            field_issue_hits = 0
            if isinstance(node, ast.ClassDef):
                for child in ast.walk(node):
                    if not isinstance(child, ast.Call):
                        continue
                    func_name = (
                        child.func.id if isinstance(child.func, ast.Name) else None
                    )
                    if not func_name or "Field" not in func_name:
                        continue
                    if not child.args:
                        continue
                    first = child.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        field_tokens = _lexical_terms(first.value)
                        if query & field_tokens or any(
                            q.startswith(t) or t.startswith(q)
                            for q in query
                            for t in field_tokens
                        ):
                            field_issue_hits += 1
            ranked.append(
                (
                    boundary_hits,
                    int(
                        boundary_hits > 0
                        and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ),
                    field_issue_hits,
                    int(
                        class_issue and isinstance(node, ast.ClassDef) and name_overlap
                    ),
                    identity_hits,
                    name_overlap,
                    name_overlap_general,
                    int(
                        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        or span <= 200
                    ),
                    len(query & terms),
                    exact_hits,
                    len(qualified),
                    -span,
                    -node.lineno,
                    ".".join(qualified),
                )
            )
            visit(node.body, qualified)

    visit(tree.body, ())
    ranked.sort(reverse=True)
    return [(tuple(row[:-1]), row[-1]) for row in ranked]


def qualified_symbol_for_issue(source: str, instruction: str) -> str | None:
    """Choose a definition using issue identifiers and boundary-shaped control flow."""

    rankings = _qualified_symbol_rankings(source, instruction)
    if not rankings:
        return None
    best_score, best_name = rankings[0]
    return (
        best_name
        if any(best_score[index] for index in _QUALIFIED_SIGNAL_INDICES)
        else None
    )


def qualified_symbol_candidates(
    source: str, instruction: str, *, top_n: int = 16
) -> tuple[str, ...]:
    """Return the top-N ranked qualified symbol candidates, best first.

    The single-symbol API is a structural recall bottleneck: an issue and its
    fix site often share zero lexical tokens, so one guess is wrong ~2/3 of the
    time. Returning an ordered top-N lets the Student make a choice instead of
    guessing a single name, while the adapter keeps the final AST anchor
    deterministic.
    """

    if type(top_n) is not int or top_n < 1:
        raise ContractError("symbol candidate top_n must be positive")
    rankings = _qualified_symbol_rankings(source, instruction)
    return tuple(name for _score, name in rankings[:top_n])


def qualified_symbol_excerpt(
    source: str, symbol: str, max_chars: int
) -> tuple[str, int] | None:
    """Return one exact, bounded definition excerpt for a frozen qualified symbol."""

    if type(max_chars) is not int or max_chars < 1:
        raise ContractError("qualified symbol excerpt budget must be positive")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    node = _definition(tree.body, symbol.split("."))
    if node is None:
        return None
    segment = ast.get_source_segment(source, node)
    if segment is None:
        return None
    return segment[:max_chars].rstrip() + "\n", node.lineno


def _normalized_replacement(
    rewrite: SymbolRewrite,
    original: _Definition,
    indentation: str,
) -> str:
    dedented = textwrap.dedent(rewrite.replacement).strip("\n") + "\n"
    try:
        parsed = ast.parse(dedented)
    except SyntaxError as exc:
        raise ContractError("replacement symbol is not valid Python") from exc
    if len(parsed.body) != 1 or not isinstance(
        parsed.body[0], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        raise ContractError("replacement must contain exactly one Python definition")
    replacement = parsed.body[0]
    if type(replacement) is not type(original) or replacement.name != original.name:
        raise ContractError(
            "replacement must define the same symbol and definition type"
        )
    if len(replacement.decorator_list) != len(original.decorator_list):
        raise ContractError("replacement must preserve the symbol decorator count")
    return "".join(
        f"{indentation}{line}\n" if line else "\n" for line in dedented.splitlines()
    )


class SymbolRewriteAdapter(StudentAdapter):
    """Construct an exact diff by locating one complete Python symbol with AST."""

    def experiment_config(self) -> dict[str, Any]:
        configured = getattr(self.generator, "generation_config", None)
        return {
            "adapter": type(self).__name__,
            "adapter_contract": "python-symbol-rewrite-v1",
            "generator": configured() if configured is not None else {},
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
        preliminary = self._classify_unstructured(raw)
        if preliminary is not None:
            return self._failure(task, revision, raw, preliminary, preliminary)
        try:
            rewrite = SymbolRewrite.from_model_output(raw)
        except ContractError as exc:
            return self._failure(task, revision, raw, "malformed-hunk", str(exc))

        allowed = task.allowed_targets or self._discover_targets(task)
        if (
            rewrite.file not in allowed
            or self._is_test_path(rewrite.file)
            or Path(rewrite.file).suffix != ".py"
        ):
            return self._failure(
                task,
                revision,
                raw,
                "wrong-target",
                f"target is not an allowed Python source: {rewrite.file}",
                target_file=rewrite.file,
            )
        try:
            target = task.resolve_target(rewrite.file)
            before = target.read_text(encoding="utf-8")
            tree = ast.parse(before)
        except (ContractError, OSError, UnicodeError, SyntaxError) as exc:
            return self._failure(
                task,
                revision,
                raw,
                "wrong-target",
                str(exc),
                target_file=rewrite.file,
            )
        original = _definition(tree.body, rewrite.symbol.split("."))
        if original is None or original.end_lineno is None:
            return self._failure(
                task,
                revision,
                raw,
                "apply-fail",
                f"symbol must resolve exactly once: {rewrite.symbol}",
                target_file=rewrite.file,
                before_sha256=_sha256_text(before),
            )
        start_line = min(
            [original.lineno, *(row.lineno for row in original.decorator_list)]
        )
        lines = before.splitlines(keepends=True)
        source_line = lines[start_line - 1]
        indentation = source_line[: len(source_line) - len(source_line.lstrip())]
        try:
            replacement = _normalized_replacement(rewrite, original, indentation)
            after = "".join(
                [*lines[: start_line - 1], replacement, *lines[original.end_lineno :]]
            )
            ast.parse(after)
        except (ContractError, SyntaxError) as exc:
            return self._failure(
                task,
                revision,
                raw,
                "apply-fail",
                str(exc),
                target_file=rewrite.file,
                before_sha256=_sha256_text(before),
            )
        patch = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{rewrite.file}",
                tofile=f"b/{rewrite.file}",
            )
        )
        if not patch.strip() or not self._git_apply_check(task.checkout, patch):
            return self._failure(
                task,
                revision,
                raw,
                "apply-fail",
                "constructed symbol patch does not apply cleanly",
                patch=patch,
                target_file=rewrite.file,
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
            target_file=rewrite.file,
            before_sha256=_sha256_text(before),
            after_sha256=_sha256_text(after),
            implementation_fingerprint=_implementation_fingerprint(rewrite.file, after),
            structural_valid=True,
            failure_reason=None,
            detail=rewrite.diagnostic,
        )


def fixed_symbol_context(
    task: StudentTask, max_context_chars: int
) -> tuple[str, str, int]:
    """Select one frozen-target symbol without consulting the teaching treatment."""
    targets = list(task.allowed_targets)
    if not targets:
        targets = list(StudentAdapter._discover_targets(task))
    relative = targets[0]
    path = task.resolve_target(relative)
    content = path.read_text(encoding="utf-8", errors="replace")
    issue_title = next(
        (line.strip() for line in task.instruction.splitlines() if line.strip()),
        task.instruction,
    )
    context_query = issue_title
    code_evidence = _issue_code_evidence(task.instruction)
    if code_evidence:
        context_query += "\n" + code_evidence
    excerpt = _python_symbol_excerpt(
        content, context_query, max_context_chars
    ) or _issue_overlap_excerpt(content, context_query, max_context_chars)
    exact_excerpt = excerpt.rstrip()
    offset = content.find(exact_excerpt)
    if offset < 0:
        raise ContractError("selected symbol excerpt is not present in its source")
    excerpt = content[offset : offset + len(exact_excerpt)]
    start_line = content.count("\n", 0, offset) + 1
    return relative, excerpt, start_line


def fixed_symbol_contexts(
    task: StudentTask,
    max_context_chars: int,
    *,
    max_targets: int = 32,
) -> tuple[tuple[str, str, int], ...]:
    """Build bounded symbol anchors from the frozen file candidate universe."""

    targets = list(task.allowed_targets)
    if not targets:
        targets = list(StudentAdapter._discover_targets(task))
    targets = targets[:max_targets]
    if not targets:
        raise ContractError("symbol task has no source target")
    issue_title = next(
        (line.strip() for line in task.instruction.splitlines() if line.strip()),
        task.instruction,
    )
    context_query = issue_title
    code_evidence = _issue_code_evidence(task.instruction)
    if code_evidence:
        context_query += "\n" + code_evidence
    per_target = max(400, max_context_chars // len(targets))
    contexts: list[tuple[str, str, int]] = []
    for relative in targets:
        content = task.resolve_target(relative).read_text(
            encoding="utf-8", errors="replace"
        )
        symbol_excerpt = _python_symbol_excerpt(content, context_query, per_target)
        excerpt = symbol_excerpt or _issue_overlap_excerpt(
            content, context_query, per_target
        )
        if not excerpt:
            continue
        if symbol_excerpt is None:
            contexts.append((relative, excerpt, 1))
            continue
        exact_excerpt = excerpt.rstrip()
        offset = content.find(exact_excerpt)
        if offset < 0:
            raise ContractError("selected symbol excerpt is not present in its source")
        excerpt = content[offset : offset + len(exact_excerpt)]
        contexts.append((relative, excerpt, content.count("\n", 0, offset) + 1))
    if not contexts:
        raise ContractError("symbol source contexts are empty")
    return tuple(contexts)


class MlxSymbolRewriteGenerator(MlxStructuredGenerator):
    """Generate one complete Python definition while the adapter owns its span."""

    def __init__(
        self,
        *,
        max_tokens: int = 1536,
        max_context_chars: int = 8_000,
        **fields: Any,
    ) -> None:
        super().__init__(
            max_tokens=max_tokens,
            max_context_chars=max_context_chars,
            max_structural_repairs=0,
            use_grounding_plan=False,
            use_semantic_critic=False,
            **fields,
        )

    def __call__(self, task: StudentTask, revision: LoopRevision) -> str:
        model, tokenizer, generate = self._runtime()
        relative, excerpt, _start_line = fixed_symbol_context(
            task, self.max_context_chars
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a code-edit student. Return exactly one JSON object "
                    "with fields file, symbol, replacement, and diagnostic. The "
                    "symbol is the qualified name of exactly one supplied Python "
                    "function or class. Replacement must contain the complete valid "
                    "definition of that same symbol, including every unchanged path. "
                    "Do not return a diff, search span, markdown, or prose.\n\n"
                    f"Protocol: {revision.protocol}\n"
                    f"Teaching Skill:\n{revision.skill_text}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task: {task.instruction}\n\n"
                    f"Allowed target: {relative}\n\n"
                    f"Python symbol source:\n### {relative}\n{excerpt}\n\n"
                    "Return the symbol rewrite JSON now. Copy the file path and "
                    "symbol name from the supplied source."
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
            max_tokens=self.max_tokens,
        )
        self._last_generation_trace = (raw,)
        self._last_generation_trace_kinds = ("symbol-rewrite-attempt-0",)
        self._last_generation_prompt_trace = (prompt,)
        self._last_generation_trace_results = ({"status": "generated"},)
        return raw

    def generation_config(self) -> dict[str, Any]:
        return {
            **super().generation_config(),
            "action_space": "python-symbol-rewrite-v1",
            "context_selector": _SYMBOL_CONTEXT_SELECTOR,
            "target_policy": "first-frozen-qualified-target",
        }


def build_symbol_conditions(
    *,
    taught_skill: str,
    parent_revision_id: str,
    source_round: int,
    generation_config: dict[str, Any],
) -> list[ExperimentCondition]:
    """Build a strict paired comparison whose user prompt and action space are fixed."""
    definitions = [
        (
            "symbol-baseline",
            "baseline",
            "No additional domain teaching is provided. Use only repository evidence.",
            None,
        ),
        ("symbol-taught", "taught", taught_skill, parent_revision_id),
    ]
    conditions = []
    for condition_id, teaching, skill_text, parent_id in definitions:
        revision = LoopRevision.create(
            skill_id="p1-local-qwen-symbol-skill",
            revision_id=f"p1-{condition_id}-r{source_round:03d}",
            parent_revision_id=parent_id,
            source_round=source_round,
            protocol="python-symbol-rewrite-v1",
            skill_text=skill_text,
            prompt_template=_SYMBOL_PROMPT_TEMPLATE,
            eval_note=(
                "P1 local symbol-rewrite mechanism comparison; the adapter owns "
                "AST span selection and the student owns the complete definition."
            ),
        )
        conditions.append(
            ExperimentCondition.create(
                condition_id=condition_id,
                mechanism="symbol",
                teaching=teaching,
                revision=revision,
                generation_config=generation_config,
            )
        )
    return conditions
