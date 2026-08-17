"""Line-anchored local edit action space bounded to one Python symbol."""

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
from .student_adapter import (
    StudentAdapter,
    StudentAttempt,
    StudentTask,
    _implementation_fingerprint,
    _sha256_text,
)
from .symbol_rewrite import (
    MlxSymbolRewriteGenerator,
    _definition,
    fixed_symbol_context,
)

_BLOCK_FIELDS = frozenset(
    {"file", "symbol", "start_line", "end_line", "replacement", "diagnostic"}
)
_SYMBOL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_BLOCK_PROMPT_TEMPLATE = (
    "Return exactly one JSON object with file, symbol, start_line, end_line, "
    "replacement, and diagnostic."
)
_BLOCK_CONTEXT_SELECTOR = "frozen-first-target-issue-title-code-numbered-symbol-v1"


@dataclass(frozen=True)
class LineBlockRewrite:
    file: str
    symbol: str
    start_line: int
    end_line: int
    replacement: str
    diagnostic: str

    @classmethod
    def from_model_output(cls, raw: str) -> LineBlockRewrite:
        candidate = raw.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
        if fenced:
            candidate = fenced.group(1)
        else:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start < 0 or end <= start:
                raise ContractError("student output contains no line-block object")
            candidate = candidate[start : end + 1]
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ContractError("student line-block rewrite is malformed JSON") from exc
        if not isinstance(data, dict) or set(data) != _BLOCK_FIELDS:
            raise ContractError("student line-block rewrite fields are invalid")
        rewrite = cls(
            file=str(data["file"]),
            symbol=str(data["symbol"]),
            start_line=data["start_line"],
            end_line=data["end_line"],
            replacement=str(data["replacement"]),
            diagnostic=str(data["diagnostic"]),
        )
        rewrite.validate()
        return rewrite

    def validate(self) -> None:
        if (
            not self.file.strip()
            or _SYMBOL_NAME.fullmatch(self.symbol) is None
            or not self.replacement.strip()
            or not self.diagnostic.strip()
        ):
            raise ContractError("student line-block fields are invalid")
        if (
            type(self.start_line) is not int
            or type(self.end_line) is not int
            or self.start_line < 1
            or self.end_line < self.start_line
        ):
            raise ContractError("student line-block range is invalid")


class LineBlockRewriteAdapter(StudentAdapter):
    """Apply a model-selected absolute line range only inside one AST symbol."""

    def experiment_config(self) -> dict[str, Any]:
        configured = getattr(self.generator, "generation_config", None)
        return {
            "adapter": type(self).__name__,
            "adapter_contract": "python-line-block-rewrite-v1",
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
            rewrite = LineBlockRewrite.from_model_output(raw)
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
        symbol = _definition(tree.body, rewrite.symbol.split("."))
        if (
            symbol is None
            or symbol.end_lineno is None
            or rewrite.start_line < symbol.lineno
            or rewrite.end_line > symbol.end_lineno
        ):
            return self._failure(
                task,
                revision,
                raw,
                "apply-fail",
                "line range must be inside the named symbol",
                target_file=rewrite.file,
                before_sha256=_sha256_text(before),
            )
        lines = before.splitlines(keepends=True)
        source_line = lines[rewrite.start_line - 1]
        indentation = source_line[: len(source_line) - len(source_line.lstrip())]
        dedented = textwrap.dedent(rewrite.replacement).strip("\n") + "\n"
        replacement = "".join(
            f"{indentation}{line}\n" if line else "\n" for line in dedented.splitlines()
        )
        after = "".join(
            [
                *lines[: rewrite.start_line - 1],
                replacement,
                *lines[rewrite.end_line :],
            ]
        )
        if after == before:
            return self._failure(
                task,
                revision,
                raw,
                "apply-fail",
                "line-block replacement is identical to source",
                target_file=rewrite.file,
                before_sha256=_sha256_text(before),
            )
        try:
            ast.parse(after)
        except SyntaxError as exc:
            return self._failure(
                task,
                revision,
                raw,
                "apply-fail",
                f"line-block replacement breaks Python syntax: {exc.msg}",
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
        if not self._git_apply_check(task.checkout, patch):
            return self._failure(
                task,
                revision,
                raw,
                "apply-fail",
                "constructed line-block patch does not apply cleanly",
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


class MlxLineBlockRewriteGenerator(MlxSymbolRewriteGenerator):
    """Generate one bounded replacement against numbered source lines."""

    def __init__(self, *, max_tokens: int = 768, **fields: Any) -> None:
        super().__init__(max_tokens=max_tokens, **fields)

    def __call__(self, task: StudentTask, revision: LoopRevision) -> str:
        model, tokenizer, generate = self._runtime()
        relative, excerpt, start_line = fixed_symbol_context(
            task, self.max_context_chars
        )
        numbered = "\n".join(
            f"{start_line + index:06d}|{line}"
            for index, line in enumerate(excerpt.splitlines())
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a code-edit student. Return exactly one JSON object "
                    "with fields file, symbol, start_line, end_line, replacement, "
                    "and diagnostic. Select one contiguous numbered range inside "
                    "the supplied Python symbol. Replacement contains only the new "
                    "code for that range without line-number prefixes. It must "
                    "change behavior and preserve valid Python. Do not return a diff, "
                    "markdown, or prose.\n\n"
                    f"Protocol: {revision.protocol}\n"
                    f"Teaching Skill:\n{revision.skill_text}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task: {task.instruction}\n\n"
                    f"Allowed target: {relative}\n\n"
                    f"Numbered Python symbol source:\n### {relative}\n{numbered}\n\n"
                    "Return the line-block rewrite JSON now."
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
        self._last_generation_trace_kinds = ("line-block-attempt-0",)
        self._last_generation_prompt_trace = (prompt,)
        self._last_generation_trace_results = ({"status": "generated"},)
        return raw

    def generation_config(self) -> dict[str, Any]:
        return {
            **super().generation_config(),
            "action_space": "python-line-block-rewrite-v1",
            "context_selector": _BLOCK_CONTEXT_SELECTOR,
            "line_numbering": "absolute-six-digit-prefix-v1",
        }


def build_block_conditions(
    *,
    taught_skill: str,
    parent_revision_id: str,
    source_round: int,
    generation_config: dict[str, Any],
) -> list[ExperimentCondition]:
    definitions = [
        (
            "block-baseline",
            "baseline",
            "No additional domain teaching is provided. Use only repository evidence.",
            None,
        ),
        ("block-taught", "taught", taught_skill, parent_revision_id),
    ]
    conditions = []
    for condition_id, teaching, skill_text, parent_id in definitions:
        revision = LoopRevision.create(
            skill_id="p1-local-qwen-block-skill",
            revision_id=f"p1-{condition_id}-r{source_round:03d}",
            parent_revision_id=parent_id,
            source_round=source_round,
            protocol="python-line-block-rewrite-v1",
            skill_text=skill_text,
            prompt_template=_BLOCK_PROMPT_TEMPLATE,
            eval_note="P1 local numbered line-block mechanism comparison.",
        )
        conditions.append(
            ExperimentCondition.create(
                condition_id=condition_id,
                mechanism="block",
                teaching=teaching,
                revision=revision,
                generation_config=generation_config,
            )
        )
    return conditions
