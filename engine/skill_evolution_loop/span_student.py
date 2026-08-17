"""Local Student adapter for the multi-language exact-span action space."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .capabilities import StudentCapabilityProfile, profile_for
from .contracts import ContractError, LoopRevision, canonical_json
from .experiment import ExperimentCondition
from .mlx_student import (
    MlxStructuredGenerator,
    _issue_code_evidence,
    _issue_overlap_excerpt,
    _project_numbered_teaching,
)
from .span_rewrite import (
    SpanBundlePlan,
    SpanPlan,
    materialize_span_bundle,
)
from .student_adapter import (
    StudentAdapter,
    StudentAttempt,
    StudentTask,
    _implementation_fingerprint,
    _sha256_text,
    parse_unresolved_abstention,
)

_SPAN_PROMPT_TEMPLATE = "Return exactly one atomic exact-span bundle JSON object."
_SPAN_CONTEXT_SELECTOR = "frozen-read-only-candidate-control-flow-context-v10"
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_ISSUE_CODE_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_MAX_SPAN_BUNDLE_CHARS = 1_200
_MAX_EXACT_SPAN_CANDIDATES = 16
_MAX_EXACT_SPAN_LINES = 12
_MAX_EXACT_SPAN_CHARS = 600
_ASSIGNMENT_START = re.compile(
    r"^(?:(?:export\s+)?(?:const|let|var)\b|return\b|"
    r"(?:if|elif|unless|while|for|switch|match|case|when|rescue|except|"
    r"catch|ensure)\b)"
)
_CONTINUATION_END = re.compile(r"(?:=|&&|\|\||\?|:|\.|,|\(|\[|\{)$")
_CODE_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CALL_TOKEN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_METHOD_CALL_TOKEN = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_ENUMERATED_LOOP = re.compile(
    r"for\s+\((?P<index>[A-Za-z_][A-Za-z0-9_]*),\s*"
    r"(?P<item>[A-Za-z_][A-Za-z0-9_]*)\)\s+in\s+"
    r"(?P<iterator>[^\n{]+?)\.enumerate\(\)\s*\{"
)
_ENDPOINT_SPLIT_ASSIGNMENT = re.compile(
    r"^(?P<bind>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
    r"(?P<port>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"(?P<endpoint>[A-Za-z_][A-Za-z0-9_]*)\.split\((?P<quote>['\"]):"
    r"(?P=quote)\)$"
)
_LOCALE_SEPARATOR_GUARD = re.compile(
    r"\$(?P<thousands>[A-Za-z_][A-Za-z0-9_]*)\s*!==\s*"
    r"(?P<tquote>['\"])(?P<tdefault>.*?)(?P=tquote)\s*\|\|\s*"
    r"\$(?P<decimal>[A-Za-z_][A-Za-z0-9_]*)\s*!==\s*"
    r"(?P<dquote>['\"])(?P<ddefault>.*?)(?P=dquote)",
    re.IGNORECASE,
)
_LOCALE_REPLACE_CALL = re.compile(
    r"^(?P<prefix>\$[A-Za-z_][A-Za-z0-9_]*\s*=\s*str_replace\(\[)"
    r"(?P<search>[^\]]+)\],\s*\[(?P<replace>[^\]]+)\]"
    r"(?P<suffix>,\s*\$[A-Za-z_][A-Za-z0-9_]*\);)$"
)
_PHP_ARRAY_LITERAL = re.compile(
    r"\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)\s*(?:,|$)"
)
_PHP_NAMED_FUNCTION = re.compile(
    r"(?m)^[^\n{]*\bfunction\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^\n{]*\)"
    r"(?:\s*:\s*[^\n{]+)?\s*\{"
)
_COLLECTION_PREDICATE_METHOD = re.compile(
    r"\b(?P<method>(?:is_|should_)?[A-Za-z_][A-Za-z0-9_]*"
    r"(?:hide|hidden|exclude|omit|skip)[A-Za-z0-9_]*)\s*\("
)
_LOCAL_JS_IMPORT = re.compile(
    r"import\s*\{(?P<names>.*?)\}\s*from\s*['\"](?P<module>\.[^'\"]+)['\"]",
    re.DOTALL,
)
_ISSUE_STOP_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "any",
        "as",
        "be",
        "for",
        "from",
        "in",
        "info",
        "is",
        "link",
        "no",
        "of",
        "on",
        "or",
        "reproduction",
        "response",
        "same",
        "should",
        "steps",
        "system",
        "the",
        "to",
        "version",
        "what",
        "with",
    }
)
_CODE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".go",
        ".java",
        ".js",
        ".jsx",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".swift",
        ".ts",
        ".tsx",
    }
)
_DECLARATION_LINE = re.compile(
    r"^(?:(?:pub|export|public|protected|private|static|final|abstract|async)\s+)*"
    r"(?:fn|func|function|class|struct|enum|interface|type|mod|def)\b"
)
_FUNCTION_START = re.compile(
    r"^(?:func\s+(?:\([^)]*\)\s*)?|"
    r"(?:(?:pub|export|public|protected|private|static|final|abstract|async)\s+)*"
    r"(?:fn|function|def)\s+)"
    r"([A-Za-z_][A-Za-z0-9_]*)"
)


def _issue_api_method_family(
    instruction: str, defined_function_names: set[str]
) -> tuple[str, ...]:
    """Return source-defined API companions with a short directional prefix."""

    issue_symbols = {
        symbol.casefold()
        for symbol in re.findall(
            r"(?:->|\.|::)([A-Za-z_][A-Za-z0-9_]*)\s*\(", instruction
        )
    }
    defined = {name.casefold() for name in defined_function_names}
    exact = sorted(defined.intersection(issue_symbols))
    companions = sorted(
        name
        for name in defined
        if name not in issue_symbols
        and any(
            len(symbol) >= 4
            and (
                (name.endswith(symbol) and len(name) - len(symbol) <= 3)
                or (symbol.endswith(name) and len(symbol) - len(name) <= 3)
            )
            for symbol in issue_symbols
        )
    )
    return tuple([*exact, *companions])


_TOKEN_ALIASES = {
    "children": "child",
    "cleared": "clear",
    "clearing": "clear",
    "commands": "command",
    "env": "environment",
    "envs": "environment",
    "hidden": "hide",
    "hides": "hide",
    "hiding": "hide",
    "inherited": "inherit",
    "inherits": "inherit",
    "mocked": "mock",
    "mocking": "mock",
    "processes": "process",
    "spawned": "spawn",
    "spawning": "spawn",
    "variables": "variable",
    "vars": "variable",
}
_API_BEHAVIOR_TOKENS = frozenset(
    {
        "clear",
        "close",
        "convert",
        "create",
        "delete",
        "deserialize",
        "emit",
        "execute",
        "hide",
        "inherit",
        "load",
        "parse",
        "read",
        "remove",
        "render",
        "save",
        "send",
        "serialize",
        "spawn",
        "transform",
        "write",
    }
)

_CANDIDATE_ROLE_ORDER = (
    "dataflow-boundary",
    "call-site-boundary",
    "exception-boundary",
    "side-effect-boundary",
    "guard-boundary",
    "exit-boundary",
)
_DATAFLOW_BOUNDARY = re.compile(
    r"^(?!.*(?:==|!=|<=|>=|=>))[^=\n]{1,160}(?<![!<>=])=(?!=)"
)
_CALL_SITE_BOUNDARY = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\s*\("
)
_EXCEPTION_BOUNDARY = re.compile(
    r"(?:^|\b)(?:rescue|except|catch|ensure|finally)(?:\b|\s|=>)|"
    r"\b(?:IO\.select|readpartial|recv)\s*\("
)
_SIDE_EFFECT_BOUNDARY = re.compile(
    r"(?:\.|\b)(?:close|commit|delete|disable|enable|remove|rollback|"
    r"shutdown|stop|terminate)!?\s*\("
    r"|(?:\.|\b)(?:close|commit|delete|disable|enable|remove|rollback|"
    r"shutdown|stop|terminate)!?(?:\s|$)"
)
_GUARD_BOUNDARY = re.compile(
    r"^(?:if|unless|elif|else\s+if|when)\b|"
    r"\b(?:nil\?|null\b|none\b|empty\?|closed\?)",
    re.IGNORECASE,
)
_EXIT_BOUNDARY = re.compile(r"^(?:return|yield|raise|break|continue|throw)\b")
_CONTROL_HEADER_ONLY = re.compile(
    r"^(?:ensure|finally|rescue(?:\s+[^\n]+)?|except(?:\s+[^\n]+)?|"
    r"catch(?:\s+[^\n]+)?|when\s+[^\n]+|case\s+[^\n]+)$"
)


def _candidate_control_flow_roles(before: str) -> tuple[str, ...]:
    """Classify public-source spans into language-neutral repair roles."""

    stripped = before.lstrip()
    roles: list[str] = []
    if stripped.rstrip().endswith("{"):
        # A line that opens a definition body (function/type) is a
        # lexical-boundary edit surface, not merely a call site.
        roles.append("lexical-boundary")
    for role, pattern in (
        ("dataflow-boundary", _DATAFLOW_BOUNDARY),
        ("call-site-boundary", _CALL_SITE_BOUNDARY),
        ("exception-boundary", _EXCEPTION_BOUNDARY),
        ("side-effect-boundary", _SIDE_EFFECT_BOUNDARY),
        ("guard-boundary", _GUARD_BOUNDARY),
        ("exit-boundary", _EXIT_BOUNDARY),
    ):
        if pattern.search(stripped):
            roles.append(role)
    return tuple(roles) or ("lexical-boundary",)


@dataclass(frozen=True)
class ExactSpanCandidate:
    """One framework-enumerated, byte-exact, unique source action."""

    candidate_id: str
    file: str
    before: str
    line: int
    occurrence: int
    score: int

    def to_prompt_dict(self, *, include_roles: bool = True) -> dict[str, Any]:
        prompt = {
            "candidate_id": self.candidate_id,
            "file": self.file,
            "before": self.before,
            "line": self.line,
            "occurrence": self.occurrence,
        }
        if include_roles:
            prompt["roles"] = list(_candidate_control_flow_roles(self.before))
        return prompt


@dataclass(frozen=True)
class SupportingSymbolContext:
    """A bounded local callee definition discovered without evaluator evidence."""

    file: str
    symbol: str
    source: str

    def to_prompt_dict(self) -> dict[str, str]:
        return {"file": self.file, "symbol": self.symbol, "source": self.source}


def _code_tokens(value: str) -> set[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value).replace("_", " ")
    return {token.lower() for token in _CODE_TOKEN.findall(expanded)}


def _normalized_code_tokens(value: str) -> set[str]:
    return {_TOKEN_ALIASES.get(token, token) for token in _code_tokens(value)}


def _semantic_code_tokens(value: str) -> set[str]:
    """Normalize common code/prose spellings without using evaluator evidence."""

    tokens = _normalized_code_tokens(value)
    if tokens.intersection({"child", "process", "spawn"}):
        tokens.update({"command", "execute", "external", "run"})
    if {"minus", "flag"}.issubset(tokens):
        tokens.update({"align", "left", "right"})
    return tokens


def _delimiter_balance(value: str) -> int:
    return sum(
        value.count(opening) - value.count(closing)
        for opening, closing in (
            ("(", ")"),
            ("[", "]"),
            ("{", "}"),
        )
    )


def _source_span_rows(content: str) -> tuple[tuple[int, str], ...]:
    """Extract bounded complete statements plus atomic lines without parsing."""

    lines = content.splitlines()
    rows: list[tuple[int, str]] = []
    seen: set[str] = set()

    def add(line_index: int, before: str) -> None:
        if not before or before in seen:
            return
        if len(before) > _MAX_EXACT_SPAN_CHARS:
            return
        if not 1 <= len(before.splitlines()) <= _MAX_EXACT_SPAN_LINES:
            return
        if content.count(before) != 1:
            return
        seen.add(before)
        rows.append((line_index + 1, before))

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "/*", "*", "#")):
            continue
        add(index, line.lstrip())
        if not _ASSIGNMENT_START.match(stripped):
            continue
        base_indent = len(line) - len(line.lstrip())
        end = index
        balance = _delimiter_balance(line)
        continued = bool(_CONTINUATION_END.search(stripped)) or balance > 0
        while end + 1 < len(lines):
            next_line = lines[end + 1]
            next_stripped = next_line.strip()
            if not next_stripped:
                break
            next_indent = len(next_line) - len(next_line.lstrip())
            if next_indent <= base_indent and not continued:
                break
            if next_indent <= base_indent:
                break
            end += 1
            balance += _delimiter_balance(next_line)
            continued = (
                balance > 0
                or bool(_CONTINUATION_END.search(next_stripped))
                or next_stripped.startswith(("?", ":", ".", "&&", "||"))
            )
            if end - index + 1 >= _MAX_EXACT_SPAN_LINES:
                break
        if end > index:
            before = "\n".join([lines[index].lstrip(), *lines[index + 1 : end + 1]])
            add(index, before)
    return tuple(rows)


def fixed_exact_span_candidates(
    task: StudentTask,
    *,
    max_candidates: int = _MAX_EXACT_SPAN_CANDIDATES,
) -> tuple[ExactSpanCandidate, ...]:
    """Rank a gold-free, frozen exact-span action set identically for both arms."""

    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("max exact span candidates must be positive")
    targets = list(task.allowed_targets)
    if not targets:
        targets = list(StudentAdapter._discover_targets(task))
    if not targets:
        raise ContractError("span task has no source target")
    issue_without_urls = re.sub(r"https?://\S+", " ", task.instruction)
    query_tokens = _semantic_code_tokens(issue_without_urls) - _ISSUE_STOP_TOKENS
    issue_title = next(
        (line for line in issue_without_urls.splitlines() if line.strip()), ""
    )
    title_tokens = _semantic_code_tokens(issue_title) - _ISSUE_STOP_TOKENS
    exact_issue_identifiers = {
        token.lower()
        for token in _CODE_TOKEN.findall(issue_without_urls)
        if token.isupper() and len(token) >= 3
    }
    compound_issue_identifiers = {
        token.casefold()
        for token in _CODE_TOKEN.findall(issue_without_urls)
        if "_" in token and len(token) >= 4 and not token.isupper()
    }
    issue_api_symbols = {
        symbol.casefold()
        for symbol in re.findall(
            r"(?:->|\.|::)([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            issue_without_urls,
        )
    }
    ranked: list[tuple[int, str, int, str]] = []
    high_signal_spans: list[tuple[str, int, str]] = []
    for relative in targets:
        content = task.resolve_target(relative).read_text(
            encoding="utf-8", errors="replace"
        )
        defined_function_names = {
            match.group(1)
            for source_text in content.splitlines()
            if (match := _FUNCTION_START.match(source_text.strip())) is not None
        }
        issue_aligned_functions = (
            set(_issue_api_method_family(issue_without_urls, defined_function_names))
            if issue_api_symbols
            else {
                name.casefold()
                for name in defined_function_names
                if len(query_tokens.intersection(_semantic_code_tokens(name))) >= 3
            }
        )
        enclosing_symbol_tokens: dict[int, set[str]] = {}
        enclosing_symbol_names: dict[int, str | None] = {}
        active_symbol_tokens: set[str] = set()
        active_symbol_name: str | None = None
        for source_line, source_text in enumerate(content.splitlines(), start=1):
            if match := _FUNCTION_START.match(source_text.strip()):
                active_symbol_name = match.group(1).casefold()
                active_symbol_tokens = _semantic_code_tokens(match.group(1))
            enclosing_symbol_tokens[source_line] = active_symbol_tokens
            enclosing_symbol_names[source_line] = active_symbol_name
        for line, before in _source_span_rows(content):
            before_tokens = _semantic_code_tokens(before)
            overlap = query_tokens.intersection(before_tokens)
            exact_overlap = exact_issue_identifiers.intersection(before_tokens)
            lowered_before = before.casefold()
            issue_verbatim_match = (
                len(before.strip()) >= 8 and before.strip() in task.instruction
            )
            compound_identifier_hits = sum(
                identifier in lowered_before
                for identifier in compound_issue_identifiers
            )
            declaration = bool(_DECLARATION_LINE.match(before.lstrip()))
            called_functions = {name.casefold() for name in _CALL_TOKEN.findall(before)}
            score = (
                3_000 * len(exact_overlap)
                + 2_000 * len(overlap)
                + 5_000
                * len(title_tokens.intersection(before_tokens))
                * (not declaration)
                + 80_000 * compound_identifier_hits
                + 30_000 * len(query_tokens.intersection(enclosing_symbol_tokens[line]))
                + 120_000 * bool(enclosing_symbol_names[line] in issue_api_symbols)
                + 90_000
                * bool(
                    enclosing_symbol_names[line] in issue_aligned_functions
                    and enclosing_symbol_names[line] not in issue_api_symbols
                )
                + 250_000
                * bool(
                    not declaration
                    and called_functions.intersection(issue_aligned_functions)
                )
                + 400 * before.lstrip().startswith(("return ", "if ", "if (", "while "))
                + 300
                * bool(
                    re.match(
                        r"^(?:(?:export\s+)?(?:const|let|var)(?:\s+mut)?\s+)",
                        before.lstrip(),
                    )
                )
                - 10_000 * declaration
                + 20 * (len(before.splitlines()) > 1)
                + 10 * ("=" in before)
                - 10_000 * bool(_CONTROL_HEADER_ONLY.fullmatch(before.strip()))
                - min(len(before), 99)
                + 60_000
                * bool(
                    query_tokens.intersection({"command", "process", "spawn"})
                    and re.search(r"\b(?:Command|CommandSys)::new\s*\(", before)
                )
                + 100_000
                * bool(
                    {"minus", "flag"}.issubset(query_tokens)
                    and len(before.splitlines()) == 1
                    and re.search(r"\balign\s*=\s*align::right\b", before)
                )
                + 1_000_000 * issue_verbatim_match
            )
            ranked.append((score, relative, line, before))
            if len(overlap) >= 2 or (bool(exact_overlap) and not declaration):
                high_signal_spans.append((relative, line, before))
    causally_ranked: list[tuple[int, str, int, str]] = []
    assignment = re.compile(
        r"^(?:(?:export\s+)?(?:const|let|var)(?:\s+mut)?\s+)"
        r"([A-Za-z_][A-Za-z0-9_]*)"
    )
    for score, relative, line, before in ranked:
        match = assignment.match(before)
        if match is not None:
            target_tokens = _semantic_code_tokens(match.group(1))
            downstream_uses = sum(
                bool(target_tokens)
                and target_tokens.issubset(_semantic_code_tokens(signal))
                for signal_file, signal_line, signal in high_signal_spans
                if signal_file == relative
                and line < signal_line <= line + 80
                and signal != before
            )
            if downstream_uses:
                score += (
                    25_000
                    + 1_000 * downstream_uses
                    + 5_000
                    * len(query_tokens.intersection(_semantic_code_tokens(before)))
                )
        causally_ranked.append((score, relative, line, before))
    ranked = causally_ranked
    ranked.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    return tuple(
        ExactSpanCandidate(
            candidate_id=f"span-{index:03d}",
            file=relative,
            before=before,
            line=line,
            occurrence=0,
            score=score,
        )
        for index, (score, relative, line, before) in enumerate(
            ranked[:max_candidates], start=1
        )
    )


def _resolve_local_js_module(task: StudentTask, source_file: str, module: str):
    base = task.resolve_target(source_file).parent / module
    variants = (
        base,
        base.with_suffix(".ts"),
        base.with_suffix(".tsx"),
        base.with_suffix(".js"),
        base.with_suffix(".jsx"),
        base / "index.ts",
        base / "index.js",
    )
    checkout = task.checkout.resolve()
    for path in variants:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(checkout)
        except ValueError:
            continue
        if resolved.is_file():
            return relative.as_posix(), resolved
    return None


def _extract_js_function_definition(content: str, symbol: str) -> str | None:
    pattern = re.compile(
        rf"(?:export\s+)?(?:async\s+)?function\s+{re.escape(symbol)}\s*\("
    )
    match = pattern.search(content)
    if match is None:
        return None
    start = match.start()
    opening = content.find("{", match.end())
    if opening < 0:
        return None
    depth = 0
    end = opening
    for index in range(opening, min(len(content), opening + 4_000)):
        char = content[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if depth != 0:
        return None
    source = content[start:end]
    if len(source) > 3_000 or len(source.splitlines()) > 60:
        return None
    return source


def fixed_supporting_symbol_contexts(
    task: StudentTask,
    candidates: tuple[ExactSpanCandidate, ...],
    *,
    max_contexts: int = 4,
) -> tuple[SupportingSymbolContext, ...]:
    """Resolve bounded local JS/TS callee contracts used by frozen candidates."""

    if type(max_contexts) is not int or max_contexts < 1:
        raise ValueError("max supporting symbol contexts must be positive")
    calls_by_file: dict[str, list[str]] = {}
    for candidate in candidates:
        calls = calls_by_file.setdefault(candidate.file, [])
        for symbol in _CALL_TOKEN.findall(candidate.before):
            if symbol not in calls:
                calls.append(symbol)
    contexts: list[SupportingSymbolContext] = []
    seen: set[tuple[str, str]] = set()
    for source_file in sorted(calls_by_file):
        content = task.resolve_target(source_file).read_text(
            encoding="utf-8", errors="replace"
        )
        calls = calls_by_file[source_file]
        imported: dict[str, tuple[str, str, Any]] = {}
        for match in _LOCAL_JS_IMPORT.finditer(content):
            imports: dict[str, str] = {}
            for item in match.group("names").split(","):
                parts = item.strip().split()
                if not parts:
                    continue
                if len(parts) == 3 and parts[1] == "as":
                    imports[parts[2]] = parts[0]
                else:
                    imports[parts[0]] = parts[0]
            resolved = _resolve_local_js_module(
                task, source_file, match.group("module")
            )
            if resolved is None:
                continue
            relative, path = resolved
            for local_name, source_name in imports.items():
                imported[local_name] = (source_name, relative, path)
        for local_name in calls:
            resolved_import = imported.get(local_name)
            if resolved_import is None:
                continue
            source_name, relative, path = resolved_import
            key = (relative, source_name)
            if key in seen:
                continue
            support_source = path.read_text(encoding="utf-8", errors="replace")
            definition = _extract_js_function_definition(support_source, source_name)
            if definition is None:
                continue
            seen.add(key)
            contexts.append(
                SupportingSymbolContext(
                    file=relative,
                    symbol=local_name,
                    source=definition,
                )
            )
            if len(contexts) >= max_contexts:
                return tuple(contexts)
    return tuple(contexts)


def fixed_repository_api_contexts(
    task: StudentTask,
    candidates: tuple[ExactSpanCandidate, ...],
    *,
    max_contexts: int = 4,
) -> tuple[SupportingSymbolContext, ...]:
    """Find bounded repository API exemplars aligned with the public issue."""

    if type(max_contexts) is not int or max_contexts < 1:
        raise ValueError("max repository API contexts must be positive")
    normalized_query = _normalized_code_tokens(task.instruction)
    candidate_methods = {
        method
        for candidate in candidates
        for method in _METHOD_CALL_TOKEN.findall(candidate.before)
    }
    ranked: list[tuple[int, str, int, str, str]] = []
    for path in sorted(task.checkout.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _CODE_SUFFIXES:
            continue
        relative = path.relative_to(task.checkout).as_posix()
        if ".git" in path.parts:
            continue
        try:
            if path.stat().st_size > 1_000_000:
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for index, line in enumerate(lines):
            for method in _METHOD_CALL_TOKEN.findall(line):
                method_tokens = _normalized_code_tokens(method)
                overlap = normalized_query.intersection(method_tokens)
                behavior_overlap = overlap.intersection(_API_BEHAVIOR_TOKENS)
                domain_overlap = overlap.intersection({"environment", "variable"})
                if (
                    len(overlap) < 2
                    or not behavior_overlap
                    or method in candidate_methods
                ):
                    continue
                start = max(0, index - 1)
                end = min(len(lines), index + 2)
                source = "\n".join(lines[start:end]).strip()
                if not source or len(source) > 600:
                    continue
                score = (
                    20_000 * len(behavior_overlap)
                    + 20_000 * len(domain_overlap)
                    + 5_000 * len(overlap)
                    + min(len(line), 200)
                )
                ranked.append((score, relative, index + 1, method, source))
    ranked.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    contexts: list[SupportingSymbolContext] = []
    seen: set[str] = set()
    for _score, relative, _line, method, source in ranked:
        if method in seen:
            continue
        seen.add(method)
        contexts.append(
            SupportingSymbolContext(file=relative, symbol=method, source=source)
        )
        if len(contexts) >= max_contexts:
            break
    return tuple(contexts)


_STATE_ASSIGNMENT = re.compile(
    r"\b(?P<object>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*"
    r"(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*="
)
_STATE_ASSIGNMENT_STATEMENT = re.compile(
    r"^\s*(?P<object>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*"
    r"(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"(?P<value>[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+)\s*;\s*$"
)
_STATE_VALUE_ASSIGNMENT = re.compile(
    r"\b(?P<object>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*"
    r"(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"(?P<value>[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+)\s*;"
)
_STATE_BEHAVIOR_TOKENS = frozenset(
    {
        "align",
        "default",
        "fallback",
        "flag",
        "left",
        "minus",
        "numeric",
        "parse",
        "right",
    }
)


def _bounded_enclosing_state_source(lines: list[str], index: int) -> str:
    """Return a bounded function-level excerpt around one state assignment."""

    start = max(0, index - 12)
    for candidate in range(index, max(-1, index - 40), -1):
        stripped = lines[candidate].strip()
        if not stripped or lines[candidate] != lines[candidate].lstrip():
            continue
        header = " ".join(
            line.strip() for line in lines[candidate : min(len(lines), candidate + 3)]
        )
        if "(" in header and not re.match(
            r"^(?:if|for|while|switch|catch)\b", stripped
        ):
            start = candidate
            break

    depth = 0
    saw_open = False
    end = min(len(lines), index + 4)
    for cursor in range(start, min(len(lines), start + 80)):
        depth += lines[cursor].count("{") - lines[cursor].count("}")
        saw_open = saw_open or "{" in lines[cursor]
        if saw_open and cursor >= index and depth <= 0:
            end = cursor + 1
            break
    source = "\n".join(lines[start:end]).strip()
    if len(source) <= 600:
        return source
    header = "\n".join(lines[start : min(index, start + 3)]).strip()
    body = "\n".join(lines[max(start, index - 4) : min(len(lines), index + 3)]).strip()
    return f"{header}\n...\n{body}"[:600].rstrip()


def fixed_causal_state_contexts(
    task: StudentTask,
    candidates: tuple[ExactSpanCandidate, ...],
    *,
    max_contexts: int = 4,
) -> tuple[SupportingSymbolContext, ...]:
    """Expose same-field producers/defaults without evaluator or patch evidence."""

    if type(max_contexts) is not int or max_contexts < 1:
        raise ValueError("max causal state contexts must be positive")
    query_tokens = _semantic_code_tokens(task.instruction) - _ISSUE_STOP_TOKENS
    candidate_fields: dict[str, set[str]] = {}
    candidate_lines: dict[str, set[int]] = {}
    for candidate in candidates:
        candidate_lines.setdefault(candidate.file, set()).add(candidate.line)
        fields = candidate_fields.setdefault(candidate.file, set())
        fields.update(
            match.group("field")
            for match in _STATE_ASSIGNMENT.finditer(candidate.before)
        )
    ranked: list[tuple[int, str, int, str, str]] = []
    for relative, fields in sorted(candidate_fields.items()):
        if not fields:
            continue
        lines = (
            task.resolve_target(relative)
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
        )
        for index, line in enumerate(lines):
            match = _STATE_ASSIGNMENT.search(line)
            if match is None or match.group("field") not in fields:
                continue
            if index + 1 in candidate_lines.get(relative, set()):
                continue
            source = _bounded_enclosing_state_source(lines, index)
            source_tokens = _semantic_code_tokens(source)
            if "flags" in _code_tokens(source):
                source_tokens.add("flag")
            overlap = query_tokens.intersection(source_tokens).intersection(
                _STATE_BEHAVIOR_TOKENS
            )
            score = (
                20_000 * len(overlap)
                + 10_000 * bool({"flag", "parse"}.intersection(source_tokens))
                + 5_000 * bool({"left", "right"}.intersection(source_tokens))
                - min(len(source), 600)
            )
            ranked.append(
                (
                    score,
                    relative,
                    index + 1,
                    f"state:{match.group('field')}",
                    source,
                )
            )
    ranked.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    contexts: list[SupportingSymbolContext] = []
    seen: set[str] = set()
    for _score, relative, line, symbol, source in ranked:
        if source in seen:
            continue
        seen.add(source)
        contexts.append(
            SupportingSymbolContext(
                file=relative,
                symbol=f"{symbol}@{line}",
                source=source,
            )
        )
        if len(contexts) >= max_contexts:
            break
    return tuple(contexts)


def fixed_repository_state_value_contexts(
    task: StudentTask,
    candidates: tuple[ExactSpanCandidate, ...],
    *,
    max_contexts: int = 2,
) -> tuple[SupportingSymbolContext, ...]:
    """Find exact neutral/default enum values for a candidate state field."""

    if type(max_contexts) is not int or max_contexts < 1:
        raise ValueError("max repository state value contexts must be positive")
    fields = {
        match.group("field")
        for candidate in candidates
        for match in _STATE_ASSIGNMENT.finditer(candidate.before)
    }
    if not fields:
        return ()
    targets = task.allowed_targets or StudentAdapter._discover_targets(task)
    ranked: list[tuple[int, str, int, str, str, str]] = []
    for relative in targets:
        path = task.resolve_target(relative)
        try:
            if path.stat().st_size > 1_000_000:
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for index, line in enumerate(lines, start=1):
            for field in fields:
                pattern = rf"\b{re.escape(field)}::([A-Za-z_][A-Za-z0-9_]*)"
                for value in re.findall(pattern, line):
                    if value not in {"none", "default", "unset", "unknown"}:
                        continue
                    source = line.strip()
                    if not source or len(source) > 300:
                        continue
                    score = 10_000 * (value == "none") - len(source)
                    ranked.append((score, relative, index, field, value, source))
    ranked.sort(key=lambda row: (-row[0], row[1], row[2], row[3], row[4]))
    contexts: list[SupportingSymbolContext] = []
    seen: set[tuple[str, str]] = set()
    for _score, relative, _line, field, value, source in ranked:
        key = (field, value)
        if key in seen:
            continue
        seen.add(key)
        contexts.append(
            SupportingSymbolContext(
                file=relative,
                symbol=f"state-value:{field}:{value}",
                source=source,
            )
        )
        if len(contexts) >= max_contexts:
            break
    return tuple(contexts)


def fixed_editable_candidate_contexts(
    task: StudentTask,
    candidates: tuple[ExactSpanCandidate, ...],
) -> tuple[SupportingSymbolContext, ...]:
    """Expose bounded control flow around editable spans without widening selectors."""

    contexts: list[SupportingSymbolContext] = []
    for candidate in candidates:
        try:
            lines = (
                task.resolve_target(candidate.file)
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()
            )
        except OSError:
            continue
        index = candidate.line - 1
        if not 0 <= index < len(lines):
            continue
        source = "\n".join(
            lines[max(0, index - 8) : min(len(lines), index + 5)]
        ).strip()
        if not source or candidate.before.strip() not in source:
            continue
        contexts.append(
            SupportingSymbolContext(
                file=candidate.file,
                symbol=f"editable-context:{candidate.candidate_id}",
                source=source,
            )
        )
    return tuple(contexts)


def fixed_typed_state_actions(
    task: StudentTask,
    candidates: tuple[ExactSpanCandidate, ...],
) -> tuple[dict[str, str], ...]:
    """Enumerate model-selectable state actions from source-only neutral values."""

    neutral_by_field: dict[str, list[str]] = {}
    for context in fixed_repository_state_value_contexts(
        task, candidates, max_contexts=4
    ):
        parts = context.symbol.split(":", 2)
        if len(parts) != 3 or parts[0] != "state-value":
            continue
        field, value = parts[1:]
        namespace_match = re.search(
            rf"\b(?P<namespace>[A-Za-z_][A-Za-z0-9_]*)::{re.escape(value)}\b",
            context.source,
        )
        if namespace_match is None:
            continue
        neutral_by_field.setdefault(field, []).append(
            f"{namespace_match.group('namespace')}::{value}"
        )
    causal_contexts = fixed_causal_state_contexts(task, candidates, max_contexts=4)
    protected_values = _issue_triggered_state_values(task, causal_contexts)
    # C2: derive transient state values with a task-level scan of every
    # candidate file instead of the candidate-count-sensitive causal context
    # set.  fixed_causal_state_contexts skips candidate lines, so once the
    # candidate set covers all assignment sites (e.g. a bounded 8-span display)
    # it returns nothing and the guard-neutral-or-transient-default action
    # silently disappears.  Scanning the files is stable and still honors the
    # same fill/width/zero guard and issue-triggered protection rules.
    transient_by_field: dict[str, list[str]] = {}
    scanned_files: set[str] = set()
    for candidate in candidates:
        if candidate.file in scanned_files:
            continue
        scanned_files.add(candidate.file)
        try:
            lines = task.resolve_target(candidate.file).read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            continue
        for index, line in enumerate(lines):
            match = _STATE_VALUE_ASSIGNMENT.search(line)
            if match is None:
                continue
            enclosing = _bounded_enclosing_state_source(lines, index)
            if not {"fill", "width", "zero"}.intersection(
                _code_tokens(enclosing)
            ):
                continue
            if match.group("value") in protected_values.get(
                match.group("field"), set()
            ):
                continue
            transient_by_field.setdefault(match.group("field"), []).append(
                match.group("value")
            )

    actions: list[dict[str, str]] = []
    for candidate in candidates:
        assignment = _STATE_ASSIGNMENT_STATEMENT.fullmatch(candidate.before)
        if assignment is None:
            continue
        field = assignment.group("field")
        original_value = assignment.group("value")
        for state_value in dict.fromkeys(neutral_by_field.get(field, [])):
            for operation in (
                "preserve-existing-state",
                "set-neutral-state",
                "guard-neutral-default",
            ):
                actions.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "operation": operation,
                        "state_value": state_value,
                    }
                )
        if neutral_by_field.get(field):
            for transient_value in dict.fromkeys(transient_by_field.get(field, [])):
                if transient_value == original_value:
                    continue
                actions.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "operation": "guard-neutral-or-transient-default",
                        "state_value": transient_value,
                    }
                )
    return tuple(actions)


def _issue_triggered_state_values(
    task: StudentTask,
    contexts: tuple[SupportingSymbolContext, ...],
) -> dict[str, set[str]]:
    """Protect parser-written values tied to an explicit issue trigger."""

    issue_tokens = _semantic_code_tokens(task.instruction)
    trigger_literals = {
        literal
        for token, literal in {
            "minus": "-",
            "plus": "+",
            "zero": "0",
            "space": " ",
        }.items()
        if token in issue_tokens
    }
    if not trigger_literals:
        return {}
    protected: dict[str, set[str]] = {}
    case_pattern = re.compile(r"\bcase\s+['\"](?P<literal>.)['\"]\s*:")
    for context in contexts:
        for match in _STATE_VALUE_ASSIGNMENT.finditer(context.source):
            prefix = context.source[max(0, match.start() - 320) : match.start()]
            cases = list(case_pattern.finditer(prefix))
            if not cases or cases[-1].group("literal") not in trigger_literals:
                continue
            protected.setdefault(match.group("field"), set()).add(match.group("value"))
    return protected


def fixed_typed_collection_actions(
    task: StudentTask,
    candidates: tuple[ExactSpanCandidate, ...],
) -> tuple[dict[str, str], ...]:
    """Derive bounded iteration filters from issue intent and repository APIs."""

    issue_tokens = _semantic_code_tokens(task.instruction)
    behaviors = issue_tokens.intersection({"hide", "exclude", "omit", "skip"})
    if not behaviors:
        return ()
    targets = task.allowed_targets or StudentAdapter._discover_targets(task)
    ranked_methods: dict[str, int] = {}
    for relative in targets:
        path = task.resolve_target(relative)
        try:
            if path.stat().st_size > 1_000_000:
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _COLLECTION_PREDICATE_METHOD.finditer(content):
            method = match.group("method")
            method_tokens = _semantic_code_tokens(method)
            behavior_overlap = behaviors.intersection(method_tokens)
            if not behavior_overlap:
                continue
            exact_behavior_setter = any(
                method == f"is_{behavior}_set" for behavior in behavior_overlap
            )
            score = (
                50_000 * exact_behavior_setter
                + 5_000 * len(issue_tokens.intersection(method_tokens))
                - 100 * len(method_tokens - issue_tokens)
                - len(method)
            )
            ranked_methods[method] = max(score, ranked_methods.get(method, score))
    if not ranked_methods:
        return ()
    ordered = sorted(ranked_methods.items(), key=lambda row: (-row[1], row[0]))
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return ()
    predicate_method = ordered[0][0]
    actions: list[dict[str, str]] = []
    for candidate in candidates:
        if _ENUMERATED_LOOP.search(candidate.before) is None:
            continue
        actions.append(
            {
                "candidate_id": candidate.candidate_id,
                "operation": "filter-iteration-item",
                "predicate_method": predicate_method,
            }
        )
    return tuple(actions)


def fixed_typed_endpoint_actions(
    task: StudentTask,
    candidates: tuple[ExactSpanCandidate, ...],
) -> tuple[dict[str, str], ...]:
    """Enumerate bounded bracket-aware endpoint parsing alternatives."""

    issue = task.instruction
    issue_tokens = _semantic_code_tokens(issue)
    if "ipv6" not in issue_tokens or re.search(r"\[[0-9A-Fa-f:]+\]:\d+", issue) is None:
        return ()
    actions: list[dict[str, str]] = []
    for candidate in candidates:
        if _ENDPOINT_SPLIT_ASSIGNMENT.fullmatch(candidate.before) is None:
            continue
        for operation in (
            "split-endpoint-at-last-colon",
            "parse-bracketed-endpoint",
            "validate-and-parse-endpoint",
        ):
            actions.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "operation": operation,
                }
            )
    return tuple(actions)


def _php_literal_array(value: str) -> tuple[tuple[str, str], ...] | None:
    rows: list[tuple[str, str]] = []
    position = 0
    while position < len(value):
        match = _PHP_ARRAY_LITERAL.match(value, position)
        if match is None:
            return None
        rows.append((match.group("quote"), match.group("value")))
        position = match.end()
    return tuple(rows)


def _materialize_locale_separator_action(
    task: StudentTask, candidate: ExactSpanCandidate
) -> tuple[str, str] | None:
    """Render a proven three-slot locale separator cycle without model-authored code."""

    issue_tokens = _semantic_code_tokens(task.instruction)
    if not {"decimal", "separator"}.issubset(issue_tokens) or not (
        "thousand" in issue_tokens or "thousands" in issue_tokens
    ):
        return None
    source = task.resolve_target(candidate.file).read_text(
        encoding="utf-8", errors="replace"
    )
    call = _LOCALE_REPLACE_CALL.fullmatch(candidate.before.strip())
    candidate_offset = source.find(candidate.before)
    function_matches = tuple(_PHP_NAMED_FUNCTION.finditer(source))
    owning_index = next(
        (
            index
            for index in range(len(function_matches) - 1, -1, -1)
            if function_matches[index].start() <= candidate_offset
        ),
        None,
    )
    if call is None or candidate_offset < 0 or owning_index is None:
        return None
    function_start = function_matches[owning_index].start()
    function_end = (
        function_matches[owning_index + 1].start()
        if owning_index + 1 < len(function_matches)
        else len(source)
    )
    function_source = source[function_start:function_end]
    guard = _LOCALE_SEPARATOR_GUARD.search(function_source)
    if guard is None:
        return None
    thousands = guard.group("thousands")
    decimal = guard.group("decimal")
    if (
        "thousand" not in thousands.casefold()
        or "decimal" not in decimal.casefold()
        or re.search(rf"\${re.escape(thousands)}\s*=\s*[^;]+;", function_source) is None
        or re.search(rf"\${re.escape(decimal)}\s*=\s*[^;]+;", function_source) is None
    ):
        return None
    search = _php_literal_array(call.group("search"))
    replace = _php_literal_array(call.group("replace"))
    if search is None or replace is None or len(search) != 3 or len(replace) != 3:
        return None
    decimal_default = guard.group("ddefault")
    thousands_default = guard.group("tdefault")
    sentinel = search[2][1]
    if (
        not sentinel
        or tuple(value for _quote, value in search)
        != (decimal_default, thousands_default, sentinel)
        or tuple(value for _quote, value in replace)
        != (sentinel, decimal_default, thousands_default)
    ):
        return None
    after = (
        f"{call.group('prefix')}"
        + ", ".join(f"{quote}{value}{quote}" for quote, value in search)
        + "], ["
        + f"{replace[0][0]}{sentinel}{replace[0][0]}, "
        + f"${thousands}, ${decimal}]"
        + call.group("suffix")
    )
    return candidate.before, after


def fixed_typed_locale_separator_actions(
    task: StudentTask,
    candidates: tuple[ExactSpanCandidate, ...],
) -> tuple[dict[str, str], ...]:
    """Expose only statically proven locale separator role substitutions."""

    return tuple(
        {
            "candidate_id": candidate.candidate_id,
            "operation": "substitute-locale-separator-roles",
        }
        for candidate in candidates
        if _materialize_locale_separator_action(task, candidate) is not None
    )


def _issue_verbatim_replacements(
    task: StudentTask,
    candidates: tuple[ExactSpanCandidate, ...],
) -> dict[str, str]:
    """Bind public issue before/after blocks to unique renderer-owned spans."""

    blocks = [
        match.group(1).strip() for match in _ISSUE_CODE_FENCE.finditer(task.instruction)
    ]
    replacements: dict[str, list[str]] = {}
    for before_block, after_block in zip(blocks, blocks[1:]):
        if (
            not before_block
            or not after_block
            or before_block == after_block
            or len(after_block) > _MAX_EXACT_SPAN_CHARS
            or len(after_block.splitlines()) > _MAX_EXACT_SPAN_LINES
        ):
            continue
        for candidate in candidates:
            if candidate.before.strip() != before_block:
                continue
            source = task.resolve_target(candidate.file).read_text(
                encoding="utf-8", errors="replace"
            )
            if source.count(candidate.before) != 1:
                continue
            replacements.setdefault(candidate.candidate_id, []).append(after_block)
    return {
        candidate_id: rows[0]
        for candidate_id, rows in replacements.items()
        if len(set(rows)) == 1
    }


def fixed_typed_issue_verbatim_actions(
    task: StudentTask,
    candidates: tuple[ExactSpanCandidate, ...],
) -> tuple[dict[str, str], ...]:
    """Expose short choices for unambiguous public issue replacement pairs."""

    replacements = _issue_verbatim_replacements(task, candidates)
    return tuple(
        {
            "candidate_id": candidate.candidate_id,
            "operation": "apply-issue-verbatim-replacement",
        }
        for candidate in candidates
        if candidate.candidate_id in replacements
    )


def fixed_typed_actions(
    task: StudentTask,
    candidates: tuple[ExactSpanCandidate, ...],
) -> tuple[dict[str, str], ...]:
    """Combine deterministic typed edit classes without widening selectors."""

    return (
        *fixed_typed_issue_verbatim_actions(task, candidates),
        *fixed_typed_state_actions(task, candidates),
        *fixed_typed_collection_actions(task, candidates),
        *fixed_typed_endpoint_actions(task, candidates),
        *fixed_typed_locale_separator_actions(task, candidates),
    )


def _materialize_collection_action(
    before: str, action: dict[str, str]
) -> tuple[str, str] | None:
    """Render one source-derived collection action inside an exact candidate."""

    match = _ENUMERATED_LOOP.search(before)
    method = action.get("predicate_method")
    if match is None or not isinstance(method, str):
        return None
    if _CODE_TOKEN.fullmatch(method) is None:
        return None
    iterator = match.group("iterator").rstrip()
    item = match.group("item")
    selector = match.group(0)
    replacement = (
        f"for ({match.group('index')}, {item}) in {iterator}"
        f".filter(|{item}| !{item}.{method}()).enumerate() {{"
    )
    return selector, replacement


def _materialize_endpoint_action(
    before: str, action: dict[str, str]
) -> tuple[str, str] | None:
    """Render a source-derived endpoint split action without model-authored code."""

    match = _ENDPOINT_SPLIT_ASSIGNMENT.fullmatch(before)
    if match is None:
        return None
    bind = match.group("bind")
    port = match.group("port")
    endpoint = match.group("endpoint")
    operation = action.get("operation")
    if operation == "split-endpoint-at-last-colon":
        after = f"{bind}, {port} = {endpoint}.split(/:(?=[^:]+\\z)/)"
    elif operation == "parse-bracketed-endpoint":
        after = (
            f"{bind}, {port} = {endpoint}.match(/\\A\\[(.*)\\]:(\\d+)\\z/)"
            f"&.captures || {endpoint}.split(':')"
        )
    elif operation == "validate-and-parse-endpoint":
        after = (
            f"match = {endpoint}.match(/\\A(?:\\[([0-9A-Fa-f:]+)\\]|"
            f"([A-Za-z0-9][A-Za-z0-9.-]*)):(\\d+)\\z/)\n"
            f'        raise Fluent::ConfigError, "Invalid rpc_endpoint: '
            f'#{{{endpoint}}}" unless match\n'
            f"        {bind}, {port} = match[1] || match[2], match[3]"
        )
    else:
        return None
    return before, after


def _typed_collection_selectors(
    task: StudentTask,
    candidates: tuple[ExactSpanCandidate, ...],
) -> dict[tuple[str, str], ExactSpanCandidate]:
    """Map renderer-owned small selectors back to their frozen candidates."""

    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selectors: dict[tuple[str, str], ExactSpanCandidate] = {}
    for action in fixed_typed_collection_actions(task, candidates):
        candidate = candidate_by_id[action["candidate_id"]]
        rendered = _materialize_collection_action(candidate.before, action)
        if rendered is None:
            continue
        before, _after = rendered
        selectors[(candidate.file, before)] = candidate
    return selectors


def _framework_owned_span_intent(
    task: StudentTask, revision: LoopRevision
) -> dict[str, str]:
    marker = "## Shared diagnosis and localization (read-only)"
    if marker in revision.skill_text:
        suffix = revision.skill_text.split(marker, 1)[1].lstrip()
        try:
            pinned, _end = json.JSONDecoder().raw_decode(suffix)
        except json.JSONDecodeError:
            pinned = None
        diagnosis = pinned.get("diagnosis") if isinstance(pinned, dict) else None
        if isinstance(diagnosis, dict):
            fields = {
                key: diagnosis.get(key)
                for key in ("defect", "trigger", "desired_boundary")
            }
            if all(
                isinstance(value, str) and value.strip() for value in fields.values()
            ):
                return {key: str(value) for key, value in fields.items()}
    first_line = next(
        (line.strip() for line in task.instruction.splitlines() if line.strip()),
        "Feedback task requires a source repair.",
    )
    return {
        "defect": first_line[:240],
        "trigger": "behavior described by the frozen feedback task",
        "desired_boundary": "satisfy the task while preserving unrelated behavior",
    }


def _editable_span_task(task: StudentTask, revision: LoopRevision) -> StudentTask:
    """Bind editable candidate harvest without shrinking read-only evidence scope."""

    marker = "## Shared diagnosis and localization (read-only)"
    if marker not in revision.skill_text:
        return task
    suffix = revision.skill_text.split(marker, 1)[1].lstrip()
    try:
        pinned, _end = json.JSONDecoder().raw_decode(suffix)
    except json.JSONDecodeError as exc:
        raise ContractError("shared span context is malformed") from exc
    files = pinned.get("target_files") if isinstance(pinned, dict) else None
    if (
        not isinstance(files, list)
        or not 1 <= len(files) <= 2
        or any(not isinstance(file, str) for file in files)
        or len(set(files)) != len(files)
        or any(file not in task.allowed_targets for file in files)
    ):
        raise ContractError("shared span editable scope is invalid")
    targets = tuple(files)
    if targets == task.allowed_targets:
        return task
    return StudentTask.create(
        task_id=task.task_id,
        checkout=task.checkout,
        instruction=task.instruction,
        allowed_targets=list(targets),
        cohort=task.cohort,
    )


def _frozen_causal_candidates(
    task: StudentTask,
    revision: LoopRevision,
    *,
    max_candidates: int = _MAX_EXACT_SPAN_CANDIDATES,
) -> tuple[ExactSpanCandidate, ...]:
    """Emit bounded lexical and control-flow roles per frozen editable file."""

    editable = _editable_span_task(task, revision)
    if editable.allowed_targets == task.allowed_targets:
        # C2: a single source of truth for both prompt display and typed-action
        # derivation.  Every schema exposes a bounded role-diverse candidate
        # set (symmetric with the pinned/taught path), so a wrong top-1 span
        # can never starve the student into an empty-patch abstention.
        candidates = fixed_exact_span_candidates(
            editable, max_candidates=max_candidates
        )
        emitted = min(max_candidates, 4 * len(editable.allowed_targets), 8)
        return candidates[:emitted]
    emitted_limit = min(max_candidates, 4 * len(editable.allowed_targets), 8)
    selected: list[ExactSpanCandidate] = []
    ranked_by_file: dict[str, tuple[ExactSpanCandidate, ...]] = {}
    editable_sources = {
        relative: editable.resolve_target(relative).read_text(
            encoding="utf-8", errors="replace"
        )
        for relative in editable.allowed_targets[:2]
    }
    for relative in editable.allowed_targets[:2]:
        source = editable_sources[relative]
        defined_types = set(
            re.findall(
                r"\b(?:class|module|struct|interface|trait)\s+"
                r"([A-Z][A-Za-z0-9_]*)",
                source,
            )
        )
        constructed_elsewhere = {
            match
            for other, other_source in editable_sources.items()
            if other != relative
            for match in re.findall(
                r"\b(?:new\s+)?(?:[A-Z][A-Za-z0-9_]*::)*"
                r"([A-Z][A-Za-z0-9_]*)\s*(?:\.new\s*\(|\()",
                other_source,
            )
        }
        ranking_instruction = editable.instruction
        if defined_types.intersection(constructed_elsewhere):
            ranking_instruction += "\nRepository constructor call edge: ->initialize()"
        per_file = StudentTask.create(
            task_id=editable.task_id,
            checkout=editable.checkout,
            instruction=ranking_instruction,
            allowed_targets=[relative],
            cohort=editable.cohort,
        )
        ranked = fixed_exact_span_candidates(
            per_file, max_candidates=max(max_candidates, 256)
        )
        ranked_by_file[relative] = ranked
        if ranked:
            selected.append(ranked[0])
        if revision.source_round >= 74:
            locale_role_candidate = next(
                (
                    candidate
                    for candidate in ranked
                    if _materialize_locale_separator_action(editable, candidate)
                    is not None
                ),
                None,
            )
            if (
                locale_role_candidate is not None
                and locale_role_candidate not in selected
            ):
                selected.append(locale_role_candidate)

        family = _issue_api_method_family(
            editable.instruction,
            {
                match.group(1)
                for line in source.splitlines()
                if (match := _FUNCTION_START.match(line.strip())) is not None
            },
        )
        if len(family) > 1:
            active_name: str | None = None
            function_by_line: dict[int, str | None] = {}
            for line_number, line in enumerate(source.splitlines(), start=1):
                if match := _FUNCTION_START.match(line.strip()):
                    active_name = match.group(1).casefold()
                function_by_line[line_number] = active_name
            for member in family:
                if len(selected) >= emitted_limit:
                    break
                if any(
                    row.file == relative and function_by_line.get(row.line) == member
                    for row in selected
                ):
                    continue
                companion = next(
                    (
                        row
                        for row in ranked
                        if row not in selected
                        and function_by_line.get(row.line) == member
                        and not _DECLARATION_LINE.match(row.before.lstrip())
                    ),
                    None,
                )
                if companion is not None:
                    selected.append(companion)

    for role in _CANDIDATE_ROLE_ORDER:
        for relative in editable.allowed_targets[:2]:
            if len(selected) >= emitted_limit:
                break
            if any(
                row.file == relative
                and role in _candidate_control_flow_roles(row.before)
                for row in selected
            ):
                continue
            ranked = ranked_by_file.get(relative, ())
            candidate = next(
                (
                    row
                    for row in ranked
                    if row not in selected
                    and role in _candidate_control_flow_roles(row.before)
                ),
                None,
            )
            if candidate is not None:
                selected.append(candidate)

    for relative in editable.allowed_targets[:2]:
        for candidate in ranked_by_file.get(relative, ()):
            if len(selected) >= emitted_limit:
                break
            if candidate not in selected:
                selected.append(candidate)

    return tuple(
        ExactSpanCandidate(
            candidate_id=f"span-{index:03d}",
            file=candidate.file,
            before=candidate.before,
            line=candidate.line,
            occurrence=candidate.occurrence,
            score=candidate.score,
        )
        for index, candidate in enumerate(selected[:emitted_limit], start=1)
    )


def _canonicalize_operation_only_output(
    raw: str, task: StudentTask, revision: LoopRevision
) -> tuple[str, bool]:
    """Merge a small-model edit list with framework-owned diagnosis fields."""

    candidate = raw.strip()
    fenced = _JSON_FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return raw, False
        candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return raw, False
    if not isinstance(data, dict):
        return raw, False
    if set(data) == {"schema_version", "actions"}:
        if data.get("schema_version") != 1 or not isinstance(data.get("actions"), list):
            return raw, False
        candidates = _frozen_causal_candidates(task, revision)
        catalog = fixed_typed_actions(task, candidates)
        by_payload = {canonical_json(row): row for row in catalog}
        candidate_by_id = {
            candidate.candidate_id: candidate for candidate in candidates
        }
        edits: list[dict[str, str]] = []
        for action in data["actions"]:
            if not isinstance(action, dict) or not {
                "candidate_id",
                "operation",
            }.issubset(action):
                return raw, False
            if not all(isinstance(action[key], str) for key in action):
                return raw, False
            if canonical_json(action) not in by_payload:
                return raw, False
            span = candidate_by_id[action["candidate_id"]]
            if action["operation"] == "apply-issue-verbatim-replacement":
                replacement = _issue_verbatim_replacements(task, candidates).get(
                    span.candidate_id
                )
                if replacement is None:
                    return raw, False
                edits.append(
                    {
                        "file": span.file,
                        "before": span.before,
                        "after": replacement,
                    }
                )
                continue
            if action["operation"] == "filter-iteration-item":
                rendered = _materialize_collection_action(span.before, action)
                if rendered is None:
                    return raw, False
                before, after = rendered
                edits.append({"file": span.file, "before": before, "after": after})
                continue
            if action["operation"] in {
                "split-endpoint-at-last-colon",
                "parse-bracketed-endpoint",
                "validate-and-parse-endpoint",
            }:
                rendered = _materialize_endpoint_action(span.before, action)
                if rendered is None:
                    return raw, False
                before, after = rendered
                edits.append({"file": span.file, "before": before, "after": after})
                continue
            if action["operation"] == "substitute-locale-separator-roles":
                rendered = _materialize_locale_separator_action(task, span)
                if rendered is None:
                    return raw, False
                before, after = rendered
                edits.append({"file": span.file, "before": before, "after": after})
                continue
            assignment = _STATE_ASSIGNMENT_STATEMENT.fullmatch(span.before)
            if assignment is None:
                return raw, False
            state = f"{assignment.group('object')}.{assignment.group('field')}"
            if action["operation"] == "preserve-existing-state":
                after = ""
            elif action["operation"] == "set-neutral-state":
                after = f"{state} = {action['state_value']};"
            elif action["operation"] == "guard-neutral-or-transient-default":
                neutral = next(
                    (
                        row["state_value"]
                        for row in catalog
                        if row["candidate_id"] == action["candidate_id"]
                        and row["operation"] == "guard-neutral-default"
                    ),
                    None,
                )
                if neutral is None:
                    return raw, False
                after = (
                    f"if ({state} == {neutral} || "
                    f"{state} == {action['state_value']})\n"
                    f"  {span.before.strip()}"
                )
            else:
                after = (
                    f"if ({state} == {action['state_value']})\n  {span.before.strip()}"
                )
            edits.append({"file": span.file, "before": span.before, "after": after})
        data = {"schema_version": 1, "edits": edits}
    if set(data) != {"schema_version", "edits"}:
        return raw, False
    if data.get("schema_version") != 1 or not isinstance(data.get("edits"), list):
        return raw, False
    edits = data["edits"]
    if not 1 <= len(edits) <= 4:
        return raw, False
    grouped: dict[str, list[dict[str, str]]] = {}
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"file", "before", "after"}:
            return raw, False
        if not all(isinstance(edit[key], str) for key in edit):
            return raw, False
        grouped.setdefault(edit["file"], []).append(
            {"before": edit["before"], "after": edit["after"]}
        )
    if len(grouped) > 2:
        return raw, False
    intent = _framework_owned_span_intent(task, revision)
    diagnosis = "Framework-owned diagnosis; Student supplied operation-only edits."
    bundle = {
        "schema_version": 1,
        "plans": [
            {
                "schema_version": 1,
                "file": file,
                "intent": intent,
                "operations": operations,
            }
            for file, operations in grouped.items()
        ],
        "diagnostic": diagnosis,
    }
    return canonical_json(bundle), True


def _canonicalize_semantic_recipe_output(
    raw: str,
    candidates: tuple[ExactSpanCandidate, ...],
    *,
    task: StudentTask | None = None,
    recipe_output_chars: int = _MAX_EXACT_SPAN_CHARS,
) -> tuple[str, bool]:
    """Project small-model candidate recipes into renderer-owned exact edits."""

    candidate = raw.strip()
    fenced = _JSON_FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return raw, False
        candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return raw, False
    if (
        not isinstance(data, dict)
        or set(data) != {"schema_version", "recipes"}
        or data.get("schema_version") != 1
        or not isinstance(data.get("recipes"), list)
        or not 1 <= len(data["recipes"]) <= 3
    ):
        return raw, False
    by_id = {row.candidate_id: row for row in candidates}
    editable_context_by_id = (
        {
            context.symbol.removeprefix("editable-context:"): context.source
            for context in fixed_editable_candidate_contexts(task, candidates)
            if context.symbol.startswith("editable-context:")
        }
        if task is not None
        else {}
    )
    selected: set[str] = set()
    edits: list[dict[str, str]] = []
    for recipe in data["recipes"]:
        if not isinstance(recipe, dict) or set(recipe) != {"candidate_id", "after"}:
            return raw, False
        candidate_id = recipe.get("candidate_id")
        after = recipe.get("after")
        if (
            not isinstance(candidate_id, str)
            or candidate_id not in by_id
            or candidate_id in selected
            or not isinstance(after, str)
            or not after.strip()
            or len(after) > recipe_output_chars
            or len(after.splitlines()) > _MAX_EXACT_SPAN_LINES
        ):
            return raw, False
        span = by_id[candidate_id]
        if "".join(after.split()) == "".join(span.before.split()):
            return raw, False
        if task is not None and _php_after_has_unbound_variable(
            after,
            editable_context_by_id.get(candidate_id, span.before),
        ):
            return raw, False
        selected.add(candidate_id)
        edits.append({"file": span.file, "before": span.before, "after": after})
    return canonical_json({"schema_version": 1, "edits": edits}), True


def _php_after_has_unbound_variable(after: str, enclosing_source: str) -> bool:
    """Reject PHP recipes that import locals unavailable at the edit site."""

    used = set(re.findall(r"\$[A-Za-z_][A-Za-z0-9_]*", after))
    if not used:
        return False
    visible = set(re.findall(r"\$[A-Za-z_][A-Za-z0-9_]*", enclosing_source)) | {"$this"}
    introduced = set(
        re.findall(
            r"(?<![!<>=])(?P<name>\$[A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)",
            after,
        )
    )
    introduced.update(
        re.findall(
            r"\b(?:as|catch\s*\([^)]*)\s+(?P<name>\$[A-Za-z_][A-Za-z0-9_]*)",
            after,
        )
    )
    return bool(used - visible - introduced)


def _typed_state_action_payload(raw: str) -> dict[str, Any] | None:
    """Parse only the strict model-facing typed-action envelope."""

    candidate = raw.strip()
    fenced = _JSON_FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(data, dict)
        or set(data) != {"schema_version", "actions"}
        or data.get("schema_version") != 1
        or not isinstance(data.get("actions"), list)
        or len(data["actions"]) != 1
    ):
        return None
    action = data["actions"][0]
    if (
        not isinstance(action, dict)
        or not {"candidate_id", "operation"}.issubset(action)
        or any(not isinstance(action.get(key), str) for key in action)
    ):
        return None
    return data


def _typed_state_action_reason(
    raw: str,
    revision: LoopRevision,
    catalog: tuple[dict[str, str], ...],
) -> str | None:
    """Validate one catalog action plus machine-readable inactive Skill preference."""

    payload = _typed_state_action_payload(raw)
    if payload is None:
        return (
            "typed state action response requires exactly one copied ALLOWED TYPED "
            "ACTION object"
        )
    action = payload["actions"][0]
    allowed = {canonical_json(row) for row in catalog}
    if canonical_json(action) not in allowed:
        return "typed state action must be copied byte-for-byte from the catalog"
    teaching = revision.skill_text.casefold()
    transient_preference = (
        "width/fill behavior" in teaching and "every writer" in teaching
    )
    if (
        transient_preference
        and any(
            row["operation"] == "guard-neutral-or-transient-default" for row in catalog
        )
        and action["operation"] != "guard-neutral-or-transient-default"
    ):
        return (
            "inactive Skill action preference requires "
            "guard-neutral-or-transient-default for a source-derived transient state"
        )
    guard_preference = "guard-neutral-default is preferred" in teaching
    if (
        guard_preference
        and any(row["operation"] == "guard-neutral-default" for row in catalog)
        and action["operation"] != "guard-neutral-default"
    ):
        return (
            "inactive Skill action preference requires guard-neutral-default "
            "instead of deletion or an unconditional assignment"
        )
    preserve_preference = (
        "preserves parser-derived state" in teaching
        and "prefer it over neutral assignment" in teaching
    )
    if (
        preserve_preference
        and any(row["operation"] == "preserve-existing-state" for row in catalog)
        and action["operation"] != "preserve-existing-state"
    ):
        return (
            "inactive Skill action preference requires preserve-existing-state "
            "instead of a neutral or guarded assignment"
        )
    endpoint_preference = "bracket-aware endpoint parsing" in teaching
    if (
        endpoint_preference
        and any(row["operation"] == "parse-bracketed-endpoint" for row in catalog)
        and action["operation"] != "parse-bracketed-endpoint"
    ):
        return (
            "inactive Skill action preference requires parse-bracketed-endpoint "
            "for a bracketed host:port input"
        )
    validation_preference = (
        "validation-preserving endpoint parsing" in teaching
        and "preserve invalid-input rejection" in teaching
    )
    if (
        validation_preference
        and any(row["operation"] == "validate-and-parse-endpoint" for row in catalog)
        and action["operation"] != "validate-and-parse-endpoint"
    ):
        return (
            "inactive Skill action preference requires "
            "validate-and-parse-endpoint to preserve invalid-input rejection"
        )
    return None


def _canonicalize_span_bundle_output(raw: str) -> tuple[str, bool]:
    """Fill a file-plan schema version when the root v1 bundle proves it."""

    candidate = raw.strip()
    fenced = _JSON_FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return raw, False
        candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return raw, False
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        return raw, False
    plans = data.get("plans")
    if not isinstance(plans, list) or not plans:
        return raw, False
    changed = False
    normalized: list[Any] = []
    allowed = {"schema_version", "file", "intent", "operations", "diagnostic"}
    required = {"file", "intent", "operations"}
    for plan in plans:
        if not isinstance(plan, dict):
            return raw, False
        if "schema_version" not in plan:
            if not required.issubset(plan) or not set(plan).issubset(allowed):
                return raw, False
            plan = {"schema_version": 1, **plan}
            changed = True
        normalized.append(plan)
    if not changed:
        return raw, False
    data["plans"] = normalized
    return canonical_json(data), True


def parse_span_plan_output(raw: str) -> SpanPlan:
    """Extract exactly one bounded exact-span plan from model output."""

    candidate = raw.strip()
    fenced = _JSON_FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ContractError("student output contains no span plan object")
        candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ContractError("student span plan is malformed JSON") from exc
    return SpanPlan.from_dict(data)


def parse_span_bundle_output(raw: str) -> SpanBundlePlan:
    """Parse a one/two-file bundle while accepting the qualified v1 plan."""

    candidate = raw.strip()
    fenced = _JSON_FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ContractError("student output contains no span bundle object")
        candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ContractError("student span bundle is malformed JSON") from exc
    if isinstance(data, dict) and "plans" in data:
        return SpanBundlePlan.from_dict(data)
    return SpanBundlePlan.from_plan(SpanPlan.from_dict(data))


def _span_failure_reason(error: ContractError) -> str:
    detail = str(error)
    if "malformed JSON" in detail:
        return "json-malformed"
    if "file targets must be unique" in detail:
        return "duplicate-file"
    if "must change source" in detail:
        return "no-op"
    if "must match exactly once" in detail:
        return "selector-no-match"
    if "frozen exact-span candidates" in detail:
        return "selector-not-enumerated"
    return "apply-fail"


def _span_failure_is_repairable(error: ContractError) -> bool:
    """Return whether a fresh bounded decode can plausibly repair the plan."""

    detail = str(error)
    return not (
        detail == "semantic recipe fields or candidate_id are invalid"
        or detail == "span bundle changed the frozen candidates"
        or detail == "before must be copied from the frozen exact-span candidates"
        or "must change source" in detail
    )


_NON_EXECUTABLE_INSERTION = re.compile(
    r"^(?://|/\*|\*|\*/|#(?!include\b|define\b)|#(?:include|define)\b|"
    r"import\b|from\b|use\b|(?:export\s+)?(?:async\s+)?(?:def|class|function|"
    r"interface|struct|enum|type)\b)"
)
_DECLARED_SYMBOL = re.compile(
    r"(?:#define|import|use|def|class|function|interface|struct|enum|type)\s+"
    r"(?:\{\s*)?([A-Za-z_][A-Za-z0-9_]*)"
)


def _non_executable_insertion_reason(
    bundle: SpanBundlePlan, instruction: str
) -> str | None:
    """Reject insertion-only pseudo-fixes unless the issue names the new symbol."""

    issue_symbols = {
        token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", instruction)
    }
    for plan in bundle.plans:
        for operation in plan.operations:
            delta = list(
                difflib.ndiff(
                    operation.before.splitlines(), operation.after.splitlines()
                )
            )
            added = [row[2:].strip() for row in delta if row.startswith("+ ")]
            removed = [row[2:].strip() for row in delta if row.startswith("- ")]
            substantive = [row for row in added if row]
            if removed or not substantive:
                continue
            if not all(_NON_EXECUTABLE_INSERTION.match(row) for row in substantive):
                continue
            declared = {
                match.group(1).lower()
                for row in substantive
                if (match := _DECLARED_SYMBOL.search(row)) is not None
            }
            if not declared.intersection(issue_symbols):
                return "non-executable-insertion"
    return None


def _flag_state_overwrite_reason(
    bundle: SpanBundlePlan, instruction: str
) -> str | None:
    """Reject overbroad flag-state edits before an expensive native run."""

    issue_tokens = _semantic_code_tokens(instruction)
    if not {"flag", "align"}.issubset(issue_tokens):
        return None
    for plan in bundle.plans:
        for operation in plan.operations:
            match = _STATE_ASSIGNMENT.search(operation.before)
            if match is None or match.group("field") != "align":
                continue
            state = f"{match.group('object')}.{match.group('field')}"
            after_without_comments = re.sub(r"(?m)//.*$", "", operation.after)
            after_without_comments = re.sub(
                r"/\*.*?\*/", "", after_without_comments, flags=re.DOTALL
            )
            stripped_after = after_without_comments.strip()
            if not stripped_after:
                continue
            direct = re.fullmatch(
                rf"{re.escape(state)}\s*=\s*"
                rf"(?P<namespace>[A-Za-z_][A-Za-z0-9_]*)::"
                rf"(?P<value>none|default|unset|unknown)\s*;",
                stripped_after,
            )
            if direct is not None:
                continue
            if "if" not in _code_tokens(after_without_comments):
                return (
                    "flag state overwrite requires deletion, a neutral/default "
                    "state assignment, or a conditional default guard"
                )
            if after_without_comments.count(state) < 2:
                return "flag state guard must test and assign the same state"
            object_roots = set(
                re.findall(
                    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:\.|->)",
                    after_without_comments,
                )
            )
            foreign = object_roots - {match.group("object"), "this"}
            if foreign:
                return "flag state guard references a foreign local scope"
            if after_without_comments.lstrip().startswith(operation.before.strip()):
                return "flag state guard preserves the unconditional overwrite"
    return None


def _has_flag_state_overwrite_candidate(
    task: StudentTask, candidates: tuple[ExactSpanCandidate, ...]
) -> bool:
    issue_tokens = _semantic_code_tokens(task.instruction)
    if not {"flag", "align"}.issubset(issue_tokens):
        return False
    return any(
        (match := _STATE_ASSIGNMENT.search(candidate.before)) is not None
        and match.group("field") == "align"
        for candidate in candidates
    )


def fixed_span_context(task: StudentTask, max_context_chars: int) -> tuple[str, str]:
    """Select one target and gold-free source windows identically for both arms."""

    targets = list(task.allowed_targets)
    if not targets:
        targets = list(StudentAdapter._discover_targets(task))
    if not targets:
        raise ContractError("span task has no source target")
    relative = targets[0]
    content = task.resolve_target(relative).read_text(
        encoding="utf-8", errors="replace"
    )
    query = task.instruction
    code_evidence = _issue_code_evidence(task.instruction)
    if code_evidence:
        query += "\n" + code_evidence
    excerpt = _issue_overlap_excerpt(content, query, max_context_chars)
    if not excerpt:
        raise ContractError("span source context is empty")
    return relative, excerpt


def fixed_span_contexts(
    task: StudentTask,
    max_context_chars: int,
    *,
    max_targets: int = 32,
) -> tuple[tuple[str, str], ...]:
    """Build the same bounded candidate anchors for baseline and taught arms."""

    targets = list(task.allowed_targets)
    if not targets:
        targets = list(StudentAdapter._discover_targets(task))
    targets = targets[:max_targets]
    if not targets:
        raise ContractError("span task has no source target")
    per_target = max(400, max_context_chars // len(targets))
    query = task.instruction
    code_evidence = _issue_code_evidence(task.instruction)
    if code_evidence:
        query += "\n" + code_evidence
    contexts: list[tuple[str, str]] = []
    for relative in targets:
        content = task.resolve_target(relative).read_text(
            encoding="utf-8", errors="replace"
        )
        excerpt = _issue_overlap_excerpt(content, query, per_target)
        if excerpt:
            contexts.append((relative, excerpt))
    if not contexts:
        raise ContractError("span source contexts are empty")
    return tuple(contexts)


class SpanPlanAdapter(StudentAdapter):
    """Render model-owned exact spans into one auditable unified diff."""

    def experiment_config(self) -> dict[str, Any]:
        configured = getattr(self.generator, "generation_config", None)
        return {
            "adapter": type(self).__name__,
            "adapter_contract": "multilanguage-exact-span-plan-v1",
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "generator": configured() if configured is not None else {},
            "renderer": {
                "name": "multilanguage-exact-span-plan-v1",
                "qualification_suite": (
                    "multilanguage-exact-span-renderer-synthetic-v1"
                ),
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
        bundle_limit = getattr(
            getattr(self.generator, "profile", None),
            "bundle_output_chars",
            _MAX_SPAN_BUNDLE_CHARS,
        )
        if len(raw) > bundle_limit:
            return self._failure(
                task,
                revision,
                raw,
                "plan-too-large",
                f"span bundle exceeds {bundle_limit} characters",
            )
        if revision.source_round >= 39:
            typed_candidates = _frozen_causal_candidates(task, revision)
            typed_catalog = fixed_typed_actions(task, typed_candidates)
            if revision.source_round >= 70 and not typed_catalog:
                projected, changed = _canonicalize_semantic_recipe_output(
                    raw,
                    typed_candidates,
                    task=task,
                    recipe_output_chars=getattr(
                        getattr(self.generator, "profile", None),
                        "recipe_output_chars",
                        _MAX_EXACT_SPAN_CHARS,
                    ),
                )
                if changed:
                    raw = projected
            typed_reason = (
                _typed_state_action_reason(raw, revision, typed_catalog)
                if typed_catalog
                else None
            )
            if typed_reason is not None:
                failure_reason = (
                    "semantic-overbroad"
                    if typed_reason.startswith("inactive Skill action preference")
                    else "schema-invalid"
                )
                return self._failure(
                    task,
                    revision,
                    raw,
                    failure_reason,
                    typed_reason,
                )
        raw, _operation_only = _canonicalize_operation_only_output(raw, task, revision)
        raw, _canonicalized = _canonicalize_span_bundle_output(raw)
        unresolved = parse_unresolved_abstention(raw)
        if unresolved is not None:
            return self._failure(task, revision, raw, "unresolved", unresolved)
        preliminary = self._classify_unstructured(raw)
        if preliminary is not None:
            return self._failure(task, revision, raw, preliminary, preliminary)
        try:
            bundle = parse_span_bundle_output(raw)
        except ContractError as exc:
            reason = _span_failure_reason(exc)
            if reason == "apply-fail":
                reason = "malformed-hunk"
            return self._failure(task, revision, raw, reason, str(exc))
        selected_candidates: list[dict[str, Any]] = []
        if revision.source_round >= 25:
            eligible_candidates = fixed_exact_span_candidates(
                _editable_span_task(task, revision)
            )
            if revision.source_round >= 27:
                eligible_candidates = _frozen_causal_candidates(task, revision)
            candidate_by_selector = {
                (candidate.file, candidate.before): candidate
                for candidate in eligible_candidates
            }
            candidate_by_selector.update(
                _typed_collection_selectors(task, eligible_candidates)
            )
            for plan in bundle.plans:
                for operation in plan.operations:
                    candidate = candidate_by_selector.get((plan.file, operation.before))
                    if candidate is None:
                        return self._failure(
                            task,
                            revision,
                            raw,
                            "selector-not-enumerated",
                            "before must be copied from the frozen exact-span candidates",
                            target_file=plan.file,
                        )
                    selected_candidates.append(candidate.to_prompt_dict())
        if revision.source_round >= 12:
            reason = _non_executable_insertion_reason(bundle, task.instruction)
            if reason is not None:
                return self._failure(task, revision, raw, reason, reason)
        if revision.source_round >= 34:
            reason = _flag_state_overwrite_reason(bundle, task.instruction)
            if reason is not None:
                return self._failure(task, revision, raw, "semantic-overbroad", reason)
        allowed = task.allowed_targets or self._discover_targets(task)
        files = tuple(plan.file for plan in bundle.plans)
        for file in files:
            if file not in allowed or self._is_test_path(file):
                return self._failure(
                    task,
                    revision,
                    raw,
                    "wrong-target",
                    f"target is not allowed: {file}",
                    target_file=file,
                )
        try:
            sources = {
                file: task.resolve_target(file).read_text(encoding="utf-8")
                for file in files
            }
            result = materialize_span_bundle(sources, bundle)
        except (ContractError, OSError, UnicodeError) as exc:
            reason = (
                _span_failure_reason(exc)
                if isinstance(exc, ContractError)
                else "apply-fail"
            )
            return self._failure(
                task,
                revision,
                raw,
                reason,
                str(exc),
                target_file=files[0],
            )
        receipt = canonical_json(
            {
                "receipt_type": "span-bundle-realization-v2",
                "diagnostic": bundle.diagnostic,
                "bundle_fingerprint": bundle.fingerprint,
                "candidate_policy": (
                    "frozen-owning-expression-role-diverse-exact-span-v6"
                    if revision.source_round >= 27
                    else "frozen-enumerable-exact-span-v1"
                    if revision.source_round >= 25
                    else None
                ),
                "selected_candidates": selected_candidates,
                "materialization": result.to_dict(),
            }
        )
        patch = "".join(
            "".join(
                difflib.unified_diff(
                    item.before.splitlines(keepends=True),
                    item.after.splitlines(keepends=True),
                    fromfile=f"a/{file}",
                    tofile=f"b/{file}",
                )
            )
            for file, item in result.results
        )
        if not patch.strip() or not self._git_apply_check(task.checkout, patch):
            return self._failure(
                task,
                revision,
                raw,
                "apply-fail",
                "materialized span patch does not apply cleanly",
                patch=patch,
                target_file=files[0],
                before_sha256=_sha256_text(canonical_json(sources)),
            )
        after = {file: item.after for file, item in result.results}
        implementation = {
            file: _implementation_fingerprint(file, item.after)
            for file, item in result.results
        }
        return StudentAttempt(
            task=task,
            revision_id=revision.revision_id,
            raw_output=raw,
            raw_output_sha256=_sha256_text(raw),
            edit=None,
            patch=patch,
            patch_sha256=_sha256_text(patch),
            target_file=files[0],
            before_sha256=_sha256_text(canonical_json(sources)),
            after_sha256=_sha256_text(canonical_json(after)),
            implementation_fingerprint=_sha256_text(canonical_json(implementation)),
            structural_valid=True,
            failure_reason=None,
            detail=receipt,
        )


class MlxSpanPlanGenerator(MlxStructuredGenerator):
    """Generate bounded exact-span operations with one structural replan."""

    def __init__(
        self,
        *,
        max_tokens: int = 1536,
        max_context_chars: int = 24_000,
        max_context_targets: int = 32,
        max_exact_span_candidates: int = _MAX_EXACT_SPAN_CANDIDATES,
        max_plan_repairs: int | None = None,
        profile: StudentCapabilityProfile | None = None,
        **fields: Any,
    ) -> None:
        self.profile = profile or profile_for(str(fields.get("model_path", "")))
        if max_plan_repairs is None:
            max_plan_repairs = self.profile.max_span_repairs
        if type(max_plan_repairs) is not int or max_plan_repairs < 0:
            raise ValueError("span max_plan_repairs must be non-negative")
        if type(max_context_targets) is not int or not 1 <= max_context_targets <= 32:
            raise ValueError("span max_context_targets must be between 1 and 32")
        if (
            type(max_exact_span_candidates) is not int
            or not 1 <= max_exact_span_candidates <= 32
        ):
            raise ValueError("span max exact span candidates must be between 1 and 32")
        super().__init__(
            max_tokens=max_tokens,
            max_context_chars=max_context_chars,
            max_structural_repairs=0,
            use_grounding_plan=False,
            use_semantic_critic=False,
            **fields,
        )
        self.max_plan_repairs = max_plan_repairs
        self.max_context_targets = max_context_targets
        self.max_exact_span_candidates = max_exact_span_candidates

    def __call__(self, task: StudentTask, revision: LoopRevision) -> str:
        model, tokenizer, generate = self._runtime()
        editable_task = _editable_span_task(task, revision)
        contexts = fixed_span_contexts(
            editable_task,
            self.max_context_chars,
            max_targets=self.max_context_targets,
        )
        allowed_files = tuple(relative for relative, _excerpt in contexts)
        candidates = _frozen_causal_candidates(
            task,
            revision,
            max_candidates=self.max_exact_span_candidates,
        )
        supporting_contexts = (
            *fixed_causal_state_contexts(task, candidates, max_contexts=3),
            *fixed_repository_state_value_contexts(task, candidates, max_contexts=1),
            *fixed_supporting_symbol_contexts(task, candidates),
            *fixed_repository_api_contexts(task, candidates),
        )[:4]
        editable_contexts = fixed_editable_candidate_contexts(task, candidates)
        typed_actions = fixed_typed_actions(task, candidates)
        candidate_selectors = {
            (candidate.file, candidate.before) for candidate in candidates
        }
        candidate_selectors.update(_typed_collection_selectors(task, candidates))
        teaching = revision.skill_text
        if revision.source_round >= 12:
            teaching = _project_numbered_teaching(
                teaching,
                task.instruction,
                mandatory_rule_numbers=(1, 2, 3),
            )
        typed_schema = revision.source_round >= 39 and bool(typed_actions)
        recipe_schema = revision.source_round >= 70 and not typed_schema
        output_contract = (
            "Return exactly one JSON object. Success uses only schema_version and "
            "actions: "
            '{"schema_version":1,"actions":[<one copied action object>]}. '
            "Copy one complete action "
            "object byte-for-byte from ALLOWED TYPED ACTIONS. Do not write source code. "
            if typed_schema
            else (
                "Return exactly one JSON object. Success uses only schema_version "
                "and recipes: "
                '{"schema_version":1,"recipes":[{"candidate_id":"span-001",'
                '"after":"complete replacement"}]}. '
                if recipe_schema
                else "Return exactly one JSON object. Success uses only schema_version "
                "and edits: "
                '{"schema_version":1,"edits":[{"file":"...","before":"...",'
                '"after":"..."}]}. '
            )
        )
        action_rules = (
            "actions contains exactly one object copied from ALLOWED TYPED ACTIONS. "
            "The framework converts it into the exact file/before/after edit; do not "
            "emit edits, source code, a diff, or candidate fields outside actions. "
            if typed_schema
            else (
                "recipes contains one to three objects with exactly candidate_id and "
                "after. The framework converts candidate_id into the exact file and "
                "byte-exact before selector. "
                if recipe_schema
                else "edits contains one to four objects with exactly file, before, "
                "and after. "
            )
        )
        selector_rules = (
            "The framework owns file, before, and after. candidate_id is required "
            "only inside the action object. The framework supplies a bounded, "
            "role-diverse candidate set; choose one allowed action whenever the "
            "action catalog is non-empty. "
            if typed_schema
            else (
                "Choose candidate_id only from ONLY EDITABLE and write the complete "
                "bounded replacement in after. Do not copy file or before into the "
                "response. "
                if recipe_schema
                else "before must be copied exactly from one framework-enumerated "
                "candidate; after is its complete replacement. The framework supplies "
                "up to four role-diverse candidates per frozen file; use only candidates "
                "that support the repair. Candidate IDs and roles are evidence labels "
                "only; do not "
                "add candidate_id to the output schema. "
            )
        )
        active_renderer_contract = (
            "\n\nActive renderer contract overrides any legacy Skill wording: "
            "semantic recipe mode does not require an issue-supplied verbatim "
            "before/after pair. Select candidate_id from ONLY EDITABLE and author "
            "only its complete bounded after replacement; the renderer owns file, "
            "before, intent, and materialization. Do not return unresolved merely "
            "because the issue does not contain a copyable replacement when supplied "
            "source supports a bounded repair."
            if recipe_schema
            else ""
        )
        system = (
            "You are a code-edit student. "
            f"{output_contract}"
            f"{action_rules}"
            "The framework owns diagnosis, intent, target files, target "
            "symbol, and fingerprint; never output those fields. "
            f"{selector_rules}"
            "Use two files only when both "
            "files are required by "
            "one invariant. Keep each span at most 80 lines. "
            "Never repeat a before span, and ensure "
            "after differs from before by non-whitespace content. Keep each "
            f"before+after operation under {self.profile.recipe_output_chars} characters. If no exact unique "
            "span supports a bounded repair, use the single failure schema "
            '{"schema_version":1,"status":"unresolved","diagnostic":"..."} '
            "rather than fabricating "
            "source. Keep the entire response under "
            f"{self.profile.bundle_output_chars} characters. Do not return a diff, full file, "
            "markdown, or prose. Trace the reported symptom backward through local "
            "assignments and supplied callee contracts before editing a downstream "
            "branch result. Supporting evidence is READ-ONLY and is never an "
            "editable before selector. The only legal edit surface is the ONLY "
            "EDITABLE candidate list.\n\n"
            f"Protocol: {revision.protocol}\n"
            f"Teaching Skill:\n{teaching}"
            f"{active_renderer_contract}"
        )
        rendered_context = "\n\n".join(
            f"### {relative}\n{excerpt}" for relative, excerpt in contexts
        )
        rendered_candidates = canonical_json(
            [
                candidate.to_prompt_dict(
                    include_roles=self.profile.show_roles_in_prompt
                )
                for candidate in candidates
            ]
        )
        rendered_support = canonical_json(
            [context.to_prompt_dict() for context in supporting_contexts]
        )
        rendered_editable_contexts = canonical_json(
            [context.to_prompt_dict() for context in editable_contexts]
        )
        rendered_typed_actions = canonical_json(list(typed_actions))
        final_request = (
            "Return the required JSON now. Select one to three candidate_id values "
            "from ONLY EDITABLE and write only each complete bounded after "
            "replacement. Do not output file or before."
            if recipe_schema
            else (
                "Return the required JSON now. Copy each selected file path and "
                "every before span exactly from the ONLY EDITABLE list, or copy one "
                "complete object from ALLOWED TYPED ACTIONS when that list is "
                "non-empty."
            )
        )
        user = (
            f"Task: {task.instruction}\n\n"
            f"Allowed candidate files: {canonical_json(list(allowed_files))}\n\n"
            f"Gold-free source anchors:\n{rendered_context}\n\n"
            f"READ-ONLY causal and repository evidence (never copy as before):\n"
            f"{rendered_support}\n\n"
            "READ-ONLY enclosing control flow for editable candidates (the exact "
            "before selector still comes only from ONLY EDITABLE):\n"
            f"{rendered_editable_contexts}\n\n"
            f"ONLY EDITABLE exact-span candidates (before must match this list):\n"
            f"{rendered_candidates}\n\n"
            f"ALLOWED TYPED ACTIONS (copy exactly when present):\n"
            f"{rendered_typed_actions}\n\n"
            f"{final_request}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        trace: list[str] = []
        prompts: list[str] = []
        results: list[dict[str, str]] = []
        kinds: list[str] = []
        raw = ""
        sources = {
            relative: task.resolve_target(relative).read_text(encoding="utf-8")
            for relative in allowed_files
        }
        for repair_index in range(self.max_plan_repairs + 1):
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                tokenize=False,
            )
            generated_raw = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
            )
            prompts.append(prompt)
            trace.append(generated_raw)
            kinds.append(f"span-plan-attempt-{repair_index}")
            raw = generated_raw
            try:
                typed_reason = (
                    _typed_state_action_reason(generated_raw, revision, typed_actions)
                    if typed_schema
                    else None
                )
                if typed_reason is not None:
                    raise ContractError(typed_reason)
                canonical_raw = generated_raw
                if recipe_schema:
                    canonical_raw, recipe_changed = (
                        _canonicalize_semantic_recipe_output(
                            generated_raw,
                            candidates,
                            task=task,
                            recipe_output_chars=self.profile.recipe_output_chars,
                        )
                    )
                    if not recipe_changed:
                        raise ContractError(
                            "semantic recipe fields or candidate_id are invalid"
                        )
                canonical_raw, _operation_only = _canonicalize_operation_only_output(
                    canonical_raw, task, revision
                )
                raw, _changed = _canonicalize_span_bundle_output(canonical_raw)
                unresolved = parse_unresolved_abstention(raw)
                if unresolved is not None:
                    if revision.source_round >= 34 and (
                        _has_flag_state_overwrite_candidate(task, candidates)
                    ):
                        raise ContractError(
                            f"forbidden unresolved: candidate {candidates[0].candidate_id} "
                            "exists; preserve parser authority using deletion, a neutral "
                            "default, or a conditional default guard"
                        )
                    results.append({"status": "unresolved", "detail": unresolved})
                    break
                bundle = parse_span_bundle_output(raw)
                if any(plan.file not in allowed_files for plan in bundle.plans):
                    raise ContractError("span bundle changed the frozen candidates")
                if any(
                    (plan.file, operation.before) not in candidate_selectors
                    for plan in bundle.plans
                    for operation in plan.operations
                ):
                    raise ContractError(
                        "before must be copied from the frozen exact-span candidates"
                    )
                if revision.source_round >= 34:
                    reason = _flag_state_overwrite_reason(bundle, task.instruction)
                    if reason is not None:
                        raise ContractError(reason)
                materialize_span_bundle(sources, bundle)
            except ContractError as exc:
                results.append({"status": "structural-rejected", "detail": str(exc)})
                if (
                    repair_index >= self.max_plan_repairs
                    or not _span_failure_is_repairable(exc)
                ):
                    break
                messages = [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": (
                            f"{user}\n\nStructural gate failure: {exc}. Replan once "
                            "without changing the diagnosis. Correct only the JSON "
                            "schema, exact before selector, bounded after span, or "
                            "target path. Return a fresh complete plan JSON."
                        ),
                    },
                ]
                continue
            results.append({"status": "structural-valid", "detail": "accepted"})
            raw = generated_raw
            break
        self._last_generation_trace = tuple(trace)
        self._last_generation_trace_kinds = tuple(kinds)
        self._last_generation_prompt_trace = tuple(prompts)
        self._last_generation_trace_results = tuple(results)
        return raw

    def generation_config(self) -> dict[str, Any]:
        return {
            **super().generation_config(),
            "action_space": "multilanguage-typed-edit-action-v20",
            "context_selector": _SPAN_CONTEXT_SELECTOR,
            "target_policy": "frozen-editable-readonly-evidence-dual-scope-v3",
            "max_context_targets": self.max_context_targets,
            "candidate_policy": ("frozen-owning-expression-role-diverse-exact-span-v6"),
            "max_exact_span_candidates": self.max_exact_span_candidates,
            "emitted_exact_span_candidates": (
                "api-scope-constructor-edge-role-diverse-max4-per-file-max8"
            ),
            "support_context_policy": ("read-only-cross-file-causal-state-evidence-v6"),
            "editable_context_policy": "bounded-enclosing-control-flow-v1",
            "max_supporting_symbol_contexts": 4,
            "max_bundle_files": 2,
            "bundle_diagnostic_fallback": True,
            "max_plan_repairs": self.max_plan_repairs,
            "renderer": "multilanguage-exact-span-plan-v1",
            "pre_schema_policy": "localize-copy-unique-or-unresolved-r010",
            "student_output_schema": "typed-edit-actions-or-operation-only-edits-v4",
            "typed_selector_policy": (
                "renderer-owned-minimal-source-derived-selector-v1"
            ),
            "endpoint_action_policy": (
                "issue-bracketed-host-source-split-validation-alternatives-v2"
            ),
            "state_role_policy": "issue-triggered-parser-state-protected-v1",
            "framework_owned_fields": ["diagnosis", "intent", "localization"],
            "teaching_projection": "mandatory-first3-max900-shared-suffix-v3",
            "max_bundle_chars": self.profile.bundle_output_chars,
        }


def build_span_conditions(
    *,
    taught_skill: str,
    parent_revision_id: str,
    source_round: int,
    generation_config: dict[str, Any],
) -> list[ExperimentCondition]:
    """Build paired span conditions varying only the teaching content."""

    definitions = [
        (
            "span-baseline",
            "baseline",
            "No additional domain teaching is provided. Use only repository evidence.",
            None,
        ),
        ("span-taught", "taught", taught_skill, parent_revision_id),
    ]
    conditions = []
    for condition_id, teaching, skill_text, parent_id in definitions:
        revision = LoopRevision.create(
            skill_id="round1-multilanguage-span-skill",
            revision_id=f"round1-{condition_id}-r{source_round:03d}",
            parent_revision_id=parent_id,
            source_round=source_round,
            protocol="multilanguage-typed-state-action-v13",
            skill_text=skill_text,
            prompt_template=_SPAN_PROMPT_TEMPLATE,
            eval_note=(
                "Round 1 multi-language exact-span comparison; Student owns intent "
                "and exact spans while the adapter owns diff construction."
            ),
        )
        conditions.append(
            ExperimentCondition.create(
                condition_id=condition_id,
                mechanism="span",
                teaching=teaching,
                revision=revision,
                generation_config=generation_config,
            )
        )
    return conditions
