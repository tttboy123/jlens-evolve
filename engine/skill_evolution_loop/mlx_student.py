"""Lazy MLX transport for the structured StudentAdapter protocol."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .contracts import ContractError, LoopRevision
from .legacy_swe_4b import HUNK_SYSTEM, MODEL, _hunk_user_prompt, _pick_target_files
from .model_transport import ModelTransport, PromptGenerationRequest
from .student_adapter import StructuredEdit, StudentTask

ModelLoader = Callable[[str], tuple[Any, Any]]
TextGenerator = Callable[..., str]

_CONTEXT_SELECTOR = "skill-ranked-python-symbol-v14-complete-critic-clean-repair"
_SEMANTIC_CRITIC_MAX_TOKENS = 768
_PATTERN_ROUTER = "lexical-symptom-overlap-v2"
_SELECTED_CARD_CONTEXT_CHARS = 8_000
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_+.-]{2,}")
_STRUCTURE_ANCHORS = frozenset(
    {
        "__all__",
        "posonlyargs",
        "rolename",
        "sentinels",
        "subclass",
        "sphinxupquote",
        "rstrip",
        "get_index_text",
        "PyTypedField",
    }
)


def _card_required_anchor_predicate(
    card: str,
) -> Callable[[str], bool] | None:
    """Content-derived relevance predicate for one card.

    A card is selected only when its predicate accepts the issue text. This is
    content-based (works for any card order) and prevents generic words such as
    "default" (HTTP header defaults) from false-matching the trailing-defaults
    card (Python positional defaults).
    """

    symptom = _card_symptom(card).lower()
    if "positional" in symptom and "default" in symptom:
        disappearance_markers = (
            "disappear",
            "vanish",
            "missing",
            "not shown",
            "not display",
            "omitted",
            "lost",
        )
        structure_markers = (
            "positional only",
            "positional-only",
            "defaults vector",
            "trailing default",
        )
        return lambda issue: (
            "default" in issue
            and any(marker in issue for marker in disappearance_markers)
            and any(marker in issue for marker in structure_markers)
        )
    if "mocked" in symptom and "inherit" in symptom:
        return lambda issue: "mocked" in issue and "inherit" in issue
    if "inline" in symptom and "wrapper" in symptom:
        return lambda issue: "inline" in issue
    if "property" in symptom and "parenth" in symptom:
        return lambda issue: "property" in issue and "paren" in issue
    if "variable" in symptom and ("role" in symptom or "obj" in symptom):
        return lambda issue: (
            "variable" in issue
            and any(marker in issue for marker in ("link", "obj", "role"))
        )
    return None


_STOPWORDS = frozenset(
    {
        "and",
        "about",
        "after",
        "also",
        "because",
        "before",
        "behavior",
        "build",
        "class",
        "current",
        "describe",
        "expected",
        "following",
        "from",
        "have",
        "into",
        "inside",
        "issue",
        "name",
        "only",
        "output",
        "project",
        "sphinx",
        "that",
        "the",
        "then",
        "this",
        "two",
        "version",
        "when",
        "while",
        "with",
    }
)
_CODE_ABBREVIATIONS = {
    "argument": "arg",
    "arguments": "arg",
    "highlighted": "highlight",
    "highlighter": "highlight",
    "highlighting": "highlight",
    "latex": "tex",
    "parameter": "param",
    "parameters": "param",
    "positional": "pos",
    "positionals": "pos",
    "spaces": "space",
    "whitespace": "space",
}


def _lexical_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in _TOKEN.findall(text):
        normalized = token.lower().strip("._+-")
        if not normalized:
            continue
        if normalized in _STOPWORDS:
            continue
        terms.add(normalized)
        abbreviation = _CODE_ABBREVIATIONS.get(normalized)
        if abbreviation is not None:
            terms.add(abbreviation)
    return terms


def _pattern_cards(skill_text: str) -> list[str]:
    """Parse numbered PatternCards without interpreting evaluator evidence."""
    marker = "## Pattern cards"
    if marker not in skill_text:
        return []
    section = skill_text.split(marker, 1)[1]
    heading = re.search(r"(?m)^##\s+", section)
    if heading is not None:
        section = section[: heading.start()]
    starts = list(re.finditer(r"(?m)^\d+\.\s+Symptom:\s*", section))
    cards: list[str] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(section)
        card = section[match.start() : end].strip()
        if card:
            cards.append(card)
    return cards


def _card_symptom(card: str) -> str:
    """Return the Symptom portion of one numbered card (before Transformation)."""
    return re.split(r"\bTransformation:\s*", card, maxsplit=1)[0]


def _score_card(instruction: str, card: str) -> float:
    """Lexical overlap plus shared structural anchors between issue and card."""
    symptom = _card_symptom(card)
    query = _lexical_terms(instruction)
    card_terms = _lexical_terms(symptom)
    overlap = query & card_terms
    lexical = sum(3 if any(char in term for char in "_+.-") else 1 for term in overlap)
    instruction_low = (instruction or "").lower()
    symptom_low = symptom.lower()
    anchors = sum(
        2.0
        for anchor in _STRUCTURE_ANCHORS
        if anchor in instruction_low and anchor in symptom_low
    )
    return float(lexical) + anchors


def _select_pattern_cards_v2(
    skill_text: str, instruction: str, *, top_k: int = 3
) -> list[str]:
    """Rank cards by lexical overlap + structural anchors; abstain when weak.

    Required per-card anchors reject semantically unrelated defects first. A
    surviving card then needs at least one shared lexical term. Cards are
    ordered by score, then source order for ties, so the result is deterministic.
    """
    ranked: list[tuple[float, int, str]] = []
    instruction_low = (instruction or "").lower()
    for index, card in enumerate(_pattern_cards(skill_text)):
        predicate = _card_required_anchor_predicate(card)
        if predicate is not None and not predicate(instruction_low):
            continue
        ranked.append((_score_card(instruction, card), -index, card))
    if not ranked:
        return []
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    # The per-card required code anchors are the primary relevance gate; the
    # score floor only confirms at least one shared lexical term.
    threshold = 1.0
    selected: list[str] = []
    for score, _order, card in ranked:
        if score < threshold:
            break
        selected.append(card)
        if len(selected) >= max(1, top_k):
            break
    return selected


def _select_pattern_card(skill_text: str, instruction: str) -> str | None:
    """Choose one task-matched card using only frozen Skill and issue text."""
    cards = _select_pattern_cards_v2(skill_text, instruction, top_k=1)
    return cards[0] if cards else None


def _commit_gate(skill_text: str) -> str | None:
    marker = "## Commit gate"
    if marker not in skill_text:
        return None
    section = skill_text.split(marker, 1)[1]
    heading = re.search(r"(?m)^##\s+", section)
    if heading is not None:
        section = section[: heading.start()]
    gate = section.strip()
    return gate or None


def _teaching_reminder(skill_text: str, instruction: str) -> tuple[str, str | None]:
    card = _select_pattern_card(skill_text, instruction)
    gate = _commit_gate(skill_text)
    fields = [
        "Selected teaching card:\n"
        + (
            card
            if card is not None
            else "No PatternCard met the lexical match threshold; do not force one."
        )
    ]
    if gate is not None:
        fields.append(f"Commit gate:\n{gate}")
    return "\n\n".join(fields), card


def _project_numbered_teaching(
    skill_text: str,
    instruction: str,
    *,
    max_rules: int = 3,
    max_chars: int = 900,
    mandatory_rule_numbers: tuple[int, ...] = (),
) -> str:
    """Retrieve a bounded task-matched projection from an inactive Skill.

    The frozen Skill remains the source of truth. Selection uses only issue text
    and rule text, so it cannot observe evaluator labels or hidden task data.
    """

    if skill_text.startswith("No additional domain teaching"):
        return skill_text
    if type(max_rules) is not int or max_rules < 1:
        raise ValueError("teaching projection max_rules must be positive")
    if type(max_chars) is not int or max_chars < 200:
        raise ValueError("teaching projection max_chars must be at least 200")
    if any(type(number) is not int or number < 1 for number in mandatory_rule_numbers):
        raise ValueError("mandatory teaching rule numbers must be positive")
    mandatory_marker = "## Shared diagnosis and localization (read-only)"
    mandatory = ""
    if mandatory_marker in skill_text:
        skill_text, suffix = skill_text.split(mandatory_marker, 1)
        mandatory = f"{mandatory_marker}{suffix}".strip()
    matches = list(
        re.finditer(
            r"(?ms)^(\d+)\.\s+(.+?)(?=^\d+\.\s+|\Z)",
            skill_text,
        )
    )
    if not matches:
        projected = skill_text.strip()
        return f"{projected}\n\n{mandatory}" if mandatory else projected
    query = _lexical_terms(instruction)
    rules = [
        (int(match.group(1)), " ".join(match.group(2).strip().split()))
        for match in matches
    ]
    mandatory_numbers = set(mandatory_rule_numbers)
    mandatory_rules = [row for row in rules if row[0] in mandatory_numbers]
    ranked_optional = sorted(
        (row for row in rules if row[0] not in mandatory_numbers),
        key=lambda row: (-len(query & _lexical_terms(row[1])), row[0]),
    )
    ranked = [
        *mandatory_rules,
        *ranked_optional[: max(0, max_rules - len(mandatory_rules))],
    ]
    ranked = (
        sorted(
            rules,
            key=lambda row: (-len(query & _lexical_terms(row[1])), row[0]),
        )[:max_rules]
        if not mandatory_rules
        else ranked
    )
    selected = sorted(ranked)
    header = "Task-matched rules from the frozen inactive Skill:\n"
    footer = "\nUse only supplied source; preserve unrelated behavior."
    rendered: list[str] = []
    for number, rule in selected:
        candidate = [*rendered, f"{number}. {rule}"]
        projected = header + "\n".join(candidate) + footer
        if len(projected) > max_chars:
            continue
        rendered = candidate
    if not rendered:
        number, rule = ranked[0]
        available = max_chars - len(header) - len(footer) - len(f"{number}. ")
        rendered = [f"{number}. {rule[:available].rstrip()}"]
    projected = header + "\n".join(rendered) + footer
    return f"{projected}\n\n{mandatory}" if mandatory else projected


def _card_retrieval_text(card: str) -> str:
    """Use invariants as gates, never as positive source-retrieval terms."""
    return re.split(r"\bValidation:\s*", card, maxsplit=1)[0].strip()


def _issue_code_evidence(instruction: str, *, limit: int = 6_000) -> str:
    """Extract public issue literals that can bind prose to source identifiers."""
    fenced = re.findall(r"```(?:[^\n]*)\n(.*?)```", instruction, flags=re.DOTALL)
    inline = re.findall(r"(?<!`)`([^`\n]+)`(?!`)", instruction)
    return "\n".join([*fenced, *inline])[:limit]


def _grounding_gate(selected_card: str | None) -> str:
    if selected_card is None:
        return "No teaching card was selected; rely only on task and source evidence."
    return (
        "Grounding gate: the selected card is the mandatory hypothesis for this "
        "task. Edit only the source path whose identifiers and control flow realize "
        "its Transformation; implement every transformation clause, not a subset. "
        "Treat Validation as invariants: never edit a path the card says must stay "
        "unchanged. In diagnostic, cite at least two concrete source identifiers "
        "that establish the data-flow match. Because only one edit object is "
        "allowed, choose one contiguous exact search span large enough to cover "
        "every adjacent line required by all transformation clauses. The search "
        "and replace strings must differ and the replacement must contain the "
        "actual implementation; a descriptive no-op is invalid."
    )


def _source_overlap_count(content: str, query: str) -> int:
    terms = _lexical_terms(query)
    source = _lexical_terms(content)
    return sum(
        term in source or (len(term) >= 3 and any(term in token for token in source))
        for term in terms
    )


def _python_symbol_excerpt(content: str, query: str, budget: int) -> str | None:
    """Select one Python function block by strong, dense issue/card overlap."""
    symbols = list(
        re.finditer(
            r"(?m)^(?P<indent>[ \t]*)(?P<kind>def|class)\s+"
            r"(?P<name>[A-Za-z_]\w*)\b[^\n]*",
            content,
        )
    )
    candidates: list[tuple[int, int, int, str]] = []
    for index, symbol in enumerate(symbols):
        if symbol.group("kind") != "def":
            continue
        indent = len(symbol.group("indent").expandtabs(8))
        end = len(content)
        for following in symbols[index + 1 :]:
            following_indent = len(following.group("indent").expandtabs(8))
            if following_indent <= indent:
                end = following.start()
                break
        block = content[symbol.start() : end].rstrip() + "\n"
        overlap = _source_overlap_count(block, query)
        if overlap:
            candidates.append((overlap, len(block), symbol.start(), block))
    if not candidates:
        return None
    maximum = max(row[0] for row in candidates)
    if maximum < 3:
        return None
    _overlap, _size, _start, block = max(
        candidates,
        key=lambda row: (
            row[0],
            row[0] * 1_000_000 // max(1, row[1]),
            -row[2],
        ),
    )
    return block[:budget]


class MlxStructuredGenerator:
    """Generate one bounded search/replace object with a cached local model."""

    def __init__(
        self,
        *,
        model_path: str = MODEL,
        max_tokens: int = 768,
        max_context_chars: int = 80_000,
        enable_thinking: bool = False,
        max_structural_repairs: int = 0,
        use_grounding_plan: bool = False,
        use_semantic_critic: bool = False,
        seed: int = 0,
        temperature: float = 0.0,
        loader: ModelLoader | None = None,
        text_generator: TextGenerator | None = None,
        model_transport: ModelTransport | None = None,
        tokenizer_loader: Callable[[str], Any] | None = None,
    ) -> None:
        if (
            max_tokens < 1
            or max_context_chars < 1
            or type(max_structural_repairs) is not int
            or max_structural_repairs < 0
            or type(use_grounding_plan) is not bool
            or type(use_semantic_critic) is not bool
        ):
            raise ValueError("MLX generation limits must be positive")
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.max_context_chars = max_context_chars
        self.enable_thinking = enable_thinking
        self.max_structural_repairs = max_structural_repairs
        self.use_grounding_plan = use_grounding_plan
        self.use_semantic_critic = use_semantic_critic
        self.seed = seed
        self.temperature = temperature
        self._loader = loader
        self._text_generator = text_generator
        self._model_transport = model_transport
        self._tokenizer_loader = tokenizer_loader
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._last_generation_trace: tuple[str, ...] = ()
        self._last_generation_trace_kinds: tuple[str, ...] = ()
        self._last_generation_prompt_trace: tuple[str, ...] = ()
        self._last_generation_trace_results: tuple[dict[str, Any], ...] = ()

    def _generate(
        self,
        model: Any,
        tokenizer: Any,
        *,
        prompt: str,
        max_tokens: int,
    ) -> str:
        """Deterministic generation entry shared by every model call.

        Remote transports already pin temperature=0 and seed=0 inside
        ``remote_generate``. For the local MLX runtime we explicitly reseed the
        global MLX PRNG before each call so identical prompts reproduce the same
        output; MLX's default sampler is greedy (temperature 0).
        """
        if self._model_transport is not None:
            _, _, generate = self._runtime()
            return generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
            )
        if self._text_generator is None:
            _, _, generate = self._runtime()
            return generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
            )
        try:
            import mlx.core as mx

            mx.random.seed(int(self.seed))
        except Exception:
            pass
        return self._text_generator(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        )

    def __call__(self, task: StudentTask, revision: LoopRevision) -> str:
        model, tokenizer, generate = self._runtime()
        messages = [
            {
                "role": "system",
                "content": self._system_prompt(revision),
            },
            {
                "role": "user",
                "content": self._user_prompt(task, revision),
            },
        ]
        trace: list[str] = []
        trace_kinds: list[str] = []
        prompt_trace: list[str] = []
        trace_results: list[dict[str, Any]] = []
        if self.use_grounding_plan:
            plan_prompt = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": self._plan_system_prompt(revision),
                    },
                    {
                        "role": "user",
                        "content": self._user_prompt(task, revision, plan_mode=True),
                    },
                ],
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                tokenize=False,
            )
            plan = self._generate(
                model,
                tokenizer,
                prompt=plan_prompt,
                max_tokens=min(384, self.max_tokens),
            )
            prompt_trace.append(plan_prompt)
            trace.append(plan)
            trace_kinds.append("grounding-plan")
            trace_results.append({"status": "generated"})
            messages[1]["content"] += (
                "\n\nGrounding plan produced by the same local model:\n"
                + plan
                + "\n\nUse this plan only where it matches the supplied source. "
                "The final edit must implement every planned transformation and "
                "preserve every planned invariant."
            )
        raw = ""
        for repair_index in range(self.max_structural_repairs + 1):
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                tokenize=False,
            )
            raw = self._generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
            )
            prompt_trace.append(prompt)
            trace.append(raw)
            trace_kinds.append(f"edit-attempt-{repair_index}")
            trace_results.append({"status": "generated"})
            try:
                edit = StructuredEdit.from_model_output(raw)
                self._preflight_edit(task, edit)
            except ContractError as exc:
                trace_results[-1] = {
                    "status": "structural-rejected",
                    "detail": str(exc),
                }
                if repair_index >= self.max_structural_repairs:
                    break
                messages = [
                    {
                        "role": "system",
                        "content": self._system_prompt(revision),
                    },
                    {
                        "role": "user",
                        "content": (
                            self._user_prompt(task, revision)
                            + "\n\nStructural preflight issue from the prior "
                            f"attempt: {exc}. Re-derive the exact unique search "
                            "span from the supplied file content and correct the "
                            "edit. Do not copy any prior response. Return only one "
                            "valid JSON object."
                        ),
                    },
                ]
                continue
            trace_results[-1] = {"status": "structural-valid"}
            if self.use_semantic_critic:
                critic, critic_prompt = self._run_semantic_critic(
                    model=model,
                    tokenizer=tokenizer,
                    generate=generate,
                    task=task,
                    revision=revision,
                    proposed_edit=raw,
                )
                prompt_trace.append(critic_prompt)
                trace.append(critic)
                trace_kinds.append(f"semantic-critic-{repair_index}")
                trace_results.append({"status": "generated"})
                try:
                    critique = self._parse_semantic_critique(critic)
                except ContractError as exc:
                    trace_results[-1] = {
                        "status": "contract-rejected",
                        "detail": str(exc),
                    }
                    if repair_index >= self.max_structural_repairs:
                        raw = f"SEMANTIC_PREFLIGHT_REJECTED: {exc}"
                        break
                    messages.extend(
                        [
                            {"role": "assistant", "content": raw},
                            {
                                "role": "user",
                                "content": (
                                    "Semantic critic preflight was invalid: "
                                    f"{exc}. Re-check every teaching clause and "
                                    "return a corrected edit JSON."
                                ),
                            },
                        ]
                    )
                    continue
                if not critique["complete"]:
                    issues = [
                        *critique["missing_clauses"],
                        *critique["violated_invariants"],
                    ]
                    detail = json.dumps(issues, sort_keys=True)
                    trace_results[-1] = {
                        "status": "semantic-rejected",
                        "missing_clause_count": len(critique["missing_clauses"]),
                        "violated_invariant_count": len(
                            critique["violated_invariants"]
                        ),
                    }
                    if repair_index >= self.max_structural_repairs:
                        raw = "SEMANTIC_PREFLIGHT_REJECTED: " + detail
                        break
                    messages = [
                        {
                            "role": "system",
                            "content": self._system_prompt(revision),
                        },
                        {
                            "role": "user",
                            "content": (
                                self._user_prompt(task, revision)
                                + "\n\nSemantic critic issues:\n"
                                + detail
                                + "\n\nRevise the one contiguous edit so every "
                                "missing Transformation clause is implemented and "
                                "every invariant is preserved. The new replacement "
                                "must be freshly derived from the supplied source "
                                "and critic issues. Do not copy any prior response. "
                                "Return only the corrected edit JSON."
                            ),
                        },
                    ]
                    continue
                trace_results[-1] = {"status": "semantic-accepted"}
            break
        self._last_generation_trace = tuple(trace)
        self._last_generation_trace_kinds = tuple(trace_kinds)
        self._last_generation_prompt_trace = tuple(prompt_trace)
        self._last_generation_trace_results = tuple(trace_results)
        return raw

    @staticmethod
    def _preflight_edit(task: StudentTask, edit: StructuredEdit) -> None:
        if edit.file not in task.allowed_targets:
            raise ContractError("student target is outside the allowed target set")
        content = task.resolve_target(edit.file).read_text(
            encoding="utf-8", errors="replace"
        )
        if content.count(edit.search) != 1:
            raise ContractError("student search span must match exactly once")
        start = content.index(edit.search)
        outside = content[:start] + content[start + len(edit.search) :]
        outside_lines = {line.strip() for line in outside.splitlines() if line.strip()}
        search_lines = {
            line.strip() for line in edit.search.splitlines() if line.strip()
        }

        def structural(line: str) -> bool:
            return bool(
                re.match(r"(?:async\s+)?(?:def|class|for|while|if|elif)\b", line)
                or line == "else:"
                or ".append(" in line
            )

        duplicates = sorted(
            {
                normalized
                for line in edit.replace.splitlines()
                if (normalized := line.strip())
                and normalized not in search_lines
                and normalized in outside_lines
                and structural(normalized)
            }
        )
        if duplicates:
            raise ContractError(
                "replacement duplicates structural lines already present outside "
                "the search span; expand the search span instead of re-adding: "
                + " | ".join(duplicates[:3])
            )

    def generation_trace(self) -> tuple[str, ...]:
        """Expose every bounded local attempt for append-only cell evidence."""
        return self._last_generation_trace

    def generation_trace_kinds(self) -> tuple[str, ...]:
        return self._last_generation_trace_kinds

    def generation_prompt_trace(self) -> tuple[str, ...]:
        """Expose the exact rendered prompt paired with every local output."""
        return self._last_generation_prompt_trace

    def generation_trace_results(self) -> tuple[dict[str, Any], ...]:
        """Expose fail-closed stage outcomes aligned with prompts and outputs."""
        return self._last_generation_trace_results

    def _run_semantic_critic(
        self,
        *,
        model: Any,
        tokenizer: Any,
        generate: TextGenerator,
        task: StudentTask,
        revision: LoopRevision,
        proposed_edit: str,
    ) -> tuple[str, str]:
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a fail-closed semantic edit critic. Return exactly "
                        "one JSON object with fields complete (boolean), "
                        "missing_clauses (string list), violated_invariants (string "
                        "list), and evidence (string list). Mark complete true only "
                        "when the proposed replacement implements every selected "
                        "Teaching Skill Transformation clause and preserves every "
                        "Validation invariant. Check repeated categories, both "
                        "boundaries, operation order, and unchanged paths literally. "
                        "For every quantified word such as both or each, require "
                        "separate literal replacement evidence for every referenced "
                        "branch. For a clause saying to index both loops against one "
                        "aligned vector, the aligned-vector identifier must appear "
                        "inside each loop body. Never accept a diagnostic claim as "
                        "evidence when the replacement code does not show it. Keep "
                        "each string under 160 characters and return at most three "
                        "evidence strings so the JSON always closes."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        self._user_prompt(task, revision, plan_mode=True)
                        + "\n\nProposed edit to audit:\n"
                        + proposed_edit
                        + "\n\nReturn the semantic critic JSON now."
                    ),
                },
            ],
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
            tokenize=False,
        )
        return (
            self._generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=min(_SEMANTIC_CRITIC_MAX_TOKENS, self.max_tokens),
            ),
            prompt,
        )

    @staticmethod
    def _parse_semantic_critique(raw: str) -> dict[str, Any]:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ContractError("semantic critic returned no JSON object")
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ContractError("semantic critic JSON is malformed") from exc
        fields = {
            "complete",
            "missing_clauses",
            "violated_invariants",
            "evidence",
        }
        if not isinstance(data, dict) or set(data) != fields:
            raise ContractError("semantic critic fields are invalid")
        if type(data["complete"]) is not bool or any(
            not isinstance(data[field], list)
            or any(not isinstance(value, str) for value in data[field])
            for field in fields - {"complete"}
        ):
            raise ContractError("semantic critic values are invalid")
        return data

    def _runtime(self) -> tuple[Any, Any, TextGenerator]:
        if self._model_transport is not None:
            if self._tokenizer is None:
                if self._tokenizer_loader is None:
                    from transformers import AutoTokenizer

                    self._tokenizer_loader = AutoTokenizer.from_pretrained
                self._tokenizer = self._tokenizer_loader(self.model_path)

            def remote_generate(
                _model: Any,
                _tokenizer: Any,
                *,
                prompt: str,
                max_tokens: int,
            ) -> str:
                response = self._model_transport.generate_prompt(
                    PromptGenerationRequest.create(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=0.0,
                        seed=0,
                    )
                )
                return response.text

            return None, self._tokenizer, remote_generate
        if self._loader is None or self._text_generator is None:
            from mlx_lm import generate, load

            self._loader = self._loader or load
            self._text_generator = self._text_generator or generate
        if self._model is None or self._tokenizer is None:
            self._model, self._tokenizer = self._loader(self.model_path)
        return self._model, self._tokenizer, self._text_generator

    def generation_config(self) -> dict[str, Any]:
        root = Path(self.model_path).resolve()
        identities: dict[str, str] = {}
        for name in (
            "config.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        ):
            path = root / name
            identities[name] = (
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path.is_file()
                else "missing"
            )
        return {
            "generator": type(self).__name__,
            "model_path": str(root),
            "model_identity_sha256": identities,
            "max_tokens": self.max_tokens,
            "max_context_chars": self.max_context_chars,
            "max_structural_repairs": self.max_structural_repairs,
            "use_grounding_plan": self.use_grounding_plan,
            "use_semantic_critic": self.use_semantic_critic,
            "seed": self.seed,
            "temperature": self.temperature,
            "context_selector": _CONTEXT_SELECTOR,
            "selected_card_context_chars": _SELECTED_CARD_CONTEXT_CHARS,
            "teaching_router": _PATTERN_ROUTER,
            "enable_thinking": self.enable_thinking,
            "decoder": "mlx-lm-default",
            "prompt_rendering": "chat-template-tokenize-false-v1",
            "execution_mode": (
                "remote-transport" if self._model_transport is not None else "local-mlx"
            ),
            "transport_identity": (
                self._model_transport.identity()
                if self._model_transport is not None
                else None
            ),
        }

    @staticmethod
    def _system_prompt(revision: LoopRevision) -> str:
        return (
            "You are a code-edit student. Return exactly one JSON object and no "
            "markdown or prose. Required fields are file, search, replace, and "
            "diagnostic. The search text must be copied exactly from one allowed "
            "non-test file and match once. Keep the replacement minimal.\n\n"
            f"Protocol: {revision.protocol}\n"
            f"Teaching Skill:\n{revision.skill_text}"
        )

    @staticmethod
    def _plan_system_prompt(revision: LoopRevision) -> str:
        return (
            "You are a code-localization planner. Return exactly one JSON object "
            "with fields target_file, target_symbol, source_identifiers, "
            "transform_steps, and invariants. Do not return a patch or edit strings. "
            "Map every teaching Transformation clause to concrete supplied source "
            "identifiers, and map every Validation clause to an unchanged path.\n\n"
            f"Protocol: {revision.protocol}\n"
            f"Teaching Skill:\n{revision.skill_text}"
        )

    def _user_prompt(
        self,
        task: StudentTask,
        revision: LoopRevision,
        *,
        plan_mode: bool = False,
    ) -> str:
        reminder, selected_card = _teaching_reminder(
            revision.skill_text, task.instruction
        )
        issue_title = next(
            (line.strip() for line in task.instruction.splitlines() if line.strip()),
            task.instruction,
        )
        context_query = issue_title
        if selected_card is not None:
            context_query += "\n" + _card_retrieval_text(selected_card)
        code_evidence = _issue_code_evidence(task.instruction)
        if code_evidence:
            context_query += "\n" + code_evidence
        targets = list(task.allowed_targets)
        if not targets:
            found, _ = _pick_target_files(task.checkout, task.instruction)
            targets = [path.relative_to(task.checkout).as_posix() for path in found]
        context_targets = targets
        remaining = self.max_context_chars
        if selected_card is not None:
            card_query = _card_retrieval_text(selected_card)
            ranked = []
            for index, relative in enumerate(targets):
                content = task.resolve_target(relative).read_text(
                    encoding="utf-8", errors="replace"
                )
                ranked.append(
                    (
                        _source_overlap_count(content, card_query),
                        _source_overlap_count(content, context_query),
                        -index,
                        relative,
                    )
                )
            context_targets = [max(ranked)[3]]
            remaining = min(remaining, _SELECTED_CARD_CONTEXT_CHARS)
        sections: list[str] = []
        for index, relative in enumerate(context_targets):
            path = task.resolve_target(relative)
            content = path.read_text(encoding="utf-8", errors="replace")
            targets_left = len(context_targets) - index
            budget = max(1, remaining // targets_left)
            excerpt = (
                _python_symbol_excerpt(content, context_query, budget)
                if selected_card is not None and path.suffix == ".py"
                else None
            )
            if excerpt is None:
                excerpt = _issue_overlap_excerpt(
                    content,
                    context_query,
                    budget,
                )
            sections.append(f"### {relative}\n{excerpt}")
            remaining -= len(excerpt)
            if remaining <= 0:
                break
        final_instruction = (
            "Return the grounding-plan JSON now. Include one transform_steps entry "
            "for every selected-card Transformation clause and one invariants entry "
            "for every Validation clause."
            if plan_mode
            else "Return the JSON edit object now."
        )
        return (
            f"Task: {task.instruction}\n\n"
            f"Prompt template: {revision.prompt_template}\n\n"
            f"Allowed targets: {', '.join(targets)}\n\n"
            "File content:\n"
            + "\n\n".join(sections)
            + f"\n\n{reminder}"
            + f"\n\n{_grounding_gate(selected_card)} "
            + final_instruction
        )


def _issue_overlap_excerpt(content: str, instruction: str, budget: int) -> str:
    """Select gold-free source windows using lexical overlap with the issue."""
    if budget < 1:
        return ""
    if len(content) <= budget:
        return content
    query = _lexical_terms(instruction)
    head_size = min(2_000, max(500, budget // 6))
    window_size = min(5_000, max(1_500, budget // 3))
    stride = max(1_000, window_size // 2)
    raw_windows: list[tuple[int, int, set[str]]] = []
    for start in range(0, len(content), stride):
        end = min(len(content), start + window_size)
        if start < head_size:
            continue
        tokens = _lexical_terms(content[start:end])
        raw_windows.append((start, end, tokens))
        if end == len(content):
            break

    def matches(term: str, tokens: set[str]) -> bool:
        return term in tokens or (
            len(term) >= 3 and any(term in token for token in tokens)
        )

    document_frequency = {
        term: sum(matches(term, tokens) for _start, _end, tokens in raw_windows)
        for term in query
    }
    windows: list[tuple[int, int, int]] = []
    for start, end, tokens in raw_windows:
        overlap = {term for term in query if matches(term, tokens)}
        score = sum(
            (1_000_000 // (document_frequency[term] ** 2))
            * (3 if any(char in term for char in "_+.-") else 1)
            for term in overlap
        )
        score += len(overlap) * 1_000_000_000
        windows.append((score, start, end))
    selected: list[tuple[int, int]] = [(0, head_size)]
    used = head_size
    for score, start, end in sorted(windows, key=lambda row: (-row[0], row[1])):
        if score == 0 or used >= budget:
            break
        uncovered = _subtract_intervals((start, end), selected)
        if not uncovered:
            continue
        start, end = max(uncovered, key=lambda row: row[1] - row[0])
        available = budget - used
        chosen_end = min(end, start + available)
        selected.append((start, chosen_end))
        used += chosen_end - start
    if used < budget:
        tail_start = max(head_size, len(content) - (budget - used))
        if not any(
            tail_start < chosen_end and len(content) > chosen_start
            for chosen_start, chosen_end in selected
        ):
            selected.append((tail_start, len(content)))
    parts = [content[start:end] for start, end in sorted(selected)]
    return "\n... [gold-free issue-overlap excerpt] ...\n".join(parts)[:budget]


def _subtract_intervals(
    candidate: tuple[int, int], selected: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    segments = [candidate]
    for chosen_start, chosen_end in sorted(selected):
        remaining: list[tuple[int, int]] = []
        for start, end in segments:
            if chosen_end <= start or chosen_start >= end:
                remaining.append((start, end))
                continue
            if start < chosen_start:
                remaining.append((start, chosen_start))
            if chosen_end < end:
                remaining.append((chosen_end, end))
        segments = remaining
    return segments


class MlxHunkGenerator(MlxStructuredGenerator):
    """Generate a small unified diff while reusing one cached MLX runtime."""

    def __init__(self, *, max_tokens: int = 512, **fields: Any) -> None:
        super().__init__(max_tokens=max_tokens, **fields)

    def __call__(self, task: StudentTask, revision: LoopRevision) -> str:
        model, tokenizer, generate = self._runtime()
        targets = list(task.allowed_targets)
        if not targets:
            found, _ = _pick_target_files(task.checkout, task.instruction)
            targets = [path.relative_to(task.checkout).as_posix() for path in found]
        target_paths = [task.resolve_target(relative) for relative in targets]
        skill = (
            HUNK_SYSTEM
            + "\n\n### Teaching Skill\n"
            + revision.skill_text
            + "\n\nProtocol: "
            + revision.protocol
        )
        user = (
            _hunk_user_prompt(task.instruction, target_paths, task.checkout)
            + "\n\nPrompt template:\n"
            + revision.prompt_template
        )
        reminder, selected_card = _teaching_reminder(
            revision.skill_text, task.instruction
        )
        user += (
            f"\n\n{reminder}\n\n"
            f"{_grounding_gate(selected_card)} "
            "Return only the unified diff now."
        )
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": skill},
                {"role": "user", "content": user},
            ],
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
            tokenize=False,
        )
        raw = self._generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
        )
        self._last_generation_trace = (raw,)
        self._last_generation_trace_kinds = ("hunk-attempt-0",)
        self._last_generation_prompt_trace = (prompt,)
        self._last_generation_trace_results = ({"status": "generated"},)
        return raw
