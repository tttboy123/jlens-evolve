"""Round 1 binding from existing Student adapters to multi-realization search."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ContractError, LoopRevision, canonical_json, sha256_json
from .mlx_student import _lexical_terms
from .operator_student import parse_operator_plan_output
from .realization_adapter import DiagnosisFrozenRealizationAdapter
from .realization_candidates import FrozenDiagnosis
from .span_student import _semantic_code_tokens, parse_span_bundle_output
from .student_adapter import StudentAdapter, StudentAttempt, StudentTask
from .symbol_rewrite import qualified_symbol_for_issue

_MECHANISMS = {"operator", "span"}
_MAX_DIAGNOSIS_CHARS = 160
_SHARED_CONTEXT_FIELDS = frozenset(
    {
        "schema_version",
        "contract",
        "task_id",
        "cohort",
        "mechanism",
        "diagnosis",
        "target_files",
        "target_symbol",
        "target_symbols",
        "seed_raw_output",
        "seed_raw_output_sha256",
        "source_policy",
        "native_labels_visible",
        "reference_patch_visible",
        "network_calls_performed",
        "evidence_sha256",
    }
)


@dataclass(frozen=True)
class SharedDiagnosisLocalization:
    """Neutral diagnosis/localization consumed unchanged by both A/B arms."""

    task_id: str
    cohort: str
    mechanism: str
    diagnosis: FrozenDiagnosis
    target_files: tuple[str, ...]
    target_symbol: str | None
    target_symbols: tuple[str, ...]
    seed_raw_output: str
    source_policy: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SharedDiagnosisLocalization:
        if not isinstance(data, dict):
            raise ContractError("shared diagnosis/localization fields are invalid")
        expected = set(_SHARED_CONTEXT_FIELDS)
        if set(data) not in (expected, expected - {"target_symbols"}):
            raise ContractError("shared diagnosis/localization fields are invalid")
        diagnosis_data = data["diagnosis"]
        if not isinstance(diagnosis_data, dict) or set(diagnosis_data) != {
            "schema_version",
            "defect",
            "trigger",
            "desired_boundary",
            "fingerprint",
        }:
            raise ContractError("shared diagnosis fields are invalid")
        diagnosis = FrozenDiagnosis.create(
            defect=str(diagnosis_data["defect"]),
            trigger=str(diagnosis_data["trigger"]),
            desired_boundary=str(diagnosis_data["desired_boundary"]),
        )
        if diagnosis_data["fingerprint"] != diagnosis.fingerprint:
            raise ContractError("shared diagnosis fingerprint mismatch")
        files = data["target_files"]
        symbols = data.get("target_symbols")
        target_symbols = (
            tuple(str(row) for row in symbols)
            if isinstance(symbols, list) and symbols
            else (
                () if data["target_symbol"] is None else (str(data["target_symbol"]),)
            )
        )
        context = cls(
            task_id=str(data["task_id"]),
            cohort=str(data["cohort"]),
            mechanism=str(data["mechanism"]),
            diagnosis=diagnosis,
            target_files=(
                tuple(str(row) for row in files) if isinstance(files, list) else ()
            ),
            target_symbol=(
                None if data["target_symbol"] is None else str(data["target_symbol"])
            ),
            target_symbols=target_symbols,
            seed_raw_output=str(data["seed_raw_output"]),
            source_policy=str(data["source_policy"]),
        )
        context.validate()
        if data["schema_version"] != 1 or data["contract"] != context.contract:
            raise ContractError("unsupported shared diagnosis/localization contract")
        if data["seed_raw_output_sha256"] != context.seed_raw_output_sha256:
            raise ContractError("shared seed raw output sha256 mismatch")
        if (
            data["native_labels_visible"] is not False
            or data["reference_patch_visible"] is not False
            or data["network_calls_performed"] is not False
        ):
            raise ContractError("shared context crosses evaluator boundary")
        content = {
            key: value for key, value in data.items() if key != "evidence_sha256"
        }
        if data["evidence_sha256"] != sha256_json(content):
            raise ContractError("shared diagnosis/localization evidence mismatch")
        return context

    @property
    def contract(self) -> str:
        return "shared-diagnosis-localization-v1"

    @property
    def seed_raw_output_sha256(self) -> str:
        return hashlib.sha256(self.seed_raw_output.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        if not self.task_id or self.cohort not in {"feedback", "holdout"}:
            raise ContractError("shared context task boundary is invalid")
        if self.mechanism not in _MECHANISMS:
            raise ContractError("shared context mechanism is invalid")
        self.diagnosis.validate()
        if not 1 <= len(self.target_files) <= 2:
            raise ContractError("shared context requires one or two target files")
        if len(set(self.target_files)) != len(self.target_files):
            raise ContractError("shared context target files must be unique")
        for raw in self.target_files:
            path = Path(raw)
            if path.is_absolute() or not path.parts or ".." in path.parts:
                raise ContractError("shared context target file is invalid")
        if self.target_symbol is not None and not self.target_symbol.strip():
            raise ContractError("shared context target symbol is invalid")
        if self.target_symbols:
            for symbol in self.target_symbols:
                if not isinstance(symbol, str) or not symbol.strip():
                    raise ContractError("shared context target symbols are invalid")
            if self.target_symbol is None:
                raise ContractError("shared context target symbol must match symbols")
            if self.target_symbol not in self.target_symbols:
                raise ContractError(
                    "shared context target symbol must be within target symbols"
                )
        if self.source_policy not in {
            "neutral-plan-intent-localization-v1",
            "issue-first-target-fallback-v1",
            "issue-static-symbol-localization-v2",
            "issue-static-operator-localization-v3",
            "issue-static-file-localization-v3",
            "issue-static-file-localization-v4",
            "issue-static-file-localization-v5",
            "issue-static-file-localization-v6",
        }:
            raise ContractError("shared context source policy is invalid")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        content = {
            "schema_version": 1,
            "contract": self.contract,
            "task_id": self.task_id,
            "cohort": self.cohort,
            "mechanism": self.mechanism,
            "diagnosis": self.diagnosis.to_dict(),
            "target_files": list(self.target_files),
            "target_symbol": self.target_symbol,
            "target_symbols": list(self.target_symbols),
            "seed_raw_output": self.seed_raw_output,
            "seed_raw_output_sha256": self.seed_raw_output_sha256,
            "source_policy": self.source_policy,
            "native_labels_visible": False,
            "reference_patch_visible": False,
            "network_calls_performed": False,
        }
        return {**content, "evidence_sha256": sha256_json(content)}


class Round1SharedRealizationAdapter:
    """Prepare one neutral context, then compare only A/B repair realization."""

    def __init__(
        self,
        *,
        mechanism: str,
        base_adapter: StudentAdapter,
        maximum_candidates: int,
    ) -> None:
        if mechanism not in _MECHANISMS:
            raise ContractError("unsupported Round 1 realization mechanism")
        if type(maximum_candidates) is not int or not 1 <= maximum_candidates <= 8:
            raise ContractError("realization candidate budget must be between 1 and 8")
        self.mechanism = mechanism
        self.base_adapter = base_adapter
        self.maximum_candidates = maximum_candidates
        self.generator = self
        self._contexts: dict[str, SharedDiagnosisLocalization] = {}
        self._inner: DiagnosisFrozenRealizationAdapter | None = None
        self._generation_trace: list[str] = []
        self._generation_trace_kinds: list[str] = []
        self._prompt_trace: list[str | None] = []
        self._generation_trace_results: list[dict[str, Any]] = []

    def experiment_config(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "adapter_contract": "shared-diagnosis-localization-paired-repair-v5",
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "maximum_candidates": self.maximum_candidates,
            "shared_context_policy": (
                "issue-anchor-source-edge-top2-candidate-harvest-v6"
            ),
            "localization_drift": "reject",
            "diagnosis_echo_policy": "framework-owned-ignore-model-paraphrase-v1",
            "native_labels_visible_to_selection": False,
            "candidate_generation": self.base_adapter.experiment_config(),
        }

    def prepare_shared_context(
        self, task: StudentTask, revision: LoopRevision
    ) -> dict[str, Any]:
        task.validate()
        revision.validate()
        neutral = _neutral_revision(revision)
        attempt = self.base_adapter.run(task, neutral)
        context = _context_from_seed(
            mechanism=self.mechanism,
            task=task,
            raw_output=attempt.raw_output,
            seed_structural_valid=attempt.structural_valid,
        )
        self._contexts[task.task_id] = context
        return context.to_dict()

    def bind_shared_context(self, evidence: dict[str, Any]) -> None:
        context = SharedDiagnosisLocalization.from_dict(evidence)
        if context.mechanism != self.mechanism:
            raise ContractError("shared context mechanism does not match adapter")
        self._contexts[context.task_id] = context

    def run(self, task: StudentTask, revision: LoopRevision) -> StudentAttempt:
        task.validate()
        revision.validate()
        context = self._contexts.get(task.task_id)
        if context is None:
            raise ContractError("shared context was not bound before repair")
        if context.cohort != task.cohort or any(
            target not in task.allowed_targets for target in context.target_files
        ):
            raise ContractError("shared context does not match Student task")

        def diagnose(_task: StudentTask, _revision: LoopRevision) -> FrozenDiagnosis:
            return context.diagnosis

        def realize(
            candidate_task: StudentTask,
            candidate_revision: LoopRevision,
            diagnosis: FrozenDiagnosis,
            index: int,
        ) -> StudentAttempt:
            pinned = _pinned_shared_revision(candidate_revision, context, index)
            # The pinned revision owns the editable scope.  Keep the original
            # task here because realization adapters may still need the remaining
            # allowed files as read-only repository evidence (for example, state
            # writers and neutral enum values).  The concrete generator narrows
            # candidate harvest separately from that evidence scope.
            attempt = self.base_adapter.run(candidate_task, pinned)
            generator = getattr(self.base_adapter, "generator", None)
            trace_reader = getattr(generator, "generation_trace", None)
            kind_reader = getattr(generator, "generation_trace_kinds", None)
            prompt_reader = getattr(generator, "generation_prompt_trace", None)
            result_reader = getattr(generator, "generation_trace_results", None)
            traces = trace_reader() if trace_reader is not None else ()
            kinds = kind_reader() if kind_reader is not None else ()
            prompts = prompt_reader() if prompt_reader is not None else ()
            results = result_reader() if result_reader is not None else ()
            if (
                not isinstance(traces, tuple)
                or any(not isinstance(raw, str) for raw in traces)
                or not isinstance(kinds, tuple)
                or any(not isinstance(kind, str) or not kind for kind in kinds)
                or not isinstance(prompts, tuple)
                or not isinstance(results, tuple)
                or any(not isinstance(result, dict) for result in results)
            ):
                raise ContractError("base realization generation trace is invalid")
            if any(
                prompt is not None and not isinstance(prompt, str) for prompt in prompts
            ):
                raise ContractError("base realization prompt trace is invalid")
            lengths = {len(traces), len(kinds), len(prompts), len(results)}
            if traces and lengths != {len(traces)}:
                raise ContractError("base realization generation trace lengths differ")
            candidate_kind = f"realization-candidate-{index + 1:03d}"
            if traces:
                self._generation_trace.extend(traces)
                self._generation_trace_kinds.extend(
                    f"{candidate_kind}/{kind}" for kind in kinds
                )
                self._prompt_trace.extend(prompts)
                self._generation_trace_results.extend(
                    dict(result) for result in results
                )
            else:
                self._generation_trace.append(attempt.raw_output)
                self._generation_trace_kinds.append(candidate_kind)
                self._prompt_trace.append(prompts[-1] if prompts else None)
                self._generation_trace_results.append(
                    {
                        "status": (
                            "structural-valid"
                            if attempt.structural_valid
                            else "structural-rejected"
                        ),
                        **(
                            {}
                            if attempt.structural_valid
                            else {
                                "failure_reason": attempt.failure_reason,
                                "detail": attempt.detail,
                            }
                        ),
                    }
                )
            if attempt.failure_reason == "eval-infra":
                return attempt
            if not _candidate_localization_matches_context(
                self.mechanism, attempt.raw_output, context
            ):
                return StudentAdapter._failure(
                    candidate_task,
                    candidate_revision,
                    attempt.raw_output,
                    "unresolved",
                    "repair candidate changed shared localization",
                )
            return attempt

        self._generation_trace = []
        self._generation_trace_kinds = []
        self._prompt_trace = []
        self._generation_trace_results = []
        self._inner = DiagnosisFrozenRealizationAdapter(
            diagnosis_provider=diagnose,
            candidate_runner=realize,
            maximum_candidates=self.maximum_candidates,
            candidate_generation_config=self.experiment_config(),
        )
        return self._inner.run(task, revision)

    def realization_evidence(self) -> dict[str, Any] | None:
        return None if self._inner is None else self._inner.realization_evidence()

    def generation_trace(self) -> tuple[str, ...]:
        return () if self._inner is None else tuple(self._generation_trace)

    def generation_trace_kinds(self) -> tuple[str, ...]:
        return () if self._inner is None else tuple(self._generation_trace_kinds)

    def generation_prompt_trace(self) -> tuple[str | None, ...]:
        return () if self._inner is None else tuple(self._prompt_trace)

    def generation_trace_results(self) -> tuple[dict[str, Any], ...]:
        return () if self._inner is None else tuple(self._generation_trace_results)


def build_round1_shared_realization_adapter(
    *, mechanism: str, base_adapter: StudentAdapter, maximum_candidates: int
) -> Round1SharedRealizationAdapter:
    """Build the explicit shared-context A/B repair mechanism."""

    return Round1SharedRealizationAdapter(
        mechanism=mechanism,
        base_adapter=base_adapter,
        maximum_candidates=maximum_candidates,
    )


def extract_round1_frozen_diagnosis(
    *, mechanism: str, raw_output: str, task: StudentTask
) -> FrozenDiagnosis:
    """Extract a seed intent, with a deterministic issue-only fail-closed fallback."""

    if mechanism not in _MECHANISMS:
        raise ContractError("unsupported Round 1 realization mechanism")
    try:
        if mechanism == "operator":
            intent = parse_operator_plan_output(raw_output).intent.to_dict()
        else:
            bundle = parse_span_bundle_output(raw_output)
            intents = [plan.intent.to_dict() for plan in bundle.plans]
            if not intents or any(row != intents[0] for row in intents[1:]):
                raise ContractError("span candidate contains diagnosis drift")
            intent = intents[0]
        return FrozenDiagnosis.create(
            defect=_bounded(intent["defect"]),
            trigger=_bounded(intent["trigger"]),
            desired_boundary=_bounded(intent["desired_boundary"]),
        )
    except (ContractError, KeyError, TypeError):
        return _issue_fallback_diagnosis(task)


def build_round1_realization_adapter(
    *,
    mechanism: str,
    base_adapter: StudentAdapter,
    maximum_candidates: int,
) -> DiagnosisFrozenRealizationAdapter:
    """Reuse candidate 1's diagnosis and diversify later deterministic prompts."""

    if mechanism not in _MECHANISMS:
        raise ContractError("unsupported Round 1 realization mechanism")

    def seed_provider(
        task: StudentTask, revision: LoopRevision
    ) -> tuple[FrozenDiagnosis, StudentAttempt]:
        attempt = base_adapter.run(task, revision)
        diagnosis = extract_round1_frozen_diagnosis(
            mechanism=mechanism,
            raw_output=attempt.raw_output,
            task=task,
        )
        return diagnosis, attempt

    def candidate_runner(
        task: StudentTask,
        revision: LoopRevision,
        diagnosis: FrozenDiagnosis,
        index: int,
    ) -> StudentAttempt:
        pinned = _pinned_revision(revision, diagnosis, index)
        attempt = base_adapter.run(task, pinned)
        parsed = _try_extract(mechanism, attempt.raw_output, task)
        if parsed is not None and parsed.fingerprint != diagnosis.fingerprint:
            return StudentAdapter._failure(
                task,
                revision,
                attempt.raw_output,
                "unresolved",
                "realization candidate changed the frozen diagnosis",
            )
        return attempt

    return DiagnosisFrozenRealizationAdapter(
        seed_candidate_provider=seed_provider,
        candidate_runner=candidate_runner,
        maximum_candidates=maximum_candidates,
        candidate_generation_config={
            "mechanism": mechanism,
            "inner_adapter": base_adapter.experiment_config(),
            "seed_policy": "candidate-001-intent-or-issue-fallback-v1",
            "diversification_policy": "candidate-index-prompt-v1",
            "diagnosis_drift": "reject",
        },
    )


def _try_extract(
    mechanism: str, raw_output: str, task: StudentTask
) -> FrozenDiagnosis | None:
    try:
        if mechanism == "operator":
            parse_operator_plan_output(raw_output)
        else:
            parse_span_bundle_output(raw_output)
    except ContractError:
        return None
    return extract_round1_frozen_diagnosis(
        mechanism=mechanism, raw_output=raw_output, task=task
    )


def _issue_static_span_target(task: StudentTask) -> str:
    """Choose one gold-free source file from issue/path/source overlap."""

    return _issue_static_span_targets(task)[0]


def _issue_static_operator_target(task: StudentTask) -> str:
    """Choose one Python target from public issue/path overlap only."""

    if not task.allowed_targets:
        raise ContractError("operator static localization has no allowed target")
    aliases = {
        "classes": "class",
        "inherited": "inherit",
        "inherits": "inherit",
        "mocked": "mock",
        "mocking": "mock",
    }

    def normalized_tokens(value: str) -> set[str]:
        return {
            aliases.get(token, token)
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", value.casefold())
        }

    query = normalized_tokens(task.instruction)
    title = next((line for line in task.instruction.splitlines() if line.strip()), "")
    title_tokens = normalized_tokens(title)
    explicit_artifacts = tuple(
        value.strip()
        for value in re.findall(r"`([^`\n]{1,160})`", task.instruction)
        if value.strip() and ("/" in value or "." in value)
    )
    ranked: list[tuple[int, int, str]] = []
    for index, target in enumerate(task.allowed_targets):
        path_tokens = normalized_tokens(target)
        stem_tokens = normalized_tokens(Path(target).stem)
        overlap = len(query & path_tokens)
        title_stem_overlap = len(title_tokens & stem_tokens)
        issue_stem_overlap = len(query & stem_tokens)
        explicit_overlap = max(
            (
                len(normalized_tokens(value) & path_tokens)
                for value in explicit_artifacts
            ),
            default=0,
        )
        ranked.append(
            (
                title_stem_overlap * 10_000
                + issue_stem_overlap * 1_000
                + explicit_overlap * 100
                + overlap,
                -index,
                target,
            )
        )
    score, _order, target = max(ranked)
    return target if score > 0 else task.allowed_targets[0]


def _issue_static_span_targets(task: StudentTask) -> tuple[str, ...]:
    """Choose a bounded issue anchor plus a source-derived adjacent definition."""

    if not task.allowed_targets:
        raise ContractError("span static localization has no allowed target")
    query = _lexical_terms(task.instruction)
    semantic_query = _semantic_code_tokens(task.instruction)
    causal_vocabulary = {
        "apply",
        "clear",
        "compile",
        "convert",
        "create",
        "delete",
        "deserialize",
        "emit",
        "execute",
        "external",
        "generate",
        "hide",
        "inherit",
        "load",
        "parse",
        "process",
        "read",
        "receive",
        "remove",
        "render",
        "run",
        "save",
        "send",
        "serialize",
        "show",
        "spawn",
        "transform",
        "write",
    }
    causal_query = semantic_query.intersection(causal_vocabulary)
    exact_identifiers = set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", task.instruction))
    qualified_api_symbols = set(
        re.findall(
            r"\b(?:[A-Za-z_][A-Za-z0-9_]*::)+([A-Za-z_][A-Za-z0-9_]*)",
            task.instruction,
        )
    )
    explicit_artifacts = tuple(
        dict.fromkeys(
            value.strip()
            for value in re.findall(r"`([^`\n]{1,80})`", task.instruction)
            if value.strip()
        )
    )
    named_identifiers = set(
        re.findall(
            r"\b(?:[A-Z][A-Za-z0-9_]{2,}|[a-z][a-z0-9]+(?:_[a-z0-9]+)+)\b",
            task.instruction,
        )
    )
    title = next((line for line in task.instruction.splitlines() if line.strip()), "")
    title_identifiers = set(
        re.findall(
            r"\b(?:[A-Z][a-z][A-Za-z0-9_]{1,}|[a-z][a-z0-9]+(?:_[a-z0-9]+)+)\b",
            title,
        )
    )
    title_component_terms = {
        piece.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", title)
        for piece in re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", token).split("_")
        if len(piece) > 1
    }

    def compound_terms(value: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9]*", value)
            if len(token) > 1
        }

    def slug(value: str) -> str:
        return "".join(re.findall(r"[A-Za-z0-9]+", value)).casefold()

    issue_component_terms = {
        piece.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", task.instruction)
        for piece in re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", token).split("_")
        if len(piece) > 1
    }

    ranked: list[tuple[int, int, str, str]] = []
    for index, target in enumerate(task.allowed_targets):
        try:
            source = task.resolve_target(target).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
        path_terms = compound_terms(target)
        semantic_path_terms = _semantic_code_tokens(target)
        path_overlap = len(semantic_query & semantic_path_terms)
        target_stem = Path(target).stem
        target_stem_slug = slug(target_stem)
        qualified_stem_hits = sum(
            1
            for symbol in qualified_api_symbols
            if target_stem_slug
            and (
                slug(symbol).endswith(target_stem_slug)
                or target_stem_slug.endswith(slug(symbol))
            )
        )
        explicit_stem_hits = sum(
            1
            for artifact in explicit_artifacts
            if slug(artifact) and slug(artifact) == slug(target_stem)
        )
        explicit_path_hits = sum(
            len(artifact_terms) ** 2
            for artifact in explicit_artifacts
            if ("/" in artifact or "." in artifact)
            and len(artifact_terms := compound_terms(artifact)) >= 2
            and artifact_terms.issubset(path_terms)
        )
        named_stem_hits = sum(
            1
            for identifier in named_identifiers
            if target_stem_slug
            and (
                slug(identifier).endswith(target_stem_slug)
                or target_stem_slug.endswith(slug(identifier))
            )
        )
        named_source_hits = sum(
            1 for identifier in named_identifiers if identifier in source
        )
        title_stem_hits = sum(
            1
            for identifier in title_identifiers
            if target_stem_slug
            and (
                slug(identifier).endswith(target_stem_slug)
                or target_stem_slug.endswith(slug(identifier))
            )
        )
        title_source_hits = sum(
            1 for identifier in title_identifiers if identifier in source
        )
        explicit_source_hits = sum(
            1 for artifact in explicit_artifacts if artifact in source
        )
        source_overlap = len(query & _lexical_terms(source))
        semantic_source_terms = _semantic_code_tokens(source)
        causal_source_overlap = len(causal_query & semantic_source_terms)
        causal_path_overlap = len(causal_query & semantic_path_terms)
        identifier_hits = sum(
            1 for identifier in exact_identifiers if identifier in source
        )
        control_density = sum(
            source.count(marker)
            for marker in ("if (", "if ", "switch (", "return ", "find", "transform")
        )
        score = (
            explicit_stem_hits * (15_000 if len(causal_query) >= 3 else 50_000)
            + title_stem_hits * 60_000
            + title_source_hits * 60_000
            + named_stem_hits * 30_000
            + qualified_stem_hits * 25_000
            + explicit_path_hits * 20_000
            + explicit_source_hits * 30_000
            + named_source_hits * 6_000
            + identifier_hits * 1_000
            + causal_path_overlap * 15_000
            + causal_source_overlap * 8_000
            + path_overlap * 2_000
            + source_overlap * 100
            + min(control_density, 20)
        )
        ranked.append((score, -index, target, source))
    if not ranked:
        return (task.allowed_targets[0],)
    ranked.sort(reverse=True)
    _anchor_score, _anchor_index, anchor, anchor_source = ranked[0]

    qualified_references = [
        tuple(qualified.split("::"))
        for qualified in re.findall(
            r"\b[A-Z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)+",
            anchor_source,
        )
    ]
    referenced_types = {
        value for qualified in qualified_references for value in qualified
    }
    referenced_types.update(
        re.findall(
            r"\b(?:new\s+)?([A-Z][A-Za-z0-9_]{2,})(?=\.new\b|\s*\()",
            anchor_source,
        )
    )
    adjacent: list[tuple[int, int, str]] = []
    for _score, neg_index, target, source in ranked[1:]:
        stem = slug(Path(target).stem)
        edge_scores: list[int] = []
        for qualified in qualified_references:
            if all(
                re.search(
                    rf"\b(?:class|module|interface|struct|trait)\s+{re.escape(component)}\b",
                    source,
                )
                is not None
                for component in qualified
            ):
                relevant_components = sum(
                    component.casefold() in semantic_query
                    or component.casefold() in issue_component_terms
                    for component in qualified
                )
                title_relevant_components = sum(
                    component.casefold() in title_component_terms
                    for component in qualified
                )
                edge_scores.append(
                    100_000
                    + relevant_components * 200_000
                    + title_relevant_components * 400_000
                    + (300_000 if relevant_components == len(qualified) else 0)
                )
        for identifier in referenced_types:
            identifier_slug = slug(identifier)
            identifier_components = {
                piece.casefold()
                for piece in re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", identifier).split(
                    "_"
                )
                if len(piece) > 1
            }
            identifier_edge_score = 0
            definition = re.search(
                rf"\b(?:class|module|interface|struct|trait)\s+{re.escape(identifier)}\b",
                source,
            )
            matched_identifier = stem == identifier_slug or definition is not None
            if stem == identifier_slug:
                identifier_edge_score += 20_000
            if definition is not None:
                identifier_edge_score += 15_000
            if matched_identifier:
                identifier_edge_score += (
                    len(identifier_components & issue_component_terms) * 100_000
                )
            if identifier_edge_score:
                edge_scores.append(identifier_edge_score)
        edge_score = max(edge_scores, default=0)
        if edge_score:
            adjacent.append((edge_score, neg_index, target))
    if not adjacent:
        return (anchor,)
    adjacent.sort(reverse=True)
    return (anchor, adjacent[0][2])


def _context_from_seed(
    *,
    mechanism: str,
    task: StudentTask,
    raw_output: str,
    seed_structural_valid: bool,
) -> SharedDiagnosisLocalization:
    diagnosis = extract_round1_frozen_diagnosis(
        mechanism=mechanism, raw_output=raw_output, task=task
    )
    seed_localization_drift = False
    target_symbols: tuple[str, ...] = ()
    try:
        if mechanism == "operator":
            plan = parse_operator_plan_output(raw_output)
            target_files = (
                (plan.file,)
                if seed_structural_valid
                else (_issue_static_operator_target(task),)
            )
            source = task.resolve_target(target_files[0]).read_text(
                encoding="utf-8", errors="replace"
            )
            static_symbol = qualified_symbol_for_issue(source, task.instruction)
            target_symbol = static_symbol or plan.symbol
            target_symbols = (target_symbol,) if target_symbol else ()
            seed_localization_drift = (
                static_symbol is not None and static_symbol != plan.symbol
            )
        else:
            bundle = parse_span_bundle_output(raw_output)
            target_files = tuple(plan.file for plan in bundle.plans)
            target_symbol = None
            target_symbols = ()
            if not seed_structural_valid:
                target_files = _issue_static_span_targets(task)
        if any(target not in task.allowed_targets for target in target_files):
            raise ContractError("neutral seed selected a non-allowed target")
        source_policy = (
            "issue-static-operator-localization-v3"
            if mechanism == "operator" and not seed_structural_valid
            else (
                "issue-static-symbol-localization-v2"
                if mechanism == "operator" and static_symbol is not None
                else (
                    "issue-static-file-localization-v6"
                    if mechanism == "span" and not seed_structural_valid
                    else "neutral-plan-intent-localization-v1"
                )
            )
        )
    except ContractError:
        if not task.allowed_targets:
            raise ContractError(
                "shared context fallback has no allowed target"
            ) from None
        target_files = (
            _issue_static_span_targets(task)
            if mechanism == "span"
            else (_issue_static_operator_target(task),)
        )
        target_symbol = None
        target_symbols = ()
        if mechanism == "operator":
            source = task.resolve_target(target_files[0]).read_text(
                encoding="utf-8", errors="replace"
            )
            target_symbol = qualified_symbol_for_issue(source, task.instruction)
            target_symbols = (target_symbol,) if target_symbol else ()
        source_policy = (
            "issue-static-operator-localization-v3"
            if mechanism == "operator"
            else (
                "issue-static-file-localization-v6"
                if mechanism == "span"
                else "issue-first-target-fallback-v1"
            )
        )
    if not seed_structural_valid or seed_localization_drift:
        diagnosis = _issue_fallback_diagnosis(task)
    return SharedDiagnosisLocalization(
        task_id=task.task_id,
        cohort=task.cohort,
        mechanism=mechanism,
        diagnosis=diagnosis,
        target_files=target_files,
        target_symbol=target_symbol,
        target_symbols=target_symbols,
        seed_raw_output=raw_output,
        source_policy=source_policy,
    )


def _candidate_localization_matches_context(
    mechanism: str,
    raw_output: str,
    context: SharedDiagnosisLocalization,
) -> bool:
    try:
        if mechanism == "operator":
            plan = parse_operator_plan_output(raw_output)
            return plan.file == context.target_files[0] and (
                context.target_symbol is None or plan.symbol == context.target_symbol
            )
        bundle = parse_span_bundle_output(raw_output)
    except ContractError:
        return False
    selected = tuple(plan.file for plan in bundle.plans)
    return (
        bool(selected)
        and len(set(selected)) == len(selected)
        and all(file in context.target_files for file in selected)
    )


def _neutral_revision(revision: LoopRevision) -> LoopRevision:
    return LoopRevision.create(
        skill_id=revision.skill_id,
        revision_id=f"{revision.revision_id}-shared-neutral",
        parent_revision_id=revision.parent_revision_id,
        source_round=revision.source_round,
        protocol=revision.protocol,
        skill_text=(
            "No additional domain teaching is provided. Use only repository "
            "evidence to emit one source-anchored plan. This neutral plan is used "
            "only to freeze diagnosis and localization; its patch is not scored."
        ),
        prompt_template=revision.prompt_template,
        eval_note="Neutral shared diagnosis/localization preparation.",
    )


def _pinned_shared_revision(
    revision: LoopRevision,
    context: SharedDiagnosisLocalization,
    index: int,
) -> LoopRevision:
    pinned = canonical_json(
        {
            "diagnosis": context.diagnosis.to_dict(),
            "target_files": list(context.target_files),
            "target_symbol": context.target_symbol,
        }
    )
    return LoopRevision.create(
        skill_id=revision.skill_id,
        revision_id=revision.revision_id,
        parent_revision_id=revision.parent_revision_id,
        source_round=revision.source_round,
        protocol=revision.protocol,
        skill_text=(
            f"{revision.skill_text}\n\n"
            "## Shared diagnosis and localization (read-only)\n"
            f"{pinned}\n"
            f"Repair candidate: {index + 1}. Keep diagnosis, target files, and "
            "target symbol byte-for-byte equivalent; vary only the supported "
            "minimal implementation."
        ),
        prompt_template=revision.prompt_template,
        eval_note=revision.eval_note,
    )


def _pinned_revision(
    revision: LoopRevision, diagnosis: FrozenDiagnosis, index: int
) -> LoopRevision:
    diagnosis_text = canonical_json(
        {
            "defect": diagnosis.defect,
            "trigger": diagnosis.trigger,
            "desired_boundary": diagnosis.desired_boundary,
        }
    )
    return LoopRevision.create(
        skill_id=revision.skill_id,
        revision_id=revision.revision_id,
        parent_revision_id=revision.parent_revision_id,
        source_round=revision.source_round,
        protocol=revision.protocol,
        skill_text=(
            f"{revision.skill_text}\n\n"
            "## Frozen diagnosis (read-only)\n"
            f"{diagnosis_text}\n"
            f"Realization candidate: {index + 1}. Keep the diagnosis byte-for-byte "
            "equivalent and produce a distinct minimal supported implementation."
        ),
        prompt_template=revision.prompt_template,
        eval_note=revision.eval_note,
    )


def _bounded(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError("Round 1 diagnosis field must be non-empty")
    return value.strip()[:_MAX_DIAGNOSIS_CHARS]


def _public_issue_section_first_line(instruction: str, heading: str) -> str | None:
    """Read one public prose line from a Markdown issue section."""

    active = False
    expected = heading.casefold()
    for raw in instruction.splitlines():
        line = raw.strip()
        normalized = line.strip("#* _:.").casefold()
        if not active:
            if normalized == expected:
                active = True
            continue
        if not line or line.startswith("```"):
            continue
        if line.startswith("#") or (
            line.startswith("**") and line.rstrip(":").endswith("**")
        ):
            return None
        if normalized in {"no response", "none", "not applicable", "n/a"}:
            return None
        return line
    return None


def _issue_fallback_diagnosis(task: StudentTask) -> FrozenDiagnosis:
    issue_title = next(
        (line.strip() for line in task.instruction.splitlines() if line.strip()),
        "feedback task behavior is incorrect",
    )
    expected_behavior = _public_issue_section_first_line(
        task.instruction, "expected behavior"
    )
    return FrozenDiagnosis.create(
        defect=_bounded(issue_title),
        trigger="behavior described by the frozen feedback task",
        desired_boundary=_bounded(
            expected_behavior or "satisfy the task while preserving unrelated behavior"
        ),
    )
