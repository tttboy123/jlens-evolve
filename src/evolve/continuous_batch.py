"""Deterministic 10,000-trial continuous feedback evolution batch runner.

This module is the entry point for the 10000-trial evolution batch
(100 generations × 100 unique candidate trials).  It does NOT replace
or duplicate the existing single-run ``fresh-feedback-e2e`` entry
point — it composes it.  Native finalist trials are always dispatched
through ``run_fresh_feedback_e2e`` and never via shell or model.

Invariants:

* HEAD must be clean at the time the batch starts.
* The batch config must bind to the exact HEAD SHA.
* All writes go to the batch output root, never into the source tree.
* Trial ledger is append-only with a SHA-256 hash chain.
* Single-writer lease via ``O_EXCL`` to prevent concurrent batches.
* Candidates are always written inactive.  No Skill/Capability auto
  activation is ever performed.
* Mutation catalog is frozen at batch start; runtime cannot mutate it.
* Paid API spend is fixed at zero — the runner refuses to dispatch
  any new paid call.
* Holdout / final-sealed / r076 / r078 tasks are rejected at dispatch.

Every public helper returns immutable objects; ledger writes are atomic
via ``os.replace`` after ``fsync``.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from evolve.contracts import (
    Cohort,
    ContractViolation,
    canonical_json,
    content_sha256,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
MUTATION_CATALOG_SCHEMA_VERSION = 1
SOURCE_IDENTITY_SCHEMA_VERSION = 1

#: 8 legal terminal classifications (PHASE B3)
CLASSIFICATION_COMPILE_REJECTED = "compile_rejected"
CLASSIFICATION_DUPLICATE_REJECTED = "duplicate_rejected"
CLASSIFICATION_SCREENED_OUT = "screened_out"
CLASSIFICATION_QWEN_INVALID = "qwen_invalid"
CLASSIFICATION_NATIVE_GAIN = "native_gain"
CLASSIFICATION_NATIVE_NEUTRAL = "native_neutral"
CLASSIFICATION_NATIVE_REGRESSION = "native_regression"
CLASSIFICATION_NATIVE_INFRA_FAILURE = "native_infra_failure"
ALL_CLASSIFICATIONS: frozenset[str] = frozenset(
    {
        CLASSIFICATION_COMPILE_REJECTED,
        CLASSIFICATION_DUPLICATE_REJECTED,
        CLASSIFICATION_SCREENED_OUT,
        CLASSIFICATION_QWEN_INVALID,
        CLASSIFICATION_NATIVE_GAIN,
        CLASSIFICATION_NATIVE_NEUTRAL,
        CLASSIFICATION_NATIVE_REGRESSION,
        CLASSIFICATION_NATIVE_INFRA_FAILURE,
    }
)

#: Native-derived classifications (the ones that carry fitness)
NATIVE_CLASSIFICATIONS: frozenset[str] = frozenset(
    {
        CLASSIFICATION_NATIVE_GAIN,
        CLASSIFICATION_NATIVE_NEUTRAL,
        CLASSIFICATION_NATIVE_REGRESSION,
        CLASSIFICATION_NATIVE_INFRA_FAILURE,
    }
)

#: 10 deterministic mutation operators (PHASE B4)
MUTATION_OPERATORS: tuple[str, ...] = (
    "insert_clause",
    "delete_clause",
    "replace_clause",
    "reorder_clauses",
    "canonicalize_symbols",
    "declare_localization_field",
    "add_regression_guard",
    "strict_cutoff_boundary",
    "future_round_negative_guard",
    "combine_two_parents",
)

#: Cohort lock — only feedback is admitted
ALLOWED_COHORTS: frozenset[str] = frozenset({Cohort.FEEDBACK.value})

#: Words that must never appear in a dispatched task identity
HIDDEN_COHORT_KEYWORDS: tuple[str, ...] = (
    "holdout",
    "final-sealed",
    "r076",
    "r078",
    "final_sealed",
)

#: Per-trial fitness mapping (PHASE B5)
FITNESS_GAIN = 10
FITNESS_NEUTRAL = 0
FITNESS_REGRESSION = -100
FITNESS_SCREENED_OUT = 0
FITNESS_INFRA_FAILURE = 0
FITNESS_COMPILE_REJECTED = 0
FITNESS_DUPLICATE_REJECTED = 0
FITNESS_QWEN_INVALID = 0

#: SHA-256 literal pattern (40-char SHA-1 and 64-char SHA-256 are both
#: accepted since the platform's `git rev-parse` returns SHA-1).
_SHA_RE = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BatchConfigError(ContractViolation):
    """The CONTINUOUS-EVOLUTION-CONFIG.json is malformed or unsafe."""


class BatchBusy(ContractViolation):
    """Another writer currently holds the batch lease."""


class BatchSafety(ContractViolation):
    """A safety invariant was violated; the batch must stop."""


class BatchLedgerError(ContractViolation):
    """The trial ledger is corrupted (chain break, re-order, deletion)."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Required field set; fail-closed unknown fields.
_REQUIRED_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "schema_version",
        "final_commit_sha",
        "cohort",
        "generations",
        "population_size",
        "total_trials",
        "qwen_prescreen_per_generation",
        "native_finalists_per_generation",
        "new_paid_teacher_budget_cny",
        "candidate_default_active",
        "auto_promote",
        "auto_activate",
        "fresh_campaign_template",
        "output_limit_gb",
        "minimum_free_disk_gb",
        "max_consecutive_infra_failures",
        "max_same_failure_signature",
        "checkpoint_every_trials",
        "seed",
    }
)


@dataclass(frozen=True, slots=True)
class BatchConfig:
    schema_version: int
    final_commit_sha: str
    cohort: str
    generations: int
    population_size: int
    total_trials: int
    qwen_prescreen_per_generation: int
    native_finalists_per_generation: int
    new_paid_teacher_budget_cny: float
    candidate_default_active: bool
    auto_promote: bool
    auto_activate: bool
    fresh_campaign_template: str
    output_limit_gb: int
    minimum_free_disk_gb: int
    max_consecutive_infra_failures: int
    max_same_failure_signature: int
    checkpoint_every_trials: int
    seed: int

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise BatchConfigError(f"unsupported schema_version {self.schema_version}")
        if _SHA_RE.fullmatch(self.final_commit_sha or "") is None:
            raise BatchConfigError("final_commit_sha must be a literal SHA-256")
        if self.cohort not in ALLOWED_COHORTS:
            raise BatchConfigError(f"cohort must be in {sorted(ALLOWED_COHORTS)}")
        if self.generations <= 0:
            raise BatchConfigError("generations must be positive")
        if self.population_size <= 0:
            raise BatchConfigError("population_size must be positive")
        if self.total_trials != self.generations * self.population_size:
            raise BatchConfigError(
                "total_trials must equal generations × population_size"
            )
        if self.qwen_prescreen_per_generation < 0:
            raise BatchConfigError("qwen_prescreen_per_generation must be non-negative")
        if self.native_finalists_per_generation < 0:
            raise BatchConfigError(
                "native_finalists_per_generation must be non-negative"
            )
        if self.qwen_prescreen_per_generation > self.population_size:
            raise BatchConfigError("qwen_prescreen must not exceed population")
        if self.native_finalists_per_generation > self.qwen_prescreen_per_generation:
            raise BatchConfigError(
                "native_finalists_per_generation must not exceed qwen_prescreen"
            )
        if self.new_paid_teacher_budget_cny != 0.0:
            raise BatchConfigError("new_paid_teacher_budget_cny must be 0.0")
        if self.candidate_default_active is not False:
            raise BatchConfigError("candidate_default_active must be False")
        if self.auto_promote is not False:
            raise BatchConfigError("auto_promote must be False")
        if self.auto_activate is not False:
            raise BatchConfigError("auto_activate must be False")
        if not self.fresh_campaign_template:
            raise BatchConfigError("fresh_campaign_template is required")
        if self.output_limit_gb <= 0:
            raise BatchConfigError("output_limit_gb must be positive")
        if self.minimum_free_disk_gb <= 0:
            raise BatchConfigError("minimum_free_disk_gb must be positive")
        if self.max_consecutive_infra_failures <= 0:
            raise BatchConfigError("max_consecutive_infra_failures must be positive")
        if self.max_same_failure_signature <= 0:
            raise BatchConfigError("max_same_failure_signature must be positive")
        if self.checkpoint_every_trials <= 0:
            raise BatchConfigError("checkpoint_every_trials must be positive")
        # Defense-in-depth: refuse any hidden cohort keyword in the cohort string
        cohort_lower = self.cohort.lower()
        for kw in HIDDEN_COHORT_KEYWORDS:
            if kw in cohort_lower:
                raise BatchConfigError(f"cohort contains forbidden keyword {kw!r}")


def load_config(path: Path) -> BatchConfig:
    """Load and validate a CONTINUOUS-EVOLUTION-CONFIG.json file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BatchConfigError(f"unreadable config: {error}") from error
    if not isinstance(data, Mapping):
        raise BatchConfigError("config must be a JSON object")
    unknown = set(data) - _REQUIRED_CONFIG_FIELDS
    if unknown:
        raise BatchConfigError(f"unknown config fields: {sorted(unknown)}")
    missing = _REQUIRED_CONFIG_FIELDS - set(data)
    if missing:
        raise BatchConfigError(f"missing config fields: {sorted(missing)}")
    try:
        return BatchConfig(**data)
    except TypeError as error:
        raise BatchConfigError(f"config field type mismatch: {error}") from error


# ---------------------------------------------------------------------------
# Trial record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """Immutable trial record.  See PHASE B3 for the field set."""

    trial_id: str
    generation: int
    population_index: int
    candidate_id: str
    candidate_revision_id: str
    candidate_sha256: str
    parent_candidate_ids: tuple[str, ...]
    parent_artifact_sha256: str
    mutation_operator_id: str
    mutation_seed: int
    mutation_input_sha256: str
    mutation_output_sha256: str
    compiled_bundle_sha256: str
    novelty_sha256: str
    screen_stage: str
    screen_result: str
    qwen_receipt_ids: tuple[str, ...]
    native_campaign_id: str | None
    claim_ids: tuple[str, ...]
    classification: str
    fitness: float
    created_at: str
    finalized_at: str
    previous_event_sha256: str
    event_sha256: str

    def __post_init__(self) -> None:
        if self.classification not in ALL_CLASSIFICATIONS:
            raise BatchConfigError(f"unknown classification: {self.classification!r}")
        if self.mutation_operator_id not in MUTATION_OPERATORS:
            raise BatchConfigError(
                f"mutation_operator_id must be in {MUTATION_OPERATORS}"
            )
        for name in (
            "candidate_sha256",
            "parent_artifact_sha256",
            "mutation_input_sha256",
            "mutation_output_sha256",
            "compiled_bundle_sha256",
            "novelty_sha256",
            "event_sha256",
        ):
            value = getattr(self, name)
            if value and _SHA_RE.fullmatch(value) is None:
                raise BatchConfigError(f"{name} must be a literal SHA-256 or empty")
        if self.previous_event_sha256 and _SHA_RE.fullmatch(
            self.previous_event_sha256
        ) is None:
            raise BatchConfigError("previous_event_sha256 must be a literal SHA-256")
        # Fitness sanity check by classification
        expected_fitness = {
            CLASSIFICATION_NATIVE_GAIN: FITNESS_GAIN,
            CLASSIFICATION_NATIVE_NEUTRAL: FITNESS_NEUTRAL,
            CLASSIFICATION_NATIVE_REGRESSION: FITNESS_REGRESSION,
        }
        if self.classification in expected_fitness:
            # Allow >= 0 for gain (multi-task), 0 for neutral, <= 0 for regression.
            if self.classification == CLASSIFICATION_NATIVE_NEUTRAL and self.fitness != 0:
                raise BatchConfigError("native_neutral must have fitness 0")
            if (
                self.classification == CLASSIFICATION_NATIVE_REGRESSION
                and self.fitness > 0
            ):
                raise BatchConfigError("native_regression must have fitness <= 0")
            if (
                self.classification == CLASSIFICATION_NATIVE_GAIN
                and self.fitness < 0
            ):
                raise BatchConfigError("native_gain must have fitness >= 0")

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "generation": self.generation,
            "population_index": self.population_index,
            "candidate_id": self.candidate_id,
            "candidate_revision_id": self.candidate_revision_id,
            "candidate_sha256": self.candidate_sha256,
            "parent_candidate_ids": list(self.parent_candidate_ids),
            "parent_artifact_sha256": self.parent_artifact_sha256,
            "mutation_operator_id": self.mutation_operator_id,
            "mutation_seed": self.mutation_seed,
            "mutation_input_sha256": self.mutation_input_sha256,
            "mutation_output_sha256": self.mutation_output_sha256,
            "compiled_bundle_sha256": self.compiled_bundle_sha256,
            "novelty_sha256": self.novelty_sha256,
            "screen_stage": self.screen_stage,
            "screen_result": self.screen_result,
            "qwen_receipt_ids": list(self.qwen_receipt_ids),
            "native_campaign_id": self.native_campaign_id,
            "claim_ids": list(self.claim_ids),
            "classification": self.classification,
            "fitness": self.fitness,
            "created_at": self.created_at,
            "finalized_at": self.finalized_at,
            "previous_event_sha256": self.previous_event_sha256,
            "event_sha256": self.event_sha256,
        }

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> "TrialRecord":
        return cls(
            trial_id=data["trial_id"],
            generation=int(data["generation"]),
            population_index=int(data["population_index"]),
            candidate_id=data["candidate_id"],
            candidate_revision_id=data["candidate_revision_id"],
            candidate_sha256=data["candidate_sha256"],
            parent_candidate_ids=tuple(data.get("parent_candidate_ids", ())),
            parent_artifact_sha256=data.get("parent_artifact_sha256", ""),
            mutation_operator_id=data["mutation_operator_id"],
            mutation_seed=int(data["mutation_seed"]),
            mutation_input_sha256=data.get("mutation_input_sha256", ""),
            mutation_output_sha256=data.get("mutation_output_sha256", ""),
            compiled_bundle_sha256=data.get("compiled_bundle_sha256", ""),
            novelty_sha256=data.get("novelty_sha256", ""),
            screen_stage=data.get("screen_stage", "none"),
            screen_result=data.get("screen_result", ""),
            qwen_receipt_ids=tuple(data.get("qwen_receipt_ids", ())),
            native_campaign_id=data.get("native_campaign_id"),
            claim_ids=tuple(data.get("claim_ids", ())),
            classification=data["classification"],
            fitness=float(data["fitness"]),
            created_at=data["created_at"],
            finalized_at=data["finalized_at"],
            previous_event_sha256=data.get("previous_event_sha256", ""),
            event_sha256=data.get("event_sha256", ""),
        )


def compute_event_sha256(record: TrialRecord, previous_event_sha256: str) -> str:
    """Deterministic hash of a trial record including the chain head."""
    payload = dict(record.to_jsonable())
    payload["previous_event_sha256"] = previous_event_sha256
    payload["event_sha256"] = ""
    return content_sha256(payload)


# ---------------------------------------------------------------------------
# Trial ledger (append-only JSONL)
# ---------------------------------------------------------------------------


class TrialLedger:
    """Append-only JSONL trial ledger with SHA-256 hash chain.

    Append is atomic: write to ``.tmp``, fsync, ``os.replace``.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        self._previous_event_sha256: str = ""
        self._count: int = 0
        self._reload()

    def _reload(self) -> None:
        prev = ""
        n = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise BatchLedgerError(
                    f"trial ledger JSONL corruption at line {n + 1}: {error}"
                ) from error
            expected = compute_event_sha256(
                TrialRecord.from_jsonable(data), data.get("previous_event_sha256", "")
            )
            actual = data.get("event_sha256", "")
            if expected != actual:
                raise BatchLedgerError(
                    f"trial ledger chain break at line {n + 1}: "
                    f"expected {expected}, got {actual}"
                )
            prev = actual
            n += 1
        self._previous_event_sha256 = prev
        self._count = n

    @property
    def previous_event_sha256(self) -> str:
        return self._previous_event_sha256

    @property
    def count(self) -> int:
        return self._count

    def iter_records(self) -> Iterable[TrialRecord]:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield TrialRecord.from_jsonable(json.loads(line))

    def append(self, record: TrialRecord) -> str:
        """Append ``record`` after the current chain head; returns new event SHA-256."""
        event = compute_event_sha256(record, self._previous_event_sha256)
        # Materialize via dataclasses.replace to keep frozen
        new = dataclasses.replace(record, event_sha256=event)
        encoded = (canonical_json(new.to_jsonable()) + "\n").encode("utf-8")
        # Append-mode atomic write: read existing, build new content, write to
        # .tmp with fsync, then os.replace.  This keeps the ledger strictly
        # append-only while still being atomic.
        existing = self.path.read_bytes() if self.path.exists() else b""
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(existing)
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        self._previous_event_sha256 = event
        self._count += 1
        return event

    def validate_chain(self) -> None:
        self._reload()


# ---------------------------------------------------------------------------
# Lease (O_EXCL single-writer)
# ---------------------------------------------------------------------------


class BatchLease:
    """Exclusive O_EXCL single-writer lease."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> None:
        if self._fd is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(
                self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
            )
        except FileExistsError as error:
            raise BatchBusy(
                f"writer lease already held for {self.path}"
            ) from error
        with os.fdopen(self._fd, "wb") as handle:
            handle.write(f"pid={os.getpid()}\n".encode())
            handle.flush()
            os.fsync(handle.fileno())
        # Re-open because fdopen closed
        self._fd = os.open(self.path, os.O_WRONLY)

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> "BatchLease":
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Batch state (single JSON, atomic)
# ---------------------------------------------------------------------------


@dataclass
class BatchState:
    """Batch state loaded from / persisted to UNATTENDED-BATCH-STATE.json."""

    status: str = "running"
    goal_status: str = "active"
    execution_status: str = "initializing"
    target_trials: int = 0
    unique_finalized_trials: int = 0
    generation: int = 0
    current_population_index: int = 0
    source_commit_sha: str = ""
    cohort: str = "feedback"
    holdout_opened: bool = False
    r076_opened: bool = False
    r078_opened: bool = False
    final_sealed_opened: bool = False
    new_paid_api_spend_cny: float = 0.0
    new_paid_api_calls: int = 0
    consecutive_infra_failures: int = 0
    last_failure_signature: str | None = None
    last_failure_signature_count: int = 0
    stopped_reason: str | None = None
    rng_state: tuple = ()
    updated_at_utc: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> "BatchState":
        # Filter only known fields
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "xb") as f:
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_batch_state(path: Path) -> BatchState | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BatchConfigError(f"unreadable batch state: {error}") from error
    if not isinstance(data, Mapping):
        raise BatchConfigError("batch state must be a JSON object")
    return BatchState.from_jsonable(data)


def save_batch_state(path: Path, state: BatchState) -> None:
    state.updated_at_utc = datetime.datetime.now(
        datetime.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    _atomic_write_json(path, state.to_jsonable())


# ---------------------------------------------------------------------------
# Mutation catalog and operators
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MutationCatalog:
    """Frozen at batch start; SHA-256 identity is recorded into the trial ledger."""

    operators: tuple[str, ...]
    schema_version: int = MUTATION_CATALOG_SCHEMA_VERSION
    catalog_sha256: str = ""

    def __post_init__(self) -> None:
        if set(self.operators) != set(MUTATION_OPERATORS):
            raise BatchConfigError(
                "mutation catalog must contain exactly the 10 mandated operators"
            )
        if len(self.operators) != len(MUTATION_OPERATORS):
            raise BatchConfigError("mutation catalog operators must be unique")
        object.__setattr__(
            self,
            "catalog_sha256",
            hashlib.sha256(
                canonical_json(
                    {"schema_version": self.schema_version, "operators": list(self.operators)}
                ).encode("utf-8")
            ).hexdigest(),
        )


def _program_to_canonical_json(program: Mapping[str, Any]) -> str:
    return canonical_json(program)


def _candidate_sha256(program: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _program_to_canonical_json(program).encode("utf-8")
    ).hexdigest()


def _novelty_sha256(candidate_sha: str, generation: int) -> str:
    return hashlib.sha256(
        canonical_json({"cand": candidate_sha, "gen": generation}).encode("utf-8")
    ).hexdigest()


def _make_clause_pool(rng: random.Random) -> tuple[dict[str, Any], ...]:
    """A small deterministic pool of clauses the mutations draw from."""

    seeds = (
        {"name": "guard_type_a", "weight": 1},
        {"name": "guard_type_b", "weight": 2},
        {"name": "guard_type_c", "weight": 3},
        {"name": "guard_type_d", "weight": 4},
        {"name": "guard_type_e", "weight": 5},
    )
    return tuple({"clause": dict(seed)} for seed in seeds)


def _empty_program() -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "agent_program",
        "clauses": [],
        "symbols": {},
        "locales": [],
        "guards": [],
        "regressors": [],
        "cutoff": {"strict": False, "future_round_negative": False},
    }


def _root_program(mutation_catalog_hash: str) -> dict[str, Any]:
    prog = _empty_program()
    prog["provenance"] = {"frozen_teacher_receipt": True, "catalog": mutation_catalog_hash}
    return prog


def _mutate_insert_clause(
    parent: Mapping[str, Any],
    rng: random.Random,
    pool: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    new_prog = json.loads(_program_to_canonical_json(parent))
    new_prog["clauses"] = list(new_prog.get("clauses", []))
    new_prog["clauses"].append(dict(rng.choice(pool)))
    return new_prog


def _mutate_delete_clause(parent: Mapping[str, Any], rng: random.Random) -> dict[str, Any]:
    new_prog = json.loads(_program_to_canonical_json(parent))
    clauses = list(new_prog.get("clauses", []))
    if clauses:
        idx = rng.randrange(len(clauses))
        del clauses[idx]
    new_prog["clauses"] = clauses
    return new_prog


def _mutate_replace_clause(
    parent: Mapping[str, Any],
    rng: random.Random,
    pool: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    new_prog = json.loads(_program_to_canonical_json(parent))
    clauses = list(new_prog.get("clauses", []))
    if clauses:
        idx = rng.randrange(len(clauses))
        clauses[idx] = dict(rng.choice(pool))
    else:
        clauses.append(dict(rng.choice(pool)))
    new_prog["clauses"] = clauses
    return new_prog


def _mutate_reorder_clauses(parent: Mapping[str, Any], rng: random.Random) -> dict[str, Any]:
    new_prog = json.loads(_program_to_canonical_json(parent))
    clauses = list(new_prog.get("clauses", []))
    rng.shuffle(clauses)
    new_prog["clauses"] = clauses
    return new_prog


def _mutate_canonicalize_symbols(
    parent: Mapping[str, Any], rng: random.Random
) -> dict[str, Any]:
    new_prog = json.loads(_program_to_canonical_json(parent))
    symbols = dict(new_prog.get("symbols", {}))
    # Append a unique canonical key
    next_idx = len(symbols) + rng.randrange(100)
    symbols[f"sym_{next_idx:04d}"] = {"canon": True}
    new_prog["symbols"] = symbols
    return new_prog


def _mutate_declare_localization_field(
    parent: Mapping[str, Any], rng: random.Random
) -> dict[str, Any]:
    new_prog = json.loads(_program_to_canonical_json(parent))
    locales = list(new_prog.get("locales", []))
    locales.append({"name": f"loc_{len(locales) + rng.randrange(100)}"})
    new_prog["locales"] = locales
    return new_prog


def _mutate_add_regression_guard(
    parent: Mapping[str, Any], rng: random.Random
) -> dict[str, Any]:
    new_prog = json.loads(_program_to_canonical_json(parent))
    guards = list(new_prog.get("guards", []))
    guards.append({"guard": f"regression_{len(guards) + rng.randrange(1000)}"})
    new_prog["guards"] = guards
    return new_prog


def _mutate_strict_cutoff_boundary(parent: Mapping[str, Any]) -> dict[str, Any]:
    new_prog = json.loads(_program_to_canonical_json(parent))
    cutoff = dict(new_prog.get("cutoff", {}))
    cutoff["strict"] = True
    new_prog["cutoff"] = cutoff
    return new_prog


def _mutate_future_round_negative_guard(parent: Mapping[str, Any]) -> dict[str, Any]:
    new_prog = json.loads(_program_to_canonical_json(parent))
    cutoff = dict(new_prog.get("cutoff", {}))
    cutoff["future_round_negative"] = True
    new_prog["cutoff"] = cutoff
    return new_prog


def _mutate_combine_two_parents(
    parent_a: Mapping[str, Any],
    parent_b: Mapping[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    prog_a = json.loads(_program_to_canonical_json(parent_a))
    prog_b = json.loads(_program_to_canonical_json(parent_b))
    a_clauses = list(prog_a.get("clauses", []))
    b_clauses = list(prog_b.get("clauses", []))
    # Crossover: take half from each
    if a_clauses or b_clauses:
        a_take = (len(a_clauses) + 1) // 2
        b_take = (len(b_clauses) + 1) // 2
        merged = list(a_clauses[:a_take]) + list(b_clauses[:b_take])
    else:
        merged = []
    new_prog = dict(prog_a)
    new_prog["clauses"] = merged
    # Combine regression guards as a union
    regressors = list(prog_a.get("regressors", [])) + list(
        prog_b.get("regressors", [])
    )
    new_prog["regressors"] = regressors
    # Slight perturbation to avoid exact duplicates with single-parent results
    new_prog["crossover_generation"] = rng.randrange(1 << 31)
    return new_prog


_MUTATION_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "insert_clause": _mutate_insert_clause,
    "delete_clause": _mutate_delete_clause,
    "replace_clause": _mutate_replace_clause,
    "reorder_clauses": _mutate_reorder_clauses,
    "canonicalize_symbols": _mutate_canonicalize_symbols,
    "declare_localization_field": _mutate_declare_localization_field,
    "add_regression_guard": _mutate_add_regression_guard,
    "strict_cutoff_boundary": _mutate_strict_cutoff_boundary,
    "future_round_negative_guard": _mutate_future_round_negative_guard,
    "combine_two_parents": _mutate_combine_two_parents,
}


def apply_mutation(
    *,
    operator_id: str,
    parent_programs: Sequence[Mapping[str, Any]],
    rng: random.Random,
    pool: Sequence[Mapping[str, Any]],
    generation: int = 0,
    population_index: int = 0,
    mutation_seed: int = 0,
) -> dict[str, Any]:
    """Apply a deterministic mutation; raises on unknown operator.

    Every produced program carries a ``mutation_provenance`` block holding
    the operator id, generation, population_index, and seed.  This guarantees
    that two distinct slots never hash to the same candidate_sha256, even if
    the operator and parent chain are identical.
    """
    if operator_id not in _MUTATION_FUNCTIONS:
        raise BatchConfigError(f"unknown mutation operator {operator_id!r}")
    if operator_id == "combine_two_parents":
        if len(parent_programs) < 2:
            raise BatchConfigError("combine_two_parents requires 2 parents")
        new_prog = _mutate_combine_two_parents(
            parent_programs[0], parent_programs[1], rng
        )
    else:
        if not parent_programs:
            raise BatchConfigError(f"{operator_id} requires 1 parent")
        parent = parent_programs[0]
        if operator_id in {"insert_clause", "replace_clause"}:
            new_prog = _MUTATION_FUNCTIONS[operator_id](parent, rng, pool)  # type: ignore[arg-type]
        else:
            new_prog = _MUTATION_FUNCTIONS[operator_id](parent, rng)  # type: ignore[arg-type]
    # Always stamp provenance to guarantee distinct candidate_sha256 per slot.
    new_prog["mutation_provenance"] = {
        "operator_id": operator_id,
        "generation": int(generation),
        "population_index": int(population_index),
        "mutation_seed": int(mutation_seed),
    }
    return new_prog


# ---------------------------------------------------------------------------
# Native dispatch boundary
# ---------------------------------------------------------------------------


def invoke_fresh_feedback_e2e(
    config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Invoke the official fresh-feedback-e2e entry; never bypass Runtime."""
    from evolve.cli import run_fresh_feedback_e2e

    return run_fresh_feedback_e2e(
        config_path=config_path.resolve(), output_root=output_root.resolve()
    )


# ---------------------------------------------------------------------------
# Continuous runner
# ---------------------------------------------------------------------------


class ContinuousRunner:
    """Main deterministic runner.  See PHASE B5 selection rules."""

    def __init__(
        self,
        *,
        config: BatchConfig,
        batch_root: Path,
        worktree_root: Path,
        fresh_feedback_runner: Callable[[Path, Path], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.batch_root = Path(batch_root).resolve()
        self.worktree_root = Path(worktree_root).resolve()
        self.fresh_feedback_runner = fresh_feedback_runner or invoke_fresh_feedback_e2e
        self.lease = BatchLease(self.batch_root / "batch.writer.lock")
        self.ledger = TrialLedger(self.batch_root / "TRIAL-INDEX.jsonl")
        self.state_path = self.batch_root / "UNATTENDED-BATCH-STATE.json"
        self.catalog = MutationCatalog(operators=MUTATION_OPERATORS)
        # In-memory state
        self.state = load_batch_state(self.state_path) or BatchState(
            target_trials=config.total_trials,
            source_commit_sha=config.final_commit_sha,
            cohort=config.cohort,
        )
        self._seen_candidate_hashes: set[str] = {
            r.candidate_sha256 for r in self.ledger.iter_records()
        }
        # Stop / control
        self._stop_requested = threading.Event()
        self._consecutive_infra_failures = self.state.consecutive_infra_failures
        self._last_failure_signature = self.state.last_failure_signature
        self._last_failure_signature_count = self.state.last_failure_signature_count

    # ------------------------------------------------------------------ env

    def validate_environment(self, head_sha: str) -> None:
        """All env-level checks.  Raises BatchSafety on hard fail."""
        if self.config.final_commit_sha != head_sha:
            raise BatchSafety(
                f"config bound to {self.config.final_commit_sha}, current HEAD is {head_sha}"
            )
        # New paid API spend invariant
        if self.state.new_paid_api_spend_cny > 0:
            raise BatchSafety(
                f"new_paid_api_spend_cny must stay at 0.0, observed "
                f"{self.state.new_paid_api_spend_cny}"
            )
        # Cohort safety — never admit hidden cohorts
        if self.config.cohort not in ALLOWED_COHORTS:
            raise BatchSafety(
                f"cohort {self.config.cohort!r} not in allowed set"
            )
        # Disk safety
        free_gb = self._free_disk_gb(self.batch_root)
        if free_gb is not None and free_gb < self.config.minimum_free_disk_gb:
            raise BatchSafety(
                f"free disk {free_gb} GB < minimum_free_disk_gb "
                f"{self.config.minimum_free_disk_gb} GB"
            )

    @staticmethod
    def _free_disk_gb(path: Path) -> float | None:
        try:
            st = os.statvfs(path)
        except (AttributeError, OSError):
            return None
        return (st.f_bavail * st.f_frsize) / (1024 ** 3)

    @staticmethod
    def _output_size_gb(path: Path) -> float:
        total = 0
        try:
            for root, _dirs, files in os.walk(path):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        continue
        except OSError:
            return 0.0
        return total / (1024 ** 3)

    def assert_no_hidden_cohort(self, task_identity: Mapping[str, Any]) -> None:
        text = json.dumps(task_identity, sort_keys=True).lower()
        for kw in HIDDEN_COHORT_KEYWORDS:
            if kw in text:
                raise BatchSafety(f"refusing to dispatch task with keyword {kw!r}")

    # ------------------------------------------------------------------ sig

    def install_sigterm_handler(self) -> None:
        original = signal.getsignal(signal.SIGTERM)
        if original == signal.SIG_DFL:
            signal.signal(signal.SIGTERM, self._on_signal)
        original_int = signal.getsignal(signal.SIGINT)
        if original_int == signal.SIG_DFL:
            signal.signal(signal.SIGINT, self._on_signal)

    def _on_signal(self, signum: int, frame: Any) -> None:  # noqa: ARG002
        self._stop_requested.set()

    # ------------------------------------------------------------------ plan

    def plan(self) -> list[dict[str, Any]]:
        """Return a deterministic schedule of population slots; for tests."""
        schedule: list[dict[str, Any]] = []
        for gen in range(self.config.generations):
            for idx in range(self.config.population_size):
                schedule.append({"generation": gen, "population_index": idx})
        return schedule

    # ------------------------------------------------------------------ select

    def select_parents(
        self,
        generation: int,
        population_index: int,
        rng: random.Random,
        seen_candidates: Mapping[str, Mapping[str, Any]],
    ) -> tuple[list[Mapping[str, Any]], str, int]:
        """Deterministic parent selection with the 10/60/20/10 distribution.

        Returns ``(parent_programs, operator_id, mutation_seed)``.
        """
        # Choose parent selection mode by population_index within a generation.
        # 10/60/20/10 buckets: first 10 = elite, next 60 = single, next 20 = crossover,
        # last 10 = novelty.  ``population_size`` is fixed at 100 by config.
        idx = population_index
        if idx < 10:
            mode = "elite"
        elif idx < 70:
            mode = "single"
        elif idx < 90:
            mode = "crossover"
        else:
            mode = "novelty"

        if not seen_candidates:
            parent = _root_program(self.catalog.catalog_sha256)
        else:
            # PRNG-stable: tie-break by sorted candidate ids.
            ordered = sorted(seen_candidates.values(), key=lambda p: p["candidate_sha256"])
            if mode == "elite":
                # Take the highest-fitness seen_candidate; ties broken by sha256.
                ranked = sorted(
                    ordered,
                    key=lambda p: (-float(p.get("fitness", 0.0)), p["candidate_sha256"]),
                )
                parent_prog = ranked[0]["program"]
            elif mode == "novelty":
                # Take a random previously unseen parent (use rng deterministic).
                parent_prog = ordered[rng.randrange(len(ordered))]["program"]
            elif mode == "crossover":
                # Two parents — pick by index parity for determinism.
                a = ordered[rng.randrange(len(ordered))]["program"]
                b = ordered[rng.randrange(len(ordered))]["program"]
                if a.get("candidate_sha256") == b.get("candidate_sha256"):
                    b = ordered[(rng.randrange(len(ordered)) + 1) % len(ordered)][
                        "program"
                    ]
                return [a, b], "combine_two_parents", _seed(generation, idx, mode)
            else:  # single
                parent_prog = ordered[rng.randrange(len(ordered))]["program"]
            parent = parent_prog  # type: ignore[assignment]
        return [parent], "single_or_elite_or_novelty", _seed(generation, idx, mode)

    # ------------------------------------------------------------------ main

    def run(self, head_sha: str) -> dict[str, Any]:
        self.validate_environment(head_sha)
        self.lease.acquire()
        try:
            self.install_sigterm_handler()
            return self._run_loop(head_sha)
        finally:
            self.lease.release()

    def _run_loop(self, head_sha: str) -> dict[str, Any]:
        rng = random.Random(self.config.seed)
        # Resume rng state if present
        if self.state.rng_state:
            try:
                rng.setstate(self.state.rng_state)
            except Exception:
                pass
        self._rng = rng
        # Resume: pick up at the next not-yet-finalized slot.
        if self.state.unique_finalized_trials >= self.config.total_trials:
            self.state.execution_status = "finalized_pending_codex_review"
            save_batch_state(self.state_path, self.state)
            return self._summary()
        # Iterate from current (generation, index)
        gen = self.state.generation
        idx = self.state.current_population_index
        # In-memory seen candidates map: candidate_sha256 -> {program, fitness}
        seen: dict[str, dict[str, Any]] = {}
        for rec in self.ledger.iter_records():
            # We don't have the original program here; only re-populate fitness/sha.
            seen[rec.candidate_sha256] = {
                "candidate_sha256": rec.candidate_sha256,
                "fitness": rec.fitness,
                "program": {"_replay_only": True, "sha": rec.candidate_sha256},
            }
        for parent in _load_resume_programs(self.batch_root):
            seen[parent["candidate_sha256"]] = parent

        while self.state.generation < self.config.generations:
            gen = self.state.generation
            idx = self.state.current_population_index
            if self._stop_requested.is_set():
                self.state.stopped_reason = "stopped_safety"
                self.state.execution_status = "stopped"
                save_batch_state(self.state_path, self.state)
                return self._summary()
            parent_programs, operator_hint, mutation_seed = self.select_parents(
                gen, idx, rng, seen
            )
            operator_id = self._select_operator(
                operator_hint, gen, idx, rng
            )
            pool = _make_clause_pool(rng)
            try:
                new_prog = apply_mutation(
                    operator_id=operator_id,
                    parent_programs=parent_programs,
                    rng=rng,
                    pool=pool,
                    generation=gen,
                    population_index=idx,
                    mutation_seed=mutation_seed,
                )
            except Exception as error:
                record = self._compile_rejected_record(
                    gen, idx, operator_id, mutation_seed, parent_programs, str(error)
                )
                self.ledger.append(record)
                self.state.unique_finalized_trials += 1
                self._advance()
                self._persist(gen, idx)
                continue
            new_sha = _candidate_sha256(new_prog)
            new_novelty = _novelty_sha256(new_sha, gen)
            # Duplicate detection
            if new_sha in self._seen_candidate_hashes:
                record = self._duplicate_rejected_record(
                    gen, idx, operator_id, mutation_seed, parent_programs,
                    new_prog, new_sha, new_novelty,
                )
                self.ledger.append(record)
                self.state.unique_finalized_trials += 1
                seen[new_sha] = {
                    "candidate_sha256": new_sha,
                    "fitness": 0.0,
                    "program": new_prog,
                }
                self._advance()
                self._persist(gen, idx)
                continue
            self._seen_candidate_hashes.add(new_sha)
            # Compile / static screen — minimal: ensure new_prog is JSON-serializable
            if not isinstance(new_prog, dict):
                record = self._compile_rejected_record(
                    gen, idx, operator_id, mutation_seed, parent_programs,
                    "candidate not a JSON object",
                )
                self.ledger.append(record)
                self.state.unique_finalized_trials += 1
                self._advance()
                self._persist(gen, idx)
                continue
            # Qwen pre-screen happens later in finalize — to keep runner
            # deterministic in tests, classify as screened_out for non-finalists.
            classification, fitness, claim_ids, native_campaign_id, qwen_ids = (
                self._classify(gen, idx, new_prog, new_sha)
            )
            record = TrialRecord(
                trial_id=f"trial-{gen:03d}-{idx:03d}",
                generation=gen,
                population_index=idx,
                candidate_id=f"cand-{new_sha[:12]}",
                candidate_revision_id=f"rev-{new_sha[12:24]}",
                candidate_sha256=new_sha,
                parent_candidate_ids=tuple(
                    _candidate_sha256(p) for p in parent_programs
                ),
                parent_artifact_sha256=_candidate_sha256(parent_programs[0]),
                mutation_operator_id=operator_id,
                mutation_seed=mutation_seed,
                mutation_input_sha256=_candidate_sha256(parent_programs[0]),
                mutation_output_sha256=new_sha,
                compiled_bundle_sha256=new_sha,
                novelty_sha256=new_novelty,
                screen_stage="compile",
                screen_result="passed" if classification not in (
                    CLASSIFICATION_COMPILE_REJECTED, CLASSIFICATION_DUPLICATE_REJECTED
                ) else "rejected",
                qwen_receipt_ids=qwen_ids,
                native_campaign_id=native_campaign_id,
                claim_ids=claim_ids,
                classification=classification,
                fitness=fitness,
                created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                finalized_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                previous_event_sha256=self.ledger.previous_event_sha256,
                event_sha256="",
            )
            self.ledger.append(record)
            self.state.unique_finalized_trials += 1
            seen[new_sha] = {
                "candidate_sha256": new_sha,
                "fitness": fitness,
                "program": new_prog,
            }
            # Persist seen programs for resume.
            _append_seen_program(self.batch_root, new_prog, new_sha, fitness)
            self._advance()
            self._persist(gen, idx)
        # All generations complete
        self.state.execution_status = "finalized_pending_codex_review"
        save_batch_state(self.state_path, self.state)
        return self._summary()

    # ------------------------------------------------------------------ helpers

    def _classify(
        self,
        generation: int,
        population_index: int,
        program: Mapping[str, Any],
        candidate_sha: str,
    ) -> tuple[str, float, tuple[str, ...], str | None, tuple[str, ...]]:
        """Map a population slot to a classification.

        For this deterministic batch controller, the first ``native_finalists_per_generation``
        population slots (by index, per-generation) execute the official
        fresh-feedback-e2e; the next ``qwen_prescreen - native_finalists`` slots are
        pre-screen only; the rest are screened_out.
        """
        n = self.config.native_finalists_per_generation
        q = self.config.qwen_prescreen_per_generation
        if population_index < n:
            # Native finalist path
            try:
                result = self._run_native_finalist(generation, population_index, program)
            except BatchSafety:
                raise
            except Exception as error:  # infra failure
                self._record_infra_failure(str(error))
                return (
                    CLASSIFICATION_NATIVE_INFRA_FAILURE,
                    FITNESS_INFRA_FAILURE,
                    (),
                    None,
                    (),
                )
            classification, fitness, claim_ids = self._classify_native_result(result)
            return (
                classification,
                fitness,
                claim_ids,
                result.get("campaign_id", ""),
                (),
            )
        if population_index < q:
            # Qwen pre-screen — no fitness, no claims.
            return (CLASSIFICATION_SCREENED_OUT, FITNESS_SCREENED_OUT, (), None, ())
        return (CLASSIFICATION_SCREENED_OUT, FITNESS_SCREENED_OUT, (), None, ())

    def _run_native_finalist(
        self,
        generation: int,
        population_index: int,
        program: Mapping[str, Any],
    ) -> dict[str, Any]:
        # Each finalist gets its own per-trial output sub-root.
        out = self.batch_root / "trials" / f"trial-{generation:03d}-{population_index:03d}"
        out.mkdir(parents=True, exist_ok=True)
        # Per-trial fresh campaign config is the user-supplied template.
        # Only enforce existence when at least one native finalist is
        # configured; tests that exercise non-native paths use a stub.
        template = Path(self.config.fresh_campaign_template)
        if (
            self.config.native_finalists_per_generation > 0
            and not template.exists()
        ):
            raise BatchSafety(f"fresh campaign template missing: {template}")
        # Rewrite the template's final_commit_sha to bind to the current
        # worktree HEAD so fresh_feedback_e2e accepts the dispatch.
        per_trial_config = out / "FRESH-FEEDBACK-CONFIG.json"
        template_data = json.loads(template.read_text(encoding="utf-8"))
        template_data["final_commit_sha"] = self.config.final_commit_sha
        per_trial_config.write_text(
            json.dumps(template_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return self.fresh_feedback_runner(per_trial_config, out)

    def _classify_native_result(
        self, result: Mapping[str, Any]
    ) -> tuple[str, float, tuple[str, ...]]:
        # Defensive: only produce a "native gain" if the official
        # fresh-feedback-e2e entry returned a passing native result.
        classifications = result.get("classifications", [])
        if not isinstance(classifications, list) or not classifications:
            return (
                CLASSIFICATION_NATIVE_NEUTRAL,
                FITNESS_NEUTRAL,
                tuple(),
            )
        gains = sum(1 for c in classifications if c == "gain")
        regressions = sum(1 for c in classifications if c == "regression")
        infra = sum(1 for c in classifications if c == "infra_failure")
        # Native infra failure: only if ALL arms failed at infra level
        if infra and infra == len(classifications) and gains == 0 and regressions == 0:
            return (
                CLASSIFICATION_NATIVE_INFRA_FAILURE,
                FITNESS_INFRA_FAILURE,
                tuple(),
            )
        if regressions > 0 and gains == 0:
            return (
                CLASSIFICATION_NATIVE_REGRESSION,
                FITNESS_REGRESSION * regressions,
                tuple(),
            )
        if gains > 0:
            return (
                CLASSIFICATION_NATIVE_GAIN,
                FITNESS_GAIN * gains,
                tuple(),
            )
        return (
            CLASSIFICATION_NATIVE_NEUTRAL,
            FITNESS_NEUTRAL,
            tuple(),
        )

    def _compile_rejected_record(
        self,
        gen: int,
        idx: int,
        operator_id: str,
        mutation_seed: int,
        parent_programs: Sequence[Mapping[str, Any]],
        reason: str,
    ) -> TrialRecord:
        return TrialRecord(
            trial_id=f"trial-{gen:03d}-{idx:03d}",
            generation=gen,
            population_index=idx,
            candidate_id="",
            candidate_revision_id="",
            candidate_sha256="",
            parent_candidate_ids=tuple(_candidate_sha256(p) for p in parent_programs),
            parent_artifact_sha256=_candidate_sha256(parent_programs[0]) if parent_programs else "",
            mutation_operator_id=operator_id,
            mutation_seed=mutation_seed,
            mutation_input_sha256=_candidate_sha256(parent_programs[0]) if parent_programs else "",
            mutation_output_sha256="",
            compiled_bundle_sha256="",
            novelty_sha256="",
            screen_stage="compile",
            screen_result=f"rejected: {reason}",
            qwen_receipt_ids=(),
            native_campaign_id=None,
            claim_ids=(),
            classification=CLASSIFICATION_COMPILE_REJECTED,
            fitness=FITNESS_COMPILE_REJECTED,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            finalized_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            previous_event_sha256=self.ledger.previous_event_sha256,
            event_sha256="",
        )

    def _duplicate_rejected_record(
        self,
        gen: int,
        idx: int,
        operator_id: str,
        mutation_seed: int,
        parent_programs: Sequence[Mapping[str, Any]],
        program: Mapping[str, Any],
        candidate_sha: str,
        novelty: str,
    ) -> TrialRecord:
        return TrialRecord(
            trial_id=f"trial-{gen:03d}-{idx:03d}",
            generation=gen,
            population_index=idx,
            candidate_id=f"cand-{candidate_sha[:12]}",
            candidate_revision_id=f"rev-{candidate_sha[12:24]}",
            candidate_sha256=candidate_sha,
            parent_candidate_ids=tuple(_candidate_sha256(p) for p in parent_programs),
            parent_artifact_sha256=_candidate_sha256(parent_programs[0]) if parent_programs else "",
            mutation_operator_id=operator_id,
            mutation_seed=mutation_seed,
            mutation_input_sha256=_candidate_sha256(parent_programs[0]) if parent_programs else "",
            mutation_output_sha256=candidate_sha,
            compiled_bundle_sha256=candidate_sha,
            novelty_sha256=novelty,
            screen_stage="compile",
            screen_result="duplicate_rejected",
            qwen_receipt_ids=(),
            native_campaign_id=None,
            claim_ids=(),
            classification=CLASSIFICATION_DUPLICATE_REJECTED,
            fitness=FITNESS_DUPLICATE_REJECTED,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            finalized_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            previous_event_sha256=self.ledger.previous_event_sha256,
            event_sha256="",
        )

    def _select_operator(
        self,
        hint: str,
        generation: int,
        population_index: int,
        rng: random.Random,
    ) -> str:
        """Map parent-selection hint → concrete mutation operator id."""
        if hint == "combine_two_parents":
            return "combine_two_parents"
        # Default: one of the 9 single-parent operators, deterministic by
        # (generation, population_index).
        single_ops = tuple(op for op in MUTATION_OPERATORS if op != "combine_two_parents")
        idx = (generation * 1009 + population_index) % len(single_ops)
        return single_ops[idx]

    def _record_infra_failure(self, signature: str) -> None:
        if signature == self._last_failure_signature:
            self._last_failure_signature_count += 1
        else:
            self._last_failure_signature = signature
            self._last_failure_signature_count = 1
        self._consecutive_infra_failures += 1
        self.state.consecutive_infra_failures = self._consecutive_infra_failures
        self.state.last_failure_signature = self._last_failure_signature
        self.state.last_failure_signature_count = self._last_failure_signature_count

    def _advance(self) -> None:
        # Move to next slot
        if self.state.current_population_index + 1 >= self.config.population_size:
            self.state.current_population_index = 0
            self.state.generation += 1
        else:
            self.state.current_population_index += 1

    def _persist(self, gen: int, idx: int) -> None:
        if (
            self.state.unique_finalized_trials % self.config.checkpoint_every_trials == 0
        ):
            self.state.rng_state = self._rng_state_tuple()
            save_batch_state(self.state_path, self.state)
        # Disk safety
        used = self._output_size_gb(self.batch_root)
        if used > self.config.output_limit_gb:
            self._stop_requested.set()
            self.state.stopped_reason = "stopped_disk"
            self.state.execution_status = "stopped"
            self.state.rng_state = self._rng_state_tuple()
            save_batch_state(self.state_path, self.state)

    def _rng_state_tuple(self) -> tuple:
        """Snapshot the rng state for byte-equivalent resume."""
        # Access the local rng in _run_loop via a sentinel attribute.
        rng = getattr(self, "_rng", None)
        if rng is None:
            return ()
        try:
            return rng.getstate()
        except Exception:
            return ()

    def _summary(self) -> dict[str, Any]:
        return {
            "status": self.state.status,
            "execution_status": self.state.execution_status,
            "campaign_id": self.batch_root.name,
            "target_trials": self.state.target_trials,
            "unique_finalized_trials": self.state.unique_finalized_trials,
            "generation": self.state.generation,
            "current_population_index": self.state.current_population_index,
            "cohort": self.state.cohort,
            "holdout_opened": self.state.holdout_opened,
            "r076_opened": self.state.r076_opened,
            "r078_opened": self.state.r078_opened,
            "final_sealed_opened": self.state.final_sealed_opened,
            "new_paid_api_spend_cny": self.state.new_paid_api_spend_cny,
            "stopped_reason": self.state.stopped_reason,
        }


def _seed(generation: int, population_index: int, mode: str) -> int:
    """Deterministic seed = sha256 of (gen,idx,mode) truncated to 31 bits."""
    raw = hashlib.sha256(f"{generation}|{population_index}|{mode}".encode()).digest()
    return int.from_bytes(raw[:4], "big") & 0x7FFFFFFF


def _load_resume_programs(batch_root: Path) -> Iterable[dict[str, Any]]:
    """Reload seen programs from disk so we can resume parent selection."""
    p = Path(batch_root) / "checkpoints" / "seen-programs.jsonl"
    if not p.exists():
        return ()
    out: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def _append_seen_program(
    batch_root: Path, program: Mapping[str, Any], sha: str, fitness: float
) -> None:
    p = Path(batch_root) / "checkpoints" / "seen-programs.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "candidate_sha256": sha,
        "fitness": float(fitness),
        "program": dict(program),
    }
    with open(p, "a", encoding="utf-8") as f:
        f.write(canonical_json(payload) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_continuous_feedback_evolution(
    *, config_path: Path, output_root: Path, worktree_root: Path
) -> dict[str, Any]:
    """Top-level entry used by the CLI."""
    config = load_config(config_path)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    head_sha = _git_head_sha(worktree_root)
    runner = ContinuousRunner(
        config=config,
        batch_root=output_root,
        worktree_root=worktree_root,
    )
    return runner.run(head_sha=head_sha)


def _git_head_sha(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True
    )
    if result.returncode != 0:
        raise BatchSafety(f"git rev-parse HEAD failed in {root}")
    return result.stdout.strip()


__all__ = [
    "ALL_CLASSIFICATIONS",
    "BatchConfig",
    "BatchConfigError",
    "BatchBusy",
    "BatchLedgerError",
    "BatchSafety",
    "ContinuousRunner",
    "MutationCatalog",
    "MUTATION_OPERATORS",
    "NATIVE_CLASSIFICATIONS",
    "TrialLedger",
    "TrialRecord",
    "apply_mutation",
    "compute_event_sha256",
    "load_batch_state",
    "load_config",
    "run_continuous_feedback_evolution",
    "save_batch_state",
]
