"""Compile frozen Round 1 identities into a gold-free executable TaskSet."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_json, sha256_json
from .eval_manifest import EvaluationTask, EvaluationTaskSet
from .round1_selection import _verified
from .target_selection import TargetSelectionManifest, TargetSelectionRecord

_LANGUAGE_EXTENSIONS = {
    "python": frozenset({".py"}),
    "c": frozenset({".c", ".h"}),
    "c++": frozenset({".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}),
    "go": frozenset({".go"}),
    "java": frozenset({".java"}),
    "javascript": frozenset({".js", ".jsx", ".mjs", ".cjs"}),
    "php": frozenset({".php"}),
    "ruby": frozenset({".rb"}),
    "rust": frozenset({".rs"}),
    "typescript": frozenset({".ts", ".tsx", ".mts", ".cts"}),
}
_EXCLUDED_PARTS = frozenset(
    {".git", "build", "dist", "docs", "examples", "node_modules", "vendor"}
)
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_MAX_CONTENT_SCAN_BYTES = 30_000_000
_QUERY_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "before",
        "behavior",
        "bug",
        "code",
        "current",
        "describe",
        "expected",
        "from",
        "have",
        "inside",
        "into",
        "issue",
        "only",
        "project",
        "should",
        "that",
        "their",
        "then",
        "this",
        "when",
        "with",
    }
)


def _is_test_path(relative: Path) -> bool:
    parts = {part.lower() for part in relative.parts}
    name = relative.name.lower()
    return bool(
        parts.intersection({"test", "tests", "__tests__"})
        or any(part.startswith("test-") for part in parts)
        or name.startswith("test_")
        or name.endswith(
            (
                "_test.c",
                "_test.cc",
                "_test.cpp",
                "_test.cxx",
                "_test.go",
                "_test.js",
                "_test.jsx",
                "_test.py",
                "_test.rb",
                "_test.rs",
                "_test.ts",
                "_test.tsx",
                "_tests.py",
                "spec.rb",
                "test.java",
                "test.php",
                "tests.java",
            )
        )
        or ".test." in name
        or ".spec." in name
    )


def _normalized_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for raw in _TOKEN.findall(text):
        lowered = raw.lower().strip("._-")
        pieces = [piece for piece in re.split(r"[_.-]+", lowered) if piece]
        camel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw)
        pieces.extend(piece.lower() for piece in camel.split())
        for term in {lowered, *pieces}:
            if len(term) >= 2 and term not in _QUERY_STOPWORDS and not term.isdigit():
                terms.add(term)
    return terms


def _query_weights(instruction: str) -> dict[str, int]:
    weights: dict[str, int] = defaultdict(int)
    for term in _normalized_terms(instruction):
        weights[term] += 1
    title = next((line for line in instruction.splitlines() if line.strip()), "")
    for term in _normalized_terms(title):
        weights[term] += 4
    code = "\n".join(
        [
            *re.findall(r"```(?:[^\n]*)\n(.*?)```", instruction, re.DOTALL),
            *re.findall(r"(?<!`)`([^`\n]+)`(?!`)", instruction),
        ]
    )
    for term in _normalized_terms(code):
        weights[term] += 8
    return dict(sorted(weights.items(), key=lambda row: (-row[1], row[0]))[:96])


def _issue_ranked_targets(
    checkout: Path, instruction: str, language: str, max_targets: int
) -> list[Path]:
    suffixes = _LANGUAGE_EXTENSIONS[language]
    query = _query_weights(instruction)
    candidates: list[tuple[Path, str, set[str], int]] = []
    for path in sorted(checkout.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() not in suffixes
            or _is_test_path(path.relative_to(checkout))
            or {part.lower() for part in path.parts}.intersection(_EXCLUDED_PARTS)
        ):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > 300_000:
            continue
        relative = path.relative_to(checkout).as_posix().lower()
        path_terms = _normalized_terms(relative)
        candidates.append((path, relative, path_terms, size))
    if not candidates:
        return []

    path_frequency = {
        term: sum(
            term in path_terms for _path, _relative, path_terms, _size in candidates
        )
        for term in query
    }
    total = len(candidates)

    def path_score(row: tuple[Path, str, set[str], int]) -> float:
        _path, _relative, path_terms, _size = row
        return sum(
            weight * (1.0 + math.log((total + 1) / (path_frequency[term] + 1))) * 5.0
            for term, weight in query.items()
            if term in path_terms
        )

    # The scan budget is allocated by public-issue path relevance first, then by
    # file size. Lexicographic traversal would permanently hide the latter half
    # of large repositories from content retrieval.
    content_terms_by_path: dict[Path, set[str]] = {}
    scanned = 0
    for path, _relative, _path_terms, size in sorted(
        candidates,
        key=lambda row: (-path_score(row), row[3], row[1]),
    ):
        if scanned + size > _MAX_CONTENT_SCAN_BYTES:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        scanned += size
        content_terms_by_path[path] = _normalized_terms(content)

    document_frequency = {
        term: sum(
            term in path_terms or term in content_terms_by_path.get(path, set())
            for path, _relative, path_terms, _size in candidates
        )
        for term in query
    }
    ranked: list[tuple[float, int, str, Path]] = []
    for path, relative, path_terms, size in candidates:
        content_terms = content_terms_by_path.get(path, set())
        score = 0.0
        for term, weight in query.items():
            if term not in path_terms and term not in content_terms:
                continue
            inverse_frequency = 1.0 + math.log(
                (total + 1) / (document_frequency[term] + 1)
            )
            score += weight * inverse_frequency * (5.0 if term in path_terms else 1.0)
        ranked.append((score, -size, relative, path))
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return [row[3] for row in ranked[:max_targets]]


def select_round1_targets(
    checkout: Path,
    instruction: str,
    language: str,
    *,
    max_targets: int = 8,
) -> tuple[str, ...]:
    """Select a bounded non-test source candidate universe without a patch."""

    if language not in _LANGUAGE_EXTENSIONS:
        raise ContractError("Round 1 target language is unsupported")
    if type(max_targets) is not int or not 1 <= max_targets <= 64:
        raise ContractError("Round 1 target limit must be between 1 and 64")
    ranked = _issue_ranked_targets(
        checkout, instruction, language, max_targets=max_targets
    )
    targets = tuple(path.relative_to(checkout).as_posix() for path in ranked)
    if not targets:
        raise ContractError("Round 1 target selector found no editable source")
    return targets


def _multi_instruction(row: dict[str, Any]) -> str:
    sections = [str(row.get("title", "")).strip(), str(row.get("body", "")).strip()]
    issues = row.get("resolved_issues", [])
    if isinstance(issues, list):
        for issue in issues:
            if isinstance(issue, dict):
                sections.extend(
                    [
                        str(issue.get("title", "")).strip(),
                        str(issue.get("body", "")).strip(),
                    ]
                )
    instruction = "\n\n".join(section for section in sections if section)
    if not instruction:
        raise ContractError("Round 1 Multi-SWE instruction is empty")
    return instruction


def _dataset_index(pool_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    paths = (
        (
            "swe-bench-verified",
            pool_root / "harness-inputs/swe-bench-verified.jsonl",
        ),
        (
            "swe-bench-multilingual",
            pool_root / "harness-inputs/swe-bench-multilingual.jsonl",
        ),
        (
            "multi-swe-bench-flash",
            pool_root / "inputs/multi-swe-bench-flash/multi_swe_bench_flash.jsonl",
        ),
    )
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for benchmark_id, path in paths:
        if not path.is_file():
            raise ContractError("Round 1 frozen dataset is missing")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = row.get("instance_id")
            key = (benchmark_id, instance_id)
            if not isinstance(instance_id, str) or key in indexed:
                raise ContractError("Round 1 dataset identity is ambiguous")
            indexed[key] = row
    return indexed


def _checkout(source: Path, commit: str, destination: Path) -> Path:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "worktree",
            "add",
            "--quiet",
            "--detach",
            str(destination),
            commit,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ContractError("Round 1 target source checkout failed")
    return destination


def _remove_checkout(source: Path, checkout: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(source), "worktree", "remove", "--force", str(checkout)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ContractError("Round 1 target worktree cleanup failed")


def _write_frozen(path: Path, report: dict[str, Any], *, label: str) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"{label} is unreadable") from exc
        if existing != report:
            raise ContractError(f"frozen {label} does not match replay")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(report) + "\n", encoding="utf-8")


def compile_round1_taskset(
    *,
    selection_path: Path,
    source_summary_path: Path,
    pool_root: Path,
    evidence_root: Path,
    workspace_root: Path,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """Compile/resume all selected tasks and freeze target provenance."""

    if max_tasks is not None and (type(max_tasks) is not int or max_tasks < 1):
        raise ContractError("Round 1 compile max_tasks must be a positive integer")
    selection_raw, selection = _verified(selection_path, label="Round 1 selection")
    selected_task_count = selection.get("task_count")
    if (
        selection.get("status") != "frozen"
        or type(selected_task_count) is not int
        or selected_task_count < 6
        or selected_task_count != len(selection.get("tasks", ()))
    ):
        raise ContractError("Round 1 selection is not frozen")
    source_path = source_summary_path.resolve()
    try:
        source_raw = source_path.read_bytes()
        source_summary = json.loads(source_raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("Round 1 source summary is unreadable") from exc
    source_content = {
        key: value for key, value in source_summary.items() if key != "summary_sha256"
    }
    if (
        source_summary.get("status") != "complete"
        or source_summary.get("summary_sha256") != sha256_json(source_content)
        or source_summary.get("verified_commit_count", 0) < selected_task_count
    ):
        raise ContractError("Round 1 sources are not complete")
    cache = Path(source_summary["cache_root"]).resolve()
    datasets = _dataset_index(pool_root.resolve())
    evidence = evidence_root.resolve()
    workspace = workspace_root.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    compiled: list[dict[str, Any]] = []
    executed = 0
    for selected in selection["tasks"]:
        task_id = selected["task_id"]
        cell_path = evidence / "cells" / task_id / "TASK.json"
        if cell_path.is_file():
            wrapper = json.loads(cell_path.read_text(encoding="utf-8"))
            content = {
                key: value for key, value in wrapper.items() if key != "evidence_sha256"
            }
            if wrapper.get("evidence_sha256") != sha256_json(content):
                raise ContractError("Round 1 compiled task evidence was tampered")
            compiled.append(wrapper)
            continue
        if max_tasks is not None and executed >= max_tasks:
            continue
        instance_id = selected["instance_id"]
        row = datasets.get((selected["benchmark_id"], instance_id))
        if row is None:
            raise ContractError("Round 1 selected dataset row is missing")
        benchmark = selected["benchmark_id"]
        if benchmark in {"swe-bench-verified", "swe-bench-multilingual"}:
            instruction = str(row.get("problem_statement", ""))
            base_commit = row.get("base_commit")
        elif benchmark == "multi-swe-bench-flash":
            instruction = _multi_instruction(row)
            base = row.get("base")
            base_commit = base.get("sha") if isinstance(base, dict) else None
        else:
            raise ContractError("Round 1 selected benchmark is unsupported")
        if base_commit != selected["base_commit"] or not instruction.strip():
            raise ContractError("Round 1 dataset row does not match selection")
        repo = selected["repo"]
        source = cache / repo
        if not (source / ".git").is_dir():
            raise ContractError("Round 1 source repository is missing")
        with tempfile.TemporaryDirectory(
            prefix=f"{task_id}-", dir=workspace
        ) as temporary:
            checkout = _checkout(source, base_commit, Path(temporary) / "checkout")
            try:
                computed_targets = select_round1_targets(
                    checkout,
                    instruction,
                    selected["language"],
                    max_targets=selected.get("retrieval_limit", 8),
                )
            finally:
                _remove_checkout(source, checkout)
        frozen_targets = selected.get("retrieved_targets")
        if frozen_targets is not None and tuple(frozen_targets) != computed_targets:
            raise ContractError("Round 1 frozen retrieval does not match replay")
        targets = computed_targets
        task = EvaluationTask.create(
            task_id=task_id,
            instance_id=instance_id,
            benchmark_id=benchmark,
            benchmark_base_commit=base_commit,
            repo=repo,
            source_repository=source,
            source_revision=base_commit,
            instruction=instruction,
            allowed_targets=list(targets),
            cohort=selected["cohort"],
        )
        target = TargetSelectionRecord.create(
            task=task,
            selector_id="issue-lexical-source-ranker-top32-v4",
            evidence=[
                "public_issue_text_only",
                f"language={selected['language']}",
                f"retrieval_limit={len(targets)}",
                *(f"ranked_source_path={relative}" for relative in targets),
            ],
        )
        content = {
            "schema_version": 1,
            "selection_task_uid": selected["task_uid"],
            "selection_identity_fingerprint": selected["identity_fingerprint"],
            "mechanism": selected["mechanism"],
            "language": selected["language"],
            "task": task.to_dict(),
            "target_selection": target.to_dict(),
            "answer_fields_read": False,
            "network_calls_performed": True,
        }
        wrapper = {**content, "evidence_sha256": sha256_json(content)}
        _write_frozen(cell_path, wrapper, label="Round 1 compiled task")
        compiled.append(wrapper)
        executed += 1
    planned = len(selection["tasks"])
    status = "complete" if len(compiled) == planned else "partial"
    if status == "complete":
        tasks = [EvaluationTask.from_dict(row["task"]) for row in compiled]
        taskset_id = (
            "round1-search-qualified-60-v3"
            if planned == 60
            else f"round1-search-qualified-{planned}-v4"
        )
        taskset = EvaluationTaskSet.create(taskset_id=taskset_id, tasks=tasks)
        target_manifest = TargetSelectionManifest.create(
            taskset=taskset,
            records=[
                TargetSelectionRecord.from_dict(row["target_selection"], task=task)
                for row, task in zip(compiled, tasks, strict=True)
            ],
        )
        _write_frozen(
            evidence / "TASKSET.json", taskset.to_dict(), label="Round 1 TaskSet"
        )
        _write_frozen(
            evidence / "TARGET-SELECTION.json",
            target_manifest.to_dict(),
            label="Round 1 target selection",
        )
        routes = {row["task"]["task_id"]: row["mechanism"] for row in compiled}
        route_content = {
            "schema_version": 1,
            "taskset_fingerprint": taskset.fingerprint,
            "target_selection_fingerprint": target_manifest.fingerprint,
            "routes": dict(sorted(routes.items())),
            "mechanism_counts": dict(sorted(Counter(routes.values()).items())),
        }
        _write_frozen(
            evidence / "MECHANISM-ROUTES.json",
            {**route_content, "evidence_sha256": sha256_json(route_content)},
            label="Round 1 mechanism routes",
        )
    content = {
        "schema_version": 1,
        "status": status,
        "planned_tasks": planned,
        "completed_tasks": len(compiled),
        "selection_file_sha256": hashlib.sha256(selection_raw).hexdigest(),
        "selection_evidence_sha256": selection["evidence_sha256"],
        "source_summary_file_sha256": hashlib.sha256(source_raw).hexdigest(),
        "source_summary_sha256": source_summary["summary_sha256"],
        "answer_fields_read": False,
        "network_calls_performed": bool(compiled),
    }
    summary = {**content, "summary_sha256": sha256_json(content)}
    name = "SUMMARY.json" if status == "complete" else "PROGRESS.json"
    (evidence / name).write_text(canonical_json(summary) + "\n", encoding="utf-8")
    return summary
