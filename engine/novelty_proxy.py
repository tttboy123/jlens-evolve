"""Fixed-budget OpenAI proxy for duplicate-aware proposal selection."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from admission_policy import source_fingerprints
from proposal_controller import load_proposal_controller
from structured_mutation import (
    apply_mutation_plan,
    build_coder_payload,
    build_planner_payload,
    derive_fallback_plan,
    extract_current_program,
    extract_public_target_failure,
    parse_mutation_plan,
    postcondition_satisfied,
)

_FENCE = re.compile(r"```(?:python|py)?[^\n]*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_candidate_source(text: str) -> str | None:
    """Extract the largest fenced Python candidate without executing it."""
    blocks = [block.strip() for block in _FENCE.findall(text) if block.strip()]
    if blocks:
        return max(blocks, key=len)
    stripped = text.strip()
    return stripped if "def solve" in stripped else None


def _response_content(response: dict[str, Any]) -> str:
    try:
        return str(response["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("upstream response has no assistant content") from exc


def _response_with_source(response: dict[str, Any], source: str) -> dict[str, Any]:
    updated = copy.deepcopy(response)
    updated["choices"][0]["message"]["content"] = f"```python\n{source.strip()}\n```"
    return updated


def _sources_in_payload(payload: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    for message in payload.get("messages", []):
        content = message.get("content", "") if isinstance(message, dict) else ""
        for block in _FENCE.findall(str(content)):
            if block.strip():
                sources.append(block.strip())
    return sources


def read_candidate_events(path: Path | None) -> list[dict[str, Any]]:
    """Read public candidate events; never inspect holdout verification rows."""
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid proposal event JSONL at {path}:{line_number}"
                ) from exc
            if row.get("event_type") == "candidate":
                rows.append(row)
    return rows


def detect_stagnation(events: list[dict[str, Any]], *, window: int) -> dict[str, Any]:
    """Detect a public-score plateau over the most recent candidate window."""
    if window < 2:
        raise ValueError("stagnation window must be at least two")
    candidates = [row for row in events if row.get("event_type") == "candidate"]
    recent = candidates[-window:]
    improvement_flags: list[bool] = []
    global_best = 0.0
    for row in candidates:
        global_best = max(global_best, float(row.get("parent_score", 0.0)))
        child_score = float(row.get("child_score", 0.0))
        improved = bool(row.get("accepted")) and child_score > global_best + 1e-12
        improvement_flags.append(improved)
        if row.get("accepted"):
            global_best = max(global_best, child_score)
    improvements = sum(improvement_flags[-window:])
    structural_duplicates = sum(
        bool(
            {"exact_duplicate", "ast_duplicate"}.intersection(
                row.get("admission_reasons", [])
            )
        )
        for row in recent
    )
    return {
        "active": len(recent) == window and improvements == 0,
        "window": window,
        "candidates_observed": len(candidates),
        "recent_improvements": improvements,
        "recent_structural_duplicates": structural_duplicates,
    }


def _duplicate_reasons(
    source: str | None, source_hashes: set[str], ast_hashes: set[str]
) -> tuple[list[str], str | None, str | None]:
    if not source:
        return ["missing_code"], None, None
    source_hash, ast_hash = source_fingerprints(source)
    reasons: list[str] = []
    if source_hash in source_hashes:
        reasons.append("exact_duplicate")
    if ast_hash in ast_hashes:
        reasons.append("ast_duplicate")
    return reasons, source_hash, ast_hash


def _retry_payload(payload: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    retried = copy.deepcopy(payload)
    messages = retried.setdefault("messages", [])
    reason_text = ", ".join(reasons) or "structural_duplicate"
    feedback = (
        "\n\n[Duplicate-aware retry; observer signal only]\n"
        f"The first proposal matched prior structure ({reason_text}). "
        "Produce one materially different control/data-flow structure while "
        "preserving already-passing public behavior. Do not merely rename, "
        "reformat, or add comments. The deterministic evaluator remains the "
        "only correctness authority."
    )
    if messages and isinstance(messages[-1], dict):
        messages[-1]["content"] = str(messages[-1].get("content", "")) + feedback
    else:
        messages.append({"role": "user", "content": feedback.strip()})
    return retried


class ProposalController:
    """Select one of two fixed-budget completions using only code novelty."""

    def __init__(self, *, mode: str) -> None:
        if mode not in {"shadow-control", "duplicate-aware"}:
            raise ValueError(f"unsupported proposal controller mode: {mode}")
        self.mode = mode
        self._seen_source_hashes: set[str] = set()
        self._seen_ast_hashes: set[str] = set()
        self._lock = threading.Lock()
        self._stats = {
            "stagnation_detector_version": "global-best-v2",
            "requests": 0,
            "upstream_calls": 0,
            "first_duplicates": 0,
            "retry_feedback_requests": 0,
            "selected_second": 0,
            "selected_novel": 0,
            "stagnation_triggers": 0,
        }

    def process_chat(
        self,
        payload: dict[str, Any],
        forward: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        stagnation: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Make exactly two calls and return the policy-selected response."""
        started = time.monotonic()
        context_source_hashes: set[str] = set()
        context_ast_hashes: set[str] = set()
        for source in _sources_in_payload(payload):
            source_hash, ast_hash = source_fingerprints(source)
            context_source_hashes.add(source_hash)
            context_ast_hashes.add(ast_hash)
        with self._lock:
            baseline_sources = context_source_hashes | self._seen_source_hashes
            baseline_asts = context_ast_hashes | self._seen_ast_hashes

        first_response = forward(copy.deepcopy(payload))
        first_source = extract_candidate_source(_response_content(first_response))
        first_reasons, first_source_hash, first_ast_hash = _duplicate_reasons(
            first_source, baseline_sources, baseline_asts
        )
        first_duplicate = bool(first_reasons)
        stagnation = stagnation or {"active": False}
        stagnation_active = bool(stagnation.get("active"))
        selection_trigger = (
            "first_duplicate"
            if first_duplicate
            else "search_stagnation"
            if stagnation_active
            else "none"
        )
        add_feedback = self.mode == "duplicate-aware" and selection_trigger != "none"
        feedback_reasons = list(first_reasons)
        if selection_trigger == "search_stagnation":
            feedback_reasons.append("search_stagnation")
        second_payload = (
            _retry_payload(payload, feedback_reasons)
            if add_feedback
            else copy.deepcopy(payload)
        )
        second_response = forward(second_payload)
        second_source = extract_candidate_source(_response_content(second_response))
        second_sources = set(baseline_sources)
        second_asts = set(baseline_asts)
        if first_source_hash:
            second_sources.add(first_source_hash)
        if first_ast_hash:
            second_asts.add(first_ast_hash)
        second_reasons, second_source_hash, second_ast_hash = _duplicate_reasons(
            second_source, second_sources, second_asts
        )

        select_second = bool(
            self.mode == "duplicate-aware"
            and selection_trigger != "none"
            and second_source is not None
            and (first_duplicate or not second_reasons)
        )
        selected_response = second_response if select_second else first_response
        selected_source_hash = (
            second_source_hash if select_second else first_source_hash
        )
        selected_ast_hash = second_ast_hash if select_second else first_ast_hash
        selected_reasons = second_reasons if select_second else first_reasons
        with self._lock:
            if selected_source_hash:
                self._seen_source_hashes.add(selected_source_hash)
            if selected_ast_hash:
                self._seen_ast_hashes.add(selected_ast_hash)
            self._stats["requests"] += 1
            self._stats["upstream_calls"] += 2
            self._stats["first_duplicates"] += int(first_duplicate)
            self._stats["retry_feedback_requests"] += int(add_feedback)
            self._stats["selected_second"] += int(select_second)
            self._stats["selected_novel"] += int(not selected_reasons)
            self._stats["stagnation_triggers"] += int(
                self.mode == "duplicate-aware" and stagnation_active
            )

        audit = {
            "request_id": uuid.uuid4().hex,
            "mode": self.mode,
            "upstream_calls": 2,
            "first_duplicate": first_duplicate,
            "first_reasons": first_reasons,
            "first_source_sha256": first_source_hash,
            "first_ast_sha256": first_ast_hash,
            "second_reasons": second_reasons,
            "second_source_sha256": second_source_hash,
            "second_ast_sha256": second_ast_hash,
            "selected_index": 2 if select_second else 1,
            "selected_reasons": selected_reasons,
            "selected_novel": not selected_reasons,
            "retry_feedback_added": add_feedback,
            "selection_trigger": (
                selection_trigger if self.mode == "duplicate-aware" else "none"
            ),
            "stagnation_active": stagnation_active,
            "stagnation": stagnation,
            "duration_seconds": time.monotonic() - started,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        return selected_response, audit

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"mode": self.mode, **self._stats}


class StructuredMutationController:
    """Run a fixed two-call planner/coder protocol with optional AST enforcement."""

    def __init__(self, *, mode: str) -> None:
        if mode not in {"planner-control", "structured-mutation"}:
            raise ValueError(f"unsupported structured controller mode: {mode}")
        self.mode = mode
        self._lock = threading.Lock()
        self._stats = {
            "protocol_version": "structured-mutation-v4",
            "requests": 0,
            "upstream_calls": 0,
            "structured_plans": 0,
            "model_plans_valid": 0,
            "fallback_plans": 0,
            "deterministic_transforms": 0,
            "model_repairs_selected": 0,
            "deterministic_fallbacks": 0,
            "free_coder_selected": 0,
        }

    def process_chat(
        self,
        payload: dict[str, Any],
        forward: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        stagnation: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del stagnation
        started = time.monotonic()
        structured = self.mode == "structured-mutation"
        current_source = extract_current_program(payload)
        public_failure = extract_public_target_failure(payload)

        planner_response = forward(
            build_planner_payload(payload, structured=structured)
        )
        planner_content = _response_content(planner_response)
        plan_origin = "control_free_plan"
        model_plan_valid = False
        if structured:
            try:
                plan = parse_mutation_plan(planner_content, public_failure)
                plan_origin = "model"
                model_plan_valid = True
            except (ValueError, json.JSONDecodeError):
                plan = derive_fallback_plan(public_failure)
                plan_origin = "public_failure_fallback"
        else:
            plan = derive_fallback_plan(None)

        mutation = (
            apply_mutation_plan(current_source, plan)
            if structured and current_source is not None
            else None
        )
        scaffold = (
            mutation.source
            if mutation is not None
            and mutation.changed
            and mutation.postcondition_valid
            else None
        )
        coder_response = forward(
            build_coder_payload(
                payload,
                plan=plan,
                planner_content=planner_content,
                scaffold=scaffold,
                structured=structured,
            )
        )
        coder_source = extract_candidate_source(_response_content(coder_response))
        current_ast = source_fingerprints(current_source)[1] if current_source else None
        coder_hashes = (
            source_fingerprints(coder_source) if coder_source else (None, None)
        )
        repair_postcondition_valid = bool(
            coder_source
            and postcondition_satisfied(coder_source, plan)
            and (not plan.structured or coder_hashes[1] != current_ast)
        )

        if not structured:
            if coder_source is not None:
                selected = coder_response
                selected_source = coder_source
                selected_origin = "free_coder"
            elif current_source is not None:
                selected = _response_with_source(coder_response, current_source)
                selected_source = current_source
                selected_origin = "current_program_fallback"
            else:
                selected = coder_response
                selected_source = None
                selected_origin = "invalid_coder"
        elif repair_postcondition_valid:
            selected = coder_response
            selected_source = coder_source
            selected_origin = "model_repair"
        elif scaffold is not None:
            selected = _response_with_source(coder_response, scaffold)
            selected_source = scaffold
            selected_origin = "deterministic_scaffold"
        elif coder_source is not None:
            selected = coder_response
            selected_source = coder_source
            selected_origin = "free_coder"
        elif current_source is not None:
            selected = _response_with_source(coder_response, current_source)
            selected_source = current_source
            selected_origin = "current_program_fallback"
        else:
            selected = coder_response
            selected_source = None
            selected_origin = "invalid_coder"

        selected_source_hash, selected_ast_hash = (
            source_fingerprints(selected_source)
            if selected_source is not None
            else (None, None)
        )
        with self._lock:
            self._stats["requests"] += 1
            self._stats["upstream_calls"] += 2
            self._stats["structured_plans"] += int(plan.structured)
            self._stats["model_plans_valid"] += int(model_plan_valid)
            self._stats["fallback_plans"] += int(
                structured and plan_origin == "public_failure_fallback"
            )
            self._stats["deterministic_transforms"] += int(scaffold is not None)
            self._stats["model_repairs_selected"] += int(
                selected_origin == "model_repair"
            )
            self._stats["deterministic_fallbacks"] += int(
                selected_origin == "deterministic_scaffold"
            )
            self._stats["free_coder_selected"] += int(selected_origin == "free_coder")

        audit = {
            "request_id": uuid.uuid4().hex,
            "mode": self.mode,
            "protocol_version": "structured-mutation-v4",
            "upstream_calls": 2,
            "public_failure": public_failure,
            "operator_id": plan.operator_id,
            "plan_origin": plan_origin,
            "plan_preserve": list(plan.preserve),
            "current_ast_sha256": current_ast,
            "deterministic_transform_applied": scaffold is not None,
            "deterministic_transform_error": (
                mutation.error if mutation is not None else None
            ),
            "repair_postcondition_valid": repair_postcondition_valid,
            "repair_source_sha256": coder_hashes[0],
            "repair_ast_sha256": coder_hashes[1],
            "selected_source_sha256": selected_source_hash,
            "selected_ast_sha256": selected_ast_hash,
            "selected_origin": selected_origin,
            "duration_seconds": time.monotonic() - started,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        return selected, audit

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"mode": self.mode, **self._stats}


class AuditStore:
    """Append proxy audits without storing prompts or source code."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class ProxyService:
    def __init__(
        self,
        *,
        controller_config: dict[str, Any],
        controller_sha256: str,
        upstream_base: str,
        audit_path: Path,
        event_archive_path: Path | None,
        timeout: float,
    ) -> None:
        self.controller_config = controller_config
        self.controller_sha256 = controller_sha256
        self.implementation_sha256 = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        self.upstream_base = upstream_base.rstrip("/")
        self.timeout = timeout
        self.event_archive_path = event_archive_path
        mode = str(controller_config["mode"])
        self.controller = (
            StructuredMutationController(mode=mode)
            if mode in {"planner-control", "structured-mutation"}
            else ProposalController(mode=mode)
        )
        self.audit_store = AuditStore(audit_path)

    def forward(
        self, payload: dict[str, Any], authorization: str | None
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.upstream_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": authorization} if authorization else {}),
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def process(
        self, payload: dict[str, Any], authorization: str | None
    ) -> dict[str, Any]:
        stagnation = detect_stagnation(
            read_candidate_events(self.event_archive_path),
            window=int(self.controller_config.get("stagnation_window", 3)),
        )
        response, audit = self.controller.process_chat(
            payload,
            lambda request: self.forward(request, authorization),
            stagnation=stagnation,
        )
        self.audit_store.append(audit)
        return response

    def descriptor(self) -> dict[str, Any]:
        return {
            **self.controller_config,
            "controller_sha256": self.controller_sha256,
            "implementation_sha256": self.implementation_sha256,
            "stats": self.controller.stats(),
        }


class ProxyHandler(BaseHTTPRequestHandler):
    server: ProposalProxyServer

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json(200, {"status": "ok"})
        elif self.path == "/proposal-controller":
            self._write_json(200, self.server.service.descriptor())
        elif self.path == "/stats":
            self._write_json(200, self.server.service.controller.stats())
        else:
            self._write_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._write_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if payload.get("stream"):
                raise ValueError("streaming is not supported by the novelty proxy")
            response = self.server.service.process(
                payload, self.headers.get("Authorization")
            )
            self._write_json(200, response)
        except (ValueError, urllib.error.URLError, TimeoutError) as exc:
            self._write_json(502, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return


class ProposalProxyServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], service: ProxyService) -> None:
        super().__init__(address, ProxyHandler)
        self.service = service


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-config", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--event-archive", type=Path)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=18081)
    parser.add_argument("--upstream-base", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--timeout", type=float, default=240.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    parsed = urllib.parse.urlparse(args.upstream_base)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("novelty proxy upstream must be loopback")
    controller_path = args.controller_config.resolve()
    config = load_proposal_controller(controller_path)
    digest = hashlib.sha256(controller_path.read_bytes()).hexdigest()
    service = ProxyService(
        controller_config=config,
        controller_sha256=digest,
        upstream_base=args.upstream_base,
        audit_path=args.audit.resolve(),
        event_archive_path=(
            args.event_archive.resolve() if args.event_archive else None
        ),
        timeout=args.timeout,
    )
    server = ProposalProxyServer((args.listen_host, args.listen_port), service)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
