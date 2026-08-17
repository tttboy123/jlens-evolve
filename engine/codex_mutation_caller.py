"""One-shot real Codex proposer calls with immutable local evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from agent_arm_runner import codex_subprocess_environment


class CodexMutationCallError(RuntimeError):
    """Raised when a proposer call cannot produce a frozen final response."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _usage_from_events(text: str) -> dict[str, int]:
    usage = {"input_tokens": 0, "output_tokens": 0}
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        candidate = event.get("usage") if isinstance(event, dict) else None
        if not isinstance(candidate, dict):
            continue
        for key, current in usage.items():
            value = candidate.get(key)
            if isinstance(value, int) and value >= 0:
                usage[key] = max(current, value)
    return {**usage, "total_tokens": usage["input_tokens"] + usage["output_tokens"]}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class CodexMutationCaller:
    """Callable adapter used by RealMutationProposerAdapter.

    The prompt hash is the call identity. A completed identical prompt is replayed
    from local evidence and never dispatched twice.
    """

    def __init__(
        self,
        *,
        codex_executable: Path,
        output_root: Path,
        working_directory: Path,
        model: str,
        reasoning: str,
        timeout_seconds: int,
    ) -> None:
        self.codex_executable = codex_executable.resolve()
        self.output_root = output_root.resolve()
        self.working_directory = working_directory.resolve()
        self.model = model
        self.reasoning = reasoning
        self.timeout_seconds = timeout_seconds
        if not self.codex_executable.is_file():
            raise CodexMutationCallError("Codex executable is missing")
        if not self.working_directory.is_dir():
            raise CodexMutationCallError("proposer working directory is missing")
        if not model.strip() or not reasoning.strip() or timeout_seconds < 1:
            raise CodexMutationCallError("proposer execution contract is invalid")

    def _argv(self, last_message_path: Path) -> tuple[str, ...]:
        argv = [
            str(self.codex_executable),
            "exec",
            "--json",
            "--ephemeral",
        ]
        # Keep the instance provider config (e.g. DeepSeek) when requested;
        # otherwise the hermetic flag discards it and auth falls back to ChatGPT.
        if os.environ.get("EVOLVE_CODEX_REQUIRE_PROVIDER_CONFIG") != "1":
            argv.append("--ignore-user-config")
        sandbox_args = (
            ["--dangerously-bypass-approvals-and-sandbox"]
            if os.environ.get("EVOLVE_CODEX_NO_SANDBOX") == "1"
            else ["--sandbox", "read-only"]
        )
        argv.extend(
            [
                "--ignore-rules",
                "--disable",
                "apps",
                "--disable",
                "enable_mcp_apps",
                "--disable",
                "plugins",
                "--disable",
                "remote_plugin",
                "--disable",
                "browser_use",
                "--disable",
                "computer_use",
                "--disable",
                "multi_agent",
                "--disable",
                "memories",
                "--skip-git-repo-check",
                "--model",
                self.model,
                "-c",
                'model_provider="custom"',
                "-c",
                f'model_reasoning_effort="{self.reasoning}"',
                "-c",
                'web_search="disabled"',
                "-c",
                "apps._default.enabled=false",
                "-c",
                "include_apps_instructions=false",
                "-c",
                "mcp_servers={}",
                "-c",
                "tool_output_token_limit=4096",
                *sandbox_args,
                "-C",
                str(self.working_directory),
                "--output-last-message",
                str(last_message_path),
                "-",
            ]
        )
        return tuple(argv)

    def __call__(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise CodexMutationCallError("proposer prompt is empty")
        prompt_bytes = prompt.encode("utf-8")
        prompt_sha256 = _sha256_bytes(prompt_bytes)
        call_root = self.output_root / prompt_sha256
        receipt_path = call_root / "receipt.json"
        response_path = call_root / "last-message.txt"
        if receipt_path.is_file() and response_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            response = response_path.read_text(encoding="utf-8")
            if (
                receipt.get("status") == "completed"
                and receipt.get("prompt", {}).get("sha256") == prompt_sha256
                and receipt.get("response", {}).get("sha256")
                == _sha256_bytes(response.encode("utf-8"))
            ):
                return response
            raise CodexMutationCallError("persisted proposer evidence was tampered")
        if call_root.exists() and any(call_root.iterdir()):
            raise CodexMutationCallError(
                "incomplete proposer call exists; explicit reconciliation is required"
            )
        call_root.mkdir(parents=True, exist_ok=True)
        (call_root / "prompt.txt").write_bytes(prompt_bytes)
        argv = self._argv(response_path)
        subprocess_environment, network_transport = codex_subprocess_environment()
        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
                env=subprocess_environment,
            )
        except subprocess.TimeoutExpired as error:
            (call_root / "events.jsonl").write_text(
                error.stdout or "", encoding="utf-8"
            )
            (call_root / "stderr.txt").write_text(error.stderr or "", encoding="utf-8")
            raise CodexMutationCallError("Codex proposer call timed out") from error
        elapsed = time.monotonic() - started
        (call_root / "events.jsonl").write_text(completed.stdout, encoding="utf-8")
        (call_root / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0 or not response_path.is_file():
            raise CodexMutationCallError(
                f"Codex proposer failed with return code {completed.returncode}"
            )
        response = response_path.read_text(encoding="utf-8")
        if not response.strip():
            raise CodexMutationCallError("Codex proposer returned an empty response")
        receipt = {
            "schema_version": 1,
            "status": "completed",
            "model": self.model,
            "reasoning": self.reasoning,
            "argv": list(argv),
            "returncode": completed.returncode,
            "elapsed_seconds": elapsed,
            "prompt": {"bytes": len(prompt_bytes), "sha256": prompt_sha256},
            "response": {
                "bytes": len(response.encode("utf-8")),
                "sha256": _sha256_bytes(response.encode("utf-8")),
            },
            "events_sha256": _sha256_bytes(completed.stdout.encode("utf-8")),
            "stderr_sha256": _sha256_bytes(completed.stderr.encode("utf-8")),
            "usage": _usage_from_events(completed.stdout),
            "sandbox": "read-only",
            "web_search": "disabled",
            "network_transport": network_transport,
        }
        _write_json(receipt_path, receipt)
        return response
