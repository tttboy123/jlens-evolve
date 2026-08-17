"""Local JLens agent: Qwen3.5-4B (MLX) with per-step hidden-state capture.

v2.5 / T4: a minimal, restricted agent loop that records per-layer hidden
states (layer_records) for each action text the model produces. Weights are
frozen (inference only); the output is observation evidence
(observational_not_causal), not an admission gate.

Design:
- Tools are an explicit allowlist executed inside the repo workdir.
- Each step: model generates an action line (JSON-ish), layer records are
  captured for that exact text via a patched DecoderLayer.__call__, the tool
  runs, and the result is appended to the transcript.
- Evidence is written as JSONL (one line per step) + meta.json + final diff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MODEL = "models/Qwen3.5-4B-mlx-4bit"

_SYSTEM = (
    "You are a debugging agent in a local sandbox. You may use these tools:\n"
    '- {"tool": "list"}\n'
    '- {"tool": "read", "path": "<relative path>"}\n'
    '- {"tool": "run", "cmd": "<git|rg|sed|grep|ls|head|tail|pytest command>"}\n'
    '- {"tool": "write", "path": "<relative path>", "content": "<new file content>"}\n'
    '- {"tool": "finish", "message": "<summary>"}\n'
    "Respond with exactly one tool JSON per turn. Finish when you have a fix."
)

_ACTION_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_ALLOWED_CMD = (
    "git",
    "rg",
    "sed",
    "grep",
    "ls",
    "head",
    "tail",
    "pytest",
    "python3",
    "node",
)


class LensAgentError(ValueError):
    """Raised for invalid actions or tool violations."""


@dataclass
class LayerRecord:
    layer: int
    is_linear: bool
    shape: list[int]
    mean: float
    l2_norm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "is_linear": self.is_linear,
            "shape": self.shape,
            "mean": self.mean,
            "l2_norm": self.l2_norm,
        }


@dataclass
class StepEvidence:
    step: int
    action_text: str
    action: dict[str, Any]
    layer_records: list[dict[str, Any]] = field(default_factory=list)
    tool_result: str = ""
    tool_ok: bool = True
    step_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action_text": self.action_text,
            "action": self.action,
            "layer_records": self.layer_records,
            "tool_result": self.tool_result[:2000],
            "tool_ok": self.tool_ok,
            "step_sha256": self.step_sha256,
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def capture_layer_records(
    model: Any, tokenizer: Any, text: str
) -> list[dict[str, Any]]:
    """Run a single forward over ``text`` and record every DecoderLayer output."""
    import mlx.core as mx
    from mlx_lm.models.qwen3_5 import DecoderLayer

    records: list[dict[str, Any]] = []
    orig = DecoderLayer.__call__
    counter = {"n": 0}

    def wrapped(self, x, **kw):
        out = orig(self, x, **kw)
        i = counter["n"]
        counter["n"] += 1
        records.append(
            {
                "layer": i,
                "is_linear": bool(self.is_linear),
                "shape": list(out.shape),
                "mean": float(mx.mean(out)),
                "l2_norm": float(mx.linalg.norm(out)),
            }
        )
        return out

    DecoderLayer.__call__ = wrapped
    try:
        lm = model.language_model.model
        ids = tokenizer.encode(text)
        _ = lm(mx.array([ids]))
    finally:
        DecoderLayer.__call__ = orig
    return records


class LocalLensAgent:
    def __init__(
        self,
        *,
        repo: Path,
        issue: str,
        out_dir: Path,
        max_steps: int = 4,
        generator: Callable[[list[dict[str, str]], int], str] | None = None,
        capture: Callable[[Any, Any, str], list[dict[str, Any]]] | None = None,
        model_path: str = _MODEL,
    ) -> None:
        self.repo = repo.resolve()
        self.issue = issue
        self.out_dir = out_dir.resolve()
        self.max_steps = max_steps
        self.generator = generator
        self.capture = capture
        self.model_path = model_path
        self.steps: list[StepEvidence] = []

    # -- tools ------------------------------------------------------------
    def _tool_list(self) -> str:
        files = sorted(
            p.relative_to(self.repo).as_posix()
            for p in self.repo.rglob("*")
            if p.is_file() and "node_modules" not in p.parts and ".git" not in p.parts
        )
        return "\n".join(files[:50]) or "(empty)"

    def _tool_read(self, path: str) -> str:
        p = (self.repo / path).resolve()
        if self.repo not in p.parents:
            raise LensAgentError(f"path escapes repo: {path}")
        if not p.is_file():
            raise LensAgentError(f"not a file: {path}")
        return p.read_text(encoding="utf-8", errors="replace")[:4000]

    def _tool_run(self, cmd: str) -> str:
        parts = cmd.split()
        if not parts or parts[0] not in _ALLOWED_CMD:
            raise LensAgentError(f"command not allowed: {cmd[:80]}")
        proc = subprocess.run(
            parts,
            cwd=self.repo,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return out[:4000]

    def _tool_write(self, path: str, content: str) -> str:
        p = (self.repo / path).resolve()
        if self.repo not in p.parents:
            raise LensAgentError(f"path escapes repo: {path}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {path} ({len(content)} bytes)"

    def _execute(self, action: dict[str, Any]) -> tuple[str, bool]:
        tool = action.get("tool")
        try:
            if tool == "list":
                return self._tool_list(), True
            if tool == "read":
                return self._tool_read(str(action.get("path", ""))), True
            if tool == "run":
                return self._tool_run(str(action.get("cmd", ""))), True
            if tool == "write":
                return self._tool_write(
                    str(action.get("path", "")), str(action.get("content", ""))
                ), True
            if tool == "finish":
                return str(action.get("message", "")), True
            raise LensAgentError(f"unknown tool: {tool}")
        except (LensAgentError, subprocess.TimeoutExpired, OSError) as exc:
            return f"ERROR: {exc}", False

    # -- loop -------------------------------------------------------------
    def _default_generate(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        from mlx_lm import generate, load

        model, tokenizer = load(self.model_path)
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)

    def _parse_action(self, text: str) -> dict[str, Any]:
        m = _ACTION_RE.search(text)
        if not m:
            raise LensAgentError("no JSON action found in model output")
        action = json.loads(m.group(0))
        if not isinstance(action, dict) or "tool" not in action:
            raise LensAgentError("action must contain 'tool'")
        return action

    def run(self) -> dict[str, Any]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Repo: {self.repo}\nIssue: {self.issue}"},
        ]
        model = None
        tokenizer = None
        if self.generator is None:
            from mlx_lm import load

            model, tokenizer = load(self.model_path)

        for step in range(1, self.max_steps + 1):
            if self.generator is not None:
                text = self.generator(messages, 512)
            else:
                prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                from mlx_lm import generate

                text = generate(model, tokenizer, prompt=prompt, max_tokens=512)
            text = text.strip()
            action = self._parse_action(text)
            result, ok = self._execute(action)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "tool", "content": result[:2000]})

            layer_records: list[dict[str, Any]] = []
            if self.capture is not None:
                layer_records = self.capture(model, tokenizer, text)
            elif model is not None:
                layer_records = capture_layer_records(model, tokenizer, text)

            payload = _canonical_json(
                {
                    "step": step,
                    "action_text": text,
                    "action": action,
                    "layer_records": layer_records,
                    "tool_result": result[:2000],
                    "tool_ok": ok,
                }
            )
            evidence = StepEvidence(
                step=step,
                action_text=text,
                action=action,
                layer_records=layer_records,
                tool_result=result,
                tool_ok=ok,
                step_sha256=hashlib.sha256(payload.encode()).hexdigest(),
            )
            self.steps.append(evidence)
            if action.get("tool") == "finish" and ok:
                break

        diff = ""
        try:
            proc = subprocess.run(
                ["git", "-C", str(self.repo), "diff"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            diff = proc.stdout
        except (OSError, subprocess.TimeoutExpired):
            diff = ""

        meta = {
            "model": self.model_path,
            "repo": str(self.repo),
            "issue": self.issue,
            "max_steps": self.max_steps,
            "steps_completed": len(self.steps),
            "diff_bytes": len(diff),
            "observational_not_causal": True,
            "weights_frozen": True,
        }
        (self.out_dir / "meta.json").write_text(
            json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        with (self.out_dir / "STEPS.jsonl").open("w", encoding="utf-8") as handle:
            for ev in self.steps:
                handle.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
        (self.out_dir / "final.diff").write_text(diff, encoding="utf-8")
        return {**meta, "evidence": str(self.out_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--issue", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--model", default=_MODEL)
    args = parser.parse_args()
    agent = LocalLensAgent(
        repo=args.repo,
        issue=args.issue,
        out_dir=args.out,
        max_steps=args.max_steps,
        model_path=args.model,
    )
    result = agent.run()
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
