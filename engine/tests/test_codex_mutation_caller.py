from __future__ import annotations

import json
from pathlib import Path

from codex_mutation_caller import CodexMutationCaller


def test_codex_mutation_caller_freezes_raw_events_stderr_usage_and_receipt(
    tmp_path: Path,
    monkeypatch,
):
    fake = tmp_path / "codex"
    fake.write_text(
        "#!/bin/sh\n"
        "while [ $# -gt 0 ]; do\n"
        '  if [ "$1" = "--output-last-message" ]; then shift; out=$1; fi\n'
        "  shift\n"
        "done\n"
        "input=$(cat)\n"
        'printf \'%s\\n\' \'{"usage":{"input_tokens":11,"output_tokens":7}}\'\n'
        "printf '%s\\n' 'diagnostic only' >&2\n"
        "printf '%s' '{\"schema_version\":1}' > \"$out\"\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setenv("EVOLVE_CODEX_HTTPS_PROXY", "http://127.0.0.1:43128")
    monkeypatch.setenv("HTTPS_PROXY", "http://untrusted.example:9999")
    caller = CodexMutationCaller(
        codex_executable=fake,
        output_root=tmp_path / "calls",
        working_directory=tmp_path,
        model="gpt-test",
        reasoning="low",
        timeout_seconds=30,
    )

    response = caller("produce one inactive ChangeSet")

    assert response == '{"schema_version":1}'
    call_roots = [path for path in (tmp_path / "calls").iterdir() if path.is_dir()]
    assert len(call_roots) == 1
    root = call_roots[0]
    receipt = json.loads((root / "receipt.json").read_text())
    assert receipt["status"] == "completed"
    assert receipt["returncode"] == 0
    assert receipt["usage"] == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }
    assert receipt["prompt"]["sha256"]
    assert receipt["response"]["sha256"]
    assert receipt["network_transport"]["loopback_proxy_configured"] is True
    argv = receipt["argv"]
    for feature in (
        "apps",
        "enable_mcp_apps",
        "plugins",
        "browser_use",
        "computer_use",
        "multi_agent",
        "memories",
    ):
        assert feature in argv
    assert "mcp_servers={}" in argv
    assert (root / "events.jsonl").is_file()
    assert (root / "stderr.txt").read_text() == "diagnostic only\n"
    assert (root / "prompt.txt").read_text() == "produce one inactive ChangeSet"


def test_codex_mutation_caller_reuses_completed_identical_prompt(tmp_path: Path):
    marker = tmp_path / "count.txt"
    fake = tmp_path / "codex"
    fake.write_text(
        "#!/bin/sh\n"
        "while [ $# -gt 0 ]; do\n"
        '  if [ "$1" = "--output-last-message" ]; then shift; out=$1; fi\n'
        "  shift\n"
        "done\n"
        f"printf x >> '{marker}'\n"
        "cat >/dev/null\n"
        "printf '%s' '{\"ok\":true}' > \"$out\"\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    caller = CodexMutationCaller(
        codex_executable=fake,
        output_root=tmp_path / "calls",
        working_directory=tmp_path,
        model="gpt-test",
        reasoning="low",
        timeout_seconds=30,
    )

    assert caller("same") == '{"ok":true}'
    assert caller("same") == '{"ok":true}'
    assert marker.read_text() == "x"
