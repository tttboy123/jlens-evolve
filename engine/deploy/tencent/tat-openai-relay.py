#!/usr/bin/env python3
"""Localhost OpenAI relay over Tencent TAT for a private worker endpoint."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import subprocess
import threading
import time
from binascii import Error as BinasciiError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MAX_HTTP_BODY = 48 * 1024
MAX_TAT_CONTENT = 64 * 1024


class RelayError(RuntimeError):
    pass


def encode_tat_payload(value: bytes) -> str:
    return base64.b64encode(gzip.compress(value, mtime=0)).decode("ascii")


class CloudSkillsMcpClient:
    """Small persistent stdio client for the credential-isolating cloud gateway."""

    def __init__(self, command: str) -> None:
        environment = dict(os.environ)
        environment["CLOUD_SKILLS_ALLOW_MUTATIONS"] = "1"
        self._process = subprocess.Popen(
            [command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=environment,
        )
        self._lock = threading.Lock()
        self._request_id = 0
        self._request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "jlens-tat-relay", "version": "1"},
            },
        )
        self._write({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call(self, *, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            result = self._request(
                "tools/call", {"name": tool, "arguments": arguments}
            )
        blocks = result.get("content")
        if not isinstance(blocks, list) or not blocks:
            raise RelayError("cloud gateway returned no content")
        text = blocks[0].get("text") if isinstance(blocks[0], dict) else None
        if not isinstance(text, str):
            raise RelayError("cloud gateway returned invalid content")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RelayError("cloud gateway returned invalid JSON") from exc
        response = payload.get("Response")
        if not isinstance(response, dict):
            raise RelayError("cloud gateway returned no Tencent response")
        error = response.get("Error")
        if isinstance(error, dict):
            code = error.get("Code", "unknown")
            raise RelayError(f"Tencent API failed: {code}")
        return response

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._request_id += 1
        request_id = self._request_id
        self._write(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
        assert self._process.stdout is not None
        while True:
            line = self._process.stdout.readline()
            if not line:
                raise RelayError("cloud gateway closed unexpectedly")
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RelayError("cloud gateway request failed")
            result = message.get("result")
            if not isinstance(result, dict):
                raise RelayError("cloud gateway returned invalid result")
            return result

    def _write(self, message: dict[str, Any]) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self._process.stdin.flush()


class TatRelay:
    def __init__(
        self,
        *,
        instance_id: str,
        region: str,
        remote_base_url: str,
        timeout_seconds: int,
        poll_seconds: float = 1.0,
        cloud_skills_mcp: str | None = None,
    ) -> None:
        self.instance_id = instance_id
        self.region = region
        self.remote_base_url = remote_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self.last_tat_content = ""
        self._mcp = (
            CloudSkillsMcpClient(cloud_skills_mcp) if cloud_skills_mcp else None
        )

    def forward(self, path: str, body: bytes) -> bytes:
        if path not in {"/v1/completions", "/v1/chat/completions"}:
            raise RelayError("unsupported OpenAI path")
        try:
            json.loads(body)
        except json.JSONDecodeError as exc:
            raise RelayError("request body is not JSON") from exc
        payload = encode_tat_payload(body)
        script = (
            "set -eu; f=$(mktemp); r=$(mktemp); e=$(mktemp); "
            f"printf '%s' '{payload}' | base64 -d | gzip -d >\"$f\"; "
            "trap 'rm -f \"$f\" \"$r\" \"$e\"' EXIT; "
            f"if ! curl -fsS --max-time {self.timeout_seconds} "
            "-H 'Content-Type: application/json' --data-binary @\"$f\" "
            f"'{self.remote_base_url}{path}' >\"$r\" 2>\"$e\"; then "
            "cat \"$e\" >&2; exit 22; fi; gzip -c \"$r\""
        )
        content = base64.b64encode(script.encode()).decode("ascii")
        self.last_tat_content = content
        if len(content) > MAX_TAT_CONTENT:
            raise RelayError("request exceeds TAT command limit")
        invocation = self._run_command(content)
        invocation_id = invocation.get("InvocationId")
        if not isinstance(invocation_id, str):
            raise RelayError("TAT did not return an invocation ID")
        deadline = time.monotonic() + self.timeout_seconds + 45
        while time.monotonic() < deadline:
            result = self._describe_invocation_tasks(invocation_id)
            tasks = result.get("InvocationTaskSet")
            if isinstance(tasks, list) and tasks:
                task = tasks[0]
                status = task.get("TaskStatus")
                if status == "SUCCESS":
                    encoded = task.get("TaskResult", {}).get("Output")
                    if not isinstance(encoded, str):
                        raise RelayError("TAT response output is missing")
                    try:
                        output = gzip.decompress(base64.b64decode(encoded))
                    except (BinasciiError, EOFError, gzip.BadGzipFile) as exc:
                        raise RelayError("TAT response envelope is invalid") from exc
                    json.loads(output)
                    return output
                if status in {"FAILED", "TIMEOUT", "CANCELLED", "TERMINATED"}:
                    raise RelayError(f"TAT invocation ended with {status}")
            time.sleep(self.poll_seconds)
        raise RelayError("TAT invocation timed out")

    def _run_command(self, content: str) -> dict[str, Any]:
        if self._mcp is not None:
            return self._mcp.call(
                tool="tencent_api_mutate",
                arguments={
                    **self._mcp_common("RunCommand"),
                    "body": {
                        "Content": content,
                        "InstanceIds": [self.instance_id],
                        "CommandName": "jlens-openai-relay",
                        "CommandType": "SHELL",
                        "Timeout": self.timeout_seconds + 30,
                    },
                    "force": True,
                },
            )
        return self._tccli(
            "tat",
            "RunCommand",
            "--region",
            self.region,
            "--Content",
            content,
            "--InstanceIds",
            json.dumps([self.instance_id]),
            "--CommandName",
            "jlens-openai-relay",
            "--CommandType",
            "SHELL",
            "--Timeout",
            str(self.timeout_seconds + 30),
        )

    def _describe_invocation_tasks(self, invocation_id: str) -> dict[str, Any]:
        if self._mcp is not None:
            return self._mcp.call(
                tool="tencent_api_read",
                arguments={
                    **self._mcp_common("DescribeInvocationTasks"),
                    "body": {
                        "Filters": [
                            {"Name": "invocation-id", "Values": [invocation_id]}
                        ],
                        "HideOutput": False,
                    },
                },
            )
        return self._tccli(
            "tat",
            "DescribeInvocationTasks",
            "--region",
            self.region,
            "--Filters",
            json.dumps([{"Name": "invocation-id", "Values": [invocation_id]}]),
            "--HideOutput",
            "False",
        )

    def _mcp_common(self, operation: str) -> dict[str, Any]:
        return {
            "auth_scheme": "tc3",
            "service": "tat",
            "operation": operation,
            "api_version": "2020-10-28",
            "region": self.region,
            "method": "POST",
            "url": "https://tat.tencentcloudapi.com/",
        }

    @staticmethod
    def _tccli(*arguments: str) -> dict[str, Any]:
        if not os.environ.get("TENCENTCLOUD_SECRET_ID") or not os.environ.get(
            "TENCENTCLOUD_SECRET_KEY"
        ):
            raise RelayError("Tencent credentials are not configured")
        process = subprocess.run(
            ["tccli", *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if process.returncode != 0:
            raise RelayError("tccli request failed")
        try:
            value = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise RelayError("tccli returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise RelayError("tccli returned invalid payload")
        return value


def handler_for(relay: TatRelay) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_HTTP_BODY:
                    raise RelayError("invalid request length")
                output = relay.forward(self.path, self.rfile.read(length))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(output)))
                self.end_headers()
                self.wfile.write(output)
            except (RelayError, ValueError, json.JSONDecodeError) as exc:
                output = json.dumps({"error": {"message": str(exc)}}).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(output)))
                self.end_headers()
                self.wfile.write(output)

        def log_message(self, format: str, *args: object) -> None:
            print(f"tat-openai-relay: {format % args}", flush=True)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--listen-port", type=int, default=18000)
    parser.add_argument("--remote-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--cloud-skills-mcp", default=None)
    args = parser.parse_args()
    relay = TatRelay(
        instance_id=args.instance_id,
        region=args.region,
        remote_base_url=args.remote_base_url,
        timeout_seconds=args.timeout_seconds,
        cloud_skills_mcp=args.cloud_skills_mcp,
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.listen_port), handler_for(relay)
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
