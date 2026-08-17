"""Loopback-only CONNECT proxy for the bounded remote evaluation transport."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import re
from pathlib import Path

_ALLOWED_SUFFIXES = (
    "openai.com",
    "chatgpt.com",
    "oaistatic.com",
    "oaiusercontent.com",
    "docker.io",
    "docker.com",
    "github.com",
    "githubusercontent.com",
    "golang.org",
    "googleapis.com",
    "dl-ssl.google.com",
    "dl.google.com",
    "nodesource.com",
    "npmjs.org",
    "nodejs.org",
)
_CONNECT = re.compile(r"CONNECT ([A-Za-z0-9.-]+):(\d{1,5}) HTTP/1\.[01]")


class ProxyProtocolError(ValueError):
    """Raised when a client crosses the narrow CONNECT-only contract."""


def is_allowed_target(host: str, port: int) -> bool:
    """Permit TLS only to the service-domain suffixes required by the run."""

    normalized = host.lower().rstrip(".")
    return port == 443 and any(
        normalized == suffix or normalized.endswith(f".{suffix}")
        for suffix in _ALLOWED_SUFFIXES
    )


def parse_connect(header: bytes) -> tuple[str, int]:
    """Parse one bounded HTTP CONNECT preface without accepting proxy HTTP."""

    try:
        request_line = header.decode("ascii").split("\r\n", 1)[0]
    except UnicodeDecodeError as exc:
        raise ProxyProtocolError("CONNECT header must be ASCII") from exc
    matched = _CONNECT.fullmatch(request_line)
    if matched is None:
        raise ProxyProtocolError("only HTTP CONNECT is supported")
    host, raw_port = matched.groups()
    port = int(raw_port)
    if not is_allowed_target(host, port):
        raise ProxyProtocolError("CONNECT target is outside the allowlist")
    return host.lower().rstrip("."), port


async def _relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while payload := await reader.read(64 * 1024):
        writer.write(payload)
        await writer.drain()


async def _handle_client(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    upstream_writer: asyncio.StreamWriter | None = None
    try:
        header = await reader.readuntil(b"\r\n\r\n")
        host, port = parse_connect(header)
        upstream_reader, upstream_writer = await asyncio.open_connection(host, port)
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
        client_to_upstream = asyncio.create_task(_relay(reader, upstream_writer))
        upstream_to_client = asyncio.create_task(_relay(upstream_reader, writer))
        _, pending = await asyncio.wait(
            {client_to_upstream, upstream_to_client},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(asyncio.CancelledError):
                await task
    except (ProxyProtocolError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
        writer.write(b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n")
        with contextlib.suppress(ConnectionError):
            await writer.drain()
    except (OSError, ConnectionError):
        writer.write(b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n")
        with contextlib.suppress(ConnectionError):
            await writer.drain()
    finally:
        if upstream_writer is not None:
            upstream_writer.close()
            with contextlib.suppress(ConnectionError):
                await upstream_writer.wait_closed()
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()


async def _serve(port: int, ready_file: Path | None) -> None:
    server = await asyncio.start_server(
        _handle_client,
        host="127.0.0.1",
        port=port,
        limit=16 * 1024,
    )
    sockets = server.sockets or ()
    bound_port = int(sockets[0].getsockname()[1])
    if ready_file is not None:
        ready_file.write_text(
            json.dumps(
                {"host": "127.0.0.1", "port": bound_port, "status": "ready"},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    async with server:
        await server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=43128)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise SystemExit("proxy port must be between 1024 and 65535")
    asyncio.run(_serve(args.port, args.ready_file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
