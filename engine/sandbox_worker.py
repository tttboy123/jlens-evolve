"""Trusted child-process worker for already allowlisted pure route functions."""

from __future__ import annotations

import json
import resource
import sys


def _apply_limits(limits):
    requested = {
        "cpu_seconds": (resource.RLIMIT_CPU, int(limits["cpu_seconds"])),
        "file_bytes": (resource.RLIMIT_FSIZE, int(limits["file_bytes"])),
        "open_files": (resource.RLIMIT_NOFILE, int(limits["open_files"])),
    }
    if hasattr(resource, "RLIMIT_AS"):
        requested["address_space_bytes"] = (
            resource.RLIMIT_AS,
            int(limits["address_space_mb"]) * 1024 * 1024,
        )
    report = {}
    for name, (resource_id, value) in requested.items():
        try:
            _, hard = resource.getrlimit(resource_id)
            effective = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(resource_id, (effective, effective))
            report[name] = {"applied": True, "value": effective}
        except (OSError, ValueError) as exc:
            report[name] = {"applied": False, "error": type(exc).__name__}
    return report


def main() -> int:
    payload = json.loads(sys.stdin.read())
    limit_report = _apply_limits(payload["limits"])
    namespace: dict[str, object] = {}
    exec(  # noqa: S102 - source was statically allowlisted by the parent process.
        compile(payload["source"], "<agent-harness-candidate>", "exec"),
        {"__builtins__": {}},
        namespace,
    )
    function = namespace["select_route"]
    rows = []
    for case in payload["cases"]:
        actual = function(*case["args"])
        rows.append(
            {
                "id": case["id"],
                "expected": case["expected"],
                "actual": actual,
                "passed": actual == case["expected"],
            }
        )
    sys.stdout.write(
        json.dumps(
            {"case_results": rows, "resource_limits": limit_report}, sort_keys=True
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
