"""Run statically allowlisted harness code in a bounded isolated subprocess."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from agent_code_mutation import MutationContractError, validate_source

WORKER = Path(__file__).resolve().with_name("sandbox_worker.py")


def run_candidate(
    *,
    source: str,
    cases: list[dict[str, Any]],
    limits: dict[str, Any],
    sandbox_parent: Path,
) -> dict[str, Any]:
    report = validate_source(source, limits=limits)
    if not report["allowed"]:
        raise MutationContractError(
            "candidate failed static gate: " + ", ".join(report["reasons"])
        )
    sandbox_parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"source": source, "cases": cases, "limits": limits}, sort_keys=True
    )
    with tempfile.TemporaryDirectory(dir=sandbox_parent) as directory:
        cwd = Path(directory)
        completed = subprocess.run(
            [sys.executable, "-I", "-S", str(WORKER)],
            input=payload,
            text=True,
            capture_output=True,
            cwd=cwd,
            env={},
            timeout=float(limits["timeout_seconds"]),
            check=False,
        )
        files_after = sorted(
            path.relative_to(cwd).as_posix() for path in cwd.rglob("*")
        )
    if completed.returncode != 0:
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "passed_cases": 0,
            "total_cases": len(cases),
            "case_results": [],
            "static_report": report,
            "sandbox": {
                "isolated_python": True,
                "empty_environment": True,
                "empty_cwd_before": True,
                "files_after": files_after,
                "limits": limits,
                "resource_limits": {},
            },
        }
    child = json.loads(completed.stdout)
    results = child["case_results"]
    return {
        "status": "completed",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed_cases": sum(row["passed"] for row in results),
        "total_cases": len(results),
        "case_results": results,
        "static_report": report,
        "sandbox": {
            "isolated_python": True,
            "empty_environment": True,
            "empty_cwd_before": True,
            "files_after": files_after,
            "limits": limits,
            "resource_limits": child["resource_limits"],
        },
    }
