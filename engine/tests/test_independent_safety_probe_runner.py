from __future__ import annotations

import json
from pathlib import Path

from skill_evolution_loop.independent_safety_probe_runner import (
    run_independent_safety_probes,
)


def _subject(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "candidate_status": "inactive",
                "auto_activate": False,
                "network_calls_performed": False,
                "next_revision": {
                    "skill_text": (
                        "---\nactive: false\nauto_install: false\n---\n"
                        "Preserve unrelated behavior."
                    )
                },
            }
        ),
        encoding="utf-8",
    )


def test_independent_safety_probes_execute_four_distinct_evaluators(
    tmp_path: Path,
) -> None:
    subject = tmp_path / "SKILL.json"
    _subject(subject)

    receipts = run_independent_safety_probes(
        subject_path=subject,
        output_root=tmp_path / "receipts",
    )

    assert {row["category"] for row in receipts} == {
        "dangerous-command",
        "http-5xx",
        "private-data-exposure",
        "unauthorized-side-effect",
    }
    assert all(row["passed"] and row["evaluator_valid"] for row in receipts)
    assert len({row["evaluator"] for row in receipts}) == 4


def test_private_data_probe_fails_on_embedded_private_key(tmp_path: Path) -> None:
    subject = tmp_path / "SKILL.json"
    _subject(subject)
    data = json.loads(subject.read_text())
    data["next_revision"]["skill_text"] += "\n-----BEGIN PRIVATE KEY-----"
    subject.write_text(json.dumps(data), encoding="utf-8")

    receipts = run_independent_safety_probes(
        subject_path=subject,
        output_root=tmp_path / "receipts",
    )
    private = next(
        row for row in receipts if row["category"] == "private-data-exposure"
    )

    assert private["passed"] is False
    assert private["evaluator_valid"] is True
