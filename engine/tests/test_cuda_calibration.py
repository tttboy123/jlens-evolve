from __future__ import annotations

import json
import time

import pytest

from skill_evolution_loop.__main__ import _parser
from skill_evolution_loop.contracts import sha256_json
from skill_evolution_loop.cuda_calibration import (
    CalibrationError,
    build_calibration_manifest,
    collect_cuda_calibration,
    evaluate_calibration,
    replay_cuda_calibration,
)


def _attempt(path, *, structural: bool, output: str = "out") -> None:
    path.mkdir(parents=True)
    (path / "generation-prompt-000.txt").write_text("prompt")
    (path / "generation-output-000.txt").write_text(output)
    (path / "ATTEMPT.json").write_text(
        json.dumps(
            {
                "attempt": {
                    "structural_valid": structural,
                    "patch_sha256": "a" * 64 if structural else None,
                },
                "generation_trace": [
                    {
                        "kind": "attempt-0",
                        "prompt_path": "generation-prompt-000.txt",
                        "path": "generation-output-000.txt",
                    }
                ],
            }
        )
    )


def test_manifest_freezes_three_complete_baseline_taught_pairs(tmp_path) -> None:
    cells = tmp_path / "cells"
    for task in ("task-a", "task-b", "task-c"):
        _attempt(cells / task / "span-baseline", structural=True)
        _attempt(cells / task / "span-taught", structural=True)

    manifest = build_calibration_manifest(
        cells_root=cells,
        task_ids=("task-a", "task-b", "task-c"),
        output_path=tmp_path / "MANIFEST.json",
    )

    assert manifest["task_count"] == 3
    assert manifest["cell_count"] == 6
    assert all(cell["prompt_sha256"] for cell in manifest["cells"])


def test_manifest_rejects_non_three_task_scope(tmp_path) -> None:
    with pytest.raises(CalibrationError, match="exactly three"):
        build_calibration_manifest(
            cells_root=tmp_path,
            task_ids=("task-a",),
            output_path=tmp_path / "MANIFEST.json",
        )


def test_calibration_gate_requires_structural_agreement_and_no_safety_regression() -> (
    None
):
    manifest = {
        "cells": [
            {"cell_id": f"cell-{index}", "mlx_structural_valid": index < 5}
            for index in range(6)
        ]
    }
    cuda = {
        "cells": [
            {
                "cell_id": f"cell-{index}",
                "structural_valid": index < 5,
                "safety_regression": False,
            }
            for index in range(6)
        ]
    }

    report = evaluate_calibration(manifest=manifest, cuda_results=cuda)

    assert report["gate_passed"] is True
    cuda["cells"][0]["safety_regression"] = True
    assert (
        evaluate_calibration(manifest=manifest, cuda_results=cuda)["gate_passed"]
        is False
    )


def test_collect_cuda_calibration_derives_results_from_frozen_attempts(
    tmp_path,
) -> None:
    experiment = tmp_path / "experiment"
    manifest_cells = []
    for index in range(6):
        task = f"task-{index // 2}"
        condition = "span-baseline" if index % 2 == 0 else "span-taught"
        cell = experiment / "cells" / task / condition
        cell.mkdir(parents=True)
        prompt = f"prompt-{index}"
        output = f"output-{index}"
        (cell / "generation-prompt-000.txt").write_text(prompt)
        (cell / "generation-output-000.txt").write_text(output)
        attempt_content = {
            "schema_version": 1,
            "task": {"task_id": task},
            "condition": {"condition_id": condition},
            "attempt": {
                "structural_valid": True,
                "failure_reason": None,
                "patch_sha256": "a" * 64,
            },
            "generation_trace": [
                {
                    "path": "generation-output-000.txt",
                    "sha256": __import__("hashlib").sha256(output.encode()).hexdigest(),
                    "prompt_path": "generation-prompt-000.txt",
                    "prompt_sha256": __import__("hashlib")
                    .sha256(prompt.encode())
                    .hexdigest(),
                }
            ],
        }
        attempt = {
            **attempt_content,
            "evidence_sha256": sha256_json(attempt_content),
        }
        (cell / "ATTEMPT.json").write_text(json.dumps(attempt))
        manifest_cells.append(
            {
                "cell_id": f"{task}/{condition}",
                "task_id": task,
                "condition_id": condition,
                "prompt_sha256": attempt_content["generation_trace"][0][
                    "prompt_sha256"
                ],
                "mlx_structural_valid": True,
            }
        )
    manifest_content = {
        "schema_version": 1,
        "gate_kind": "three-task-mlx-vs-cuda",
        "cells": manifest_cells,
    }
    manifest = {
        **manifest_content,
        "evidence_sha256": sha256_json(manifest_content),
    }

    report = collect_cuda_calibration(
        manifest=manifest,
        experiment_root=experiment,
        output_path=tmp_path / "CUDA-RESULTS.json",
    )

    assert report["cell_count"] == 6
    assert report["cells"][0]["structural_valid"] is True
    assert report["cells"][0]["safety_regression"] is False
    assert report["evidence_sha256"]


def test_collect_cuda_calibration_rejects_prompt_drift(tmp_path) -> None:
    manifest = {
        "cells": [
            {
                "cell_id": "task-a/span-baseline",
                "task_id": "task-a",
                "condition_id": "span-baseline",
                "prompt_sha256": "a" * 64,
                "mlx_structural_valid": True,
            }
        ]
    }
    cell = tmp_path / "cells/task-a/span-baseline"
    cell.mkdir(parents=True)
    (cell / "generation-prompt-000.txt").write_text("different")
    (cell / "generation-output-000.txt").write_text("output")
    content = {
        "task": {"task_id": "task-a"},
        "condition": {"condition_id": "span-baseline"},
        "attempt": {"structural_valid": True, "failure_reason": None},
        "generation_trace": [
            {
                "path": "generation-output-000.txt",
                "sha256": __import__("hashlib").sha256(b"output").hexdigest(),
                "prompt_path": "generation-prompt-000.txt",
                "prompt_sha256": __import__("hashlib").sha256(b"different").hexdigest(),
            }
        ],
    }
    (cell / "ATTEMPT.json").write_text(
        json.dumps({**content, "evidence_sha256": sha256_json(content)})
    )

    with pytest.raises(CalibrationError, match="prompt drift"):
        collect_cuda_calibration(
            manifest=manifest,
            experiment_root=tmp_path,
            output_path=tmp_path / "RESULT.json",
        )


def test_replay_cuda_calibration_is_resumable_and_append_only(tmp_path) -> None:
    prompts = []
    manifest_cells = []
    for index in range(2):
        prompt = tmp_path / f"prompt-{index}.txt"
        prompt.write_text(f"prompt-{index}")
        prompts.append(prompt)
        manifest_cells.append(
            {
                "cell_id": f"task-{index}/span-baseline",
                "prompt_path": prompt.as_posix(),
                "prompt_sha256": __import__("hashlib")
                .sha256(prompt.read_bytes())
                .hexdigest(),
            }
        )
    manifest = {"cells": manifest_cells, "evidence_sha256": "b" * 64}

    class Transport:
        def __init__(self) -> None:
            self.calls = 0

        def generate_prompt(self, request):
            self.calls += 1
            time.sleep(0.001)
            return type(
                "Response",
                (),
                {
                    "text": "completion",
                    "request_sha256": request.fingerprint,
                    "response_sha256": "c" * 64,
                    "model": "same-base-cuda",
                    "finish_reason": "stop",
                    "usage": {"completion_tokens": 1},
                },
            )()

        def identity(self):
            return {"transport_implementation_sha256": "d" * 64}

    transport = Transport()
    first = replay_cuda_calibration(
        manifest=manifest,
        evidence_root=tmp_path / "replay",
        transport=transport,
        max_tokens=32,
    )
    second = replay_cuda_calibration(
        manifest=manifest,
        evidence_root=tmp_path / "replay",
        transport=transport,
        max_tokens=32,
    )

    assert first == second
    assert first["completed_cells"] == 2
    assert first["usage_totals"] == {"completion_tokens": 2}
    assert first["effective_token_ratio"] == 1.0
    assert first["generation_seconds"] > 0
    assert transport.calls == 2
    assert (tmp_path / "replay/cells/task-0/span-baseline/RESPONSE.json").is_file()


def test_cuda_calibration_cli_exposes_replay_collect_and_evaluate() -> None:
    replay = _parser().parse_args(
        [
            "cuda-calibration-replay",
            "--manifest",
            "MANIFEST.json",
            "--out",
            "replay",
            "--transport-base-url",
            "http://127.0.0.1:18000",
            "--transport-model",
            "same-base-cuda",
        ]
    )
    collect = _parser().parse_args(
        [
            "cuda-calibration-collect",
            "--manifest",
            "MANIFEST.json",
            "--experiment",
            "experiment",
            "--out",
            "CUDA-RESULTS.json",
        ]
    )
    evaluate = _parser().parse_args(
        [
            "cuda-calibration-evaluate",
            "--manifest",
            "MANIFEST.json",
            "--cuda-results",
            "CUDA-RESULTS.json",
            "--out",
            "CALIBRATION-REPORT.json",
        ]
    )

    assert replay.transport_base_url.endswith(":18000")
    assert collect.experiment.name == "experiment"
    assert evaluate.cuda_results.name == "CUDA-RESULTS.json"


def test_calibration_commands_reject_tampered_manifest(tmp_path) -> None:
    from skill_evolution_loop.cuda_calibration import load_calibration_evidence

    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps({"cells": [], "evidence_sha256": "a" * 64}))

    with pytest.raises(CalibrationError, match="evidence SHA"):
        load_calibration_evidence(path, "manifest")
