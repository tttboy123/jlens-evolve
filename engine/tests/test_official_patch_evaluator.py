from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from official_patch_evaluator import (
    OfficialEvaluatorError,
    OfficialEvaluatorTimeout,
    OfficialPatchEvaluator,
    freeze_official_harness_runtime,
)


def _invocation(tmp_path: Path, benchmark_id: str, instance_id: str):
    evidence = tmp_path / "agent-evidence"
    evidence.mkdir()
    prediction = evidence / "prediction.patch"
    prediction.write_text("diff --git a/a b/a\n", encoding="utf-8")
    return SimpleNamespace(
        round_id="g0-observe-task",
        arm="original",
        benchmark_id=benchmark_id,
        instance_id=instance_id,
        agent_program_sha256="a" * 64,
        evidence_dir=str(evidence),
    ), {"prediction": {"path": str(prediction)}}


def _harness_repo(root: Path, entrypoint: str) -> str:
    source = root / entrypoint
    source.parent.mkdir(parents=True)
    parent = source.parent
    while parent != root:
        (parent / "__init__.py").touch()
        parent = parent.parent
    source.write_text("def main():\n    return 0\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q"), cwd=root, check=True)
    subprocess.run(("git", "add", "."), cwd=root, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ),
        cwd=root,
        check=True,
    )
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_official_harness_runtime_fails_fast_when_editable_source_disappears(
    tmp_path: Path,
):
    swe = tmp_path / "swe"
    multi = tmp_path / "multi"
    swe.mkdir()
    multi.mkdir()
    _harness_repo(multi, "multi_swe_bench/harness/run_evaluation.py")

    with pytest.raises(OfficialEvaluatorError, match="source entry point is missing"):
        freeze_official_harness_runtime(
            swe_python=Path("/usr/bin/python3"),
            multi_python=Path("/usr/bin/python3"),
            swe_harness_root=swe,
            multi_harness_root=multi,
            output_root=tmp_path / "official",
        )

    assert not (tmp_path / "official/HARNESS-RUNTIME-RECEIPT.json").exists()


def test_official_harness_runtime_rejects_an_unimportable_source_tree(tmp_path: Path):
    swe = tmp_path / "swe"
    multi = tmp_path / "multi"
    swe.mkdir()
    multi.mkdir()
    _harness_repo(swe, "swebench/harness/run_evaluation.py")
    _harness_repo(multi, "multi_swe_bench/harness/run_evaluation.py")
    (multi / "multi_swe_bench/harness/__init__.py").write_text(
        "import dependency_that_does_not_exist\n", encoding="utf-8"
    )

    with pytest.raises(OfficialEvaluatorError, match="module import failed"):
        freeze_official_harness_runtime(
            swe_python=Path("/usr/bin/python3"),
            multi_python=Path("/usr/bin/python3"),
            swe_harness_root=swe,
            multi_harness_root=multi,
            output_root=tmp_path / "official",
        )


def test_official_harness_runtime_binds_pinned_revisions_and_source_hashes(
    tmp_path: Path,
):
    swe = tmp_path / "swe"
    multi = tmp_path / "multi"
    swe.mkdir()
    multi.mkdir()
    swe_revision = _harness_repo(swe, "swebench/harness/run_evaluation.py")
    setup = multi / "setup.py"
    setup.write_text("from setuptools import setup\nsetup()\n", encoding="utf-8")
    multi_revision = _harness_repo(multi, "multi_swe_bench/harness/run_evaluation.py")
    assets = tmp_path / "NATIVE-ASSETS.json"
    assets.write_text(
        json.dumps(
            {
                "swe_harness": {"revision": swe_revision},
                "multi_swe_harness": {
                    "revision": multi_revision,
                    "setup_py_sha256": hashlib.sha256(setup.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    receipt_path = freeze_official_harness_runtime(
        swe_python=Path("/usr/bin/python3"),
        multi_python=Path("/usr/bin/python3"),
        swe_harness_root=swe,
        multi_harness_root=multi,
        output_root=tmp_path / "official",
        native_assets_path=assets,
    )
    receipt = json.loads(receipt_path.read_text())

    assert receipt["status"] == "ready"
    assert receipt["official_patch_evaluator_sha256"]
    assert receipt["runtimes"]["swe"]["revision"] == swe_revision
    assert receipt["runtimes"]["swe"]["entrypoint_sha256"]
    assert receipt["runtimes"]["multi_swe"]["revision"] == multi_revision
    assert (
        receipt["native_assets_sha256"]
        == hashlib.sha256(assets.read_bytes()).hexdigest()
    )


def test_official_swe_evaluator_freezes_prediction_and_copies_native_report(
    tmp_path: Path,
):
    invocation, receipt = _invocation(tmp_path, "swe-bench-verified", "example__repo-1")
    pool = tmp_path / "pool"
    dataset = pool / "harness-inputs/swe-bench-verified.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}\n", encoding="utf-8")
    harness = tmp_path / "swe-harness"
    harness.mkdir()

    def run_command(args, cwd, _timeout):
        assert args[args.index("--namespace") + 1] == "none"
        assert args[args.index("--cache_level") + 1] == "instance"
        predictions = Path(args[args.index("--predictions_path") + 1])
        row = json.loads(predictions.read_text())
        run_id = args[args.index("--run_id") + 1]
        report = (
            cwd
            / "logs/run_evaluation"
            / run_id
            / row["model_name_or_path"]
            / row["instance_id"]
            / "report.json"
        )
        report.parent.mkdir(parents=True)
        report.write_text(
            json.dumps(
                {
                    row["instance_id"]: {
                        "resolved": True,
                        "patch_successfully_applied": True,
                        "tests_status": {},
                    }
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, "ok", "")

    evaluator = OfficialPatchEvaluator(
        swe_python=Path("/usr/bin/python3"),
        multi_python=Path("/usr/bin/python3"),
        swe_harness_root=harness,
        multi_harness_root=tmp_path,
        pool_root=pool,
        output_root=tmp_path / "native",
        run_command=run_command,
    )

    report = evaluator(invocation, {}, receipt)

    assert report.is_file()
    frozen = json.loads(
        (tmp_path / "native/g0-observe-task/original/prediction.jsonl").read_text()
    )
    assert frozen["instance_id"] == "example__repo-1"
    assert frozen["model_patch"].startswith("diff --git")
    native_receipt = json.loads(
        (
            tmp_path / "native/g0-observe-task/original/NATIVE-EVALUATOR-RECEIPT.json"
        ).read_text()
    )
    assert native_receipt["status"] == "completed"
    assert native_receipt["native_report"]["sha256"]
    command = native_receipt["command"]
    run_id = command[command.index("--run_id") + 1]
    assert run_id.startswith("g0-observe-task-original-")
    assert len(run_id.rsplit("-", 1)[1]) == 12


def test_official_swe_timeout_freezes_failure_and_replays_without_rerun(
    tmp_path: Path,
):
    invocation, receipt = _invocation(tmp_path, "swe-bench-verified", "example__repo-1")
    pool = tmp_path / "pool"
    dataset = pool / "harness-inputs/swe-bench-verified.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}\n", encoding="utf-8")
    harness = tmp_path / "swe-harness"
    harness.mkdir()
    calls = 0

    def run_command(args, _cwd, timeout):
        nonlocal calls
        calls += 1
        raise subprocess.TimeoutExpired(
            cmd=args,
            timeout=timeout,
            output="partial stdout",
            stderr=b"partial stderr",
        )

    evaluator = OfficialPatchEvaluator(
        swe_python=Path("/usr/bin/python3"),
        multi_python=Path("/usr/bin/python3"),
        swe_harness_root=harness,
        multi_harness_root=tmp_path,
        pool_root=pool,
        output_root=tmp_path / "native",
        run_command=run_command,
        timeout_seconds=17,
    )

    for _ in range(2):
        with pytest.raises(OfficialEvaluatorTimeout, match="timed out") as captured:
            evaluator(invocation, {}, receipt)
        assert captured.value.error_code == "evaluator_timeout"
        assert captured.value.evidence_path is not None

    assert calls == 1
    root = tmp_path / "native/g0-observe-task/original"
    failure = json.loads((root / "NATIVE-EVALUATOR-FAILURE.json").read_text())
    assert failure["status"] == "failed"
    assert failure["error_code"] == "evaluator_timeout"
    assert failure["timeout_seconds"] == 17
    assert (root / "stdout.log").read_text() == "partial stdout"
    assert (root / "stderr.log").read_text() == "partial stderr"


def test_multilingual_swe_uses_local_images_and_rejects_schema2_infrastructure_error(
    tmp_path: Path,
):
    invocation, receipt = _invocation(
        tmp_path, "swe-bench-multilingual", "example__repo-1"
    )
    pool = tmp_path / "pool"
    dataset = pool / "harness-inputs/swe-bench-multilingual.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}\n", encoding="utf-8")
    harness = tmp_path / "swe-harness"
    harness.mkdir()

    def run_command(args, cwd, _timeout):
        assert args[args.index("--namespace") + 1] == "none"
        assert args[args.index("--cache_level") + 1] == "instance"
        model_name = json.loads(
            Path(args[args.index("--predictions_path") + 1]).read_text()
        )["model_name_or_path"]
        run_id = args[args.index("--run_id") + 1]
        aggregate = cwd / f"{model_name}.{run_id}.json"
        (cwd / "run_instance.log").write_text("instance details\n", encoding="utf-8")
        build_log = (
            cwd
            / "logs/build_images/instances"
            / "sweb.eval.x86_64.example__repo-1__latest/build_image.log"
        )
        build_log.parent.mkdir(parents=True)
        build_log.write_text("compiler failed\n", encoding="utf-8")
        aggregate.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "submitted_ids": ["example__repo-1"],
                    "completed_ids": [],
                    "resolved_ids": [],
                    "unresolved_ids": [],
                    "error_ids": ["example__repo-1"],
                    "empty_patch_ids": [],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, "native error summary", "")

    evaluator = OfficialPatchEvaluator(
        swe_python=Path("/usr/bin/python3"),
        multi_python=Path("/usr/bin/python3"),
        swe_harness_root=harness,
        multi_harness_root=tmp_path,
        pool_root=pool,
        output_root=tmp_path / "native",
        run_command=run_command,
    )

    with pytest.raises(OfficialEvaluatorError, match="infrastructure error"):
        evaluator(invocation, {}, receipt)

    root = tmp_path / "native/g0-observe-task/original"
    assert json.loads((root / "native-invalid-report.json").read_text())[
        "error_ids"
    ] == ["example__repo-1"]
    assert (root / "NATIVE-EVALUATOR-FAILURE.json").is_file()
    failure = json.loads((root / "NATIVE-EVALUATOR-FAILURE.json").read_text())
    assert failure["diagnostic_artifacts"] == {
        "build-image.log": hashlib.sha256(b"compiler failed\n").hexdigest(),
        "run-instance.log": hashlib.sha256(b"instance details\n").hexdigest(),
    }
    assert (root / "build-image.log").read_text() == "compiler failed\n"
    assert not (root / "NATIVE-EVALUATOR-RECEIPT.json").exists()


def test_official_swe_rejects_stale_tests_after_build_failure(tmp_path: Path):
    invocation, receipt = _invocation(
        tmp_path, "swe-bench-multilingual", "example__repo-1"
    )
    pool = tmp_path / "pool"
    dataset = pool / "harness-inputs/swe-bench-multilingual.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}\n", encoding="utf-8")
    harness = tmp_path / "swe-harness"
    harness.mkdir()

    def run_command(args, cwd, _timeout):
        prediction = json.loads(
            Path(args[args.index("--predictions_path") + 1]).read_text()
        )
        run_id = args[args.index("--run_id") + 1]
        evidence = (
            cwd
            / "logs/run_evaluation"
            / run_id
            / prediction["model_name_or_path"]
            / prediction["instance_id"]
        )
        evidence.mkdir(parents=True)
        (evidence / "report.json").write_text(
            json.dumps(
                {
                    prediction["instance_id"]: {
                        "resolved": True,
                        "patch_successfully_applied": True,
                        "tests_status": {},
                    }
                }
            ),
            encoding="utf-8",
        )
        (evidence / "test_output.txt").write_text(
            "cmake --build build\nsource.cc:12:7: error: unknown identifier\n"
            "gmake: *** [target] Error 2\n"
            ">>>>> Start Test Output\n[ RUN ] stale-test\n[ OK ] stale-test\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, "harness says resolved", "")

    evaluator = OfficialPatchEvaluator(
        swe_python=Path("/usr/bin/python3"),
        multi_python=Path("/usr/bin/python3"),
        swe_harness_root=harness,
        multi_harness_root=tmp_path,
        pool_root=pool,
        output_root=tmp_path / "native",
        run_command=run_command,
    )

    with pytest.raises(OfficialEvaluatorError, match="build failed before tests"):
        evaluator(invocation, {}, receipt)

    root = tmp_path / "native/g0-observe-task/original"
    assert (root / "native-invalid-report.json").is_file()
    assert (root / "test-output.log").is_file()
    assert not (root / "NATIVE-EVALUATOR-RECEIPT.json").exists()
    failure = json.loads((root / "NATIVE-EVALUATOR-FAILURE.json").read_text())
    assert failure["reason"] == "official SWE evaluator build failed before tests"


def test_official_multi_swe_evaluator_uses_native_org_repo_number_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    invocation, receipt = _invocation(
        tmp_path, "multi-swe-bench-flash", "example__repo-7"
    )
    pool = tmp_path / "pool"
    dataset = pool / "inputs/multi-swe-bench-flash/multi_swe_bench_flash.jsonl"
    dataset.parent.mkdir(parents=True)
    source_row = {
        "instance_id": "example__repo-7",
        "org": "example",
        "repo": "repo",
        "number": 7,
        "base": None,
    }
    canonical_source = json.dumps(
        source_row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    dataset.write_text(json.dumps(source_row) + "\n", encoding="utf-8")
    harness = tmp_path / "multi-harness"
    harness.mkdir()
    monkeypatch.setenv("EVOLVE_BENCHMARK_HTTPS_PROXY", "http://127.0.0.1:43128")

    def repository_command(args, _cwd, _timeout):
        assert "http.proxy=http://127.0.0.1:43128" in args
        target = Path(args[-1])
        (target / ".git").mkdir(parents=True)
        return subprocess.CompletedProcess(args, 0, "cloned", "")

    def run_command(args, _cwd, _timeout):
        config = json.loads(Path(args[-1]).read_text())
        patch = json.loads(Path(config["patch_files"][0]).read_text())
        assert patch == {
            "org": "example",
            "repo": "repo",
            "number": 7,
            "fix_patch": "diff --git a/a b/a\n",
        }
        report = Path(config["workdir"]) / "example/repo/evaluation/image/report.json"
        report.parent.mkdir(parents=True)
        report.write_text(
            json.dumps(
                {
                    "org": "example",
                    "repo": "repo",
                    "number": 7,
                    "valid": True,
                    "test_patch_result": {"passed_tests": []},
                    "fix_patch_result": {"failed_tests": []},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, "ok", "")

    evaluator = OfficialPatchEvaluator(
        swe_python=Path("/usr/bin/python3"),
        multi_python=Path("/usr/bin/python3"),
        swe_harness_root=tmp_path,
        multi_harness_root=harness,
        pool_root=pool,
        output_root=tmp_path / "native",
        run_command=run_command,
        repository_command=repository_command,
    )

    report = evaluator(
        invocation,
        {
            "source_content_sha256": hashlib.sha256(
                canonical_source.encode("utf-8")
            ).hexdigest()
        },
        receipt,
    )

    assert json.loads(report.read_text())["number"] == 7
    config = json.loads(
        (tmp_path / "native/g0-observe-task/original/config.json").read_text()
    )
    assert config["specifics"] == ["example/repo:pr-7"]
    assert config["max_workers_run_instance"] == 1
    prefetch = json.loads(
        (
            tmp_path
            / "native/g0-observe-task/original/REPOSITORY-PREFETCH-RECEIPT.json"
        ).read_text()
    )
    assert prefetch["status"] == "completed"
    assert prefetch["repository"] == "example/repo"
    assert prefetch["strategy"] == "loopback-proxy"


def test_multi_swe_prefetch_falls_back_to_direct_when_proxy_unset(tmp_path: Path):
    invocation, receipt = _invocation(
        tmp_path, "multi-swe-bench-flash", "example__repo-7"
    )
    pool = tmp_path / "pool"
    dataset = pool / "inputs/multi-swe-bench-flash/multi_swe_bench_flash.jsonl"
    dataset.parent.mkdir(parents=True)
    source_row = {
        "instance_id": "example__repo-7",
        "org": "example",
        "repo": "repo",
        "number": 7,
        "base": None,
    }
    dataset.write_text(json.dumps(source_row) + "\n", encoding="utf-8")
    harness = tmp_path / "multi-harness"
    harness.mkdir()

    def repository_command(args, _cwd, _timeout):
        assert not any(arg.startswith("http.proxy=") for arg in args)
        target = Path(args[-1])
        target.mkdir(parents=True)
        subprocess.run(
            ["git", "init", "-q"], cwd=target, check=True, capture_output=True
        )
        return subprocess.CompletedProcess(args, 0, "cloned", "")

    def run_command(args, _cwd, _timeout):
        config = json.loads(Path(args[-1]).read_text())
        report = Path(config["workdir"]) / "example/repo/evaluation/image/report.json"
        report.parent.mkdir(parents=True)
        report.write_text(
            json.dumps(
                {
                    "org": "example",
                    "repo": "repo",
                    "number": 7,
                    "valid": True,
                    "test_patch_result": {"passed_tests": []},
                    "fix_patch_result": {"failed_tests": []},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, "ok", "")

    evaluator = OfficialPatchEvaluator(
        swe_python=Path("/usr/bin/python3"),
        multi_python=Path("/usr/bin/python3"),
        swe_harness_root=tmp_path,
        multi_harness_root=harness,
        pool_root=pool,
        output_root=tmp_path / "native",
        run_command=run_command,
        repository_command=repository_command,
    )

    evaluator(
        invocation,
        {
            "source_content_sha256": hashlib.sha256(
                json.dumps(
                    source_row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        },
        receipt,
    )

    prefetch = json.loads(
        (
            tmp_path
            / "native/g0-observe-task/original/REPOSITORY-PREFETCH-RECEIPT.json"
        ).read_text()
    )
    assert prefetch["strategy"] == "direct"


def test_multi_swe_failure_persists_process_and_reason_before_raising(tmp_path: Path):
    invocation, receipt = _invocation(
        tmp_path, "multi-swe-bench-flash", "example__repo-7"
    )
    pool = tmp_path / "pool"
    dataset = pool / "inputs/multi-swe-bench-flash/multi_swe_bench_flash.jsonl"
    dataset.parent.mkdir(parents=True)
    source_row = {
        "instance_id": "example__repo-7",
        "org": "example",
        "repo": "repo",
        "number": 7,
        "base": None,
    }
    canonical_source = json.dumps(
        source_row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    dataset.write_text(json.dumps(source_row) + "\n", encoding="utf-8")
    harness = tmp_path / "multi-harness"
    harness.mkdir()

    def repository_command(args, _cwd, _timeout):
        target = Path(args[-1])
        (target / ".git").mkdir(parents=True)
        return subprocess.CompletedProcess(args, 0, "cloned", "")

    evaluator = OfficialPatchEvaluator(
        swe_python=Path("/usr/bin/python3"),
        multi_python=Path("/usr/bin/python3"),
        swe_harness_root=tmp_path,
        multi_harness_root=harness,
        pool_root=pool,
        output_root=tmp_path / "native",
        run_command=lambda args, _cwd, _timeout: subprocess.CompletedProcess(
            args, 1, "runner-out", "runner-err"
        ),
        repository_command=repository_command,
    )

    with pytest.raises(OfficialEvaluatorError, match="found=0"):
        evaluator(
            invocation,
            {
                "source_content_sha256": hashlib.sha256(
                    canonical_source.encode("utf-8")
                ).hexdigest()
            },
            receipt,
        )

    root = tmp_path / "native/g0-observe-task/original"
    assert (root / "stdout.log").read_text() == "runner-out"
    assert (root / "stderr.log").read_text() == "runner-err"
    failure = json.loads((root / "NATIVE-EVALUATOR-FAILURE.json").read_text())
    assert failure["status"] == "failed"
    assert failure["returncode"] == 1
    assert failure["reason"] == "expected one Multi-SWE native report; found=0"
