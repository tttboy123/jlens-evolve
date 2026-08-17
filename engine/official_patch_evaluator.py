"""Pinned one-arm official evaluator adapters for three patch benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

_REPOSITORY_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class OfficialEvaluatorError(RuntimeError):
    """Raised when an official harness does not produce exact native evidence."""

    error_code = "evaluator_error"

    def __init__(self, message: str, *, evidence_path: Path | None = None) -> None:
        super().__init__(message)
        self.evidence_path = evidence_path


class OfficialEvaluatorTimeout(OfficialEvaluatorError):
    """Raised after a native timeout has been frozen as replayable evidence."""

    error_code = "evaluator_timeout"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not rendered:
        raise OfficialEvaluatorError("native evaluator identity is empty")
    return rendered[:120]


def _default_run(
    args: tuple[str, ...], cwd: Path, timeout: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _git_revision(root: Path) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    revision = completed.stdout.strip()
    if completed.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise OfficialEvaluatorError(
            f"official harness Git revision is unavailable: {root}"
        )
    return revision


def _probe_harness_module(
    *, python: Path, root: Path, module_name: str, entrypoint: Path
) -> None:
    completed = subprocess.run(
        (
            str(python),
            "-c",
            (
                "import importlib; from pathlib import Path; "
                f"module = importlib.import_module({module_name!r}); "
                "print(Path(module.__file__).resolve())"
            ),
        ),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    imported = completed.stdout.strip().splitlines()
    if (
        completed.returncode != 0
        or not imported
        or Path(imported[-1]).resolve() != entrypoint.resolve()
    ):
        raise OfficialEvaluatorError(
            f"official harness module import failed: {module_name}"
        )


def freeze_official_harness_runtime(
    *,
    swe_python: Path,
    multi_python: Path,
    swe_harness_root: Path,
    multi_harness_root: Path,
    output_root: Path,
    native_assets_path: Path | None = None,
) -> Path:
    """Fail fast on missing/drifted harness source and freeze its identity.

    Editable installations can outlive their source checkout.  Checking only the
    Python launcher and checkout directory therefore permits a whole native batch
    to start even though ``python -m`` can no longer import the harness.  This
    preflight binds the actual source entry points and Git revisions before any
    benchmark cell is evaluated.
    """

    roots = {
        "swe": swe_harness_root.resolve(),
        "multi_swe": multi_harness_root.resolve(),
    }
    pythons = {
        "swe": swe_python.absolute(),
        "multi_swe": multi_python.absolute(),
    }
    entrypoints = {
        "swe": roots["swe"] / "swebench/harness/run_evaluation.py",
        "multi_swe": roots["multi_swe"]
        / "multi_swe_bench/harness/run_evaluation.py",
    }
    modules = {
        "swe": "swebench.harness.run_evaluation",
        "multi_swe": "multi_swe_bench.harness.run_evaluation",
    }
    for label, python in pythons.items():
        if not python.is_file():
            raise OfficialEvaluatorError(
                f"{label} official harness Python executable is missing"
            )
    for label, entrypoint in entrypoints.items():
        if not entrypoint.is_file():
            raise OfficialEvaluatorError(
                f"{label} official harness source entry point is missing"
            )
        _probe_harness_module(
            python=pythons[label],
            root=roots[label],
            module_name=modules[label],
            entrypoint=entrypoint,
        )

    expected: dict[str, dict[str, Any]] = {}
    native_assets_sha256: str | None = None
    if native_assets_path is not None:
        assets_path = native_assets_path.resolve()
        if not assets_path.is_file():
            raise OfficialEvaluatorError("native assets manifest is missing")
        try:
            assets = json.loads(assets_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise OfficialEvaluatorError("native assets manifest is unreadable") from exc
        swe_assets = assets.get("swe_harness")
        multi_assets = assets.get("multi_swe_harness")
        if not isinstance(swe_assets, dict) or not isinstance(multi_assets, dict):
            raise OfficialEvaluatorError("native assets harness pins are missing")
        expected = {"swe": swe_assets, "multi_swe": multi_assets}
        native_assets_sha256 = _sha256_file(assets_path)

    runtimes: dict[str, dict[str, Any]] = {}
    for label in ("swe", "multi_swe"):
        revision = _git_revision(roots[label])
        expected_revision = expected.get(label, {}).get("revision")
        if expected_revision is not None and revision != expected_revision:
            raise OfficialEvaluatorError(
                f"{label} official harness revision does not match native assets"
            )
        runtimes[label] = {
            "root": str(roots[label]),
            "python": str(pythons[label]),
            "revision": revision,
            "entrypoint": str(entrypoints[label].relative_to(roots[label])),
            "entrypoint_sha256": _sha256_file(entrypoints[label]),
            "module_import_verified": True,
        }

    expected_setup_sha = expected.get("multi_swe", {}).get("setup_py_sha256")
    multi_setup = roots["multi_swe"] / "setup.py"
    if expected_setup_sha is not None:
        if not multi_setup.is_file() or _sha256_file(multi_setup) != expected_setup_sha:
            raise OfficialEvaluatorError(
                "Multi-SWE harness setup.py does not match native assets"
            )
        runtimes["multi_swe"]["setup_py_sha256"] = expected_setup_sha

    payload = {
        "schema_version": 1,
        "status": "ready",
        "native_assets_sha256": native_assets_sha256,
        "official_patch_evaluator_sha256": _sha256_file(Path(__file__).resolve()),
        "runtimes": runtimes,
    }
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_path = output_root / "HARNESS-RUNTIME-RECEIPT.json"
    if receipt_path.is_file():
        try:
            frozen = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise OfficialEvaluatorError(
                "official harness runtime receipt is unreadable"
            ) from exc
        if frozen != payload:
            raise OfficialEvaluatorError("official harness runtime receipt drifted")
        return receipt_path
    _write_json(receipt_path, payload)
    return receipt_path


class OfficialPatchEvaluator:
    """Run the frozen official harness once for one already-frozen patch."""

    def __init__(
        self,
        *,
        swe_python: Path,
        multi_python: Path,
        swe_harness_root: Path,
        multi_harness_root: Path,
        pool_root: Path,
        output_root: Path,
        run_command: Callable[
            [tuple[str, ...], Path, int], subprocess.CompletedProcess[str]
        ] = _default_run,
        repository_command: Callable[
            [tuple[str, ...], Path, int], subprocess.CompletedProcess[str]
        ] = _default_run,
        timeout_seconds: int = 7200,
    ) -> None:
        # A venv's bin/python is intentionally a symlink. Keep that launcher
        # path so Python discovers the venv rather than the system site-packages.
        self.swe_python = swe_python.absolute()
        self.multi_python = multi_python.absolute()
        self.swe_harness_root = swe_harness_root.resolve()
        self.multi_harness_root = multi_harness_root.resolve()
        self.pool_root = pool_root.resolve()
        self.output_root = output_root.resolve()
        self.run_command = run_command
        self.repository_command = repository_command
        self.timeout_seconds = timeout_seconds
        for path, label in (
            (self.swe_python, "SWE Python"),
            (self.multi_python, "Multi-SWE Python"),
        ):
            if not path.is_file():
                raise OfficialEvaluatorError(f"{label} executable is missing")
        for path, label in (
            (self.swe_harness_root, "SWE harness"),
            (self.multi_harness_root, "Multi-SWE harness"),
            (self.pool_root, "benchmark pool"),
        ):
            if not path.is_dir():
                raise OfficialEvaluatorError(f"{label} root is missing")
        if timeout_seconds < 1:
            raise OfficialEvaluatorError("native evaluator timeout is invalid")

    @staticmethod
    def _prediction(receipt: dict[str, Any]) -> Path:
        prediction = receipt.get("prediction")
        path = prediction.get("path") if isinstance(prediction, dict) else None
        if not isinstance(path, str):
            raise OfficialEvaluatorError("Agent receipt prediction path is missing")
        result = Path(path).resolve()
        if not result.is_file():
            raise OfficialEvaluatorError("frozen Agent prediction is missing")
        return result

    def _root(self, invocation: Any) -> Path:
        return self.output_root / _safe(invocation.round_id) / _safe(invocation.arm)

    def _resume(self, root: Path) -> Path | None:
        receipt_path = root / "NATIVE-EVALUATOR-RECEIPT.json"
        report_path = root / "native-report.json"
        if not receipt_path.is_file() and not report_path.exists():
            return None
        if not receipt_path.is_file() or not report_path.is_file():
            raise OfficialEvaluatorError("native evaluator evidence is incomplete")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "completed" or receipt.get("native_report", {}).get(
            "sha256"
        ) != _sha256_file(report_path):
            raise OfficialEvaluatorError("native evaluator evidence was tampered")
        return report_path

    def _resume_failure(self, root: Path, prediction_input: Path) -> None:
        failure_path = root / "NATIVE-EVALUATOR-FAILURE.json"
        if not failure_path.is_file():
            return
        try:
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise OfficialEvaluatorError(
                "native evaluator failure evidence is unreadable"
            ) from exc
        expected = failure.get("prediction_input", {}).get("sha256")
        if (
            failure.get("status") != "failed"
            or expected != _sha256_file(prediction_input)
        ):
            raise OfficialEvaluatorError("native evaluator failure evidence was tampered")
        reason = failure.get("reason")
        if not isinstance(reason, str):
            raise OfficialEvaluatorError("native evaluator failure reason is missing")
        error_type = (
            OfficialEvaluatorTimeout
            if failure.get("error_code") == "evaluator_timeout"
            else OfficialEvaluatorError
        )
        raise error_type(reason, evidence_path=failure_path)

    def _complete(
        self,
        *,
        root: Path,
        source_report: Path,
        args: tuple[str, ...],
        completed: subprocess.CompletedProcess[str],
        prediction_input: Path,
    ) -> Path:
        self._persist_process(root, completed)
        if completed.returncode != 0 or not source_report.is_file():
            reason = (
                f"official evaluator failed with return code {completed.returncode}"
            )
            failure_path = self._persist_failure(
                root=root,
                args=args,
                completed=completed,
                prediction_input=prediction_input,
                reason=reason,
            )
            raise OfficialEvaluatorError(reason, evidence_path=failure_path)
        report_path = root / "native-report.json"
        shutil.copyfile(source_report, report_path)
        _write_json(
            root / "NATIVE-EVALUATOR-RECEIPT.json",
            {
                "schema_version": 1,
                "status": "completed",
                "command": list(args),
                "returncode": completed.returncode,
                "prediction_input": {
                    "path": prediction_input.name,
                    "sha256": _sha256_file(prediction_input),
                },
                "native_report": {
                    "path": report_path.name,
                    "sha256": _sha256_file(report_path),
                },
            },
        )
        return report_path

    @staticmethod
    def _persist_process(
        root: Path, completed: subprocess.CompletedProcess[str]
    ) -> None:
        (root / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (root / "stderr.log").write_text(completed.stderr, encoding="utf-8")

    def _persist_failure(
        self,
        *,
        root: Path,
        args: tuple[str, ...],
        completed: subprocess.CompletedProcess[str],
        prediction_input: Path,
        reason: str,
        error_code: str = "evaluator_error",
        timeout_seconds: int | None = None,
    ) -> Path:
        self._persist_process(root, completed)
        diagnostic_artifacts = {
            path.name: _sha256_file(path)
            for path in (root / "run-instance.log", root / "build-image.log")
            if path.is_file()
        }
        payload = {
            "schema_version": 1,
            "status": "failed",
            "error_code": error_code,
            "reason": reason,
            "command": list(args),
            "returncode": completed.returncode,
            "prediction_input": {
                "path": prediction_input.name,
                "sha256": _sha256_file(prediction_input),
            },
            "stdout_sha256": _sha256_file(root / "stdout.log"),
            "stderr_sha256": _sha256_file(root / "stderr.log"),
            "diagnostic_artifacts": diagnostic_artifacts,
        }
        if timeout_seconds is not None:
            payload["timeout_seconds"] = timeout_seconds
        failure_path = root / "NATIVE-EVALUATOR-FAILURE.json"
        _write_json(failure_path, payload)
        return failure_path

    def _persist_swe_diagnostics(
        self, *, root: Path, source_report: Path, instance_id: str
    ) -> None:
        run_log = source_report.parent / "run_instance.log"
        if run_log.is_file():
            shutil.copyfile(run_log, root / "run-instance.log")
        build_logs = sorted(
            (
                self.swe_harness_root
                / "logs/build_images/instances"
            ).glob(f"*{_safe(instance_id)}*/build_image.log"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if build_logs:
            shutil.copyfile(build_logs[0], root / "build-image.log")

    def _persist_timeout(
        self,
        *,
        root: Path,
        args: tuple[str, ...],
        error: subprocess.TimeoutExpired,
        prediction_input: Path,
    ) -> Path:
        def rendered(value: str | bytes | None) -> str:
            if value is None:
                return ""
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return value

        completed = subprocess.CompletedProcess(
            args=args,
            returncode=-1,
            stdout=rendered(error.stdout),
            stderr=rendered(error.stderr),
        )
        reason = f"official evaluator timed out after {error.timeout} seconds"
        return self._persist_failure(
            root=root,
            args=args,
            completed=completed,
            prediction_input=prediction_input,
            reason=reason,
            error_code="evaluator_timeout",
            timeout_seconds=int(error.timeout),
        )

    def _reject_swe_infrastructure_report(
        self,
        *,
        root: Path,
        source_report: Path,
        instance_id: str,
        args: tuple[str, ...],
        completed: subprocess.CompletedProcess[str],
        prediction_input: Path,
    ) -> None:
        if completed.returncode != 0 or not source_report.is_file():
            return
        try:
            report = json.loads(source_report.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return
        error_ids = report.get("error_ids") if isinstance(report, dict) else None
        if (
            not isinstance(report, dict)
            or report.get("schema_version") != 2
            or not isinstance(error_ids, list)
        ):
            return
        if instance_id not in error_ids:
            return
        reason = "official SWE evaluator returned an infrastructure error"
        shutil.copyfile(source_report, root / "native-invalid-report.json")
        self._persist_swe_diagnostics(
            root=root,
            source_report=source_report,
            instance_id=instance_id,
        )
        failure_path = self._persist_failure(
            root=root,
            args=args,
            completed=completed,
            prediction_input=prediction_input,
            reason=reason,
        )
        raise OfficialEvaluatorError(reason, evidence_path=failure_path)

    def _reject_swe_failed_build(
        self,
        *,
        root: Path,
        source_report: Path,
        instance_id: str,
        test_output: Path,
        args: tuple[str, ...],
        completed: subprocess.CompletedProcess[str],
        prediction_input: Path,
    ) -> None:
        """Reject reports produced by stale tests after an ignored build failure."""

        if completed.returncode != 0 or not source_report.is_file():
            return
        if not test_output.is_file():
            return
        try:
            output = test_output.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        shutil.copyfile(test_output, root / "test-output.log")
        build_output = output.split(">>>>> Start Test Output", 1)[0]
        failure_patterns = (
            re.compile(r"(?m)\berror:\s+"),
            re.compile(r"(?m)^(?:g?make|ninja)(?:\[[0-9]+\])?: .*\*\*\*"),
            re.compile(r"(?m)^FAILED:\s+"),
        )
        if not any(pattern.search(build_output) for pattern in failure_patterns):
            return
        reason = "official SWE evaluator build failed before tests"
        shutil.copyfile(source_report, root / "native-invalid-report.json")
        self._persist_swe_diagnostics(
            root=root,
            source_report=source_report,
            instance_id=instance_id,
        )
        failure_path = self._persist_failure(
            root=root,
            args=args,
            completed=completed,
            prediction_input=prediction_input,
            reason=reason,
        )
        raise OfficialEvaluatorError(reason, evidence_path=failure_path)

    def _evaluate_swe(self, invocation: Any, prediction: Path, root: Path) -> Path:
        benchmark = invocation.benchmark_id
        dataset = self.pool_root / f"harness-inputs/{benchmark}.jsonl"
        if not dataset.is_file():
            raise OfficialEvaluatorError("frozen SWE dataset input is missing")
        model_name = _safe(
            f"evolve-{invocation.arm}-{invocation.agent_program_sha256[:12]}"
        )
        row = {
            "instance_id": invocation.instance_id,
            "model_name_or_path": model_name,
            "model_patch": prediction.read_text(encoding="utf-8"),
        }
        prediction_input = root / "prediction.jsonl"
        prediction_input.write_text(_canonical_json(row) + "\n", encoding="utf-8")
        # The official harness persists results under run_id and silently skips
        # an instance when that identifier already exists. Append-only retry
        # roots therefore need distinct harness namespaces; otherwise a clean
        # retry can replay a stale infrastructure failure from an earlier root.
        evidence_namespace = hashlib.sha256(
            str(root.resolve()).encode("utf-8")
        ).hexdigest()[:12]
        run_id = _safe(
            f"{invocation.round_id}-{invocation.arm}-{evidence_namespace}"
        )
        args = (
            str(self.swe_python),
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            str(dataset),
            "--split",
            "test",
            "--predictions_path",
            str(prediction_input),
            "--max_workers",
            "1",
            "--run_id",
            run_id,
            "--instance_ids",
            invocation.instance_id,
        )
        # Prebuilt instance images are not complete for either frozen SWE
        # family (Verified can also return registry 404s). Use the official
        # harness's local-image mode for both families so evaluator
        # availability cannot vary by registry coverage. Preserve the clean
        # instance image so matched arms do not rebuild it; patches are
        # applied only to disposable containers created from that image.
        args += ("--namespace", "none", "--cache_level", "instance")
        self._resume_failure(root, prediction_input)
        try:
            completed = self.run_command(
                args, self.swe_harness_root, self.timeout_seconds
            )
        except subprocess.TimeoutExpired as exc:
            failure_path = self._persist_timeout(
                root=root,
                args=args,
                error=exc,
                prediction_input=prediction_input,
            )
            raise OfficialEvaluatorTimeout(
                f"official evaluator timed out after {exc.timeout} seconds",
                evidence_path=failure_path,
            ) from exc
        instance_report = (
            self.swe_harness_root
            / "logs/run_evaluation"
            / run_id
            / model_name
            / invocation.instance_id
            / "report.json"
        )
        aggregate_report = self.swe_harness_root / f"{model_name}.{run_id}.json"
        source_report = (
            instance_report if instance_report.is_file() else aggregate_report
        )
        self._reject_swe_infrastructure_report(
            root=root,
            source_report=source_report,
            instance_id=invocation.instance_id,
            args=args,
            completed=completed,
            prediction_input=prediction_input,
        )
        self._reject_swe_failed_build(
            root=root,
            source_report=source_report,
            instance_id=invocation.instance_id,
            test_output=instance_report.parent / "test_output.txt",
            args=args,
            completed=completed,
            prediction_input=prediction_input,
        )
        return self._complete(
            root=root,
            source_report=source_report,
            args=args,
            completed=completed,
            prediction_input=prediction_input,
        )

    def _multi_row(self, instance_id: str) -> tuple[dict[str, Any], Path]:
        dataset = (
            self.pool_root / "inputs/multi-swe-bench-flash/multi_swe_bench_flash.jsonl"
        )
        if not dataset.is_file():
            raise OfficialEvaluatorError("frozen Multi-SWE dataset input is missing")
        matches = []
        for line in dataset.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("instance_id") == instance_id:
                    matches.append(row)
        if len(matches) != 1:
            raise OfficialEvaluatorError("Multi-SWE instance identity is ambiguous")
        return matches[0], dataset

    @staticmethod
    def _benchmark_proxy() -> str | None:
        proxy = os.environ.get("EVOLVE_BENCHMARK_HTTPS_PROXY")
        if proxy is None:
            return None
        if re.fullmatch(r"http://127\.0\.0\.1:[1-9][0-9]{3,4}", proxy) is None:
            raise OfficialEvaluatorError(
                "benchmark repository proxy must be loopback HTTP"
            )
        if int(proxy.rsplit(":", 1)[1]) > 65535:
            raise OfficialEvaluatorError("benchmark repository proxy port is invalid")
        return proxy

    def _ensure_multi_repository(
        self, *, row: dict[str, Any], repo_dir: Path, root: Path
    ) -> None:
        org = row.get("org")
        repo = row.get("repo")
        if (
            not isinstance(org, str)
            or _REPOSITORY_COMPONENT.fullmatch(org) is None
            or not isinstance(repo, str)
            or _REPOSITORY_COMPONENT.fullmatch(repo) is None
        ):
            raise OfficialEvaluatorError("Multi-SWE repository identity is unsafe")
        target = repo_dir / org / repo
        receipt_path = root / "REPOSITORY-PREFETCH-RECEIPT.json"
        if (target / ".git").is_dir():
            if not receipt_path.exists():
                _write_json(
                    receipt_path,
                    {
                        "schema_version": 1,
                        "status": "reused",
                        "repository": f"{org}/{repo}",
                        "path": str(target),
                    },
                )
            return
        if target.exists():
            raise OfficialEvaluatorError("Multi-SWE repository cache is incomplete")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Multi-SWE repository prefetch is a GitHub-only clone. Direct GitHub
        # egress from the temporary ap-guangzhou instance is intermittent, so
        # the verified loopback CONNECT tunnel is tried first when configured,
        # then direct egress. The target is verified to contain the pinned base
        # commit before the prefetch is accepted. See
        # EXECUTION-PROTOCOL-AMENDMENT-016.
        proxy = self._benchmark_proxy()
        strategies: list[tuple[str, list[str]]] = []
        if proxy is not None:
            strategies.append(("loopback-proxy", ["-c", f"http.proxy={proxy}"]))
        strategies.append(("direct", []))
        failures: list[dict[str, Any]] = []
        for label, extra in strategies:
            args = [
                "git",
                "-c",
                "http.version=HTTP/1.1",
                *extra,
                "clone",
                f"https://github.com/{org}/{repo}.git",
                str(target),
            ]
            completed = self.repository_command(
                tuple(args), target.parent, min(self.timeout_seconds, 1800)
            )
            (root / f"repository-prefetch-{label}.stdout.log").write_text(
                completed.stdout, encoding="utf-8"
            )
            (root / f"repository-prefetch-{label}.stderr.log").write_text(
                completed.stderr, encoding="utf-8"
            )
            base = row.get("base")
            base_sha = base.get("sha") if isinstance(base, dict) else None
            if (
                completed.returncode == 0
                and (target / ".git").is_dir()
                and (
                    base_sha is None
                    or subprocess.run(
                        ["git", "cat-file", "-e", f"{base_sha}^{{commit}}"],
                        cwd=target,
                        capture_output=True,
                        check=False,
                    ).returncode
                    == 0
                )
            ):
                payload = {
                    "schema_version": 1,
                    "status": "completed",
                    "strategy": label,
                    "repository": f"{org}/{repo}",
                    "base_commit_verified": base_sha,
                    "path": str(target),
                    "command": list(args),
                    "returncode": completed.returncode,
                    "stdout_sha256": _sha256_file(
                        root / f"repository-prefetch-{label}.stdout.log"
                    ),
                    "stderr_sha256": _sha256_file(
                        root / f"repository-prefetch-{label}.stderr.log"
                    ),
                }
                _write_json(receipt_path, payload)
                return
            failures.append(
                {
                    "strategy": label,
                    "returncode": completed.returncode,
                    "target_exists": target.exists(),
                    "stdout_sha256": _sha256_file(
                        root / f"repository-prefetch-{label}.stdout.log"
                    ),
                    "stderr_sha256": _sha256_file(
                        root / f"repository-prefetch-{label}.stderr.log"
                    ),
                }
            )
            if target.exists():
                shutil.rmtree(target)
            (root / "repository-prefetch.stdout.log").write_text(
                completed.stdout, encoding="utf-8"
            )
            (root / "repository-prefetch.stderr.log").write_text(
                completed.stderr, encoding="utf-8"
            )
        payload = {
            "schema_version": 1,
            "status": "failed",
            "repository": f"{org}/{repo}",
            "path": str(target),
            "attempts": failures,
            "stdout_sha256": _sha256_file(root / "repository-prefetch.stdout.log"),
            "stderr_sha256": _sha256_file(root / "repository-prefetch.stderr.log"),
        }
        _write_json(root / "REPOSITORY-PREFETCH-FAILURE.json", payload)
        raise OfficialEvaluatorError("Multi-SWE repository prefetch failed")

    def _evaluate_multi(
        self,
        invocation: Any,
        materialized: dict[str, Any],
        prediction: Path,
        root: Path,
    ) -> Path:
        row, dataset = self._multi_row(invocation.instance_id)
        if (
            materialized.get("source_content_sha256")
            != hashlib.sha256(_canonical_json(row).encode()).hexdigest()
        ):
            raise OfficialEvaluatorError("Multi-SWE source row hash mismatch")
        identity = {
            "org": row["org"],
            "repo": row["repo"],
            "number": int(row["number"]),
        }
        patch = {**identity, "fix_patch": prediction.read_text(encoding="utf-8")}
        prediction_input = root / "prediction.jsonl"
        prediction_input.write_text(_canonical_json(patch) + "\n", encoding="utf-8")
        workdir = root / "workdir"
        output_dir = root / "output"
        log_dir = root / "logs"
        repo_dir = self.output_root / "multi-repos"
        for path in (workdir, output_dir, log_dir, repo_dir):
            path.mkdir(parents=True, exist_ok=True)
        self._ensure_multi_repository(row=row, repo_dir=repo_dir, root=root)
        config = {
            "mode": "evaluation",
            "workdir": str(workdir),
            "patch_files": [str(prediction_input)],
            "dataset_files": [str(dataset)],
            "force_build": False,
            "output_dir": str(output_dir),
            "specifics": [
                f"{identity['org']}/{identity['repo']}:pr-{identity['number']}"
            ],
            "skips": [],
            "repo_dir": str(repo_dir),
            "need_clone": True,
            "global_env": [],
            "clear_env": True,
            "stop_on_error": True,
            "max_workers": 1,
            "max_workers_build_image": 1,
            "max_workers_run_instance": 1,
            "log_dir": str(log_dir),
            "log_level": "INFO",
            "log_to_console": True,
            "human_mode": True,
        }
        config_path = root / "config.json"
        _write_json(config_path, config)
        args = (
            str(self.multi_python),
            "-m",
            "multi_swe_bench.harness.run_evaluation",
            "--config",
            str(config_path),
        )
        self._resume_failure(root, prediction_input)
        try:
            completed = self.run_command(
                args, self.multi_harness_root, self.timeout_seconds
            )
        except subprocess.TimeoutExpired as exc:
            failure_path = self._persist_timeout(
                root=root,
                args=args,
                error=exc,
                prediction_input=prediction_input,
            )
            raise OfficialEvaluatorTimeout(
                f"official evaluator timed out after {exc.timeout} seconds",
                evidence_path=failure_path,
            ) from exc
        self._persist_process(root, completed)
        reports = []
        for candidate in workdir.rglob("report.json"):
            try:
                value = json.loads(candidate.read_text(encoding="utf-8"))
            except (UnicodeError, json.JSONDecodeError):
                continue
            if all(value.get(key) == expected for key, expected in identity.items()):
                reports.append(candidate)
        if len(reports) != 1:
            reason = f"expected one Multi-SWE native report; found={len(reports)}"
            failure_path = self._persist_failure(
                root=root,
                args=args,
                completed=completed,
                prediction_input=prediction_input,
                reason=reason,
            )
            raise OfficialEvaluatorError(reason, evidence_path=failure_path)
        return self._complete(
            root=root,
            source_report=reports[0],
            args=args,
            completed=completed,
            prediction_input=prediction_input,
        )

    def __call__(
        self, invocation: Any, materialized: dict[str, Any], receipt: dict[str, Any]
    ) -> Path:
        root = self._root(invocation)
        root.mkdir(parents=True, exist_ok=True)
        resumed = self._resume(root)
        if resumed is not None:
            return resumed
        prediction = self._prediction(receipt)
        if invocation.benchmark_id in {
            "swe-bench-verified",
            "swe-bench-multilingual",
        }:
            return self._evaluate_swe(invocation, prediction, root)
        if invocation.benchmark_id == "multi-swe-bench-flash":
            return self._evaluate_multi(invocation, materialized, prediction, root)
        raise OfficialEvaluatorError(
            "official patch evaluator benchmark is unsupported"
        )
