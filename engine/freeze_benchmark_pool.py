"""Freeze four real benchmark sources into a deterministic 300-task pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmark_adapters import (
    BenchmarkContractError,
    BenchmarkRegistry,
    StaticBenchmarkAdapter,
    TaskPool,
)
from benchmark_catalog import (
    PINNED_SOURCES,
    build_execution_command,
    normalize_multi_swe_task,
    normalize_swe_task,
    normalize_terminal_tasks,
)

USER_AGENT = "evolve-jlens-cluster-v2.1.0/benchmark-freezer"
EXPECTED_SOURCE_COUNTS = {
    "swe-bench-verified": 500,
    "swe-bench-multilingual": 300,
    "multi-swe-bench-flash": 300,
    "terminal-bench-2": 89,
}
PARTITION_QUOTAS = {
    adapter_id: {"search": 40, "promotion": 20, "final_sealed": 15}
    for adapter_id in PINNED_SOURCES
}
PREVIOUSLY_OPENED_INSTANCE_IDS = frozenset(
    {
        "django__django-14122",
        "django__django-14238",
        "scikit-learn__scikit-learn-14087",
        "sphinx-doc__sphinx-9367",
        "sympy__sympy-13551",
        "sympy__sympy-20590",
        "astropy__astropy-7336",
        "django__django-15987",
        "django__django-13741",
        "sympy__sympy-13480",
        "django__django-16429",
        "django__django-16612",
        "sphinx-doc__sphinx-9673",
        "pydata__xarray-6744",
        "matplotlib__matplotlib-25960",
        "django__django-12273",
    }
)


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def _fetch_bytes(url: str, *, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(_request(url), timeout=timeout) as response:
        return response.read()


def _fetch_json(url: str) -> dict[str, Any]:
    value = json.loads(_fetch_bytes(url))
    if not isinstance(value, dict):
        raise BenchmarkContractError(f"expected JSON object from {url}")
    return value


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(
        path,
        (
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8"),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hub_api_url(dataset_repo: str) -> str:
    return f"https://huggingface.co/api/datasets/{dataset_repo}"


def _assert_source_revision(adapter_id: str, metadata: dict[str, Any]) -> None:
    source = PINNED_SOURCES[adapter_id]
    if metadata.get("sha") != source.dataset_revision:
        raise BenchmarkContractError(
            f"source revision drift for {adapter_id}: "
            f"expected={source.dataset_revision} actual={metadata.get('sha')}"
        )


def _fetch_viewer_rows(
    adapter_id: str,
    *,
    input_dir: Path,
) -> list[dict[str, Any]]:
    source = PINNED_SOURCES[adapter_id]
    metadata = _fetch_json(_hub_api_url(source.dataset_repo))
    _assert_source_revision(adapter_id, metadata)
    source_dir = input_dir / adapter_id
    _atomic_json(source_dir / "hub-metadata.before.json", metadata)
    rows: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        query = urllib.parse.urlencode(
            {
                "dataset": source.dataset_repo,
                "config": "default",
                "split": source.split,
                "offset": offset,
                "length": 100,
            }
        )
        content = _fetch_bytes(f"https://datasets-server.huggingface.co/rows?{query}")
        _atomic_bytes(source_dir / "pages" / f"page-{offset:04d}.json", content)
        page = json.loads(content)
        page_rows = page.get("rows")
        if not isinstance(page_rows, list) or not page_rows:
            raise BenchmarkContractError(f"empty viewer page for {adapter_id}@{offset}")
        rows.extend(item["row"] for item in page_rows)
        total = int(page["num_rows_total"])
        offset += len(page_rows)
    after = _fetch_json(_hub_api_url(source.dataset_repo))
    _assert_source_revision(adapter_id, after)
    _atomic_json(source_dir / "hub-metadata.after.json", after)
    if len(rows) != total:
        raise BenchmarkContractError(f"viewer pagination mismatch for {adapter_id}")
    return rows


def _download_multi_swe(input_dir: Path) -> list[dict[str, Any]]:
    adapter_id = "multi-swe-bench-flash"
    source = PINNED_SOURCES[adapter_id]
    metadata = _fetch_json(_hub_api_url(source.dataset_repo))
    _assert_source_revision(adapter_id, metadata)
    source_dir = input_dir / adapter_id
    _atomic_json(source_dir / "hub-metadata.before.json", metadata)
    url = (
        f"https://huggingface.co/datasets/{source.dataset_repo}/resolve/"
        f"{source.dataset_revision}/multi_swe_bench_flash.jsonl"
    )
    target = source_dir / "multi_swe_bench_flash.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with (
        urllib.request.urlopen(_request(url), timeout=180) as response,
        temporary.open("wb") as output,
    ):
        shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(target)
    rows = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    after = _fetch_json(_hub_api_url(source.dataset_repo))
    _assert_source_revision(adapter_id, after)
    _atomic_json(source_dir / "hub-metadata.after.json", after)
    return rows


def _fetch_terminal_paths(input_dir: Path) -> list[str]:
    adapter_id = "terminal-bench-2"
    source = PINNED_SOURCES[adapter_id]
    metadata = _fetch_json(_hub_api_url(source.dataset_repo))
    _assert_source_revision(adapter_id, metadata)
    _atomic_json(input_dir / adapter_id / "hub-metadata.json", metadata)
    paths = [
        str(item["rfilename"])
        for item in metadata.get("siblings", [])
        if isinstance(item, dict) and item.get("rfilename")
    ]
    if not paths:
        raise BenchmarkContractError("Terminal-Bench file index is empty")
    return paths


def assemble_registry(
    *,
    verified_rows: list[dict[str, Any]],
    multilingual_rows: list[dict[str, Any]],
    multi_swe_rows: list[dict[str, Any]],
    terminal_paths: list[str],
) -> BenchmarkRegistry:
    """Build the offline registry after network inputs have been frozen."""

    registry = BenchmarkRegistry()
    rows_by_adapter = {
        "swe-bench-verified": verified_rows,
        "swe-bench-multilingual": multilingual_rows,
    }
    for adapter_id, rows in rows_by_adapter.items():
        source = PINNED_SOURCES[adapter_id]
        registry.register(
            StaticBenchmarkAdapter(
                adapter_id=adapter_id,
                revision=source.dataset_revision,
                executable=True,
                tasks=tuple(
                    normalize_swe_task(source, row, row_index=index)
                    for index, row in enumerate(rows)
                ),
            )
        )
    multi_source = PINNED_SOURCES["multi-swe-bench-flash"]
    registry.register(
        StaticBenchmarkAdapter(
            adapter_id=multi_source.adapter_id,
            revision=multi_source.dataset_revision,
            executable=True,
            tasks=tuple(
                normalize_multi_swe_task(multi_source, row, row_index=index)
                for index, row in enumerate(multi_swe_rows)
            ),
        )
    )
    terminal_source = PINNED_SOURCES["terminal-bench-2"]
    registry.register(
        StaticBenchmarkAdapter(
            adapter_id=terminal_source.adapter_id,
            revision=terminal_source.dataset_revision,
            executable=True,
            tasks=normalize_terminal_tasks(terminal_source, terminal_paths),
        )
    )
    return registry


def _input_manifest(input_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in input_dir.rglob("*") if item.is_file()):
        files.append(
            {
                "path": str(path.relative_to(input_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "schema_version": "1.0",
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def _write_harness_inputs_and_contracts(
    output_dir: Path,
    *,
    verified_rows: list[dict[str, Any]],
    multilingual_rows: list[dict[str, Any]],
    registry: BenchmarkRegistry,
) -> dict[str, Any]:
    harness_inputs = output_dir / "harness-inputs"
    rows_by_adapter = {
        "swe-bench-verified": verified_rows,
        "swe-bench-multilingual": multilingual_rows,
    }
    local_inputs: dict[str, dict[str, Any]] = {}
    for adapter_id, rows in rows_by_adapter.items():
        path = harness_inputs / f"{adapter_id}.jsonl"
        content = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        ).encode()
        _atomic_bytes(path, content)
        local_inputs[adapter_id] = {
            "path": str(path.relative_to(output_dir)),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }

    task_by_adapter = {
        adapter_id: next(
            task for task in registry.tasks() if task.benchmark_id == adapter_id
        )
        for adapter_id in PINNED_SOURCES
    }
    adapters: dict[str, Any] = {}
    for adapter_id, source in sorted(PINNED_SOURCES.items()):
        if source.harness_kind == "swebench":
            dataset_path = (
                f"/workspace/benchmark-pool/{local_inputs[adapter_id]['path']}"
            )
            staging = {
                "method": "copy_hashed_local_jsonl",
                **local_inputs[adapter_id],
                "runtime_path": dataset_path,
            }
        elif source.harness_kind == "multi_swe_bench":
            dataset_path = (
                "/workspace/benchmark-pool/inputs/multi-swe-bench-flash/"
                "multi_swe_bench_flash.jsonl"
            )
            staging = {
                "method": "copy_hashed_source_jsonl",
                "path": ("inputs/multi-swe-bench-flash/multi_swe_bench_flash.jsonl"),
                "runtime_path": dataset_path,
                "dataset_revision": source.dataset_revision,
            }
        else:
            dataset_path = (
                f"/workspace/eval-datasets/{adapter_id}@{source.dataset_revision}"
            )
            staging = {
                "method": "git_clone_checkout_with_lfs",
                "repo": f"https://huggingface.co/datasets/{source.dataset_repo}",
                "revision": source.dataset_revision,
                "runtime_path": dataset_path,
                "verify_checkout_before_task_open": True,
            }
        adapter = {
            "source": asdict(source),
            "harness_checkout": {
                "repo": f"https://github.com/{source.harness_repo}.git",
                "revision": source.harness_revision,
            },
            "dataset_staging": staging,
            "sample_command": list(
                build_execution_command(
                    source,
                    predictions_path="/workspace/predictions/frozen.jsonl",
                    run_id="round-000-sample",
                    instance_ids=(task_by_adapter[adapter_id].instance_id,),
                    dataset_path=dataset_path,
                    model="gpt-5.6-sol",
                    reasoning="low",
                    arm="baseline",
                    agent_program_sha256="0" * 64,
                    baseline_contract_sha256="1" * 64,
                )
            ),
        }
        if source.harness_kind == "multi_swe_bench":
            adapter["config_template"] = {
                "mode": "evaluation",
                "workdir": "/workspace/multi-swe/workdir",
                "patch_files": ["/workspace/predictions/frozen.jsonl"],
                "dataset_files": [dataset_path],
                "force_build": False,
                "output_dir": "/workspace/results",
                "specifics": [task_by_adapter[adapter_id].instance_id],
                "skips": [],
                "repo_dir": "/workspace/multi-swe/repos",
                "need_clone": True,
                "global_env": [],
                "clear_env": True,
                "stop_on_error": True,
                "max_workers": 1,
                "max_workers_build_image": 1,
                "max_workers_run_instance": 1,
                "log_dir": "/workspace/logs",
                "log_level": "INFO",
            }
        adapters[adapter_id] = adapter
    payload = {"schema_version": "1.0", "adapters": adapters}
    _atomic_json(output_dir / "ADAPTER_CONTRACTS.json", payload)
    return payload


def load_frozen_inputs(
    input_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    """Reload the immutable local snapshots without contacting the network."""

    def viewer_rows(adapter_id: str) -> list[dict[str, Any]]:
        rows = []
        for path in sorted((input_dir / adapter_id / "pages").glob("page-*.json")):
            page = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(item["row"] for item in page["rows"])
        return rows

    verified_rows = viewer_rows("swe-bench-verified")
    multilingual_rows = viewer_rows("swe-bench-multilingual")
    multi_swe_rows = [
        json.loads(line)
        for line in (
            input_dir / "multi-swe-bench-flash" / "multi_swe_bench_flash.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    terminal_metadata = json.loads(
        (input_dir / "terminal-bench-2" / "hub-metadata.json").read_text(
            encoding="utf-8"
        )
    )
    terminal_paths = [
        str(item["rfilename"])
        for item in terminal_metadata["siblings"]
        if item.get("rfilename")
    ]
    return verified_rows, multilingual_rows, multi_swe_rows, terminal_paths


def rebuild_pool_from_frozen_inputs(output_dir: Path) -> dict[str, Any]:
    """Rebuild only derived pool files from already hashed source snapshots."""

    output_dir = output_dir.resolve()
    inputs = load_frozen_inputs(output_dir / "inputs")
    registry = assemble_registry(
        verified_rows=inputs[0],
        multilingual_rows=inputs[1],
        multi_swe_rows=inputs[2],
        terminal_paths=inputs[3],
    )
    _write_harness_inputs_and_contracts(
        output_dir,
        verified_rows=inputs[0],
        multilingual_rows=inputs[1],
        registry=registry,
    )
    pool = TaskPool.build(
        registry=registry,
        seed_material="evolve-jlens-v2.1.0-continuous-ab",
        target_count=300,
        promotion_count=80,
        final_sealed_count=60,
        partition_quotas=PARTITION_QUOTAS,
        retired_instance_ids=PREVIOUSLY_OPENED_INSTANCE_IDS,
    )
    pool.save(output_dir / "TASK_POOL.json")
    actual_counts = {
        adapter_id: len(
            [task for task in registry.tasks() if task.benchmark_id == adapter_id]
        )
        for adapter_id in PINNED_SOURCES
    }
    partitions = Counter(record.assigned_partition for record in pool.records)
    adapters = Counter(record.benchmark_id for record in pool.records)
    summary = {
        "schema_version": "1.0",
        "status": "frozen",
        "source_task_count": sum(actual_counts.values()),
        "selected_task_count": len(pool.records),
        "source_counts": actual_counts,
        "selected_by_adapter": dict(sorted(adapters.items())),
        "selected_by_partition": dict(sorted(partitions.items())),
        "duplicate_count": len(pool.duplicates),
        "prior_opened_exclusion_count": len(pool.retired_exclusions),
        "all_tasks_unopened": all(
            record.state == "unopened" for record in pool.records
        ),
    }
    _atomic_json(output_dir / "POOL_SUMMARY.json", summary)
    return summary


def freeze(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    input_dir = output_dir / "inputs"
    verified_rows = _fetch_viewer_rows("swe-bench-verified", input_dir=input_dir)
    multilingual_rows = _fetch_viewer_rows(
        "swe-bench-multilingual", input_dir=input_dir
    )
    multi_swe_rows = _download_multi_swe(input_dir)
    terminal_paths = _fetch_terminal_paths(input_dir)
    actual_counts = {
        "swe-bench-verified": len(verified_rows),
        "swe-bench-multilingual": len(multilingual_rows),
        "multi-swe-bench-flash": len(multi_swe_rows),
        "terminal-bench-2": len(
            normalize_terminal_tasks(PINNED_SOURCES["terminal-bench-2"], terminal_paths)
        ),
    }
    if actual_counts != EXPECTED_SOURCE_COUNTS:
        raise BenchmarkContractError(
            f"frozen source counts changed: expected={EXPECTED_SOURCE_COUNTS} "
            f"actual={actual_counts}"
        )
    registry = assemble_registry(
        verified_rows=verified_rows,
        multilingual_rows=multilingual_rows,
        multi_swe_rows=multi_swe_rows,
        terminal_paths=terminal_paths,
    )
    pool = TaskPool.build(
        registry=registry,
        seed_material="evolve-jlens-v2.1.0-continuous-ab",
        target_count=300,
        promotion_count=80,
        final_sealed_count=60,
        partition_quotas=PARTITION_QUOTAS,
        retired_instance_ids=PREVIOUSLY_OPENED_INSTANCE_IDS,
    )
    catalog = {
        "schema_version": "1.0",
        "sources": {
            adapter_id: asdict(source)
            for adapter_id, source in sorted(PINNED_SOURCES.items())
        },
        "source_counts": actual_counts,
        "tasks": [task.to_dict() for task in registry.tasks()],
    }
    _write_harness_inputs_and_contracts(
        output_dir,
        verified_rows=verified_rows,
        multilingual_rows=multilingual_rows,
        registry=registry,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pool.save(output_dir / "TASK_POOL.json")
    _atomic_json(output_dir / "BENCHMARK_CATALOG.json", catalog)
    manifest = _input_manifest(input_dir)
    _atomic_json(output_dir / "SOURCE_MANIFEST.json", manifest)
    partitions = Counter(record.assigned_partition for record in pool.records)
    adapters = Counter(record.benchmark_id for record in pool.records)
    summary = {
        "schema_version": "1.0",
        "status": "frozen",
        "source_task_count": sum(actual_counts.values()),
        "selected_task_count": len(pool.records),
        "source_counts": actual_counts,
        "selected_by_adapter": dict(sorted(adapters.items())),
        "selected_by_partition": dict(sorted(partitions.items())),
        "duplicate_count": len(pool.duplicates),
        "prior_opened_exclusion_count": len(pool.retired_exclusions),
        "all_tasks_unopened": all(
            record.state == "unopened" for record in pool.records
        ),
    }
    _atomic_json(output_dir / "POOL_SUMMARY.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-frozen-inputs", action="store_true")
    args = parser.parse_args()
    result = (
        rebuild_pool_from_frozen_inputs(args.output_dir)
        if args.reuse_frozen_inputs
        else freeze(args.output_dir)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
