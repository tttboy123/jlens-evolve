"""CLI for the A-E diversity/emergence experiment suite.

Examples
--------
Deterministic (default, no model calls, fully reproducible):

    .venv/bin/python -m epistasis run \\
        --real paid,payout,refund --synthetic 12 --seeds 3 --budget 8 \\
        --out runs/epistasis-smoke

LLM mode (any OpenAI-compatible endpoint; model vendors plug in here):

    .venv/bin/python -m epistasis run \\
        --mode llm --model-config /path/to/model.json \\
        --real paid,payout --synthetic 0 --seeds 1 --budget 4 \\
        --out runs/epistasis-llm

model.json::

    {"provider": "openai", "base_url": "http://127.0.0.1:18080/v1",
     "model": "Qwen3.5-4B-mlx-4bit", "api_key_env": "OPENAI_API_KEY",
     "temperature": 0.7, "max_tokens": 1024}
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .experiments import (
    analyze_A,
    analyze_B,
    analyze_C,
    analyze_D,
    analyze_E,
    run_cell_plan,
    serialize_events,
)
from .model_transport import load_transport
from .report import default_meta, write_reports
from .tasks import build_task_matrix

EXPERIMENTS = ("A", "B", "C", "D", "E")


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_seeds(value: str) -> list[int]:
    if "," in value:
        return [int(item) for item in _parse_list(value)]
    return list(range(int(value)))


def _load_existing_cells(out_dir: Path) -> set[str]:
    cells_dir = out_dir / "cells"
    if not cells_dir.is_dir():
        return set()
    return {path.stem for path in cells_dir.glob("*.json")}


def cmd_run(args: argparse.Namespace) -> int:
    experiments = tuple(
        item for item in _parse_list(args.experiments) if item in EXPERIMENTS
    )
    if not experiments:
        raise SystemExit("--experiments must select at least one of A,B,C,D,E")
    mode = args.mode
    if mode not in {"deterministic", "llm"}:
        raise SystemExit("--mode must be deterministic or llm")
    llm_style = getattr(args, "llm_style", "scaffold")
    if mode == "llm" and llm_style not in {"free", "scaffold"}:
        raise SystemExit("--llm-style must be free or scaffold")
    model = load_transport(args.model_config) if mode == "llm" else None
    seeds = _parse_seeds(args.seeds)
    budget = int(args.budget)
    if budget < 1:
        raise SystemExit("--budget must be >= 1")

    tasks = build_task_matrix(
        real_tasks=tuple(_parse_list(args.real)),
        synthetic=int(args.synthetic),
        synthetic_seed=int(args.synthetic_seed),
        coupling=int(args.coupling),
        uncovered_extra=int(args.uncovered_extra),
    )
    real_count = len(_parse_list(args.real))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    skip_existing = set()
    results: dict[str, Any] = {}
    if args.resume:
        skip_existing, results = _load_cells(out_dir)

    results = dict(results)
    futures: list[Any] = []
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        for task in tasks:
            for seed in seeds:
                futures.append(
                    executor.submit(
                        run_cell_plan,
                        task,
                        seed,
                        budget,
                        mode=mode,
                        model=model,
                        skip_existing=skip_existing,
                        llm_style=llm_style,
                    )
                )
        for index, future in enumerate(as_completed(futures), start=1):
            results.update(future.result())
            if args.verbose:
                print(
                    f"[{index}/{len(futures)}] cells done: {len(results)}",
                    file=sys.stderr,
                )

    # Persist every cell immediately (idempotent resume).
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    for result in results.values():
        cell_path = cells_dir / _cell_filename(result)
        cell_path.write_text(
            json.dumps(
                result.to_full_dict(), indent=2, sort_keys=True, ensure_ascii=False
            )
            + "\n",
            encoding="utf-8",
        )

    reports: dict[str, Any] = {}
    if "B" in experiments:
        reports["B"] = analyze_B(tasks, results, seeds, budget=budget, mode=mode)
    if "A" in experiments:
        reports["A"] = analyze_A(tasks, results, seeds, budget=budget, mode=mode)
    if "C" in experiments:
        reports["C"] = analyze_C(tasks, results, seeds, budget=budget, mode=mode)
    if "D" in experiments:
        reports["D"] = analyze_D(tasks, results, seeds, budget=budget, mode=mode)
    if "E" in experiments:
        reports["E"] = analyze_E(tasks, results, seeds, budget=budget, mode=mode)

    cells = [result.to_dict() for result in results.values()]
    events = serialize_events(results.values())
    command = " ".join(["python", "-m", "epistasis", *sys.argv[1:]])
    meta = default_meta(
        command=command,
        mode=mode,
        budget=budget,
        seeds=seeds,
        task_count=len(tasks),
        real_task_count=real_count,
        synthetic_task_count=max(0, len(tasks) - real_count),
    )
    meta["attempt_count"] = sum(len(result.events) for result in results.values())
    meta["experiments"] = list(experiments)
    meta["model"] = model.model_id if model is not None else None
    meta["llm_style"] = llm_style if mode == "llm" else None
    write_reports(out_dir, meta=meta, reports=reports, cells=cells, events=events)
    print(f"wrote experiment reports to {out_dir / 'EXPERIMENTS.json'}")
    print(f"summary:  {out_dir / 'SUMMARY.md'}")
    return 0


def _cell_filename(result: Any) -> str:
    return (
        f"{result.task_id}-s{result.seed}-{result.mode}-"
        f"{'-'.join(result.operator_ids) or 'empty'}-b{result.budget}.json"
    )


def _load_cells(out_dir: Path) -> tuple[set[str], dict[str, Any]]:
    """Load persisted full cells for resume; returns (skipped_keys, results)."""
    from .search import CellResult, cell_key

    cells_dir = out_dir / "cells"
    results: dict[str, Any] = {}
    if not cells_dir.is_dir():
        return set(), results
    for path in cells_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result = CellResult.from_full_dict(payload)
        except Exception as exc:  # noqa: BLE001 - skip unreadable cell
            print(f"warning: skipping unreadable cell {path}: {exc}", file=sys.stderr)
            continue
        key = cell_key(
            result.task_id, result.seed, result.operator_ids, result.mode, result.budget
        )
        results[key] = result
    return set(results), results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="epistasis",
        description="A-E diversity/emergence experiments for the RSI operator set.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run experiment cells and write reports")
    run.add_argument("--experiments", default="A,B,C,D,E")
    run.add_argument("--real", default="paid,payout,refund")
    run.add_argument("--synthetic", type=int, default=0)
    run.add_argument("--synthetic-seed", type=int, default=0)
    run.add_argument("--coupling", type=int, default=1)
    run.add_argument("--uncovered-extra", type=int, default=1)
    run.add_argument("--seeds", default="3")
    run.add_argument("--budget", type=int, default=8)
    run.add_argument(
        "--mode", default="deterministic", choices=["deterministic", "llm"]
    )
    run.add_argument(
        "--llm-style",
        default="scaffold",
        choices=["free", "scaffold"],
        help="llm mode only: free = model writes the full program (control); "
        "scaffold = model repairs a deterministic operator scaffold (treatment).",
    )
    run.add_argument("--model-config", default=None)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--out", required=True)
    run.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
