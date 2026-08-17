#!/usr/bin/env python3
"""Offline, dev-only localization recall oracle.

Uses golden patch locations ONLY as a measurement oracle; it is never shown
to the Student. Measures how often the harvested target file/symbol contains
the real fix location, grouped by localization policy. With ``--source-dir``
it additionally measures top-N symbol candidate recall from the new
``qualified_symbol_candidates`` API (still dev-only, never prompt-visible).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from skill_evolution_loop.symbol_rewrite import qualified_symbol_candidates

ROOT = Path("artifacts/v2.1.0/v2.1.0-continuous-ab/configs/benchmark-pool/harness-inputs/swe-bench-verified.jsonl")
RUNS = Path("artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop")


def golden_map() -> dict[str, dict]:
    out = {}
    with ROOT.open() as fh:
        for line in fh:
            d = json.loads(line)
            patch = d.get("patch", "") or ""
            files = re.findall(r"^diff --git a/(.+?) b/", patch, re.M)
            # enclosing symbol from hunk header: @@ ... @@ def name( or class Name
            syms = set()
            for m in re.finditer(
                r"^@@[^\n]*?@@[^\n]*?\b(def|class)\s+([A-Za-z_]\w*)",
                patch,
                re.M,
            ):
                syms.add(m.group(2))
            out[d["instance_id"]] = {
                "files": set(files),
                "symbols": syms,
                "base_commit": d.get("base_commit"),
            }
    return out


def _leaf_symbols(qualified: tuple[str, ...]) -> set[str]:
    """Extract leaf definition names from qualified symbol candidates."""
    out: set[str] = set()
    for name in qualified:
        leaf = name.rsplit(".", 1)[-1]
        out.add(leaf)
        out.update(part for part in re.split(r"[_]+", leaf) if part)
    return out


def _discover_source_roots(source_root: Path) -> list[Path]:
    """Locate repo roots under a multi-repo checkout directory.

    Repos are laid out as ``<root>/<org>/<repo>/`` and each repo root carries a
    SOURCE-RECEIPT.json marker. When the marker is absent we fall back to any
    directory that directly contains source files (best effort).
    """
    roots = sorted(p.parent for p in source_root.rglob("SOURCE-RECEIPT.json"))
    if roots:
        return roots
    two_level = sorted(p for p in source_root.glob("*/*") if p.is_dir())
    if two_level:
        return two_level
    return [source_root]


def _read_source(
    root: Path, relative: str, *, commit: str | None = None
) -> str | None:
    """Read one source file from disk, or from git objects when the worktree is bare."""
    path = root / relative
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit or 'HEAD'}:{relative}"],
            capture_output=True,
            check=True,
            text=True,
        )
        return completed.stdout
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default=None)
    parser.add_argument(
        "--json", action="store_true", help="print report JSON to stdout"
    )
    parser.add_argument(
        "--policy-filter", default=None, help="only keep rows with this source_policy"
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="optional checkout root to measure top-N symbol candidate recall",
    )
    parser.add_argument("--top-n", type=int, default=16)
    parser.add_argument(
        "--min-symbol-recall",
        type=float,
        default=None,
        help="optional exit-code gate on overall top-N symbol recall",
    )
    args = parser.parse_args()
    gold = golden_map()
    instance_of = {}
    for p in RUNS.rglob("ATTEMPT.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        t = d.get("task", {})
        if t.get("task_id"):
            instance_of[t["task_id"]] = t.get("instance_id")

    rows = []
    for p in sorted(RUNS.rglob("shared-contexts/*/operator.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("cohort") != "feedback":
            continue
        tid = d.get("task_id")
        iid = instance_of.get(tid)
        if not iid:
            m = re.match(r"(?:round1|p1)-(.+)$", tid or "")
            iid = m.group(1) if m else tid
        if iid not in gold:
            continue
        files = set(d.get("target_files", []))
        sym = d.get("target_symbol", "")
        syms = {s for s in re.findall(r"\w+", sym or "")}
        policy = d.get("source_policy", "?")
        if args.policy_filter and policy != args.policy_filter:
            continue
        file_hit = bool(gold[iid]["files"] & files)
        sym_hit = (
            bool(gold[iid]["symbols"] & syms) if gold[iid]["symbols"] else None
        )
        diagnosis = (
            d.get("diagnosis", {}) if isinstance(d.get("diagnosis"), dict) else {}
        )
        probe_instruction = (
            f"{diagnosis.get('defect', '')} {diagnosis.get('trigger', '')}".strip()
        )
        rows.append(
            {
                "instance_id": iid,
                "policy": policy,
                "file_hit": file_hit,
                "symbol_hit": sym_hit,
                "gold_files": sorted(gold[iid]["files"]),
                "gold_symbols": sorted(gold[iid]["symbols"]),
                "harvested_files": sorted(files),
                "harvested_symbol": sym,
                "probe_instruction": probe_instruction,
                "base_commit": gold[iid].get("base_commit"),
            }
        )

    stats = defaultdict(
        lambda: {
            "n": 0,
            "file_hit": 0,
            "sym_measurable": 0,
            "sym_hit": 0,
            "topn_measurable": 0,
            "topn_hit": 0,
        }
    )
    source_roots = (
        _discover_source_roots(Path(args.source_dir)) if args.source_dir else []
    )
    for r in rows:
        s = stats[r["policy"]]
        s["n"] += 1
        s["file_hit"] += r["file_hit"]
        if r["symbol_hit"] is not None:
            s["sym_measurable"] += 1
            s["sym_hit"] += r["symbol_hit"]
        if source_roots and r["probe_instruction"]:
            org = (
                r["instance_id"].split("__", 1)[0]
                if "__" in r["instance_id"]
                else ""
            )
            candidate_roots = (
                [root for root in source_roots if root.parent.name == org]
                or source_roots
            )
            for relative in r["harvested_files"]:
                source = None
                for root in candidate_roots:
                    source = _read_source(
                        root, relative, commit=r.get("base_commit")
                    )
                    if source is not None:
                        break
                if source is None:
                    continue
                candidates = qualified_symbol_candidates(
                    source,
                    r["probe_instruction"],
                    top_n=args.top_n,
                )
                if gold[r["instance_id"]]["symbols"] & _leaf_symbols(candidates):
                    s["topn_hit"] += 1
                s["topn_measurable"] += 1
                break

    report = {
        "schema_version": 2,
        "total_feedback_instances_measured": len(rows),
        "top_n_symbol_candidates": args.top_n,
        "file_level_recall_by_policy": {
            p: {"n": s["n"], "hit": s["file_hit"], "recall": round(s["file_hit"] / s["n"], 4)}
            for p, s in sorted(stats.items(), key=lambda kv: -kv[1]["n"])
        },
        "symbol_level_recall_by_policy": {
            p: {
                "n": s["sym_measurable"],
                "hit": s["sym_hit"],
                "recall": (
                    round(s["sym_hit"] / s["sym_measurable"], 4)
                    if s["sym_measurable"]
                    else None
                ),
            }
            for p, s in sorted(stats.items(), key=lambda kv: -kv[1]["n"])
        },
        "symbol_topn_recall_by_policy": {
            p: {
                "n": s["topn_measurable"],
                "hit": s["topn_hit"],
                "recall": (
                    round(s["topn_hit"] / s["topn_measurable"], 4)
                    if s["topn_measurable"]
                    else None
                ),
            }
            for p, s in sorted(stats.items(), key=lambda kv: -kv[1]["n"])
        },
        "misses": [r for r in rows if not r["file_hit"]],
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {out}")
    if args.json or not args.output:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.min_symbol_recall is not None:
        measurable = sum(s["topn_measurable"] for s in stats.values())
        hits = sum(s["topn_hit"] for s in stats.values())
        overall = hits / measurable if measurable else 0.0
        print(
            f"symbol top-{args.top_n} recall gate: {overall:.4f} "
            f"vs required {args.min_symbol_recall:.4f}"
        )
        if overall < args.min_symbol_recall:
            return 1
    print(f"total measured: {len(rows)}")
    total = len(rows)
    hits = sum(1 for r in rows if r["file_hit"])
    print(
        f"file-level overall recall: {hits}/{total} = "
        f"{hits / total:.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
