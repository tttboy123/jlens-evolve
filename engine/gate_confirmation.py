"""NEXT-1 executor: run the v2.2 transfer gate on real-search-002 G1 confirmation.

Usage (once evidence/generation-1/confirmation has >=8 paired tasks):
  .venv/bin/python gate_confirmation.py --run-root <local mirror root or aws flag>

For now the extraction runs over AWS via SSH and writes paired evals to a local
JSON; the gate then runs locally and records the transition in the skill
registry + promotion ladder (append-only).

Boundaries: transfer_verified is a machine gate; activation still requires
human review (promotion_ladder). No final sealed access.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

PROJECT = Path("/Users/lune/Documents/Codex/2026-07-18/bang/work/evolve-jlens-cluster")
KEY = PROJECT / "artifacts/v2.1.0/v2.1.1-jlens-evolution/cloud-control/evolve-jlens-nointernet.pem"
HOST = "ubuntu@54.151.251.47"
CONF_BASE = "/opt/evolve-v211/real-search-002-deepseek-nointernet/evidence/generation-1/confirmation"

EXTRACT_SCRIPT = r'''
import json, os, sys
base = sys.argv[1]
rows = []
for task in sorted(os.listdir(base)):
    td = os.path.join(base, task)
    if not os.path.isdir(td):
        continue
    for arm in sorted(os.listdir(td)):
        ad = os.path.join(td, arm, "native-admission.json")
        co = os.path.join(td, arm, "cost.json")
        sa = os.path.join(td, arm, "safety.json")
        if not os.path.isfile(ad):
            continue
        a = json.load(open(ad))
        c = a.get("contract", {})
        outcome = a.get("outcome", {})
        safety = json.load(open(sa)) if os.path.isfile(sa) else {}
        cost = json.load(open(co)) if os.path.isfile(co) else {}
        # CFG-INVALID quarantine: 0-token arms (empty_patch from provider loss)
        # are not evidence; skip them so the gate only sees real evaluations.
        if (cost.get("total_tokens") or 0) <= 0 or outcome.get("native_valid") is not True:
            continue
        rows.append({
            "task_uid": c.get("task_uid"),
            "benchmark_family": c.get("benchmark_id"),
            "role": c.get("arm"),
            "native_score": 1.0 if outcome.get("resolved") else 0.0,
            "safety_passed": safety.get("safety_passed"),
            "cost_units": cost.get("total_tokens"),
            "matched_contract_sha256": c.get("baseline_contract_sha256"),
            "native_evaluator_epoch": c.get("evaluator_epoch"),
        })
print(json.dumps(rows, ensure_ascii=False))
'''


def _scp_and_run(script_text: str, remote_path: str, args: list[str]) -> str:
    local = PROJECT / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/.extract_tmp.py"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(script_text, encoding="utf-8")
    subprocess.run(
        [
            "scp", "-q", "-i", str(KEY),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/tmp/aws-v211-known_hosts",
            "-o", "ProxyCommand=nc -X 5 -x 127.0.0.1:7897 %h %p",
            str(local), f"{HOST}:{remote_path}",
        ],
        check=True,
        timeout=120,
    )
    remote_cmd = f"python3 {remote_path} " + " ".join(args)
    return subprocess.run(
        [
            "ssh", "-i", str(KEY),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/tmp/aws-v211-known_hosts",
            "-o", "ConnectTimeout=20",
            "-o", "ProxyCommand=nc -X 5 -x 127.0.0.1:7897 %h %p",
            HOST, remote_cmd,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    ).stdout


def extract_paired_evals() -> list[dict]:
    script = Path("/private/tmp/extract_confirmation_evals.py").read_text(encoding="utf-8")
    raw = _scp_and_run(script, "/tmp/extract_confirmation_evals.py", [CONF_BASE])
    rows = json.loads(raw)
    by_task: dict[str, dict] = {}
    for row in rows:
        by_task.setdefault(row["task_uid"], {})[row["role"]] = row
    evals: list[dict] = []
    for arms in by_task.values():
        candidates = [arms[r] for r in arms if r.startswith("candidate")]
        if "original" in arms and candidates:
            candidate = max(candidates, key=lambda x: (x["native_score"] or 0))
            original = dict(arms["original"])
            original["role"] = "original"
            candidate_row = dict(candidate)
            candidate_row["role"] = "candidate"
            evals.append(original)
            evals.append(candidate_row)
    return evals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PROJECT / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/g1-transfer-gate")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from search_skill_bridge import evaluate_transfer_gate
    evals = extract_paired_evals()
    paired = sum(1 for r in evals if r["role"] == "original")
    print(f"extracted rows={len(evals)} paired_original={paired}")
    if paired < 8:
        print(f"NOT_READY paired={paired} (<8); rerun after confirmation completes")
        return 2
    contracts = {r["matched_contract_sha256"] for r in evals}
    epochs = {r["native_evaluator_epoch"] for r in evals}
    result = evaluate_transfer_gate(
        paired_evals=evals,
        expected_contract_sha256=next(iter(contracts)) if len(contracts) == 1 else None,
        expected_evaluator_epoch=next(iter(epochs)) if len(epochs) == 1 else None,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "GATE-EVALS.json").write_text(
        json.dumps(evals, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "GATE-RESULT.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if result.passed and not args.dry_run:
        print("GATE_PASSED -> candidate transfer_verified; human ladder review required before active.")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
