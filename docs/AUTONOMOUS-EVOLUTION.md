# Autonomous Skill/Harness Evolution

`autonomous-evolve` evolves external behavior for a frozen model. It does not
train or modify model weights. Each round selects three feedback tasks, runs the
current parent Harness as a real baseline, asks a budgeted Teacher for one
inactive Candidate, compiles that Candidate, runs the model with it, evaluates
matched baseline/taught patches through the official native evaluator, and uses
the resulting Claims to choose the next parent and construct the next Teacher
request.

## Required inputs

Copy `configs/AUTONOMOUS-EVOLUTION-CONFIG.example.json` and provide:

- a frozen local model directory and the files that identify its revision;
- a feedback-only SWE-bench task-pool JSON file;
- a source-pool directory containing clean, revision-pinned task checkouts;
- pinned official harness and evaluator paths;
- a Teacher provider, model, endpoint, API-key environment-variable name, and
  total CNY budget;
- a goal, maximum rounds, no-progress patience, and deterministic seed.

The task-pool file is either a JSON list of task objects or an object with
`tasks` and `runtime`. Real local-Qwen/native execution requires `runtime`:

```json
{
  "runtime": {
    "legacy_root": "/absolute/path/to/evolve-jlens-cluster",
    "routes_path": "/absolute/path/to/MECHANISM-ROUTES.json",
    "swe_python": "/absolute/path/to/swe-venv/bin/python",
    "multi_python": "/absolute/path/to/multi-venv/bin/python",
    "multi_harness_root": "/absolute/path/to/multi-swe-checkout",
    "pool_root": "/absolute/path/to/benchmark-pool",
    "native_assets_path": "/absolute/path/to/NATIVE-ASSETS.json",
    "native_timeout_seconds": 7200
  },
  "tasks": [
    {
      "instance_id": "project__issue-1",
      "project": "project",
      "benchmark_id": "swe-bench-verified",
      "cohort": "feedback",
      "source_uri": "/absolute/source-pool/project__issue-1",
      "base_revision": "<40-character Git SHA>"
    }
  ]
}
```

Every `source_uri` must resolve below the configured `source_pool`. The loader
rejects holdout, final-sealed, burned, r076, and r078 identifiers before task
selection. Each frozen selection contains the complete selected task records and
their literal SHA-256 fingerprints.

## Run and resume

Export the Teacher key without putting it in the config or output, then run:

```bash
export DEEPSEEK_API_KEY='...'
PYTHONPATH=src python3 -m evolve autonomous-evolve \
  --config /absolute/path/to/AUTONOMOUS-EVOLUTION-CONFIG.json \
  --output /absolute/path/to/evolution-run \
  --worktree-root /absolute/path/to/clean/committed/source
```

The source worktree must be clean and committed. Stop the process normally (or
send SIGTERM from an external supervisor) to pause. Run the exact command again
to resume. Completed baseline, Teacher, compiled Candidate, Qwen prescreen,
native campaign, Receipt Store, and sealed round artifacts are replayed by
identity; a completed paid Teacher request is not sent again. The append-only
Teacher cost ledger reserves before dispatch and recovers across restart.

The current search produces one Candidate per round. Set
`qwen_prescreen_count` to `1` to run a real model-only prescreen or `0` to
explicitly skip it; skipped output is labelled `status=skipped` and never claims
a ModelReceipt. `native_finalist_count` must be `1`.

## Inspect results

- `EVOLUTION-STATE.json`: active/terminal product state and next round.
- `TASK-SELECTION-INDEX.jsonl`: hash-chained task selections.
- `TEACHER-CALL-LEDGER.jsonl` and `teacher/COST-LEDGER.jsonl`: frozen call
  identity and authoritative budget events.
- `rounds/round-XXXX/`: task, baseline, Teacher, compiled Harness, prescreen,
  Campaign result, Receipts, Evidence Graph, native evidence, and manifest.
- `best/BEST-HARNESS.json`: current best inactive external asset.
- `EVOLUTION-RESULT.json`: stop reason, gains, cost, evidence mode, and safety
  attestations.

To verify a sealed manifest:

```bash
PYTHONPATH=src python3 -m evolve verify-manifest \
  --manifest /absolute/path/to/evolution-run/EVIDENCE-MANIFEST.json \
  --root /absolute/path/to/evolution-run
```

Load the paths in `best/BEST-HARNESS.json` with the same frozen model. Baseline
in the next round loads this Harness as the parent; the proposed Candidate is
only consumed by taught execution. `active` remains `false`: Governance does not
auto-activate a Skill or Capability.

## Stop conditions and safety

The loop stops only for goal reached, maximum rounds, no progress, exhausted
Teacher budget, repeated evaluator infrastructure failure, integrity failure, or
an explicit process stop. A normal round immediately advances without waiting
for Codex or a user message.

Confirm `holdout_opened=false` and `skill_active=false` in
`EVOLUTION-RESULT.json`, and inspect every task's `cohort` in the frozen task
selection. The product state machine contains no review/waiting state.

## Strategy status

```bash
python3 -m evolve campaign run --strategy skill-paired --config FRESH.json --output RUN
python3 -m evolve campaign import --strategy legacy --config IMPORT.json --output RUN
python3 -m evolve campaign run --strategy agent-program --config PROGRAM.json --output RUN
```

Skill-paired is live. Legacy import is a compatibility path. AgentProgram uses
the common Strategy and CampaignRunner contracts but deliberately returns
`not-yet-live` (exit code 2) until a tournament authority adapter exists. The
project does not claim that all three are live.
