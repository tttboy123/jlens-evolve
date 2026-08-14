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
- a goal, maximum rounds, no-progress patience, infrastructure/failure/disk
  stop limits, and deterministic seed.

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
      "schema_version": 1,
      "instance_id": "project__issue-1",
      "task_id": "legacy-route-project__issue-1",
      "project": "owner/project",
      "repo": "owner/project",
      "benchmark_id": "swe-bench-verified",
      "cohort": "feedback",
      "source_uri": "/absolute/source-pool/project__issue-1",
      "source_repository": "/absolute/read-only/source-cache/owner/project",
      "base_revision": "<40-character Git SHA>",
      "source_revision": "<same 40-character Git SHA>",
      "benchmark_base_commit": "<same 40-character Git SHA>",
      "catalog_fingerprint": "<64-character SHA-256 from the frozen catalog>",
      "fingerprint": "<same catalog SHA-256 used by the legacy Qwen adapter>",
      "instruction": "The complete feedback-task instruction.",
      "instruction_sha256": "<SHA-256 of the instruction bytes>",
      "allowed_targets": [
        "src/project/module.py",
        "tests/test_module.py"
      ]
    }
  ]
}
```

This is the real local-Qwen task-row shape, not a minimal catalog sketch.
`catalog_fingerprint` is mandatory for native execution. `task_id`,
`benchmark_base_commit`, `fingerprint`, `instruction`, and `allowed_targets`
are consumed by the compatibility Qwen adapter. `task_fingerprint_sha256` is
not an input field: the autonomous selector derives it from the complete row
and freezes it in `TASK-SELECTION.json`.

Every `source_uri` must resolve below the configured `source_pool`, identify a
clean checkout at `base_revision`, and agree with the catalog and benchmark
revision fields. The loader rejects holdout, final-sealed, burned, r076, and
r078 identifiers before task selection. Each frozen selection contains the
complete selected task records and their derived literal SHA-256 fingerprints.

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
- `rounds/round-XXXX/AUTONOMOUS-ROUND-RESULT.json`: sealed round decision,
  including whether that exact candidate was accepted as the search parent.
- `best/BEST-HARNESS.json`: hash-verified convenience projection of the current
  accepted parent, not an independent parent-selection authority.
- `EVOLUTION-RESULT.json`: stop reason, gains, cost, evidence mode, and safety
  attestations.

To verify a sealed manifest:

```bash
PYTHONPATH=src python3 -m evolve verify-manifest \
  --manifest /absolute/path/to/evolution-run/EVIDENCE-MANIFEST.json \
  --root /absolute/path/to/evolution-run
```

Load and hash-verify the projection with the same frozen model identity:

```python
from evolve.autonomous import load_best_harness

compiled = load_best_harness(
    "/absolute/path/to/run/best/BEST-HARNESS.json",
    expected_model_identity_sha256="<GOAL.json:model_identity_sha256>",
)
```

`compiled is None` means the current best is the explicit `empty-harness-v1`;
otherwise it is a fully reloadable `CompiledRevision`, including its frozen
Teacher request/response, compile spec, cost/model receipts, Candidate,
Skill/Operator/Router and optional Memory Policy. Baseline in the next round
loads the compiled artifacts from the latest accepted, manifest-verified sealed
round. On resume, `ROUND-INDEX.jsonl` plus that round's verified manifest and
`accepted_as_best` decision are the parent authority. `BEST-HARNESS.json` is
exported from those same artifacts and must hash-reload to the same bundle; it
cannot promote a candidate or override the sealed-round decision. The proposed
Candidate is only consumed by taught execution. `active` remains `false`:
Governance does not auto-activate a Skill or Capability.

Each completed round also freezes Teacher-safe `campaign_feedback`, rebuilt
from authoritative model, external-trace and native receipts. The next round
carries it unchanged as `prior_campaign_feedback`; patch bytes, prompts and
unrecognized raw error text are not sent to the Teacher.

## Stop conditions and safety

The loop stops only for `goal_reached`, `max_rounds_reached`, `no_progress`,
`budget_exhausted`, `max_consecutive_infra_failures`,
`max_same_failure_signature`, `disk_limit`, `stopped_by_user`, or
`blocked_integrity`. The three numeric safety limits are configured in `goal`.
A normal round immediately advances without waiting for Codex or a user message.

Confirm `holdout_opened=false` and `skill_active=false` in
`EVOLUTION-RESULT.json`, and inspect every task's `cohort` in the frozen task
selection. The product state machine contains no review/waiting state.

## Strategy status

```bash
python3 -m evolve campaign run --strategy skill-paired --config FRESH.json --output RUN
python3 -m evolve campaign import --strategy legacy --config IMPORT.json --output RUN
python3 -m evolve campaign run --strategy agent-program --config PROGRAM.json --output RUN
```

Skill-paired is the current LIVE native campaign. Legacy import is a read-only
compatibility campaign, not a new evidence-minting path. The public
`campaign run --strategy agent-program` command remains a deterministic
`execution_profile=fixture` facade: it records inactive AgentProgram Registry
rows, reports `claims=[]`, `native_gain_claimed=false`, and
`promotion_eligible=false`.

For application integrations, `AgentProgramSearchStrategy(execution_profile="live")`
and `HashVerifiedAgentProgramTransport` provide a LIVE non-fixture execution
seam. The transport reloads a complete Program revision, verifies its bundle,
prompt, context, tool-policy, capability and parent-lineage hashes, and only then
delegates to an injected `AgentProgramExecutor` through the common
`ExecutionRuntime`. Search-parent advance first rebuilds the configured
ReceiptStore and EvidenceGraph, then requires exactly one persisted E1+ Claim
per participant and verifies native-to-model receipt, Program bundle, revision
and plan identity before making the parent win ties. It never mints Claims,
promotes a capability, or activates a revision.
There is not yet a public non-fixture executor configuration or an automatic
cross-strategy scheduler. Automatic `CapabilityGap` creation and a Portfolio
Orchestrator joining Skill campaigns to live AgentProgram tournaments remain
TARGET architecture, not current product behavior.
