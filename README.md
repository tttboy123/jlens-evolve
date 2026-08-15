# JLens Evolve v3

Evidence-centric evolution platform for frozen Agent programs. Strategies submit
neutral `ExecutionPlan` values; the single Runtime emits immutable Receipts;
Observers emit Evidence; Claim Engine classifies native outcomes; only Governance
may approve promotion. Candidates and capabilities enter registries inactive.

```bash
python3 -m pytest -q
PYTHONPATH=src python3 -m evolve --help
```

The product entry point can execute a fresh three-task, six-arm feedback campaign:

```bash
PYTHONPATH=src /path/to/legacy/.venv/bin/python -m evolve fresh-feedback-e2e \
  --config /absolute/path/to/FRESH-FEEDBACK-CONFIG.json \
  --output /absolute/path/to/run
PYTHONPATH=src python3 -m evolve verify-manifest \
  --manifest /absolute/path/to/run/EVIDENCE-MANIFEST.json \
  --root /absolute/path/to/run
```

For continuous feedback-only Skill/Harness evolution from a frozen Model and a
budgeted Teacher, use:

```bash
PYTHONPATH=src python3 -m evolve autonomous-evolve \
  --config /absolute/path/to/AUTONOMOUS-EVOLUTION-CONFIG.json \
  --output /absolute/path/to/evolution-run \
  --worktree-root /absolute/path/to/clean/committed/source
```

See [`docs/AUTONOMOUS-EVOLUTION.md`](docs/AUTONOMOUS-EVOLUTION.md) and the
example config. The loop automatically selects feedback tasks, runs the current
best Harness as baseline, compiles the Teacher Candidate, performs real local
model generation and official matched native evaluation, feeds authoritative
Claims into the next round, and exports an inactive `BEST-HARNESS.json`.

For autonomous Skill evolution, that command is the authoritative live loop. It
validates three clean,
revision-pinned feedback checkouts; freezes the local Qwen and official evaluator
identities; compiles a byte-frozen DeepSeek request/response into an immutable
`CandidateChangeSet`, Skill, zero-argument Operator and Router; and dispatches
baseline/taught generation and native evaluation only via `ExecutionRuntime`.
Baseline never reads the compiled candidate, while taught must consume and hash
it or fail closed. An E2 claim binds a frozen `MatchedCounterfactualPair`: both
model receipts, external traces, native outcomes, the matched execution identity,
and the taught Candidate revision/bundle. Native metadata alone remains E1. E3
additionally requires a `MechanismPrediction` receipt frozen before model dispatch
and independently signed `trusted-jlens-v1` observations whose artifact bytes,
prediction receipt, model subject and receipt order reverify against a process-local
trust root;
an `observer_id`, renamed receipt, self-reported hash or model-provided
`internal_trace` is never sufficient. Before E3 projection, every E2 pair is
rebuilt from the Receipt Store and its literal receipt kinds/artifacts. Missing
trusted evidence or receipt replay remains E2.

Every round starts with a hash-bound Qwen/native/Teacher health preflight. A
failed evaluator is recorded as infrastructure feedback with no gain Claim; it
does not update BEST, and the configured consecutive-infrastructure threshold
stops the loop before an unattended batch can spin indefinitely. Resume rebuilds
state and next-round feedback from the last verified round rather than trusting
mutable summaries.

Every legal outcome enters Candidate Registry. Governance projects neutral as
`no_change`, regression as `rejected`, and evaluator infrastructure failure as
`blocked`; only regression creates a Rejected Registry record. Only an approved
E3 decision with explicit human approval can create an inactive Capability
revision. Approved decisions are signed by the configured process-local
Governance authority; the decision log re-verifies that signature before the
Capability Registry will project it. Teacher cost authorization is an append-only
sequence/hash chain with an atomic head anchor, restart recovery and replay
deduplication. `budget_integrity_status=validated` is projected only by replaying
that ledger; report metadata cannot self-assert budget validity. The command does
not open holdout, retrain model weights, auto-activate Skills or mutate legacy data.

Fresh configuration must bind the final Git SHA, exactly three feedback tasks,
the frozen model/harness paths, and byte-frozen `teacher_request` and
`teacher_response` paths. Historical `operator_skill_path`/`span_skill_path`
inputs are no longer accepted by the live taught path.

Trusted JLens is optional. Without it the command can reach at most E2. To enable
the E3 observation path, add this object to the frozen config and provide the
secret only through the named process environment variable:

```json
{
  "trusted_jlens": {
    "trace_root": "/absolute/path/to/canonical-jlens-traces",
    "secret_env": "JLENS_OBSERVER_SECRET",
    "key_id": "jlens-production-key-1",
    "implementation_id": "jlens-observer-v1",
    "implementation_sha256": "<64 lowercase hex>",
    "observer_config_sha256": "<64 lowercase hex>",
    "prediction_id": "role-commitment-v1",
    "expected_internal_effect": {
      "concept": "declared-role",
      "phase": "symbol-selection",
      "min_final_score": 0.7,
      "min_location_count": 2,
      "require_non_decreasing": true
    }
  }
}
```

The trace source must create one canonical JSON file named `<plan_id>.json` in
`trace_root` before that plan's observation stage. Its only top-level field is
`locations`; each location carries `layer`, `token_position`, `phase`, and a
`concept_scores` object. The secret is never serialized into campaign artifacts.

`legacy-feedback-e2e` remains available only as a historical evidence-import
compatibility command. Public AgentProgram campaigns support a deterministic
fixture profile and one allowlisted local live profile. The bounded
`PortfolioOrchestrator` connects one authoritative AgentProgram failure to an
inactive validated Capability and a new live tournament without minting Claims
or activating assets. See `docs/MIGRATION-v3.md` and
`docs/TRUST-BOUNDARIES.md` for authority and threat boundaries.
