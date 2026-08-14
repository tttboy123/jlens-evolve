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

That command is the sole live orchestration entry. It validates three clean,
revision-pinned feedback checkouts; freezes the local Qwen and official evaluator
identities; compiles a byte-frozen DeepSeek request/response into an immutable
`CandidateChangeSet`, Skill, zero-argument Operator and Router; and dispatches
baseline/taught generation and native evaluation only via `ExecutionRuntime`.
Baseline never reads the compiled candidate, while taught must consume and hash
it or fail closed. An E2 claim binds a frozen `MatchedCounterfactualPair`: both
model receipts, external traces, native outcomes, the matched execution identity,
and the taught Candidate revision/bundle. Native metadata alone remains E1. E3
additionally requires independently signed `trusted-jlens-v1` observations whose
artifact bytes and model subject reverify against a process-local trust root;
an `observer_id`, renamed receipt, self-reported hash or model-provided
`internal_trace` is never sufficient. Before E3 projection, every E2 pair is
rebuilt from the Receipt Store and its literal receipt kinds/artifacts. Missing
trusted evidence or receipt replay remains E2.

Every legal outcome enters Candidate Registry. Governance projects neutral as
`no_change`, regression as `rejected`, and evaluator infrastructure failure as
`blocked`; only regression creates a Rejected Registry record. Only an approved
E3 decision with explicit human approval can create an inactive Capability
revision. Approved decisions are signed by the configured process-local
Governance authority; the decision log re-verifies that signature before the
Capability Registry will project it. Teacher cost authorization is an append-only sequence/hash chain with
an atomic head anchor, restart recovery and replay deduplication. The command does
not open holdout, retrain model weights, auto-activate Skills or mutate legacy data.

Fresh configuration must bind the final Git SHA, exactly three feedback tasks,
the frozen model/harness paths, and byte-frozen `teacher_request` and
`teacher_response` paths. Historical `operator_skill_path`/`span_skill_path`
inputs are no longer accepted by the live taught path.

`legacy-feedback-e2e` remains available only as a historical evidence-import
compatibility command. See `docs/MIGRATION-v3.md` for authority boundaries.
