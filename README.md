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
it or fail closed. Matched native pairs produce E2 counterfactual claims. E3 is
available only for cross-task native gains with aligned external/internal
prediction evidence; missing evidence remains E2.

Every legal outcome enters Candidate Registry. Neutral, regression and
infrastructure failure never enter Capability Registry. Governance records an
immutable decision, and only an approved E3 decision with explicit human approval
can create an inactive Capability revision. Teacher cost authorization is an
append-only ledger that survives restart and deduplicates replay. The command does
not open holdout, retrain model weights, auto-activate Skills or mutate legacy
data.

Fresh configuration must bind the final Git SHA, exactly three feedback tasks,
the frozen model/harness paths, and byte-frozen `teacher_request` and
`teacher_response` paths. Historical `operator_skill_path`/`span_skill_path`
inputs are no longer accepted by the live taught path.

`legacy-feedback-e2e` remains available only as a historical evidence-import
compatibility command. See `docs/MIGRATION-v3.md` for authority boundaries.
