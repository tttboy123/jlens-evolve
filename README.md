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
identities; dispatches baseline/taught generation and native evaluation only via
`ExecutionRuntime`; projects external-trace, native and cost evidence; classifies
matched pairs; and records the resulting candidate/capability as inactive. A
frozen real Teacher receipt may be replayed, but it is never allowed to promote a
candidate. It does not open holdout, retrain model weights or mutate legacy data.

`legacy-feedback-e2e` remains available only as a historical evidence-import
compatibility command. See `docs/MIGRATION-v3.md` for authority boundaries.
