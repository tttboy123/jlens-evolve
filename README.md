# JLens Evolve v3

Evidence-centric evolution platform for frozen Agent programs. Strategies submit
neutral `ExecutionPlan` values; the single Runtime emits immutable Receipts;
Observers emit Evidence; Claim Engine classifies native outcomes; only Governance
may approve promotion. Candidates and capabilities enter registries inactive.

```bash
python3 -m pytest -q
PYTHONPATH=src python3 -m evolve --help
```

The v3 vertical slice imports three real feedback-only matched native pairs from
the read-only v2.5 evidence repository. It does not open holdout or retrain models.
See `docs/MIGRATION-v3.md` for compatibility boundaries.

