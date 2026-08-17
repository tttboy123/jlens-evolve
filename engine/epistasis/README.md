# Epistasis / Diversity-Collapse Experiments (A-E)

Empirical suite for the RSI diversity-collapse question:

> 坍缩是否因为"单个 RSI 算子变更不够"，必须多个算子同时修改（涌现/临界效应）
> 才会让全局状态发生改变？

The suite answers this with five falsifiable experiments over a deterministic,
schema-parameterized operator universe, and is explicitly built to scale to
large task matrices and to accept **any model vendor's** OpenAI-compatible
endpoint.

## What it measures

| Exp | Question | Method |
|---|---|---|
| A | Does composing operators escape a plateau that single operators cannot? | Paired cells: singles vs both orderings of every mechanism pair; sign test + additive-prediction delta |
| B | Is there a *critical coverage* threshold (coverage -> yield)? | 2^4 subset sweep per (task, seed); Spearman/Pearson + bootstrap CI + permutation p + threshold-vs-linear BIC |
| C | Is *simultaneous* multi-operator change required, or does sequential lineage suffice? | Rotating single-operator lineage cells vs composed cells |
| D | Does the conjunction improve *more than the sum of parts* (epistasis)? | 2^k factorial interaction `gain(i,j)-gain(i)-gain(j)+gain(empty)` with bootstrap CI |
| E | Does cross-task validation show a threshold? | Per-operator cumulative gain vs number of tasks + changepoint |

## Architecture

```
epistasis/
├── operators.py        schema-parameterized deterministic AST operators + composition
├── tasks.py            real (paid/payout/refund) + synthetic task generator (coupled cases)
├── search.py           fixed-budget cell engine: deterministic / llm / lineage
├── model_transport.py  stub + OpenAI-compatible transport (vendor plug-in)
├── analysis.py         bootstrap / permutation / sign test / threshold BIC / changepoint
├── experiments.py      A-E cell plans + analysis
├── report.py           JSON + SUMMARY.md evidence writer
└── cli.py              `python -m epistasis run`
```

Design invariants (same as the rest of the repo):

- fixed budget per cell; admission = non-regression + exact/AST/behavior dedup
  (`AdmissionPolicy`, behavior-equivalent limit 2);
- deterministic mode makes **zero model calls** and is byte-reproducible per seed;
- operators are pure `source -> source` transforms whose postcondition is a
  *behavioral probe* (minimal case only that mechanism can satisfy);
- every cell persists to `out/cells/*.json` with full events -> `--resume` is idempotent.

## Run

Deterministic (default; no model, fully reproducible):

```bash
.venv/bin/python -m epistasis run \
  --real paid,payout,refund --synthetic 12 --seeds 3 --budget 8 \
  --workers 8 --out runs/epistasis-scale
```

LLM mode (any OpenAI-compatible endpoint - local `mlx_lm.server`, vLLM,
DeepSeek, hosted vendors).  `model.json`:

```json
{"provider": "openai", "base_url": "http://127.0.0.1:18080/v1",
 "model": "Qwen3.5-4B-mlx-4bit", "api_key_env": "OPENAI_API_KEY",
 "temperature": 0.7, "max_tokens": 1024}
```

```bash
export OPENAI_API_KEY=...
.venv/bin/python -m epistasis run \
  --mode llm --model-config model.json \
  --real paid,payout --synthetic 0 --seeds 1 --budget 4 \
  --out runs/epistasis-llm
```

Outputs: `SUMMARY.md`, `EXPERIMENTS.json`, `CELLS.json`, `EVENTS.jsonl`,
`cells/*.json` (resume).

## Scale / vendor knobs

- `--synthetic N` generates N deterministic record-cleaning task variants
  (mechanism subsets + coupled cases + uncovered noise) so coverage/epistasis
  statistics have many data points without needing real benchmarks.
- `--seeds` `--budget` `--workers` control the cell matrix; every cell is
  independent so the sweep parallelizes trivially.
- Swap the model by pointing `--model-config` at any OpenAI-compatible endpoint;
  the analysis (A-E) is model-independent, so vendors can compare models on the
  same evidence and correlation reports.

## Deterministic operators (operator universe)

| mechanism | operator |
|---|---|
| status | `canonicalize_status` |
| amount | `reject_invalid_amounts` |
| identity | `normalize_identity` |
| empty | `drop_empty_identity` |

## Interpretation of results

- **A**: `composition_helps` if wins > losses (sign test p<0.05); `order_sensitive_cells`
  tells you whether operator order matters.
- **B**: `threshold_yield.bic_advantage > 0` means a step (critical coverage) model
  beats a linear model; `threshold` is the critical coverage. The ladder table
  shows the mean gain per coverage level.
- **C**: `lineage_sufficient` means sequential single-operator generations reach
  the same score as composed mutation - simultaneous change is **not** required
  when operators compose and admission preserves them.
- **D**: `synergistic_pairs` are mechanism pairs whose interaction CI excludes
  zero - the quantitative "emergence" signature.
- **E**: `threshold_detected` flags jumps in the cumulative-gain curve. With the
  deterministic per-task search there is no cross-task learning, so jumps
  reflect *mechanism prevalence* across tasks rather than a learned validation
  mass; a true cross-task threshold must be tested in the PSI/lessons layer.

## Tests

```bash
.venv/bin/python -m pytest tests/test_epistasis_operators.py \
  tests/test_epistasis_tasks.py tests/test_epistasis_search.py \
  tests/test_epistasis_analysis.py tests/test_epistasis_experiments.py -q
```
