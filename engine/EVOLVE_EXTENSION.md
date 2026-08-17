# Guarded Evolve extension

This project now runs OpenEvolve through a project-local guard instead of changing the
upstream checkout. The extension separates four things that the original experiment
mixed together: candidate evaluation, admission into the breeding population, durable
experience, and self-improvement claims.

## What is implemented

| Capability | Durable form | Promotion or acceptance rule |
|---|---|---|
| Same-search resume | OpenEvolve checkpoints plus `run_manifest.json` | task, initial program, evaluator, and config hashes must match |
| Full attempt archive | `state/archive.jsonl` | append-only, idempotent event ids, file-locked writes |
| Search population | checkpoint programs, islands, archive, MAP cells | parent-case non-regression plus source/AST/behavior dedup |
| Reusable experience | `state/lessons.jsonl` | accepted improvement, hidden-holdout non-regression, repeated evidence |
| Reviewable procedural memory | `state/skills/<family>/SKILL.md` | rendered as `status: candidate`; never installed globally automatically |
| Meta-policy evidence | `state/meta_policy_trials.jsonl` | compare accepted-improvement yield before and after bounded operator revisions |
| Run evidence | `state/run_manifests.jsonl` and per-run reports | immutable execution records; latest manifest also lives in the run directory |

This gives OpenEvolve the useful persistence properties from the compared systems
without pretending that all of them expose the same public protocol:

| Project | Same search across processes | Cross-task experience | Public durable form |
|---|---:|---:|---|
| autoresearch | manual | no automatic migration | dedicated Git branch and commits; `results.tsv` normally uncommitted |
| AlphaEvolve | not public | not public | paper describes an internal program database |
| OpenEvolve upstream | yes | no | checkpoint, program database, archive, metadata |
| AutoResearchClaw + MetaClaw | yes | optional | `lessons.jsonl` to reviewable skill files |
| DGM | yes | within an evolutionary lineage | Git organisms, node metadata, JSONL archive |
| Hermes Self-Evolution | partial | through reviewed merged skills | reports, branch, human-reviewed PR |
| This extension | yes, auto-compatible | yes, task-family scoped | checkpoints + manifests + archive + lessons + local skill candidates |

## Search and evaluation repair

The live evaluator returns one numeric metric per public case. Fitness is lexicographic:
one additional passing case always outweighs any legacy group-weight difference. The
legacy weighted score remains a tie-breaker and diagnostic.

Before a child may enter an island or archive, it must:

1. preserve every public case passed by its parent;
2. pass the static evaluator boundary;
3. differ from all accepted exact-source and AST hashes; and
4. remain below the configured cap for behavior-equivalent structural variants.

Rejected children are still retained in the database, checkpoint, persistent archive,
and evolution trace. They are evidence, but they cannot become parents.

MAP-Elites now uses `case_pass_rate` and AST node complexity. It no longer treats text
length or character-set differences as useful behavioral diversity. The prompt contains
one accepted reference, the passing-case set, and one target failure. The measured initial
prompt is about 2.6K characters, well below the local 4B context budget.

Six deterministic holdout cases cover permutation and malformed-row invariance, status
and identity normalization, integer/float equivalence, bool/non-finite rejection,
split/merge aggregation, and unique finite sorted output. Their ids and results are never
returned to the proposer.

## RSI and PSI definitions

This project uses explicit definitions because the acronyms are overloaded.

- **RSI — recursive self-improvement:** candidate gains are insufficient. The run must
  show a strict multi-generation improvement chain, zero accepted regressions, at least
  one change to the improvement operator, and higher accepted-improvement yield after
  that operator change.
- **PSI — persistent self-improvement:** same-search gains must survive process restart,
  and a holdout-verified lesson from another task identity must be retrieved and retain a
  non-negative hidden-set gain on the target task. Resume and cross-task transfer are
  reported separately; overall PSI passes only when both have evidence.

## Running it

Use the project virtual environment and disable Qwen thinking for this short code task;
otherwise the model can spend the full completion budget on hidden reasoning and return
`content: null`.

```bash
.venv/bin/mlx_lm.server \
  --model models/Qwen3.5-4B-mlx-4bit \
  --host 127.0.0.1 --port 18080 \
  --temp 0.85 --top-p 0.95 --max-tokens 512 \
  --chat-template-args '{"enable_thinking":false}'
```

Start a fresh five-iteration segment:

```bash
.venv/bin/python evolve_runtime.py \
  --output runs/my-run --iterations 5 --resume none --run-id my-run
```

Resume for fifteen additional iterations:

```bash
.venv/bin/python evolve_runtime.py \
  --output runs/my-run --iterations 15 --resume auto
```

`--iterations` means additional iterations after a checkpoint. Auto-resume refuses a
changed task/evaluator/config contract. A report can be rebuilt without a model call:

```bash
.venv/bin/python evolve_runtime.py --output runs/my-run --report-only
```

## Verified 20-iteration smoke result

The repaired run is in `runs/repaired-smoke-v2`.

| Measure | Initial | Best/final |
|---|---:|---:|
| Public cases | 3/13 | 11/13 |
| Public fitness | 0.2304 | 0.8464 |
| Hidden holdout | 0/6 | 3/6 |

The run used 20 official iterations and two replayed attempts caused by deliberate
process-stop tests. Of 22 attempts, 3 entered the breeding population and 19 were
rejected. Accepted parent regressions, accepted exact duplicates, and accepted AST
duplicates were all zero. The attempt archive contains six unique source hashes, five
unique AST hashes, and three behavior signatures. `checkpoint_20` contains all 20
official programs but only three selectable island members.

PSI same-search resume passed: checkpoint state, best public score, and hidden score
survived multiple process restarts. Cross-task transfer was not run, so overall PSI is
correctly `false`. RSI found a strict improvement depth of two, but neither bounded
operator revision increased mutation yield; RSI is correctly `false`.

The remaining best-program failures are normalized status filtering and complete invalid
amount rejection. The local Qwen3.5-4B proposer repeatedly generated the same 11/13
behavior even after removing the fixed LLM seed. That is now a proposer/model limitation,
not archive contamination or a scalar-score collision.

## Boundaries

The evaluator uses an AST screen plus restricted Python builtins. That is a useful task
boundary, not an OS security sandbox. Untrusted general code still requires a real local
sandbox. J-Lens remains an observational analysis sidecar and does not control admission,
generation, inference output, or model weights.
