# engine/ (evolve-jlens-cluster harness)

Auditable infrastructure for project-local **Agent Skill evolution experiments**:
frozen model weights, JLens observation, Evolve application-layer search, native
evaluator authority, and human-gated promotion.  Companion codebase of
`/Users/lune/Documents/Jlens Evolve` (v3 evidence-centric platform).

> Note: this root README was regenerated from the surviving project docs after an
> experiment-suite edit overwrote the previous index.  The previous README body
> was not recoverable (this directory is not a git checkout).  Project content
> lives in the linked docs below; the new experiment suite is documented in
> [`epistasis/README.md`](epistasis/README.md).

## Canonical docs (zh-CN)

- [ARCHITECTURE-AND-PRINCIPLES.zh-CN.md](ARCHITECTURE-AND-PRINCIPLES.zh-CN.md) - architecture and invariants
- [ROADMAP.zh-CN.md](ROADMAP.zh-CN.md) - product scope and version roadmap
- [STATUS.zh-CN.md](STATUS.zh-CN.md) - current stage and changelog
- [EVOLVE_EXTENSION.zh-CN.md](EVOLVE_EXTENSION.zh-CN.md) - guarded OpenEvolve extension
- [STRUCTURED_MUTATION_RSI_PSI.zh-CN.md](STRUCTURED_MUTATION_RSI_PSI.zh-CN.md) - structured mutation v4, RSI/PSI results
- [HANDOFF.zh-CN.md](HANDOFF.zh-CN.md), [HANDOFF-MINIMAXCODE.md](HANDOFF-MINIMAXCODE.md)

## Key capabilities

- RSI / PSI evolution loops (`evolve_runtime.py`, `psi_runtime.py`,
  `meta_evolution_runtime.py`, `structured_mutation.py`)
- Duplicate-aware novelty proxy (`novelty_proxy.py`) and admission guards
  (`admission_policy.py`)
- Deterministic record-cleaning tasks (`tasks/payout_cleaning`, `tasks/refund_cleaning`)
- Experience -> candidate-skill pipeline (`experience_store.py`, `search_skill_bridge.py`)
- JLens observation sidecar (`local_lens_agent.py`, `trace_observer.py`, `lens_features.py`)
- **Epistasis / diversity-collapse experiment suite A-E** (`epistasis/`) - see
  [epistasis/README.md](epistasis/README.md)

## Test

```bash
.venv/bin/python -m pytest tests -q
```
