# Evolve × J-lens clustering experiment eval contract

## Capability gates

- OpenEvolve evaluates child mutations with a deterministic, non-LLM evaluator.
- Every public case is a numeric metric; one more passing case dominates weighted ties.
- Accepted children preserve every public case passed by their parent.
- Exact-source and AST duplicates are rejected; behavior-equivalent variants are capped.
- Rejected children remain traceable but never enter islands, archives, or best tracking.
- MAP-Elites uses public case pass rate and AST complexity.
- Prompt-hidden deterministic holdout cases are evaluated only after search.
- A checkpoint exists at every fifth iteration and the trace retains parent/child code.
- A compatible checkpoint is discovered and resumed automatically across processes.
- Candidate events, lessons, skill candidates, meta-policy trials, and manifests persist
  under project-scoped `state/`.
- Every executable candidate is statically screened before restricted execution.
- Every unique program in the lineage receives J-lens and logit-lens signatures at the
  same fixed prompt position and the same 31 fitted source layers.
- Mutation clustering uses lens features only; evaluator scores are joined afterward.
- The analysis reports cluster stability, silhouette, outcome association, permutation
  uncertainty, component deltas, AST changes, and representative mutations.
- J-lens is compared with logit-lens and a shuffled-feature null.

## Data-quality gates

- Unique mutation edge IDs; conflicting duplicates fail validation.
- Parent and child code plus component metrics are present for every analyzed edge.
- All lens and attribution numeric features are finite.
- No cluster is interpreted without its sample count and score-delta distribution.
- Claims remain observational: no head/MLP attribution and no causal language.
- RSI requires productive operator self-change; candidate improvement alone is not RSI.
- PSI reports same-search resume and cross-task transfer separately.

## Deliverables

- Raw OpenEvolve lineage and checkpoints under `runs/`.
- Raw per-program lens signatures and per-mutation feature tables under `results/`.
- Executed clustering notebook and figures under `analysis/`.
- Detailed technical report and updated handoff under the user-facing `outputs/` folder.
