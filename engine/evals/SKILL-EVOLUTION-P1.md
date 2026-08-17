# EVAL DEFINITION: Skill Evolution P1

## Capability evals

1. **TaskSet contract**
   - accepts exactly versioned, fingerprinted task records;
   - requires at least 3 feedback and 3 hold-out tasks;
   - rejects duplicate instance IDs, unknown fields, test-only targets, gold patches or test patches.
2. **No-leak preflight**
   - every pinned source revision exists in its local source repository;
   - every allowed target exists at that revision and is not a test path;
   - hold-out instructions never enter `FeedbackPackage` / `ParentModelRequest`.
   - evaluator-only reference patches may expose implementation target paths and
     patch SHA only; patch content is never copied into Student or target-selection
     evidence.
   - every allowed implementation target covers the evaluator reference target set,
     and the target count fits the selected mechanism's file-edit capacity.
3. **Isolated materialization**
   - each arm starts from the pinned source revision in a fresh checkout;
   - materialization does not change source HEAD or source worktree state.
4. **Mechanism comparison**
   - structured-edit and hunk mechanisms run on the same TaskSet;
   - baseline and taught revisions differ only in the injected Skill revision;
   - raw output, parsed patch, rejection code, timing, model/revision/task fingerprints are frozen per arm.
   - evidence reused across a TaskSet revision must pass the composition contract:
     identical task, target-selection and condition fingerprints plus intact cell
     and artifact SHA values; reused evidence is indexed, never rewritten.
5. **True capability gate**
   - apply validity is a structural prerequisite only;
   - completion requires at least one feedback task changing from native unresolved to resolved;
   - hold-out native score must not regress and infrastructure errors fail closed.

## Regression evals

- Existing P0 contracts, parent-call ledger and append-only registry remain valid.
- Root `swe_4b_patch.py` remains a thin compatible CLI.
- Frozen engine surfaces are not modified.
- Full pytest, Ruff lint and Ruff format checks pass.

## Stop conditions

- Stop after two consecutive no-progress rounds.
- Stop before any DeepSeek, Codex proxy, AWS, paid call or formal Skill promotion
  unless the user provides specific authorization.
- Stop if fewer than 3+3 source-valid, gold-free paired tasks can be frozen.

## Success metrics

- TaskSet preflight: 6/6 ready, split 3 feedback + 3 hold-out.
- Structural P1 report: all 24 planned cells accounted for
  (`6 tasks × 2 mechanisms × baseline/taught`), including classified failures.
- Capability completion: feedback native gain ≥ 1 resolved task and hold-out native
  gain ≥ 0.
- Regression: pass^1 for the full local verification gate on every code change;
  repeat to pass^3 before a promotion recommendation.
