# Repository Cleanup Report

## Scope and classification

Cleanup covers the autonomous-evolution, AgentProgram and Portfolio additions
between `e31bec5071e655664321a0f747914822ce89099d` and the integrated local
branch. Immutable runs, registries, cost ledgers, historical catalogs, review
artifacts, models and external task checkouts are outside deletion scope.

- keep: Campaign Kernel, `ExecutionRuntime`, Receipt/Evidence/Claim authorities,
  Governance, autonomous runner, public strategy entries and inactive registries;
- compatibility: Legacy import, pinned Qwen/native adapters and fixture
  AgentProgram profile used by historical replay;
- deprecated: direct historical execution authorities in the separate v2.5
  repository; they are not v3 product authorities;
- delete-now: duplicate cleanup reports under `docs/`, monolithic mixed-responsibility
  AgentProgram campaign layout, and duplicate local receipt/native/Claim projection;
- generated/cache: ignored Python, pytest and Ruff caches plus local `runs/`.

## Consolidation and deletion

- Deleted the duplicate tracked `docs/REPOSITORY-CLEANUP-REPORT.md/json`; this
  root report is the sole cleanup authority.
- Split the AgentProgram implementation into fixture/profile dispatch,
  `agent_program_live` orchestration and `agent_program_authority` replay/report
  projection. Receipt/native/Claim identity now has one projector.
- Exported Runtime receipt and native-execution identity functions and reused
  them during replay instead of duplicating their canonical hash formulas.
- Kept Portfolio to one product path: authoritative failure → CapabilityGap →
  inactive Skill → Governance-approved inactive Capability → complete
  AgentProgram revision → live tournament.
- Added an explicit trust-boundary document rather than multiplying local
  full-filesystem rewrite tests.

No immutable evidence, registry history, cost event, public CLI or compatibility
reader was deleted. No model cache, credential or Docker layer is tracked.

## Post-cleanup inventory and verification

- Source Python: 69 files, 20,451 lines.
- Test Python: 28 files, 12,364 lines.
- Targeted AgentProgram/Portfolio/autonomous suites: passed.
- Final integrated suite: 259 passed.
- Ruff: passed for `src` and `tests`.
- mypy: passed for 69 source files.
- `git diff --check`: passed.

Counts are informational, not quality targets. Final commands and commit identity
are also recorded in the final handoff; the independent real Teacher experiment
is bound to its explicitly authorized source commit and immutable run directory.

## Deferred deletion

Legacy v2.5 readers and Qwen/native bridges remain until semantic replay
equivalence no longer depends on them. Broader Portfolio optimization and
production activation are future product work, not hidden v3 stubs.
