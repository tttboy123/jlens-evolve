# Repository Cleanup Report

## Scope

This cleanup starts from `b98e1ffe458fc70f705ac26727d76a797aeceae8` on
`codex/v3-evidence-platform`. It covers the v3 causal-closure source and tests;
immutable runs, legacy catalogs, review artifacts, model files and external
checkouts are explicitly outside the deletion scope.

## Classification

- Keep: Campaign Kernel, `ExecutionRuntime`, Receipt/Evidence/Claim authorities,
  Governance decision log, split registries, durable Teacher cost ledger,
  Candidate compiler, and the fresh feedback entry point.
- Compatibility: `LegacyImportStrategy`, `LegacyQwenPairTransport`,
  `LegacyOfficialNativeEvaluator`, and `legacy-feedback-e2e`. These remain needed
  for historical replay or the pinned local model/native harness bridge.
- Deprecated: direct historical evidence import through `legacy-feedback-e2e`.
  It now writes Candidate Registry only and cannot create a Capability.
- Delete now: the frozen taught-Skill loader/fallback, duplicate candidate prompt
  rendering, the in-memory Teacher spend authority, direct Capability append in
  the legacy CLI, and obsolete helper functions `_teacher_evidence` and
  `_freeze_candidate_bundle`.
- Generated/cache: `.pytest_cache`, `.ruff_cache`, Python `__pycache__` trees and
  `src/jlens_evolve.egg-info`; these are removed after final verification and are
  reproducible.

## Removed or consolidated code

- `fresh_feedback.py` no longer accepts `operator_skill_path` or
  `span_skill_path`; it compiles the frozen Teacher exchange and fails closed on
  those legacy fallback fields.
- `LegacyQwenCellRunner` no longer loads old frozen Skill wrappers. Baseline does
  not read the compiled revision; taught revalidates and consumes it.
- Candidate prompt rendering is centralized in `compiled_candidate_prompt` and
  shared by the generic and legacy runtime transports.
- `CandidateProposer._spent_cny` is replaced by one append-only
  `DurableCostLedger` authority with dispatch-once semantics.
- The compatibility CLI no longer creates Capability records directly.

No project file was deleted merely because static search found no caller. Public
interfaces, dynamic adapters and compatibility readers were retained unless a
tested replacement covered their responsibility.

## Authority scan

- Campaign lifecycle: `evolve.kernel.CampaignController`
- Execution dispatch: `evolve.runtime.ExecutionRuntime`
- Paid Teacher budget: `evolve.kernel.DurableCostLedger`
- Native evaluation: injected `NativeEvaluator`, with the legacy official adapter
  behind Runtime
- Candidate compilation: `evolve.proposals.CandidateCompiler`
- Claims: `evolve.evidence.ClaimEngine`
- Promotion: `evolve.governance.GovernanceService` plus immutable decision log
- Product assets: Candidate, Capability and Rejected registries with distinct
  admission rules

The in-memory `BudgetManager` remains because it accounts authorized campaign
work items; it is not a second paid-Teacher ledger.

## Inventory and verification

- Source files before cleanup baseline: 36 Python files, 5,018 lines.
- Source files after v3 causal implementation: 40 Python files, 7,293 lines.
- File/line counts are informational only; new lines are primarily compiler,
  evidence-grade, governance, durable-ledger and adversarial-test code.
- Pre-commit verification: 100 pytest tests passed; Ruff, mypy, compileall and
  `git diff --check` passed.
- Final commit, post-commit real E2E, replay and manifest verification are recorded
  in the immutable run directory rather than asserted by this pre-commit report.

## Deferred deletion

The legacy CLI, strategy facades and native/Qwen adapters remain until historical
replay equivalence no longer depends on them. Removing them now would break an
explicit compatibility responsibility. There are no known unused v3 placeholders
or silent Skill fallbacks in the release path.
