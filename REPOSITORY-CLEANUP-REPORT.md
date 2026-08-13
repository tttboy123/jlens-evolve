# Repository Cleanup Report

## Inventory and classification

- keep: `src/evolve/**`, tests, architecture and migration documentation.
- compatibility: the three Strategy adapters, historical evidence importer,
  `LegacyQwenCellRunner`, and `LegacyOfficialNativeEvaluator`.
- deprecated: historical execution authorities remain only in the separate,
  read-only v2.5 repository; none were copied into this v3 repository.
- delete-now: no source file qualified after reference and public-interface audit.
- generated/cache: Python bytecode, pytest/ruff caches, build output and `runs/` are
  excluded by `.gitignore`; run evidence remains local and content-addressed.

## Cleanup action

No project source qualified for deletion. Import/reference, CLI, test, dynamic
adapter and historical-replay audits found one authority for each v3 product
responsibility. The legacy Qwen and official-harness implementations are reached
only through narrow Runtime adapters; they were not copied. Empty stubs and
duplicate Campaign, Runtime, Transport, Evaluator, Budget, Catalog or Registry
authorities were not found. Rebuildable cache/run products remain ignored.

Post-cleanup inventory: 36 source Python files, 10 test Python files, and 7,415
Python lines before final formatting (informational only). The pre-extension Git
revision contained 48 tracked files; this extension adds the bounded live adapters,
orchestrator and their contract tests without adding a duplicate authority.

## Verification

Post-cleanup verification covers import/compile, targeted tests, the full suite,
Ruff, Git diff checks, the live campaign and manifest replay. Raw final-HEAD
outputs are stored under the final run. Historical immutable evidence, manifests,
reviews, registries and cost ledgers were not modified or deleted.

## Deferred deletion

Legacy v2.5 modules remain because historical replay and schema-reader obligations
have not yet completed semantic equivalence testing. They are not v3 authorities.
