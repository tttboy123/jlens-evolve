# Repository Cleanup Report

## Inventory and classification

- keep: `src/evolve/**`, tests, architecture and migration documentation.
- compatibility: the three Strategy adapters and legacy evidence importer.
- deprecated: historical execution authorities remain only in the separate,
  read-only v2.5 repository; none were copied into this v3 repository.
- delete-now: no source file qualified after reference and public-interface audit.
- generated/cache: Python bytecode, pytest/ruff caches, build output and `runs/` are
  excluded by `.gitignore`; run evidence remains local and content-addressed.

## Cleanup action

No project source was deleted. This repository began as an architecture-only Git
worktree, so v3 was implemented once behind a single formal entry point instead of
copying legacy implementations. The cleanup therefore consisted of excluding
rebuildable cache/run products and documenting the compatibility boundary.

## Verification

Pre-cleanup and post-cleanup test results are recorded under the final run's raw
test outputs. Historical immutable evidence, manifests, reviews, registries and
cost ledgers were not modified or deleted.

## Deferred deletion

Legacy v2.5 modules remain because historical replay and schema-reader obligations
have not yet completed semantic equivalence testing. They are not v3 authorities.
