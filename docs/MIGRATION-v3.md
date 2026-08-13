# v3 Migration and Compatibility Map

| Legacy responsibility | v3 authority | Status |
|---|---|---|
| Per-generation controller and round state | `evolve.kernel` | deprecated; legacy remains read-only |
| Direct strategy model/evaluator calls | `evolve.runtime.ExecutionRuntime` | replaced |
| Per-loop evidence/catalog facts | `evolve.evidence.ReceiptStore` and `EvidenceGraph` | replaced for new campaigns |
| Round reports as truth | receipts + claims, projected by `evolve.reporting` | replaced |
| First-generation replay | `LegacyImportStrategy` | compatibility facade |
| Third-generation paired Skill A/B | `SkillPairedStrategy` | compatibility adapter |
| Second-generation tournaments | `AgentProgramSearchStrategy` | compatibility adapter |

The historical repository remains immutable and is referenced by content hash. No
legacy schema reader, sealed artifact, Catalog, review, or cost ledger was deleted.
Physical deletion from the legacy repository is deferred until semantic replay
equivalence is independently demonstrated.

