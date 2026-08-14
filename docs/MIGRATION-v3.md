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
| Local Qwen operator/span execution | `LegacyQwenPairTransport` behind `ExecutionRuntime` | live compatibility transport |
| Official SWE-bench invocation | `LegacyOfficialNativeEvaluator` behind `ExecutionRuntime` | live compatibility evaluator |
| Mutable task checkout selection | `FrozenSourceWorkspaceManager` | replaced with clean, exact-revision admission |
| Frozen taught Skill selection | `CandidateCompiler` + compiled Skill/Operator/Router | replaced; no live fallback |
| In-memory Teacher `_spent_cny` | `DurableCostLedger` | replaced for new calls and replay |
| Direct Capability append | `GovernanceService` + `PromotionDecisionLog` | rejected by authoritative registry |

## One live authority path

`fresh-feedback-e2e` composes the live path without adding a second Runtime,
budget, evaluator or registry authority. `LegacyQwenCellRunner` and
`LegacyOfficialNativeEvaluator` are adapters: they cannot create Claims, activate
Skills or bypass campaign authorization. `run_skill_paired_campaign` submits six
plans to the neutral Kernel and dispatches every plan through `ExecutionRuntime`.
The Runtime emits model, external trace, cost, native and terminal Receipts. The
Observer Hub emits Evidence; alignment and Claim Engine classify the three
matched pairs; registries accept only inactive revisions.

The Teacher request and response are copied byte-for-byte into a self-contained
compiled revision. The baseline branch does not read that revision; the taught
branch revalidates every compiled artifact and consumes its Skill, Operator and
Router. Strict native pairs become E2 claims. E3 additionally requires repeated
cross-project gains and matching external/JLens prediction evidence. Candidate,
Rejected and Capability registries are separate: only a decision-log-backed,
human-approved E3 decision may create a Capability, still inactive by default.

The CLI rejects non-feedback tasks, source revision drift, dirty checkouts,
evaluator/model hash drift, a config not bound to the current Git commit, and
receipt/manifest hash mismatch. `r076`, `r078`, fresh holdout and final-sealed
cohorts are outside this path and cannot be authorized by its configuration.

The historical repository remains immutable and is referenced by content hash. No
legacy schema reader, sealed artifact, Catalog, review, or cost ledger was deleted.
Physical deletion from the legacy repository is deferred until semantic replay
equivalence is independently demonstrated.
