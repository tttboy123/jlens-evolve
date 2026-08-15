# v3 Migration and Compatibility Map

| Legacy responsibility | v3 authority | Status |
|---|---|---|
| Per-generation controller and round state | `evolve.kernel` | deprecated; legacy remains read-only |
| Direct strategy model/evaluator calls | `evolve.runtime.ExecutionRuntime` | replaced |
| Per-loop evidence/catalog facts | `evolve.evidence.ReceiptStore` and `EvidenceGraph` | replaced for new campaigns |
| Round reports as truth | receipts + claims, projected by `evolve.reporting` | replaced |
| First-generation replay | `LegacyImportStrategy` | read-only compatibility facade; LIVE for import only |
| Third-generation paired Skill A/B | `SkillPairedStrategy` | LIVE native campaign |
| Second-generation tournaments | `AgentProgramSearchStrategy` | fixture compatibility + allowlisted public non-fixture LIVE profile |
| Cross-strategy capability-gap scheduling | Portfolio Orchestrator + `CapabilityGap` | minimal LIVE product seam; no automatic activation |
| Local Qwen operator/span execution | `LegacyQwenPairTransport` behind `ExecutionRuntime` | live compatibility transport |
| Official SWE-bench invocation | `LegacyOfficialNativeEvaluator` behind `ExecutionRuntime` | live compatibility evaluator |
| Mutable task checkout selection | `FrozenSourceWorkspaceManager` | replaced with clean, exact-revision admission |
| Frozen taught Skill selection | `CandidateCompiler` + compiled Skill/Operator/Router | replaced; no live fallback |
| In-memory Teacher `_spent_cny` | chained `DurableCostLedger` + head anchor | replaced for new calls and replay |
| Post-hoc internal-effect target | pre-dispatch `MechanismPrediction` Receipt | replaced for trusted JLens campaigns |
| Self-described JLens evidence | signed observation + exact Receipt Store replay | rejected for E3 |
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
Router. Strict native pairs become E2 only after `MatchedCounterfactualPair`
binds the Candidate revision/bundle and each arm's model receipt, external trace
and native outcome. Old unbound E2 records require an explicit legacy-read flag
and cannot be minted by the new Claim path. E3 additionally requires repeated
cross-project gains and a cryptographically attested independent JLens
observation bound to the same model receipt/artifact as external and native
evidence. The expected internal effect is frozen as a MechanismPrediction receipt
before model dispatch, so it cannot be selected after seeing the trace. E3
projection first rebuilds every counterfactual pair and then replays the exact
prediction, model and trusted-observation receipts from the Receipt Store. It
rejects missing, reordered, renamed, wrong-kind or artifact-drifted receipts.
Generic legacy JLens traces remain readable but are never E3 eligible.
Candidate, Rejected and Capability registries are separate: neutral remains
`no_change`, infra remains `blocked`, regression is `rejected`, and only a
decision-log-backed, human-approved E3 decision signed by the configured
Governance authority may create a Capability, still inactive by default. The
Capability Registry accepts only the concrete verifying decision log, not a
duck-typed reader returning hand-built approvals.

For autonomous Skill evolution, the next baseline parent is authorized by the
latest `accepted_as_best` decision in a sealed round whose manifest and
hash-chained round-index entry verify. `best/BEST-HARNESS.json` is a reloadable,
hash-verified projection exported from that accepted round. It is not a second
registry, cannot make an unaccepted candidate authoritative, and cannot
override the sealed round during resume.

The CLI rejects non-feedback tasks, source revision drift, dirty checkouts,
evaluator/model hash drift, a config not bound to the current Git commit, and
receipt/manifest hash mismatch. `r076`, `r078`, fresh holdout and final-sealed
cohorts are outside this path and cannot be authorized by its configuration.
The optional `trusted_jlens` config reads canonical per-plan traces from an
external trace root and takes its signing secret only from a named environment
variable. If absent, the product path remains explicitly E2. Budget validity is
likewise authority-derived: only a successfully replayed DurableCostLedger can
project `validated`; report metadata alone projects no such claim.

The historical repository remains immutable and is referenced by content hash. No
legacy schema reader, sealed artifact, Catalog, review, or cost ledger was deleted.
Physical deletion from the legacy repository is deferred until semantic replay
equivalence is independently demonstrated.

## Current strategy boundary

- **Skill:** LIVE for feedback-only matched baseline/taught Qwen execution and
  official native evaluation.
- **Legacy:** LIVE only as a read-only compatibility import. It does not mint
  current native Claims or become the autonomous parent authority.
- **AgentProgram:** the public CLI supports the deterministic fixture profile and
  one allowlisted local non-fixture profile. The live path verifies complete
  Program revisions, replays Runtime Receipt/Evidence authorities and derives a
  hash-bound tournament decision. It cannot dynamically import an executor,
  mint Claims, promote or activate.
- **Portfolio closure:** `PortfolioOrchestrator` implements one bounded product
  path from an authoritative AgentProgram failure to CapabilityGap, inactive
  Skill, signed Governance-approved inactive Capability, complete AgentProgram
  revision and live tournament. Broader portfolio optimization remains future
  work; this seam is deliberately not a second Campaign or Governance authority.
