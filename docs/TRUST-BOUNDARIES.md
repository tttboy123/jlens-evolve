# v3 Trust Boundaries

This document defines the integrity model for new v3 campaigns. It prevents
product code from gradually treating every writable local file as a remotely
attested fact, while keeping corruption, replay, leakage and authority failures
fail-closed.

## In scope

The platform assumes the checked-out v3 source, the current OS account, configured
local signing keys, and the process that owns a campaign are trusted. Within that
boundary it defends against:

- malformed or adversarial Model and Teacher output;
- task, model, Candidate, evaluator, cohort, plan and parent-lineage drift;
- partial writes, duplicate dispatch, stale resume and conflicting concurrent
  writers;
- missing, reordered, hash-drifted or cross-plan Receipts and Evidence;
- self-reported internal observations, direct Skill activation and promotion
  without Governance authority;
- holdout, final-sealed, burned `r076` and burned `r078` admission.

## Authority boundaries

| Boundary | What it guarantees | What it does not guarantee |
|---|---|---|
| Frozen source identity | A run is bound to the declared Git revision or explicitly frozen snapshot and to revision-pinned task checkouts. | The truth of a remote repository or protection after a privileged actor replaces both source and its identity record. |
| `ExecutionRuntime` | Authorized model, workspace and native-evaluator dispatch uses one ordered Receipt path; terminal replay does not repeat completed work. | Prevention of arbitrary programs outside the product from invoking shell, Docker or APIs directly. |
| `ReceiptStore` / ledgers | Single-writer admission, append semantics, canonical payload hashes, content-addressed artifacts, duplicate detection and restart replay. | Tamper-proof storage against an actor that can rewrite the log, all artifacts and the running verifier code. Use read-only media or external notarization for that threat. |
| Observer / Evidence Graph | Evidence is deterministically projected from admitted Receipts; Claims bind the required evidence and counterfactual lineage. | Independent truth when the underlying trusted Receipt authority itself has been replaced. E3 additionally requires a trusted Observer attestation. |
| Native evaluator receipt | The outcome is bound to the exact plan, model artifact, evaluator identity and official native report admitted by Runtime. | A guarantee that an administrator did not replace the evaluator implementation and every identity file before execution. |
| Governance signature | Only the configured Governance authority can approve an E3-backed inactive Capability projection. | Security after disclosure or replacement of the signing secret; secret custody belongs to the host boundary. |
| Evidence manifest | Every sealed artifact is present and byte-equal to its literal SHA-256; replay detects accidental or partial drift. | Authenticity by itself. A party able to rewrite all artifacts and re-seal the manifest is outside this local-manifest threat model. |

## Operational consequence

Tests keep one representative happy path, tamper path and resume path for each
core invariant. Coordinated full-filesystem rewrite variants are not multiplied
field by field. If protection from a malicious local administrator is required,
the release must add an external transparency log, hardware-backed signature or
immutable object-store retention rather than more self-signed local manifests.

The current release remains feedback-only. Skill, Capability and AgentProgram
assets are inactive by default; BEST is a verified search-parent projection, not
a production activation pointer.
