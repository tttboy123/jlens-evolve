# Minimax Autonomous Evolution Output Contract

This contract is the single source of truth for Minimax output and Codex review
artifacts. It applies whenever Minimax starts or resumes autonomous evolution.

## Stable root

All coordination artifacts live under:

`runs/skill-evolution-loop/autonomous/`

Evidence may remain in the existing `artifacts/` and evolution-catalog trees.
Coordination documents point to that evidence with project-relative paths and
SHA-256 hashes; they do not duplicate it.

## Ownership

Minimax is the only writer of:

- `AUTO-EVOLUTION-STATE.md`
- `ROUND-INDEX.jsonl`
- `rounds/<round-id>/`
- evolution-catalog records created by its run

Codex review is the only writer of:

- `CODEX-REVIEW-STATE.json`
- `LATEST-CODEX-REVIEW.md`
- `reviews/<review-id>.md`

Each writer uses an atomic temporary-file rename for mutable JSON/Markdown.
Minimax appends a round to `ROUND-INDEX.jsonl` only after all required round
files and catalog records are durable. An indexed round is immutable.

## Minimax startup

1. Read `AUTO-EVOLUTION-STATE.md`, `ROUND-INDEX.jsonl`, and the latest Codex
   review named in `LATEST-CODEX-REVIEW.md`.
2. Resolve every required action in that review or record why a different
   evidence-backed priority is safer.
3. Create a UTC round ID in the form `YYYYMMDDTHHMMSSZ-<short-slug>`.
4. Work only inside the declared round scope and project guardrails.

Startup is complete when the new round ID and falsifiable hypothesis are
written to `AUTO-EVOLUTION-STATE.md`.

## Required files for every finalized round

Directory:

`runs/skill-evolution-loop/autonomous/rounds/<round-id>/`

Files:

### `ROUND-REPORT.json`

Machine-readable object with these required keys:

```json
{
  "schema_version": 1,
  "round_id": "YYYYMMDDTHHMMSSZ-short-slug",
  "started_at": "RFC3339 UTC",
  "ended_at": "RFC3339 UTC",
  "status": "validated_gain | validated_neutral | disproven | blocked | regression",
  "hypothesis": {
    "cause": "specific claimed cause",
    "change": "smallest tested change",
    "expected_observation": "falsifiable expected result",
    "success_criteria": ["checkable criterion"],
    "failure_criteria": ["checkable criterion"]
  },
  "scope": {
    "feedback_task_ids": [],
    "holdout_opened": false,
    "frozen_core_changed": false
  },
  "changed_files": [],
  "verification": [
    {
      "command": "exact command",
      "exit_code": 0,
      "result": "exact counts or outcome",
      "output_path": "project-relative path",
      "output_sha256": "64 lowercase hex characters"
    }
  ],
  "generation": {
    "provider": "provider or local",
    "model": "exact model identity",
    "temperature": 0,
    "seeds": [],
    "sample_count": 0,
    "vote_counts": {}
  },
  "native_evaluation": {
    "baseline_outcome": "resolved | unresolved | structural_invalid | no_op | not_run",
    "taught_outcome": "resolved | unresolved | structural_invalid | no_op | not_run",
    "evaluator_failure_count": 0,
    "report_paths": []
  },
  "guardrails": {
    "skill_auto_activated": false,
    "catalog_single_writer": true,
    "holdout_leakage_detected": false,
    "paid_or_remote_action_authorized": false
  },
  "cost": {
    "teacher_tokens": 0,
    "student_tokens": 0,
    "estimated_cost_cny": 0
  },
  "catalog_record_ids": [],
  "evidence_manifest_path": "project-relative path",
  "evidence_manifest_sha256": "64 lowercase hex characters",
  "next_hypothesis": "next evidence-backed hypothesis"
}
```

Use explicit `not_run` and zero values; do not omit failed or unavailable
measurements. A claim is valid only when its exact verifier appears in
`verification` or `native_evaluation`.

### `EVIDENCE-MANIFEST.json`

Machine-readable object:

```json
{
  "schema_version": 1,
  "round_id": "same round ID",
  "entries": [
    {
      "kind": "test_output | patch | native_report | model_receipt | catalog_record | other",
      "path": "project-relative path",
      "sha256": "64 lowercase hex characters"
    }
  ]
}
```

Every evidence path must exist when the round is indexed. The manifest must
include the catalog records and every file used to justify the round status.

### `ROUND-REPORT.md`

Human-readable report with exactly these headings:

1. `# Round <round-id>`
2. `## Verdict`
3. `## Hypothesis`
4. `## Changes`
5. `## Verification`
6. `## Native evidence`
7. `## Guardrails and cost`
8. `## Failures and uncertainty`
9. `## Next round`

It summarizes `ROUND-REPORT.json`; the JSON remains authoritative.

## Round index

After finalization, append one canonical JSON object as one line to
`ROUND-INDEX.jsonl`:

```json
{"schema_version":1,"round_id":"...","ended_at":"...","status":"...","report_path":"runs/skill-evolution-loop/autonomous/rounds/.../ROUND-REPORT.json","report_sha256":"...","manifest_path":"runs/skill-evolution-loop/autonomous/rounds/.../EVIDENCE-MANIFEST.json","manifest_sha256":"..."}
```

Round IDs are unique, ordered by completion time, and never rewritten.

## Live state

`AUTO-EVOLUTION-STATE.md` is a compact resume index with exactly these
headings:

- `# Auto Evolution State`
- `## Long-term gate`
- `## Current round`
- `## Last finalized round`
- `## Latest Codex review`
- `## Highest-priority failure clusters`
- `## Next hypothesis`
- `## Blockers requiring user input`

Keep paths and counts here; detailed evidence belongs in immutable round files.

## Codex review output

Codex reviews every newly indexed round since
`CODEX-REVIEW-STATE.json:last_reviewed_round_id`. A review verifies referenced
file existence and hashes, result/claim consistency, frozen-core and holdout
boundaries, regression coverage, native evidence, catalog ownership, and cost.

Review IDs use `YYYYMMDDTHHMMSSZ-codex-review`. Each review is written to
`reviews/<review-id>.md` with these headings:

- `# Codex Evolution Review`
- `## Review scope`
- `## Verdict`
- `## Evidence integrity`
- `## Guardrail audit`
- `## Findings`
- `## Required actions before the next risky step`
- `## Recommended next hypothesis`

Findings use severity `P0`, `P1`, or `P2` and cite exact paths. `P0` blocks
further work; `P1` blocks promotion, holdout opening, or a completion claim;
`P2` is follow-up debt.

If no new round is indexed, Codex updates only
`CODEX-REVIEW-STATE.json:last_checked_at` and reports `no new finalized round`.

## Completion criteria

A Minimax round is complete only when all three required files exist, their
hashes agree with the index, every claimed verifier has fresh evidence, the
catalog append is durable, `AUTO-EVOLUTION-STATE.md` points to the finalized
round, and the index line is appended last.

A Codex review is complete only when every newly indexed round is accounted
for, every manifest entry is checked, findings cite evidence, the immutable
review file is written, `LATEST-CODEX-REVIEW.md` points to it, and
`CODEX-REVIEW-STATE.json` advances through the reviewed round.
