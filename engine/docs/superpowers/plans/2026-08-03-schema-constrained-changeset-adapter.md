# Schema-Constrained ChangeSet Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict project-local ChangeSet JSON adapter with at most one model repair, then measure it against the frozen raw-output control on the same local Qwen models and profiles.

**Architecture:** A new pure `changeset_adapter.py` owns the exported JSON Schema, deterministic validation, constrained first-pass prompt, and one bounded repair contract. `multi_model_eval.py` remains the execution/evaluator authority and records every raw attempt, usage, latency, adapter decision, and final grader result. Existing suites without `adapter_modes` retain their current raw-only behavior.

**Tech Stack:** Python 3.12+, dataclasses, built-in `json`, pytest, Ruff, local MLX OpenAI-compatible server.

---

### Task 1: Freeze the stage contract

**Files:**
- Create: `artifacts/v2.0.0/v2.0.0-meta-evolution/schema-adapter-001/PLAN.zh-CN.md`
- Create: `artifacts/v2.0.0/v2.0.0-meta-evolution/schema-adapter-001/configs/changeset-schema.json`
- Create: `artifacts/v2.0.0/v2.0.0-meta-evolution/schema-adapter-001/configs/eval-suite.json`

- [ ] **Step 1: Declare the only changed variable**

Record `raw` versus `schema_repair`, with identical model, profile, task, temperature, token cap, and grader.

- [ ] **Step 2: Freeze the capability gates**

Require no more than one repair; preserve all raw attempts; reject a second invalid output; keep `auto_apply=false`; require at least 3/4 schema-repair cells to be safe and schema-valid. Do not use this diagnostic for model or profile promotion.

- [ ] **Step 3: Hash before any model execution**

The run must save the suite, model registry, schema, profile hashes, and grader implementation hash before starting MLX.

### Task 2: Implement strict validation with TDD

**Files:**
- Create: `changeset_adapter.py`
- Create: `tests/test_changeset_adapter.py`

- [ ] **Step 1: Write failing tests**

Tests must cover exact keys, enum/constant fields, repeated-proposal hypothesis, matched A/B verification, terminal token removal, valid first-pass acceptance, one successful repair, and rejection after one failed repair.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_changeset_adapter.py -q`

Expected: collection fails because `changeset_adapter` does not exist.

- [ ] **Step 3: Implement the minimum pure adapter**

Expose `CHANGESET_SCHEMA`, `validate_changeset`, `build_constrained_prompt`, `build_repair_prompt`, and `adapt_changeset_response`. The repair function is injected; the adapter itself performs no network call and cannot apply a ChangeSet.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_changeset_adapter.py -q`

Expected: all adapter unit tests pass.

### Task 3: Integrate the frozen model matrix with TDD

**Files:**
- Modify: `multi_model_eval.py`
- Modify: `tests/test_multi_model_eval.py`

- [ ] **Step 1: Write failing integration tests**

Cover raw-only backward compatibility, adapter-mode expansion, token aggregation across repair calls, persisted attempt evidence, and adapter summary grouped independently from model/profile summary.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_multi_model_eval.py -q`

Expected: new adapter-mode assertions fail before integration.

- [ ] **Step 3: Implement the matrix integration**

Default old suites to one `raw` mode. For `schema_repair`, constrain the first prompt, invoke `_chat` at most once more after deterministic validation failure, canonicalize only an accepted object, and persist all attempts plus errors.

- [ ] **Step 4: Verify GREEN and regressions**

Run: `.venv/bin/python -m pytest tests/test_changeset_adapter.py tests/test_multi_model_eval.py tests/test_evolve_service.py -q`

Expected: all targeted tests pass.

### Task 4: Run the predeclared local diagnostic

**Files:**
- Create: `artifacts/v2.0.0/v2.0.0-meta-evolution/schema-adapter-001/runs/local-matrix-001/`

- [ ] **Step 1: Freeze contracts before loading models**

Run: `.venv/bin/python evolve_service.py benchmark-run --registry artifacts/v2.0.0/v2.0.0-meta-evolution/benchmark-suite-001/configs/model-registry.json --suite artifacts/v2.0.0/v2.0.0-meta-evolution/schema-adapter-001/configs/eval-suite.json --output artifacts/v2.0.0/v2.0.0-meta-evolution/schema-adapter-001/runs/local-matrix-001`

- [ ] **Step 2: Preserve negative outcomes**

Do not edit schema, grader, task, or threshold after observing outputs. Report first-pass, repaired, rejected, token, and latency results separately.

### Task 5: Package evidence and verify

**Files:**
- Create: `artifacts/v2.0.0/v2.0.0-meta-evolution/schema-adapter-001/RESULT.zh-CN.md`
- Create: `artifacts/v2.0.0/v2.0.0-meta-evolution/schema-adapter-001/EXPERIMENT-DECISION.json`
- Create: `artifacts/v2.0.0/v2.0.0-meta-evolution/schema-adapter-001/NEXT.zh-CN.md`
- Create: `artifacts/v2.0.0/v2.0.0-meta-evolution/schema-adapter-001/source-snapshot/`
- Modify: `artifacts/v2.0.0/INDEX.zh-CN.md`
- Modify: `STATUS.zh-CN.md`
- Modify: `HANDOFF.zh-CN.md`
- Modify: `CHANGELOG.md`
- Modify: `artifacts/v2.0.0/MANIFEST.json`

- [ ] **Step 1: Separate evidence classes**

`RESULT.zh-CN.md` must distinguish prior observation, deterministic schema intervention, model repair behavior, and the absence of sealed cross-task generalization.

- [ ] **Step 2: Snapshot source and regenerate hashes**

Copy only changed source/tests/configs into the stage source snapshot and regenerate the v2 manifest without overwriting old evidence.

- [ ] **Step 3: Run fresh completion verification**

Run the full pytest suite, Ruff check, Ruff format check, and artifact verifier. Record exact outputs; do not claim completion from earlier runs.

## Self-review

- Spec coverage: strict schema, one bounded repair, frozen matched control, raw evidence, token/latency, no auto-apply, and Chinese evidence packaging are each mapped to a task.
- Placeholder scan: no implementation placeholder or unbounded follow-up is part of this stage.
- Type consistency: the stage uses `adapter_id=schema-constrained-changeset-v1`, modes `raw/schema_repair`, and the existing `changeset-json-v1` grader throughout.
- Repository constraint: this directory is not a Git repository; source snapshots and SHA-256 manifests replace commit steps, matching the authoritative handoff.
