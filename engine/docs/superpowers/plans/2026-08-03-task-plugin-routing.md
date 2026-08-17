# Task-specific G2 Plugin Routing Implementation Plan

**Goal:** Replace unconditional monolithic G2 injection with a project-local, auditable task-family router and compare baseline, monolithic G2, and routed G2 on the frozen local benchmark.

**Architecture:** `multi_model_eval.py` remains the runner and admission-neutral evaluator. A strict suite contract maps a profile arm and task family to exactly one project-local plugin file. The runner compiles the base profile plus only the selected plugin, freezes each compiled task prompt hash before starting MLX, and preserves the existing behavior when no routing contract exists.

**Constraints:** Model weights frozen; no downloads; no auto-application; existing benchmark tasks are public diagnostic data, not sealed promotion evidence; routing paths must remain inside the project; missing or unknown routes fail before model calls.

---

1. Write the stage PLAN, task plugins, suite contract, and predeclared gates.
2. Add RED tests for route validation, one-plugin-only compilation, legacy compatibility, and frozen task-prompt hashes.
3. Implement the minimal router and integrate it into `run_suite` without changing graders.
4. Run targeted tests, full tests, lint, and format checks.
5. Run the local 2-model × 3-arm × 4-task diagnostic and preserve raw evidence.
6. Write Chinese RESULT/EXPERIMENT-DECISION/NEXT, update status/index/changelog, snapshot sources, rebuild and verify the v2 manifest.
