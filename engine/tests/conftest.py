"""Engine test collection hook: skip data-dependent modules in a source clone.

Several engine tests verify against frozen round evidence stored in
``artifacts/`` (and ``state/`` / ``runs/``), which is intentionally excluded
from the git repository (it can be tens of GB).  In a fresh source clone those
tests cannot run, so the modules that are known to read that data are
auto-skipped rather than failing.  Keep the data present to exercise them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MISSING = tuple(
    name
    for name in ("artifacts", "state", "runs")
    if not (_ROOT / name).is_dir()
)

# Modules that read frozen round evidence and therefore need artifacts/state/runs.
_DATA_DEPENDENT_MODULES = frozenset(
    {
        "test_agent_code_mutation",
        "test_agent_code_runtime",
        "test_agent_program",
        "test_agent_program_runtime",
        "test_artifact_verifier",
        "test_backend_poc",
        "test_codex_changeset",
        "test_codex_evolution_runtime",
        "test_durable_service",
        "test_evaluator_shadow",
        "test_evolve_service",
        "test_failure_taxonomy",
        "test_governance",
        "test_integration_runtime",
        "test_meta_evolution_runtime",
        "test_observation_artifact",
        "test_observer_runtime",
        "test_psi_runtime",
        "test_release_candidate",
        "test_size_invariant",
        "test_skill_evolution_round1_feedback",
        "test_skill_evolution_target_audit",
        "test_supervised_evolution_poc",
        "test_taxonomy_extension",
        "test_v21_artifacts",
        "test_codex_target_runtime",
        "test_continuous_ab_smoke",
        "test_hardening_runtime",
        "test_multi_model_eval",
        "test_pilot_admission",
        "test_real_evolution_bridge",
        "test_release_verification",
        "test_skill_evolution_round1_run",
    }
)


def pytest_collection_modifyitems(config, items):  # noqa: ANN001
    if not _MISSING:
        return
    for item in items:
        module = item.getparent(pytest.Module)
        if module is None:
            continue
        name = Path(str(module.fspath)).stem
        if name in _DATA_DEPENDENT_MODULES:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"frozen evidence data ({', '.join(_MISSING)}) "
                    "is not present in this source clone"
                )
            )
