from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from observation_artifact import (
    ObservationArtifact,
    ObservationContractError,
    collect_observation,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/v1.0.0/v0.2.0-agent-program/runs/agent-program-final-pass3-3"
LENS = ROOT / "analysis/agent-baseline/agent_strategy.json"
CONFIG_HASHES = {"observer_experiment": "1" * 64}


@pytest.mark.parametrize(
    ("mode", "status"),
    [
        ("off", "disabled"),
        ("trace", "completed"),
        ("logit_lens", "completed"),
        ("jlens", "completed"),
    ],
)
def test_observation_artifacts_are_canonical_and_valid(mode, status):
    artifact = collect_observation(
        mode=mode,
        runtime_run_dir=RUN,
        lens_source=LENS,
        config_hashes=CONFIG_HASHES,
    )

    artifact.validate()
    assert artifact.status == status
    assert artifact.causal_boundary == "observational_not_causal"
    assert artifact.used_for_admission is False
    assert ObservationArtifact.from_dict(artifact.to_dict()) == artifact
    assert len(artifact.artifact_fingerprint) == 64


def test_off_has_no_features_and_trace_summarizes_the_runtime():
    off = collect_observation(
        mode="off",
        runtime_run_dir=RUN,
        lens_source=LENS,
        config_hashes=CONFIG_HASHES,
    )
    trace = collect_observation(
        mode="trace",
        runtime_run_dir=RUN,
        lens_source=LENS,
        config_hashes=CONFIG_HASHES,
    )

    assert off.features == {}
    assert off.source_refs == ()
    assert trace.features["public_evaluations"] == 12
    assert trace.features["sealed_evaluations"] == 6
    assert trace.features["accepted_candidates"] == 3
    assert trace.features["runtime_decision"] == "accepted"


def test_logit_and_jlens_use_the_same_source_with_distinct_metrics():
    logit = collect_observation(
        mode="logit_lens",
        runtime_run_dir=RUN,
        lens_source=LENS,
        config_hashes=CONFIG_HASHES,
    )
    jlens = collect_observation(
        mode="jlens",
        runtime_run_dir=RUN,
        lens_source=LENS,
        config_hashes=CONFIG_HASHES,
    )

    assert logit.source_refs == jlens.source_refs
    assert logit.features["score_eta_squared"] == pytest.approx(0.9970722769529216)
    assert jlens.features["score_eta_squared"] == pytest.approx(0.9953156431246746)
    assert jlens.features["jlens_incremental_supported"] is False
    assert jlens.features["unique_transitions"] == 7


def test_artifact_rejects_causal_claims_and_source_hash_drift():
    artifact = collect_observation(
        mode="trace",
        runtime_run_dir=RUN,
        lens_source=LENS,
        config_hashes=CONFIG_HASHES,
    )
    with pytest.raises(ObservationContractError, match="causal boundary"):
        replace(artifact, causal_boundary="causal").validate()

    source = artifact.source_refs[0]
    with pytest.raises(ObservationContractError, match="source sha256 mismatch"):
        replace(artifact, source_refs=(replace(source, sha256="0" * 64),)).validate()
