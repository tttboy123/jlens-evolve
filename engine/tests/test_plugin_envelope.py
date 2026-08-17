from __future__ import annotations

from pathlib import Path

import pytest

from plugin_envelope import ArtifactRef, EnvelopeContractError, PluginEnvelope


def _ref(path: Path, role: str) -> ArtifactRef:
    path.write_text("{}\n", encoding="utf-8")
    return ArtifactRef.from_path(path, role=role)


def test_observer_envelope_cannot_gain_admission_or_active_authority(tmp_path: Path):
    result_ref = _ref(tmp_path / "result.json", "observer_result")

    with pytest.raises(EnvelopeContractError, match="observe"):
        PluginEnvelope.create(
            operation_id="op-1",
            plugin_id="observer",
            plugin_version="0.3.0",
            authority="observe",
            status="completed",
            input_hashes={"input": "a" * 64},
            config_hashes={"config": "b" * 64},
            candidate_ref=None,
            active_ref=result_ref,
            result_refs=(result_ref,),
            evidence_refs=(result_ref,),
            used_for_admission=False,
            error=None,
        )
    with pytest.raises(EnvelopeContractError, match="observe"):
        PluginEnvelope.create(
            operation_id="op-2",
            plugin_id="observer",
            plugin_version="0.3.0",
            authority="observe",
            status="completed",
            input_hashes={"input": "a" * 64},
            config_hashes={"config": "b" * 64},
            candidate_ref=None,
            active_ref=None,
            result_refs=(result_ref,),
            evidence_refs=(result_ref,),
            used_for_admission=True,
            error=None,
        )


def test_only_admission_can_publish_active_ref(tmp_path: Path):
    candidate = _ref(tmp_path / "candidate.json", "candidate")
    active = _ref(tmp_path / "active.json", "active")
    evidence = _ref(tmp_path / "evidence.json", "evidence")
    envelope = PluginEnvelope.create(
        operation_id="op",
        plugin_id="admission-gate",
        plugin_version="0.7.0",
        authority="admit",
        status="completed",
        input_hashes={"input": "a" * 64},
        config_hashes={"config": "b" * 64},
        candidate_ref=candidate,
        active_ref=active,
        result_refs=(active,),
        evidence_refs=(evidence,),
        used_for_admission=True,
        error=None,
    )

    assert PluginEnvelope.from_dict(envelope.to_dict()) == envelope
    assert len(envelope.envelope_fingerprint) == 64


def test_artifact_ref_detects_post_creation_tamper(tmp_path: Path):
    path = tmp_path / "evidence.json"
    ref = _ref(path, "evidence")
    path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(EnvelopeContractError, match="sha256 mismatch"):
        ref.validate()
