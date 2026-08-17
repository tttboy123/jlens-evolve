from pathlib import Path


def test_checkpoint_stop_never_supports_instance_termination() -> None:
    script = (
        Path(__file__).parents[1] / "deploy/tencent/cuda-worker/checkpoint-stop.sh"
    ).read_text(encoding="utf-8")

    assert "FINAL_ACTION=${FINAL_ACTION:-stop}" in script
    assert 'if [[ "$FINAL_ACTION" != stop ]]; then' in script
    assert "StopInstances" in script
    assert "--StoppedMode STOP_CHARGING" in script
    assert "TerminateInstances" not in script
    assert "--DisableApiTermination false" not in script
