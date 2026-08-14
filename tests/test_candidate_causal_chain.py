from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from evolve.contracts import (
    Cohort,
    ContractViolation,
    ExecutionLimits,
    ExecutionPlan,
    ModelIdentity,
    TaskRevision,
)
from evolve.proposals import CandidateCompiler, CompiledRevision, CompileSpec
from evolve.runtime.candidate_prompt import CandidatePromptTransport


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _teacher_exchange(tmp_path: Path) -> tuple[Path, Path]:
    request = {
        "model": "deepseek-v4-flash",
        "messages": [
            {
                "role": "system",
                "content": "Return only an inactive external Agent capability.",
            },
            {
                "role": "user",
                "content": "Feedback-only typed zero-argument source rewrite.",
            },
        ],
        "max_tokens": 500,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    request_path = tmp_path / "source" / "TEACHER-REQUEST.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(request, sort_keys=True) + "\n")
    response = {
        "request_sha256": _sha_bytes(request_path.read_bytes()),
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "candidate_status": "inactive",
        "auto_activate": False,
        "candidate": {
            "protocol": "inactive_external_agent",
            "prompt_template": "Select the routed zero-argument operator.",
            "skill_text": "CAUSAL-SENTINEL: canonicalize field declarations.",
            "eval_note": "Validate with matched native feedback evidence.",
        },
        "usage": {
            "prompt_tokens": 112,
            "completion_tokens": 197,
            "total_tokens": 309,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 112,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        "estimated_cost_cny": 0.000506,
        "network_calls": 1,
        "elapsed_seconds": 2.67,
        "pricing_cny_per_million": {"input_cache_miss": 1.0, "output": 2.0},
    }
    response_path = tmp_path / "source" / "TEACHER-RESPONSE.json"
    response_path.write_text(json.dumps(response, sort_keys=True) + "\n")
    return request_path, response_path


def _plan(*, arm: str, candidate_revision_id: str) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id=f"plan-{arm}",
        campaign_id="campaign-1",
        strategy_id="skill-paired-v3",
        task=TaskRevision(
            task_id="sphinx-7757",
            revision_id="sphinx-7757-feedback-r1",
            project="sphinx",
            cohort=Cohort.FEEDBACK,
            source_sha256="a" * 64,
            evaluator_id="native-v1",
        ),
        candidate_revision_id=candidate_revision_id,
        arm=arm,
        model=ModelIdentity("local-mlx", "qwen3.5-4b", "frozen-r1"),
        context_policy_id="context-v3",
        tool_policy_id="tools-v3",
        observer_policy_ids=("native-v1",),
        native_evaluator_id="native-v1",
        limits=ExecutionLimits(256, 60, 0),
        holdout_scope="feedback-only",
    )


class RecordingPromptBackend:
    remote = False

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def infer(
        self,
        plan: ExecutionPlan,
        workspace: Mapping[str, Any],
        prompt: str,
    ) -> Mapping[str, Any]:
        self.prompts.append(prompt)
        return {
            "output": f"prediction:{plan.arm}",
            "cost_cny": 0,
            "input_tokens": 1,
            "output_tokens": 1,
        }


def _compile_spec() -> CompileSpec:
    return CompileSpec(
        candidate_id="candidate-symbol-field",
        revision_id="candidate-symbol-field-r2",
        parent_revision_id="prompt-baseline-r1",
        cohort=Cohort.FEEDBACK,
        operator_id="field_issue_hits",
        operator_instruction="Rewrite only the declared field symbol.",
        routes=(("sphinx-7757", "field_issue_hits"),),
    )


def test_teacher_change_set_compiles_to_immutable_lineage_and_only_taught_consumes_it(
    tmp_path: Path,
) -> None:
    request_path, response_path = _teacher_exchange(tmp_path)

    compiled = CandidateCompiler().compile(
        request_path=request_path,
        response_path=response_path,
        compile_spec=_compile_spec(),
        output_root=tmp_path / "compiled",
    )

    assert compiled == CompiledRevision.load(compiled.root)
    assert compiled.change_set.revision_id == "candidate-symbol-field-r2"
    assert compiled.skill.protocol == "inactive_external_agent"
    assert compiled.skill.prompt_template == (
        "Select the routed zero-argument operator."
    )
    assert compiled.skill.skill_text.startswith("CAUSAL-SENTINEL")
    assert compiled.operator.arguments == ()
    assert compiled.router.routes == (("sphinx-7757", "field_issue_hits"),)
    assert (compiled.root / "TEACHER-REQUEST.json").read_bytes() == (
        request_path.read_bytes()
    )
    assert (compiled.root / "TEACHER-RESPONSE.json").read_bytes() == (
        response_path.read_bytes()
    )

    manifest = json.loads(compiled.manifest_path.read_text())
    assert manifest["request_sha256"] == _sha_bytes(request_path.read_bytes())
    assert manifest["response_sha256"] == _sha_bytes(response_path.read_bytes())
    assert manifest["provider"] == "deepseek"
    assert manifest["model"] == "deepseek-v4-flash"
    assert manifest["cost_cny"] == 0.000506
    assert (compiled.root / "MODEL-RECEIPT.json").is_file()
    cost_receipt = json.loads((compiled.root / "COST-RECEIPT.json").read_text())
    assert cost_receipt["cost_kind"] == "estimated"
    assert cost_receipt["cost_cny"] == 0.000506
    for artifact in manifest["artifacts"]:
        artifact_path = compiled.root / artifact["path"]
        assert artifact_path.is_file()
        assert _sha_bytes(artifact_path.read_bytes()) == artifact["sha256"]
    frozen_before = {path.name: path.read_bytes() for path in compiled.root.iterdir()}
    replay = CandidateCompiler().compile(
        request_path=request_path,
        response_path=response_path,
        compile_spec=_compile_spec(),
        output_root=tmp_path / "compiled",
    )
    assert replay == compiled
    assert {
        path.name: path.read_bytes() for path in compiled.root.iterdir()
    } == frozen_before

    backend = RecordingPromptBackend()
    transport = CandidatePromptTransport(
        backend=backend,
        compiled=compiled,
        base_prompt_builder=lambda plan, _workspace: f"Fix {plan.task.task_id}",
    )
    baseline = transport.infer(
        _plan(arm="baseline", candidate_revision_id="prompt-baseline-r1"), {}
    )
    taught = transport.infer(
        _plan(
            arm="taught",
            candidate_revision_id=compiled.change_set.revision_id,
        ),
        {},
    )

    assert "CAUSAL-SENTINEL" not in backend.prompts[0]
    assert baseline["candidate_consumed"] is False
    assert baseline["candidate_bundle_sha256"] is None
    assert "CAUSAL-SENTINEL" in backend.prompts[1]
    assert taught["candidate_consumed"] is True
    assert taught["candidate_bundle_sha256"] == compiled.bundle_sha256
    assert taught["candidate_revision_id"] == compiled.change_set.revision_id


def test_compiler_rejects_malformed_teacher_pricing_before_writing_artifacts(
    tmp_path: Path,
) -> None:
    request_path, response_path = _teacher_exchange(tmp_path)
    response = json.loads(response_path.read_text())
    response["pricing_cny_per_million"]["input_cache_miss"] = "unknown"
    response_path.write_text(json.dumps(response, sort_keys=True) + "\n")
    output_root = tmp_path / "compiled"

    with pytest.raises(ContractViolation, match="pricing"):
        CandidateCompiler().compile(
            request_path=request_path,
            response_path=response_path,
            compile_spec=_compile_spec(),
            output_root=output_root,
        )

    assert not output_root.exists()


def test_only_teacher_skill_change_changes_taught_prompt_and_hash(
    tmp_path: Path,
) -> None:
    request_a, response_a = _teacher_exchange(tmp_path / "a")
    request_b, response_b = _teacher_exchange(tmp_path / "b")
    changed = json.loads(response_b.read_text())
    changed["candidate"]["skill_text"] = (
        "CAUSAL-SENTINEL-B: preserve declared field ownership."
    )
    response_b.write_text(json.dumps(changed, sort_keys=True) + "\n")
    compiler = CandidateCompiler()
    compiled_a = compiler.compile(
        request_path=request_a,
        response_path=response_a,
        compile_spec=_compile_spec(),
        output_root=tmp_path / "compiled-a",
    )
    compiled_b = compiler.compile(
        request_path=request_b,
        response_path=response_b,
        compile_spec=_compile_spec(),
        output_root=tmp_path / "compiled-b",
    )
    backend_a = RecordingPromptBackend()
    backend_b = RecordingPromptBackend()
    transport_a = CandidatePromptTransport(
        backend=backend_a,
        compiled=compiled_a,
        base_prompt_builder=lambda plan, _workspace: f"Fix {plan.task.task_id}",
    )
    transport_b = CandidatePromptTransport(
        backend=backend_b,
        compiled=compiled_b,
        base_prompt_builder=lambda plan, _workspace: f"Fix {plan.task.task_id}",
    )
    baseline_plan = _plan(arm="baseline", candidate_revision_id="prompt-baseline-r1")
    taught_plan = _plan(arm="taught", candidate_revision_id=_compile_spec().revision_id)

    baseline_a = transport_a.infer(baseline_plan, {})
    taught_a = transport_a.infer(taught_plan, {})
    baseline_b = transport_b.infer(baseline_plan, {})
    taught_b = transport_b.infer(taught_plan, {})

    assert backend_a.prompts[0] == backend_b.prompts[0]
    assert baseline_a["prompt_sha256"] == baseline_b["prompt_sha256"]
    assert backend_a.prompts[1] != backend_b.prompts[1]
    assert taught_a["prompt_sha256"] != taught_b["prompt_sha256"]
    assert taught_a["candidate_bundle_sha256"] != taught_b["candidate_bundle_sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("empty-skill", "non-empty"),
        ("candidate-list", "candidate fields"),
        ("bad-request-hash", "request hash"),
    ],
)
def test_empty_or_bad_teacher_response_fails_before_compilation(
    tmp_path: Path, mutation: str, message: str
) -> None:
    request_path, response_path = _teacher_exchange(tmp_path)
    response = json.loads(response_path.read_text())
    if mutation == "empty-skill":
        response["candidate"]["skill_text"] = ""
    elif mutation == "candidate-list":
        response["candidate"] = []
    else:
        response["request_sha256"] = "0" * 64
    response_path.write_text(json.dumps(response, sort_keys=True) + "\n")
    output_root = tmp_path / "compiled"
    backend = RecordingPromptBackend()

    with pytest.raises(ContractViolation, match=message):
        CandidateCompiler().compile(
            request_path=request_path,
            response_path=response_path,
            compile_spec=_compile_spec(),
            output_root=output_root,
        )

    assert not output_root.exists()
    assert backend.prompts == []


def test_tampered_or_missing_compiled_revision_fails_before_backend_call(
    tmp_path: Path,
) -> None:
    request_path, response_path = _teacher_exchange(tmp_path)
    compiled = CandidateCompiler().compile(
        request_path=request_path,
        response_path=response_path,
        compile_spec=_compile_spec(),
        output_root=tmp_path / "compiled",
    )
    taught_plan = _plan(
        arm="taught", candidate_revision_id=compiled.change_set.revision_id
    )
    backend = RecordingPromptBackend()
    missing_transport = CandidatePromptTransport(
        backend=backend,
        compiled=None,
        base_prompt_builder=lambda _plan, _workspace: "base prompt",
    )

    with pytest.raises(ContractViolation, match="fallback is forbidden"):
        missing_transport.infer(taught_plan, {})
    assert backend.prompts == []

    compiled_skill = compiled.root / "COMPILED-SKILL.json"
    compiled_skill.write_text("tampered", encoding="utf-8")
    tampered_transport = CandidatePromptTransport(
        backend=backend,
        compiled=compiled,
        base_prompt_builder=lambda _plan, _workspace: "base prompt",
    )
    baseline = tampered_transport.infer(
        _plan(arm="baseline", candidate_revision_id="prompt-baseline-r1"), {}
    )
    assert baseline["candidate_consumed"] is False
    calls_before_taught = len(backend.prompts)
    with pytest.raises(ContractViolation, match="hash mismatch"):
        tampered_transport.infer(taught_plan, {})
    assert len(backend.prompts) == calls_before_taught
