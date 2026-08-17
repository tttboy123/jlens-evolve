from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from dataclasses import replace
from pathlib import Path

from skill_evolution_loop.contracts import LoopRevision
from skill_evolution_loop.round1_realization import (
    Round1SharedRealizationAdapter,
    _issue_static_span_targets,
    build_round1_realization_adapter,
    build_round1_shared_realization_adapter,
    extract_round1_frozen_diagnosis,
)
from skill_evolution_loop.span_student import _editable_span_task
from skill_evolution_loop.student_adapter import (
    StudentAdapter,
    StudentAttempt,
    StudentTask,
)


def _fixture(tmp_path: Path) -> tuple[StudentTask, LoopRevision]:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "example.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    task = StudentTask.create(
        task_id="round1-realization-fixture",
        checkout=checkout,
        instruction="Value should be two.",
        allowed_targets=["example.py"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="round1-realization",
        revision_id="round1-realization-r011",
        parent_revision_id=None,
        source_round=11,
        protocol="python-typed-operator-plan-v1",
        skill_text="Use one supported source-anchored operation.",
        prompt_template="Return one plan.",
        eval_note="fixture",
    )
    return task, revision


def _raw_plan(*, boundary: str = "value equals two") -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "file": "example.py",
            "symbol": "module",
            "intent": {
                "defect": "value equals one",
                "trigger": "module loads",
                "desired_boundary": boundary,
            },
            "operations": [
                {
                    "operator": "replace_constant",
                    "selector": {"value": 1, "occurrence": 0},
                    "arguments": {"new_value": 2},
                }
            ],
            "diagnostic": "replace value",
        }
    )


def _attempt(
    task: StudentTask, revision: LoopRevision, raw: str, patch: str
) -> StudentAttempt:
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()  # noqa: E731
    return StudentAttempt(
        task=task,
        revision_id=revision.revision_id,
        raw_output=raw,
        raw_output_sha256=digest(raw),
        edit=None,
        patch=patch,
        patch_sha256=digest(patch),
        target_file="example.py",
        before_sha256="a" * 64,
        after_sha256="b" * 64,
        implementation_fingerprint="c" * 64,
        structural_valid=True,
        failure_reason=None,
        detail="fixture",
    )


def test_extract_round1_diagnosis_uses_plan_intent_or_issue_fallback(
    tmp_path: Path,
) -> None:
    task, _revision = _fixture(tmp_path)

    parsed = extract_round1_frozen_diagnosis(
        mechanism="operator", raw_output=_raw_plan(), task=task
    )
    fallback = extract_round1_frozen_diagnosis(
        mechanism="operator", raw_output="not-json", task=task
    )

    assert parsed.desired_boundary == "value equals two"
    assert fallback.defect == "Value should be two."


def test_issue_fallback_freezes_public_expected_behavior_as_desired_boundary(
    tmp_path: Path,
) -> None:
    task, _revision = _fixture(tmp_path)
    task = StudentTask.create(
        task_id=task.task_id,
        checkout=task.checkout,
        instruction=(
            "autodoc: empty __all__ attribute is ignored\n\n"
            "**To Reproduce**\n"
            "Define `__all__ = []` and enable `:members:`.\n\n"
            "**Expected behavior**\n"
            "No entries should be shown because `__all__` is empty.\n\n"
            "**Environment info**\nPython 3.9\n"
        ),
        allowed_targets=task.allowed_targets,
        cohort=task.cohort,
    )

    diagnosis = extract_round1_frozen_diagnosis(
        mechanism="operator", raw_output="not-json", task=task
    )

    assert diagnosis.desired_boundary == (
        "No entries should be shown because `__all__` is empty."
    )


def test_issue_fallback_ignores_no_response_placeholder(tmp_path: Path) -> None:
    task, _revision = _fixture(tmp_path)
    task = StudentTask.create(
        task_id=task.task_id,
        checkout=task.checkout,
        instruction=(
            "Inline code highlighting adds whitespace at both boundaries.\n\n"
            "### Describe the bug\n"
            "A space is inserted at the start and end of inline code.\n\n"
            "### Expected behavior\n\n_No response_\n"
        ),
        allowed_targets=task.allowed_targets,
        cohort=task.cohort,
    )

    diagnosis = extract_round1_frozen_diagnosis(
        mechanism="operator", raw_output="not-json", task=task
    )

    assert diagnosis.desired_boundary == (
        "satisfy the task while preserving unrelated behavior"
    )


def test_round1_realization_wrapper_pins_and_rejects_diagnosis_drift(
    tmp_path: Path,
) -> None:
    task, revision = _fixture(tmp_path)
    patch = "--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"

    class BaseAdapter:
        def __init__(self) -> None:
            self.revisions: list[LoopRevision] = []
            self.generator = self

        def experiment_config(self):
            return {"adapter": "fixture"}

        def run(self, candidate_task, candidate_revision):
            self.revisions.append(candidate_revision)
            raw = _raw_plan(
                boundary=(
                    "value equals two"
                    if len(self.revisions) == 1
                    else "value equals three"
                )
            )
            return _attempt(candidate_task, candidate_revision, raw, patch)

    base = BaseAdapter()
    adapter = build_round1_realization_adapter(
        mechanism="operator", base_adapter=base, maximum_candidates=2
    )

    selected = adapter.run(task, revision)

    assert selected.raw_output == _raw_plan()
    assert "## Frozen diagnosis (read-only)" in base.revisions[1].skill_text
    assert "Realization candidate: 2" in base.revisions[1].skill_text
    decisions = adapter.realization_evidence()["selection"]["candidate_decisions"]
    assert [row["status"] for row in decisions] == [
        "selected",
        "structural-invalid",
    ]


def test_round1_shared_realization_freezes_neutral_diagnosis_and_localization(
    tmp_path: Path,
) -> None:
    task, baseline = _fixture(tmp_path)
    patch = "--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"

    class BaseAdapter:
        def __init__(self) -> None:
            self.revisions: list[LoopRevision] = []
            self.generator = self

        def experiment_config(self):
            return {"adapter": "fixture"}

        def run(self, candidate_task, candidate_revision):
            self.revisions.append(candidate_revision)
            return _attempt(candidate_task, candidate_revision, _raw_plan(), patch)

        def generation_prompt_trace(self):
            return (f"prompt for call {len(self.revisions)}",)

    base = BaseAdapter()
    adapter = build_round1_shared_realization_adapter(
        mechanism="operator", base_adapter=base, maximum_candidates=1
    )
    config = adapter.experiment_config()

    shared = adapter.prepare_shared_context(task, baseline)
    adapter.bind_shared_context(shared)
    baseline_attempt = adapter.run(task, baseline)
    taught = LoopRevision.create(
        skill_id=baseline.skill_id,
        revision_id="round1-realization-taught-r012",
        parent_revision_id=baseline.revision_id,
        source_round=12,
        protocol=baseline.protocol,
        skill_text="Use the inactive teaching Skill.",
        prompt_template=baseline.prompt_template,
        eval_note=baseline.eval_note,
    )
    adapter.bind_shared_context(shared)
    taught_attempt = adapter.run(task, taught)

    assert (
        config["implementation_sha256"]
        == hashlib.sha256(
            Path(inspect.getfile(Round1SharedRealizationAdapter)).read_bytes()
        ).hexdigest()
    )
    assert shared["contract"] == "shared-diagnosis-localization-v1"
    assert shared["diagnosis"]["desired_boundary"] == "value equals two"
    assert shared["target_files"] == ["example.py"]
    assert shared["target_symbol"] == "module"
    assert shared["native_labels_visible"] is False
    assert shared["reference_patch_visible"] is False
    assert base.revisions[0].skill_text.startswith("No additional domain teaching")
    assert all(
        "## Shared diagnosis and localization (read-only)" in revision.skill_text
        for revision in base.revisions[1:]
    )
    assert baseline_attempt.structural_valid is True
    assert taught_attempt.structural_valid is True
    assert adapter.generation_prompt_trace() == ("prompt for call 3",)
    assert (
        adapter.realization_evidence()["diagnosis"]["fingerprint"]
        == (shared["diagnosis"]["fingerprint"])
    )


def test_round1_shared_realization_preserves_inner_repair_trace(tmp_path: Path) -> None:
    task, revision = _fixture(tmp_path)

    class BaseAdapter:
        def __init__(self) -> None:
            self.generator = self
            self.calls = 0

        @staticmethod
        def experiment_config():
            return {"adapter": "fixture"}

        def run(self, candidate_task, candidate_revision):
            self.calls += 1
            if self.calls == 1:
                return StudentAdapter._failure(
                    candidate_task,
                    candidate_revision,
                    "neutral-seed",
                    "unresolved",
                    "neutral seed unresolved",
                )
            return StudentAdapter._failure(
                candidate_task,
                candidate_revision,
                "final-unresolved",
                "unresolved",
                "repair remained unresolved",
            )

        def generation_trace(self):
            return ("foreign-scope-edit", "final-unresolved")

        def generation_trace_kinds(self):
            return ("span-plan-attempt-0", "span-plan-attempt-1")

        def generation_prompt_trace(self):
            return ("prompt-0", "prompt-1")

        def generation_trace_results(self):
            return (
                {"status": "structural-rejected", "detail": "foreign scope"},
                {"status": "unresolved", "detail": "no repair"},
            )

    base = BaseAdapter()
    adapter = build_round1_shared_realization_adapter(
        mechanism="operator", base_adapter=base, maximum_candidates=1
    )
    shared = adapter.prepare_shared_context(task, revision)
    adapter.bind_shared_context(shared)

    adapter.run(task, revision)

    assert adapter.generation_trace() == (
        "foreign-scope-edit",
        "final-unresolved",
    )
    assert adapter.generation_trace_kinds() == (
        "realization-candidate-001/span-plan-attempt-0",
        "realization-candidate-001/span-plan-attempt-1",
    )
    assert adapter.generation_prompt_trace() == ("prompt-0", "prompt-1")
    assert adapter.generation_trace_results() == (
        {"status": "structural-rejected", "detail": "foreign scope"},
        {"status": "unresolved", "detail": "no repair"},
    )


def test_round1_shared_realization_binds_frozen_files_before_candidate_harvest(
    tmp_path: Path,
) -> None:
    task, revision = _fixture(tmp_path)
    target = "src/directives/vShow.ts"
    distractor = "src/compat/global.ts"
    for relative, source in (
        (target, "export function setDisplay(el) { el.style.display = '' }\n"),
        (distractor, "export const Vue = createCompatVue()\n"),
    ):
        path = task.checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    task = StudentTask.create(
        task_id="round1-vshow-shared-localization",
        checkout=task.checkout,
        instruction="vShow should preserve the user display value in vShow.ts",
        allowed_targets=[distractor, target],
        cohort="feedback",
    )

    class BaseAdapter:
        def __init__(self) -> None:
            self.seen_targets: list[tuple[str, ...]] = []
            self.seen_editable_targets: list[tuple[str, ...]] = []

        @staticmethod
        def experiment_config():
            return {"adapter": "fixture"}

        def run(self, candidate_task, candidate_revision):
            self.seen_targets.append(candidate_task.allowed_targets)
            self.seen_editable_targets.append(
                _editable_span_task(candidate_task, candidate_revision).allowed_targets
            )
            return StudentAdapter._failure(
                candidate_task,
                candidate_revision,
                '{"schema_version":1,"status":"unresolved","diagnostic":"fixture"}',
                "unresolved",
                "fixture",
            )

    base = BaseAdapter()
    adapter = build_round1_shared_realization_adapter(
        mechanism="span", base_adapter=base, maximum_candidates=1
    )
    shared = adapter.prepare_shared_context(task, revision)
    adapter.bind_shared_context(shared)

    adapter.run(task, revision)

    assert shared["target_files"] == [target]
    assert base.seen_targets == [(distractor, target), (distractor, target)]
    assert base.seen_editable_targets == [(distractor, target), (target,)]


def test_static_span_localization_adds_source_derived_adjacent_definition(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "supervisor.rb").write_text(
        "@rpc_server = RPC::Server.new(@rpc_endpoint)\n",
        encoding="utf-8",
    )
    (checkout / "rpc.rb").write_text(
        "module RPC\n  class Server\n  end\nend\n",
        encoding="utf-8",
    )
    (checkout / "child_process.rb").write_text(
        "class ChildProcess\n  def run; end\nend\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    task = StudentTask.create(
        task_id="round1-rpc-edge",
        checkout=checkout,
        instruction="rpc_endpoint fails for IPv6 in the supervisor",
        allowed_targets=["child_process.rb", "rpc.rb", "supervisor.rb"],
        cohort="feedback",
    )

    assert _issue_static_span_targets(task) == ("supervisor.rb", "rpc.rb")


def test_round1_shared_realization_rejects_localization_drift(tmp_path: Path) -> None:
    task, revision = _fixture(tmp_path)
    patch = "--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"

    class BaseAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def experiment_config(self):
            return {"adapter": "fixture"}

        def run(self, candidate_task, candidate_revision):
            self.calls += 1
            raw = _raw_plan()
            if self.calls > 1:
                data = json.loads(raw)
                data["file"] = "other.py"
                raw = json.dumps(data)
            return _attempt(candidate_task, candidate_revision, raw, patch)

    adapter = build_round1_shared_realization_adapter(
        mechanism="operator", base_adapter=BaseAdapter(), maximum_candidates=1
    )
    shared = adapter.prepare_shared_context(task, revision)
    adapter.bind_shared_context(shared)

    attempt = adapter.run(task, revision)

    assert attempt.structural_valid is False
    assert attempt.failure_reason == "unresolved"
    decisions = adapter.realization_evidence()["selection"]["candidate_decisions"]
    assert decisions[0]["status"] == "structural-invalid"


def test_round1_shared_realization_ignores_repair_diagnosis_echo_drift(
    tmp_path: Path,
) -> None:
    task, revision = _fixture(tmp_path)
    patch = "--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"

    class BaseAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def experiment_config(self):
            return {"adapter": "fixture"}

        def run(self, candidate_task, candidate_revision):
            self.calls += 1
            raw = _raw_plan(
                boundary=(
                    "value equals two"
                    if self.calls == 1
                    else "the value must become exactly two"
                )
            )
            return _attempt(candidate_task, candidate_revision, raw, patch)

    adapter = build_round1_shared_realization_adapter(
        mechanism="operator", base_adapter=BaseAdapter(), maximum_candidates=1
    )
    shared = adapter.prepare_shared_context(task, revision)
    adapter.bind_shared_context(shared)

    attempt = adapter.run(task, revision)

    assert attempt.structural_valid is True
    assert (
        adapter.realization_evidence()["diagnosis"]["fingerprint"]
        == (shared["diagnosis"]["fingerprint"])
    )


def test_round1_shared_realization_relocalizes_an_invalid_neutral_seed(
    tmp_path: Path,
) -> None:
    task, revision = _fixture(tmp_path)
    (task.checkout / "example.py").write_text(
        "class Documenter:\n"
        "    def generate(self, all_members=False):\n"
        "        return all_members\n\n"
        "class ModuleDocumenter(Documenter):\n"
        "    def import_object(self):\n"
        "        self.__all__ = object()\n\n"
        "    def get_object_members(self):\n"
        "        if not self.__all__:\n"
        "            return []\n"
        "        return list(self.__all__)\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id=task.task_id,
        checkout=task.checkout,
        instruction=(
            "empty __all__ attribute is ignored and should exclude members\n\n"
            "**Expected behavior**\n"
            "No members should be returned when `__all__` is empty.\n"
        ),
        allowed_targets=task.allowed_targets,
        cohort=task.cohort,
    )
    raw = json.loads(_raw_plan())
    raw["symbol"] = "Documenter"
    seed_raw = json.dumps(raw)

    class BaseAdapter:
        def experiment_config(self):
            return {"adapter": "fixture"}

        def run(self, candidate_task, candidate_revision):
            attempt = _attempt(candidate_task, candidate_revision, seed_raw, "")
            return replace(
                attempt,
                structural_valid=False,
                failure_reason="selector-no-match",
            )

    adapter = build_round1_shared_realization_adapter(
        mechanism="operator", base_adapter=BaseAdapter(), maximum_candidates=1
    )

    shared = adapter.prepare_shared_context(task, revision)

    assert shared["target_symbol"] == "ModuleDocumenter.get_object_members"
    assert shared["diagnosis"]["desired_boundary"] == (
        "No members should be returned when `__all__` is empty."
    )


def test_round1_shared_realization_relocalizes_a_valid_but_wrong_seed_symbol(
    tmp_path: Path,
) -> None:
    task, revision = _fixture(tmp_path)
    (task.checkout / "example.py").write_text(
        "class Documenter:\n"
        "    def generate(self, all_members=False):\n"
        "        return all_members\n\n"
        "class ModuleDocumenter(Documenter):\n"
        "    def import_object(self):\n"
        "        self.__all__ = object()\n\n"
        "    def get_object_members(self):\n"
        "        if not self.__all__:\n"
        "            return []\n"
        "        return list(self.__all__)\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id=task.task_id,
        checkout=task.checkout,
        instruction="empty __all__ attribute is ignored and should exclude members",
        allowed_targets=task.allowed_targets,
        cohort=task.cohort,
    )
    raw = json.loads(_raw_plan())
    raw["symbol"] = "Documenter"
    seed_raw = json.dumps(raw)

    class BaseAdapter:
        def experiment_config(self):
            return {"adapter": "fixture"}

        def run(self, candidate_task, candidate_revision):
            return _attempt(candidate_task, candidate_revision, seed_raw, "patch")

    adapter = build_round1_shared_realization_adapter(
        mechanism="operator", base_adapter=BaseAdapter(), maximum_candidates=1
    )

    shared = adapter.prepare_shared_context(task, revision)

    assert shared["target_symbol"] == "ModuleDocumenter.get_object_members"
    assert shared["source_policy"] == "issue-static-symbol-localization-v2"


def test_round1_shared_operator_relocalizes_invalid_seed_by_issue_file_and_symbol(
    tmp_path: Path,
) -> None:
    task, revision = _fixture(tmp_path)
    mock_file = "sphinx/ext/autodoc/mock.py"
    unrelated = "sphinx/domains/python.py"
    (task.checkout / mock_file).parent.mkdir(parents=True)
    (task.checkout / unrelated).parent.mkdir(parents=True)
    (task.checkout / mock_file).write_text(
        "class _MockObject:\n"
        "    __display_name__ = '_MockObject'\n\n"
        "    def __init__(self):\n"
        "        self.__qualname__ = ''\n\n"
        "    def __mro_entries__(self, bases):\n"
        "        return (self.__class__,)\n",
        encoding="utf-8",
    )
    (task.checkout / unrelated).write_text(
        "class PyObject:\n"
        "    def document_inherited_classes_correctly(self):\n"
        "        return True\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="round1-invalid-operator-static-localization",
        checkout=task.checkout,
        instruction=(
            "Inherited classes not correctly documented when mocked. "
            "The Bases section truncates torch.nn.Module."
        ),
        allowed_targets=[unrelated, mock_file],
        cohort="feedback",
    )
    seed = json.loads(_raw_plan())
    seed["file"] = unrelated
    seed["symbol"] = "PyObject"
    seed["operations"] = []
    seed_raw = json.dumps(seed)

    class BaseAdapter:
        def experiment_config(self):
            return {"adapter": "fixture"}

        def run(self, candidate_task, candidate_revision):
            attempt = _attempt(candidate_task, candidate_revision, seed_raw, "")
            return replace(
                attempt,
                structural_valid=False,
                failure_reason="unresolved",
                target_file=None,
            )

    adapter = build_round1_shared_realization_adapter(
        mechanism="operator", base_adapter=BaseAdapter(), maximum_candidates=1
    )

    shared = adapter.prepare_shared_context(task, revision)

    assert shared["target_files"] == [mock_file]
    assert shared["target_symbol"] == "_MockObject"
    assert shared["source_policy"] == "issue-static-operator-localization-v3"


def test_round1_shared_span_relocalizes_invalid_seed_by_issue_path_and_source(
    tmp_path: Path,
) -> None:
    task, revision = _fixture(tmp_path)
    patch_flags = "packages/shared/src/patchFlags.ts"
    transform_element = "packages/compiler-core/src/transforms/transformElement.ts"
    v_for = "packages/compiler-core/src/transforms/vFor.ts"
    (task.checkout / patch_flags).parent.mkdir(parents=True)
    (task.checkout / patch_flags).write_text(
        "export enum PatchFlags { KEYED_FRAGMENT, UNKEYED_FRAGMENT }\n",
        encoding="utf-8",
    )
    (task.checkout / v_for).parent.mkdir(parents=True)
    (task.checkout / transform_element).write_text(
        "export function transformElement(node) {\n"
        "  const key = findProp(node, `key`) // shorthand handling\n"
        "  return key\n"
        "}\n",
        encoding="utf-8",
    )
    (task.checkout / v_for).write_text(
        "export function transformFor(node) {\n"
        "  const keyProp = findProp(node, `key`)\n"
        "  return keyProp ? PatchFlags.KEYED_FRAGMENT : "
        "PatchFlags.UNKEYED_FRAGMENT\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="round1-vue-span-localization",
        checkout=task.checkout,
        instruction=(
            "fix(compiler-core): :key shorthand compiles to UNKEYED_FRAGMENT "
            "instead of KEYED_FRAGMENT"
        ),
        allowed_targets=[patch_flags, transform_element, v_for],
        cohort="feedback",
    )
    seed_raw = json.dumps(
        {
            "schema_version": 1,
            "plans": [
                {
                    "schema_version": 1,
                    "file": patch_flags,
                    "intent": {
                        "defect": "wrong fragment flag",
                        "trigger": "key shorthand",
                        "desired_boundary": "keyed fragment",
                    },
                    "operations": [{"before": "missing", "after": "wrong"}],
                }
            ],
            "diagnostic": "wrong target",
        }
    )

    class BaseAdapter:
        def experiment_config(self):
            return {"adapter": "fixture"}

        def run(self, candidate_task, candidate_revision):
            attempt = _attempt(candidate_task, candidate_revision, seed_raw, "")
            return replace(
                attempt,
                structural_valid=False,
                failure_reason="unresolved",
                target_file=None,
            )

    adapter = build_round1_shared_realization_adapter(
        mechanism="span", base_adapter=BaseAdapter(), maximum_candidates=1
    )

    shared = adapter.prepare_shared_context(task, revision)

    assert shared["target_files"] == [v_for]
    assert shared["source_policy"] == "issue-static-file-localization-v6"


def test_round1_shared_span_prefers_explicit_command_name_in_target_path(
    tmp_path: Path,
) -> None:
    task, revision = _fixture(tmp_path)
    hide_env = "crates/nu-cmd-lang/src/core_commands/hide_env.rs"
    plugin = "crates/nu-plugin/src/plugin/interface/mod.rs"
    (task.checkout / hide_env).parent.mkdir(parents=True)
    (task.checkout / plugin).parent.mkdir(parents=True)
    (task.checkout / hide_env).write_text(
        "pub fn hide_env(stack: &mut Stack, name: &str) {\n"
        "    stack.remove_env_var(name);\n"
        "}\n",
        encoding="utf-8",
    )
    (task.checkout / plugin).write_text(
        "pub fn child_process_environment(engine: &EngineState) {\n"
        '    let environment_variable = engine.get_env_var("TEST");\n'
        "    if environment_variable.is_none() { return; }\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="round1-nushell-span-localization",
        checkout=task.checkout,
        instruction=(
            "Child processes inherit initial environment variables after `hide-env`. "
            "The TEST environment variable should not be found."
        ),
        allowed_targets=[plugin, hide_env],
        cohort="feedback",
    )
    seed_raw = json.dumps(
        {
            "schema_version": 1,
            "plans": [
                {
                    "schema_version": 1,
                    "file": plugin,
                    "intent": {
                        "defect": "child environment inherits TEST",
                        "trigger": "hide-env",
                        "desired_boundary": "TEST absent",
                    },
                    "operations": [{"before": "missing", "after": "wrong"}],
                }
            ],
            "diagnostic": "wrong target",
        }
    )

    class BaseAdapter:
        def experiment_config(self):
            return {"adapter": "fixture"}

        def run(self, candidate_task, candidate_revision):
            attempt = _attempt(candidate_task, candidate_revision, seed_raw, "")
            return replace(
                attempt,
                structural_valid=False,
                failure_reason="unresolved",
                target_file=None,
            )

    adapter = build_round1_shared_realization_adapter(
        mechanism="span", base_adapter=BaseAdapter(), maximum_candidates=1
    )

    shared = adapter.prepare_shared_context(task, revision)

    assert shared["target_files"] == [hide_env]
    assert shared["source_policy"] == "issue-static-file-localization-v6"


def test_round1_shared_span_prefers_causal_process_sink_over_trigger_command(
    tmp_path: Path,
) -> None:
    task, revision = _fixture(tmp_path)
    hide_env = "crates/nu-cmd-lang/src/core_commands/hide_env.rs"
    run_external = "crates/nu-command/src/system/run_external.rs"
    for relative in (hide_env, run_external):
        (task.checkout / relative).parent.mkdir(parents=True, exist_ok=True)
    (task.checkout / hide_env).write_text(
        "pub fn hide_env(stack: &mut Stack, name: &str) {\n"
        "    stack.remove_env_var(name);\n"
        "}\n",
        encoding="utf-8",
    )
    (task.checkout / run_external).write_text(
        "pub fn spawn_external(env_vars: &Env) {\n"
        '    let mut process = std::process::Command::new("nu");\n'
        "    process.envs(env_vars);\n"
        "    process.spawn();\n"
        "}\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="round1-nushell-causal-sink-localization",
        checkout=task.checkout,
        instruction=(
            "Child processes can inherit initial environment variables after "
            "`hide-env`. When we spawn child processes, we do not clear "
            "environment variables before the external command starts."
        ),
        allowed_targets=[hide_env, run_external],
        cohort="feedback",
    )

    class BaseAdapter:
        @staticmethod
        def experiment_config():
            return {"adapter": "fixture"}

        @staticmethod
        def run(candidate_task, candidate_revision):
            return StudentAdapter._failure(
                candidate_task,
                candidate_revision,
                "not-json",
                "unresolved",
                "neutral seed unresolved",
            )

    adapter = build_round1_shared_realization_adapter(
        mechanism="span", base_adapter=BaseAdapter(), maximum_candidates=1
    )

    shared = adapter.prepare_shared_context(task, revision)

    assert shared["target_files"] == [run_external]
    assert shared["source_policy"] == "issue-static-file-localization-v6"


def test_round1_shared_span_maps_qualified_api_to_matching_header_stem(
    tmp_path: Path,
) -> None:
    task, revision = _fixture(tmp_path)
    os_header = "include/fmt/os.h"
    printf_header = "include/fmt/printf.h"
    for relative in (os_header, printf_header):
        (task.checkout / relative).parent.mkdir(parents=True, exist_ok=True)
    (task.checkout / os_header).write_text(
        "template <typename... T> void sprintf_file(T... args);\n",
        encoding="utf-8",
    )
    (task.checkout / printf_header).write_text(
        "template <typename Char> class printf_arg_formatter {\n"
        "  void format_char(Char value) { fmt_specs.align = align::right; }\n"
        "};\n",
        encoding="utf-8",
    )
    task = StudentTask.create(
        task_id="round1-fmt-qualified-api-localization",
        checkout=task.checkout,
        instruction=(
            "fmt::sprintf ignores the minus flag for char while std::printf "
            "left-aligns the value."
        ),
        allowed_targets=[os_header, printf_header],
        cohort="feedback",
    )

    class BaseAdapter:
        @staticmethod
        def experiment_config():
            return {"adapter": "fixture"}

        @staticmethod
        def run(candidate_task, candidate_revision):
            return StudentAdapter._failure(
                candidate_task,
                candidate_revision,
                "not-json",
                "unresolved",
                "neutral seed unresolved",
            )

    adapter = build_round1_shared_realization_adapter(
        mechanism="span", base_adapter=BaseAdapter(), maximum_candidates=1
    )

    shared = adapter.prepare_shared_context(task, revision)

    assert shared["target_files"] == [printf_header]
    assert shared["source_policy"] == "issue-static-file-localization-v6"
