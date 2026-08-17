"""Project-local construction of the first real Qwen 3+3 TaskSet."""

from __future__ import annotations

import json
from pathlib import Path

from .contracts import ContractError, canonical_json
from .eval_manifest import EvaluationTask, EvaluationTaskSet
from .target_selection import TargetSelectionManifest, TargetSelectionRecord

_SPHINX_PROBLEMS = Path("/private/tmp/qwen-sphinx/problems.json")
_TASK_INPUT_ROOT = Path("/private/tmp/evolve-v3-sync/tasks")
_P1_TARGET_SELECTION_EVIDENCE = {
    "p1-sphinx-7757": [
        "issue-phrase:positional-only-default",
        "repository-search:sphinx/util/inspect.py",
        "adjacent-parser:sphinx/domains/python.py",
    ],
    "p1-sphinx-9658": [
        "issue-phrase:mocked-base-class",
        "repository-search:sphinx/ext/autodoc/mock.py",
    ],
    "p1-sphinx-10435": [
        "issue-phrase:latex-inline-highlight-whitespace",
        "repository-search:sphinx/writers/latex.py",
        "direct-dependency:sphinx/highlighting.py",
    ],
    "p1-darkreader-6747": [
        "issue-phrase:first-generic-fix",
        "semantic-seed:src/generators/dynamic-theme.ts",
        "relative-import:getSitesFixesFor->src/generators/utils/parse.ts",
    ],
    "p1-dayjs-938": [
        "issue-symbol:LocaleData",
        "plugin-path:src/plugin/localeData/index.js",
        "direct-dependency:src/utils.js",
    ],
    "p1-django-16493": [
        "issue-symbol:FileField.deconstruct",
        "repository-search:django/db/models/fields/files.py",
    ],
    "p1-clap-2075": [
        "issue-phrase:conflicting-required-arguments",
        "repository-search:src/parse/validator.rs",
    ],
    "p1-sphinx-8638": [
        "issue-phrase:instance-variable-autolink",
        "repository-search:sphinx/domains/python.py",
        "field-definition:PyTypedField-variable",
    ],
    "p1-django-13794": [
        "issue-phrase:lazy-string-concatenation",
        "repository-search:django/utils/functional.py",
        "issue-symbol:Promise.__radd__",
    ],
    "p1-jq-2674": [
        "issue-symbol:nth/2",
        "repository-search:src/builtin.jq",
        "issue-phrase:index-out-of-range",
    ],
}


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"P1 source artifact is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"P1 source artifact must be an object: {path}")
    return value


def build_local_p1_taskset(project_root: Path) -> EvaluationTaskSet:
    """Build the frozen pilot split exclusively from existing local evidence."""
    root = project_root.resolve()
    sphinx = _read_json(_SPHINX_PROBLEMS)
    task_input_ids = {
        "darkreader__darkreader-6747": (
            "25c73a2d16c14222619fc1d83960f1d79f0b45e64c2b4aa97bf2a615913cc504"
        ),
        "iamkun__dayjs-938": (
            "5e6f68b2893aa6891647ed50853c0e19e1d863c9137a978ad698a534a726e0f8"
        ),
        "cli__cli-545": (
            "e8474c3bbc4317fd548e48bc53d7f8e06d5a43fa35d3c605336d1208c8c471c6"
        ),
    }
    holdout_inputs = {
        instance_id: _read_json(_TASK_INPUT_ROOT / task_uid / "task-input.json")
        for instance_id, task_uid in task_input_ids.items()
    }

    tasks = [
        EvaluationTask.create(
            task_id="p1-sphinx-7757",
            instance_id="sphinx-doc__sphinx-7757",
            benchmark_id="swe-bench-verified",
            benchmark_base_commit="212fd67b9f0b4fae6a7c3501fdf1a9a5b2801329",
            repo="sphinx-doc/sphinx",
            source_repository=Path("/private/tmp/qwen-sphinx/sphinx-7757"),
            source_revision="6d4215c6076de5c9e116803d65d0d88d2774c9da",
            instruction=sphinx["sphinx-doc__sphinx-7757"]["problem_statement"],
            allowed_targets=["sphinx/util/inspect.py", "sphinx/domains/python.py"],
            cohort="feedback",
        ),
        EvaluationTask.create(
            task_id="p1-sphinx-9658",
            instance_id="sphinx-doc__sphinx-9658",
            benchmark_id="swe-bench-verified",
            benchmark_base_commit="232dbe41c5250eb7d559d40438c4743483e95f15",
            repo="sphinx-doc/sphinx",
            source_repository=Path("/private/tmp/qwen-sphinx/sphinx-9658"),
            source_revision="55f2a0a5dd711903df2308befb9cc8e1a9db7312",
            instruction=sphinx["sphinx-doc__sphinx-9658"]["problem_statement"],
            allowed_targets=["sphinx/ext/autodoc/mock.py"],
            cohort="feedback",
        ),
        EvaluationTask.create(
            task_id="p1-sphinx-10435",
            instance_id="sphinx-doc__sphinx-10435",
            benchmark_id="swe-bench-verified",
            benchmark_base_commit="f1061c012e214f16fd8790dec3c283d787e3daa8",
            repo="sphinx-doc/sphinx",
            source_repository=Path("/private/tmp/qwen-sphinx/sphinx-10435"),
            source_revision="5013a34d9989f363136431b737b33bdcce498410",
            instruction=sphinx["sphinx-doc__sphinx-10435"]["problem_statement"],
            allowed_targets=["sphinx/writers/latex.py", "sphinx/highlighting.py"],
            cohort="feedback",
        ),
        _holdout_task(
            task_id="p1-darkreader-6747",
            data=holdout_inputs["darkreader__darkreader-6747"],
            source_repository=Path("/private/tmp/swe-4b/darkreader-6747-base"),
            source_revision="a787eb511f45159c8869d30e5a6ba1f91cb67709",
            allowed_targets=[
                "src/utils/async-queue.ts",
                "src/generators/dynamic-theme.ts",
            ],
        ),
        _holdout_task(
            task_id="p1-dayjs-938",
            data=holdout_inputs["iamkun__dayjs-938"],
            source_repository=Path("/private/tmp/swe-4b/dayjs-base"),
            source_revision="72cdb20552dd5af6c2295bf52075ba19cceb000f",
            allowed_targets=["src/plugin/localeData/index.js", "src/utils.js"],
        ),
        _holdout_task(
            task_id="p1-cli-545",
            data=holdout_inputs["cli__cli-545"],
            source_repository=(
                root / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/"
                "ds-teaching-samples/real-search-002/native-evaluator/"
                "multi-repos/cli/cli"
            ),
            source_revision="7950023a8775066fcf2c5552e2cd6ec582ceb34e",
            allowed_targets=[
                "internal/cobrafish/completion.go",
                "cmd/gen-docs/main.go",
            ],
        ),
    ]
    return EvaluationTaskSet.create(
        taskset_id="p1-local-qwen-3x3-v1",
        tasks=tasks,
    )


def _holdout_task(
    *,
    task_id: str,
    data: dict,
    source_repository: Path,
    source_revision: str,
    allowed_targets: list[str],
) -> EvaluationTask:
    return _gold_free_task(
        task_id=task_id,
        data=data,
        source_repository=source_repository,
        source_revision=source_revision,
        allowed_targets=allowed_targets,
        cohort="holdout",
    )


def _gold_free_task(
    *,
    task_id: str,
    data: dict,
    source_repository: Path,
    source_revision: str,
    allowed_targets: list[str],
    cohort: str,
) -> EvaluationTask:
    if data.get("gold_fields_included") is not False:
        raise ContractError("P1 task source must exclude gold fields")
    return EvaluationTask.create(
        task_id=task_id,
        instance_id=str(data["instance_id"]),
        benchmark_id=str(data["benchmark_id"]),
        benchmark_base_commit=str(data["base_commit"]),
        repo=str(data["repo"]),
        source_repository=source_repository,
        source_revision=source_revision,
        instruction=str(data["instruction"]),
        allowed_targets=allowed_targets,
        cohort=cohort,
    )


def freeze_local_p1_taskset(project_root: Path, output_path: Path) -> EvaluationTaskSet:
    """Write the taskset once; an existing manifest is never overwritten."""
    target = output_path.resolve()
    if target.exists():
        raise ContractError(f"P1 taskset already exists: {target}")
    taskset = build_local_p1_taskset(project_root)
    preflight = taskset.preflight()
    if not preflight.ready:
        raise ContractError(
            "P1 taskset preflight failed: " + "; ".join(preflight.errors)
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(taskset.to_dict()) + "\n", encoding="utf-8")
    return taskset


def build_local_p1_v2_taskset(project_root: Path) -> EvaluationTaskSet:
    """Build the target-qualified replacement without mutating frozen P1 v1."""
    root = project_root.resolve()
    original = build_local_p1_taskset(root)
    feedback = [task for task in original.tasks if task.cohort == "feedback"]
    darkreader = _read_json(
        _TASK_INPUT_ROOT
        / "25c73a2d16c14222619fc1d83960f1d79f0b45e64c2b4aa97bf2a615913cc504"
        / "task-input.json"
    )
    dayjs = _read_json(
        _TASK_INPUT_ROOT
        / "5e6f68b2893aa6891647ed50853c0e19e1d863c9137a978ad698a534a726e0f8"
        / "task-input.json"
    )
    django = _read_json(
        _TASK_INPUT_ROOT
        / "0d07080f42f2daa82176ff8a3b5bd6efe18e99882d820273a2b5f2103206f961"
        / "task-input.json"
    )
    holdout = [
        _holdout_task(
            task_id="p1-darkreader-6747",
            data=darkreader,
            source_repository=Path("/private/tmp/swe-4b/darkreader-6747-base"),
            source_revision="a787eb511f45159c8869d30e5a6ba1f91cb67709",
            allowed_targets=[
                "src/generators/dynamic-theme.ts",
                "src/generators/utils/parse.ts",
            ],
        ),
        _holdout_task(
            task_id="p1-dayjs-938",
            data=dayjs,
            source_repository=Path("/private/tmp/swe-4b/dayjs-base"),
            source_revision="72cdb20552dd5af6c2295bf52075ba19cceb000f",
            allowed_targets=["src/plugin/localeData/index.js", "src/utils.js"],
        ),
        _holdout_task(
            task_id="p1-django-16493",
            data=django,
            source_repository=Path("/private/tmp/qwen-django/django-16493"),
            source_revision="e3a4cee081cf60650b8824f0646383b79cb110e7",
            allowed_targets=["django/db/models/fields/files.py"],
        ),
    ]
    return EvaluationTaskSet.create(
        taskset_id="p1-local-qwen-3x3-v2",
        tasks=[*feedback, *holdout],
    )


def build_local_p1_v2_target_selection(
    taskset: EvaluationTaskSet,
) -> TargetSelectionManifest:
    """Bind independently selected targets; no reference patch evidence enters."""
    return TargetSelectionManifest.create(
        taskset=taskset,
        records=[
            TargetSelectionRecord.create(
                task=task,
                selector_id="issue-symbol-dependency-v1",
                evidence=_P1_TARGET_SELECTION_EVIDENCE[task.task_id],
            )
            for task in taskset.tasks
        ],
    )


def freeze_local_p1_v2_bundle(
    project_root: Path,
    taskset_path: Path,
    target_selection_path: Path,
) -> tuple[EvaluationTaskSet, TargetSelectionManifest]:
    """Freeze both gold-free v2 qualification inputs append-only."""
    taskset_target = taskset_path.resolve()
    selection_target = target_selection_path.resolve()
    if taskset_target.exists() or selection_target.exists():
        raise ContractError("P1 v2 qualification bundle already exists")
    taskset = build_local_p1_v2_taskset(project_root)
    preflight = taskset.preflight()
    if not preflight.ready:
        raise ContractError(
            "P1 v2 taskset preflight failed: " + "; ".join(preflight.errors)
        )
    selection = build_local_p1_v2_target_selection(taskset)
    taskset_target.parent.mkdir(parents=True, exist_ok=True)
    selection_target.parent.mkdir(parents=True, exist_ok=True)
    taskset_target.write_text(
        canonical_json(taskset.to_dict()) + "\n", encoding="utf-8"
    )
    selection_target.write_text(
        canonical_json(selection.to_dict()) + "\n", encoding="utf-8"
    )
    return taskset, selection


def build_local_p1_v3_taskset(project_root: Path) -> EvaluationTaskSet:
    """Replace the multi-file hold-out with a single-file solvable task."""
    previous = build_local_p1_v2_taskset(project_root)
    tasks = [task for task in previous.tasks if task.task_id != "p1-darkreader-6747"]
    clap = _read_json(
        _TASK_INPUT_ROOT
        / "c2279b8f00feb584a8d39cc652e2be229597f3dd24bc7c1ecd77a932767e612f"
        / "task-input.json"
    )
    tasks.append(
        _holdout_task(
            task_id="p1-clap-2075",
            data=clap,
            source_repository=Path("/private/tmp/qwen-clap/clap-2075"),
            source_revision="b04e00b8c670a96c91a56b59a8cd8d7a5a20f9b9",
            allowed_targets=["src/parse/validator.rs"],
        )
    )
    return EvaluationTaskSet.create(
        taskset_id="p1-local-qwen-3x3-v3",
        tasks=tasks,
    )


def freeze_local_p1_v3_bundle(
    project_root: Path,
    taskset_path: Path,
    target_selection_path: Path,
) -> tuple[EvaluationTaskSet, TargetSelectionManifest]:
    """Freeze the single-file-capacity-qualified 3+3 bundle append-only."""
    taskset_target = taskset_path.resolve()
    selection_target = target_selection_path.resolve()
    if taskset_target.exists() or selection_target.exists():
        raise ContractError("P1 v3 qualification bundle already exists")
    taskset = build_local_p1_v3_taskset(project_root)
    preflight = taskset.preflight()
    if not preflight.ready:
        raise ContractError(
            "P1 v3 taskset preflight failed: " + "; ".join(preflight.errors)
        )
    selection = build_local_p1_v2_target_selection(taskset)
    taskset_target.parent.mkdir(parents=True, exist_ok=True)
    selection_target.parent.mkdir(parents=True, exist_ok=True)
    taskset_target.write_text(
        canonical_json(taskset.to_dict()) + "\n", encoding="utf-8"
    )
    selection_target.write_text(
        canonical_json(selection.to_dict()) + "\n", encoding="utf-8"
    )
    return taskset, selection


def build_local_p1_v4_taskset(project_root: Path) -> EvaluationTaskSet:
    """Build a 3+3 split expressible by both frozen Student mechanisms."""
    previous = build_local_p1_v3_taskset(project_root)
    retained_ids = {
        "p1-sphinx-7757",
        "p1-sphinx-10435",
        "p1-django-16493",
    }
    tasks = [task for task in previous.tasks if task.task_id in retained_ids]
    inputs = {
        "sphinx": _read_json(
            _TASK_INPUT_ROOT
            / "ed26211d683de2284a72e8f56fed3ce689279ea8dad66c760195e8e40c1caf56"
            / "task-input.json"
        ),
        "django": _read_json(
            _TASK_INPUT_ROOT
            / "3fe9ae4a274a2102bace83dac50948d61c3eeeb21ee64cda45ae68099266467f"
            / "task-input.json"
        ),
        "jq": _read_json(
            _TASK_INPUT_ROOT
            / "b02eccb5ab1f58997bf3a1dc313b919c00fc2e10eae3bec63ae5574ecfaaaea7"
            / "task-input.json"
        ),
    }
    tasks.extend(
        [
            _gold_free_task(
                task_id="p1-sphinx-8638",
                data=inputs["sphinx"],
                source_repository=Path("/private/tmp/qwen-sphinx/sphinx-8638"),
                source_revision="f220d52f06889ccd854d432832efdbbd10a41f12",
                allowed_targets=["sphinx/domains/python.py"],
                cohort="feedback",
            ),
            _gold_free_task(
                task_id="p1-django-13794",
                data=inputs["django"],
                source_repository=Path("/private/tmp/qwen-django/django-13794"),
                source_revision="e35e5088406cf3138ac76d0814ef8fca6ea9982a",
                allowed_targets=["django/utils/functional.py"],
                cohort="holdout",
            ),
            _gold_free_task(
                task_id="p1-jq-2674",
                data=inputs["jq"],
                source_repository=Path("/private/tmp/qwen-jq/jq-2674"),
                source_revision="3f48a829e564b1e9da173bcd30e1cba13046fbbd",
                allowed_targets=["src/builtin.jq"],
                cohort="holdout",
            ),
        ]
    )
    return EvaluationTaskSet.create(
        taskset_id="p1-local-qwen-3x3-v4-mechanism-qualified",
        tasks=tasks,
    )


def freeze_local_p1_v4_bundle(
    project_root: Path,
    taskset_path: Path,
    target_selection_path: Path,
) -> tuple[EvaluationTaskSet, TargetSelectionManifest]:
    """Freeze the dual-mechanism-capacity-qualified 3+3 bundle append-only."""
    taskset_target = taskset_path.resolve()
    selection_target = target_selection_path.resolve()
    if taskset_target.exists() or selection_target.exists():
        raise ContractError("P1 v4 qualification bundle already exists")
    taskset = build_local_p1_v4_taskset(project_root)
    preflight = taskset.preflight()
    if not preflight.ready:
        raise ContractError(
            "P1 v4 taskset preflight failed: " + "; ".join(preflight.errors)
        )
    selection = build_local_p1_v2_target_selection(taskset)
    taskset_target.parent.mkdir(parents=True, exist_ok=True)
    selection_target.parent.mkdir(parents=True, exist_ok=True)
    taskset_target.write_text(
        canonical_json(taskset.to_dict()) + "\n", encoding="utf-8"
    )
    selection_target.write_text(
        canonical_json(selection.to_dict()) + "\n", encoding="utf-8"
    )
    return taskset, selection
