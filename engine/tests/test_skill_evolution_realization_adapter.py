from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from skill_evolution_loop.contracts import LoopRevision
from skill_evolution_loop.realization_adapter import DiagnosisFrozenRealizationAdapter
from skill_evolution_loop.realization_candidates import FrozenDiagnosis
from skill_evolution_loop.student_adapter import StudentAttempt, StudentTask


def _fixture(tmp_path: Path) -> tuple[StudentTask, LoopRevision]:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    source = checkout / "example.py"
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    task = StudentTask.create(
        task_id="candidate-fixture",
        checkout=checkout,
        instruction="Change value.",
        allowed_targets=["example.py"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="candidate-skill",
        revision_id="candidate-r001",
        parent_revision_id=None,
        source_round=1,
        protocol="diagnosis-frozen-realization-v1",
        skill_text="Use the frozen diagnosis.",
        prompt_template="Return one candidate.",
        eval_note="fixture",
    )
    return task, revision


def _attempt(
    task: StudentTask,
    revision: LoopRevision,
    *,
    raw: str,
    patch: str,
    valid: bool,
) -> StudentAttempt:
    def digest(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    return StudentAttempt(
        task=task,
        revision_id=revision.revision_id,
        raw_output=raw,
        raw_output_sha256=digest(raw),
        edit=None,
        patch=patch,
        patch_sha256=digest(patch) if patch else None,
        target_file="example.py" if valid else None,
        before_sha256="a" * 64 if valid else None,
        after_sha256="b" * 64 if valid else None,
        implementation_fingerprint="c" * 64 if valid else None,
        structural_valid=valid,
        failure_reason=None if valid else "selector-no-match",
        detail="candidate",
    )


def test_diagnosis_frozen_adapter_selects_one_bounded_candidate(tmp_path: Path) -> None:
    task, revision = _fixture(tmp_path)
    diagnosis_calls = 0
    seen_diagnoses: list[str] = []

    def diagnose(_task, _revision):
        nonlocal diagnosis_calls
        diagnosis_calls += 1
        return FrozenDiagnosis.create(
            defect="value is too small",
            trigger="module is loaded",
            desired_boundary="value equals two",
        )

    patches = [
        "--- a/example.py\n+++ b/example.py\n@@ -1 +1,2 @@\n-value = 1\n+value = 2\n+other = 3\n",
        "--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n",
        "",
    ]

    def realize(candidate_task, candidate_revision, diagnosis, index):
        seen_diagnoses.append(diagnosis.fingerprint)
        return _attempt(
            candidate_task,
            candidate_revision,
            raw=f"candidate-{index}",
            patch=patches[index],
            valid=bool(patches[index]),
        )

    adapter = DiagnosisFrozenRealizationAdapter(
        diagnosis_provider=diagnose,
        candidate_runner=realize,
        maximum_candidates=3,
    )

    selected = adapter.run(task, revision)
    evidence = adapter.realization_evidence()

    assert diagnosis_calls == 1
    assert len(set(seen_diagnoses)) == 1
    assert selected.patch == patches[1]
    assert evidence["selection"]["selected_candidate_id"] == "candidate-002"
    assert len(evidence["candidates"]) == 3
    assert adapter.generation_trace() == (
        "candidate-0",
        "candidate-1",
        "candidate-2",
    )


def test_diagnosis_frozen_adapter_fails_closed_without_eligible_candidate(
    tmp_path: Path,
) -> None:
    task, revision = _fixture(tmp_path)
    diagnosis = FrozenDiagnosis.create(
        defect="value is too small",
        trigger="module is loaded",
        desired_boundary="value equals two",
    )

    adapter = DiagnosisFrozenRealizationAdapter(
        diagnosis_provider=lambda _task, _revision: diagnosis,
        candidate_runner=lambda candidate_task, candidate_revision, _diagnosis, index: (
            _attempt(
                candidate_task,
                candidate_revision,
                raw=f"invalid-{index}",
                patch="",
                valid=False,
            )
        ),
        maximum_candidates=2,
    )

    selected = adapter.run(task, revision)

    assert selected.structural_valid is False
    assert selected.failure_reason == "unresolved"
    assert adapter.realization_evidence()["selection"]["selected_candidate_id"] is None


def test_candidate_runner_exception_is_preserved_as_eval_infra_detail(
    tmp_path: Path,
) -> None:
    task, revision = _fixture(tmp_path)
    diagnosis = FrozenDiagnosis.create(
        defect="value is too small",
        trigger="module is loaded",
        desired_boundary="value equals two",
    )
    adapter = DiagnosisFrozenRealizationAdapter(
        diagnosis_provider=lambda _task, _revision: diagnosis,
        candidate_runner=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("offline replay exposed stale transport trace")
        ),
        maximum_candidates=1,
    )

    selected = adapter.run(task, revision)
    evidence = adapter.realization_evidence()

    assert selected.failure_reason == "unresolved"
    assert evidence["candidates"][0]["failure_reason"] == "eval-infra"
    assert evidence["candidates"][0]["detail"] == (
        "candidate runner failed: offline replay exposed stale transport trace"
    )


def test_seeded_realization_adapter_reuses_first_candidate_diagnosis(
    tmp_path: Path,
) -> None:
    task, revision = _fixture(tmp_path)
    diagnosis = FrozenDiagnosis.create(
        defect="value is too small",
        trigger="module is loaded",
        desired_boundary="value equals two",
    )
    seed_patch = (
        "--- a/example.py\n+++ b/example.py\n@@ -1 +1,2 @@\n"
        "-value = 1\n+value = 2\n+other = 3\n"
    )
    better_patch = (
        "--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    )
    seen: list[tuple[str, int]] = []

    def seed(candidate_task, candidate_revision):
        return diagnosis, _attempt(
            candidate_task,
            candidate_revision,
            raw="seed",
            patch=seed_patch,
            valid=True,
        )

    def realize(candidate_task, candidate_revision, frozen, index):
        seen.append((frozen.fingerprint, index))
        return _attempt(
            candidate_task,
            candidate_revision,
            raw=f"candidate-{index}",
            patch=better_patch,
            valid=True,
        )

    adapter = DiagnosisFrozenRealizationAdapter(
        seed_candidate_provider=seed,
        candidate_runner=realize,
        maximum_candidates=2,
    )

    selected = adapter.run(task, revision)

    assert selected.patch == better_patch
    assert seen == [(diagnosis.fingerprint, 1)]
    assert adapter.generation_trace() == ("seed", "candidate-1")
    assert adapter.realization_evidence()["selection"]["selected_candidate_id"] == (
        "candidate-002"
    )
