# Project execution rules

## Autonomous Minimax evolution

- Before starting or resuming an autonomous Minimax evolution round, read
  `MINIMAX-OUTPUT-CONTRACT.md` and follow its ownership, evidence, and path
  rules. A round is not complete until its immutable report is indexed under
  `runs/skill-evolution-loop/autonomous/`.
- Current user scope lock: do not start holdout, generalization, Docker/native
  harness, or multi-model validation rounds until the user explicitly resumes
  that work. Do not start any autonomous round while Codex repair or review is
  active in this shared directory; resume only after an explicit user request.

## Persistent CUDA worker lifecycle

- CUDA workers are project-level reusable infrastructure, not per-run resources.
- Never call `TerminateInstances`, install a terminate action timer, delete the
  system disk, or disable API termination protection as part of an evolution run.
- New workers must be created with API termination protection enabled and without
  a terminate action timer.
- Completion, failure, timeout, and operator interruption must checkpoint first,
  then use `StopInstances` with `StoppedMode=STOP_CHARGING` and retain the worker.
- Reuse a stopped compatible worker before considering creation of another one.
- Destruction is outside the normal automation contract and requires a separate,
  explicit user request naming the exact resource at the time of deletion.
