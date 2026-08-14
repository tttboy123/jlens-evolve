"""Compatibility import for the public autonomous evolution product entry."""

from __future__ import annotations

from pathlib import Path

from evolve.autonomous import (
    AutonomousEvolutionConfig,
    AutonomousEvolutionError,
    AutonomousEvolutionRunner,
    EvolutionDependencies,
    FeedbackTaskSelector,
    build_default_dependencies,
)


def run_autonomous_evolution(
    *,
    config_path: Path,
    output_root: Path,
    worktree_root: Path,
    dependencies: EvolutionDependencies | None = None,
) -> dict[str, object]:
    """Load the user config and run/resume the real autonomous product loop."""

    config = AutonomousEvolutionConfig.load(config_path)
    runner = AutonomousEvolutionRunner(
        config=config,
        output_root=output_root,
        worktree_root=worktree_root,
        dependencies=dependencies or build_default_dependencies(config),
    )
    return runner.run()


__all__ = [
    "AutonomousEvolutionConfig",
    "AutonomousEvolutionError",
    "AutonomousEvolutionRunner",
    "EvolutionDependencies",
    "FeedbackTaskSelector",
    "run_autonomous_evolution",
]
