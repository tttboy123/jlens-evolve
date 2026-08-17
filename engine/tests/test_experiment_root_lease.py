from __future__ import annotations

from pathlib import Path

import pytest

from skill_evolution_loop.contracts import ContractError
from skill_evolution_loop.experiment import ExperimentRootLease


def test_experiment_root_lease_rejects_a_concurrent_writer_and_releases(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experiment"

    with ExperimentRootLease(root):
        assert (root / ".writer.lock").is_file()
        with pytest.raises(ContractError, match="already has an active writer"):
            with ExperimentRootLease(root):
                pass

    with ExperimentRootLease(root):
        pass
