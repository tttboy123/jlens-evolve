.PHONY: bootstrap doctor test-infra verify

LOOP_VENV := .venv-loop
LOOP_PYTHON := $(LOOP_VENV)/bin/python
LOOP_RUFF := $(LOOP_VENV)/bin/ruff

bootstrap:
	UV_PROJECT_ENVIRONMENT=$(LOOP_VENV) uv sync --locked --extra local-mlx

doctor:
	$(LOOP_PYTHON) -m skill_evolution_loop doctor --root .

test-infra:
	$(LOOP_PYTHON) -m pytest -q \
		tests/test_skill_evolution_infrastructure.py \
		tests/test_skill_evolution_loop.py \
		tests/test_skill_evolution_taskset.py \
		tests/test_skill_evolution_target_audit.py \
		tests/test_skill_evolution_target_selection.py \
		tests/test_skill_evolution_experiment.py \
		tests/test_skill_evolution_p1_parent.py

verify:
	$(LOOP_PYTHON) -m pytest -q
	$(LOOP_RUFF) check *.py skill_evolution_loop tests tasks
	$(LOOP_RUFF) format --check skill_evolution_loop tests tasks
