#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${1:?usage: run-r075-calibration.sh REPO_ROOT}
RUN_ROOT="$REPO_ROOT/artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop"
MANIFEST="$RUN_ROOT/aws-migration-r075/CALIBRATION-MANIFEST.json"
CUDA_ROOT="$RUN_ROOT/aws-migration-r075/cuda"
MODEL_PATH="$REPO_ROOT/models/Qwen3.5-4B-mlx-4bit"
PYTHON=${PYTHON:-"$REPO_ROOT/.venv/bin/python"}
test -x "$PYTHON"

"$PYTHON" -m skill_evolution_loop cuda-calibration-replay \
  --manifest "$MANIFEST" \
  --out "$CUDA_ROOT/prompt-replay" \
  --transport-base-url http://127.0.0.1:18000 \
  --transport-model same-base-cuda \
  --max-tokens 1536 \
  --json

"$PYTHON" -m skill_evolution_loop round1-feedback-run \
  --manifest "$RUN_ROOT/round3-sharded-30x30-taskset-v2/TASKSET.json" \
  --routes "$RUN_ROOT/round3-sharded-30x30-taskset-v2/MECHANISM-ROUTES.json" \
  --target-audit "$RUN_ROOT/round3-sharded-30x30-target-audit/TARGET-COVERAGE.json" \
  --operator-skill "$RUN_ROOT/round3-r074-inactive-skills/OPERATOR-SKILL.json" \
  --span-skill "$RUN_ROOT/round3-r074-inactive-skills/SPAN-SKILL.json" \
  --model "$MODEL_PATH" \
  --out "$CUDA_ROOT/formal-runner" \
  --workspace /private/tmp/jlens-cuda-calibration-workspaces \
  --task round1-apache__lucene-13170 \
  --task round1-laravel__framework-52684 \
  --task round1-phpoffice__phpspreadsheet-3903 \
  --realization-candidates 1 \
  --shared-diagnosis-localization \
  --transport-base-url http://127.0.0.1:18000 \
  --transport-model same-base-cuda \
  --shared-context-source "$RUN_ROOT/round3-r074-locale-role-feedback-smoke" \
  --json

"$PYTHON" -m skill_evolution_loop cuda-calibration-collect \
  --manifest "$MANIFEST" \
  --experiment "$CUDA_ROOT/formal-runner" \
  --out "$CUDA_ROOT/CUDA-RESULTS.json" \
  --json

"$PYTHON" -m skill_evolution_loop cuda-calibration-evaluate \
  --manifest "$MANIFEST" \
  --cuda-results "$CUDA_ROOT/CUDA-RESULTS.json" \
  --out "$CUDA_ROOT/CALIBRATION-REPORT.json" \
  --json
