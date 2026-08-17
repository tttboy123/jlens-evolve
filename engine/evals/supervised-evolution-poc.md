# EVAL: supervised-evolution-poc

## POC 目标

用现有 JLens 观察产物、结构化 mutation 和固定 evaluator 跑通一条最小闭环：

```text
观察证据 -> Supervisor 计划 -> RSI 候选 -> 固定公开/隐藏评测 -> Gate
```

## 能力评测

- [x] 读取真实 `analysis/agent-baseline/agent_strategy.json`，保留
      `observational_not_causal` 边界。
- [x] Supervisor 只选择白名单结构化 operator，不调用网络或模型。
- [x] 使用真实 `initial_program.py` 和 `evaluator_core.py`。
- [x] 公开集通过数从 `3/13` 提升到至少 `4/13`。
- [x] 每一步不得丢失父候选已经通过的公开案例。
- [x] 最终隐藏集通过数不得低于 baseline。
- [x] 输出 JSON、中文 Markdown、HTML 和最终候选源码。
- [x] Gate 只允许 `poc_candidate_accepted`，明确 `production_ready=false`。

## 回归评测

- [x] 现有完整 pytest 通过。
- [x] Ruff check 通过。
- [x] Ruff format check 通过。
- [x] 报告重建不需要调用 LLM、OpenEvolve 或 JLens 模型。

## 验证命令

```bash
.venv/bin/python -m pytest -q tests/test_supervised_evolution_poc.py
.venv/bin/python supervised_evolution_poc.py
.venv/bin/python -m pytest -q
.venv/bin/ruff check *.py tests tasks
.venv/bin/ruff format --check *.py tests tasks
```

运行记录见 `supervised-evolution-poc.log`。
