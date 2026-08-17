# 历史监督演化 POC

> 本文只保留早期确定性实验记录，不再承担产品路线图职责。当前后台入口见
> `BACKEND_POC.zh-CN.md`，产品范围与版本以 `ROADMAP.zh-CN.md` 为准。

## 实验目的

早期 POC 用真实 `initial_program.py`、既有 JLens 观察和两个白名单 AST operator，验证
最小的“观察 → 候选 → evaluator → Gate”链路。它不调用网络、LLM 或在线 JLens。

```text
既有 JLens 观察
  → Rule Supervisor
  → 两个确定性 task-program operator
  → public evaluator
  → post-search holdout
  → POC candidate
```

## 实测结果

- JLens 历史观察：65% 搜索迁移重复；证据边界为 `observational_not_causal`。
- public evaluator：`3/13 → 4/13 → 6/13`。
- sealed holdout：`0/6 → 0/6`。
- 每一步丢失的父代公开案例：0。
- 停止原因：剩余公开失败没有白名单 operator。
- `task_solved=false`、`production_ready=false`。

后台 POC 已修复旧实现中 holdout 逐步参与 Gate 的问题：搜索阶段只使用 public/dev，
baseline/final holdout 都在搜索停止后审计。具体证据见 `BACKEND_POC.zh-CN.md`。

## 这项实验能证明什么

1. Evolve Kernel 可以消费观察产物并执行有界 mutation。
2. 确定性 operator 对局部 public 用例有可重放收益。
3. evaluator 可以阻止父代公开能力回归。
4. 收敛、任务解决和生产晋升可以被明确区分。

## 不能证明什么

1. 任务程序提升不等于 Agent 自身得到优化。
2. hidden `0/6` 不能证明泛化。
3. JLens 只影响实验方向，不能据此声称因果增益。
4. 两个固定 operator 不构成开放 RSI。
5. 本实验与模型训练无关，也不生成训练模型的完成条件。

## 后续位置

该 POC 作为 `v0.1.0` Kernel smoke 保留。下一产品切片是 `v0.2.0 AgentProgram`：冻结
模型，只演化 Prompt、Skill selection 和 retry policy，并用相同模型、任务、预算与 seed
进行 baseline/evolved A/B。

JLens 在线 Sidecar、跨任务 Skill Library、DGM 式 Agent code mutation 和 RQGM 式
evaluator shadow 的顺序见 `ROADMAP.zh-CN.md`。
