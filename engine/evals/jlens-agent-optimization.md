# EVAL：JLens 驱动的 Agent 优化闭环

## 目标

把 JLens 的离线观测转成一个可审核的 Agent 搜索策略候选，并通过匹配 A/B 决定是否晋升。JLens 不直接决定候选正确性、准入或模型权重更新。

## Capability evals

1. 策略编译器按唯一源码迁移折叠重复谱系，不能把重复采样当独立证据。
2. 策略记录 JLens、logit-lens、trace 和分析摘要哈希，并明确 `observational_not_causal`。
3. 若 JLens 没有优于 logit-lens，策略只能是 `candidate`，不得作为候选准入门。
4. 运行器只在选中带 `agent_strategy_file` 的 policy 时注入策略；control 不得读取该策略。
5. 允许的自动干预仅限 proposer 提示和搜索多样性参数，不能修改 evaluator、hidden cases 或 admission 规则。

## Agent A/B 契约

- 相同 task、initial、evaluator、OpenEvolve config、model 和迭代预算；
- control 使用 `focused-v1`；treatment 使用 `jlens-guided-v1`；
- 两组均关闭跨任务经验；
- 独立输出和 state 目录，禁止跨 arm 污染；
- 报告公开集、隐藏集、唯一源码/AST/行为、精确重复率和接受回归。

## 晋升门槛

以下条件必须同时满足：

1. treatment 隐藏集最终分数和增益不劣于 control；
2. treatment 公开最佳分数不劣于 control；
3. treatment 精确重复率低于 control，且唯一源码数不低于 control；
4. 两组接受后的父代回归均为 0；
5. A/B 契约匹配，策略来源哈希可追溯。

若任一条件失败，实验仍算完成，但策略保持 `candidate/rejected`，不能描述为 JLens 已经优化了 Agent。
