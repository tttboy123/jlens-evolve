# EVAL：structured mutation proposer v4

## Failure Capture

- v3 的有限自然语言重提案在 10 个候选中触发 6 次，结构重复仍为 `2/10`。
- 一次重提案形成新 AST 但 evaluator 无效；停滞触发后仍会原样复现 AST。
- 根因假设：4B proposer 对软提示的条件响应不足，必须把结构变化变成机器可验证的协议。

## Capability Evals

1. Call 1 只产生或解析 `MutationPlan`；plan 只能引用公开 target failure。
2. `canonicalize_before_predicate` 对 status 判断形成可验证的 `strip/lower` AST 变化。
3. `finite_numeric_guard` 在 amount 消费前加入 bool、数值类型、有限性和正值保护。
4. Call 2 只修复受限 scaffold；若删除 operator 后置条件或恢复旧 AST，回退到确定性 scaffold。
5. `planner-control` 与 `structured-mutation` 都严格使用两次模型调用。
6. audit 记录 plan、operator、来源、AST 前后哈希、repair 是否合规和选择来源，不保存 hidden 数据。
7. controller 配置哈希和实现哈希在运行前绑定；JLens 保持 observer-only。

## Regression Evals

1. v3 的 shadow-control、duplicate-aware 和 global-best 停滞行为不变。
2. admission、公开 evaluator、隐藏 evaluator、模型和初始程序不变。
3. hidden/holdout 不进入 planner、operator 选择或 repair prompt。
4. 现有测试、Ruff 和 py_compile 全部通过。

## v4 Matched A/B

- control：`planner-control-v4`，Call 1 自由计划、Call 2 自由编码。
- treatment：`structured-mutation-v4`，Call 1 结构化计划、确定性 AST scaffold、Call 2 受限 repair。
- 每个 seed、每个 arm：10 个候选，每候选 2 次调用。
- 固定模型、温度、top-p、max tokens、evaluator、initial program、admission 和 hidden partition。
- 正式结论至少需要 3 个独立 seed；单 seed 只能作为 capability trial。

## Promotion Gate

- pooled 结构重复率严格下降，且至少 2/3 seed 不高于 control；
- 唯一 AST 和唯一行为不低于 control；
- 公开与隐藏性能非劣；
- 接受回归为 0；
- operator 后置条件执行率可审计；
- 不满足任一条件即 `rejected`。

## RSI Gate

- 从已完成窗口生成版本化 operator evidence 和下一版权重候选；
- 新权重只能在下一个固定预算窗口生效；
- 下一窗口 improvement yield 与任务非劣门同时通过，才记为 operator-level RSI。

## PSI Gate

- 仅持久化通用 operator schema、适用条件和聚合证据，不保存目标任务代码；
- 新进程恢复只算持久化基础；
- 在相邻数据清洗任务做 transfer/off matched A/B 且隐藏非劣，才记为跨任务 PSI。

## Formal Evidence

- seeds：`20260802`、`20260803`、`20260804`；每个 arm 共 30 个候选、60 次模型调用。
- pooled 结构重复：control `23/30`，treatment `20/30`，严格下降。
- 单 seed 结构重复非升：仅 `1/3`，未达到 `2/3`。
- 唯一 AST 逐 seed 求和：`7 → 10`；唯一行为：`5 → 8`。
- 三个 seed 公开与隐藏性能均非劣；接受回归为 0。
- 正式决定：`rejected`。
- RSI：pooled policy 仍为 candidate，`rsi_pass=false`。
- PSI：只生成通用 skill candidate，尚无相邻任务 transfer/off A/B。
