# EVAL：duplicate-aware proposer v3

## Failure Capture

- Session / task：JLens 发现规律后优化 Agent 搜索。
- Goal in progress：减少结构等价提案，同时保持公开集和隐藏集性能。
- Error：提示级 JLens-guided 策略未晋升。
- Last successful step：分阶段 arm 完成固定 10 候选 A/B。
- Repeated pattern seen：后半程 5 个提案中 4 个被 admission 标记为 AST 重复。
- Environment：Qwen3.5-4B、本地 MLX、固定 evaluator、单 worker。

## Root Cause

JLens 观测到的 65% 重复迁移被转换成提示词，但 4B proposer 只改变了文本表面，没有稳定改变 AST 或行为。问题属于 Agent 提案选择策略，而不是 JLens 应拥有正确性或准入权限。

## Capability Evals

1. 每个外层提案固定产生两次上游 completion；control 与 treatment 调用数相同。
2. control 始终返回第一次 completion，第二次只作 shadow，不改变基线行为。
3. treatment 仅当第一次与 prompt/archive 的源码或 AST 重复时，向第二次请求追加有限、无隐藏信息的重复反馈，并优先返回第二次。
4. controller 只使用源码/AST 指纹，不读取 evaluator 分数、隐藏案例或 JLens cluster 作为正确性信号。
5. 每次请求记录 first/second/selected 指纹、重复原因、选择序号、调用数和延迟，不保存完整 prompt 或代码。
6. controller ID、模式、配置哈希和固定调用数写入 run manifest，并由运行前 endpoint 绑定检查验证。

## Regression Evals

1. evaluator、evaluator core、initial program、模型和外层候选数不变。
2. admission 的非回归、源码/AST/行为去重规则不变。
3. JLens 继续是 `observational_not_causal`，不能拒绝候选或修改模型权重。
4. 现有测试全部通过。

## Matched A/B Promotion Gate

- control 与 treatment 每个外层候选均为 2 次上游调用；总请求数和总上游调用数完全相同；
- treatment 公开与隐藏分数均不劣于 control；
- treatment 的 `(exact_duplicate + ast_duplicate) / candidate_attempts` 严格下降；
- treatment 唯一 AST、唯一行为均不低于 control；
- 两组接受回归均为 0；
- 不满足任一条件即 `rejected`，不得晋升。

## Recovery Action

- Smallest action：项目本地 OpenAI-compatible proposal proxy，不修改上游 OpenEvolve checkout。
- Safety：只在 proposer 响应选择层工作；固定两次调用；不接触 hidden/evaluator/admission。
- Proof：单元测试、manifest 绑定、proxy 审计、匹配 A/B 和完整回归测试。

## Introspection：状态机缺陷与恢复

- Failure：初版停滞检测把 `child_score > parent_score` 记为改进；从较差父代恢复到已存在的历史最佳也会清空停滞状态。
- Root cause：比较基准错误地使用局部父代，而不是运行截至当前的公开集全局最佳。
- Containment：保留 v2 运行与 `parent-relative-v1` 标记，不把它作为停滞状态机证据；新增回归用例覆盖“恢复旧最佳不算改进”。
- Recovery：按时间顺序维护 running global best，只有 accepted child 严格超过此前全局最佳才记改进；版本升级为 `global-best-v2`，写入配置、endpoint、proxy stats 和正式 A/B gate。
- Runtime proof：v3 在连续三个候选没有超过 `11/13` 后触发，10 个候选中共触发 4 次；完整 trace 恢复为 10/10。

## Final Outcome

- control：20 次模型调用，公开 `11/13`，隐藏 `3/6`，结构重复 `2/10`，唯一 AST 5。
- treatment v3：20 次模型调用，公开 `11/13`，隐藏 `4/6`，结构重复 `2/10`，唯一 AST 5。
- 判决：性能非劣，但结构新颖性未改善，`rejected`；不得晋升为默认 proposer 策略。
