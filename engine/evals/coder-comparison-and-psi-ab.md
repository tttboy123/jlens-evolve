# EVAL：Coder 模型对照与跨任务 PSI A/B

## 模型对照

唯一自变量是 proposer 模型：

- control：Qwen3.5-4B MLX 4-bit；
- treatment：Qwen2.5-Coder-7B-Instruct MLX 4-bit。

两组必须使用同一初始程序、公开/隐藏 evaluator、20 轮预算、随机种子、`focused-v1`、temperature、top-p、max tokens、MAP-Elites、准入规则和经验模式。比较最终公开/隐藏分数、接受率、回归数、唯一行为数和耗时。

## 第二任务能力评测

第二任务使用不同 `task_id`，但保持 `record-cleaning` 任务族。它要求清洗 settled USD payout 记录：字段规范化、过滤、数值验证、按 account 聚合、两位小数和稳定排序。

成功条件：

- 公开与隐藏案例互不泄漏；
- 参考实现通过全部公开与隐藏案例；
- 初始实现不是满分；
- 运行器的 task-core 路径进入 evaluator 契约哈希；
- 第二任务不会调用第一任务的隐藏评分器。

## PSI A/B

- control：`experience_mode=off`；
- treatment：`experience_mode=cross-task`；
- 其他变量完全一致；
- treatment 必须记录来自不同 task ID 的 lesson 来源；
- A/B 契约哈希、task、model、iteration budget 必须一致；
- treatment 隐藏集最终分数和相对初始增益均不得低于 control；
- 是否取得严格正收益单独报告，不用非劣结果冒充显著提升。

## 回归门槛

- 原交易清洗任务全部测试继续通过；
- 拒绝候选仍不可进入 island/archive/best；
- 原有 checkpoint 兼容检查继续有效；
- JLens 仍为只读观测侧车。

