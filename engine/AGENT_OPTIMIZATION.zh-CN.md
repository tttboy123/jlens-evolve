# JLens → Agent 优化：当前实现、证据与结论

## 1. 目标与边界

本轮目标不是让 JLens 直接控制模型输出，而是建立一条可审计链路：

1. 从真实演化谱系收集 JLens 与 logit-lens 观测；
2. 折叠重复源码迁移，避免把重复样本当成独立证据；
3. 将观测编译为有哈希、有限权限的 Agent 候选策略；
4. 固定 evaluator、隐藏集、初始程序、模型和候选预算，做 control/treatment A/B；
5. 只有公开集与隐藏集不劣、重复下降、唯一候选不减少且无接受回归时，才允许晋升。

JLens 的权限固定为 `observational_not_causal`。它不能参与候选准入、改变 evaluator、读取隐藏案例、修改模型权重或直接抑制推理输出。

## 2. JLens 实际发现的规律

校准数据来自 Qwen3.5-4B 的 20 轮真实搜索。修复缺失 trace 后得到 20 条边、21 个程序，但只有 7 个唯一源码迁移：**65% 的边是已出现迁移的重复**。

| 观测 | 数值 | 正确解释 |
|---|---:|---|
| 唯一迁移结果 | 改善 4 / 中性 2 / 回归 1 | 样本很小，只能形成候选假设 |
| JLens 聚类与分数 η² | 0.9953 | 关联强，但谱系边不独立，不能解释为因果 |
| logit-lens 聚类与分数 η² | 0.9971 | 不弱于 JLens |
| JLens / logit-lens 聚类 ARI | 0.8737 | 两种 lens 大体看到同一结构 |
| JLens 增量证据 | 否 | 当前不能证明必须使用 Jacobian 信息 |

唯一改善迁移在后层表现为较平衡的 aggregation、filtering、rounding 变化；唯一回归迁移在 filtering、aggregation、normalization 上同时出现大幅变化。这个模式适合生成“避免宽泛重写、优先小步结构变化”的候选策略，但不具备成功预测或准入权。

## 3. Agent 改动

- `agent_optimizer.py`：从分析产物编译 `agent_strategy.json`，保存证据文件哈希、重复比例、唯一迁移、语义剖面和因果边界。
- `evolve_runtime.py`：只对显式绑定策略的 proposer 追加提示并应用白名单搜索参数；策略文件哈希进入搜索协议与 run manifest。
- 分阶段 manifest：恢复搜索时累计 `iterations_requested_total`，并记录完整 `operator_policy_schedule`，防止把两段搜索伪装成单一策略。
- `agent_ab_report.py`：检查匹配契约、策略绑定、公开/隐藏性能、重复率、唯一源码和接受回归，自动给出 `approved` 或 `rejected`。
- JLens 采集修复：新增 `--max-seq-len`，本轮以 640 token 上限完成 21 个程序观测，无截断。
- `novelty_proxy.py`：项目本地、OpenAI-compatible 的固定预算 proposal controller；每个外层候选严格调用两次模型，control 返回首案，treatment 只在首案重复或搜索停滞时进行一次有界重提案。
- `proposal_controller.py`：限制 controller 权限，校验模式、调用预算、配置哈希、实现哈希和 endpoint 绑定；JLens 仍为 observer-only。
- `novelty_ab_report.py`：增加固定调用预算、停滞检测器版本、结构重复率、唯一 AST/行为、公开/隐藏非劣和零接受回归的联合晋升门。

## 4. 两轮正式 A/B

| arm | 策略时机 | 公开集 | 隐藏集 | 唯一源码 | 唯一 AST | 精确重复 | 接受回归 | 决定 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| control | 10 轮 focused | 11/13 | 4/6 | 6 | 5 | 1/10 | 0 | 基线 |
| v1 | 10 轮全程 JLens-guided | 11/13 | 3/6 | 7 | 7 | 2/10 | 0 | rejected |
| v2 | 5 轮 focused → 5 轮 JLens-guided | 11/13 | 3/6 | 7 | 5 | 1/10 | 0 | rejected |

v1 说明从冷启动开始施加停滞期策略会损害泛化。v2 修正了介入时机，守住了公开集并把精确重复恢复到 control 水平，但仍未守住隐藏集，也没有提高 AST 或行为多样性；后半程 5 个提案中有 4 个被标记为 AST 重复。

因此当前策略 ID `jlens-agent-683e918e61e8` 继续保持 `candidate`，正式晋升决定为 `rejected`。这不是“JLens 优化成功”，而是优化闭环成功识别并阻止了一个看似增加新颖性、实际损害泛化的策略。

### 4.1 重复感知 proposer 正式 A/B

第二阶段把上述诊断落实为 Agent 控制器，不修改上游 OpenEvolve。两组都固定为 10 个外层候选、每候选 2 次 Qwen3.5-4B 调用；固定 evaluator、初始程序、隐藏集、模型和搜索配置。

| arm | 状态机 | 公开集 | 隐藏集 | 结构重复 | 唯一 AST | 唯一行为 | 模型调用 |
|---|---|---:|---:|---:|---:|---:|---:|
| shadow-control v2 | 行为不生效 | 11/13 | 3/6 | 2/10 | 5 | 3 | 20 |
| duplicate-aware v2 | `parent-relative-v1`，无效 | 11/13 | 4/6 | 2/10 | 5 | 3 | 20 |
| duplicate-aware v3 | `global-best-v2`，有效 | 11/13 | 4/6 | 2/10 | 5 | 3 | 20 |

v3 首案重复 6 次，执行 6 次有反馈的有限重提案，修正后的停滞状态触发 4 次。一次重提案确实生成了新源码/AST，但 evaluator 判为无效；停滞后的多次重提案仍复现相同 AST。最终结构重复率没有低于 control，唯一 AST 和行为也没有增加，所以联合门判为 `rejected`。隐藏集从 3/6 到 4/6 的提升保留为观察结果，但不能归因为结构去重，因为预声明的机制指标没有改善。

v3 复用 v2 shadow-control，而没有再次消耗 20 次相同行为调用。理由是 shadow arm 无条件返回第一次 completion，第二次 completion 和停滞状态均不影响返回值；该不变式有单元测试覆盖。正式报告仍逐项核对 run、controller、配置、evaluator、模型、外层候选数和调用数。

### 4.2 结构化 AST Mutation v4 正式三 Seed A/B

v4 将自然语言“不要重复”改成 `MutationPlan → 白名单 AST operator → 确定性 scaffold → 受限 repair → 后置条件检查`。三组独立 seed 均固定每个 arm 10 个候选、每候选 2 次模型调用。

| seed | control 公开/隐藏 | treatment 公开/隐藏 | 结构重复 | 唯一 AST | 结果 |
|---|---:|---:|---:|---:|---|
| 20260802 | 3/13，0/6 | 6/13，0/6 | 9/10 → 4/10 | 1 → 6 | pass |
| 20260803 | 4/13，0/6 | 4/13，0/6 | 7/10 → 8/10 | 3 → 2 | fail |
| 20260804 | 4/13，0/6 | 4/13，0/6 | 7/10 → 8/10 | 3 → 2 | fail |

pooled 重复率从 `23/30` 降到 `20/30`，唯一 AST 从 7 增到 10，但只有 1/3 seed 的重复率不高于 control，未达到预声明的 2/3 稳定性门，因此正式决定为 `rejected`。

operator 证据进一步定位了原因：`canonicalize_before_predicate` 22 次全部 evaluator 有效，但形成 17 次结构重复；status 修复后的 8 次 identity 自由编码全部 evaluator 无效。模型生成的 30 个计划没有一个通过 JSON schema，实际依赖公开 failure 的确定性映射兜底。

## 5. 当前问题的原因

1. **主要是 proposer 的模式坍缩。** 4B 模型能够快速得到 11/13，但在剩余 status 规范化和非法 amount 两组错误上反复生成结构等价程序。提示它“不要重复”不足以改变行为。
2. **结构化 operator 提高了可执行性，但单一 scaffold 形成了新坍缩。** status operator 可靠，却在相同父 AST 和失败上反复生成同一结果。
3. **评估分辨率有限。** 多个程序公开分同为 11/13，但隐藏分不同；仅凭公开 best 无法选择泛化更好的同分程序。
4. **JLens 证据量不足且没有增量优势。** 折叠后仅 7 个唯一迁移，而且 logit-lens 的分数关联略高。当前不能把结果归因于 Jacobian 信息。
5. **不是简单换 coder 就能解决。** 已有 Qwen2.5-Coder-7B 对照在该任务上仅得公开约 0.4625、隐藏 0，弱于 Qwen3.5-4B；模型名称或参数量不能替代匹配评测。
6. **operator 覆盖不足。** status 后的 `normalize_user_identity` 没有结构化 operator，回退自由编码后 8/8 无效。
7. **当前 4B planner 不遵循 JSON schema。** 三 seed 共 30 次模型计划解析成功为 0，不能把 fallback 的效果归因为模型规划能力。

## 6. 已实施的优化与下一步

本轮已经完成 **duplicate-aware proposer loop**：

1. 提案后、正式 evaluator 前计算源码与 AST 指纹；
2. 若与 prompt、本轮 archive 重复或全局最佳停滞，最多进行 1 次 bounded re-proposal；
3. retry 只反馈结构重复类型，不提供隐藏案例，也不读取 evaluator 分数作为正确性信号；
4. 将所有重试计入模型调用预算，分别报告 proposal novelty yield 与最终任务性能；
5. 运行前校验 controller 配置与代理实现哈希，运行后由联合门自动拒绝未降低重复的策略。

结构化 mutation v4 已实现并完成三 seed A/B，但没有通过稳定性门。下一版需要补齐 identity、空身份、聚合与数值验证 operator，并按 `(parent AST, failure, operator, variant)` 去重和轮换；已出现的 selected AST 不应再次提交 evaluator。随后重新跑三个独立 seed，先证明 proposer 策略稳定，再进入下一窗口 RSI 和相邻任务 PSI A/B。

## 7. 关键产物

- 观测与编译策略：`analysis/agent-baseline/`
- control：`runs/agent-ab-control/`
- 全程介入 v1：`runs/agent-ab-jlens-treatment/`
- 分阶段 v2：`runs/agent-ab-jlens-staged-treatment/`
- 正式机器报告：`../../outputs/jlens-agent-optimization-ab-2026-08-02.json`
- 正式中文报告：`../../outputs/JLens驱动Agent优化报告-2026-08-02.md`
- 重复感知 v2 报告：`../../outputs/JLens重复感知Agent优化报告-2026-08-02.md`
- 修正状态机后的 v3 报告：`../../outputs/JLens重复感知Agent优化-v3报告-2026-08-02.md`
- v3 运行证据：`runs/agent-ab-novelty-v3-treatment/`
- 结构化 mutation 完整中文说明：`STRUCTURED_MUTATION_RSI_PSI.zh-CN.md`
- v4 正式三 seed 报告：`../../outputs/JLens结构化Mutation-v4正式三Seed报告-2026-08-02.md`
- v4 pooled operator evidence：`runs/agent-ab-structured-v4-formal/operator_evidence.json`

结论：已经完成“JLens 发现规律 → 结构化 Agent mutation → 三 seed 固定预算 A/B → pooled operator evidence → RSI/PSI 候选 → 自动拒绝不稳定优化”的闭环；尚未证明 JLens 能提高任务泛化。下一技术瓶颈是 operator 覆盖和 archive-aware variant 调度。
