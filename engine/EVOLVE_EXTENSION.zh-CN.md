# 受保护的 Evolve 扩展：中文技术说明

## 1. 项目解决了什么问题

原实验把四个概念混在了一起：

1. 候选程序是否被执行和评分；
2. 候选是否有资格进入种群并成为后续父代；
3. 搜索证据是否能跨进程、跨任务保存；
4. 一次分数提升是否足以称为“自改进”。

本扩展在项目本地包装 OpenEvolve，不修改上游 checkout，并把这四层拆开。核心原则是：**评估过不等于允许繁殖；保存过不等于已经学会；候选变好不等于 RSI。**

## 2. 总体架构

```text
候选生成器（可替换的 Qwen3.5-4B / Qwen2.5-Coder-7B）
          │
          ▼
确定性公开评估器 ──► 每个案例的独立指标
          │
          ▼
候选准入层
  ├─ 父代已通过案例不得回归
  ├─ 静态执行边界
  ├─ 源码与 AST 去重
  └─ 行为等价变体限额
          │
    ┌─────┴─────┐
    │           │
  接受         拒绝
    │           │
    ▼           └─► 仍写入 checkpoint、trace、持久 archive
island / MAP-Elites / best
    │
    ├─► 隐藏集验证 ──► lessons.jsonl ──► 本地候选技能
    └─► RSI / PSI 证据聚合

JLens：独立观测侧车，只分析搜索轨迹，不参与以上决策。
```

## 3. 持久化能力

| 能力 | 实际持久化形态 | 接受或晋升规则 |
|---|---|---|
| 同一搜索恢复 | OpenEvolve checkpoint + `run_manifest.json` | 任务、初始程序、评估器、配置哈希必须一致 |
| 全量尝试档案 | `state/archive.jsonl` | 追加写、事件 ID 幂等、文件锁保护 |
| 搜索种群 | checkpoint programs、islands、archive、MAP cells | 父代案例非回归 + 源码/AST/行为去重 |
| 可复用经验 | `state/lessons.jsonl` | 已接受改进、隐藏集不回归、重复证据达到阈值 |
| 可审核程序记忆 | `state/skills/<family>/SKILL.md` | 仅生成 `status: candidate`，不自动全局安装 |
| 元策略证据 | `state/meta_policy_trials.jsonl` | 比较搜索算子修改前后的有效改进率 |
| 运行证据 | `state/run_manifests.jsonl` 与运行目录报告 | 不可变执行记录，运行目录保留最新 manifest |

与其他 evolve 系统的对应关系：

| 项目 | 同一搜索跨进程恢复 | 跨任务经验复用 | 公开持久化形态 |
|---|---:|---:|---|
| autoresearch | 手工恢复 | 无自动迁移 | 专用 Git branch、保留 commits；`results.tsv` 通常不提交 |
| AlphaEvolve | 未公开 | 未公开 | 论文描述内部 programs database |
| OpenEvolve 上游 | 支持 | 不支持 | checkpoint、程序数据库、archive、metadata |
| AutoResearchClaw + MetaClaw | 支持 | 可选 | `lessons.jsonl` 转为可审核技能文件 |
| DGM | 支持 | 限于同一演化谱系 | Git organisms、节点 metadata、JSONL archive |
| Hermes Self-Evolution | 部分 | 通过人工合并后的技能 | 验证报告、branch、人工审核 PR |
| 本扩展 | 自动兼容恢复 | 按任务族复用 | checkpoint + manifest + archive + lessons + 本地候选技能 |

## 4. 搜索与评估修复

### 4.1 逐案例评分

评估器为 13 个公开案例分别返回数值指标。综合 fitness 使用词典序思想构造：多通过一个案例产生的收益，始终大于旧分组权重差异；旧 weighted score 只保留为平局裁决和诊断指标。

这修复了旧设计中“不同失败组合得到相同总分”的问题，也让父代非回归检查能够定位到具体案例。

### 4.2 候选准入

候选进入 island、archive 或 best 之前必须同时满足：

1. 保留父代已经通过的全部公开案例；
2. 通过静态 evaluator 边界；
3. 与已接受候选的源码哈希和 AST 哈希不同；
4. 同一公开行为签名的结构变体没有超过配置限额。

拒绝候选不会消失：它仍保留在程序数据库、checkpoint、持久事件档案和 evolution trace 中，但不能成为后续父代。这既保留失败证据，也防止种群被等价或回归候选污染。

### 4.3 MAP-Elites 与提示词

MAP-Elites 的两个维度改为：

- `case_pass_rate`：公开案例通过率；
- `ast_complexity`：AST 节点复杂度。

不再把文本长度、字符集差异当作有价值的行为多样性。提示词只包含一个已接受参考、已通过案例集合和一个目标失败，初始 system + user 提示约 2.6K 字符，适合本地 4B 模型上下文。

### 4.4 隐藏测试

新增 6 个确定性隐藏案例：

- 输入排列不变性；
- 插入畸形数据后的结果不变性；
- status 与 user 的规范化；
- 整数/浮点等价，以及 bool、非有限数拒绝；
- 拆分/合并记录后的聚合一致性；
- 输出唯一、有限、稳定排序。

这些案例的 ID、输入和逐项结果不会返回 proposer。隐藏集只用于搜索后的验证、经验晋升和 RSI/PSI 证据。

## 5. 经验提炼与跨任务复用

只有满足公开改进且隐藏集不回归的接受事件，才有资格写入经验档案。相同任务族中重复出现的验证经验达到阈值后，可以生成：

```text
state/skills/<task-family>/SKILL.md
```

该文件始终标记为 `status: candidate`，需要人工审核；运行器只把匹配任务族的经验检索到下一次运行，不修改全局 Codex/Agent 技能目录。

当前 `record-cleaning` 任务族已有 1 条达到复用阈值的经验。正式 A/B 中，第二任务 `payout-record-cleaning-v1` 的 transfer arm 已检索到来自 `transaction-record-cleaning-v1` 的经验，来源记录在 `retrieved_lessons.json` 和 manifest 中。检索链路得到验证，但目标隐藏集从 control 的 `4/6` 降到 `0/6`，因此是负迁移，不能晋升为有效 PSI 经验。

## 5.1 JLens 观测到 Agent 策略

`agent_optimizer.py` 将折叠重复迁移后的 JLens 观测编译成项目内的候选策略。编译产物包含输入文件哈希、唯一迁移清单、语义剖面、白名单搜索参数与 `observational_not_causal` 边界。`evolve_runtime.py` 只在显式绑定策略时改变 proposer，并把策略 ID、哈希和分阶段 `operator_policy_schedule` 写入 manifest；evaluator、隐藏集和 admission 保持固定。

正式结果中，20 条真实谱系边只有 7 个唯一源码迁移，重复比例为 65%。全程 JLens-guided 与“5 轮 focused → 5 轮 JLens-guided”两种 treatment 都保持公开集 `11/13`，但隐藏集均为 `3/6`，低于 control 的 `4/6`。因此策略 `jlens-agent-683e918e61e8` 没有晋升。详细证据见 `AGENT_OPTIMIZATION.zh-CN.md`。

## 6. RSI 与 PSI 评测定义

缩写存在多种用法，本项目固定采用以下工程定义。

### 6.1 RSI：递归自改进

候选分数提升还不够。RSI 必须同时满足：

- 存在候选增益；
- 存在至少两代的严格连续改进链；
- 被接受候选没有父代案例回归；
- 改进算子或搜索策略本身发生修改；
- 修改后的有效改进率高于修改前。

最后一条用于区分“程序被反复优化”和“系统改善了自身的改进能力”。

### 6.2 PSI：持久自改进

PSI 分为两个独立子项：

1. **同一搜索恢复：**进程重启后，checkpoint、最佳公开分数与隐藏分数得到保留；
2. **跨任务迁移：**另一个 task ID 产生、经隐藏集验证的经验被目标任务检索；control 与 transfer 的任务、模型、预算、初始程序、evaluator 和搜索协议必须匹配；transfer 的隐藏集最终分数与增益均不得劣于 control，且自身增益不得为负。

只有两个子项都有真实证据，总 PSI 才通过。系统不会用“文件已经保存”替代“经验迁移有效”。

## 7. 20 轮实测结果

运行目录：`runs/repaired-smoke-v2`

| 指标 | 初始程序 | 最佳/最终程序 |
|---|---:|---:|
| 公开案例 | 3/13 | 11/13 |
| 公开综合分数 | 0.2304 | 0.8464 |
| 隐藏案例 | 0/6 | 3/6 |

搜索统计：

| 项目 | 结果 |
|---|---:|
| 正式搜索迭代 | 20 |
| 因主动中断恢复产生的重放尝试 | 2 |
| 候选尝试总数 | 22 |
| 接受进入种群 | 3 |
| 拒绝 | 19 |
| 接受率 | 13.64% |
| 接受后的父代回归 | 0 |
| 接受后的源码重复 | 0 |
| 接受后的 AST 重复 | 0 |
| 唯一源码 / AST / 行为签名 | 6 / 5 / 3 |
| 隐藏集验证晋升事件 | 2 |

拒绝原因不是互斥候选数统计；一个候选可能命中多个诊断条件。持久报告记录到的原因包括 AST 重复 8 次、行为等价超限 9 次、evaluator 拒绝 1 次、源码重复 1 次、父代回归 1 次。

### RSI 结果

- 候选确实提升：是；
- 严格改进深度：2；
- 接受回归：0；
- 搜索算子调整：2 次；
- 能提高有效改进率的算子调整：0 次；
- **RSI：未通过。**

### PSI 结果

- 同一搜索恢复：通过，4 次恢复实验；
- 跨任务匹配 A/B：已完成；
- control：公开 `10/14`，隐藏 `4/6`，隐藏增益 `+4/6`；
- transfer：公开 `6/14`，隐藏 `0/6`，隐藏增益 `0`；
- 跨任务来源：通过，实际检索自 `transaction-record-cleaning-v1`；
- 隐藏分数和增益相对 control 均为 `-4/6`；
- **跨任务 PSI：未通过；总 PSI：未通过。**

这次结论不再是证据不足，而是匹配实验观测到负迁移。持久化与检索机制工作正常，但当前经验文本和 proposer 协议没有把源任务规律可靠地转化为目标任务改进。

审计边界：最初的 `runs/psi-payout-control` 暴露了 worker `sys.path` 优先级导致的 evaluator 导入冲突，所有迭代均失败，已从 A/B 排除。修复后运行器增加 worker 等价导入预检；正式结论只使用隔离的 `psi-payout-control-v2-20260802` 与 `psi-payout-transfer-v2-20260802`。两组初始 lesson 快照的 SHA-256 均为 `1992e20f14efd5d764accac32f51f265df8d198d3b4618b56e75f733154de171`。

## 8. Coder proposer 固定协议对照

两组均使用源任务、同一 evaluator/initial hash、20 轮预算、`focused-v1`、相同搜索协议哈希，并关闭经验检索；唯一实验变量是 proposer 模型。

| 指标 | Qwen3.5-4B control | Qwen2.5-Coder-7B treatment |
|---|---:|---:|
| 公开案例 | 11/13 | 6/13 |
| 隐藏案例 | 3/6 | 0/6 |
| 接受率 | 15% | 10% |
| 唯一行为签名 | 3 | 1 |
| 运行秒数 | 1400.2 | 651.8 |

本协议下 7B Coder 快约 2.15 倍，但 20 个候选中有 17 个被判定为源码完全重复，质量明显低于 4B control。合理解释是模型与“单一失败目标 + 固定解码”协议发生交互，而不是模型参数量本身决定结果；因此不把这一次对照外推为模型通用排名。

## 9. 当前瓶颈与下一步

最终程序仍失败于：

- `filter_normalized_status`；
- `reject_invalid_amounts`。

Qwen3.5-4B 在取消固定 LLM seed 后仍反复产生相同的 `11/13` 行为。因此，当前瓶颈更像 proposer 的代码修复能力，而不是 archive 污染、评分碰撞或提示词过长。

本轮已完成原定的模型对照、第二任务和 PSI A/B。新的最有价值步骤是：

1. 保持 evaluator 不变，单独对解码多样性与父代选择做消融，确认重复候选来自固定解码还是 prompt 约束；
2. 将经验从一段抽象文本升级为带适用条件、反例和目标字段映射的结构化 lesson；
3. 在 proposer 生成前增加目标任务差异检查，避免把源任务字段名和局部修复机械迁移；
4. 重新跑独立 seed 的匹配 PSI A/B；只有 transfer 在隐藏集上不劣于 control 才通过，严格提升继续单独报告。

## 10. 运行命令

Qwen 短代码任务需要关闭 thinking，否则模型可能把 completion 预算全部花在隐藏推理上并返回 `content: null`：

```bash
.venv/bin/mlx_lm.server \
  --model models/Qwen3.5-4B-mlx-4bit \
  --host 127.0.0.1 --port 18080 \
  --temp 0.85 --top-p 0.95 --max-tokens 512 \
  --chat-template-args '{"enable_thinking":false}'
```

新建 5 轮搜索：

```bash
.venv/bin/python evolve_runtime.py \
  --output runs/my-run --iterations 5 --resume none --run-id my-run
```

自动恢复并增加 15 轮：

```bash
.venv/bin/python evolve_runtime.py \
  --output runs/my-run --iterations 15 --resume auto
```

注意：`--iterations` 表示从 checkpoint 起额外执行的轮数，不是总轮数。自动恢复会拒绝任务、评估器、配置或初始程序契约发生变化的 checkpoint。

仅重建报告：

```bash
.venv/bin/python evolve_runtime.py \
  --output runs/my-run --report-only
```

## 11. 安全与解释边界

- evaluator 使用 AST 筛查和受限 Python builtins，只是任务执行边界，不是 OS 安全沙箱。
- 不能用当前机制执行任意不可信代码；这仍需要真正的本地隔离沙箱。
- JLens 是观测和诊断工具，只用于分析候选概念、层和位置；它不决定候选准入，也不控制推理输出或修改模型权重。
- 当前隐藏集提升只说明这一任务上的泛化证据，不能直接外推为通用自改进能力。
