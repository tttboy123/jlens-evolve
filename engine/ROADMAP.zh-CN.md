# Evolve × JLens Agent 自进化路线图

本文是项目范围和产品版本的唯一权威路线图。实验编号（例如 proposer v3、structured
mutation v4）不代表产品版本。

## 1. 产品目标

在一个实验窗口内冻结模型权重，通过 JLens/trace 观察 Agent 行为，使用 evolve 搜索并
验证 Agent 应用层候选，持续改进：

- Prompt 与 demonstrations；
- Skills 的内容、选择和组合；
- tool、context、retry 与 routing policy；
- proposer/search policy；
- 经过独立沙箱验证的 Agent harness code。

模型训练、LoRA、SFT、RL、Trainer backend 和模型部署 registry 不属于当前产品路线。评测用
ModelAdapter registry 只冻结 provider/path/预算，不管理权重，也不改变这一边界。未来
如果应用层优化已经达到可测量上限，再作为独立研究轨讨论，不得反向改变当前边界。

## 2. 不可破坏的边界

```text
模型权重：实验内冻结
JLens：观察和诊断，不参与候选准入
Evolve：提出并搜索应用层候选
Evaluator：固定 epoch 内唯一晋升依据
Supervisor/LLM：只能提出候选，不能直接写 active 版本
Skill：先是项目内 candidate，跨任务验证后才可人工晋升
```

JLens 与 logit-lens 只能生成优化假设。是否有效必须由固定 evaluator、sealed audit 和
matched A/B 证明。

## 3. 演化对象

当前 `candidate.py` 的确定性修改只用于验证 Evolve Kernel。产品级演化对象统一称为
`AgentProgram`：

```text
AgentProgram
├── system_prompt_ref
├── demonstration_refs
├── skill_refs
├── tool_policy_ref
├── context_policy_ref
├── retry_policy_ref
├── routing_policy_ref
└── harness_code_ref
```

任务程序是 Agent 的输出，不等于 Agent 自身。只有上述应用层对象或搜索策略发生可验证
改进，才属于本项目的 Agent 自进化。

## 4. 前沿项目如何映射到本项目

| 参考系统 | 借鉴内容 | 本项目对应模块 | 明确不照搬的部分 |
|---|---|---|---|
| Agent Lightning | Sidecar、Tracing、运行与优化解耦 | Observer | 当前不接 RL/SFT Trainer |
| DSPy | Program、Metric、Optimizer 分离 | `AgentProgram` + Evolve Optimizer | 当前不优化模型权重 |
| AlphaEvolve | evaluator、programs database、进化选择 | Evolve Kernel + Candidate Archive | 不假设未公开的跨 session 协议 |
| DGM | Agent 代码变体、谱系 archive、逐变体验证 | 受限 `AgentCodeMutation` 插件 | 不允许无沙箱的开放自改代码 |
| Voyager | 环境反馈、迭代技能、可执行 Skill Library | PSI Skill Registry | 不假设技能天然可跨任务迁移 |
| RQGM | 固定 evaluator epoch、epoch 边界更新 | Evaluator Shadow | v1.0 前不允许 evaluator 自动晋升 |

## 5. 持久化系统调研结论

“存在文件”不等于“能够自动恢复”，“跨 run 恢复”也不等于“跨任务经验有效”。本项目
采用下面的严格区分：

| 项目 | 同一搜索跨进程恢复 | 跨任务经验复用 | 公开持久化形态 | 对本项目的启示 |
|---|---:|---:|---|---|
| autoresearch | 手工 | 无自动迁移 | 专用 Git branch、commits；`results.tsv` 通常不提交 | Git 历史可审计，但不能替代运行状态协议 |
| AlphaEvolve | 未公开 | 未公开 | 论文描述内部 programs database | 借鉴候选库，不声明未公开能力 |
| OpenEvolve | 支持同一搜索 | 不支持自动跨任务迁移 | checkpoint、program database、archive、metadata | 作为搜索底座，外加本项目准入与经验层 |
| AutoResearchClaw + MetaClaw | 支持 | 可选 | `lessons.jsonl` → `~/.metaclaw/skills/arc-*/SKILL.md` | 借鉴“经验转技能”，但默认只写项目内 candidate |
| DGM | 支持 | 限于同一演化谱系 | Git organisms、节点 metadata、JSONL archive | 每个 Agent 变体保留独立谱系和验证 |
| Hermes Self-Evolution / hermes-evo | 部分 | 通过审核合并后的技能 | 五阶段路线、验证报告、Git branch、人工审核 PR | 借鉴 reviewer/PR 晋升，不自动部署技能 |
| 本项目 | 自动兼容恢复 | 必须通过 matched PSI A/B | SQLite/checkpoint + manifest + archive + candidate skills | 机制持久化与能力迁移分开验收 |

## 6. 产品版本

| 版本 | 核心目标 | 主要交付 | 退出门槛 |
|---|---|---|---|
| `v0.1.0` | 稳定 Evolve Kernel | CLI、SQLite、搜索/holdout 隔离、收敛与 evidence | 同输入 `pass^3`；holdout 不参与搜索；如实报告未解决任务 |
| `v0.2.0` | 首个 Agent 层自进化 | `AgentProgram`、Prompt/Skill/Policy mutation、Replay Supervisor | 固定模型下应用层候选严格提升，sealed 不退化，至少 2/3 seed 非劣 |
| `v0.3.0` | JLens 进入观察闭环 | Trace/JLens/logit-lens Sidecar、ObservationArtifact、matched A/B | 关闭 Observer 不改变运行；JLens 增量价值单独报告 |
| `v0.4.0` | 跨任务 PSI | Verified Skill Library、任务族与经验 lineage | transfer 隐藏结果不劣于 control，严格收益单独报告 |
| `v0.5.0` | 受限 Agent 代码自改 | sandboxed harness mutation、独立回归与谱系 archive | 每个变体可重放、可回滚、无权限扩张 |
| `v0.6.0` | Evaluator Shadow | anchor、cross-play、固定 epoch、人工晋升 | 新 evaluator 只在 epoch 边界且经人工审核切换 |
| `v0.7.0` | 纵向集成 | PluginEnvelope、统一运行与幂等 replay | 单一 admission authority；Observer 故障不改变结果 |
| `v0.8.0` | 恢复与安全加固 | durable operation、lease、crash recovery、资源门 | 并发不双执行；部分写可检测并重建；失败不发布结果 |
| `v0.9.0` | Release Candidate | 三套 RC、clean-room replay、阶段合同审计 | pass^3 一致；完整 raw event 与 public→sealed 顺序证据 |
| `v1.0.0` | 稳定 Agent 自进化服务 | 可恢复、可审计、可回滚的完整应用层闭环 | 多任务长期运行；无隐藏泄漏、无自动越权晋升 |
| `v1.1.0` | 真实 Codex Target 接入 | 真实历史 adapter、首个 Prompt/Skill/Policy ChangeSet、中文报告、正反 patch | runtime sealed 顺序、matched A/B、patch 往返、pass³；明确 live 未证明 |
| `v2.0.0` | MetaProgram 外层演化 | proposer/search/routing 三代谱系、单位成本选择、协议留出审计 | 软件门与 RSI 声明门分离；无 fresh task/live run 时 RSI 必须拒绝 |
| `v2.0.x` | 多模型与外部 benchmark 扩展 | Codex/MLX adapter、冻结 grader、Qwen 4B/7B Coder、SWE-bench adapter | 同任务同预算；adapter bug 与模型失败分离；无官方 harness 不报 resolved-rate |
| `v2.1.0` | 持续多评测集 matched A/B | 300-task 生命周期池、永久 shadow baseline、周期性 ChangeSet、one-shot sealed | 四个 pinned adapter；每轮 matched；任务不复用；60 个最终 sealed；付费执行有明确上限 |
| `v2.1.1` | JLens 多代进化发动机 | Pattern/Advantage Miner、mutation population、archive/lineage、4→2→1 tournament、search-parent selector | JLens observer-only；candidate 同时对 original/parent；负候选保留；pass³；不打开 final sealed |

## 7. 已完成的执行顺序

1. 冻结 `v0.1.0` 后台 Kernel；保留任务程序 mutation 作为确定性 smoke。
2. 定义最小 `AgentProgram` schema，首先开放 Prompt、Skill selection、retry policy。
3. 使用同一模型、任务、预算和 seed 对 baseline/evolved AgentProgram 做 A/B。
4. `v0.2.0` 通过后再接 JLens 在线 Sidecar，并加入 trace-only 与 logit-lens 对照。
5. 只有经跨任务 sealed A/B 验证的经验，才进入项目内 Skill Library。
6. DGM 式 Agent code mutation 和 RQGM 式 evaluator shadow 后置，不与首个闭环并行开发。
7. 完成统一集成、durable recovery、RC 与最终只读 Release 验证。
8. 接入真实 Codex 历史与项目原生表面，生成首个未自动应用的 AgentChangeSet。
9. 完成受限三代 MetaProgram；软件机制通过，但 RSI 因 fresh/live 门缺失而拒绝。
10. 完成首条 live A/B、多模型 diagnostic 和 SWE-bench adapter；结果混合，未晋升模型或 G2。
11. 完成官方 SWE-bench 固定 5-task capability pilot：5/5 resolved、证据回传和云资源回收通过；
    因小样本、task patch 可见且无 matched pass³，不晋升 Agent/profile/RSI。下一步是 fail-closed
    runner 与全新任务上的 baseline/evolved A/B。
12. 完成 fail-closed runner 与 10 个新任务的 Batch 002：官方 9/10 resolved、0 errors；Matplotlib
    负例暴露 4 个既有布局回归。该批只作为 capability/regression evidence；云资源四类对象已
    控制面归零，不保留付费实例。下一阶段才是在另一组新任务上的 baseline/evolved matched pass³。
13. 完成 v2.1 本地持续协议：四个 pinned adapter、300 个全新任务、永久原始 baseline、逐轮 matched
    A/B、8 个 ChangeSet 周期和 60-task one-shot sealed 均已冻结；当前 0 个真实 round，不宣称优化。
    下一步需明确授权 10-round 小流量 pilot，实测后再决定完整 300-round 预算。
14. 将固定 candidate A/B 修正为 OpenEvolve 式多代搜索：v2.1.1 已以本地 pass³ 验证 Observer→
    PatternCard→4 个 inactive mutations→4→2→1 tournament→experimental parent 的完整闭环；100 个
    fixture tasks 不消耗真实 Codex 调用。下一步接真实 Codex/native evaluator adapter，并在
    100-task、2000-call、1 instance、24h、¥30 硬上限内执行；final sealed 继续禁止打开。

## 8. 版本与实验编号

每次运行都应分别记录：

```json
{
  "system_version": "0.1.0",
  "run_schema_version": 1,
  "agent_program_schema_version": 1,
  "evaluator_epoch": "record-cleaning-v1",
  "search_protocol_hash": "...",
  "experiment_id": "exp-backend-smoke-001"
}
```

产品版本、数据协议、evaluator epoch 和实验编号不得混用。
