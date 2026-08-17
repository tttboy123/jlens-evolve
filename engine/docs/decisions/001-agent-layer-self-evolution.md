# ADR-001：冻结模型权重，优先演化 Agent 应用层

## 状态

Accepted

## 日期

2026-08-03

## 背景

项目原目标是使用 JLens 分析 Agent 搜索轨迹，再由 evolve 优化 Agent。随着 RSI、PSI、
监督模块和 Kimi 训练方法进入讨论，路线逐渐扩展到训练数据、SFT/RL、Trainer backend
和模型 registry。这使演化对象从 Agent 应用层漂移到模型权重，并让首个 POC 复杂化。

现有参考系统给出了更合适的分层：Agent Lightning 的 Sidecar 解耦、DSPy 的
Program/Metric/Optimizer、AlphaEvolve 的 evaluator/programs database、DGM 的 Agent
变体 archive、Voyager 的 Skill Library，以及 RQGM 的固定 evaluator epoch。

## 决策

在 `v1.0.0` 之前，产品主线冻结模型权重，只演化 `AgentProgram`：Prompt、
demonstrations、Skills、tool/context/retry/routing policy，以及后期经过沙箱验证的 Agent
harness code。

JLens 是独立 Observer，只产生诊断证据和实验假设；固定 evaluator 是候选晋升的唯一
权威。模型训练从产品路线移出，标记为 `deferred_research`。

## 备选方案

### 立即建设模型训练闭环

拒绝。当前只有少量任务和未证明可迁移的轨迹，训练会放大过拟合和数据泄漏风险，也
不能回答 JLens 是否改善 Agent 应用策略。

### 继续只演化任务程序

部分保留。确定性任务程序 mutation 适合作为 Kernel smoke，但不能代表 Agent 自进化。

### 同时建设 Agent、Trainer 和 evaluator 共演化平台

拒绝。三个轴同时变化会失去因果归因，也无法维持固定 evaluator epoch。

## 后果

- `v0.1.0` 后台 POC 保留，定位为 Evolve Kernel；
- `v0.2.0` 必须引入显式 `AgentProgram`；
- JLens 的增量价值必须通过 trace-only、logit-lens、JLens matched A/B 证明；
- Skill 只有经过跨任务 PSI A/B 才能晋升；
- DGM 式代码自改和 RQGM 式 evaluator shadow 分别后置到 `v0.5.0` 与 `v0.6.0`；
- 模型训练不作为当前里程碑或完成条件。
