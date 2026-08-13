# v3.0 Evidence-Centric Evolution Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将前三代进化能力和 AHE/JLens 观测能力整合为一个以 Evidence Graph 为事实核心、支持多种可插拔进化策略、可审计且可人工晋升的 Agent 外部能力进化平台。

**Architecture:** v3.0 使用一个中性的 Campaign Kernel 管理授权、预算、恢复和执行，以统一 Execution Runtime 运行模型、工具和原生评测，以 Observer Hub 采集外部行为、内部表征、结果、成本和安全证据。第一代退为历史导入策略，第三代负责局部 Skill/Operator 因果验证，第二代负责完整 AgentProgram 组合搜索；Capability Registry 和 Portfolio Orchestrator 连接三种策略。

**Tech Stack:** Python、内容寻址文件、append-only JSON/JSONL receipts、现有 native evaluator/harness、MLX/DeepSeek/Codex transports、Mermaid C4 文档。

---

## 1. 版本定位

v3.0 不是继续增加一套 Loop，而是把现有能力重构为一个产品平台：

```text
第一代：历史 ChangeSet、Prompt、Policy 和 replay → Legacy Import Strategy
第二代：多代 AgentProgram tournament → AgentProgram Search Strategy
第三代：Skill/Operator baseline-taught 验证 → Skill Paired Strategy
AHE/JLens：独立轨迹分析 → Observer Hub 插件
```

系统的事实权威固定为：

```text
Receipt Store        = 运行事实权威
Evidence Graph       = 证据关系权威
Claim Engine         = 判断权威
Capability Registry  = 局部能力权威
AgentProgram Registry = 完整配置权威
Governance Authority = 晋升与发布权威
Report               = 可重建视图，不是事实来源
```

## 2. v3.0 产品能力

v3.0 必须交付以下十项用户可感知能力：

1. **任务数据工厂**：导入、清洗、冻结、分区和防泄漏。
2. **无人值守 Campaign**：授权、预算、恢复、租约、暂停和终止。
3. **统一执行环境**：本地与远程模型、工具、workspace、patch 和 native evaluator。
4. **多通道 Observer**：AHE 外部行为、JLens 内部表征、结果、成本和安全观测。
5. **Evidence Graph**：对齐 task/run/arm/turn/token/layer 并追溯全部结论。
6. **机制研究台**：从 Evidence 生成 Failure Signature 和可证伪机制假设。
7. **外部能力生成器**：Teacher 生成 Prompt、Skill、Operator、Router 和 Memory Policy 候选。
8. **三种进化策略**：历史 replay、局部 paired A/B、完整 AgentProgram tournament。
9. **能力与 Agent 注册中心**：管理版本、lineage、适用条件、组合和回滚。
10. **治理与发布中心**：回归、transfer、holdout、独立审计、人工晋升和 release manifest。

## 3. 目标模块结构

```text
src/evolve/
├── data/                 # Task、Dataset、cohort、materialization、leakage
├── kernel/               # Campaign、Stage、WorkItem、预算、授权、恢复
├── runtime/              # Model、tool、workspace、patch、native evaluator
├── observers/            # AHE、JLens、LB、outcome、cost、safety
├── alignment/            # run/time/token/layer/cross-arm 对齐
├── evidence/             # Receipt Store、Evidence Graph、Claim Engine
├── analysis/             # Fusion、Failure Signature、Mechanism Hypothesis
├── proposals/            # Teacher context、ChangeSet、prediction
├── strategies/           # legacy_import、skill_paired、agent_program
├── registry/             # Capability、AgentProgram、Candidate
├── orchestration/        # Portfolio Goal、Capability Gap、campaign scheduler
├── governance/           # gate profiles、promotion、holdout、release
└── reporting/            # projectors、audit、replay、dashboards
```

## 4. 现有模块新增、改造与下线

“删除”在 v3.0 中默认表示停止承担权威职责并保留只读兼容层；完成一个版本的 replay 等价性验证后，才允许物理删除代码。

| 产品能力 | 新增模块 | 改造现有模块 | 下线的旧职责 |
|---|---|---|---|
| 任务数据工厂 | `data/task_registry.py`、`cohort_manager.py`、`leakage_guard.py` | `benchmark_catalog.py`、`eval_manifest.py`、`real_evolution_bridge.py` 的 materializer | 每代独立 task materialization、路径推断 cohort |
| Campaign 控制 | `kernel/campaign_controller.py`、`budget_manager.py`、`checkpoint_manager.py` | `evolution_controller.py`、`continuous_ab.py`、第三代 `loop.py` | 多套预算、resume、LATEST 状态和 Round 驱动控制 |
| 统一执行 | `runtime/execution_runtime.py`、`model_transport.py`、`workspace_manager.py` | `real_evolution_bridge.py`、`mlx_student.py`、`student_adapter.py`、`parent_model.py` | Strategy 直接执行模型、shell 和 evaluator |
| Observer 平台 | `observers/observer_hub.py` 和 adapters | `collect_lens.py`、现有 tool trace、native/cost/safety 记录 | AHE/JLens 两套独立身份和结果系统 |
| Evidence Graph | `evidence/receipt_store.py`、`evidence_graph.py`、`claim_engine.py` | append-only cells、`evolution_catalog.py`、`operator_evidence.py` | Catalog 同时作为事实、判断和展示权威；手填报告 |
| 机制分析 | `analysis/evidence_fusion.py`、`failure_signature.py`、`mechanism_hypothesis.py` | `pattern_miner.py`、failure taxonomy、PatternCard schema | 单次相关性直接称为 repair mechanism |
| Teacher 提案 | `proposals/candidate_proposer.py`、`teacher_context_builder.py` | `real_mutation_proposer.py`、`parent_model.py`、`experience_store.py` | Teacher 直接写正式 Skill 或自行宣布验证成功 |
| 局部能力进化 | `strategies/skill_paired/` | 第三代 `loop.py`、`experiment.py`、operator/symbol rewrite | 第三代独立顶层 Campaign 和人工 CLI 串联 |
| AgentProgram 搜索 | `strategies/agent_program/` | `evolution_runtime.py`、`evolution_controller.py`、`meta_evolution_runtime.py` | Kernel 硬编码 G0–G3，tournament 直接影响 production |
| 历史导入 | `strategies/legacy_import/` | 第一代 ChangeSet/replay/reader | 第一代继续承担新实验和独立 promotion |
| 能力资产 | `registry/capability_registry.py` | `skill_registry.py`、`operator_evidence.py` | Skill、Operator、Experience 各自维护状态权威 |
| Agent 产品资产 | `registry/agent_program_registry.py` | 第二代 AgentProgram profile/archive | 用可变目录表示 search parent/production |
| 跨策略编排 | `orchestration/portfolio_orchestrator.py`、`capability_gap.py` | 长期 Goal 状态和现有调度入口 | 三种 Strategy 永久并列、无法相互提供能力 |
| 治理发布 | `governance/governance_service.py`、gate profiles | `governance.py`、`cost_guard.py`、`convergence_gate.py`、promotion ladder | Strategy、Observer 或 Teacher 自行晋升；Skill 自动激活 |
| 审计报告 | `reporting/report_projector.py`、`audit_verifier.py` | Round report/index/manifest、Codex review state | 报告作为事实来源、每个微轮阻塞等待审计 |

## 5. 不可破坏的架构边界

1. 模型权重冻结；v3.0 不包含 SFT、LoRA、RL 或 checkpoint 训练。
2. Observer 只产生 Evidence，不产生 PromotionDecision。
3. Teacher 只产生 Candidate，不产生 ValidatedClaim。
4. Strategy 只生成 ExecutionPlan 并解释结果，不直接操作 runtime 基础设施。
5. Native evaluator 是结果裁判；内部轨迹不能替代 native outcome。
6. Evidence、Receipt 和审计记录 append-only；修正通过 superseding Claim 完成。
7. holdout/final-sealed 必须由独立授权开启，且 burned 数据不得重新成为 fresh holdout。
8. Skill 默认 inactive；生产激活必须经过统一 Governance 和人工批准。
9. Kernel 不理解 G0–G3、baseline/taught 或 PatternCard 等 Strategy 语义。
10. v3.0 首版 Evidence Graph 使用文件化 append-only facts 和可重建图投影，不引入图数据库。

## 6. 核心接口基线

平台集成应围绕以下稳定接口，不围绕旧文件布局：

```python
class EvolutionStrategy(Protocol):
    def plan(self, context: "StrategyContext") -> list["ExecutionPlan"]: ...
    def interpret(self, receipts: list["Receipt"]) -> list["ClaimProposal"]: ...
    def next_action(self, claims: list["Claim"]) -> "StrategyDecision": ...


class Observer(Protocol):
    def observe(self, rollout: "FrozenRollout") -> list["EvidenceEnvelope"]: ...


class GateProfile(Protocol):
    def evaluate(self, candidate_id: str, claims: list["Claim"]) -> "GateDecision": ...
```

统一执行计划至少包含：

```text
task revision
candidate/agent-program revision
arm
model identity
context policy
tool policy
observer policy
native evaluator identity
token/time/cost limits
holdout scope
```

## 7. 迁移阶段

### Phase 0：冻结现状和行为基线

目标：重构前证明三代关键路径可以重放。

- [ ] 记录第一代历史 replay 的输入、输出和 evaluator hashes。
- [ ] 记录第二代 original/parent/candidate tournament 的 golden campaign。
- [ ] 记录第三代 baseline/taught paired experiment 的 golden campaign。
- [ ] 冻结 AHE trace 和 JLens observation 的样本证据。
- [ ] 建立“相同事实、相同 Claim”的语义等价验证，不要求报告字节一致。
- [ ] 禁止在 Phase 0 修改任何 evaluator、holdout 或历史 sealed evidence。

验收：四条 golden path 均可独立验证，现有结论能追溯到具体 artifact hash。

### Phase 1：统一身份、Receipt 和 Execution Runtime

目标：先统一执行事实，不改变三代控制流。

- [ ] 创建 `src/evolve/evidence/receipt_schema.py`，定义 task、model、workspace、output、patch、evaluation 和 cost receipts。
- [ ] 创建 `src/evolve/runtime/execution_plan.py`，定义中性执行输入。
- [ ] 创建 `src/evolve/runtime/execution_runtime.py`，统一物化、模型、工具、patch 和 evaluator 顺序。
- [ ] 将 MLX、DeepSeek、Codex 封装为 transport adapters。
- [ ] 让三条 golden path 同时写入统一 Receipt Store。
- [ ] 验证旧结果和新 receipts 对 native outcome、runtime identity、cost 的表达一致。

验收：三个 Strategy 的执行事实使用同一 receipt schema，且无行为变化。

### Phase 2：Campaign Kernel

目标：统一授权、预算、lease、checkpoint 和终止语义。

- [ ] 创建中性的 Campaign、Stage、WorkItem、ExecutionCell 状态机。
- [ ] 从第二代 Controller 抽取授权、预算、claim、resume 和 signal handling。
- [ ] 将第三代 append-only writer lease 接入 Kernel。
- [ ] 实现 terminal、partial、cancelled 和 blocked 的互斥终态。
- [ ] 将 Round 降级为 finalized receipts 的审计批次。
- [ ] 验证进程中断后不会重复模型调用或覆盖 evidence。

验收：任何 Strategy 都能暂停、恢复和预算停止，且 finalized receipt 不被重写。

### Phase 3：Observer Hub、Alignment 和 Evidence Graph

目标：把 AHE、JLens 和 outcome 放到同一个 run/time axis。

- [ ] 定义 `Observer` 与 `EvidenceEnvelope`。
- [ ] 将外部工具轨迹包装成 `ExternalTraceObserver`。
- [ ] 将 `collect_lens.py` 包装成 `JacobianLensObserver`。
- [ ] 将 native、cost、safety 包装成 Observer adapters。
- [ ] 建立 run/turn/token/layer/phase 对齐映射。
- [ ] 建立从 receipts 和 evidence 重建的图投影。
- [ ] 实现分级 observation policy：全量外部、抽样内部、晋升前深度对照。

验收：同一 baseline/taught pair 能查询外部行为、内部表征和 native outcome 的对齐窗口。

### Phase 4：Claim、机制假设与 Teacher 提案

目标：建立 Evidence → Hypothesis → Candidate 的受控链路。

- [ ] 创建版本化 Claim Engine，区分 E0、E1、E2、E3。
- [ ] 扩展 PatternCard 为 precondition、failure signature、intervention、expected effects、invariants 和 falsification。
- [ ] 让 failure taxonomy 只作为分类标签，不自动成为机制。
- [ ] 统一 DeepSeek/Codex Teacher 请求、响应和调用账本。
- [ ] 禁止 Teacher request 包含 holdout outcome 或 reference patch。
- [ ] 让每个 Candidate 引用来源 hypothesis 和预期内部/外部变化。

验收：任何 `validated mechanism` Claim 都能追溯到 counterfactual、native 和跨任务证据。

### Phase 5：三种 Strategy 插件化

目标：保留实验语义，移除重复基础设施。

- [ ] 将第一代包装为 `LegacyImportStrategy`，只做 provenance、replay 和 seed publish。
- [ ] 将第三代包装为 `SkillPairedStrategy`，保留严格 baseline/taught、operator、ablation 和 transfer。
- [ ] 将第二代包装为 `AgentProgramSearchStrategy`，保留 candidate DAG、tournament 和 search-parent advance。
- [ ] 让三种 Strategy 只提交 ExecutionPlan，不直接调用模型和 evaluator。
- [ ] 为每种 Strategy 定义独立 convergence 和 gate profile。
- [ ] 对三条 golden path 运行新旧实现语义对照。

验收：三种 Strategy 可独立运行，共用 Kernel、Runtime、Evidence 和 Governance。

### Phase 6：Capability Registry 与跨策略编排

目标：让第三代生产能力积木，第二代组合能力积木。

- [ ] 创建 Candidate、CapabilityRevision 和 AgentProgramRevision 的不可变 registry。
- [ ] 将 Skill、Operator 和 mechanism 状态迁入 Capability Registry。
- [ ] 将第二代 AgentProgram profile/archive 迁入 AgentProgram Registry。
- [ ] 创建 `CapabilityGap`，允许 AgentProgram Strategy 请求局部能力研究。
- [ ] 创建 Portfolio Orchestrator，调度 capability campaign 和 program campaign。
- [ ] 禁止未达到 `native_validated` 的能力进入正式 AgentProgram 候选池。

验收：AgentProgram 失败能创建 Capability Gap；验证完成的 Skill/Operator 能返回组合搜索。

### Phase 7：统一治理、报告和切换

目标：建立单一晋升权威并安全下线旧入口。

- [ ] 为 legacy seed、capability、agent program 和 release 定义四个 Gate Profile。
- [ ] 实现 regression、transfer、holdout、sealed audit 和 human approval。
- [ ] 从 receipts/claims 自动生成 Catalog、Round、Campaign、Goal 和 Release reports。
- [ ] 将旧 Catalog、Round CLI 和各代 promotion 入口改成只读 facade。
- [ ] 完成一个完整 feedback campaign 和一个未开启 holdout 的 dry run。
- [ ] 经人工授权后，使用 fresh holdout 完成 v3.0 release audit。

验收：只有 Governance Authority 能推进 production pointer 或激活 Skill，所有报告均可从事实重建。

## 8. 删除与兼容策略

采用三步下线，禁止一次性重写：

```text
Release N：旧入口继续工作，同时写新 receipts，标记 deprecated
Release N+1：旧入口只读，只允许 replay 和 export
Release N+2：删除执行代码，保留 schema reader 和历史 artifacts
```

首批应下线的权威职责：

1. 第一代独立执行与 promotion。
2. 第二代 Controller 作为全平台 Kernel。
3. 第三代手工 CLI 阶段编排。
4. AHE/JLens 独立运行身份和结果目录。
5. Strategy 自带的模型、预算、workspace 和 evaluator 实现。
6. 手写 Round Report 和多套 gain 计数。
7. Teacher、Observer 或 Strategy 自行激活 Skill。

## 9. v3.0 发布门

v3.0 只有同时满足以下条件才可标记完成：

- [ ] 第一、二、三代 golden campaigns 均可在新平台重放。
- [ ] 统一 Runtime 记录模型、Prompt、workspace、patch、evaluator 和 cost identity。
- [ ] AHE 和 JLens evidence 能在同一 baseline/taught pair 上对齐。
- [ ] Evidence Graph 能从 append-only receipts 完整重建。
- [ ] Report 中所有计数由 Claim Engine 派生，不允许手填。
- [ ] Skill Paired Strategy 保持严格配对和 holdout 隔离。
- [ ] AgentProgram Search 保持 tournament 和 lineage 语义。
- [ ] 至少一个 Capability Gap 完成跨 Strategy 往返。
- [ ] 至少一个 E3 mechanism 被 Capability Registry 接收。
- [ ] feedback 回归无 evaluator infrastructure failure。
- [ ] 未经授权不能打开 holdout、激活 Skill 或移动 production pointer。
- [ ] Codex 可在只读条件下验证全部 release evidence。

## 10. 明确不进入 v3.0 的范围

- 模型微调、SFT、LoRA、RL 或权重更新；
- 自动激活 Skill；
- 自动开放 fresh holdout；
- 分布式集群调度；
- 强依赖独立图数据库；
- 用 JLens 内部信号替代 native evaluator；
- 一次性物理删除三代旧代码和历史 artifacts；
- 对所有廉价 trial 强制执行完整内部表征采集。

## 11. 架构文档

- [C4 System Context](../../architecture/c4-context.md)
- [C4 Container Architecture](../../architecture/c4-containers.md)
- [Dynamic Evolution Flow](../../architecture/c4-dynamic-evolution-flow.md)

## 12. 实施决策

v3.0 的关键决策可以浓缩为：

> 第一代提供历史先验，第三代生产经过 counterfactual 和 native 验证的能力积木，第二代搜索这些积木组成的完整 AgentProgram。Campaign Kernel 保证执行可靠，Evidence Graph 保证事实可追溯，Claim Engine 保证判断可重算，Governance Authority 保证晋升不被任何模型或 Strategy 越权。
