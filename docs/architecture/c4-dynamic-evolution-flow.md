# v3.0 Dynamic Evolution Flow

该流程展示一次完整的“Evidence → 局部能力 → AgentProgram → 晋升”循环。第三代局部能力验证嵌套在第二代完整 AgentProgram 搜索中，第一代历史资产作为可重放 seed 输入。

```mermaid
C4Dynamic
  title Dynamic Flow - Evidence to Validated Agent Capability

  Person(operator, "Research Operator", "Defines goals and authorization")

  Container(orchestrator, "Portfolio Orchestrator", "Python", "Coordinates strategies")
  Container(kernel, "Campaign Kernel", "Python", "Controls lifecycle and budget")
  Container(strategy, "Strategy Host", "Python plugins", "Runs legacy, Skill and AgentProgram protocols")
  Container(runtime, "Execution Runtime", "Python workers", "Executes frozen rollouts")
  Container(observer, "Observer Hub", "Python adapters", "Collects multi-source evidence")
  Container(analysis, "Evidence Analysis", "Python", "Aligns evidence and derives hypotheses")
  Container(registry, "Capability Registry", "Versioned projection", "Stores validated components")
  Container(governance, "Governance Service", "Python", "Applies promotion gates")
  ContainerDb(store, "Receipt Store", "Append-only JSONL", "Stores immutable facts and claims")

  Rel(operator, orchestrator, "1. Defines goal, budget and risk authorization", "CLI/API")
  Rel(orchestrator, kernel, "2. Creates baseline AgentProgram campaign", "Campaign API")
  Rel(kernel, strategy, "3. Requests an AgentProgram execution plan", "Strategy protocol")
  Rel(strategy, runtime, "4. Submits frozen baseline plan through Kernel", "ExecutionPlan")
  Rel(runtime, observer, "5. Publishes rollout events for configured observation", "Observer protocol")
  Rel(runtime, store, "6. Appends model, patch and native evaluation receipts", "Atomic append")
  Rel(observer, store, "7. Appends external and internal evidence", "EvidenceEnvelope")
  Rel(analysis, store, "8. Reads aligned evidence and appends failure hypothesis", "Claim API")
  Rel(orchestrator, strategy, "9. Starts Skill paired campaign for capability gap", "Campaign request")
  Rel(strategy, runtime, "10. Runs matched baseline, taught and ablation plans", "ExecutionPlan")
  Rel(analysis, store, "11. Derives counterfactual and cross-task mechanism claims", "Claim API")
  Rel(strategy, registry, "12. Publishes native-validated Skill or Operator", "Registry API")
  Rel(orchestrator, strategy, "13. Resumes AgentProgram tournament with validated components", "Campaign request")
  Rel(governance, store, "14. Verifies regression, transfer, holdout and audit evidence", "Gate query")
  Rel(governance, registry, "15. Records human-approved promotion or retirement", "State event")
```

## 关键流转

```text
Legacy evidence
→ verified seed
→ AgentProgram baseline
→ multi-source Evidence
→ Capability Gap
→ Skill/Operator paired validation
→ Capability Registry
→ AgentProgram tournament
→ regression/holdout/audit
→ human promotion
```

任何步骤失败都必须生成可重放 Receipt；失败、回归、中性和基础设施错误均保留，不得只保存成功路径。
