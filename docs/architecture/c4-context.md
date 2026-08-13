# v3.0 System Context

该图描述 v3.0 Evidence-Centric Evolution Platform 的系统边界及外部参与者。平台不训练模型权重；模型、Benchmark 和原生 evaluator 都作为外部依赖接入。

```mermaid
C4Context
  title System Context - v3.0 Evidence-Centric Evolution Platform

  Person(operator, "Research Operator", "Defines goals, budgets and risky-step authorizations")
  Person(reviewer, "Independent Reviewer", "Audits evidence and approves promotion")

  System(platform, "Evidence-Centric Evolution Platform", "Evolves external Agent capabilities with immutable evidence and native evaluation")

  System_Ext(benchmarks, "Task and Benchmark Sources", "SWE-bench, Multi-SWE-bench and curated tasks")
  System_Ext(models, "Frozen Models", "Qwen MLX, DeepSeek, Codex and future model transports")
  System_Ext(evaluators, "Native Evaluators", "Official project tests and benchmark harnesses")
  System_Ext(probes, "Interpretability Probes", "Jacobian Lens, LB, logit and activation observers")

  Rel(operator, platform, "Creates goals, campaigns and authorizations", "CLI/API")
  Rel(reviewer, platform, "Reads receipts, verifies claims and records decisions", "Read-only review API")
  Rel(platform, benchmarks, "Imports and freezes task revisions", "Adapter API")
  Rel(platform, models, "Requests frozen-model inference", "Local/remote transport")
  Rel(platform, evaluators, "Executes native validation", "Isolated harness")
  Rel(platform, probes, "Requests configured observations", "Observer adapter")

  UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

## 边界说明

- Frozen Models 只执行推理，v3.0 不更新模型权重。
- Interpretability Probes 只产生 Evidence，不拥有晋升权限。
- Native Evaluators 是结果裁判，不能被 Strategy 或 Teacher 修改以适配候选。
- Research Operator 授权成本和风险边界；Independent Reviewer 独立验证晋升证据。
