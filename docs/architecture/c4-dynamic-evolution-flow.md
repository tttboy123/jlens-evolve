# v3.0 Dynamic Evolution Flow

该动态图把当前可执行的 autonomous Skill round 与未来跨策略闭环放在同一视图。`[LIVE]` 步骤由当前代码支持；`[TARGET]` 步骤仅描述目标架构，当前没有自动 `CapabilityGap`、Portfolio Orchestrator 或 non-fixture AgentProgram 执行路径。

```mermaid
C4Dynamic
  title Dynamic Flow - Live Autonomous Skill Round and Target Portfolio Loop

  Person(operator, "Research Operator", "Defines authorization and bounded inputs")

  Container(cli, "[LIVE] Product CLI", "Python CLI", "Starts and resumes campaigns")
  Container(runner, "[LIVE] Autonomous Skill Runner", "Python", "Selects feedback tasks and seals round decisions")
  Container(kernel, "[LIVE] Campaign Kernel", "Python", "Admits plans and controls campaign lifecycle")
  Container(strategy, "[LIVE] Skill Paired Strategy", "Python", "Plans matched baseline and taught execution")
  Container(runtime, "[LIVE] Execution Runtime", "Python", "Runs model, external-trace and native evaluation")
  Container(evidence, "[LIVE] Evidence and Governance", "Python", "Verifies receipts, Claims and acceptance gates")
  ContainerDb(store, "[LIVE] Receipt and Artifact Store", "Append-only files", "Stores runtime facts, round results and manifests")
  ContainerDb(best, "[LIVE] BEST Projection", "Hash-verified JSON", "Reloadable projection of the accepted sealed-round parent")

  Container(gap, "[TARGET] CapabilityGap", "Not implemented", "Would scope a reusable failure-derived capability need")
  Container(portfolio, "[TARGET] Portfolio Orchestrator", "Not implemented", "Would schedule cross-strategy research")
  Container(localResearch, "[TARGET] Local Capability Research", "Planned Skill campaign", "Would validate an inactive Skill or Operator")
  Container(program, "[TARGET] Non-fixture AgentProgram Tournament", "Not implemented", "Would compose validated components and produce new evidence")

  Rel(operator, cli, "[LIVE 1] Supplies explicit authorization and run configuration", "CLI")
  Rel(cli, runner, "[LIVE 2] Starts or resumes autonomous Skill evolution", "In-process call")
  Rel(runner, store, "[LIVE 3] Verifies the latest accepted sealed round as parent authority", "Manifest and round index")
  Rel(runner, kernel, "[LIVE 4] Submits baseline and taught feedback campaigns", "Campaign protocol")
  Rel(kernel, strategy, "[LIVE 5] Requests matched, bounded execution plans", "Strategy protocol")
  Rel(strategy, runtime, "[LIVE 6] Executes baseline and candidate-consuming taught plans through Kernel", "ExecutionPlan")
  Rel(runtime, store, "[LIVE 7] Appends model, external-trace and official native receipts", "Atomic files")
  Rel(evidence, store, "[LIVE 8] Verifies paired receipts and appends Claims and campaign feedback", "Hash-verified projection")
  Rel(runner, store, "[LIVE 9] Freezes AUTONOMOUS-ROUND-RESULT, manifest and accepted decision", "Sealed artifacts")
  Rel(runner, best, "[LIVE 10] Exports and hash-reloads the accepted parent projection", "BEST-HARNESS.json")

  Rel(evidence, gap, "[TARGET 11] Derives a verified reusable capability gap", "Planned protocol")
  Rel(gap, portfolio, "[TARGET 12] Queues scoped capability research", "Planned protocol")
  Rel(portfolio, localResearch, "[TARGET 13] Starts matched local validation", "Planned campaign")
  Rel(localResearch, portfolio, "[TARGET 14] Returns a validated inactive component", "Planned registry event")
  Rel(portfolio, program, "[TARGET 15] Resumes a non-fixture tournament with that component", "Planned campaign")
  Rel(program, evidence, "[TARGET 16] Produces official evidence and new failure facts", "Planned receipts")
```

## 当前 LIVE 链路

```text
authorized feedback tasks
→ matched baseline/taught Skill execution
→ model + external-trace + official native receipts
→ verified Claims and Teacher-safe campaign feedback
→ sealed AUTONOMOUS-ROUND-RESULT + manifest + round-index decision
→ accepted sealed round becomes next parent authority
→ BEST-HARNESS.json is exported only as its hash-verified projection
```

只有 `accepted_as_best` 的 sealed round 在 manifest 和 round-index 验证后才能成为下一轮 parent。单独出现或被修改的 `BEST-HARNESS.json` 不能晋升候选、覆盖 round 决策或充当第二套 authority。新 revision 仍默认 inactive。

## 完整 TARGET 闭环

```text
verified failure evidence
→ CapabilityGap
→ Portfolio Orchestrator
→ local Skill/Operator paired validation
→ validated inactive component
→ non-fixture AgentProgram tournament
→ official evidence and new failure facts
→ CapabilityGap (next iteration)
```

这个闭环目前尚未实现。当前 AgentProgram 仅支持显式 `execution_profile=fixture`：它可验证 revision/search-parent 机制，但不声明 native gain，也不具备 promotion eligibility。Legacy 仅是只读兼容导入。已准入执行产生的失败事实会作为 hash-bound receipts 保留；在准入前被拒绝的输入 fail closed，不伪造执行 Receipt。
