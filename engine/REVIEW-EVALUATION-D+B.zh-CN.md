# real-search-002 D+B 评估（基于冻结证据，2026-08-05 18:47Z）

决策：**D + B** —— 让当前 run 继续到自然 timeout（08-06 09:03Z），现在评估已冻结证据，
并把“修正案 027 未部署”等作为限定条件记录；timeout 后由操作者补写 `RESULT.partial`。

## 1. 当前证据能证明什么（工程可用性）

- **真实执行链可用**：G0 32/32 臂真实 DeepSeek Codex 调用 + 官方 native evaluator，
  逐臂证据（prediction.patch / agent-receipt / native-admission / cost / tool-events）全部冻结；
  patch 全部非空（沙箱修复生效）；全部 safety=True。
- **官方结果与负证据**：16 任务中 11 个双臂 resolved；5 个未解析（1ba29c60、2f96e04a、
  3359744e、3fe9ae4a、a373685c），负证据保留在 evidence（未丢弃、计入淘汰）。
- **观察→模式→提案链路**：G0 挖出 6 张 PatternCard（3 advantage + 3 failure，django 为主）；
  成功提出 3 个 inactive 候选（prompt/policy/harness）；skills 候选因 response schema 违规
  被 fail-closed 拒绝并记入 PROPOSAL-FAILURES.jsonl（优雅降级 4→3 生效）。
- **锦标赛推进**：G1 scout 23/30 臂，3 候选在 5 个新任务上验证中（含 resolved/负例）。
- **记账正确**：real_codex_calls=63（55 臂 + 7 proposer aux + 1 在途），reserve-before-dispatch，
  未超 2000；任务 retired=16、claimed=5、unopened=70（91 题 resume 计划）。

结论措辞：当前证据**证明“真实搜索工程可用”**（观察→模式→提案→锦标赛→记账全链路）；
**不证明 Agent 已优化**（agent_optimized=false）、不证明收敛、Skill 迁移路径尚未验证。

## 2. 必须作为评估限定条件的已知缺口（评审 A1/C1/D3/C3/E6）

1. **C1 收敛门不可达风险**：vs-original + vs-parent 双约束，original/parent 差 D 时
   mean|Δ| 下界 ≈ D/2 > ε=0.05 —— 当前 run **极不可能触发 converged**；
   评估不应以“是否收敛”作为通过判据，且“converged 未发生”不是失败证据。
2. **D3 无独立安全套件**：safety 仅来自 native admission（native.safe &&
   regression_failures==0）；**没有危险命令/5xx/私密外泄/越权的独立安全评测**；
   因此 transfer_verified 只保证性能不回归，不保证安全边界。
3. **C3/D4 人工晋升无 SLA**：transfer_verified 候选 append-only、无 review 时限，
   会堆积；后续评估可能基于 stale 候选。
4. **E6 修正案 027 未部署**：当前 run 代码 = v2.2（含 convergence_metrics）+ **未含 027**
   （无 SIGTERM partial / 无 auto-skill / worker_count=1）；timeout 会直接杀进程，
   不写 RESULT.partial —— 逐臂证据与 controller STATE 完整在盘，timeout 后操作者补写汇总；
   “027 vs 实际跑版”的差距应作为工程化短板单独评估。

## 3. 给评估者的双视角

- **视角 B**：基于当前冻结证据评估 observation+PatternCard+proposer+锦标赛工程可用性
  （上节结论），不等待完整 run；
- **视角 D**：把“v2.2+未部署027”与 C1/D3/C3 记为限定条件，评估新版（修正案 027/收敛门
  修复/安全套件）vs 实际跑版的差距——这部分是评估的**主要贡献点**。
