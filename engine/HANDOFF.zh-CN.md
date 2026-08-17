# Agent 自进化项目接续说明
# 交接：v2.1.1 real-search-001-deepseek-v3 窗口已收尾


## 实例生命周期策略（2026-08-05 用户指示）

- **默认 stop，不 terminate**：以后窗口收尾/暂停一律用 EC2 stop（根卷、docker 镜像、
  构建缓存全部保留，start 后可续跑）；**只有用户明确说“项目不做了”才终止实例**。
- 计费备忘：运行 m7i-flex.large ≈ $0.1197/h；停止后计算费停止、根卷 250GB gp3 存储
  约 $24/月照付；公网 IP 会变，需固定则绑 EIP（停止时 $0.005/h）。
- 当前实例 i-0b49272b2d58d034b（54.151.251.47）保持运行（real-search-002 prebuild 中）。

## 给下一个会话/操作者的要点

- run 已停止（deadline TERM），证据完整在本地
  `artifacts/v2.1.0/v2.1.1-jlens-evolution/runs/real-search-001-deepseek-v3/`
  （STATE 摘要 a5c06e0a774a…，SHA256SUMS 8349 条）；
- AWS 已释放：i-08bacc8f91ce598c3 / sg-0a08296238f95812f / evolve-jlens-aws 全部删除，
  控制面归零；本地密钥在 `cloud-control/archived-keys/evolve-jlens-aws.pem`；
- G1 confirmation 有 1 条中断 reservation（c4a54ad9/3727f492），若续跑需先对账
  （abort+重派发或归档）并重设 timeout；
- 网络：SSH 走 `ProxyCommand="nc -X 5 -x 127.0.0.1:7897 %h %p"`，aws/codex/git 先
  `source cloud-control/aws-proxy.env`；
- 下一步按用户确认做 v2.2 收敛引擎（单独 PLAN + 预声明门，先读 NEXT.zh-CN.md）。


## 读取顺序

新任务开始时按顺序读取：

1. `README.md`：项目入口；
2. `ROADMAP.zh-CN.md`：产品范围和版本权威；
3. `STATUS.zh-CN.md`：当前真实状态；
4. 本文：完成状态、复核入口与后续边界；
5. `BACKEND_POC.zh-CN.md`：后台命令、数据库和 evidence；
6. `EVAL.zh-CN.md`：RSI/PSI 与评测边界。

代码、测试和 SQLite run 证据高于本文；如果发生冲突，先核对实际运行，再更新文档。

## 2026-08-04 紧急接续点（HUMAN_REQUIRED）

### 08-04 最新：G0 完成、G1 被 Codex 用量配额阻断

⚠️ 更正（08-04 复核）：此前“G0 完成 16/25 题”不成立。逐臂核查 receipt 确认
配额在腾讯云 #2 启动前已耗尽，腾讯云 #2 14 臂与 AWS 18 臂全部为
`0 token / rc=1 / usage-limit`，32 臂 evidence 与 4 张 PatternCard 无效，
真实有效证据=0。污染 run 已归档 `runs/real-search-001-resume-QUOTA-INVALID/`
（含失败 receipt 与已记账 reservation），详见 INCIDENT-019。

real-search-001-resume 在 AWS 上完成 G0 观察代（16/25 题、32 臂 evidence，全部
native_score=0.0、0/32 过安全门、4 张失败 PatternCard），证据已同步本地；
但 MutationProposer 首次真实调用即被 ChatGPT/Codex 用量配额拒绝（重置约
2026-08-08T04:10Z），0 候选、G1-G3 未开始。partial RESULT 见
`runs/real-search-001-resume/RESULT.partial-G0-complete.zh-CN.md`，事件见
`cloud-control/INCIDENT-019-CODEX-USAGE-QUOTA.md`。

恢复需要用户解决 Codex 配额（购买 credits 或切换有额度的账号）后：重跑
proposer 并从 G1 继续（controller 状态保留，G0 已完成无需重跑）；AWS 实例
`i-08bacc8f91ce598c3` 暂保留（空闲约 $0.086/h），可随时释放后按自动化 bootstrap
重建。

`real-search-001` 远端证据已不可恢复：实例 `ins-gdqrjr5a` 于 2026-08-03T22:41:41Z 被 root
账号经控制台（微信 MFA 已验）执行 `TerminateInstances` 终止（CloudAudit requestID
`d5cffce7-8e34-4025-8329-5a366792a17e`）；无快照、无 COS 备份、本机无副本。中断前 G0 完成
9/25 题（18 臂 evidence、约 23–24 次真实 Codex 调用），最后进度见
`artifacts/v2.1.0/v2.1.1-jlens-evolution/runs/real-search-001-interrupted/RESULT.partial.zh-CN.md`，
事故细节见 `artifacts/v2.1.0/v2.1.1-jlens-evolution/cloud-control/INCIDENT-016-EVIDENCE-LOSS.md`。
两条恢复路径的预声明计划（含已核验的 original/parent 哈希与逐题证据同步协议）见
`artifacts/v2.1.0/v2.1.1-jlens-evolution/RECOVERY-PLAN.zh-CN.md`。

本地侧完好：348 项 pytest 通过，SEARCH-TASK-SCHEDULE.json 与 300 题池、baseline authority、
修正案 002–015 均在本机；`/private/tmp/evolve-v211-runtime.tgz` SHA-256
`50a54fda71a5af9bc631a7570f4e087bb385e792defb7e4973fdb4989e2c7d88`。

按目标执行顺序第 3 条停在 HUMAN_REQUIRED。恢复选项需用户决策：

1. 接受 Task 1–9 证据丢失，以新 controller 从 Task 10 起续跑同一冻结 schedule
   （G0 将缺 9 题的 matched evidence，只能按 partial 处理，不可外推结论）；
2. 放弃本轮授权窗口，把 24h/¥30 等硬上限视为已消耗（已用约 7.3h、约 ¥4–6），
   重新申请新的授权窗口/新 schedule 从头开始；
3. 由用户确认是否删除残留的临时安全组 `sg-gx6m9m9x` 与密钥 `skey-a1xnwt11`。

任何选项均不重发已记账的模型调用、不打开 final sealed、不降低门。

## v2.1 当前接续点（2026-08-03）

最新权威入口是 `artifacts/v2.1.0/INDEX.zh-CN.md`。持续任务流、四评测集 300 题池、永久 shadow
baseline、每轮 matched A/B ledger、周期性 ChangeSet 晋升/回滚和 60-task one-shot sealed auditor
已完成本地实现。300 题当前全部 unopened，真实 completed round 为 0，不能宣称 Agent 已优化。

方向已修正：不能把 100 个 search tasks 消耗在一对固定 Agent 上。新增权威阶段
`artifacts/v2.1.0/v2.1.1-jlens-evolution/` 已实现真正的多代进化发动机：Observer evidence →
advantage/failure PatternCards → 4 个 inactive ChangeSets → 4→2→1 native-evaluator tournament →
experimental search parent。CandidateArchive 保存 selected/rejected/failed/inactive 全量谱系、正反操作
和 hash-chained events；每个候选同时对 original shadow 与当前 parent 报告。

本地 fixture pass³ 已完成：四代 100 tasks、344 逻辑 arm、16 逻辑 proposer、真实 Codex calls=0；
10 PatternCards、16 proposals、3 selected、9 rejected、4 terminal inactive；三个 run 指纹均为
`432d2b698e9cc0a930eb0acc4173bf47888e964a789860ab83c2a71b6af87442`。该结果只接受为软件 POC，
不是实际 Agent 优化。

完成性审计又修复了 freeze 后文件篡改、状态 JSON 篡改、崩溃重复领题、partition 越序、未绑定
10 个新 pair 的 promotion，以及 Terminal-Bench 缺 pre-verifier freeze receipt 六类缺口。旧 ready
runtime 保存在两个 `ready-pre-*-superseded` 目录；当前权威 pass³ 目录名含 `integrity-authoritative`。
首批 10 题身份在 `configs/PILOT_PLAN.json`，未 claim、未读题面、未产生外部动作。

后续本地补齐的 execution bridge 已覆盖 `claimed task → matched Codex arms → frozen prediction →
native evaluator report → paired evidence admission → retired`。新建文件不会再漏出 patch，空/失败
prediction 不会被过滤；四个 adapter 的 normalizer、receipt 与 ledger 入账有 31 个定向测试。权威
证据为 `evidence/EXECUTION-BRIDGE-VERIFICATION.json`。该证据的真实 Agent/native evaluator 调用均为
0，不改变 `agent_optimized=false`。

本地复核：

```bash
cd /Users/lune/Documents/Codex/2026-07-18/bang/work/evolve-jlens-cluster
.venv/bin/python verify_continuous_ab.py \
  --stage-dir artifacts/v2.1.0/v2.1.0-continuous-ab
.venv/bin/python -m pytest -q
```

下一步是把既有 `agent_arm_runner.py` / `benchmark_execution.py` / `native_result_adapter.py` 接入
`EvolutionAdapters`，然后做腾讯云实时询价。最新硬上限为 100 search tasks、2000 次真实 Codex
calls、1 台实例、24 小时、¥30；允许 inactive proposals 和实验 parent advance。final sealed、
production/global promotion 和全局 Skill 安装仍禁止。

## 工作目录

```text
/Users/lune/Documents/Codex/2026-07-18/bang/work/evolve-jlens-cluster
```

## 当前已完成层

`v0.1.0-poc` 后台 Evolve Kernel 已有实现和文档：

- `backend_poc.py run`；
- `backend_poc.py inspect`；
- SQLite `runs/evaluations/iterations`；
- public 搜索与 post-search holdout 隔离；
- 收敛原因和 evidence；
- 确定性 task-program mutation smoke。

这只能证明 Kernel，不证明 AgentProgram 已经自进化。

该基线已在 `artifacts/v1.0.0/v0.1.0-kernel/` 完成只读哈希冻结：旧证据没有移动或
覆盖，SQLite integrity、三个 final run、evaluation partition 顺序和唯一结果指纹均已
保存。统一索引是 `artifacts/v1.0.0/INDEX.zh-CN.md`。

## 当前完成状态

`v1.1.0` 与 `v2.0.0` 的新增证据统一位于 `artifacts/v2.0.0/`；v2.1 证据位于
`artifacts/v2.1.0/`：

- `v1.1.0-codex-target`：真实历史 parser、本地 Codex identity、三表面 ChangeSet、中文报告、
  apply/rollback patch 和 final pass³ 已 accepted；候选未应用，live 模型改进未证明。
- `v2.0.0-meta-evolution`：三代 MetaProgram、父代 hash、候选预算、profile、协议留出和安全
  回归已通过；release accepted，但 Agentic RSI 因 fresh/live 两门为 false 而 rejected。

- `live-audit-001`：用户授权后，以一条 G2 freeze 后真实任务运行 baseline/G2 各三次。六次均
  return code 0 且安全；纠错后质量 `0.875→0.958333`，token 成本增加 `2.6281%`。grader 修复
  发生在查看输出后，故只记 provisional，G2 未全局应用，Agentic RSI 仍 rejected。

- `benchmark-suite-001`：Qwen3.5-4B 与 Qwen2.5-Coder-7B 经统一 MLX adapter 跑完四任务
  baseline/G2 矩阵。adapter 的 stop-token/patch 等价缺陷已保留 pre-fix 后修复；正式矩阵表明
  4B G2 有格式收益但高成本和结果回归，7B G2 无收益。该阶段只完成 SWE-bench adapter；后续
  `swe-bench-pilot-001` 已在云端补上官方运行证据。

- `schema-adapter-001`：strict ChangeSet schema 与一次 bounded repair 已接入。保留首次 validator
  词形漂移负 run 后纠正重跑；Coder-7B 两个 profile 均 first-pass accepted，4B 两格仍 rejected，
  最终 `2/4` 未过 `3/4` 能力门。软件完成，但没有模型/profile/ChangeSet 晋升或应用。

- `task-plugin-routing-001`：确定性 task-family router、项目内路径门和逐 task prompt hash freeze 已
  完成。24-cell public matrix 中 routed 相对 monolithic 在 4B/Coder-7B 上同时提分并将 input tokens
  约减半；但 `agent_change` 两格仍 unsafe，只有 `3/4 safe`，所以候选未晋升。

- `swe-bench-pilot-001`：固定 revision 的 Verified 数据集按哈希预选 5 个任务；prediction 在
  official evaluator 前一次冻结。官方结果 5/5 resolved、0 errors，F2P 8/8、P2P 279/279；
  gold 1/1 仅作环境验证。云证据 archive 和 150/150 文件哈希已回传核验，实例、系统盘、两个
  安全组与云 SSH 公钥已删除。只接受为 capability pilot，不晋升 Agent/profile/RSI。

- `swe-bench-batch-002`：fail-closed runner 已实现并覆盖版本、任务、prediction、evidence 与资源
  生命周期；10 个新任务在题面读取前预声明，prediction 在 official evaluator 前冻结。官方结果
  `9/10 resolved`、0 errors；唯一 Matplotlib 失败由 4 个既有 PASS_TO_PASS 布局回归造成。远端
  archive 与 143/143 内部哈希已在本地验证。实例自动销毁后，系统盘、临时安全组与临时 SSH
  公钥也已删除，四类资源的控制面精确 ID 计数均为 0。

当前不保留付费云实例，等待下一个新任务窗口。下一项工作是在全新任务上预声明并运行
baseline/evolved Prompt/Skill/Policy matched pass³。既有 15 题只能做 regression。不要复用同一
任务调参、事后改 rubric、把 5/5 或 9/10 当榜单、或降低门。

`v0.2.0` 已完成并保存在 `artifacts/v1.0.0/v0.2.0-agent-program/`。它在冻结
FrozenReplayAgent 上通过三个单轴 AgentProgram mutation 获得 public `3→4→10→13`、
sealed `0→6`，但不证明 Direct LLM 或其他任务。

`v0.3.0` 已完成并保存在 `artifacts/v1.0.0/v0.3.0-jlens-observer/`。四模式 matched
matrix 与故障注入通过；JLens 增量结论是 `not_supported`，不得进入 admission。

`v0.4.0` 已完成并保存在 `artifacts/v1.0.0/v0.4.0-psi-skill-library/`。项目内 Registry、
历史 negative candidate 和 payout/refund 双任务 pass^3 evidence 均已封存；新 Skill 只在
deterministic replay 范围标记 `transfer_verified`，未激活、未安装。

`v0.5.0` 已完成并保存在 `artifacts/v1.0.0/v0.5.0-agent-code-mutation/`。AST allowlist、
越权静态拒绝、隔离重放、public/sealed admission、lineage 与 rollback drill 均有 pass^3
证据；最终 active 保持 parent，不能把它称作任意 Python 沙箱。

`v0.6.0` 已完成并保存在 `artifacts/v1.0.0/v0.6.0-evaluator-shadow/`。lax/strict drift、
robust review proposal、shadow on/off anchor 等价和 active epoch 未变化均有 pass^3 证据；
没有请求或执行人工 evaluator 切换。

`v0.7.0` 已完成并保存在 `artifacts/v1.0.0/v0.7.0-integration/`。统一 envelope、三次纵向
运行、JLens failure isolation、幂等 replay 和单一 admission authority 均已保存。

`v0.8.0` 已完成并保存在 `artifacts/v1.0.0/v0.8.0-hardening/`。SQLite schema v2、事务
events、lease、两个 crash point、真实并发、迁移、部分写、timeout 与预算门均有 pass^3 证据。

`v0.9.0` 已完成并保存在 `artifacts/v1.0.0/v0.9.0-release-candidate/`。三套 RC、9 个
operation、seed/task/partition raw audit、五份中文文档与 clean-room replay 均已封存。

`v1.0.0` 已完成并保存在 `artifacts/v1.0.0/v1.0.0-release/`。最终 Release 合同包括：

```text
StableRelease
├── evolve_service.py run / inspect / verify / rollback-plan
├── read-only artifact manifest verifier
├── v0.1-v0.9 acceptance matrix
├── final pass^3 / recovery / rollback evidence links
├── documentation consistency audit
└── FINAL-REPORT.zh-CN.md
```

CLI 只组合既有 release_candidate/durable service，`verify` 保持只读。三个 post-fix final
run 全部 accepted 且语义 fingerprint 一致；九阶段退出门已映射到原始 evidence。

v1.0 历史 Goal 已关闭；当前 v2 Goal 的本地软件工作和首轮 live execution 已完成。确认性新任务
A/B 与跨领域 sealed audit 尚未满足，不得把单任务 provisional 结果写成 Direct LLM 通用收益、
生产部署或 Agentic RSI。

## `v1.0.0` 验收门

- CLI run/inspect/verify/rollback-plan 参数稳定且无隐式外部动作；
- MANIFEST 中所有已列文件存在、bytes/SHA 正确，stage decision 全 accepted；
- v0.2 AgentProgram 改进、v0.3 JLens A/B、v0.4 PSI、v0.8 recovery、v0.9 RC 均有直接引用；
- 最终 pass^3、全量 tests/lint/format、rollback 与 clean-room 复核通过；
- ROADMAP/STATUS/HANDOFF/ADR/CHANGELOG/INDEX 与代码状态一致；
- 最终中文报告明确能做、不能做、负结果和准确恢复命令。

## 明确禁止继续的方向

- 不把公开训练集导入当前主线；
- 不实现或调用模型权重训练；
- 不把 JLens 分数用作 admission/reward；
- 不自动安装项目生成的 Skill；
- 不在同一 evaluator epoch 内改变评分标准；
- 不把 task program 的提升描述成 Agent RSI 已通过。

## 后台核对命令

```bash
.venv/bin/python evolve_service.py codex-run --output /tmp/evolve-codex-v11
.venv/bin/python evolve_service.py meta-run --output /tmp/evolve-codex-v20
.venv/bin/python evolve_service.py model-probe
.venv/bin/python evolve_service.py benchmark-run --output /tmp/evolve-model-matrix
.venv/bin/python evolve_service.py swe-probe
.venv/bin/python live_codex_ab.py \
  --config artifacts/v2.0.0/v2.0.0-meta-evolution/live-audit-001/configs/experiment.json \
  --output /tmp/evolve-codex-live-audit-replay
.venv/bin/python artifact_verifier.py --manifest artifacts/v2.0.0/MANIFEST.json
.venv/bin/python backend_poc.py inspect
.venv/bin/python agent_program_runtime.py inspect \
  --result artifacts/v1.0.0/v0.2.0-agent-program/runs/agent-program-final-pass3-3/result.json
.venv/bin/python observer_runtime.py \
  --output artifacts/v1.0.0/v0.3.0-jlens-observer/runs/manual-replay
.venv/bin/python psi_runtime.py \
  --output artifacts/v1.0.0/v0.4.0-psi-skill-library/runs/manual-replay
.venv/bin/python agent_code_runtime.py \
  --output artifacts/v1.0.0/v0.5.0-agent-code-mutation/runs/manual-replay
.venv/bin/python evaluator_shadow.py \
  --output artifacts/v1.0.0/v0.6.0-evaluator-shadow/runs/manual-replay
.venv/bin/python integration_runtime.py \
  --config artifacts/v1.0.0/v0.7.0-integration/configs/experiment.json \
  --output artifacts/v1.0.0/v0.7.0-integration/runs/manual-replay
.venv/bin/python durable_service.py \
  --config artifacts/v1.0.0/v0.8.0-hardening/configs/experiment.json \
  --database artifacts/v1.0.0/v0.8.0-hardening/runs/manual/service.sqlite3 \
  --output-root artifacts/v1.0.0/v0.8.0-hardening/runs/manual/operations \
  --worker-id manual
.venv/bin/python release_candidate.py \
  --config artifacts/v1.0.0/v0.9.0-release-candidate/configs/experiment.json \
  --output artifacts/v1.0.0/v0.9.0-release-candidate/runs/manual-rc
.venv/bin/python -m pytest -q
.venv/bin/ruff check *.py tests tasks
.venv/bin/ruff format --check *.py tests tasks
```

恢复或复核时，先运行：

```bash
.venv/bin/python evolve_service.py verify --manifest artifacts/v1.0.0/MANIFEST.json
.venv/bin/python evolve_service.py inspect \
  --result artifacts/v1.0.0/v1.0.0-release/runs/release-final-pass3-3/result.json
.venv/bin/python evolve_service.py rollback-plan \
  --kind agent-program
```

不要重复构建 v0.1-v1.0 已验证模块，也不要覆盖任何历史 rejected/pre-fix evidence。

## 2026-08-05 快照 incident（已收口：修正案 024）

- 4 个冻结文件不可恢复（用户确认无副本，授权自行修正）；修正案 024 已激活：
  阶段标记 degraded、防篡改测试 `test_frozen_v21_stage_is_internally_recoverable_and_tamper_evident`
  xfail(strict=True)、钉住哈希保留为权威；若日后获得字节级副本，恢复后移除 xfail（XPASS 会报警）。
- 见 `artifacts/v2.1.0/v2.1.0-continuous-ab/evidence/INCIDENT-OPERATOR-SNAPSHOT-OVERWRITE-2026-08-05.json`。
