# 项目状态
## 2026-08-05 实例生命周期策略变更（用户指示）

- 以后实例清理默认 **stop（保留数据与 docker 缓存）**，不再 terminate；
  只有用户明确说“项目不做了”才终止。详见 RESOURCE-LEDGER.instance_lifecycle_policy。

## 2026-08-05 健壮性批次 1/2/3 落地（修正案 027）

- #1 冻结保护：v2.1.0-continuous-ab 加 FROZEN.json；v21_artifacts 拒绝重快照冻结阶段，
  --freeze-override <reason> 写审计记录；CLI 自动跳过冻结阶段；
- #2 真实 run 接线：real_evolution_run.py 支持 --worker-count / --skill-registry / --auto-gate，
  搜索结束自动编译候选 Skill 并用 confirmation 配对证据跑迁移门；
- #3 SIGTERM 优雅收尾：TERM/INT 原子写 RESULT.partial（terminal_state=partial），退出 128+signum；
- 全量 pytest 375 passed + 1 xfail（修正案 024）；ruff 全过；v2.1/v2.3 MANIFEST 有效。

## 2026-08-05 v2.3 搜索→候选 Skill 落地（修正案 026）

- 用户指示完成下一版本迭代（另一 session 跑 real-search-002）；
- search_skill_bridge.py：v2.2 搜索结果编译为项目内候选 Skill（inactive）+ 跨任务迁移门
  （>=8 对、native 非劣、无 safety 回归、cost<=10%、合同/epoch 一致 → transfer_verified/rejected）；
- evolve_service.py skill-candidates 命令；边界：不 active、不 auto_install、不全局、人工晋升；
- v1.0.0 release 维护：RC 指纹重钉（04843f90…）、bundle 成员 108→110、v1.0.0 MANIFEST 重建验证；
- 全量 pytest 367 passed + 1 xfail（修正案 024）；ruff 全过；
- 产物：artifacts/v2.3.0/v2.3.0-jlens-search-skill/（bridge-pass1：3 候选、1 transfer_verified）。

## 2026-08-05 v2.2 收敛引擎已实现（修正案 025）

- 用户确认门后实现：convergence_metrics、A 修复 B → generalized_fix、
  K=2 |Δ|<0.05 且无 safety 回归 → converged 停止、候选验证并行化（worker_count 1-4）；
- 全量 pytest 363 passed + 1 xfail（修正案 024）；fixture pass³ 指纹
  `2a7bfd14bd1a5f41ebeda70999277b69dc29776b1c7d8e4ca76fe547925f4734`；
- 产物：`artifacts/v2.2.0/v2.2.0-jlens-convergence/`（PLAN/PREDECLARED_GATES/DECISION/
  runs/local-pass3-{1,2,3}/MANIFEST）；修正案 025 已生效；真实搜索未启动（待预算/实例）。

## 2026-08-05 补充：快照 incident（已按用户授权修正案 024 收口）

- 收尾时 `v21_artifacts.py --snapshot-sources` 误覆盖 v2.1.0-continuous-ab 冻结快照 5 个钉住文件；
  1 个已从会话记录恢复（哈希匹配），4 个（agent_arm_runner / benchmark_execution /
  native_result_adapter / tests/test_native_result_adapter）本地不可恢复，钉住哈希保留在
  evidence/EXECUTION-BRIDGE-VERIFICATION.json；防篡改测试按设计失败并已记录。
- 详见 `artifacts/v2.1.0/v2.1.0-continuous-ab/evidence/INCIDENT-OPERATOR-SNAPSHOT-OVERWRITE-2026-08-05.json`
  与 `cloud-control/EXECUTION-PROTOCOL-AMENDMENT-024-DRAFT.json`；已按用户 2026-08-05 指示（无副本，自行修正）激活修正案 024：阶段标记 degraded、防篡改测试 xfail(strict=True)、钉住哈希保持权威；文件恢复后 XPASS 会提示移除标记。

## 2026-08-04 v3 窗口收尾（HANDOFF 就绪）

- real-search-001-deepseek-v3 于 2026-08-04T17:30:24Z 被 run 自带 timeout TERM 停止；
- 完成：G0 observe（32/32）、4 候选（0 失败）、G1 scout（30/30）、G1 semifinal（32/32，
  晋级 skills 3727f492…）；G1 confirmation claimed 且 2/36 臂完成，候选臂在途中断；
- 记账 real_codex_calls=102、任务打开 41、实例 1 台；final sealed unopened、无 promotion、
  parent 未前进、agent_optimized=false；
- AWS 已释放并控制面归零（实例/SG/密钥），本地密钥归档；
- 下一步：v2.2 收敛引擎（单独 PLAN + 预声明门，见 NEXT.zh-CN.md）。

更新时间：2026-08-04


更新时间：2026-08-03

## 2026-08-04 事件：real-search-001 云证据不可恢复（HUMAN_REQUIRED）

实例 `ins-gdqrjr5a` 已于 2026-08-03T22:41:41Z 被 root 账号经腾讯云控制台（微信 MFA 已验）
`TerminateInstances` 终止（CloudAudit requestID `d5cffce7-8e34-4025-8329-5a366792a17e`）。
远端 `/opt/evolve-v211/real-search-001` 证据随实例销毁，无快照/无 COS 备份/无本地副本，
**不可恢复**。中断前 G0 完成 9/25 题（18 臂 evidence、约 23–24 次真实 Codex 调用），
final sealed 未打开、无 promotion、不降低任何门；已记账调用不可重发。
本地软件完好（348 pytest 通过），schedule/题池/baseline/修正案 002–015 保留。
详见 [INCIDENT-016](artifacts/v2.1.0/v2.1.1-jlens-evolution/cloud-control/INCIDENT-016-EVIDENCE-LOSS.md)
与 [partial RESULT](artifacts/v2.1.0/v2.1.1-jlens-evolution/runs/real-search-001-interrupted/RESULT.partial.zh-CN.md)。
当前停在 HUMAN_REQUIRED，恢复路径需用户决策。

## 当前阶段

```text
product_scope = agent-layer-self-evolution
current_version = v2.1.1-jlens-multigeneration-evolution
current_focus = local-engine-accepted-real-codex-native-adapter-next
long_term_goal = observer-mined-multigeneration-agent-application-layer-evolution
artifact_root = artifacts/v2.1.0
model_weights = frozen
training_track = deferred
```

当前首选入口是 `evolve_service.py`。v1 命令仍为 `run / inspect / verify /
rollback-plan`；v2 已有 `codex-run / meta-run / model-probe / benchmark-run / swe-probe`；v2.1.1
新增只读 `evolution-plan / evolution-inspect / evolution-verify`。
`backend_poc.py` 仍只代表 v0.1 Kernel smoke。

v2.1.1 已补上此前缺失的实际 mutation/population/selection 发动机：JLens/trajectory evidence 经
Pattern/Advantage Miner 形成同时包含优势、失败、条件、反例和置信度的 PatternCard；provider-neutral
proposer 只产生 inactive Prompt/Skill/Policy/router/memory/受限 code ChangeSet；CandidateArchive
保留全部谱系和负候选；native-evaluator-only tournament 做 4→2→1 淘汰，并同时报告 candidate-vs-
original 和 candidate-vs-parent。实验 parent 可前进，但 production/global ref 永远不写。

本地 4 代 × 25 fixture task 已 pass³：每 run 100 tasks、344 个逻辑 arm、16 个逻辑 proposer、
10 PatternCards、16 proposals、3 selected、9 rejected、4 terminal inactive、3 次 experimental parent
advance；三个 run 语义指纹均为
`432d2b698e9cc0a930eb0acc4173bf47888e964a789860ab83c2a71b6af87442`。fixture 的真实 Codex calls=0，
实例=0、费用=¥0，因此 `agent_optimized=false`、`agentic_rsi=false`。

v2.1 本地协议和软件已完成：四个 pinned benchmark adapter 从 1189 个源任务中冻结 300 个新任务，
每个 adapter 75 个；search/promotion/final_sealed 为 `160/80/60`。16 个历史已打开身份已永久排除，
14 个跨数据集重叠已记录。永久 shadow baseline、逐轮 prediction freeze、matched ledger、周期性
ChangeSet proposal、paired promotion/rollback 和 one-shot final sealed auditor 均已实现。

当前真实完成 multi-generation search task 为 `0`，60 个 final sealed 任务全部 unopened；v2.1.1
最终全仓测试数见 `artifacts/v2.1.0/v2.1.1-jlens-evolution/TEST-VERIFICATION.json`。完成性审计已补上 prediction/ledger/registry/SERVICE tamper、崩溃
重复领题、partition 越序、10-pair promotion 绑定和 Harbor pre-verifier ATIF receipt。因此
`software_protocol=accepted`，但 `agent_optimized=false`。首批 10 个 search 身份和 matched contract
已零副作用预声明。用户最新授权为 100 个 search tasks、最多 2000 次真实 Codex 调用、最多 1 台
临时云实例、24 小时、云费用硬上限 ¥30；允许 inactive proposal 和实验 parent advance，禁止 final
sealed 与 production/global promotion。当前先接真实 adapter 和实时价格门，未创建付费资源。

本轮已补齐正式执行桥：claim 后才物化题面且不泄漏 gold；两臂 Codex 使用相同合同和隔离
workspace；patch 会包含未 tracked 新文件并保留空/失败 prediction；Verified、Multilingual、
Multi-SWE 与 Terminal/Harbor 的原生结果统一转为带 SHA evidence 的 `ArmResult`，两臂 evidence 成对
冻结后任务才可 retired。31 个定向测试与 29 项阶段复核通过，但实际 Agent/native evaluator 调用
仍为 0，所以不改变预算门和质量结论。

v1.1 已基于 14 条真实 Codex 用户历史生成首个 Prompt/Skill/Policy ChangeSet。public 合同分
`0.1875→1.0`，protocol-order sealed `0.1667→1.0`；正反 patch 往返树哈希一致，候选未应用。

v2.0 已完成三代 MetaProgram：G0/G1/G2 的 public mean 为 `0.1667/1.0/1.0`，候选预算
为 `3/2/1`，因此单位成本分数为 `0.0556/0.5/1.0`。软件 release accepted；由于没有
developer-blind 新任务和 live Codex 执行，Agentic RSI decision 保持 rejected。

freeze 后的 `live-audit-001` 已消费一条真实 fresh task，在相同 `gpt-5.6-sol`、low reasoning、
evidence 和只读 sandbox 下完成 baseline/G2 各三次。六次全部执行成功且安全。原正则 grader 因
中文同义表达和跨行匹配缺陷得到 `0.666667=0.666667/not_promoted`；保留原结果后修复实现并只重评
原回答，baseline 为 `0.875`、G2 为 `0.958333`，delta `+0.083333`，但 G2 token 增加
`2.6281%`。纠错发生在查看输出后，因此最终 promotion 仍 deferred，Agentic RSI 仍 rejected。

`benchmark-suite-001` 已使用本地 Qwen3.5-4B 与 Qwen2.5-Coder-7B 跑完 16-cell post-fix
diagnostic。4B baseline/G2 为 `0.239583/0.510417`，但 G2 token 增加 73.01%；7B Coder 为
`0.520833/0.479167`，G2 回归且 token 增加 77.62%。两个模型都通过简单 SWE prediction
schema，四个 ChangeSet JSON cell 全失败。没有模型或 profile 晋升。

`schema-adapter-001` 已实现 strict ChangeSet schema、逐次 raw evidence 与最多一次 bounded repair。
首次矩阵暴露 `repetition` 词形与 validator/grader 的确定性合同漂移；保留负 run 后 TDD 纠正并以
相同冻结变量重跑。纠正后 raw 为 `0/4 valid、0/4 safe`，schema treatment 为 `2/4 valid、4/4
safe`：Coder-7B baseline/G2 均 first-pass accepted，4B 两格仍 rejected。软件门通过，但未达到
预声明 `3/4` 能力门，adapter、模型和 G2 均不晋升，候选未应用。

`task-plugin-routing-001` 已将 baseline、monolithic G2、routed G2 做 2 模型 × 3 arms × 4 public
tasks 的 24-cell diagnostic。routed 相对 monolithic：4B mean score `0.510417→0.625`、input
tokens 减少 `50.3147%`；Coder-7B `0.479167→0.572917`、input tokens 减少 `49.9803%`。
质量/成本子门通过，但两模型 routed 都只有 `3/4 safe`，`agent_change` 仍因因果越界失败，故 public
总门和 promotion rejected。router 软件完成，未修改 active profile。

`swe-bench-pilot-001` 已在一次性腾讯云 x86_64 实例完成官方 harness 运行。任务 ID 在查看题面前
按固定哈希协议选出，5 个 prediction 在 official evaluator 前一次冻结；结果为 5 submitted、
5 completed、5 resolved、0 errors，F2P `8/8`、P2P `279/279`。gold 环境验证独立为 1/1，
未计入 Agent 分数。云端压缩包和 150/150 内部文件哈希已在本地校验；实例、250 GiB 系统盘、
两个临时安全组和云 SSH 公钥均已删除。该结论只接受为 5-task capability pilot；因为 task patch
可见、没有 matched A/B 和 pass³，模型/profile/Agentic RSI 均未晋升。

`swe-bench-batch-002` 已把上述人工编排收敛为 fail-closed runner，并在另一台一次性腾讯云
x86_64 实例运行 10 个新预声明任务。selection 与 10 条 prediction 均在 official evaluator
前冻结；官方结果为 10 submitted、10 completed、9 resolved、1 unresolved、0 errors。唯一失败
`matplotlib__matplotlib-25960` 的新增测试通过，但 4 个既有 PASS_TO_PASS 布局测试回归，证明只跑
目标测试不足以保护高耦合模块。云证据外层 SHA 与 143/143 内部文件已在本地核验；实例、系统盘、
临时安全组和临时 SSH 公钥也已在控制面终检为 0。该结果仍仅是 capability/regression evidence，不是榜单、
matched pass³ 或 Agentic RSI。

v0.1.0 已完成只读冻结：22 个历史文件哈希一致，SQLite integrity `ok`，三个 final
pass^3 run 只有一个 outcome fingerprint。权威产物索引见
`artifacts/v1.0.0/INDEX.zh-CN.md`。

v0.2.0 已通过 Replay 范围验收：三个 AgentProgram 单轴候选依次使 public
`3→4→10→13/13`，搜索后 sealed `0→6/6`，三个独立 run 一个结果指纹；这只证明固定
Replay Agent 上的应用层机制，不外推为 Direct LLM、JLens 或跨任务收益。

v0.3.0 已通过 Observer 机制验收：四模式各三次重放保持同一 runtime fingerprint、
active program、public `13/13` 与 sealed `6/6`，collector 故障隔离成立。JLens 相对
logit-lens 的增量为 `-0.0017566`，未过 `+0.01` 门，结论为 `not_supported`。当前进入
v0.4.0 PSI Skill Library。

v0.4.0 已完成项目内 Skill Registry 和两个相邻任务的 matched A/B。新 candidate 在
payout/refund 两 target、三 seed 上从 control 的 sealed `1/6` 提升到 `6/6`，来源 replay
仍为 `6/6`；三次独立运行一个 fingerprint。该结论限于 deterministic replay，Skill
保持 inactive、未全局安装。

v0.5.0 已完成受限 route harness code mutation。三个 import/open/while 越权候选从未执行，
行为回归停在 public；安全候选 public `3/6→6/6`、sealed `2/4→4/4`，三次运行一致。
rollback drill 后 active 恢复 parent。它不是通用 Python 沙箱；macOS address-space rlimit
未生效的证据已保留。

v0.6.0 已完成 Evaluator Shadow。lax 产生 2 个 false accept，strict 产生 2 个 false
reject；robust-v2 在冻结 corpus 上 decision/champion 等价，只生成不可激活 review proposal。
shadow on/off anchor admission 完全相同，active epoch 未变。

v0.7.0 已完成统一 PluginEnvelope 和纵向集成 CLI。三次独立运行的语义 fingerprint 一致，
六组件 decision 与 11 个合同门全部通过；重复 operation 不追加日志或改变结果，只有 admission
envelope 有 active ref。

v0.8.0 已完成 SQLite v2 durable operation wrapper。prepared/running 两个 crash point 均可恢复，
实际并发 worker 不双执行，截断结果保留后由 attempt 2 重建，v1 migration 保留旧 rows；
timeout/超资源均 failed 且不发布结果。三次矩阵 fingerprint 一致，当前进入 v0.9.0 RC。

v0.9.0 已完成三套 RC、合计 9 个 durable operation。每个 operation 都从 raw events 覆盖
11/22/33 与 payout/refund，public checkpoint 位于 sealed 前；clean-room 16 tests 和 CLI
replay 通过。pre-repair 的绝对路径哈希与缺 checkpoint 缺陷已修复并保留负证据。进入 v1.0。

v1.0.0 已完成本地 Release 验收：稳定 CLI、只读 artifact verifier、九阶段 acceptance
matrix、恢复与 rollback 证据映射、中文最终报告均已封装。修复了首次正式 pass^3 暴露的
pytest duration 非语义漂移，随后三个独立 Release run 得到同一语义 fingerprint；本 Goal
已完成，但不等于生产部署或开放式 LLM 泛化已经证明。

## 已有证据

- 后台最小任务：public `3/13 → 6/13`。
- sealed holdout：`0/6 → 0/6`。
- 停止原因：`operator_space_exhausted`。
- `task_solved=false`，`production_ready=false`。
- 历史 JLens 观察：20 条边折叠为 7 个唯一迁移，重复率 65%；结论仍为
  `observational_not_causal`。
- 既有 JLens treatment 未稳定优于 control，JLens 目前仍是诊断插件。
- 跨任务 PSI 已观测到负迁移，现有 lesson 不得晋升为 active skill。

## 尚未证明

- 开放式 Direct LLM 上 Prompt、Skill 或 Agent Policy 的稳定自动演化收益；
- 开放式 LLM Agent 上的跨任务 Skill 稳定迁移；
- 开放式、多 capability 的 DGM 式 Agent harness code 自改；
- evaluator 共演化的可靠性。

## 当前范围锁

当前不实现模型训练、LoRA、SFT、RL、Trainer backend 或模型部署 registry。评测用 adapter
registry 不管理权重。多 LLM 可以作为 Proposer/Supervisor/Reviewer 候选，但不能绕过 evaluator。

## 后续方向

当前不保留付费云实例，等待下一个新任务窗口；下一项产品工作是在全新冻结任务上做
baseline/evolved Prompt/Skill/Policy 的 matched pass³。既有 5+10 题只作为 regression，不再用于
调参或晋升；Matplotlib 负例用于改进“冻结前回归门”的协议，而不是回改已冻结 prediction。4B
精确输出另走语法约束解码或确定性填槽，不能增加 repair 循环。正式 Agent 晋升仍需要新鲜多任务
对照，不能复用单臂 pass@1 capability run 冒充 sealed 泛化。

## 当前验证

2026-08-03 v2.0.0 关键验证：

- v1.1 final pass³ fingerprint：
  `d542039f30d97c5e4bc7080349e1aade45705bab435fc3b7f5da87fbcde2b05e`；
- v2.0 pass³ fingerprint：
  `99631241e2444c2ec3b23be5ad0e78e78e0936b9fde3021c12cecb40054d3c47`；
- 完整 pytest：`222 passed in 18.12s`；Ruff check 全过，`113 files already formatted`；
- v2 manifest：2 stages、1284 files，`valid=true`，fingerprint
  `fcf3d747dce0b5b8026fb19c36a28c6ff53c67976dfea12413352b6f7393c7fa`；
- live audit：6/6 调用成功且安全；纠错后 baseline/G2 `0.875/0.958333`，delta
  `+0.083333`，token reduction `-0.026281`，final promotion deferred；
- multi-model diagnostic：16 cells；4B baseline/G2 `0.239583/0.510417`，7B Coder
  `0.520833/0.479167`；正式模型晋升 rejected；
- schema adapter diagnostic：8 cells；raw `0/4 valid、0/4 safe`，schema treatment
  `2/4 valid、4/4 safe`；最大 repair 1，能力门 rejected；
- task-plugin routing diagnostic：24 cells；两个模型 routed mean score 均高于 monolithic，
  input tokens 均约减半，但 `agent_change` 使 routed 仅 `3/4 safe`，promotion rejected；
- SWE-bench official pilot：固定 5 tasks，prediction SHA
  `8834e7d70e2f797d2a67f3919c679b0eb820a3b898e840680fc410fb5b750d5d`，
  `5/5 resolved`、0 errors；证据 archive SHA
  `1e83e1d32842b5d699e250b627cafece3cb897681760b412aedf081041f26078`，
  内部 150/150 hash 通过；只接受 capability pilot；
- SWE-bench Batch 002：固定 10 个新 tasks，selection SHA
  `5ca666eb99728d97d55ea7a39744886e9caab1fa0379106624ace94dce7b8b83`，prediction SHA
  `6cb3ee0bd5351f05cc76719cb3c0102c663d736e7acc03f90624740507832be3`；官方 `9/10`
  resolved、0 errors；archive SHA
  `42a2c2ad41dfea1cf0c44895e997bf6aa91dd5b2ffc91e754661ed749577ad56`，内部 143/143 hash
  通过；实例、系统盘、临时安全组和临时 SSH 公钥的控制面精确 ID 计数均为 0；
- Release pass^3 fingerprint：
  `04dd9c6a5143f38c5cccff61a8c64176dca4962005e4ee1482e75ad8ac34efcc`；
- 当前 timing-independent RC fingerprint：
  `aa49e4257905c18133ba45dccce34bc477a9fd91119d7806481fb988770abac9`；
- 三套 final release 均 accepted，九阶段门全部通过。
