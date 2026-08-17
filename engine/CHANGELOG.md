# Changelog

## Unreleased

### 2026-08-06 v2.5 Agent Team 批次（T0-T5）

- T0 JLens 探针：MLX 手动逐层 forward（32 层 0.8s/步）可行，D3 定为不依赖外部 lens 库；
- T1 4B SWE 基线：swe_4b_patch.py + local_lens_agent.py（MLX 4bit）；裸 4B 0/4 resolved
  （dayjs-938 patch 可应用但 FAIL 5/498），验证 Skills 模式价值来自教学管道而非裸 4B；
- T2 教学 case 管道：T2-CASE.json（sphinx-8638，含 evidence sha256）+ SKILL.md 草稿（inactive）；
- T3 多模型教学 API：teacher_api.py（DS/OpenAI/Claude adapter + 加权输入，默认不调用外部 API），
  6 tests；
- T4 Skills 模式落地：promotion_ladder.py（人工晋升阶梯，append-only、30d TTL、reviewed/active/
  rejected、绝不自动）+ search_skill_bridge.compile_local_lens_candidate（本地轨→候选 Skill）
  + evolve_service skill-ladder/skill-ladder-status CLI；7+3 tests；
- v2.5 收口补充：pattern_card.py + t4_integration.py（PatternCard 聚合 + 本地轨候选编译→迁移门→晋升阶梯，6 tests）；
- v1.0.0 RC 重钉：member_count 120→124、expected_current/rc_evidence_fingerprint → d9574b40…、MANIFEST 全量重建 5946 文件（修正案 029 流程，问题 #6 待显式清单根治）；
- NEXT-2 JLens-vs-工具轨迹判别力复测：jlens_vs_tool_eta2.py（η² + LOOCV）；G0 32 臂工具轨 η²≤0.15、LOOCV 0.75（基线 0.72）→ 工具轨迹单侧判别力弱；JLens 侧缺配对 native 标签（协议已定义）；
- NEXT-3 问题 #6 根治：v1.0.0 bundle 改显式冻结清单（bundle-manifest.json，124 文件），新 root/test 模块不再改变 bundle 成员/指纹（回归测试锁定）；
- **INC-20260806-restart-missing-env（已恢复）**：首次重启漏设 EVOLVE_CODEX_REQUIRE_PROVIDER_CONFIG=1/EVOLVE_CODEX_NO_SANDBOX=1 → confirmation 8/9 任务 CFG-INVALID（0 token empty_patch，argv 含 --ignore-user-config）；已隔离无效证据、删除 g1:confirmation claim、重置 12 任务、重算 STATE.sha256，带 env 重启验证 argv 含 deepseek-v4 且无 --ignore-user-config；首个 GATE-RESULT（cost 0.2414）已标记 INVALID 作废，官方迁移门待新配对 ≥8；
- **INC-20260806-cpp-language-condition（已恢复）**：real-search-002 G1 confirmation 崩溃——trace_observer 对 C++ 题生成 `language:c++`，pattern_miner._IDENTIFIER 不允许 `+`；修复 = trace_observer._safe_condition（c++→cpp + 标识符合法化），回归测试 7 passed；resume 复用持久化 receipt（无重复扣款，real_codex_calls 100→105），paired 1→2/8；
- NEXT-1 执行器：gate_confirmation.py（paired≥8 时跑迁移门）；G1 confirmation 监控中（monitor_confirmation.sh，每 2min 轮询）；
- 全量 pytest 405 passed + 1 xfailed；
- 全量 pytest 404 passed + 1 xfailed（修正案 024）；ruff check/format 通过；
- v1.0.0 RC：member_count 120、指纹重钉 ace17a30…（修正案 029）、release.json/MANIFEST 同步（问题 #6 待显式清单根治）；
- 边界保持：权重冻结、本地轨只算数据不计计算预算、teacher 外部调用需单独授权、不打开 final sealed。

### 2026-08-05 实例生命周期策略：默认 stop、不 terminate

- 用户指示“以后逻辑改为停止吧，不要删除了，除非我说项目不做了”；
- RESOURCE-LEDGER 新增 instance_lifecycle_policy（stop_by_default）+ aws_instance_current
  （i-0b49272b2d58d034b）；HANDOFF/STATUS 同步；
- 计费备忘：运行 $0.1197/h；停止后仅 EBS 存储约 $24/月；公网 IP 会变（可绑 EIP）。

### 2026-08-05 修正案 027：健壮性批次 1/2/3（冻结保护 + runner 接线 + SIGTERM）

- #1 冻结保护：v21_artifacts 新增 FROZEN.json 守卫与 --freeze-override 审计；
  v2.1.0-continuous-ab 已加 FROZEN.json，--snapshot-sources 自动跳过冻结阶段；
- #2 真实 run 接线：real_evolution_run.py 新增 --worker-count、--skill-registry、
  --auto-gate（用 confirmation 配对证据跑跨任务迁移门），搜索结束自动产出候选 Skill；
- #3 SIGTERM 优雅收尾：TERM/INT 原子写 RESULT.partial.json/zh-CN.md（terminal_state=partial），
  退出码 128+signum；
- 全量 pytest 375 passed + 1 xfail（修正案 024）；ruff 全过；
- 未新增源文件（v1.0.0 bundle 不受影响）。

### 2026-08-05 修正案 026：v2.3 搜索→候选 Skill 落地 + 跨任务迁移门

- search_skill_bridge.py：把 v2.2 搜索结果（selected candidate + 冻结证据）编译为
  项目内候选 Skill（skill_registry，inactive），并加跨任务迁移门
  （>=8 对 fresh-task、native 非劣、无 safety 回归、cost<=10%、合同/epoch 一致
  → transfer_verified / rejected，负证据保留）；
- evolve_service.py skill-candidates 命令；
- 边界：project_local_only=true、auto_install=false、active=false、人工晋升；
- v1.0.0 release 维护：新增 2 个源文件使 RC bundle 成员 108→110、RC 指纹变化；
  已重钉 expected_current_rc_fingerprint=04843f90…、重建并验证 v1.0.0 MANIFEST
  （5946 文件，valid），门语义不变；
- 全量 pytest 367 passed + 1 xfail（修正案 024）；ruff 全过；
- 产物：artifacts/v2.3.0/v2.3.0-jlens-search-skill/（PLAN/PREDECLARED_GATES/DECISION/
  runs/bridge-pass1：3 候选编译、1 个 transfer_verified）。

### 2026-08-05 修正案 025：v2.2 收敛引擎（用户确认门后实现）

- 每代 convergence_metrics（candidate-vs-original / candidate-vs-parent 的
  native_score/cost delta + safety）；Pattern 反馈合并（A 修复 B → generalized_fix，
  仍 observational_not_causal）；连续 K=2 代 mean |Δ|<0.05 且无 safety 回归 →
  converged 停止（terminal_state: converged/exhausted/completed）；
- 候选验证并行化：worker_count 1-4，任务级并发、controller 变更串行化、
  证据逐 arm 冻结、并行/串行语义指纹一致；
- 新增 4 项测试（收敛指标 / K=2 停止 / A-fixes-B 合并 / 并行一致性）；
- 全量 pytest 363 passed + 1 xfail（修正案 024 文档化）；fixture pass³ 指纹
  2a7bfd14bd1a5f41ebeda70999277b69dc29776b1c7d8e4ca76fe547925f4734；
- 产物：artifacts/v2.2.0/v2.2.0-jlens-convergence/（PLAN/PREDECLARED_GATES/DECISION/
  runs/local-pass3-{1,2,3}）。

### 2026-08-05 修正案 024：冻结快照操作者修改（用户授权收口）

- 收尾重建 MANIFEST 时 `v21_artifacts.py --snapshot-sources` 误覆盖
  v2.1.0-continuous-ab 冻结 source-snapshot 5 个钉住文件；1 个（tests/test_agent_arm_runner.py）
  从会话记录逐字节恢复（哈希匹配），4 个本地不可恢复（用户确认无副本）；
- 修正案 024（用户授权）：阶段标记 degraded、钉住哈希保持权威、
  防篡改测试 xfail(strict=True)（恢复后 XPASS 提示移除）；
- 全量 pytest：359 passed + 1 xfailed（27.15s）；ruff check/format 通过；
  MANIFEST 有效；incident 状态 resolved_via_amendment_024。

### 2026-08-04 v3 窗口收尾（deadline 17:30:24Z）：G0+G1 scout/semifinal 完成，AWS 释放

- real-search-001-deepseek-v3 被 run 自带 timeout TERM 停止（无 SIGTERM 处理器）；
- 完成：G0 observe 32/32 臂、4 个 inactive 候选（0 失败）、G1 scout 30/30 臂
  （semifinal 晋级 skills+policy）、G1 semifinal 32/32 臂（confirmation 晋级 skills）；
- 中断：G1 confirmation claimed 且 2/36 臂完成，3727f492 候选臂在途被杀
  （incident `terminated-confirmation-c4a54ad9-3727f492`，fail-closed，不退款）；
- 记账：real_codex_calls=102（96 agent + 4 proposer + 2 重派发），未超 2000；
  任务打开 41/100，实例 1 台，未超上限；
- 结论：parent 未前进（parent_history 空）、agent_optimized=false、agentic_rsi=false、
  final sealed unopened、无 promotion；
- AWS 已释放：实例 i-08bacc8f91ce598c3 终止、SG sg-0a08296238f95812f 与密钥
  evolve-jlens-aws 删除；控制面归零（运行实例=0、自定义 SG=0、密钥=0）；
  本地私钥归档到 cloud-control/archived-keys/；
- 文档：run root 新增 RESULT.partial.zh-CN.md / DECISION.json / NEXT.zh-CN.md；
  RESOURCE-LEDGER 更新为 released；STATE 摘要 a5c06e0a774a… 与 STATE.sha256 一致。

### 2026-08-04 协议修正案 023：优雅降级 + rollback schema + v3 重启

- 优雅降级（用户确认）：proposer 逐候选 try/except（MutationContractError /
  RealProposalError / CodexMutationCallError），失败写入 PROPOSAL-FAILURES.jsonl，
  候选数允许 0-4；tournament 按实际候选数跑 scout/semifinal/confirmation
  （G1 最低 1 候选），0 候选走 _finalize_no_signal 写 partial RESULT；
  expected_agent_calls 按实际臂数动态计算；报告注明候选数<4 的原因；
- proposer rollback schema：response_schema.path_rule 明确“每项必须恰好
  {op,path,after}；op=delete 时 after 必须为 null”；
- v3 重启（08:57:07Z）：复用 controller 从断点续跑；被中断的 0dfaa203 parent 臂
  按协议 abort（incident resume-interrupted-parent-0dfaa203，evidence 8c506d…）
  后重派发（不退款，calls 11→12）；已完成 9 臂 evidence 保留；
- 本地 360 pytest + ruff 全过；8 个运行时文件远端哈希与本地一致；
- baseline 合同核验：文件 sha 834ba66d… 两端一致，
  baseline_contract_sha256=0826c478…，token_budget=20,000,000。

### 2026-08-04 协议修正案 021：DeepSeek 无 token 门 baseline + v3 全新任务

- 按用户指示采用 `permanent-baseline-deepseek-notokengate.json`
  （token_budget=20,000,000，contract_sha256 0826c478…；原 baseline 均未改动）；
- safety_passed 不再包含 token 用量维度（报告将如实标注）；
- 新建 `SEARCH-TASK-SCHEDULE-DEEPSEEK.json`：91 个全新未打开任务
  （G0=16，G1-G3=25×3，seed evolve-jlens-deepseek-2026-08-04，
  fingerprint 3129b795…），排除 INCIDENT-016 丢失题、v2 已打开题与
  terminal-bench-2；
- v2（TOKENGATE）归档：32 臂真实 G0（token 门导致 safety=false）、
  4 个 admitted inactive 候选、G1 scout 已开始；
- v3（real-search-001-deepseek-v3）已启动：首个真实臂 safety_passed=true，
  G0 25 题推进中；修正案 021 已落盘。

### 2026-08-04 DeepSeek provider（AWS）真实 G0 启动

- 腾讯云已确认归零（ap-guangzhou CVM/Lighthouse=0）；
- 按用户指示改用 DeepSeek（cc-switch 配置的 custom provider，
  deepseek-v4-flash，api.deepseek.com），创建新 baseline authority
  `permanent-baseline-deepseek.json`（contract sha 19e74d08…，原 baseline 冻结）；
- 修复 `build_codex_argv`：`EVOLVE_CODEX_REQUIRE_PROVIDER_CONFIG=1` 时保留
  实例 provider 配置并传 `-c model_provider=custom`（否则 --ignore-user-config
  会丢弃 DeepSeek 配置，arms 以 0-token 失败——已归档
  `real-search-001-deepseek`）；
- `real-search-001-deepseek-v2` 已启动：Task 10（dayjs）original/parent 均为
  真实成功调用（returncode=0，tokens 221K/137K），后续任务推进中；
- 两个无效 run（QUOTA-INVALID、deepseek CFG-INVALID）已归档，详见
  EXECUTION-PROTOCOL-AMENDMENT-020。

### 2026-08-04 更正：配额耗尽，真实 G0 未执行

- 逐臂复核 32 份 receipt 证实：ChatGPT/Codex 配额在腾讯云 #2 启动前已耗尽，
  腾讯云 #2 14 臂与 AWS 18 臂全部为 0 token / rc=1 / usage-limit；官方 evaluator
  对空补丁给出 resolved=false 被误记为有效证据；
- 此前“G0 完成 16/25 题”报告作废；真实有效证据=0，G0 未完成；
- 污染 run 归档为 `runs/real-search-001-resume-QUOTA-INVALID/`（保留失败 receipt
  与 33 个已记账 reservation）；配额重置约 08-08T04:10Z；
- 恢复需用户解决配额后以新 run root 重跑 G0（Task 10-25 未真正暴露，可复用）。

### 2026-08-04 迁移到 AWS 续跑（用户决策）

- 用户决定切换到 AWS（新账号 829485866839，ap-southeast-1），不再使用腾讯云；
- 腾讯云实例 `ins-fscmj63s` 已终止、安全组 `sg-rqlenhez` 与密钥 `skey-ld4i6yon`
  已删除（requestID 见 RESOURCE-LEDGER）；中断前证据（G0 已完成 7/16 题、
  14 臂 evidence、real calls=14）已同步到本地并部署到 AWS；
- AWS 实例 `i-08bacc8f91ce598c3`（52.77.226.250，m7i-flex.large 2vCPU/8GB，
  250GB 根卷，新加坡）已创建并 bootstrap；账号受 Free Tier 限制，m7i-flex.large
  是可用的最大 x86 类型；
- 本机 Clash TUN 会黑洞 AWS 出网流量：已为 AWS 新加坡 IP 段加入 DIRECT 规则并重载
  mihomo（原配置备份 clash-verge.yaml.bak-evolve-aws）；
- AWS 直连互联网，无需反向隧道/loopback 代理；运行将不再设置任何代理环境变量。

### 2026-08-04 恢复路径 A：real-search-001-resume 已启动

- 用户决策选 A（从 Task 10 续跑、G0 partial）；清理旧临时安全组 `sg-gx6m9m9x` 与
  密钥 `skey-a1xnwt11`（已核验删除）；
- 新增 resume 计划支持：`EvolutionPlan.build_resume`（91 题：G0=16/32 calls，
  G1-G3=25 各 98 calls，planned 366 calls）、schedule loader `1.1-resume`、
  集成测试；全仓 351 pytest 通过；
- 生成 `SEARCH-TASK-SCHEDULE-RESUME.json`（fingerprint `4601ba19…`）与
  `runs/real-search-001-resume/lost-tasks.json`（Task 1-9，引用 INCIDENT-016，
  不可重发）；
- 新实例 `ins-fscmj63s`（129.204.18.20，同规格）已创建并 bootstrap；
  反向隧道与 loopback 代理就绪；preflight 通过；
- 修正案 016：Multi-SWE 仓库 prefetch 经隧道代理克隆（github.com 直连不稳定，
  隧道克隆已实测含 base commit 校验），启动同时设置
  `EVOLVE_CODEX_HTTPS_PROXY` 与 `EVOLVE_BENCHMARK_HTTPS_PROXY`；
- 续跑已启动（G0 Task 10 dayjs original 的官方 Multi-SWE evaluator 运行中），
  逐题证据/controller rsync 回本地循环运行中。

### 2026-08-04 INCIDENT-016：real-search-001 云证据不可恢复

- 实例 `ins-gdqrjr5a` 于 2026-08-03T22:41:41Z 被 root 账号经腾讯云控制台（微信 MFA 已验）
  `TerminateInstances` 终止；远端 `/opt/evolve-v211/real-search-001`（controller STATE、
  18 臂 evidence、native-evaluator、incidents、archive）随实例与数据盘销毁，无快照/无 COS
  备份/无本地副本，证据不可恢复；
- 中断前 G0 完成 9/25 题（18 个有效 arm evidence、约 23–24 次真实 Codex 调用），
  final sealed 未打开、无 promotion；已记账调用不可重发；
- 本机侧 348 项 pytest 通过，schedule/题池/baseline/修正案 002–015 均保留；
- 已写 partial RESULT 与 INCIDENT-016，更新 RESOURCE-LEDGER（`released_resource_ids=
  ["ins-gdqrjr5a"]`，安全组/密钥保留待用户决定）；当前停在 HUMAN_REQUIRED。

### v2.1.1 JLens multi-generation evolution engine — 2026-08-03

- 将主线从固定 baseline/candidate 大规模 A/B 修正为 Observer→PatternCard→multi-candidate mutation→
  tournament→search-parent 的多代 Agent 应用层进化；
- 新增 advantage/failure 双向 Pattern Miner，证据卡包含 hash 引用、条件、反例、置信度、作用面和
  `observational_not_causal` 边界；
- 新增 provider-neutral inactive MutationProposer，允许 Prompt、Skills、Policy、router、memory
  policy 与受限 harness code，拒绝权重/evaluator/global Skill/auto apply；
- 新增 immutable CandidateArchive、hash-chained lineage events、负候选/rollback 保留，以及
  native-evaluator-only 4→2→1 tournament；
- 新增四代 100-task EvolutionController、logical/real call 分账和 100 tasks/2000 calls/1 instance/
  24h/¥30 fail-closed budget；
- 本地 pass³ 每 run 完成 100 fixture tasks、344 逻辑 arm、16 逻辑 proposer，真实 Codex calls=0；
  10 PatternCards、16 proposals、3 selected、9 rejected、4 inactive，三个指纹一致；
- v1 clean-room 继续冻结为 108 members；final sealed 未打开、production/global ref 未写，明确不声明
  Agent 已优化或 Agentic RSI。

### v2.1.0 Continuous multi-benchmark matched A/B — 2026-08-03

- 新增 SWE-bench Verified、SWE-bench Multilingual、Multi-SWE-bench-flash 与 Terminal-Bench 2
  四个 version-pinned adapter，并冻结数据、harness 和原始输入 SHA-256；
- 从 1189 个源任务识别 14 个跨集重叠、排除 16 个历史已打开身份，形成四集各 75 个、合计
  300 个新任务的 search/promotion/final_sealed `160/80/60` 生命周期池；
- 新增永久 shadow baseline、matched prediction freeze、可恢复 round ledger、任务永久退役、
  周期性 ChangeSet proposal、paired promotion/rollback 和 one-shot final sealed auditor；
- 预声明最多 8 个 search/promotion 周期，60 个 final sealed 任务当前全部 unopened；
- 三次无模型协议 smoke 得到同一语义指纹；v2.1 文件与冻结 v1 clean-room
  发布包隔离，历史 108-member RC 指纹保持不变；
- 完成正式 300-round 的 tokens/墙钟情景预算并设置 `HUMAN_REQUIRED`：建议首批仅授权 10 个
  search task、20 次真实 Codex 调用、1 台 24 小时内临时实例和 ¥200 云费用硬上限；
- 当前 `completed_rounds=0`、无 ChangeSet 晋升、未打开 final sealed，明确不宣称 Agent 已优化。
- 完成性反证审计发现原 accepted 范围不足，新增 prediction 评分前重验、round/registry/SERVICE
  integrity SHA、规划前 crash reconciliation、partition/cycle candidate 冻结和 10-pair proposal 绑定；
- 以 pinned Harbor 源码确认 agent-sync-verifier 顺序，新增 `FrozenCodexAgent` 在 verifier 前生成
  ATIF prediction receipt，替换无法证明 phase boundary 的直接 `--agent codex` 合同；
- 零副作用预声明覆盖四 adapter 的首批 10 个 search task；preflight 仍为 `HUMAN_REQUIRED`，未调用
  Agent 或创建云资源；加固 pass³ 指纹一致，全仓更新为 `257 passed`。
- 新增 claim 后按需物化任务、隔离 matched Codex arm runner 和 native result admission；patch 会纳入
  未 tracked 新文件，空/失败 prediction 保留为 unresolved，不再形成成功样本选择偏差；
- Verified、Multilingual、Multi-SWE 与 Terminal/Harbor 四条 evaluator 路径统一冻结 receipt、原生
  report 和归一化 `ArmResult`，双臂 evidence 入账后任务才可 retired；31 个定向测试、29 项阶段检查
  和全仓 `269 passed` 通过，真实 Agent/native evaluator 调用仍为 0，质量结论保持 none。

### Fail-closed SWE-bench Batch 002 — 2026-08-03

- 新增本地 fail-closed runner ledger，固定版本、任务 selection、prediction、证据与云资源生命周期；
- 在查看题面前预声明 10 个新 Verified 任务，不换题；10 条非空 prediction 在 official evaluator
  前一次冻结；
- official harness 一次完成 10/10，得到 9 resolved、1 unresolved、0 errors；
- `matplotlib__matplotlib-25960` 的新增测试通过，但 4 个既有 PASS_TO_PASS 布局测试失败，原样保留
  为回归证据，official run 后未修改 prediction；
- 云端 archive SHA 在本地一致，143/143 内部文件通过；即时销毁被微信 MFA 拦截后使用预设自动销毁；
  实例、系统盘、临时安全组和临时 SSH 公钥最终均删除，控制面精确 ID 计数全部为 0；
- 只接受为 capability/regression evidence；不宣称榜单、matched pass³ 或 Agentic RSI。

### Official SWE-bench 5-task capability pilot — 2026-08-03

- 在临时腾讯云 x86_64 按量实例上固定官方 SWE-bench commit、`datasets==5.0.1` 和
  `SWE-bench_Verified` revision；
- 在查看题面前以冻结哈希算法从 499 个非 gold 任务中选出 5 个，不换题；
- gold 环境验证 1/1 resolved，独立于 Agent 分数；Docker Hub 失败后以相同 prediction 配置
  腾讯云镜像代理重试，保留首次失败；
- 5 个 Codex prediction 在 official evaluator 前一次冻结，最终 5/5 resolved、0 errors，
  F2P 8/8、P2P 279/279；
- 保留错误 image key、shell 伪成功、目录顺序、错误 bind、SymPy 首候选与 hunk count 等负证据；
- 云端证据 archive 本地 SHA 一致，内部 150/150 文件通过；实例、250 GiB 系统盘、两个安全组
  和云 SSH 公钥已删除；
- 只接受为官方 5-task capability pilot；不晋升 G2、Skill、Policy、模型或 Agentic RSI。

### Task-specific G2 plugin routing — 2026-08-03

- 新增严格 task-family router，只允许项目内相对路径并要求 routed profile 覆盖全部 family；
- 运行前冻结 base profile、唯一 plugin 与最终 task system-prompt SHA，旧 suite 保持兼容；
- 完成 2 模型 × baseline/monolithic/routed × 4 public tasks 的 24-cell 真实 MLX matrix；
- routed 相对 monolithic：4B mean score `+0.114583`、input tokens `-50.3147%`；Coder-7B
  mean score `+0.09375`、input tokens `-49.9803%`；
- `agent_change` 在两个模型上仍为 `0/unsafe`，导致 routed 只有 `3/4 safe`，预声明 public 总门 rejected；
- router 软件保留，active profile/G2/模型/Agentic RSI 均未晋升；下一步组合 route-level schema harness 与 fresh tasks。

### Schema-constrained ChangeSet adapter — 2026-08-03

- 新增 strict ChangeSet validator、前置 JSON Schema、最多一次 bounded repair 和逐次 raw/usage/latency evidence；
- 旧 suite 未声明 adapter 时保持 raw-only，所有候选 `auto_apply=false`；
- 首次矩阵发现 PLAN/prompt 允许 `repetition`、实现只匹配 `repeat` 的 evaluator 漂移，保留负 run；
- TDD 纠正后重跑：raw `0/4 valid、0/4 safe`，schema treatment `2/4 valid、4/4 safe`；
- Coder-7B 两格 first-pass accepted，4B 仍复述 schema 或缺 matched A/B；预声明 `3/4` 能力门 rejected；
- 修复 v1 clean-room 后置文件排除边界，保持冻结 108-member RC 指纹与发布测试；
- 没有晋升模型、G2、ChangeSet 或 Agentic RSI，下一步拆分 task-specific G2 plugins。

### Multi-model and SWE-bench adapters — 2026-08-03

- result-handoff grader 增加 schema v2 freeze；多模型 grader bundle 同时冻结 rubric 与实现 SHA；
- 新增 localhost-only MLX adapter、Qwen3.5-4B / Qwen2.5-Coder-7B 实际模型矩阵和可选 7B general probe；
- 修复 MLX `<|im_end|>` 尾 token 与 inline unified-diff 等价修复误杀，保留 pre-fix 原始矩阵；
- post-fix 4B baseline/G2 `0.239583/0.510417`，7B Coder `0.520833/0.479167`；
- 新增 SWE-bench prediction/harness adapter、test-path poisoning preflight 和 Docker daemon readiness；
- 本机资源门未满足，官方 SWE-bench resolved-rate 保持 null；没有晋升模型、G2 或全局 Skill。

### Live Codex matched A/B — 2026-08-03

- 用户授权后，以 G2 freeze 后第一条真实任务运行 baseline/G2 各三次，冻结
  `gpt-5.6-sol`、low reasoning、task、evidence 与 read-only sandbox；
- 六次真实调用均成功且安全，原始 JSONL、回答、usage、latency 和失败信息完整保留；
- 发现并保留原正则 grader 对中文同义表达、跨行 decision 和 artifact path 的假阴性，原决定为
  `not_promoted`；
- TDD 修复 grader 后只重评原回答：baseline `0.875`，G2 `0.958333`，delta `+0.083333`；
- G2 token 增加 `2.6281%`，且修复发生在输出检查后，因此只记 provisional，正式 promotion
  deferred、Agentic RSI 继续 rejected；
- 未修改全局 Codex、未安装全局 Skill、未 push/publish。

### v2.0.0 MetaProgram — 2026-08-03

- 新增三代带 hash lineage 的 MetaProgram，演化 proposer、search budget 与 routing；
- public mean `0.1667→1.0→1.0`，候选预算 `3→2→1`，单位成本分数 `0.0556→0.5→1.0`；
- protocol-order meta-sealed `0.2167→1.0`，原 v1.1 任务 `1.0→1.0` 无退化；
- v2 软件 release accepted，Agentic RSI 因 fresh task/live execution 缺失而 rejected；
- 三次运行 fingerprint 一致，保留 MetaProgram transition/rollback patch、archive 和中文报告。

### v1.1.0 Real Codex Target — 2026-08-03

- 只读索引 14 条真实 Codex 用户任务，排除系统/开发者/assistant/tool/reasoning；
- 接入本地 Codex CLI identity 与项目原生 AGENTS/Skill/Policy 表面，不调用模型；
- 生成首个三表面 AgentChangeSet、中文报告、apply/rollback patch，未自动应用；
- public `0.1875→1.0`，protocol-order sealed `0.1667→1.0`，patch 往返树哈希一致；
- 明确 sealed 非 developer-blind、live improvement 未证明，避免把机制分数冒充 RSI。

### v1.0.0 Local Release — 2026-08-03

- 新增稳定后台 CLI：`run / inspect / verify / rollback-plan`，无隐式网络或外部写入；
- 新增只读 artifact manifest verifier 与九阶段 release gate verifier；
- 首次 formal pass^3 暴露 pytest duration 进入语义 identity 的非确定性，保留 rejected run 后修复；
- post-fix 三次 Release 均 accepted，语义 fingerprint 为
  `04dd9c6a5143f38c5cccff61a8c64176dca4962005e4ee1482e75ad8ac34efcc`；
- 170 tests 与 93 个维护 Python 文件 lint/format 通过；生成中文最终报告与 source snapshot；
- 本地 v1 Goal 完成；未 push、merge、publish，不宣称生产就绪或 Direct LLM 泛化。

### v0.9.0 Release Candidate — 2026-08-03

- 三套 RC 共完成 9 个独立 durable operation，全部 attempt 1 completed 且幂等 replay；
- 每个 operation 从 raw events 证明三 seed、payout/refund 与 public checkpoint→sealed 顺序；
- 修复 AgentProgram 缺磁盘 checkpoint、Integration contract 绝对路径污染两个发布阻断；
- 保留 pre-repair config/负证据，未覆盖 v0.2-v0.8 历史 artifacts；
- clean-room tar 108 成员安全，核心 16 tests 与 integration CLI replay 通过；
- 159 tests、87 个维护文件通过；进入 v1.0.0 最终 Release。

### v0.8.0 Reliability and Security Hardening — 2026-08-03

- 新增 SQLite schema v2 DurableOperation 状态机、事务 events、lease 与 contract conflict；
- prepared/running 两处 crash injection 均可恢复，未过期 worker 竞争被拒绝，过期后安全接管；
- completed result SHA 不符时保留损坏字节并以新 attempt 重建，不把部分写冒充完成；
- legacy v1→v2 migration 保留 operation/event，真实并发测试只执行一次；
- timeout、artifact budget、路径/schema/input budget 失败均关闭且不发布 result；
- 三次故障矩阵一个 fingerprint；158 tests、85 个维护文件验证通过；进入 v0.9.0 RC。

### v0.7.0 Integration — 2026-08-03

- 新增严格可哈希的 PluginEnvelope，显式区分 execute/observe/propose/persist/admit 权限；
- 直接复用 v0.2-v0.6 运行入口组成一个 CLI 纵向切片，没有复制组件实现；
- 三次独立运行均 accepted 且语义 fingerprint 一致；六组件 decision 与 11 个合同门全过；
- JLens failure injection 被隔离，Observer 不参与 admission；只有 Admission Gate 有 active ref；
- 同 operation 重放返回已有结果且不改变 result/log hash；changed contract 被拒绝；
- 144 tests 与 81 个维护 Python 文件 lint/format 通过；当前进入 v0.8.0 Hardening。

### v0.6.0 Evaluator Shadow — 2026-08-03

- 新增不可写 admission 的 evaluator shadow cross-play runtime；
- 冻结 7-program × 10-outcome corpus、anchor truth、当前 epoch 和三个 candidate；
- lax-v1 检出 2 个 false accept，strict-v1 检出 2 个 false reject；
- robust-v2 无 decision disagreement、champion 稳定、rank correlation=1.0，只生成
  `activation_allowed=false` 的 epoch-boundary review proposal；
- shadow on/off 的 anchor score/admission/champion 完全相同，active evaluator 未切换；
- 当前进入 v0.7.0 Integration。

### v0.5.0 受限 Agent Code Mutation — 2026-08-03

- 新增极窄 `select_route` AST capability allowlist，拒绝 import/call/attribute/loop/IO 等；
- 只有静态门通过的候选才进入 `python -I -S`、空环境、空 cwd、无 builtins 子进程；
- 三个 import/open/while 越权候选未执行；行为回归候选仅跑 public 后拒绝；
- 安全路由候选 public `3/6→6/6`、sealed `2/4→4/4`，三 seed 一致；
- verified 候选短暂激活后完成 rollback drill，最终 active hash 恢复 parent；
- 明确保留 macOS address-space rlimit 未生效与“非通用 Python 沙箱”边界；
- 当前进入 v0.6.0 Evaluator Shadow。

### v0.4.0 PSI Skill Library — 2026-08-03

- 新增严格、append-only、项目内 `SkillCandidate` Registry；禁止 active、auto-install 和
  全局 Skill 写入；
- candidate 强制带来源 task/evidence hash、适用 predicates、反例和已知失败模式；
- 历史 payout 负迁移作为 `legacy-record-cleaning-bundle-v1` rejected revision 保留；
- 新增 refund 相邻任务，与 payout 一起完成双 target、双 arm、三 seed matched PSI A/B；
- 全部 public 落盘后才打开 sealed；两个 target transfer 均为 public/sealed 100%，来源
  replay 保持 `13/13`、`6/6`；
- 三次独立运行一个 fingerprint，新 candidate 仅标记 inactive `transfer_verified`，未安装；
- 当前进入 v0.5.0 受限 Agent Code Mutation。

### v0.3.0 JLens Observer — 2026-08-03

- 新增严格 `ObservationArtifact` schema，统一 off、trace、logit-lens、JLens 输出；
- runtime 结果先落盘，再在线程 Sidecar 中运行 Observer，artifact 永不进入 admission；
- 四模式各三次 matched replay 的 runtime、active program、public 与 sealed 完全一致；
- JLens collector 故障注入被隔离，runtime 仍然 accepted；
- JLens 相对 logit-lens 的预声明增量是 `-0.0017566`，未过 `+0.01` 门，明确记录为
  `not_supported`，只保留诊断插件；
- 236 个运行文件、源码 tar 与阶段文件全部写入 SHA-256 manifest；
- 当前进入 v0.4.0 PSI Skill Library。

### v0.2.0 AgentProgram replay — 2026-08-03

- 新增版本化 `AgentProgram`、Component Registry、MutationProposal 和 ReplaySupervisor；
- 只开放 Prompt instruction、Skill composition、retry policy 三个 mutation 轴；
- 固定 Replay Agent、evaluator、预算和三个 seed，public `3→4→10→13/13`；
- sealed 仅在搜索后打开，baseline/final 为 `0→6/6`，三个 seed 全部非劣；
- 三次独立运行一个 outcome fingerprint，保存 events、archive、active program 和 evidence；
- 不调用模型或网络，不修改 task program/harness，不宣称 Direct LLM、JLens 或 PSI 收益；
- 当前进入 v0.3.0 ObservationArtifact 与 Observer 隔离。

### v0.1.0 Kernel freeze — 2026-08-03

- 建立 `artifacts/v1.0.0/` 统一证据根目录和长期 Goal 索引；
- 只读索引 22 个既有 Kernel 输入/证据文件，没有移动或覆盖旧产物；
- 验证 SQLite integrity、三个 final pass^3 run、唯一结果指纹及 public/holdout 顺序；
- 明确 task-program mutation 只是 Kernel smoke，不代表 Agent 自进化；
- 当前进入 v0.2.0 `AgentProgram + ReplaySupervisor`。
