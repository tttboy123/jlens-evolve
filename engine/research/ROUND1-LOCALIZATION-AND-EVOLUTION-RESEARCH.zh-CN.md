# Round 1：目标定位、受限编辑与自进化闭环的前沿做法

> 调研日期：2026-08-09  
> 范围：只采用论文作者、项目官方仓库或实验室官方页面；结论用于旁路实验框架，
> 不修改冻结引擎。

## 结论

当前的 24/60 目标覆盖不是“再写一版 Skill”能解决的问题，而是上游
**localization scaffold 不足**。社区和前沿实验室的共同做法是把问题拆成：

`定位候选 → 受限生成多个修改 → 外部 evaluator 验证/排序 → 结果进入 archive`

因此本项目应在 `skill_evolution_loop/` 中加入独立、gold-free 的分层定位阶段，且将其在
baseline/taught 两臂之间冻结。Skill 因果 A/B 只在相同定位、动作空间、采样预算和 verifier
下比较；黄金 patch 只允许进入 evaluator-only 覆盖审计。

## 一、社区工程路线

### Agentless：定位、修复、验证三阶段

Agentless 官方仓库把流程明确拆成三段：先按文件、类/函数、细粒度编辑位置进行层级定位；
再对编辑位置采样多个候选 patch；最后选择回归测试、生成 reproduction test，并按测试结果
重排候选。它还公开了预处理后的仓库结构 artifact。

对本项目的直接含义：

- `issue-lexical-source-ranker` 只能做第一阶段召回，不能直接把第一名当最终目标；
- Round 1 应保留 top-K 文件，再做 symbol/excerpt 级定位；
- 一个诊断可以生成多个 realization，native evaluator 之外的结构/语法/定向测试只做筛选；
- 定位结果与候选生成结果分别冻结，避免把定位错误归因给 Skill。

来源：[Agentless 官方仓库](https://github.com/OpenAutoCoder/Agentless)

### SWE-agent / mini-SWE-agent：让模型拥有可审计的代码检索与执行界面

SWE-agent 的核心主张是 Agent-Computer Interface；当前项目官方又将默认实现收敛为
mini-SWE-agent：只保留 bash、线性消息历史和独立 `subprocess.run` 动作。官方说明这种设计
便于 sandbox、扩展、轨迹调试和微调研究，并减少 scaffold 过拟合。

对本项目的直接含义：

- 不需要把复杂 agent runtime 塞进冻结引擎；旁路中提供少量确定性检索动作即可；
- 每次定位动作和输出都应线性记录并带 SHA；
- 学生若不能可靠调用工具，可由固定 localizer 先生成候选文件/符号，再交给 4B 做 typed edit；
- 保留一个极简 bash/search baseline，防止复杂定位器的提升被误认为 Skill 提升。

来源：[SWE-agent 官方仓库](https://github.com/SWE-agent/SWE-agent)、
[mini-SWE-agent 官方仓库](https://github.com/SWE-agent/mini-swe-agent)

## 二、前沿实验室路线

### Google DeepMind AlphaEvolve：宽度模型、深度模型、自动 evaluator 与候选数据库

Google DeepMind 对 AlphaEvolve 的官方描述是：用高效模型扩大探索宽度，用更强模型提供
深度建议；候选程序被自动 evaluator 执行和评分，并存入实现进化选择的程序数据库，后续
prompt 从数据库中的高价值候选继续演化。该方法适用于能够给出客观量化评价的领域。

对本项目的直接含义：

- DeepSeek V4 Flash 适合做高信息量的 localization/strategy/critic，而不是无限重复同一
  4B prompt；
- Qwen 4B 负责受限 typed operator/span realization，确定性 renderer 负责落盘；
- 只有 external native evaluator 能给 resolver credit；模型自评不能修改候选或验收条件；
- append-only Skill/trajectory archive 应保留失败候选，后续按失败类型抽样进入 teacher prompt；
- 3M tokens 是 campaign 总预算，仍应按“是否增加覆盖/能力”停止低价值分支。

来源：[Google DeepMind：AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)

### SWE-bench：人工可解性、容器化复现与私有测试边界

SWE-bench 官方仓库说明 Verified 是 500 个由真实软件工程师确认可解的问题；官方 harness
使用 Docker 做可复现评测。其 Multimodal 测试 split 保持私有并通过云端工具提交，体现了
生成侧与最终裁判侧的隔离。

对本项目的直接含义：

- `search` / `feedback` / evaluator-only `holdout` 必须继续隔离；
- gold patch 只可用于资格审计，不能进入 TaskSet、Skill、prompt 或 feedback；
- native harness receipt 是最终结果，apply/AST/syntax 仅为成本较低的前置门；
- 任务身份、base commit、容器/环境版本和输出 SHA 必须冻结，避免评测漂移。

来源：[SWE-bench 官方仓库](https://github.com/SWE-bench/SWE-bench)

## 三、落到本项目的目标架构

```text
Public issue + pinned source
        |
        v
Gold-free retrieval (top-K files)
        |
        v
Frozen localizer (file -> symbol/excerpt)
        |
        +---- same localization receipt ----+
        |                                    |
 baseline 4B                            taught 4B + Skill
        |                                    |
        +------- typed operator/span --------+
                         |
                         v
 deterministic materializer -> structural gates -> native evaluator
                         |
                         v
 append-only trajectory/Skill archive (inactive)
```

实施顺序：

1. 用 evaluator-only artifact 量化 top-K target recall 和单/多文件容量；
2. 将 top-K 检索与最终编辑目标选择拆开，不再固定第一文件；
3. 定位 receipt 在两臂之间完全一致，先证明定位覆盖，再做 Skill A/B；
4. 只从覆盖且机制容量合格的候选中冻结 60 题；
5. feedback 先跑，出现 native gain 后一次性打开 holdout；
6. DeepSeek 只看 public issue、source summaries 与匿名失败轨迹，永不接触 holdout/gold。

## 四、初始框架评估

- 正确：旁路隔离、inactive Skill、typed action、deterministic renderer、native final judge、
  append-only evidence、feedback/holdout 门。
- 尚缺：独立层级 localizer、top-K context、candidate reranking、候选池级 qualification、
  两文件事务型 edit plan。
- 当前 60 题 v2 的正式 evaluator-only 审计仅有 **24/60** 覆盖；因此不能直接启动 120 cells。
  该负证据应保留，并用新 localizer/selection 生成 v3，而不是覆盖 v2 artifact。

## 五、落地结果（2026-08-09）

上述建议已经在旁路框架内落地：检索扩展为冻结的 top-32 文件候选，operator 接收多 symbol
context，span 扩展为最多两个文件的 atomic exact-span bundle；candidate universe、source
cache、qualification、selection、TaskSet 和 target audit 均分别冻结，旧 v2 负证据没有覆盖。

最终从 109 个 search-only 候选中冻结 60 题，八语言覆盖，30 feedback + 30 holdout；正式
evaluator-only target audit 为 **60/60 ready**。`round1-feedback-run` 只允许执行恰好 30 个
feedback tasks，holdout 仍封闭。首个 taught 结构改善样本已经通过官方 native judge 检查：
补丁可应用但目标测试未通过，说明当前瓶颈已从 broad localization 转移到定位正确后的
semantic realization；它不能被误记为 Skill gain。

## 六、Native evaluator 的平台长尾与框架边界（2026-08-09）

Round 1 在 Apple Silicon 本地首次构建 `django__django-15277` 的官方 x86_64 instance image
时，环境镜像构建完成后，`setup_repo.sh` 在 qemu 下执行了约 42 分钟的
`git gc --prune=now --aggressive`。进程持续占用约 360% CPU，临时 pack 持续增长，随后正常
完成并进入 `pip install -e .`；因此这是**可观测的架构仿真长尾**，不是 harness 死锁。

该命令不能从本项目裁掉。SWE-bench 官方 Python test spec 用它清除目标提交之后的 reflog、
tag 和不可达对象，并在后续检查中确认 future history 不可见。跳过它会改变官方裁判的隔离
语义，形成潜在数据泄漏。当前正确边界是：

- 官方 harness、setup script、test patch 和 resolved 判定保持不变；
- 旁路框架在 A/B 前建立独立的 image-prewarm 阶段，记录 host/docker 架构、instance image
  identity、构建日志 SHA 和完成状态；
- `cache_level=instance` 保证同一任务的 baseline/taught 复用同一干净镜像，patch 只进入各自
  的 disposable container；
- image build timeout、repository/network error、Docker/qemu failure 归类为 infrastructure，
  不记作模型 unresolved，也不解锁 holdout；
- 中断后只允许从已校验的 instance image 或完整 native receipt 继续，禁止操作者补写结果；
- 大批量正式评测优先放到原生 x86_64 Linux runner；Apple Silicon 保留为本地冒烟与缓存后
  的单任务复核环境。

这与社区实践一致：SWE-bench 用 Docker 固定可复现环境，官方 harness 本身支持 image cache；
mini-SWE-agent 将执行环境与 agent 轨迹分离；AlphaEvolve 类系统也把候选生成与外部 evaluator
分开。对本项目而言，基础设施预热可以优化成本和恢复性，但不得降低 native 完成门。

来源：[SWE-bench 官方仓库](https://github.com/SWE-bench/SWE-bench)、
[SWE-bench 官方 Python test spec](https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/test_spec/python.py)、
[mini-SWE-agent 官方仓库](https://github.com/SWE-agent/mini-swe-agent)
