# Evolve × JLens 评测与验收契约

## 一、能力门槛

- OpenEvolve 必须使用确定性的非 LLM evaluator 评估候选变体。
- 每个公开案例必须输出独立数值指标；多通过一个案例必须压倒旧权重产生的平局差异。
- 被接受候选必须保留父代已经通过的全部公开案例。
- 拒绝源码重复和 AST 重复；行为等价结构变体必须设上限。
- 被拒绝候选保留在证据链中，但不得进入 islands、archive 或 best 跟踪。
- MAP-Elites 使用公开案例通过率和 AST 复杂度。
- 提示词不可见的确定性隐藏案例只用于搜索后的验证。
- 每 5 轮必须生成 checkpoint，trace 必须保留父代与子代代码。
- 跨进程只允许自动恢复兼容 checkpoint；不兼容契约必须拒绝恢复。
- 候选事件、经验、候选技能、元策略实验和 manifest 必须持久化在项目作用域 `state/` 下。
- 每个可执行候选必须先经过静态筛查，再进入受限执行。
- 演化谱系中的每个唯一程序，必须在同一固定提示位置、相同 31 个拟合源层上生成 JLens 与 logit-lens 签名。
- 变体聚类只能使用 lens 特征，evaluator 分数必须在聚类完成后再关联。
- 分析必须报告聚类稳定性、silhouette、结果关联、置换不确定性、组件变化、AST 变化和代表变体。
- JLens 必须与 logit-lens 和特征打乱空基线对照。

## 二、数据质量门槛

- 变体边 ID 唯一；出现内容冲突的重复 ID 时必须失败。
- 每条被分析边都必须存在父代代码、子代代码和组件级指标。
- lens 与 attribution 的全部数值特征必须有限。
- 解释任何聚类时必须同时给出样本量和分数变化分布。
- 所有结论保持观测性质：不声称已经做 head/MLP attribution，不使用因果措辞。
- RSI 必须包含能提高改进产率的搜索算子自修改；只有候选提升不能通过 RSI。
- PSI 必须把同一搜索恢复与跨任务经验迁移分开报告。

## 三、RSI 验收

以下条件必须同时满足：

1. 候选程序相对初始程序提升；
2. 严格连续改进深度至少为 2；
3. 接受候选的父代案例回归数为 0；
4. 搜索算子至少修改 1 次；
5. 至少一次修改使修改后的有效改进率高于修改前。

## 四、PSI 验收

### 同一搜索恢复

- 至少发生一次真实进程重启；
- checkpoint 状态、当前最佳公开分数、隐藏分数都得到保留；
- task、初始程序、evaluator 和 config 哈希一致。

### 跨任务迁移

- 经验必须来自不同的 task ID；
- 经验在源任务上经过隐藏集验证；
- 目标运行必须记录检索到的经验来源；
- control 与 transfer 的 task、task family、config、evaluator、initial、search protocol、model 和迭代预算必须一致；
- control 必须使用 `experience_mode=off`，transfer 必须使用 `experience_mode=cross-task`；
- transfer 自身隐藏集增益不得为负，且最终隐藏分数和隐藏增益均不得劣于 control；
- 是否取得严格正收益必须单独报告，打平只能判定“不劣”，不能描述为经验提升。

总 PSI 只有在两个子项都通过时才为真。

## 五、交付物

- `runs/`：OpenEvolve 原始谱系、日志、trace、checkpoint 和最佳程序；
- `results/`：每个程序的 lens 签名及每个变体的特征表；
- `analysis/`：执行后的聚类分析、表格和图；
- `state/`：追加式候选档案、经验、元策略实验、运行 manifest 和候选技能；
- `outputs/`：面向使用者的中文成果报告、HTML 报告和 notebook。

## 六、多模型与外部 Benchmark

### 本地 diagnostic

- model 与 profile 是两个独立轴；不得把“换模型”和“AgentProgram 演化”混成一个 treatment；
- 运行前冻结 task、grader contract SHA、grader implementation SHA、temperature、max tokens；
- 保存原始响应、usage、latency、逐项 rubric 和所有安全失败；
- pass@1 只用于发现问题，默认模型晋升至少要求多任务 pass³ 和 sealed 非退化；
- schema compliance、patch correctness、container tests、跨任务泛化分别报告。

### SWE-bench

- prediction 必须使用 `instance_id / model_name_or_path / model_patch` 三字段合同；
- 本地 preflight 拒绝不安全路径、额外字段和 tests 路径 mutation，但不能替代官方 harness；
- 正式 resolved-rate 只能来自官方 Docker evaluator，并保存 prediction、image/build/test logs；
- 缺 `swebench`、dataset、Docker daemon 或资源门时只能报告
  `adapter_ready_runtime_blocked`；
- schema smoke 和简单 patch 不能写成 SWE-bench Lite/Verified 分数。
