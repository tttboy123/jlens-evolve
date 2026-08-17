# 后台 Evolve POC

## 目标与边界

当前 POC 只证明一件事：在不依赖 UI、网络、LLM 和在线 JLens 的情况下，后台可以
稳定完成一次最小 evolve 搜索，并留下完整、可查询的过程证据。

它不是模型训练系统，也不把 JLens 当成正确性信号。JLens 产物只用于说明为什么选择
“小步、结构化 mutation”；候选是否有效完全由固定 evaluator 决定。

## 一条命令运行

```bash
cd /Users/lune/Documents/Codex/2026-07-18/bang/work/evolve-jlens-cluster
.venv/bin/python backend_poc.py run
```

指定独立 run、数据库和产物目录：

```bash
.venv/bin/python backend_poc.py run \
  --run-id my-first-run \
  --db runs/backend-poc/evolve.sqlite3 \
  --output runs/backend-poc
```

检查某次 run：

```bash
.venv/bin/python backend_poc.py inspect \
  --db runs/backend-poc/evolve.sqlite3 \
  --run-id my-first-run
```

不传 `--run-id` 时，`inspect` 返回数据库中最近开始的 run。

## 后台流程

```text
读取 initial_program.py 与 JLens 历史观察
  -> 冻结 evaluator/operator/protocol 哈希
  -> public baseline
  -> 找到第一个有白名单 operator 的公开失败
  -> 生成候选并只用 public evaluator Gate
  -> 重复，直到公开目标全通过、operator 空间耗尽或预算耗尽
  -> 搜索停止
  -> sealed holdout 分别审计 baseline 与最终候选
  -> 写 SQLite、candidate.py、result.json、evidence.json、summary.md
```

## SQLite 数据

数据库只有三张核心表：

| 表 | 内容 |
|---|---|
| `runs` | 输入/协议哈希、生命周期状态、收敛原因、最终结果和 evidence |
| `evaluations` | 每次评测的顺序、阶段、partition、候选哈希与完整指标 |
| `iterations` | 每步 operator、父子哈希、增益、回归和 Gate 决定 |

常用查询：

```sql
SELECT run_id, status, convergence_reason, decision, outcome_fingerprint
FROM runs
ORDER BY started_at DESC;

SELECT ordinal, phase, partition, candidate_role, passed_cases, total_cases
FROM evaluations
WHERE run_id = 'my-first-run'
ORDER BY ordinal;

SELECT iteration, public_failure, operator_id, decision, public_gain
FROM iterations
WHERE run_id = 'my-first-run'
ORDER BY iteration;
```

正常最小 run 的 partition 顺序必须是：

```text
public, public, public, holdout, holdout
```

前 3 次是搜索阶段的 baseline 与两个候选；后 2 次是在搜索已经停止后进行的 baseline/
final sealed audit。Holdout 结果不会触发新的 mutation。

## 收敛结果如何解释

POC 明确区分：

- `converged=true`：当前有限搜索协议已经停止。
- `reason=operator_space_exhausted`：剩余失败没有白名单 operator。
- `task_solved=false`：整个任务并没有解决。
- `production_ready=false`：候选不能晋升为生产版本。

因此“收敛”不等于“答案正确”或“泛化完成”。2026-08-03 的 `pass^3` 实测停止点是 public
`3/13 -> 6/13`、holdout `0/6 -> 0/6`：两个 operator 有确定性局部收益，但现有
operator 覆盖不足，且没有隐藏集泛化收益。

## Evidence 的因果等级

`evidence.json` 把证据分成三层：

1. `observational`：历史 JLens/trace 发现 65% 重复迁移，仅用于提出实验方向。
2. `deterministic_parent_child_intervention`：固定父候选、应用一个 operator、运行同一
   public evaluator，记录新增/丢失用例；这是当前最强的局部干预证据。
3. `post_search_audit`：搜索结束后运行 holdout，判断是否出现泛化收益。

当前主要原因会记录为 `operator_coverage_exhausted`：系统不是因为 evaluator 或数据库
失败而停下，而是因为仅有两个白名单 operator，无法处理 identity、aggregation、
rounding、sorting 等剩余失败。

## 稳定性验收

```bash
.venv/bin/python -m pytest -q tests/test_backend_poc.py
.venv/bin/python -m pytest -q
.venv/bin/ruff check *.py tests tasks
.venv/bin/ruff format --check *.py tests tasks
```

测试会独立运行三次并比较 `outcome_fingerprint`。指纹排除 run ID、时间和输出路径，
只覆盖协议、候选哈希、逐步 Gate、收敛与最终分数；三次完全一致才算 `pass^3`。

## 下一步最小扩展

在这条后台链路稳定后，按以下顺序扩展即可：

1. 把确定性 task-program mutation 保留为 Kernel smoke，不将其描述为 Agent RSI。
2. 定义最小 `AgentProgram`，只开放 Prompt、Skill selection 和 retry policy mutation。
3. 先接可录制重放的 `ReplaySupervisor`；LLM Supervisor 仍然只能返回候选。
4. 做 trace-only、logit-lens、JLens matched A/B，验证 JLens 是否有独立增益。
5. 只有跨任务验证通过的经验，才生成项目内 PSI Skill candidate。

产品版本与完整顺序见 `ROADMAP.zh-CN.md`；模型训练不属于当前产品路线。
