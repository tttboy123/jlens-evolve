# EVAL: backend-evolve-poc

## 目标

用一条本地命令稳定跑完最小 evolve 任务；不依赖 UI、网络、LLM 或在线
JLens。运行过程、收敛结论和 evidence 必须能够从 SQLite 重新读取。

## 能力评测

- [x] 搜索阶段只调用 public/dev evaluator。
- [x] sealed holdout 只在搜索停止后评测 baseline 与最终候选。
- [x] 每个 run 持久化输入哈希、协议哈希、状态和最终结果。
- [x] 每次 public/holdout 评测都持久化 partition、phase、候选哈希和分数。
- [x] 每次 mutation 都持久化 operator、父子哈希、增益、回归和 Gate 决定。
- [x] 明确区分“搜索空间收敛”与“任务解决”。
- [x] evidence 区分 JLens 观察、确定性 operator 干预和泛化审计。
- [x] 同一输入连续运行三次，得到相同 outcome fingerprint。
- [x] CLI 可以运行新任务，也可以从数据库检查已有 run。

## 当前 POC 成功门槛

- public/dev：从 `3/13` 提升到 `6/13`。
- 两个白名单 operator 均被接受，且不丢失父代已通过用例。
- 收敛原因：`operator_space_exhausted`。
- hidden：搜索完成后才运行；当前允许保持 `0/6`，但必须明确
  `task_solved=false`、`production_ready=false`。
- 稳定性：`pass^3`，三次独立 run 的 outcome fingerprint 完全一致。

## 回归评测

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check *.py tests tasks
.venv/bin/ruff format --check *.py tests tasks
```

实测记录见 `evals/backend-poc.log`。

