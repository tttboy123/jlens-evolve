# Skill 自进化闭环：基础工程设施

## 当前范围

P0 已从“仅基础设施”推进到可执行的旁路闭环。默认验证仍不运行真实 Qwen、
DeepSeek、AWS 或 SWE-bench；真实模型 transport 必须另行授权：

- `skill_evolution_loop.contracts`：Loop revision、失败证据、反馈包和父模型
  请求/响应的严格合约；
- `LoopRevisionRegistry`：与正式 `SkillRegistry` 分离的实验版本注册表，
  append-only、幂等、强制线性父子关系；
- `ParentCallLedger`：父模型调用前预留预算，完成或中止后成为终态；中止调用
  不退款、不自动重发；
- `ParentModelAdapter`：只接受显式授权和注入的 transport，验证响应后冻结完整
  回放证据；
- `StudentAdapter`：把学生输出收敛成严格的 `{file, search, replace,
  diagnostic}`，只允许唯一匹配、非测试目标；先 `git apply --check`，显式调用
  `apply` 才修改 checkout；
- `MlxStructuredGenerator`：惰性加载并缓存 Qwen MLX，注入当前 revision 的
  protocol / Skill / prompt template，输出交给 `StudentAdapter` 校验；
- `LoopEvaluator`：同时要求结构有效、feedback native 提升、hold-out 不回归，
  基础设施错误 fail-closed；
- `LoopDriver`：负责回合上限、no-progress 停止、最佳 revision 回滚、feedback
  组证据隔离、父模型再生成和 round 证据冻结；
- `doctor`：只读、零网络的环境诊断。

P0 不包含真实付费父模型调用、SWE-bench Round 1/2、正式 Skill 晋升或对冻结引擎
的修改。闭环 revision 仍在 loop-local registry 中；人工 review 后才能进入既有
promotion ladder。

## 环境

项目使用 `uv` 和根目录 `pyproject.toml`。OpenEvolve 继续使用相邻目录的本地
checkout：`../openevolve`。为避免清理或改写历史实验使用的 `.venv`，闭环使用
独立的 `.venv-loop`。首次创建环境：

```bash
make bootstrap
```

如果只开发闭环的纯 Python 基础设施，可以在独立环境中不安装 `local-mlx`：

```bash
UV_PROJECT_ENVIRONMENT=.venv-loop uv sync --locked
```

## 验证入口

```bash
make doctor
make test-infra
make verify
```

完整零网络冒烟（输出目录必须不存在，证据只增不改）：

```bash
python -m skill_evolution_loop offline-smoke \
  --out artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/p0-offline-smoke
```

当前冻结样例完成 2 回合：round-000 为 `reasoning-only`、三项通过率均 0；
父 transport 用本地确定性 fixture 返回新 revision；round-001 的结构、feedback
native、hold-out native 均为 1.0，最终 `converged`。`RESULT.json` 明确记录
`network_calls_performed=false`。

`doctor` 不读取 API key、不探测代理，也不发出网络请求。真实父模型 transport
必须在后续阶段单独实现，并继续受 authorization artifact 和 parent-call ledger
约束。

`verify` 运行全量 pytest、全项目 lint，并对闭环包、tests、tasks 执行格式检查。
根目录有 5 个历史实验脚本尚未采用当前 Ruff 格式，为保护冻结实验面，本阶段不对
它们做机械重排。

## 稳定公共接口

- `LoopRevision.create/from_dict/to_dict`
- `FeedbackPackage.create/from_dict/to_dict`
- `LoopRevisionRegistry.append/latest/read_revisions`
- `ParentCallLedger.reserve/complete/abort/records`
- `ParentModelAdapter.generate`
- `StudentTask.create` / `StudentAdapter.run/apply`
- `MlxStructuredGenerator`
- `LoopEvaluator.evaluate`
- `LoopDriver.run`
- `python -m skill_evolution_loop doctor`
- `python -m skill_evolution_loop offline-smoke`

测试只通过这些接口验证行为，不依赖内部实现。
