# C 方案：Span 候选本地化修复计划（与 typed-action 契约对齐）

> 状态：**已实施**（见 §0 实施记录）
> 依据：实验 A（Catch2-2288 全链路验证，2026-08-16）
> 目标代码：`skill_evolution_loop/span_student.py`（cluster，无 git，改动前必须手动备份）

---

## 0. 实施记录（2026-08-16）

> 状态：**C1 与 C2 均已实施并通过验证**；C3 未做（后续演进）。

- **C1（已实施）**：`_frozen_causal_candidates` baseline 路径按 `source_round < 39` 门控扩展到 `min(max_candidates, 4*len(targets), 8)`；`_candidate_control_flow_roles` 对 `{` 结尾定义行加 `lexical-boundary`。验证：真实 Catch2-2288 @ source_round=0 → 8 候选、`operator()` 第 3 位；span_student 53 测试全绿。
- **C2（已实施）**：
  1. 移除 `source_round` 门控 → 所有 schema 统一有界多候选（typed 流程的 prompt 也受益）；
  2. 修复 `fixed_typed_state_actions` 的 transient 推导：从依赖 `fixed_causal_state_contexts`（跳过候选行、随候选数非单调变化：8 候选时 causal context 从 2 变 0）改为**任务级全文件扫描**（稳定、与候选数无关），使 `guard-neutral-or-transient-default` 动作在扩展候选集下稳定存在。
  验证：transient-guard 测试在统一 8 候选下通过（无需改测试）；Catch2-2288 @ source_round=0/50 均 8 候选含 `operator()`；cluster 全量 **871 passed**；ruff 干净。
- **备份**：`/tmp/span_student.py.bak.c1.*`、`/tmp/span_student.py.bak.c2.*`。
- **对运行中 v6 无影响**：进程内已加载旧代码；C1+C2 对下一次进程/重启生效。

## 1. 背景与问题（含证据）

### 1.1 症状
多语言 flash 任务在 baseline 阶段产出**空 patch**（`patch_sha256 = e3b0c442…`，即空串 SHA）-> native 无可评 -> `infra_failure` -> `baseline campaign did not complete`，整轮中止。

### 1.2 根因链（实验 A 证实）
1. baseline（未锚定）路径 `_frozen_causal_candidates` 执行：
   ```python
   if editable.allowed_targets == task.allowed_targets:
       return fixed_exact_span_candidates(editable, max_candidates=max_candidates)[:1]
   ```
   只给 student **1 个候选 span**。
2. `fixed_exact_span_candidates` 跨文件按 score 排序，**正确位置可能排不到第 1**。Catch2-2288 实测：`Approx operator()( T const& value ) {`（catch_approx.h:36，score 24963）排第 **3**，第 1 是测试文件 `Approx.tests.cpp:108` 的 `const double dZero = 0;`（score 46287）。
3. student 只能编辑 ONLY EDITABLE 列表里的候选 -> 正确位置不可达 -> **正确弃权**（unresolved）-> 空 patch。
4. 附带不对称：baseline 只见 1 个候选，pinned/taught 路径最多 8 个（`emitted_limit = min(max_candidates, 4*len(targets), 8)`）——**baseline 被无意识削弱**。

### 1.3 边界（诚实声明）
实验 A 进一步证明：**即使把正确候选（#3）放进列表，Qwen3.5-4B 仍弃权**（复读同一句错误诊断，未读角色标签），recipe schema 下则 `plan-too-large`。因此 **C 修复的是"候选被卡死"这一真实缺陷与臂对称，不承诺解锁 flash**；flash 解锁仍需更强 student（A2：Qwen2.5-Coder-7B）或 C3 分层 localizer。

---

## 2. 修复目标

- 消除 baseline 因单一错误候选导致的空 patch（候选本地化缺陷）。
- 恢复 baseline / taught 两臂候选可见性对称。
- 不与 typed-action 推导/校验契约冲突（现有测试全绿）。
- 保持确定性、可审计、可回滚。

---

## 3. 方案设计

### 3.1 C1（主修复）：baseline 候选有界扩展，按 schema 门控

**文件**：`skill_evolution_loop/span_student.py` -> `_frozen_causal_candidates`

**现状**
```python
editable = _editable_span_task(task, revision)
if editable.allowed_targets == task.allowed_targets:
    return fixed_exact_span_candidates(editable, max_candidates=max_candidates)[:1]
```

**改为**
```python
editable = _editable_span_task(task, revision)
if editable.allowed_targets == task.allowed_targets:
    candidates = fixed_exact_span_candidates(
        editable, max_candidates=max_candidates
    )
    if revision.source_round < 39:
        # edits/recipe schema（autonomous baseline 用 source_round=0）：
        # 暴露有界、role-diverse 候选集，与 pinned/taught 路径同口径（对称）。
        emitted = min(max_candidates, 4 * len(editable.allowed_targets), 8)
        return candidates[:emitted]
    # typed-action schema（source_round >= 39）保持既有契约，避免改变
    # typed_catalog 推导/校验语义。
    return candidates[:1]
```

**门控依据（已核实）**：
- autonomous baseline 的 `LoopRevision.source_round = 0`（`worktree/src/evolve/runtime/qwen_transport.py:505`）-> 走 edits schema -> 扩展生效。
- 被实验 A 破坏的测试 `test_typed_state_actions_derive_and_materialize_transient_guard` 用 `source_round=50`（typed-action 路径）-> 保持 `[:1]` -> 不受影响。

**为什么这个门控是对的**：
- `_frozen_causal_candidates` 有两个消费方：① 生成 prompt 候选（给 student 看/改）；② `SpanPlanAdapter.run` 在 `source_round>=39` 时用它推导 `typed_catalog` 并做语义校验。简单扩候选会同时改 ② 的语义（实测把 `align::left` 引入 catalog 后，transient-guard 校验失败）。按 `source_round` 门控，让 ② 的契约完全不变，只在 ①（edits 路径）修复。

### 3.2 附带小修：定义行角色分类

**文件**：`skill_evolution_loop/span_student.py` -> `_candidate_control_flow_roles`

```python
stripped = before.lstrip()
roles: list[str] = []
if stripped.rstrip().endswith("{"):
    # 以 { 结尾 = 函数/类型定义体起始，是 lexical-boundary 编辑面，
    # 不应只被标成 call-site。例：`Approx operator()( T const& value ) {`
    roles.append("lexical-boundary")
```

效果：`operator()( T const& value ) {` 从 `('call-site-boundary',)` -> `('lexical-boundary','call-site-boundary')`；普通调用行、赋值行不受影响（已实测）。

### 3.3 演进路径（本次不做，作为后续）

| 方案 | 内容 | 工程量 | 何时做 |
|---|---|---|---|
| **C2** | 把"prompt 候选展示"与"typed-action 推导/校验"解耦：prompt 恒给有界 role-diverse 集；`typed_catalog` 用稳定子集。彻底修 typed-schema 的候选可见性 | 中 | 需要 typed 流程也受益时 |
| **C3** | 分层 localizer（file->symbol->excerpt、top-K 文件、按文件多样候选），根治"正确位置在错误文件/排名靠后" | 大 | 决定投入多语言时，对齐 ROUND1 调研 |

---

## 4. 改动清单

| 文件 | 改动 | 类型 |
|---|---|---|
| `skill_evolution_loop/span_student.py` | `_frozen_causal_candidates` 扩展 + 门控 | 主修复 |
| `skill_evolution_loop/span_student.py` | `_candidate_control_flow_roles` 定义行角色 | 附带 |
| `tests/test_skill_evolution_span_student.py` | 新增 2 个测试（见 §5） | 测试 |

**备份**（cluster 无 git）：
```bash
cp skill_evolution_loop/span_student.py /tmp/span_student.py.bak.$(date +%s)
```

---

## 5. 测试计划

新增断言：
1. **baseline 有界多候选**：构造 Catch2-2288 任务（`source_round=0`，无 shared-diagnosis marker），断言 `_frozen_causal_candidates` 返回 `min(16, 4*8, 8)=8` 个候选，且 `Approx operator()( T const& value ) {` 在列。
2. **角色分类**：`_candidate_control_flow_roles("Approx operator()( T const& value ) {")` 含 `"lexical-boundary"`。

回归：
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_skill_evolution_span_student.py -q      # 51 个，须全绿（含 transient-guard）
PYTHONPATH=. .venv/bin/python -m pytest tests/test_skill_evolution_span_rewrite.py \
  tests/test_skill_evolution_operator_student.py -q                                        # 相邻模块
```

候选级验证（无需模型）：
```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
# 构建 Catch2-2288 StudentTask（source_uri / instruction / allowed_targets 来自 FEEDBACK-TASK-POOL）
# source_round=0 的 LoopRevision
# 断言 _frozen_causal_candidates(...) 返回 8 个且含 operator() #3
PY
```

---

## 6. 验收标准

- [ ] `_frozen_causal_candidates`（source_round=0）对 Catch2-2288 返回 8 候选，`operator()` 定义在第 3 位。
- [ ] `_candidate_control_flow_roles` 对 `{` 结尾定义行返回 `lexical-boundary`。
- [ ] `tests/test_skill_evolution_span_student.py` 51 个全绿（含 transient-guard）。
- [ ] 相邻 span/operator student 测试全绿。
- [ ] 改动可回滚（备份在 /tmp）。
- [ ] 运行中的 v6 campaign 不受影响（进程内已加载旧代码；改动在下次进程/重启生效）。

---

## 7. 风险与边界

- **不承诺解锁 flash**：实验 A 已证 4B 对 C++/模板是能力瓶颈（复读弃权、recipe 超长）。C1 消除的是"候选被卡死 + 臂不对称"，对 Python/PHP 任务（4B 能编辑）价值最大。
- **typed-schema 的 prompt 候选仍是 1 个**：`source_round>=39` 流程的候选可见性缺陷保留，由 C2 解决。
- **cluster 无 git**：改动依赖 /tmp 手动备份；若需版本化，应把该目录纳入 git 或单独提交。
- **确定性**：扩展与门控都是纯函数，随 CompileSpec/候选 hash 绑定，replay 一致。

---

## 8. 执行步骤（实施时照此）

1. `cp span_student.py /tmp/span_student.py.bak.$(date +%s)`（备份）。
2. 应用 §3.1 与 §3.2 两个 diff。
3. 跑 §5 回归 + 新增测试，直到全绿。
4. 跑候选级验证脚本，确认 Catch2-2288 在 source_round=0 下 8 候选含 operator()。
5. 记录结果；如要外发，按 cluster 的实际版本管理方式提交（当前无 git，建议先 `git init` 或并入工作区仓库）。
6. （可选）观察下一次新 campaign 的 baseline 空 patch 率是否下降。
