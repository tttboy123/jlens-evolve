# Handoff: evolve-jlens-cluster 自进化项目 → MinimaxCode

> 生成时间：2026-08-13。这是把"冻结模型 + Skills/机制让模型做到原本做不到的事"这条线完整移交给 MinimaxCode 的交接文档。
> 工作目录：`/Users/lune/Documents/Codex/2026-07-18/bang/work/evolve-jlens-cluster`

## 1. 已验证的产出（当前真实状态）

- **5 个 native 增益候选**，其中 2 个具备可独立重建的 baseline/taught
  paired evidence：
  - paired native 完整：sphinx-7757（`align_trailing_defaults`）、
    sphinx-10435（`normalize_inline_wrapper_boundaries`，baseline 为
    structural-invalid，历史 p1-r8）。
  - taught native resolved、但 catalog 尚缺 baseline native receipt：
    sphinx-9658（`initialize_generated_subclass_identity`）、sphinx-8638
    （`remove_variable_obj_role`）、sphinx-9698
    （`remove_property_index_parens`）。在补齐 baseline receipt 前，不把这 3
    个对外表述为完整的官方 failed→resolved paired gain。
- **关键结论**：弱学生（Qwen3.5-4B MLX）只有在"确定性 zero-arg operator"下可靠——renderer 替它改代码，它只需认出模式并选对 operator。给 `replace_constant` 这种要自己填 value 的操作会改错。
- **教学回归已修复**：无 card 匹配时 taught 退化为 baseline；边界重写由 issue oracle 确定性纠正；单 operator 卡路径同时规范化 operator + symbol（9658 现 3/3 稳定）。

## 2. 核心机制（MinimaxCode 应继续使用）

1. **确定性 operator**（`skill_evolution_loop/operator_rewrite.py`）：新增模式 = 零参 operator + structural matcher + renderer 确定性改。已有：align_trailing_defaults / normalize_inline_wrapper_boundaries / initialize_generated_subclass_identity / remove_property_index_parens / remove_variable_obj_role。
2. **PatternCard 路由**（`mlx_student.py`）：内容派生必需锚点谓词，弱词面匹配弃权，杜绝"default"类假阳性。
3. **定位信号**（`symbol_rewrite.py`）：`field_issue_hits`——类体声明了 issue 提到的字段（含单复数前缀匹配）则强提升；`qualified_symbol_candidates` top-N API 已提供但**尚未接进 prompt**（接入曾回归 9658，已回退，需更好的排序再接）。
4. **生成确定性**（`mlx_student.py` + `student_adapter.py`）：seed/temperature 配置 + `run_majority` 多采样多数投票。
5. **边界 oracle**（`operator_student.py`）：empty/absent 等边界重写由 issue 语义 oracle 确定性纠正。
6. **catalog 单写者 lease**（`evolution_catalog.py`）：`os.O_EXCL` writer lock，并发写 fail-fast。
7. **治理产出**（`skill_evolution_loop/governance.py`）：runtime identity / cost ledger / evolution overview。

## 3. 规则（必须遵守）

- **holdout 零泄漏**：不开 holdout、不用 holdout 内容训练/改 Skill；holdout 只用于完成门的安全评测。
- **Skill 永不自动激活**：全部 candidate/inactive；晋升 = baseline/taught + native failed→resolved 对照 + 零回归 + 零 evaluator failure。
- **core 冻结**：不动 evolution_runtime/bridge/controller/修正案/MANIFEST；实验轨（skill_evolution_loop）可改。
- **catalog 单写者**：root 唯一 append，subagent 只产提案。
- **成本记账**：每次 teacher（DeepSeek）与本地模型调用记 provider/model/tokens。
- **镜像直连拉取**：`docker pull swebench/sweb.eval.x86_64.<id 把 __ 换 _1776_>:latest`，再 `docker tag` 本地名；不要走 Clash。

## 4. 本地运行面

- 模型：`models/Qwen3.5-4B-mlx-4bit`（mlx_lm 0.31.3，温度 0 / seed 固定）。
- native harness：`.runtime/swebench-f7bbbb2`（rev f7bbbb2…）+ `.runtime/official-evaluator-venv`。
- 已有镜像：7757/9658/8638/9698/8595 等 feedback；10435 镜像已移除（历史证据仍在 catalog）。
- 测试：`818 passed, 1 skipped, 1 xfailed`；Ruff lint 与 format gate 通过
  （2026-08-13 本地 `make verify`）。

## 5. 待办（MinimaxCode 优先级）

1. **django/sympy/matplotlib/pytest/requests 等 feedback 任务的定位 + operator 编码**：每个 = 拉镜像 → 分析 golden（dev-only）→ 编码确定性 operator → 生成（多数投票）→ native。
2. **holdout 门（完成门核心）**：≥3 evaluator-valid holdout pairs。需 feedback 增益确认（已确认）→ 拉 holdout 镜像 → 跑 native safety。
3. **top-N 定位接回 prompt**：先改进排序（当前 field 信号只解决 8638 类），再接入避免 9658 回归。
4. **全 feedback 回归扫描**：多数投票批量，确认无 taught-worse-than-baseline。

## 6. 关键文件

- operator 实现：`skill_evolution_loop/operator_rewrite.py`、`operator_student.py`
- 定位：`skill_evolution_loop/symbol_rewrite.py`、`round1_realization.py`
- 生成确定性：`skill_evolution_loop/mlx_student.py`、`student_adapter.py`
- catalog：`skill_evolution_loop/evolution_catalog.py` + `artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/evolution-catalog/`
- native：`official_patch_evaluator.py`、`skill_evolution_loop/p1_native.py`
- 治理：`skill_evolution_loop/governance.py`、`scripts/measure_localization_recall.py`
