---
title: "VACE-Validation-Gated-Alternating-Co-Evolution-of-Agent-Mode"
source: https://arxiv.org/pdf/2609.37105v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:58:30"
---

# 论文速读：VACE-Validation-Gated-Alternating-Co-Evolution-of-Agent-Mode

## 一句话总结
本文提出 VACE 框架，通过交替执行智能体强化学习与轨迹驱动的 harness 精炼，并在每次修订后引入基于更新后模型权重的验证门控，实现模型权重与执行系统的协同演化；在 Qwen3.5-9B 上于 OfficeQA 和 Automation-Bench 均显著超越仅训模型、仅调 harness 及无门控交替基线。

## 研究问题与动机
1. 智能体性能同时依赖模型权重（决定推理与工具调用能力）和 harness（包含指令、技能、执行规则），两者高度耦合：权重更新会改变 harness 的使用方式，而 harness 调整又会改变后续 RL 训练所依赖的轨迹分布。
2. 现有工作多单向优化：agentic RL 仅更新权重而固定 harness，或仅调优 harness 而冻结模型，无法捕获两者的双向反馈与协同增益。
3. 从轨迹中提取的 harness 修改建议并非总能提升任务表现（例如强制增加多轮检索可能消耗交互预算），盲目采纳会拖累后续 RL 阶段，亟需一种与当前模型状态对齐的评估机制。
4. 缺乏一套将 agentic RL、轨迹诊断、harness 提议与验证门控串联的闭环流程，难以保证交替演化过程中的稳定性与可复现性。

## 核心贡献（创新点）
1. 提出 VACE 联合优化框架，将 agentic RL 与轨迹驱动的 harness 精炼耦合为交替闭环，实现模型权重 $W$ 与执行配置 $H$ 的持续协同演化；与单组件优化工作相比，首次在同一训练循环内显式建模两者的双向依赖关系。
2. 引入 checkpoint-specific 验证门控机制，在每次 harness 提议后固定更新后的模型权重，在同一验证集上严格比较 incumbent 与 candidate，仅当验证分数严格提升时才采纳新 harness；区别于无门控交替方法（如 WHALE），该机制有效阻断了有害修订向后续 RL 阶段的传播。
3. 在 OfficeQA 与 Automation-Bench 上系统性对比了联合演化、单组件优化及不同调度策略（SIA-style、WHALE-style），量化揭示了验证门控的必要性（44 次提议中 17 次因降低验证性能被拒绝），为 agent self-evolution 提供了可复现的基线与方法学参考。

## 方法详解
- **联合优化目标**：定义 agent 状态为 $(W, H)$，目标为最大化期望任务回报 $J(W, H) = \mathbb{E}_{x \sim \mathcal{P}} \mathbb{E}_{\tau \sim p_{W,H}(\cdot|x)}[r(x,\tau)]$。权重决定模型行为，harness 决定执行流程，两者通过轨迹生成过程 $p_{W,H}$ 耦合。
- **交替更新循环（每轮 $t$）**：
  1. **模型训练阶段**：固定当前 harness $H_t$，使用 agentic RL 优化器 $\mathcal{M}$（基于 GRPO + DAPO 风格过采样/过滤，仅使用最终结果奖励）在训练集 $\mathcal{D}_{\text{train}}$ 上更新权重，同时收集轨迹批次 $\mathcal{T}_t$，得到 $W_{t+1}$。
  2. **Harness 提议阶段**：利用轨迹诊断（失败案例、重复错误模式）驱动 harness 优化器 $S$（基于 HarnessEvolve 思路），针对技能与执行指引提出单一候选修订 $H'_t = S(H_t, \mathcal{T}_t)$，无需额外 roll-out。
  3. **验证门控阶段**：固定使用更新后的权重 $W_{t+1}$，在独立验证集 $\mathcal{V}$ 上分别评估 incumbent $H_t$ 与 candidate $H'_t$，计算 $\Delta_t^H = \widehat{R}_{\mathcal{V}}(W_{t+1}, H'_t) - \widehat{R}_{\mathcal{V}}(W_{t+1}, H_t)$。若 $\Delta_t^H > 0$ 则采纳 $H_{t+1}=H'_t$，否则保留 $H_{t+1}=H_t$（平局保留 incumbent）。
- **关键设计细节**：严格改进门控确保每次 harness 变更都与当前模型状态兼容；权重更新不受门控影响直接保留至下一轮；测试集仅用于最终性能报告，不参与优化回路。

## 实验与结果
- **数据集与设置**：基础模型 Qwen3.5-9B；OfficeQA（文档 grounding 推理，84 训练/53 验证/109 测试，二值正确性奖励）；Automation-Bench（职场工作流工具调用，209 任务子集按 Table 1 划分为 80/58/71，密集部分得分奖励）。使用 Uni-Agent 框架连接执行、RL 与 harness 精炼，所有方法共享相同基础模型、初始 harness 与评估器。
- **评估基线**：Static agent、Harness-only、RL-only、SIA-style（先优化 harness 再固定跑 RL）、WHALE-style（交替但无门控）。
- **主要结果**：
  - **OfficeQA**：VACE 测试准确率 **45.26%**，超越 RL-only（38.83%）6.43 pp，超越 WHALE-style（40.67%）4.59 pp，超越 Static 13.15 pp。
  - **Automation-Bench**：VACE 平均部分得分 **75.19%**，超越 RL-only（66.10%）9.09 pp，超越 WHALE-style（68.25%）6.95 pp，超越 SIA-style（65.35%）9.85 pp。
- **门控有效性分析**：共 44 次 harness 对比，25 次接受、17 次拒绝、2 次平局（Table 3）。同 checkpoint 下 OfficeQA 第 30 轮提议使验证分从 60.37% 降至 49.06%，Automation-Bench 第 15 轮从 70.93% 降至 64.50%，均被门控拦截，证明提议与采纳需分离决策。
- **演化动态**：保留 harness 的验证分在 OfficeQA 上从 30.28% 升至 47.17%（+16.89 pp），Automation-Bench 从 53.
