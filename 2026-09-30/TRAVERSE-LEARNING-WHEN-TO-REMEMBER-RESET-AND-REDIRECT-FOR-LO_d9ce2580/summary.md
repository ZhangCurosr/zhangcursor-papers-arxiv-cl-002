---
title: "TRAVERSE-LEARNING-WHEN-TO-REMEMBER-RESET-AND-REDIRECT-FOR-LO"
source: https://arxiv.org/pdf/2609.37082v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 16:53:11"
field: "长周期信息检索智能体"
keywords: ["long-horizon search", "agent context management", "reinforcement learning", "Seal Memory", "Rubric-Answer-Verify", "BrowseComp"]
innovations: ["三状态自主搜索工作流（Rubric-Answer-Verify）", "Seal Memory 主动上下文压缩机制", "最终段 RL 解决 Seal Collapse 训练失效问题"]
benchmarks: ["BrowseComp", "BrowseComp-ZH", "WideSearch", "xbench", "DeepSearchQA", "GAIA", "FinSearchComp", "ShoppingComp"]
---

# 论文速读：TRAVERSE-LEARNING-WHEN-TO-REMEMBER-RESET-AND-REDIRECT-FOR-LO

## 一句话总结
本文提出了 TRAVERSE，一个支持长周期 Web 搜索的自主智能体框架，通过 Rubric-Answer-Verify 三状态机与 Seal Memory 上下文管理机制，使智能体能够自主控制搜索流程与上下文压缩，并在解决"Seal Collapse"训练失效问题后，实现了 35B 模型在 BrowseComp 上 72.83 分的领先性能。

## 研究问题与动机
- **核心问题**：长周期信息检索智能体在探索过程中容易累积噪声或误导性上下文，导致早期错误持续放大且难以恢复。
- **现有方法不足**：
  - 上下文摘要与折叠方法仅关注维护有用的搜索状态，未考虑智能体应具备的自我验证能力。
  - 将轨迹级奖励广播到所有上下文的 RL 训练策略会导致信用分配模糊与轨迹重加权问题，引发 Seal Collapse。
  - 自动触发式上下文压缩（如达到阈值自动摘要）无法及时响应语义停滞状态，可能错过最佳压缩时机。

## 核心贡献（创新点）
- **三状态自主搜索工作流**：提出 Rubric-Answer-Verify 状态机，使智能体能够自主构建验证标准、收集证据、独立验证答案，与前人仅关注状态维护的方法形成本质区别。
- **Seal Memory 主动上下文管理**：引入由智能体自主触发的 Seal Memory 工具，使其能在任意时刻压缩上下文并携带有意义信息进入新段，而非依赖固定阈值自动压缩。
- **发现并解决 Seal Collapse**：识别出全段 RL 训练导致的"Seal Collapse"失效模式，提出仅优化最终段的策略，从算法层面解决信用分配模糊与轨迹重加权问题。
- ** masked SFT + 最终段 RL 训练管线**：结合掩码监督微调（过滤低质量交互轮次）与最终段相对优势 RL，在保证训练稳定性的同时实现搜索行为的涌现。
- **多基准领先性能**：35B 模型在 BrowseComp 达到 72.83 分，超过 QUEST-35B、AREX-Turbo 等同规模开源系统，并在泛化任务（金融搜索、商品搜索）上实现显著提升。

## 方法详解
- **Rubric-Answer-Verify 状态机**：
  - **Rubric 状态**：给定查询 $q$，智能体分解问题并生成 $R = \{r_i\}_{i=1}^m$ 验证标准，每条标准对应独立可验证的属性。
  - **Answer 状态**：携带 Rubric 进行证据搜索，智能体可自主决定是否调用 Seal Memory 压缩上下文；上下文重置后从 $c_{k+1} = q \oplus m_{k+1}$ 继续。
  - **Verify 状态**：以验证者角色评估候选答案 $y$ 是否满足 $R$，输出 $d \in \{PASS, REVISE\_ANSWER\}$；若需修改则携带反馈回到 Answer 状态。
- **Seal Memory 工具设计**：
  - 智能体按结构化模板总结需保留信息：$m_{k+1} \sim \pi_\theta(\cdot | q, h_k, m_k)$，包括 verified/conflicting/partial 事实、已探索路径、死胡同、下一步计划等。
  - 每个工具调用返回当前 token/turn 预算使用量，智能体可显式跟踪剩余预算。
  - Read Memory 工具用于按需检索最近一次 Seal 的详细信息。
- **Masked SFT 训练**：
  - 对教师轨迹中的不良轮次（过早 Seal、答案已找到仍 Seal 等）赋予掩码 $\mathcal{M}_i = 0$，避免模仿错误行为，但保留恢复能力的学习。
  - 损失函数为 $\mathcal{L}_{SFT} = -\frac{1}{\sum \mathcal{M}_i T_i} \sum \mathcal{M}_i \sum \log \pi_\theta(x_{i,j}|x_{<i}, x_{i,<j})$。
- **最终段 RL 策略（解决 Seal Collapse）**：
  - 仅对每段轨迹的最终段 $\tau_i^{(K_i)}$ 计算组内归一化优势 $\hat{A}_i$ 并优化：$\mathcal{L}_{RL}^{last} = \frac{1}{N}\sum_i \mathcal{L}(\pi_\theta, \tau_i^{(K_i)}, \hat{A}_i)$。
  - 隐含学习机制：每段末尾若需继续搜索则必须 Seal，否则因上下文溢出成为最终段并被惩罚，从而无需显式奖励 Seal 调用。
  - 采用 GSPO 稳定 MoE 训练，并施加 turn/ token 预算惩罚与并行工具调用奖励。
- **上下文预算随机化**：$B_\mathcal{G} \sim Uniform(\mathcal{B})$，使智能体学会根据实时进度与剩余预算决定 Seal 时机，而非依赖固定阈值。

## 实验与结果
- **数据集与基准**：
  - 主评测：BrowseComp（深度信息检索）、BrowseComp-ZH（中文）、xbench-DeepSearch（多步证据合成）、WideSearch（广度检索）、DeepSearchQA。
  - 泛化评测：GAIA（通用 AI 助手）、FinSearchComp（金融搜索）、ShoppingComp（商品搜索）。
- **基线对比**：
  - 闭源：GPT-5.5、Claude Opus 5、Gemini 3.1 Pro。
  - 开源：QUEST-35B、AREX-Turbo、FORT-Searcher、DeepSeek-V4-Pro、Kimi K3 等。
- **主要结果**：
  - **BrowseComp**：Traverse-35B 达到 **72.83** 分，超过 QUEST-35B（64.6）、AREX-Turbo（70.7）、FORT-Searcher（72.2），仅次于 GPT-5.5（84.4）和 DeepSeek-V4-Pro（83.4）。
  - **WideSearch**：Item-F1 **72.1**，显著优于多数基线。
  - **DeepSearchQA**：Macro-F1 **82.8**，领先 AREX-Turbo（78.5）。
  - **泛化提升**：GAIA 提升 **+18.45pp**（61.81→80.26）、FinSearchComp 提升 **+19.70pp**（38.36→58.06）、ShoppingComp SoP 提升 **+0.1331**（0.2245→0.3576）。
- **消融结论**：
  - 最终段 RL 优于全段 RL（BC185 Avg@3：65.2 vs 51.7）。
  - Seal Memory 主动压缩比 Auto Compaction 提升 Avg@3 **10.28pp**（65.23 vs 54.95），首次 Seal 发生在 67 轮/67.5K tokens，远早于自动压缩的 217 轮/231.4K tokens。
  - RL 在 SFT 基础上进一步带来 BC185 **+4.8 分**、WideSearch Row F1 **+6.0 分**。

## 相关工作脉络
- **Web Search Agents**：WebGPT、ReAct 奠定基础，后续 IterResearch、WebSailor、WebThinker 等扩展长周期搜索；本文在三状态工作流与主动上下文管理上区别于现有系统。
- **RL for Long-Horizon Search**：Search-R1、R1-Searcher 探索基于环境反馈的训练；Resum、Memory as Action 等方法聚焦上下文摘要/折叠；本文通过 Seal Memory + 最终段 RL 避免轨迹碎片化带来的信用分配问题。
- **Context Management**：Auto Compaction（固定阈值触发）与本文 Active Seal 形成对比，后者能识别语义停滞并提前重置搜索方向。
- **Verification Mechanisms**：DeepSeekMath-V2 启发 Rubric-Verify 设计；Wan et al. (2026) 在推理时通过 Rubric 引导验证，本文将其嵌入训练闭环。
- **Benchmark 定位**：在 BrowseComp、WideSearch 等基准上与 QUEST、AREX、FORT 直接竞争，展现开源 35B 模型的竞争力。

## 局限性与未来方向
- **推理时缩放潜力**：当前结果主要来自训练时行为塑造，未系统探索推理时计算扩展（如多轮验证、Best-of-N 采样）对性能的提升空间。
- **Seal 内容质量依赖**：结构化记忆模板的质量直接影响后续搜索方向，复杂任务中可能遗漏关键上下文或引入偏差。
- **单步验证限制**：Verify 状态目前仅输出 PASS/REVISE，未支持多候选答案排序或置信度量化。
- **工具调用开销**：长周期搜索涉及大量 search/link_summary 调用，实际部署的延迟与成本尚未充分评估。
- **基准覆盖局限**：主要在英文检索场景验证，中文及其他语言能力的提升仍有探索空间（BrowseComp-ZH 得分略低于 BrowseComp）。

## 研究启发与可借鉴点
- **最终段 RL 策略可迁移**：任何引入上下文重置的工具型智能体（如代码 agent 的文件操作、multi-agent 协作）均可借鉴"仅优化最终段"避免信用分配模糊的思路。
- **主动 vs 被动上下文管理**：从"达到阈值触发"转向"智能体自主决策"是一个强信号，可启发其他需要长上下文的 agent 设计。
- **Rubric-Driven 验证范式**：先分解验证标准再搜索、后验证的工作流具有通用性，可扩展到数学证明、代码生成、文献综述等任务。
- **Masked SFT 过滤策略**：对教师轨迹中部分轮次进行掩码而非整条丢弃，兼顾行为学习与恢复能力训练，值得在指令微调中推广。
- **上下文预算随机化**：训练时随机化 token 预算可迫使模型学会动态决策，避免对固定阈值的依赖，适用于任何有长度约束的 agent 场景。

## 关键术语表
- **Rubric-Answer-Verify 状态机**：三阶段搜索工作流，智能体先构建验证标准、再搜索答案、最后独立验证并决定终止或继续。
- **Seal Memory**：由智能体自主触发的上下文压缩工具，将当前探索状态、已验证事实、死胡同与下一步计划封装为结构化记忆。
- **Seal Collapse**：全段 RL 训练导致的失效模式，表现为 Seal 调用逐渐减少直至为零，模型无法学会何时压缩上下文。
- **最终段 RL（Final-Segment RL）**：仅对轨迹中最后一次上下文压缩后的段计算优势并更新参数，避免早期探索噪声的信用分配问题。
- **Masked SFT**：对教师轨迹中识别出的不良轮次施加掩码（$\mathcal{M}_i=0$），阻止模型模仿错误行为但保留恢复能力学习。
- **GSPO（Group Sequence Policy Optimization）**：聚合 token 级策略比为序列级重要性比率，用于稳定 MoE 架构的 RL 训练。
- **BrowseComp-Lite-185**：从 BrowseComp 筛选的 185 题子集，用于高效训练过程中的 checkpoint 评估，误差控制在 ±3-5pp。

## 可复现要素
- **数据集**：训练数据为论文合成的知识图谱问答数据（遵循 WebShaper 流程），公开于 HuggingFace：`ByteDance-BandAI/Traverse-AutoGen`；Benchmark 为 BrowseComp、BrowseComp-ZH、WideSearch、xbench、DeepSearchQA、GAIA、FinSearchComp、ShoppingComp。
- **代码**：开源地址 `github.com/ByteDance-BandAI/Traverse`；权重与数据集公开于 HuggingFace。
- **关键超参**：
  - 基座模型：Qwen3.5-35B-A3B（MoE）。
  - 最大 agent 轮次：512；上下文长度：256K；采样温度：1.0。
  - RL 训练：96 GPU（32 策略优化 + 64 异步 rollout）。
  - 上下文预算随机化：$B_\mathcal{G} \sim Uniform(\mathcal{B})$。
  - Turn 惩罚触发阈值：$\rho_{turn} = 5\%$。
