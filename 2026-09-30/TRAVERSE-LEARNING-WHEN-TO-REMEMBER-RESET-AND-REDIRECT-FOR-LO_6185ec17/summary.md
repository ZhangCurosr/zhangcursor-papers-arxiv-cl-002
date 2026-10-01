---
title: "TRAVERSE-LEARNING-WHEN-TO-REMEMBER-RESET-AND-REDIRECT-FOR-LO"
source: https://arxiv.org/pdf/2609.37082v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 16:53:26"
field: "多智能体搜索与长上下文管理"
keywords: ["Long-Horizon Search", "Agent Memory", "Reinforcement Learning", "Context Management", "Self-Verification", "BrowseComp"]
innovations: ["提出Rubric-Answer-Verify状态机与自主Seal Memory工具实现长程搜索的自管理", "发现并解决Seal Collapse问题，设计仅优化最终分段的RL策略稳定记忆工具学习", "在BrowseComp等基准上取得SOTA级性能，验证自主上下文压缩优于固定阈值自动压缩"]
benchmarks: ["BrowseComp", "BrowseComp-ZH", "DeepSearchQA", "WideSearch", "xbench", "GAIA", "FinSearchComp", "ShoppingComp"]
---

# 论文速读：TRAVERSE: LEARNING WHEN TO REMEMBER, RESET, AND REDIRECT FOR LONG-HORIZON WEB SEARCH

## 一句话总结
本文提出一种面向长程网页搜索的自主智能体框架 TRAVERSE，通过 Rubric-Answer-Verify 状态机与智能体自主触发的 Seal Memory 上下文压缩工具，使模型能够动态管理搜索历史并独立验证答案；同时针对强化学习中常见的“Seal Collapse”现象，设计仅优化最终分段的 RL 策略，在 BrowseComp 等基准上取得 SOTA 级性能。

## 研究问题与动机
- 长程信息检索智能体在多次搜索后会不断累积噪声、无关或误导性上下文，导致早期错误持续放大并陷入难以恢复的退化状态。
- 现有上下文管理方法（如周期性摘要、上下文折叠）主要关注维持可用状态，却忽视了智能体自身判断“当前答案是否真正正确”的自检能力。
- 使用环境反馈（RL）训练记忆工具时，若将轨迹级奖励广播至所有上下文分段，会产生严重的信用分配模糊与分段诱导的轨迹重加权，引发训练不稳定与“Seal Collapse”。
- 如何在保持长程搜索自主性的同时，稳定教会智能体“何时压缩、压缩什么、何时停止搜索”，是该方法旨在解决的核心训练与架构难题。

## 核心贡献（创新点）
1. **自主状态机与 Agent-Triggered 记忆工具**：设计 Rubric-Answer-Verify 三态工作流，结合智能体按需调用的 Seal Memory / Read Memory 工具，使压缩时机与信息取舍由模型自主决策，而非依赖固定阈值规则。
2. **揭示并解决 Seal Collapse 问题**：首次系统分析轨迹级优势广播导致的信用分配模糊与分段重加权现象，提出仅对最终分段执行 RL 更新的简单高效策略，稳定记忆工具学习且训练开销不随重置次数增长。
3. **多基准 SOTA 与强泛化验证**：Traverse-35B 在 BrowseComp 达到 72.83，在 DeepSearchQA 与 WideSearch 上领跑同规模开源系统；在金融、电商及多语言任务上均显著超越基座模型，证明自主压缩与自检机制的工程有效性。

## 方法详解
- **三态工作流（Rubric → Answer → Verify）**：智能体首先根据问题 $q$ 生成结构化验证清单 $R=\{r_i\}$；在 Answer 状态下调用搜索/浏览工具收集证据，并依据剩余 token/turn 预算决定是否触发 Seal Memory；在 Verify 状态下以验证者身份逐条核对 $y$ 是否满足 $R$，输出 $\text{PASS}$ 或 $\text{REVISE ANSWER}$，后者携带具体未满足项反馈回到 Answer 状态继续搜索。
- **Seal Memory 自主上下文管理**：智能体调用工具时按结构化模板生成记忆 $m_{k+1} \sim \pi_\theta(\cdot \mid q, h_k, m_k)$，包含已验证事实、冲突项、死胡同路径与下一步计划；随后重置上下文 $c_{k+1} = q \oplus m_{k+1}$ 继续搜索，并提供 Read Memory 按需回溯细节。
- **Masked SFT**：针对教师轨迹中的格式错误、过早 Seal 等坏行为，引入回合级掩码 $\mathcal{M}_i \in \{0,1\}$，仅对有效 Assistant 回合计算 next-token loss，避免模仿缺陷模式同时保留恢复行为信号。
- **Final-Segment-Only RL**：轨迹经 Seal 操作被切分为 $\{\tau_i^{(1)},\dots,\tau_i^{(K_i)}\}$，传统做法将组归一化优势 $\hat{A}_i$ 广播至所有分段；本文证明此方式会放大优化噪声并隐式重加权样本，因此 RL 损失仅作用于最后一段：$\mathcal{L}_{\mathrm{RL}}^{\mathrm{last}} = \frac{1}{N}\sum_i \mathcal{L}(\pi_\theta, \tau_i^{(K_i)}, \hat{A}_i)$。采用 GSPO 稳定 MoE 训练，并叠加 turn/context 预算惩罚、格式惩罚与并行工具效率奖励。

## 实验与结果
- **数据集与基线**：在 BrowseComp、BrowseComp-ZH、xbench-DeepSearch、DeepSearchQA、WideSearch 及 GAIA、FinSearchComp、ShoppingComp 上评测；基线涵盖 GPT-5.5、Claude Opus 5、Gemini 3.1 Pro、DeepSeek-V4-Pro、Kimi K3、QUEST-35B、AREX-Turbo、FORT-Searcher 等闭源/开源系统。
- **主结果**：Traverse-35B 在 BrowseComp 取得 **72.83**（超越 QUEST-35B 的 64.6、AREX-Turbo 的 70.7，与 FORT-Searcher 的 72.2 持平）；在 DeepSearchQA 达 **82.8 Macro-F1**、WideSearch 达 **72.1 Item-F1**，均为同规模搜索专用模型最优。相比 Qwen3.5-35B-A3B 基座，BC-ZH +1.2、DeepSearchQA +14.3、WideSearch +15.0，泛化到 GAIA（+18.45pp）、FinSearchComp（+19.70pp）、ShoppingComp SoP（+0.1331）亦表现稳定。
- **消融关键数字**：
  - RL 策略对比（BC185 Avg@3）：SFT 60.4 → All-segment RL **51.7**（崩溃）→ 加权 All-segment 57.8 → **Final-segment 65.2**。
  - SFT vs RL 增益：BC185 +4.8 分、WideSearch Row/Item F1 +6.0/+7.9，且搜索轮次减少。
  - Active Seal vs Auto Compaction：Active Seal Avg@3 **65.23** vs 54.95（+10.28），首次 Seal 中位点从 217 轮/231.4K token 提前至 67 轮/67.5K token，轨迹平均长度从 112.2 提升至 217.7。

## 相关工作脉络
- **Web Search Agents**：继承 WebGPT、ReAct 的具身推理范式，但区别于 OpenAI Deep Research、Gemini Deep Research 等黑盒工作流，本文显式引入 Rubric 分解与自我验证循环，强调智能体对搜索节奏的完全控制。
- **上下文摘要/折叠方法**（Resum、Context Folding、Memory as Action）：现有工作多以固定轮次或长度阈值触发压缩；本文将其改为语义驱动的 Agent-Triggered 机制，压缩时机与内容均由策略网络决策。
- **长程 Agent RL**（Search-R1、Research、R1-Searcher）：这类方法多将终端奖励广播全轨迹；本文指出该设置在含记忆重置的轨迹中会导致信用模糊与样本重加权偏差，仅优化最终分段即可兼顾训练稳定性与工具学习能力。
- **评测基准**：BrowseComp（深度检索）、WideSearch（广度检索）、DeepSearchQA（多步证据合成）构成互补评估三角；GAIA/FinSearchComp/ShoppingComp 用于检验跨领域迁移能力。

## 局限性与未来方向
- 当前方法依赖强教师模型蒸馏初始行为，Masked SFT 的掩码质量受限于数据清洗管线，自动化程度仍有提升空间。
- 实验以 256K 上下文与 512 turn 上限为主，虽引入预算随机化缓解过拟合，但在更超长上下文（百万级 token）下的 Seal 触发策略仍需进一步验证。
- Verify 状态主要面向多约束实体链接与数值问答；面对开放型文献综述、多智能体协作或需要外部工具链反馈的复杂任务，自检准则的设计可能面临挑战。
- 未来可探索更细粒度的信用分配（如分段级价值网络）而不牺牲训练效率，或将 Rubric 生成扩展至动态演化型任务。

## 研究启发与可借鉴点
- **Final-segment-only RL 可作为通用记忆工具训练范式**：凡智能体可在交互中途“存盘并清空历史”的场景（如代码工程 Agent、多轮对话客服），均可沿用此策略避免全轨迹广播带来的信用稀释。
- **结构化记忆模板强制区分事实与假设**：知识图谱字段（verified/conflicting/partial）、导航状态（visited/dead_ends/frontier）与元学习记录的设计，能有效防止压缩后死胡同被遗忘，值得迁移至任意长程规划任务。
- **Rubric-Verify 自检闭环可显著提升搜索效率**：将“答案正确性判定”从终端奖励前置为中间状态决策，配合失败反馈的定向重定向，是降低无效搜索轮次的关键设计。
- **上下文预算随机化（$B_\mathcal{G} \sim \text{Uniform}(\mathcal{B})$）防止阈值过拟合**：训练时动态调整每组的上下文上限，迫使模型学习基于搜索进展而非固定 token 计数来触发 Seal，实验表明该技巧对策略鲁棒性影响显著。

## 关键术语表
- **Rubric State**：智能体进入的首个状态，负责将模糊查询拆解为若干可独立验证的结构化条件清单。
- **Seal Memory Tool**：由智能体按需调用的上下文压缩接口，按模板生成结构化记忆并重置对话历史，保留关键证据与失败路径。
- **Seal Collapse**：在 RL 中将终端奖励广播至所有被压缩分段时，因信用分配模糊与样本隐式重加权导致记忆工具使用率趋近于零的训练崩溃现象。
- **Final-Segment-Only RL**：仅对上下文重置后的最后一个分段计算策略梯度，避免全轨迹广播噪声的稳定化训练策略。
- **MASKED SFT**：通过回合级二元掩码屏蔽教师轨迹中的格式错误与行为缺陷，仅在合法 Assistant 回合计算损失。
- **Verify State**：智能体扮演验证者角色，对照 Rubric 重新检索证据并输出 PASS 或 REVISE ANSWER，构成自我纠错闭环。

## 可复现要素
- **数据集**：BrowseComp、BrowseComp-ZH、xbench-DeepSearch、DeepSearchQA、WideSearch、GAIA、FinSearchComp、ShoppingComp 均为公开基准；合成任务管线见 Appendix A，配套训练数据集已开源至 HuggingFace `ByteDance-BandAI/Traverse-AutoGen`。
- **代码/权重**：框架代码开源至 `github.com/ByteDance-BandAI/Traverse`；基座模型为 Qwen3.5-35B-A3B（公开），Traverse 微调权重论文未明确声明开源。
- **关键超参**：最大轮次 512、上下文长度 256K、采样温度 1.0；RL 使用 96 GPU（32 策略 + 64 异步 Rollout）；优化器采用 GSPO + GRPO 组内 Z-score 归一化；Turn/Context 预算惩罚阈值 $\rho_{\text{turn}}=5\%$；预算随机化范围见公式 (20)。
