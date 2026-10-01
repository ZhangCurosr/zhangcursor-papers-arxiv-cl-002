---
title: "What-Does-Post-Training-Change-in-Multilingual-Reasoning"
source: https://arxiv.org/pdf/2609.37104v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 11:00:45"
---

块）中目标语言文本占比过半的判定标准。
     - **Termination Failure (终止失败)**：生成未正常结束，包括触发 token 上限或检测到重复循环两类。
     - **Gated Reward (门控奖励)**：将语言奖励与正确性相乘（C×ℓ），确保只有答对才能获得语言bonus，防止错误答案套利。
     - **Pass@k (通过率)**：在 k 次采样中至少有一次成功的概率估计，本文用于计算 C@16 与 J@16。
     - **Delivery Efficiency (交付效率)**：每生成 1000 token 所能获得的联合成功（J=1）数量，用于量化多语言访问的实际成本。
   - **可复现要素:**
     - 数据集：70道竞赛数学题（AIME-2026/AMC-2023），经翻译本地化；DAPO-Math-17K 用于 RL。论文声明所有生成迹与样本级测量均在配套 artifact 中公开。
     - 代码/权重：基于开源 Qwen3-4B/8B 发布版 checkpoint 进行微调与 RL；分析脚本与训练日志随论文发布（Appendix B.5 提及）。
     - 关键超参：SFT 学习率 2e-5，cosine decay，bfloat16，batch 16/4；PPO clip ratio 0.2，γ=λ=1，actor lr 1e-6，critic lr 1e-5，rollouts per prompt 8；详细配置见 Tables S6/S7。
   - **META JSON:** 提取关键词、field、innovations、benchmarks。

4.  **Draft Generation (Mental Refinement - Chinese):**
   I will now write out the full response strictly following the template, ensuring tone matches a senior researcher's note, keeping all key numbers and English names intact. I'll double-check each section against the paper text.

   *Self-Correction/Verification during drafting:*
   - Check J formula: J = C ∧ L ∧ T. Correct.
   - Check numbers: Released 4B: C=90.6%, J=35.7%. 8B: C=92.6%, J=30.6%. SFT 4B: J=84.6%, SFT 8B: J=86.3%. RL gated 4B: J=81.7%, efficiency=102%. All match Table 1 & text.
   - Check RL reward formulas: R_corr, R_add, R_gate. Match Eq 2.
   - Check limitations: Single model family, competition math, single seed, translation artifacts, config differences. Matches "Limitations" section.
   - Check related work: Mentions English-pivoted representations, long CoT gaps, Polymath, MTM-Bench, LANG, ReasonXL. I'll frame the comparison points carefully.
   - Check reproducibility: Appendix B.5 says "All generated traces and per-sample measurements are retained in the accompanying artifact." Code/scripts referenced. I'll state this accurately.
   - Format: Strictly follow the requested headings. No omissions.
   - META: keywords 3-8, innovations 1-3, field, benchmarks.

   Everything looks solid. I will generate the response now. 
   Structure:
   # 论文速读：...
   ## 一句话总结
   ## 研究问题与动机
   ## 核心贡献（创新点）
   ## 方法详解
   ## 实验与结果
   ## 相关工作脉络
   ## 局限性与未来方向
   ## 研究启发与可借鉴点
   ## 关键术语表
   ## 可复现要素
   <!--META ... -->
   All in Simplified Chinese. Faithful to text. No extra fluff.✅
   Proceed. 
   Output matches the final response. 
   *Self-Correction/Note during thought*: I need to make sure the JSON is valid and placed exactly as requested. The template says "最后，在笔记末尾另起一行输出一个机器可读的元信息注释块". I will ensure it's on a new line after the last section.
   Check format: `<!--META\n{"keywords": [...], "field": "...", "innovations": [...], "benchmarks": [...]}\n-->`
   All good. 
   Generating. 
   [Output Generation] -> *Proceeds*
</think>

# 论文速读：What-Does-Post-Training-Change-in-Multilingual-Reasoning

## 一句话总结
本文在固定协议下审计 Qwen3 系列模型从发布版经多语言 SFT 到强化学习各端点的演化轨迹，揭示多语言推理的瓶颈会随训练阶段迁移：发布版受限于“语言选择”，SFT 修复遵从但引入“非终止循环”，仅奖励正确性的 RL 会使模型回退英语，唯有在正确性门控下注入语言奖励，才能实现端到端可靠的多语言交付。

## 研究问题与动机
1. **指标盲区掩盖访问不平等**：传统最终答案准确率（C）会将“用英语写出完整推理迹但答案正确”的响应计入成功，导致多语言可用性与实际性能严重脱节。
2. **孤立缺陷缺乏阶段归因**：既有工作分别记录语言遵从、终止性、效率等失败模式，但未在同一模型族、固定评测协议下追踪后训练各干预手段如何改变主导瓶颈。
3. **SFT 的真实代价未被隔离**：多语言混合训练是否必然损害通用推理能力？还是 SFT 本身带来的精度下降被误归因为语言混合？现有结论缺乏控制实验支撑。
4. **RL 奖励设计的语言效应未知**：仅优化正确性能否维持目标语言输出？语言 bonus 应如何设计才能避免被错误答案套利，同时修复循环终止问题？

## 核心贡献（创新点）
1. **提出以“交付（Delivery）”为中心的三维联合评估**：定义 $J = C \land L \land T$，联合报告正确性、语言遵从、终止性及其协同效率，暴露最高 94 个百分点的“正确但未交付”性能缺口。
2. **隔离 SFT 带来的通用精度代价与多语特有失败模式**：通过英语-only SFT 与 5 个单语专家对照，证明准确率下降是 SFT 通用现象而非多语混合特有，同时明确非终止循环是仅出现在非英语推理迹中的新增瓶颈。
3. **揭示 RL 奖励形式对语言保留的决定性作用**：对比仅正确性奖励与带语言项的奖励，证明语言 term 是决定模型是否回退英语的关键；提出正确性门控奖励（$R_{gate} = C_{tr} \cdot (0.7 + 0.3\ell_{tr})$）以阻断错误答案对语言 bonus 的套利。
4. **构建可复现的分阶段后训练路径**：给出从发布版→多语 SFT→加法阶段 RL→门控阶段 RL 的完整干预链，证实多语言可靠交付可通过分治策略逐步修复。

## 方法详解
- **统一评估协议**：70 道竞赛数学题（30 AIME-2026 + 40 AMC-2023），11 种语言（6 种联合国官方语言 + 5 种训练集外低资源语言），每问题 16 次采样（temperature 0.7），24,576 token 响应预算。
- **三维指标定义**：
  - $C$：最终 `\boxed{}` 答案经 `math_verify` 校验正确。
  - $L$：提取可见推理迹（`</think>` 块），去除数学表达式后按句子切分，目标语言占比 ≥50% 即视为遵从（使用 `langid` 并在 6 种评估语言内限制候选集）。
  - $T$：生成未触达响应上限且重复得分（$1 - \text{compressed}/\text{raw}$ via zlib）≤0.85，否则判为终止失败。
  - $J = C \land L \land T$，直接按样本级布尔交计算，而非边缘概率相乘。
- **SFT 数据构建**：以 45K 英语 OpenR1 长 CoT 数学迹为源，用本地部署的 Qwen3-14B 翻译为 5 种目标语言（经严格门控：保留 `\boxed`、结构完整、无重复、跨语言长度比合理）；4B 语料 43,218 条，8B 语料 56,307 条。控制组包含 9,800 条英语-only SFT 与 5 个单语专家各 9-10K 条。
- **RL 奖励设计**（基于 DAPO-Math-17K 本地化提示，PPO + GAE，$\gamma=\lambda=1$）：
  - $R_{corr} = C_{tr}$（正确性控制臂）
  - $R_{add} = 0.7C_{tr} + 0.3\ell_{tr}$（加法阶段，错误答案也可获最高 0.3 语言分）
  - $R_{gate} = C_{tr} \cdot (0.7 + 0.3\ell_{tr})$（门控阶段，错误答案奖励归零）
- **交付效率度量**：$\text{joint\_per\_lk}(\ell) = 1000 \times \#J=1 / \#\text{调整后 token}$，跨语言归一化到同端点英语值为 $\text{Eff. vs. En}$，100% 表示 parity。

## 实验与结果
- **评测基线**：Qwen3-4B/8B 发布版、多语 SFT、英语-only SFT、5 个单语专家、3 个 RL 端点（共 13 个 checkpoint）。
- **发布版瓶颈（语言选择）**：5 种主要非英语语言平均 $C@16 = 90.6\%$（4B）/ $92.6\%$（8B），但 $J@16$ 仅 $35.7\%$ / $30.6\%$；西班牙语、法语、阿拉伯语遵从率接近 0%，可见推理迹几乎全为英语。
- **多语 SFT 修复遵从但引入循环**：$L$ 升至 $99.4\%$（4B）/ $99.7\%$（8B），$J@16$ 跃升至 $84.6\%$ / $86.3\%$；但非英语终止失败从 $4.7\%$ / $5.8\%$ 升至 $21.1\%$ / $18.2\%$（主要为检测到重复循环），通用准确率下降约 6–8%。
- **RL-仅正确性臂回退英语**：终止失败降至 $0.3\%$，但 $L$ 暴跌至 $8.7\%$，$J@16$ 仅 $15.4\%$，证明单纯优化正确性会抹除 SFT 建立的语言遵从。
- **RL-门控语言臂实现端到端交付**：$L$ 保持 $99.8\%$，终止失败仅 $0.5\%$，$J@16 = 81.7\%$，交付效率达到英语的 $102\%$（调整后）；加法阶段 reward 记录显示错误轨迹曾吸收 44%–82% 奖励，门控后彻底消除该套利路径。
- **最强结果**：Qwen3-4B 经多语 SFT + 门控 RL 后，五种主要非英语语言的端到端交付效率与英语持平甚至略优（raw token 条件下 J-only 效率达英语的 100%）。

## 相关工作脉络
1. **English-pivoted 内部表征研究**（Wendler et al. 2024; Schut et al. 2025）指出多语言 LLM 隐式偏向英语；本文从可见推理迹层面量化该偏向对实际交付的冲击，并给出后训练修复路径。
2. **多语言推理基准**（Polymath、MTM-Bench、CoCo-CoLa）侧重单项指标报告；本文首次在同一协议下联合追踪 C/L/T/J 四维演化，明确“正确性掩盖交付失败”的系统性偏差。
3. **长 CoT 训练对小型模型的影响**（Luo et al. 2025）警告有限监督会退化能力；本文证实 SFT 精度代价具有普遍性（单语对照组同样下降），并非多语混合特有。
4. **语言自适应强化学习**（LANG、ReasonXL）尝试在 RL 阶段引导语言行为；本文进一步区分加法奖励与门控奖励的差异，证明无正确性约束的语言 bonus 会被错误答案劫持。
5. **推理循环现象研究**（Pipis et al. 2026）描述推理模型陷入重复消耗 budget 的行为；本文将其定位为非英语 SFT 后的主导失败模式，并展示 RL 可有效修复。

## 局限性与未来方向
- 结论仅限 Qwen3 家族与竞赛数学任务，未验证其他架构、更多样任务（如代码、开放问答）或更长上下文场景。
- 翻译虽经严格质量门控，但仍可能轻微改变题目自然度或难度分布；尽管跨语言问题难度相关性检验显示 Pearson $r$ 均值达 0.907，翻译引入的潜在偏差仍存。
- 实验仅使用单一随机种子，且英语-only 与单语专家对照组在数据量与 batch size 上与主实验存在差异，因果推断为诊断性而非严格匹配。
- 未来方向包括：拓展至多模型族与多任务域、探索内在循环检测与惩罚机制、研究更鲁棒的多语言奖励几何设计、验证未训练低资源语言（如冰岛语、斯瓦希里语）的零样本迁移潜力。

## 研究启发与可借鉴点
1. **评估框架可迁移**：将可用性问题拆解为 $C/L/T$ 三维联合指标（$J = C \land L \land T$）并报告交付效率，适用于任何需要跨语言/跨模态可靠输出的系统评测，能有效暴露单一准确率指标的虚假繁荣。
2. **分阶段瓶颈修复范式**：SFT 负责语言遵从与知识注入，RL 负责生成结构与终止性规整，二者呈互补瓶颈转移关系；后续多语言 agent/推理系统可复用“监督对齐语言 → RL 修复终止”的两阶段训练流程。
3. **门控奖励设计技巧**：$R_{gate} = C \cdot (\alpha + \beta \cdot \ell)$ 通过正确性门控阻断错误答案对软奖励的套利，该思路可直接迁移至多目标 RLHF/RLAIF、安全对齐或任何需联合优化任务性能与格式遵从的场景。
4. **对照实验设计范式**：设置单语专家、英语-only SFT 与同一多语 checkpoint 出发但不同奖励的 RL 臂，能有效分离“多语混合效应”与“SFT/RL 通用副作用”，为后续语言扩展研究提供可复用的消融范式。

## 关键术语表
**Delivery（交付）**：响应同时满足答案正确、推理迹遵从请求语言且正常终止的端到端可用状态，本文用 $J = C \land L \land T$ 刻画。
**Language Adherence（语言遵从）**：可见推理迹中目标语言文本占比过半即判定为遵从，用于衡量模型是否真正“用用户语言思考”。
**
