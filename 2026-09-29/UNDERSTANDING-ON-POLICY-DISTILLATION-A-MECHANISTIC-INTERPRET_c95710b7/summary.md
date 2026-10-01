---
title: "UNDERSTANDING-ON-POLICY-DISTILLATION-A-MECHANISTIC-INTERPRET"
source: https://arxiv.org/pdf/2609.35210v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:22:25"
field: "大模型蒸馏与机制可解释性"
keywords: ["on-policy distillation", "mechanistic interpretability", "sparse crosscoder", "feature reweighting", "SFT warm-up", "decision token"]
innovations: ["提出swap readout追踪训练前后特征使用变化", "揭示OPD本质为共享特征重新加权而非特征创造", "量化SFT预热沿OPD方向与额外方向的分解效应"]
benchmarks: ["AIME 2024", "AIME 2025", "AMC 2023", "DAPO-Math-17k"]
---

# 论文速读：UNDERSTANDING-ON-POLICY-DISTILLATION-A-MECHANISTIC-INTERPRET

## 一句话总结
本文从机制可解释性视角揭示：在线策略蒸馏（OPD）并不为学生创造新特征或传递教师自有特征，而是重新加权学生已共享的特征；其前置的SFT预热同样如此，通过提前完成部分重新加权并引入额外方向的变化来提升OPD效果。

## 研究问题与动机
- OPD已成为主流后训练范式（被Qwen3、MiMo-V2-Flash、GLM-5、Kimi K3等采用），但现有研究多关注学生输出端表现，缺乏对学生内部表征变化的理解。
- OPD的实际效果存在不确定性：有时会失败甚至退化（Li et al., 2026; Zhu et al., 2026），且主要提升采样效率而非扩展能力边界（Ge et al., 2026），引发"学生到底学到了什么"的核心问题。
- 现有交叉编码器分析只能识别某一模型特有的特征，无法可靠追踪训练过程中学生特征使用方式的变化（Minder et al., 2026）。
- SFT预热（在教师rollouts上的监督微调）普遍提升OPD效果，但其作用机制尚未被理解。

## 核心贡献（创新点）
- 提出swap readout机制：将学生checkpoint同时置于交叉编码器的两个学生输入槽位，固定教师输入，从而测量该checkpoint对每个共享特征的独立使用，并可泛化到训练时未见过的checkpoint。
- 发现OPD的本质是特征重新加权：在三种OPD设置下，OPD既不创建学生自有特征，也不传递教师特征，仅轻微调整学生已共享特征的触发频率（>98%频繁特征的触发率变化<20%）。
- 揭示SFT预热的作用机制：预热不创造新特征，而是沿OPD方向完成约一半的重新加权，并额外改变对话格式、推理风格和数学符号等特征，施加该重新加权可在不修改权重的情况下恢复约46%的教师优势。

## 方法详解
- **稀疏交叉编码器（Sparse Crosscoders）**：对多个模型学习共享特征字典，编码器公式为 $z = \sigma(\sum_m W_m h_m + b)$，重建公式为 $\hat{h}_m = D_m z$，使用BatchTopK保持稀疏性（每token平均50个激活特征）。
- **模型归属分数（MAS）**：$MAS(m,j) = \frac{\|d_{mj}\|_1 / \sqrt{n_m}}{\sum_{m'} \|d_{m'j}\|_1 / \sqrt{n_{m'}}}$，衡量特征j属于模型m的占比，>0.5表示模型主导。
- **Swap Readout**：对于checkpoint s，令 $z_s(x) = enc(\bar{h}_s(x), \bar{h}_s(x), h_T(x))$，其中两个学生槽位输入相同激活，教师槽位固定。理论证明可消除未确定项 $N\delta$，使激活仅由学生自身决定。
- **特征干预实验**：计算预热的特征变化 $\Delta h = D_O(z_S - z_B)/s_O$，在交叉编码器所在层直接加/减到残差流，验证因果效应。

## 实验与结果
- **数据集**：训练使用DAPO-Math-17k；评估使用AIME 2024/2025（各30题）和AMC 2023（40题）。
- **三种OPD设置**：JustRL（1.5B/1.5B）、Skywork（1.5B/7B）、R1-7B（1.5B/7B）。
- **核心发现**：
  - 98.2%（JustRL）、99.6%（Skywork）、99.4%（R1-7B）的频繁特征触发率变化<20%。
  - 决策token特征（Wait/Hmm/So等）仅占0.4%，但占变化最大的50个特征的12%（JustRL）。
  - 决策token处OPD改变10–44%的活跃特征，教师与学生KL在决策token处为平均值的3.0–3.6倍。
- **SFT预热效果**（Qwen3设置，avg@8）：
  - Base: 9.7%，OPD: 17.6%（恢复30%），Warm-up+OPD: 22.0%（恢复46%）。
  - 特征干预：施加预热变化使直接蒸馏学生从18.6%→21.3%，移除使预热学生从21.9%→19.2%。

## 相关工作脉络
- Agarwal et al. (2024) 提出OPD原始框架，本文从其表征层面补充机制解释。
- Li et al. (2026) 分析OPD现象学与失败模式，本文从特征层面揭示"兼容性思维模式"的具体含义。
- Shi et al. (2026) 发现SFT可引入新特征，本文区分了"教师rollouts预热"与"强模型解SFT"的本质差异。
- Lindsey et al. (2024) 提出交叉编码器，本文改进其readout机制以追踪训练引起的特征使用变化。
- Ge et al. (2026) 从测试时扩展角度指出OPD不扩展能力边界，本文从特征重加权给出底层解释。

## 局限性与未来方向
- 仅分析中间层（block 14/18）表征，未覆盖全层或输出层特征动态。
- 研究集中在数学推理任务，结论在其他领域（如代码生成、对话）的推广性待验证。
- 交叉编码器字典大小固定为32,768，可能遗漏细粒度特征。
- 未深入探索"哪些特征变化可直接用于指导蒸馏策略设计"。

## 研究启发与可借鉴点
- Swap readout可迁移到其他训练前后对比场景（如SFT、RL），用于追踪特征使用变化而非仅识别模型特有特征。
- 决策token层面的特征分析为理解推理链关键节点提供新视角，可指导蒸馏时的加权策略设计。
- 特征干预实验（不改权重仅改激活）为验证表征变化因果效应提供可靠方法。
- "预热沿OPD方向+额外方向"的分解框架可推广至其他先训练后蒸馏流程的分析。

## 关键术语表
**On-policy distillation (OPD)**：在线策略蒸馏，学生用自己的rollouts配合教师dense token级监督进行训练的后训练范式。
**Sparse crosscoder**：稀疏交叉编码器，为多模型共享单一特征字典的稀疏自编码器变体，用于模型间特征对比。
**Swap readout**：交换读出，将同一checkpoint同时输入交叉编码器的两个学生槽位以隔离其独立特征使用的分析方法。
**Model Attribution Score (MAS)**：模型归属分数，衡量某特征解码器范数在模型间的占比，>0.5表示该模型主导此特征。
**Decision token**：决策token，推理链中标志下一步行动的关键词（如Wait/So/Hmm），OPD在此处的特征变化最显著。
**Firing rate**：触发率，特征在 token 上激活的比例，用于衡量特征使用情况。

## 可复现要素
- **数据集**：DAPO-Math-17k（公开）、AIME 2024/2025、AMC 2023（公开竞赛题）、OpenThoughts-114k、RedPajama-Data-1T-Sample。
- **代码/权重**：项目页面 https://yzc-666.github.io/understanding-opd-crosscoders/；模型为开源模型（DeepSeek-R1-Distill-Qwen、Qwen3系列）。
- **关键超参**：Crosscoder字典32,768、BatchTopK k=50、OPD学习率1e-6、预热LoRA rank 32 α=32 lr 5e-5、训练epoch约2。
