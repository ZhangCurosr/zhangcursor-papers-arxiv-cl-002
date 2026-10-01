---
title: "UNDERSTANDING-ON-POLICY-DISTILLATION-A-MECHANISTIC-INTERPRET"
source: https://arxiv.org/pdf/2609.35210v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:55:01"
field: "大语言模型后训练与可解释性"
keywords: ["on-policy distillation", "mechanistic interpretability", "sparse crosscoder", "feature reweighting", "SFT warm-up", "decision token"]
innovations: ["提出swap readout方法追踪训练前后特征使用变化", "揭示OPD本质是重加权而非特征创建", "阐明SFT预热通过特征重加权而非引入新特征提升OPD效果"]
benchmarks: ["AIME 2024", "AIME 2025", "AMC 2023"]
---

# 论文速读：UNDERSTANDING-ON-POLICY-DISTILLATION-A-MECHANISTIC-INTERPRET

## 一句话总结
本文通过稀疏Crosscoder和提出的swap readout方法，从机理可解释性视角揭示：On-policy Distillation (OPD) 并不为学生创建新特征，也不传递教师独有特征，而是对学生已与教师共享的特征进行**重加权**；SFT预热同样仅重加权共享特征，且其带来的性能提升可通过直接施加特征层面的变化复现。

## 研究问题与动机
1. OPD已成为LLM推理后训练的标准范式（被Qwen3、MiMo-V2-Flash、GLM-5、Kimi K3等采用），但"OPD究竟向学生内部表征中蒸馏了什么"这一基本问题仍未解决。
2. 现有OPD分析仅研究学生输出层面（token概率、准确率、训练信号），无法揭示学生**内部表征**的变化。
3. 标准Crosscoder分析通过decoder norm识别特定于某一模型的特征，但**无法追踪模型如何使用特征的变化**（joint encoding将所有模型编码为单一特征激活集合）。
4. SFT预热（在教师rollout上先做SFT）常被用于提升OPD效果，但其作用机制不明；一种直觉认为预热为学生"提供了新特征"，但未见检验。

## 核心贡献（创新点）
1. **提出swap readout方法**：将学生checkpoint同时置于crosscoder的两个学生槽位、教师激活固定，从而读取该checkpoint独立使用的特征激活；与已有工作（通过decoder attribution识别model-specific特征）的本质区别在于，它追踪的是**模型对共享特征使用方式的变化**，而非特征归属。
2. **揭示OPD的本质是特征重加权**：OPD不创建学生自己的新特征，也不传递教师的独有特征；>98%的高频特征 firing rate 变化低于20%。与已有OPD分析（仅关注输出层面）的本质区别在于，该发现直接面向**学生内部表征**。
3. **阐明SFT预热的作用机制**：预热同样不创造新特征，也不使教师特征变为共享；它沿OPD方向预完成部分重加权，并额外引入OPD不会做出的重加权（涉及对话格式、推理风格、数学符号），且这种重加权具有**因果效应**。与已有SFT特征分析（Shi et al., 2026，SFT在独立强模型解上训练会引入新特征）的本质区别在于，本文预热仅在教师自身rollout上训练，作用机制完全不同。

## 方法详解
**Sparse Crosscoder（BatchTopK）**：
- 学习一个所有模型共享的特征字典 $\mathbf{z}$，每个模型有独立的decoder $\mathbf{D}_m$。
- 编码公式：$\mathbf{z} = \sigma\left(\sum_m \mathbf{W}_m \mathbf{h}_m + \mathbf{b}\right)$，解码：$\hat{\mathbf{h}}_m = \mathbf{D}_m \mathbf{z}$。
- 稀疏性通过 $\sigma$（BatchTopK，每batch平均保留 $k=50$ 个最高激活特征）实现，避免 $\ell_1$ 惩罚导致的shared特征假象。

**Model Attribution Score (MAS)**：
- $\mathrm{MAS}(m, j) = \frac{\|\mathbf{d}_{mj}\|_1 / \sqrt{n_m}}{\sum_{m'} \|\mathbf{d}_{m'j}\|_1 / \sqrt{n_{m'}}}$，衡量特征 $j$ 归属于模型 $m$ 的比例；MAS > 0.5 表示模型 $m$ 主导该特征。

**Swap Readout（核心方法）**：
- 将checkpoint $s$ 的激活同时放入两个学生槽位：$\mathbf{z}_s(x) = \mathrm{enc}(\bar{\mathbf{h}}_s(x), \bar{\mathbf{h}}_s(x), \mathbf{h}_T(x))$。
- 原理：两个学生槽位的encoder差异 $\mathbf{N} = \frac{1}{2}(\mathbf{W}_B - \mathbf{W}_O)$ 在训练后几乎未被确定（$\|\mathbf{N}\|_F / \|\mathbf{S}\|_F \approx 0.5$，与随机初始化相近），只有和 $\mathbf{S} = \mathbf{W}_B + \mathbf{W}_O$ 在训练中被确定；将同一activation放入两槽使 $\mathbf{N}$ 项抵消，仅保留 $\mathbf{S}\bar{\mathbf{h}}_s + \mathbf{c}(x)$，从而可靠读取特征使用。
- 即使对crosscoder训练时**未见过的checkpoint**（如SFT预热后的student），也能可靠读取。

**损失函数**：OPD使用vanilla OPD配方，token级信号为reverse KL $D_{KL}(q\|p)$ 的学生top-16 token log-ratio加权，无其他loss/奖励。

## 实验与结果
**实验设置**：
- Base student：DeepSeek-R1-Distill-Qwen-1.5B（维度1536）；Teacher三种：JustRL-DeepSeek-1.5B（同维度）、Skywork-OR1-Math-7B（3584）、R1-Distill-Qwen-7B（3584）。
- 训练数据：DAPO-Math-17k，一pass（279步），64 prompts × 4 rollouts/step。
- 评估：AIME 2024、AIME 2025、AMC 2023，avg@$n$ 指标。

**主要结果**：
- **JustRL设置**：OPD avg@256 = 56.6%，恢复教师优势80%。
- **Skywork设置**：OPD avg@256 = 71.1%，恢复教师优势26%（师生差距大，OPD进步最大）。
- **R1-7B设置**：OPD avg@256 = 43.5%，仅恢复教师优势9%（师生分布高度重叠，OPD几乎不移动学生）。
- **SFT预热+OPD（Qwen3实验）**：预热前base avg@8 = 9.7%，直接OPD = 17.6%（恢复30%），预热+OPD = 22.0%（恢复46%）。

**最强结果**：Skywork设置下OPD学生AIME 2024正确率67.2%，AIME 2025为52.1%，AMC 2023为93.9%，显著提升。

**特征干预实验（因果验证）**：将预热产生的特征变化直接施加于未预热的直接蒸馏学生（不改权重），avg@8从18.6%→21.3%（+2.7，bootstrap [0.5, 6.0]）；施加于打乱特征的控制组则无改善（15.9%，−2.7）；从预热学生中移除该变化降至19.2%（−2.7）。证明重加权的**因果效应**。

**决策token特征**：OPD改变最大的特征中，decision token特征（Wait/Actually、Hmm/Maybe、So/Therefore等）占比远超其 baseline（0.4%→12%/4%），且firing rate向教师方向移动；在这些token处教师-学生KL偏差最大（3.0–3.6倍平均值）。

## 相关工作脉络
1. **OPD基础算法**（Agarwal et al., ICLR 2024）：提出on-policy distillation框架，学生采样轨迹、教师提供dense token级监督。本文在此框架上做表征层面的机理分析。
2. **OPD成功失败条件**（Li et al., 2026）：发现OPD需师生thinking pattern兼容，否则无效或退化。本文从特征重加权角度给出了表征层解释。
3. **OPD与test-time scaling**（Ge et al., 2026）：发现OPD主要改善sampling效率而非扩展能力边界。本文的"仅重加权共享特征"发现为其提供了表征机制解释。
4. **SFT vs RL特征分析**（Shi et al., ACL 2026）：发现SFT引入model-specific特征，RL基本保留base模型表征。本文对比指出：在教师rollout上的SFT预热**不**引入新特征，作用机制不同。
5. **Sparse Crosscoder**（Lindsey et al., 2024；Minder et al., NeurIPS 2026）：学习多模型共享特征字典以比较模型。本文在其基础上提出swap readout，解决"如何追踪同一特征使用变化"的问题。
6. **Simple-OPD**（Liu et al., 2026）：发现SFT预热能显著提升OPD效果。本文给出了预热的特征层面作用机制解释。

## 局限性与未来方向
1. 实验仅聚焦**数学推理任务**，结论在更广泛任务领域的泛化性有待验证。
2. 分析仅在残差流的**单个中间层**（block 14/18）进行，未探索多层特征变化模式。
3. Crosscoder字典规模32,768，相对较小，可能无法完全捕获模型表征的复杂性。
4. Swap readout需对每个checkpoint运行两次前向传播（尽管共享encoder），计算开销较高。
5. 未来可将swap readout推广至更多后训练范式（如SFT、RLHF），以及探索特征干预直接用于模型改进的可能性。

## 研究启发与可借鉴点
1. **Swap readout的可迁移价值**：该方法可复用于分析任意"训练前后同一模型变化"的问题（如不同SFT策略、RL训练、continued pretraining），无需重新训练crosscoder即可追踪未见过checkpoint的特征使用变化。
2. **特征干预实验的设计**：通过decode code difference并add到residual stream来实施特征级干预，是一种无需改权重的"轻量"因果验证方法，可直接用于评估其他表征变化的实际影响。
3. **对预训练/后训练策略的启示**：若目标仅是提升sampling效率而非扩展能力，"让师生共享特征的重加权更充分"比"引入新特征"更重要；预热应聚焦于调整推理风格和决策token使用模式。
4. **可结合本团队方向**：若团队研究LLM后训练效率优化，可借鉴此视角设计"特征对齐度"作为OPD适配性的快速评估指标，替代昂贵的端到端训练。
5. **决策token特征的发现**：可进一步探索decision token相关特征是否在其他任务（如代码生成、对话）中也有类似的关键作用，形成通用的"trace decision monitoring"分析工具。

## 关键术语表
**On-policy Distillation (OPD)**：学生模型在自己生成的轨迹上接受教师token级监督的后训练技术，减少exposure bias。
**Sparse Crosscoder**：学习多个模型共享的特征字典，每个特征有统一激活但每个模型有独立decoder方向，支持跨模型feature-level比较。
**Swap Readout**：将同一checkpoint激活同时放入crosscoder两个学生槽位来读取其特征使用的方法，消除未确定的encoder差异项。
**Model Attribution Score (MAS)**：通过decoder norm比例衡量某特征归属于某模型的程度，MAS > 0.5表示模型主导该特征。
**Firing Rate**：特征在多少比例的token上激活（entry of z为正），反映特征的活跃程度。
**Decision Token**：推理链中标志"下一步动作"的词（Wait/Actually反思、Hmm/Maybe犹豫、So/Therefore推进），OPD最大特征变化集中于此类token。
**SFT Warm-up**：在OPD之前，在教师自身rollout上对student做少量SFT，本文发现其通过重加权共享特征而非创造新特征来提升OPD效果。
**Reverse KL Distillation Loss**：$D_{KL}(q\|p)$，以教师分布$p$为主体、学生分布$q$为次体的KL散度，近似为学生top-k token的log-ratio加权。

## 可复现要素
- **数据集**：训练集DAPO-Math-17k（公开）、Crosscoder训练数据OpenThoughts-114k + RedPajama-Data-1T-Sample（均公开）；评估集AIME 2024/2025、AMC 2023（公开竞赛题）。
- **代码**：项目页面 https://yzc-666.github.io/understanding-opd-crosscoders/，论文声明使用AI辅助编写分析代码，但具体开源状态论文未明确说明代码仓库链接；建议关注项目页面获取。
- **模型权重**：Base student（R1-Distill-Qwen-1.5B）、Teachers（JustRL-DeepSeek-1.5B、Skywork-OR1-Math-7B、R1-Distill-Qwen-7B）均为开源模型（DeepSeek/Qwen/Skywork官方发布）。
- **关键超参**：Crosscoder字典32,768、k=50、learning rate 1.41e-4、warmup 1000步；OPD learning rate 1e-6、AdamW β₁=0.9 β₂=0.999、weight decay 0.01；SFT预热LoRA rank=32、α=32、lr=5e-5、175步。
