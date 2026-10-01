---
title: "TIME-ANCHORED-DIFFUSION-LANGUAGE-MODELS-LATENT-SPACE-CACHING"
source: https://arxiv.org/pdf/2609.37924v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 16:53:02"
field: "扩散语言模型高效推理"
keywords: ["diffusion language models", "latent-space caching", "time-anchored diffusion", "inference acceleration", "self-supervised anchoring"]
innovations: ["提出时间锚定的自监督 latent-space caching 框架，无需显式锚 token 监督即可跨步复用隐式表征", "设计轻量融合模块通过残差校正融合当前态与过时锚，支持任意 K 间隔复用", "提出 TADM:Post-train 和 TADM:Pretraining 两种实例化方案，分别在 DiffusionGemma-26B 和 OpenWebText 上实现最高 79% 吞吐提升和 38% 计算减少"]
benchmarks: ["GSM8K", "AIME26", "GPQA-Diamond", "LiveCodeBench-v6", "HumanEval", "MMLU-Pro", "OpenWebText MAUVE/Gen PPL"]
---

# 论文速读：TIME-ANCHORED-DIFFUSION-LANGUAGE-MODELS-LATENT-SPACE-CACHING

## 一句话总结
本文提出时间锚定扩散语言模型（TADM），通过将锚网络学习到的隐式语义表征进行跨扩散步复用（latent-space caching），将昂贵的锚网络计算从每步执行降为每 K 步执行一次，从而在不损失生成质量的前提下显著提升 DLM 推理吞吐量。

## 研究问题与动机
- **DLM 推理昂贵**：扩散语言模型需要在每个去噪步重复评估整个模型，推断成本远高于自回归解码。
- **现有锚定方法依赖监督信号**：ADLM 通过预测重要锚 token 降低条件不确定性，但需要任务相关的锚 token 标注，泛化性受限。
- **现有 KV-cache 复用方法未显式学习时序一致性**：dKV-cache、d²cache 等方法依赖相邻扩散态相似性复用 KV 状态，但这些局部表示未经过时序复用的训练，无法保证跨多步的有效性。
- **锚的语义内容具有跨时间不变性**：锚表征编码序列的持久属性（语义意图、全局结构），其语义内容在邻近扩散时间保持有用，尽管隐式表示会随 token canvas 演化而过时，可通过当前状态修正。

## 核心贡献（创新点）
1. **时间锚定的自监督锚定框架**：将锚从显式 token 预测转为可跨步复用的隐式表征，无需标注即可学习。与 ADLM 的本质区别在于用隐式 latent anchor 替换了监督锚 token 目标，实现了无监督时序复用。
2. **两阶段分解 + 融合模块架构**：将模型拆分为轻量共享网络、昂贵锚网络、小融合模块和轻量去噪器，融合模块用当前状态修正过时锚表示。与 ADLM 的本质区别在于引入融合模块实现"旧+新"的校正，而非仅依赖新鲜锚。
3. **TADM:Post-train 快速适配方案**：仅训练 3.1M 参数的融合模块即可使预训练 DLM 获得 latent-cache 能力。与 ADLM 的本质区别在于无需重新预训练，直接对已训好的 DiffusionGemma-26B 做轻量 post-training。
4. **TADM:Pretraining 端到端训练**：从 scratch 预训练时联合优化时序锚定目标（T-ANELBO），实现计算量与质量的帕累托改进。与 ReMDM/MDLM 的本质区别在于显式暴露模型于过时锚状态以学习时序复用能力。

## 方法详解
- **模型分解**：对噪声序列 $\mathbf{z}_t$，定义共享网络 $S_{\theta_S}$、锚网络 $A_{\theta_A}$、融合模块 $\Phi_\phi$、去噪器 $D_{\theta_D}$。锚刷新步 $t'$ 计算 $\mathbf{h}_{t'} = A_{\theta_A}(S_{\theta_S}(\mathbf{z}_{t'}))$ 并缓存；后续步计算 $\mathbf{c}_t = S_{\theta_S}(\mathbf{z}_t)$，融合得到 $\widetilde{\mathbf{h}}_{t|t'} = \Phi_\phi(\mathbf{c}_t, \mathbf{h}_{t'})$，再送入去噪器。
- **融合模块设计**：核心公式为残差修正形式 $\widetilde{\mathbf{h}}_{t|t'} = \mathbf{h}_{t'} + \mathbf{g}_\phi(\mathbf{c}_t, \mathbf{h}_{t'}) \odot \Delta_\phi(\mathbf{c}_t, \mathbf{h}_{t'})$，门控项 $\mathbf{g}$ 控制融合强度。Post-train 场景中使用 paired attention 实现 $\mathbf{M}_{t,t'} = \phi(\mathbf{Z}, \mathbf{U}) - \phi(\mathbf{Z}_0, \mathbf{U})$，当 $\mathbf{C}=\mathbf{C}_0$ 时修正量精确为零，保证锚刷新瞬间无干扰。
- **TADM:Post-train 训练**：从干净响应 canvas $\mathbf{x}_0$ 出发，采样 anchor 时间 $t'$ 和缓存年龄 $k \in \{0,1,2\}$，通过 stop-gradient rollout 生成当前态 $\widetilde{\mathbf{z}}_t$，再反向传播计算 loss：$\mathcal{L}_{\text{cache}} = \mathbb{E}_{t',k}[\mathcal{L}_{CE} + \lambda_{KD} D_{KL}(p_{\theta_0}^l(\cdot|\widetilde{\mathbf{z}}_t) \| \widehat{\mathbf{x}}^l)]$，其中教师为冻结的预训练模型。
- **TADM:Pretraining 训练（T-ANELBO）**：基于 ReMDM 逆过程，训练时从干净序列采样 $\mathbf{z}_t$ 并进一步加噪得 $\mathbf{z}_{t'}$，使用 stale anchor 训练的 NELBO 下界：$\mathcal{L}_{\text{T-ANELBO}} = \mathbb{E}[{-}\log p_\theta(\mathbf{x}|\mathbf{z}_0)] + \sum_i \mathbb{E}[\lambda_{t(i)} \sum_l \log\langle\mathbf{x}_\theta^l(\mathbf{z}_{t(i)}, \mathbf{z}_{t'(i)}), \mathbf{x}^l\rangle + \gamma \lambda_{t'(i)} \sum_l \log\langle\mathbf{y}_{A_{\theta_A}}^l(\mathbf{z}_{t'(i)}), \mathbf{y}^l\rangle]$。令 $\gamma=0$ 实现完全自监督。
- **计算节省分析**：$C_{\text{cache}}/C_{\text{full}} \approx (L_S + L_D + L_A/K)/(L_S + L_A + L_D)$，增大 $K$ 可降低 Transformer 层调用次数。

## 实验与结果
- **TADM:Post-train（DiffusionGemma-26B）**：在 GSM8K、AIME26、GPQA-Diamond、LiveCodeBench-v6、HumanEval、MMLU-Pro 六个 benchmark 上评估，K=3 时吞吐提升 **49%–79%**（1.49×–1.79×），准确率基本保持不变（最大降幅仅 -1.11pp on AIME26，HumanEval 反而 +0.41pp）。LCB-v6 提升最大达 **1.79×**。
- **TADM:Pretraining（OpenWebText，224M 参数）**：与 ADLM/MDLM/ReMDM/SEDD 等基线对比，T=2048 时 MAUVE 0.650（vs ReMDM 0.610），使用 25% 更少 Transformer 层评估；相对于 ADLM 实现 **最高 73% 吞吐提升**（1.73× @ T=512），Transformer 层调用减少 **38%**。
- **消融**：增大 K 单调提升吞吐；小 T 时质量下降明显，大 T 时对 stale anchor 容忍度高（T=2048 时 Gen PPL 仅从 21.49 升至 22.34，K=1→8）；ADLM 直接使用 stale anchor 会快速退化（T=1024, K=8 时 Gen PPL 从 25.97 恶化到 144.97），证明 stale-anchor 训练与融合模块缺一不可。

## 相关工作脉络
- **ADLM（Rout et al., 2025）**：两阶段锚定 DLM，需显式锚 token 监督；本文扩展为无监督时序复用，锚可跨步缓存。
- **dKV-cache / d²cache（Ma et al., 2025; Jiang et al., 2026）**：复用相邻扩散态的 KV 状态以加速；本文的核心差异在于训练的是可跨步复用的 latent 表征而非 layer-specific KV，且通过融合模块显式校正。
- **MDLM / ReMDM（Sahoo et al., 2024; Wang et al., 2025）**：标准单阶段 masked diffusion LM；本文证明在两阶段分解基础上引入时序锚定可获得更优质量-计算权衡。
- **SEDD / MDLM+DFM（Lou et al., 2024; Gat et al., 2024）**：其他离散扩散建模方案；本文在相同参数预算下取得更具竞争力的 MAUVE 和 Gen PPL。
- **Attention is all you need for KV cache（Nguyen-Tri et al., 2026）**：用注意力机制学习 KV cache 复用；本文与之互补，聚焦于 latent representation 的时序重用而非仅 attention 头级别。

## 局限性与未来方向
- **生成质量在大 K 时退化**：anchor refresh interval 过大会导致 stale anchor 信息不足，质量下降，限制了缓存复用的最大有效跨度。
- **Post-train 受限于冻结 backbone**：TADM:Post-train 仅训练融合模块，性能上限受预训练 backbone 能力约束。
- **未来方向**：探索更大的 K 值、自适应刷新策略（根据生成难度动态调整 K）、将方法扩展到 block diffusion 和其他 DLM 架构、以及结合 SD-RL 等蒸馏技术进一步优化。

## 研究启发与可借鉴点
- **Latent-space caching 作为通用加速范式**：将"学习可复用表征"替代"检测不变的 KV 状态"，为其他迭代生成模型（如流匹配模型、ODE 求解器）提供借鉴。
- **Stale-anchor 训练策略**：在训练中主动注入时序错位（stale anchor age），使模型适应过时表征并学会校正，这一技巧可迁移至其他需要跨步共享信息的生成场景中。
- **融合模块的残差校正设计**：paired attention + 门控残差修正的结构可复用于其他缓存式推理系统，尤其是当缓存与当前状态存在分布偏移时。
- **结合 T-ANELBO 的预训练框架**：将时间锚定目标整合进标准 NELBO 下界，为从 scratch 训练具有缓存能力的 DLM 提供了理论保证。

## 关键术语表
- **TADM（Time-Anchored Diffusion Model）**：将锚表征作为可跨扩散步复用的 latent cache 的扩散语言模型框架。
- **Anchor refresh interval（K）**：锚网络重新计算的间隔步数，控制质量-计算权衡的超参。
- **Fusion module**：轻量模块，将当前扩散态表示与缓存锚表示融合，通过残差修正补偿锚的过时性。
- **T-ANELBO（Time-Anchored Negative Evidence Lower Bound）**：TADM 预训练的变分下界损失，包含时序交叉熵项和可选锚监督项。
- **Rollout construction（Post-train）**：通过 stop-gradient 的 k 步反向扩散采样生成当前态，以近似 inference 分布用于 post-training。
- **MAUVE**：衡量模型生成文本分布与人类文本分布之间距离的指标，越高越好。
- **Gen PPL（Generative Perplexity）**：生成序列的 perplexity，越低越好。
- **Remasking（ReMDM）**：允许已解码 token 重新进入 mask 状态的逆扩散操作，支持生成过程中的错误修正。

## 可复现要素
- **数据集**：TADM:Post-train 使用 UltraData-SFT-2605 和 Nemotron-Post-Training-Dataset-v2；TADM:Pretraining 使用 OpenWebText。**公开**。
- **代码/权重**：论文未提及开源声明。
- **关键超参**：Post-train：融合模块参数 3.1M，K ∈ {1,2,3}，rollout cache age k ∈ {0,1,2}，18750 步，batch size 16，LR 1.5e-4；Pretraining：1M 步，batch size 512，K ∈ {1,2,4,8}，T ∈ {128,256,512,1024,2048,4096}，LR 3e-4，BF16。
