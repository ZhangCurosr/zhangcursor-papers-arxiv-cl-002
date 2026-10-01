---
title: "WHEN-CAN-ATTENTION-HEADS-BE-STATICALLY-DEFINED"
source: https://arxiv.org/pdf/2609.34650v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:23:34"
field: "高效大语言模型训练"
keywords: ["attention freezing", "efficient training", "fixed attention pattern", "kernel fusion", "head heterogeneity", "causal language model"]
innovations: ["提出 SAF：通过注意力方差选择低波动 head 并用固定均值模式替换其 QK 计算", "设计 α+ρ 线性紧凑表示将 O(T²) 存储降至 O(T) 并配合融合 kernel 跳过 softmax", "证明保留固定模式 token mixing 在关联回忆任务中显著优于同 head 数剪枝"]
benchmarks: ["FineWeb-Edu pretraining", "SST-2", "BoolQ", "QuALITY", "MQAR", "Qwen3-4B zero-shot"]
---

# 论文速读：WHEN CAN ATTENTION HEADS BE STATICALLY-DEFINED

## 一句话总结
本文提出了选择性注意力冻结（Selective Attention Freezing, SAF），在预训练中途选出注意力模式方差较低的注意力头，将其注意力权重替换为拟合后的固定因果模式，通过紧凑存储与融合 kernel 实现训练加速。在 124M/4K 与 1B/8K 设置下，25% 替换率仅带来约 0.5–0.8% 的 perplexity 增加，而优化器更新速度提升约 1.06×，并显著改善了长上下文微调与因果 prefill 的效率。

## 研究问题与动机
- 标准自注意力为每个 head 和每个输入动态计算 query-key 分数与 softmax，计算复杂度随序列长度呈二次增长。
- 部分 attention head 的权重主要取决于位置而非 token 内容（如 RoPE 下的位置敏感 head），这些 head 在不同输入间表现出低方差，存在用固定模式替代的潜力。
- 现有工作（如 PAPA、固定位置模式、随机 token mixing）仅验证了"无内容依赖也可有效"，但未系统回答"哪些 head 可替换、何时替换、混合层能否在现代 kernel 上同时节省内存与耗时"等问题。
- 替换后保留 value 和输出投影的 trainable 性，可保留 input-dependent 的 token mixing，这与直接剪枝不同，是一种新的效率–表达能力权衡路径。

## 核心贡献（创新点）
- 提出了 **Selective Attention Freezing (SAF)** 方法，在预训练中途一次性选中低方差 head，将其注意力权重固定为拟合的因果模式，同时保留 value/output 投影的训练能力。
- 设计了基于**绝对位置（α）与相对距离（ρ）**的线性紧凑表示，将存储从 O(T²) 降至 O(T)，并给出对应的交叉熵拟合损失（Eq. 4）。
- 实现了**融合 kernel**，在单次前向发射中同时执行普通 FlashAttention 与固定模式 head 的 token mixing，跳过 softmax 与密集模式读取。
- 通过受控对比实验，系统比较了五种固定模式（post-softmax mean/sharp mean/Gaussian/Dirichlet/structured random）、两种选择策略（attention variance/forward KL）以及多种调度方案，给出了"SAF 配方"（方差选择 + 均值模式 + 一次式中点替换 + 25% 替换率）。
- 证明保留固定模式的 token mixing 在关联回忆任务（MQAR）中优于同等 head 数量的剪枝方案，泛化到更多 key-value 对时优势明显（124M 模型在 64 对上 SAF 达 54.4%，剪枝仅 ~26–29%）。

## 方法详解
- **标定与 head 选择**：在替换 checkpoint 处，使用 N 条标定序列（eval 模式，不改变权重和数据采样器）测量各 head 的注意力矩阵 A(ℓh)(n)，计算 Empirical mean Â(ℓh)。按注意力方差分 s_var 全局排序，选取 k=round(rLH) 个 head；另对比 forward-KL 分 s_KL。
  - 方差分：$s_{\ell h}^{\mathrm{var}} = \frac{1}{(N-1)T^2}\sum_n \|\mathbf{A}_{\ell h}^{(n)} - \widehat{\mathbf{A}}_{\ell h}\|_F^2$
  - Forward-KL 分：$s_{\ell h}^{\mathrm{KL}} = \frac{1}{NT}\sum_{n,i} \mathrm{KL}(\mathbf{A}_{\ell h}^{(n)}[i,1:i] \| \widehat{\mathbf{A}}_{\ell h}[i,1:i])$
- **固定模式构建**：比较五种候选因果行随机模式——post-softmax mean（$\widehat{\mathbf{A}}$）、sharp mean（softmax of mean logits）、Gaussian sample、Dirichlet sample、structured random（控制）。Post-softmax mean 在匹配 token 预算下取得最低或接近最低的 PPL。
- **紧凑表示与拟合**：用长度 T 的向量 α（绝对位置偏好）和 ρ（相对距离偏好）参数化：
  $\widehat{\mathbf{P}}(i,j) = \exp(\alpha(j)+\rho(i-j))/Z(i),\ j\le i$
  拟合目标为行平均交叉熵：
  $\mathcal{L}_{\mathrm{fit}}(\alpha,\rho) = \frac{1}{T}[\sum_i \log Z(i) - \sum_j \eta_{abs}(j)\alpha(j) - \sum_\delta \eta_{rel}(\delta)\rho(\delta)]$
  其中 η_abs/η_rel 为目标的绝对位置与相对距离边缘统计量；拟合容差 ε_fit=0.2 nats/行。
- **替换调度**：对比一次性替换、渐进替换（分 5 组）、验证损失预算调度、训练损失触发调度。实验表明固定的一次性中点替换效果最佳。
- **融合执行**：普通 head 使用 FlashAttention-2 online softmax tiling；替换 head 在寄存器中重建因果 tile 并与 V tile 相乘，跳过 score/softmax。反向传播中，replacement head 的 Q/K 梯度被省略（或在 GQA 路径中省略 Q 投影但保留共享 K/V）。α、ρ 及归一化器不参与梯度更新。

## 实验与结果
- **数据集与模型**：FineWeb-Edu 预训练；124M 模型（12 层，12 head，d=768，RoPE/Absolute pos，4K/8K/16K，共 2.4576B tokens）；1B 模型（32 层，16 head，d=1536，RoPE，8K，19.667B tokens，4×GH200）；Qwen3-4B 零样本评估。
- **基线**：Gate-Taylor pruning、随机 head 替换、不同固定模式、不同选择分数、不同调度方案。
- **主要结果（124M，4K，25% 替换）**：
  - ΔPPL = +0.768 ± 0.067%；post-replacement 更新速度 1.056×；峰值内存 −1.96%。
  - Post-softmax mean 表现最优（+0.177%），Sharp mean 次之（+0.662%），Gaussian/Dirichlet/Structured random 误差更大。
  - Variance selection 优于 Forward KL（25% 时 0.665% vs 1.068%）。
  - 方差选中 head 集中在较浅层，注意力更弥散（熵 0.837 vs 0.678，最近 64 token 注意力 19.0% vs 40.7%）。
- **主要结果（1B，8K，25% 替换）**：
  - ΔPPL = +0.507%；4×GH200 更新速度 1.068×；微调准确率变化 <0.72pp。
  - 若累计 100 万 GPU 小时，25% SAF 可节省约 6.4 万 GPU 小时。
- **长上下文**：16K 下 25% 替换仅 ΔPPL=+0.450%，prefill 加速 1.09–1.12×（25%）与 1.20–1.24×（50%）。
- **MQAR 关联回忆**：适配 8 对后，在 64 对/512 token 下 SAF 达 54.4% 准确率，远超 ordinary attention（26.2%）和两种剪枝基线（26.2–29.0%）。
- **微调任务**：SST-2/BoolQ/QuALITY 上绝对准确率变化均 <0.8pp；QuALITY 16K 50% 替换下更新加速 1.118×。

## 相关工作脉络
- **PAPA (Hassid et al., 2022)**：在 encoder 中使用输入平均注意力；本文将其思想迁移至 causal LM 的中间替换，并引入紧凑存储与融合执行。
- **Synthesizer (Tay et al., 2021) / Fixed positional patterns (Raganato et al., 2020)**：使用输入无关模式；本文强调"选择性"（仅替换低方差 head）与训练中期一次性干预。
- **FLAP (An et al., 2024) / Gate-Taylor pruning (Michel et al., 2019)**：结构化剪枝；本文指出方差低的 head 适合"固定权重但保留 token mixing"，而非直接移除。
- **DuoAttention (Xiao et al., 2025) / MInference (Jiang et al., 2024)**：利用 head 异质性设计 sparse pattern；本文通过方差选择 + 固定均值模式，在训练阶段即减少 Q/K 投影与 softmax 计算。
- **FlashAttention (Dao et al., 2022/2024)**：高效 exact attention；本文融合 kernel 在其基础上扩展支持 fixed-pattern head，消除 score/softmax 并合并 kernel launch。

## 局限性与未来方向
- 替换率超过 25% 后 perplexity 代价显著上升（50% 时 ΔPPL ≈ 2.0–2.5%），高替换率场景适用性受限。
- 仅在预训练中途进行一次替换；动态/渐进替换的自适应调度虽被测试但未优于固定方案。
- 当前验证主要在 decoder-only causal LM 上， encoder 或多模态场景的泛化性未探索。
- MQAR 等特定任务的增益机制（固定模式仍保留有用 mixing）尚需更深入的理论解释。
- 论文未讨论与 KV-cache 剪枝/压缩等推理阶段优化技术的结合方式。

## 研究启发与可借鉴点
- **"方差选择 + 均值拟合"的配对逻辑**：同一目标（最小化重构误差）同时决定"选哪些 head"和"用什么模式"，概念简洁且易于复用到其他 head-heterogeneity 场景。
- **紧凑线性参数化（α+ρ）**：将 O(T²) 的固定矩阵压缩为两个 O(T) 向量，兼顾存储与拟合效率，可推广至其他需要固定 attention pattern 的场景（如长上下文推理）。
- **保留 token mixing 的"半冻结"思路**：与剪枝相比，固定权重但保留 V 投影训练，在关联回忆等任务中展现出更强的泛化能力，为"结构效率"与"功能保留"的权衡提供新视角。
- **融合 kernel 设计范式**：通过 head-state flag 在单次 launch 内分支处理普通/固定 head，避免额外 concat 开销，对异构头架构的 kernel 实现具有参考价值。
- **校准数据与报告数据分离**：标定序列独立于训练顺序和最终报告集，确保评估无泄漏，这一实验规范值得在类似干预方法中遵循。

## 关键术语表
- **Selective Attention Freezing (SAF)**：一种在预训练中途选择性冻结低方差注意力头权重的方法，用固定因果模式替代其动态 QK 计算。
- **Attention variance score**：衡量单个 head 在不同输入间注意力矩阵波动程度的指标，用于排序和选择可被固定的 head。
- **Post-softmax mean pattern**：将校准输入上的注意力概率取平均作为固定模式，是 forward-KL 意义下的最优固定近似。
- **Compact pattern (α+ρ 表示)**：用绝对位置偏好向量 α 和相对距离偏好向量 ρ 参数化的 O(T) 存储形式，通过 Eq.3 重建因果注意力矩阵。
- **Fused mixed-head kernel**：在单次 kernel launch 中同时执行普通 FlashAttention 与固定模式 token mixing，跳过 softmax 并合并输出。
- **MQAR (Multi-Query Associative Recall)**：要求模型从更早位置检索配对 value 的合成任务，用于评估固定模式 head 的关联泛化能力。
- **Gate-Taylor pruning**：基于损失对 head 输出标量 gate 的梯度幅值评估 head 重要性的剪枝方法，作为本文的剪枝基线。
- **Replacement schedule**：控制何时、以何种节奏安装固定 head 的策略，包括一次性、渐进、验证预算和训练损失触发等变体。

## 可复现要素
- **数据集**：FineWeb-Edu（开源）、SST-2、BoolQ、QuALITY、HellaSwag、PIQA、ARC-Easy；MQAR 使用 Zoology 生成器（Apache 2.0）。
- **代码**：https://github.com/waylonli/Selective-Attention-Freezing（已开源）。
- **模型权重**：124M 从头训练，1B 与 Qwen3-4B 使用公开权重或自身训练 checkpoint。
- **关键超参**：124M 模型 AdamW lr_peak=6e-4，warmup 500 steps，cosine decay；calibration 32 序列；fitting 400 Adam steps lr=0.05；ε_fit=0.2 nats。1B 模型 lr_peak=3e-4，375 warmup steps。
