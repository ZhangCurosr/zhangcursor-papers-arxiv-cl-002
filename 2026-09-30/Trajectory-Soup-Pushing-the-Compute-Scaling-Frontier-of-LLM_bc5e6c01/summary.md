---
title: "Trajectory-Soup-Pushing-the-Compute-Scaling-Frontier-of-LLM"
source: https://arxiv.org/pdf/2609.37169v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:57:00"
field: "大语言模型中期训练与计算分配"
keywords: ["mid-training", "model merging", "trajectory diversity", "compute scaling", "weight averaging", "LLM"]
innovations: ["将轨迹数量作为独立的计算分配轴，突破单条轨迹的串行计算缩放饱和墙", "提出双层融合框架（intra + inter trajectory averaging）并给出偏差-方差理论分解", "证明均匀权重在方差意义上最优，并揭示方向多样性可预测插值增益"]
benchmarks: ["ARC-Easy/Challenge", "MMLU-Pro", "GSM8K", "HumanEval", "LiveCodeBench", "C-Eval", "IFEval", "BFCL v4"]
---

# 论文速读：Trajectory-Soup-Pushing-the-Compute-Scaling-Frontier-of-LLM

## 一句话总结
论文提出 **Trajectory Soup**，将 LLM mid-training 的计算预算分配给多条从共享 checkpoint 分叉的独立优化轨迹，分别在每个轨迹内选取最佳 checkpoint 取平均得到"分支锚点"，再跨轨迹取平均，从而在相同计算量下超越单条长轨迹的串行训练上限，并将优势传递至后续 SFT 阶段。

## 研究问题与动机
1. **Mid-training 的串行计算缩放存在饱和墙**：下游性能随 token 预算快速上升后迅速 plateau，过度训练甚至会退化（Fig. 1a），单纯延长单条轨迹的收益有限。
2. **现有融合方法的局限性**：现有工作多聚焦 intra-trajectory averaging（如 SWA）或 inter-trajectory endpoint averaging（如 Model Soups），二者如何协同利用仍未被系统探索。
3. **计算分配轴未受重视**：已有 scaling law 研究仅关注参数/数据/轨迹长度，而未将"轨迹数量"视为独立的计算分配维度。
4. **可控配方扰动能否产生互补方向**：从同一 checkpoint 出发、经数据顺序/学习率/批次大小/调度/优化器等受控扰动产生的轨迹，是否探索了参数空间中几何上足够不同的优化方向。

## 核心贡献（创新点）
1. **将 mid-training 计算分配重新定义为"轨迹数 vs 轨迹长度"问题**，指出当单条轨迹长度已饱和后，轨迹数量仍是一个有效的缩放轴；与已有工作本质区别在于把 trajectory count 提升为与参数/数据/token 同等地位的计算资源维度。
2. **提出 Trajectory Soup 双层融合框架**：先在内轨迹内按验证 loss 排序选 Top-K 取平均得分支锚点，再跨分支均匀平均；本质区别在于同时利用 intra- 与 inter-trajectory 两个层次的互补信息，而非仅靠 endpoint 或全量 checkpoint 池。
3. **提供局部偏差-方差分析严格解释双层平均的作用分工**：证明 intra-trajectory 平均消除的是沿轨迹的短期波动 $V_{\text{intra}}$，inter-trajectory 平均消除的是分支间持续误差 $V_{\text{inter}}$，且均匀权重在方差意义上是最优的；与已有经验性工作本质区别在于给出了可量化的误差分解。
4. **控制实验证实收益来自轨迹多样性而非采样密度**：在候选池大小与合并数量完全相等条件下，稀疏采样的多轨迹仍优于密集采样的单轨迹，明确了增益来源。

## 方法详解

### 基本设定
- 共享预训练 checkpoint $\theta_0$，目标分布 $\mathcal{P}$ 对所有分支相同。
- 第 $n$ 条分支使用配方 $\psi_n$（在 baseline 基础上扰动数据 shuffle seed / peak LR / global batch / schedule / optimizer 中的一个或多个），产出参数序列 $\theta_n(i)$。
- 兼容筛选：分支在 horizon $t$ 处的相对 training-loss gap 不超过 $\varepsilon = 0.01$。
- 总预算 $T = N \cdot t$（Limited）或 per-branch $t$ 固定（Extended）。

### 第一阶段：Intra-Trajectory Merging（分支锚点）
对每条分支 $n$，将所有保存的 checkpoint 按验证 accuracy（文中实际使用）升序排列，取前 $K$ 位后均匀平均：

$$
\bar{\theta}_n(t, K) = \frac{1}{K}\sum_{i \in \mathcal{T}_n(t,K)} \theta_n(i)
$$

其中 $\mathcal{T}_n$ 是验证集排名最高的 $K$ 个 checkpoint 索引集合。这一步消除单条轨迹内部的短期随机波动。

### 第二阶段：Inter-Trajectory Merging（最终模型）
将所有 $N$ 个分支锚点均匀平均：

$$
\theta_{\text{Traj-Soup}}(N,t,K) = \frac{1}{N}\sum_{n=1}^{N} \bar{\theta}_n(t,K)
$$

该模型保持原架构、tokenizer 和推理成本不变。

### 理论分析要点
- **局部二次损失模型**：$\mathcal{L}_Q(\theta) \doteq \mathcal{L}^\star + \frac{1}{2}\|\theta - \theta^\star\|_H^2$。
- **偏差-方差分解（Theorem 4.2）**：
  $$\mathbb{E}[\mathcal{L}_Q(\bar{\theta}_n)] - \mathcal{L}^\star = \underbrace{\frac{1}{2}\|\mathbb{E}[\bar{\theta}_n] - \theta^\star\|_H^2}_{B_n(K)} + \underbrace{\frac{1}{2}\mathbb{E}[\text{tr}(H\,\text{Cov}(\bar{\theta}_n\mid\mathcal{F}_n))]}_{V_{\text{intra},n}(K)} + \underbrace{\frac{1}{2}\text{tr}(H\,\text{Cov}(\mathbb{E}[\bar{\theta}_n\mid\mathcal{F}_n]))}_{V_{\text{inter},n}(K)}$$
  三项均非负，分别代表偏置、分支内方差、分支间方差。
- **分层方差衰减（Theorem 4.3）**：
  $$\mathbb{E}[\mathcal{L}_Q(\theta_{\text{Traj-Soup}})] - \mathcal{L}^\star = B_{\text{soup}}(K) + \frac{V_{\text{intra}}}{N\,K_{\text{eff}}(K)} + \frac{V_{\text{inter}}}{N}$$
  $K_{\text{eff}}(K)$ 为有效 checkpoint 数量（考虑时序相关性），其值 $\le K$。
- **均匀权重方差最优性（Corollary A.1）**：在分支独立且各 checkpoint 与所选集合具相同曲率加权协方差的条件下，内外两层均匀平均最小化随机项。
- **Selection 偏置**：随 $K$ 增大 bias $B_{\text{soup}}(K)$ 单调增加，方差项单调减少，存在内点最优 $K^\star_N$，且 $K^\star_N$ 随 $N$ 增大而减小。

## 实验与结果

**模型与设置**：Ling-3.0-Tiny（7.9B 总参数，1.3B 激活参数的稀疏 MoE），默认每条分支 horizon $t = 600\text{B}$ tokens，每 25B tokens 保存一个 checkpoint。

**基线方法**：Single EXP Merge（单条轨迹内 Top-K 平均）、Model Soup（跨分支仅取最终 checkpoint 平均）、Full Soup（无选择地平均所有候选）、Trajectory Soup。

**主要结果（Table 2，Mid-training 阶段）**：

| 方法 | Overall Average |
|---|---|
| Single-Trajectory Merge | 68.55 |
| Model Soup (Limited) | 68.43 |
| Model Soup (Extended) | 68.67 |
| Trajectory Soup (Limited) | 68.72 |
| **Trajectory Soup (Extended)** | **68.96** |

- Limited 设置（$N=3$，每支 200B tokens）下比最强单轨迹基线高 **+0.17**；Extended 设置（$N=3$，每支 600B tokens）下高 **+0.41**。
- **SFT 后优势保留（Table 3）**：Trajectory Soup (Extended) SFT 后 Overall Average 达 **61.52**，单条轨迹为 61.11，差值 **+0.41**。
- **小模型复现（Table 4，2B MoE + WSD 调度）**：Trajectory Soup (Extended) 49.64 / 50.07，优于 Single-Trajectory 49.55。
- **Scaling 趋势（Fig. 5）**：Trajectory Soup 的精度-计算前沿始终位于 Raw 曲线上方，且随 $N$ 增大持续上升（Raw 曲线在此区间弯曲下降）。
- **多样性 vs 采样密度控制实验（Fig. 7）**：在候选池大小与合并数严格匹配（M=12）时，Trajectory Soup 仍然全面超越 Dense Single-Trajectory Merge，证明增益来自轨迹多样性。
- **轨迹数缩放（Fig. 8）**：从 2→5 条轨迹，性能单调提升（68.79→69.08），边际递减；每支 Top-K 的最优窗口在 **10~16** 个 checkpoint 附近，超出后性能振荡或下降。
- **方向多样性预测插值增益（Fig. 9）**：两分支 PCA 主方向余弦相似度与插值增益 Pearson $r = -0.80$（$p = 0.005$），Spearman $\rho = -0.94$，方向越正交增益越大，可作为事前筛选指标。

## 相关工作脉络
1. **Scaling Laws（Kaplan et al., 2020; Hoffmann et al., 2022）**：刻画性能与参数/数据/计算的幂律关系；本文将其思想移植至 mid-training，增加"轨迹数"这一新的分配轴。
2. **Stochastic Weight Averaging (SWA, Izmailov et al., 2018)**：沿单条轨迹均匀平均 checkpoints；本文在其基础上增加跨分支维度，并给出偏差-方差理论解释。
3. **Model Soups (Wortsman et al., 2022)**：仅合并各分支的最终 endpoint；本文证明通过 intra-trajectory 筛选再跨分支合并远优于 endpoint-only 平均。
4. **Branch-Train-Merge / Branch-Train-MiX (Li et al., 2022; Sukhbaatar et al., 2024)**：并行训练专家后合并；本文不同在于所有分支共享同一数据分布，依靠 recipe 扰动产生方向多样性。
5. **Extra-Merge (Zhou et al., 2026)**：利用 late-stage 单条轨迹的近 rank-1 结构做外推；本文与之互补——在饱和区利用多条轨迹的互补方向而非单条轨迹的深层结构。
6. **WSM (Tian et al., 2026)**：通过 decay-free schedule + checkpoint merging 改善 pretraining；本文关注 mid-training 阶段，并将 selection + 双层平均理论化。

## 局限性与未来方向
1. **理论模型的适用范围**：偏差-方差分析建立在局部二次损失假设之上，对实际高维非凸 loss landscape 的保真度有限。
2. **计算会计不完整**：仅统计训练 token，未计入 checkpoint 存储、validation 开销、搜索超参的成本，以及不同并发度下的 accelerator 利用率差异。
3. **配方扰动选择依赖启发式**：当前通过 validation 事后筛选"足够多样且兼容"的分支，而非事前理论预测哪些扰动能产生可被平均消除的误差分量。
4. **数据 mixture 扰动未探索**：文中仅扰动调度超参与数据顺序，数据配比变化的效果留待未来工作。
5. **轨迹数/分支数的最佳配比有待自动学习**：目前 $N$ 和 $K$ 均通过 validation sweep 确定；作者提议训练过程中自适应调整分配策略。

## 研究启发与可借鉴点
1. **"计算分配轴"的扩展思路**：将 trajectory count 视为与 token 数、参数量并列的 scaling 维度，这一视角可迁移至 continued pretraining、domain adaptation、post-training 等多个场景。
2. **双层融合框架的结构化通用性**：内层去噪（intra）+ 外层聚合（inter）的分层平均设计，可推广至任何存在"多次独立训练启动"的设置（如多 seed 实验、多数据混合策略）。
3. **方向多样性作为事前筛选指标**：Fig. 9 中余弦相似度与插值增益的强负相关（$r=-0.80$）提示，可在训练前用低成本的方向预估来指导分支选择，避免盲目增加轨迹数。
4. **有效性保障的理论工具**：Corollary A.1 证明均匀权重在方差意义下最优，这为后续工作的消融实验提供了理论默认基线——除非有明确理由，否则不应引入复杂权重。
5. **小模型复现验证泛化性**：在 2B MoE + WSD 调度上的复现表明方法不依赖特定模型规模或调度，有利于团队在自己架构上快速验证可行性。

## 关键术语表
- **Trajectory Soup**：论文提出的双层 checkpoint 融合方法——先在每条轨迹内选 Top-K 平均得分支锚点，再跨分支均匀平均得最终模型。
- **Intra-Trajectory Merging**：单条优化轨迹内部对多个 checkpoint 的加权平均，用于平滑该轨迹的随机波动。
- **Inter-Trajectory Merging**：跨多条独立轨迹（不同配方）的平均，用于整合不同优化方向带来的互补信息。
- **Compatibility Screen（兼容筛选）**：以 relative training-loss gap ≤ 0.01 为阈值，过滤掉偏离过多可能导致融合劣化的分支。
- **Bias-Variance Decomposition（偏差-方差分解）**：将验证 loss 超额分解为 bias 项、intra-trajectory 方差项和 inter-trajectory 方差项，定量刻画两阶段平均各自的贡献。
- **Limited / Extended Budget**：Limited 指总 token 预算固定、分支均分；Extended 指每分支预算固定、累加分支数，用于区分计算匹配与计算扩展两种评估范式。
- **$K_{\text{eff}}(K)$（有效 checkpoint 数）**：考虑时序相关性和排名依赖后，实际等效的独立 checkpoint 数量，满足 $K_{\text{eff}} \le K$。
- **Directional Diversity（方向多样性）**：各分支在参数空间中主优化方向的余弦相似性；方向越正交，插值/平均带来的性能增益越大。

## 可复现要素
- **数据集**：Ling-3.0-Tiny mid-training corpus（论文未公开具体数据组成，仅注明使用高质量 mid-training 语料，见 Appendix D.4 列出的 41 个 benchmark）；代码/权重**论文未提及是否开源**。
- **关键超参**：每条分支 horizon $t = 600\text{B}$ tokens（默认），checkpoint 保存间隔 25B tokens，Top-K 默认 $K = 4$（三分支），兼容阈值 $\varepsilon = 0.01$，统一均匀权重，每支激活 8 个 expert 的 7.9B MoE 模型。
- **模型架构**：Ling-3.0-Tiny，24 层，hidden width 1536，128 routed experts / 8 active，MLA + KDA linear attention，QA KV LoRA rank 256/512，vocab size 157,184，sequence length 262,144。
- **训练设备与 FLOPs 估算**：36.74 GFLOPs/token，compute 轴使用 log 尺度拟合。
