---
title: "Trajectory-Soup-Pushing-the-Compute-Scaling-Frontier-of-LLM"
source: https://arxiv.org/pdf/2609.37169v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:56:55"
field: "大语言模型训练效率与scaling"
keywords: ["mid-training", "model merging", "trajectory diversity", "weight averaging", "compute scaling", "checkpoint fusion"]
innovations: ["提出Trajectory Soup双层合并框架，同时利用轨迹内筛选与轨迹间融合突破mid-training计算饱和", "将轨迹数量作为独立计算分配轴，证明扩展并行轨迹可持续提升下游性能", "给出局部偏差-方差理论分解，证明uniform averaging在两层均为方差最优"]
benchmarks: ["ARC-Easy/Challenge", "AGIEval", "MMLU/MMLU-Pro", "GSM8K/MATH", "HumanEval/MBPP/LiveCodeBench", "IFEval/BFCL"]
---

# 论文速读：Trajectory-Soup-Pushing-the-Compute-Scaling-Frontier-of-LLM

## 一句话总结
论文提出 **Trajectory Soup**，将 mid-training 的计算预算分配给多条从共同 checkpoint 分叉的独立优化轨迹，在各轨迹内按验证集筛选 Top-K 最佳 checkpoint 并平均，再跨轨迹平均融合，从而突破串行 mid-training 的计算饱和墙，在匹配计算预算下持续提升下游性能，且优势可传递至 SFT 阶段。

## 研究问题与动机
- **mid-training 计算饱和问题**：与 pretraining 不同，单纯增加 mid-training token 数量并不能持续改善下游性能；Figure 1a 显示性能先快速上升后趋于饱和，继续训练甚至可能衰退。
- **串行扩展收益递减的根源**：固定模型容量下重复使用有限语料存在边际收益递减（Muennighoff et al., 2023），且单条轨迹内的随机优化波动虽可通过 SWA 缓解，但一旦局部波动被衰减，额外 checkpoint 的噪声降低效果急剧下降。
- **轨迹多样性未被充分利用**：现有方法要么只做单条轨迹内 checkpoint 平均（SWA 等），要么只做多轨迹 endpoint 平均（Model Soup），但两者结合是否能在同一预算下产生互补增益尚不明确。
- **计算分配视角有待探索**：受 scaling law 启发，作者将"轨迹数量"作为一个与"单条轨迹长度"并列的计算分配轴，探讨在多轨迹并行 vs. 单轨迹串行之间的最优权衡。

## 核心贡献（创新点）
1. **提出 Trajectory Soup 双层合并框架**：先从共享 checkpoint 分叉多条独立轨迹，各轨迹内按验证 loss 选择 Top-K checkpoint 平均得 branch anchor，再跨轨迹均匀平均合成单一模型——与已有工作（仅 intra 或仅 inter）的本质区别在于同时利用轨迹多样性与 checkpoint 质量筛选。
2. **重构 mid-training 计算分配视角**：将 trajectory count 视为与参数规模、token 数量同级的 scaling axis，证明在单条轨迹达到饱和后，增加并行兼容轨迹数仍可持续提升下游性能。
3. **给出局部偏差-方差理论分解**：Theorem 4.2/4.3 将验证 loss 拆分为 bias（共享偏移 + recipe 特定偏移 + 选择偏差）、intra-trajectory variance、inter-trajectory variance 三部分，严格证明 uniform averaging 在两个层面均为方差最优，且预测存在有限的最优 checkpoint 合并数量。
4. **系统性实验验证与可扩展性分析**：在 Ling-3.0-Tiny（7.9B MoE）和 2B 小模型上验证，扩展 trajectory 数量可稳步提升性能上限（68.79→69.08），且每轨迹仅需合并约 10–16 个 checkpoint 即可逼近最优。

## 方法详解
**整体流程（Algorithm 1 / Figure 4）**：
1. 从共享 pretrained checkpoint $\theta_0$ 出发，对训练分布 $\mathcal{P}$ 并行运行 $N$ 条独立分支，每条分支 $\psi_n$ 通过扰动一条超参维度（数据 shuffle seed / peak LR / batch size / LR schedule / optimizer momentum）生成，但需满足兼容性筛选：相对训练 loss 差距 $\leq \varepsilon=0.01$。
2. **Step 1（Intra-Trajectory Merging）**：对每条分支 $n$，按验证 loss 对保存的 checkpoint 排序，取前 $K$ 个位置 $\mathcal{I}_n(t,K)$，均匀平均得 branch anchor：
$$\bar{\theta}_n(t, K) = \frac{1}{K}\sum_{i \in \mathcal{I}_n(t,K)} \theta_n(i)$$
3. **Step 2（Inter-Trajectory Merging）**：对所有分支 anchor 做均匀平均：
$$\theta_{\text{Traj-Soup}}(N, t, K) = \frac{1}{N}\sum_{n=1}^{N} \bar{\theta}_n(t, K) = \frac{1}{NK}\sum_{n=1}^{N}\sum_{i \in \mathcal{I}_n(t,K)} \theta_n(i)$$
4. 最终模型保持原有架构、tokenizer 和推理流程不变。

**两种预算设置**：
- **Limited（匹配预算）**：总 token 预算 $T = Nt$ 固定，每条分支获得 $T/N$ 个 token，与单条跑满 $T$ token 的基线公平比较。
- **Extended（扩展预算）**：每条分支已跑满 $t$ token，逐步增加 $N$ 以扩大总计算量，检验能否将额外预算转化为性能提升。

**理论分解（Theorem 4.2/4.3）**：
在局部二次验证 loss 模型 $\mathcal{L}_Q(\theta) \doteq \mathcal{L}^* + \frac{1}{2}\|\theta - \theta^*\|_H^2$ 下，Trajectory Soup 的期望 excess loss 分解为：
$$\mathbb{E}[\mathcal{L}_Q(\theta_{\text{Traj-Soup}})] - \mathcal{L}^* = B_{\text{soup}}(K) + \frac{V_{\text{intra}}}{NK_{\text{eff}}(K)} + \frac{V_{\text{inter}}}{N}$$
其中 $B_{\text{soup}}(K)$ 为 selection-dependent 偏差项（随 $K$ 增大而增大），$V_{\text{intra}}/K_{\text{eff}}(K)$ 为轨迹内方差（随 $K$ 增大而减小），$V_{\text{inter}}/N$ 为跨轨迹方差（不受 $K$ 影响）。三者间的权衡解释了为何存在有限的优选 $K$，且 $K^*$ 随 $N$ 增大而递减。

## 实验与结果
- **模型**：Ling-3.0-Tiny（稀疏 MoE，7.9B 总参数，1.3B 激活参数/ token，24 层，128 experts/layer，8 active），基于 30T token pretraining checkpoint 进行 mid-training。
- **基线**：Single EXP Merge（单条轨迹 Top-K 平均）、Model Soup（仅跨轨迹 endpoint 平均）、Full Soup（无筛选的全量平均）。
- **主要结果（Table 2，mid-training 阶段）**：
  - Trajectory Soup (Limited) Overall Average **68.72**，优于 Single-Trajectory Merge（68.55）和 Model Soup (Limited)（68.43）。
  - Trajectory Soup (Extended) Overall Average **68.96**，进一步提升，优于 Model Soup (Extended)（68.67）。
  - 优势在各能力类别均有体现，尤其是 Math（+0.33/−0.22 区间）、Code（+0.83 提升最大）。
- **SFT 后性能保留（Table 3）**：Trajectory Soup (Extended) 在统一 SFT 后仍保持最优，Overall Average **61.52**，vs. Single-Trajectory 61.11，证明 mid-training 阶段的优势可端到端传递。
- **小模型验证（Table 4，2B MoE + WSD 调度）**：Trajectory Soup (Limited) **49.64** vs. 最强 intra 基线 49.55；(Extended) **50.07**，再次复现主趋势。
- **扩展分析**：增加 trajectory 数量从 2→5 使性能从 68.79 升至 69.08；最优 merge 规模集中在每轨迹 **10–16 个** checkpoint，超过此范围性能开始震荡或下降（Figure 8）。
- **方向多样性预测增益（Figure 9）**：两轨迹间 leading 方向余弦相似度与插值增益呈强负相关（Pearson $r=-0.80$, $p=0.005$；Spearman $\rho=-0.94$），即方向越正交，合并收益越大。

## 相关工作脉络
1. **Stochastic Weight Averaging (SWA, Izmailov et al., 2018)**：单条轨迹内 checkpoint 平均的经典方法；本文在此基础上引入跨轨迹合并，解决 SWA 在单条轨迹饱和后的收益递减问题。
2. **Model Soups (Wortsman et al., 2022)**：多模型 endpoint 均匀平均；本文区分了"仅 endpoint 平均"与"quality-aware 筛选后平均"，证明前者无法充分利用轨迹多样性。
3. **Extra-Merge (Zhou et al., 2026)**：利用 late-stage 轨迹近似 rank-1 结构做外推；本文不依赖该低秩假设，而是通过偏差-方差分解给出更一般性的合并理论。
4. **Branch-Train-Merge / Branch-Train-MiX**：并行训练专家语言模型后 ensemble；本文方法无需额外专家路由，直接将多条轨迹融合为单一稠密/稀疏模型，推理成本不变。
5. **WSM (Tian et al., 2026)**：无 decay 的 LR schedule 配合 checkpoint 合并；本文的轨迹分配视角可与其 complement，构成更完整的 mid-training scaling 策略。
6. **Compute-Optimal Scaling Laws (Hoffmann et al., 2022)**：平衡参数量与 token 数量；本文扩展了 scaling law 的维度，将"轨迹数"纳入计算分配决策空间。

## 局限性与未来方向
- **计算账目不完整**：当前只统计训练 token 消耗，未计入 checkpoint 存储、验证 pass、超参搜索等开销；不同加速器利用率（长串行 vs. 短并行）也未纳入比较。
- **理论模型假设限制**：偏差-方差分析基于局部二次 loss 模型，实际 loss landscape 的非二次性未被覆盖；几何诊断（余弦相似度、插值曲线）仅为事后观测，无法在训练前预测哪些 recipe 扰动能产生可被平均消除的错误分量。
- **强相关轨迹存在误差下限**：高度相关的轨迹共享相同误差，平均无法消除，且过度激进的扰动会使分支落入合并后比成员更差的区域。
- **超参仍依赖手工搜索**：$N$、$K$、$\varepsilon$ 等关键超参目前通过验证搜索确定；未来可在训练中自适应调整轨迹分配与 checkpoint 筛选策略。

## 研究启发与可借鉴点
1. **计算分配维度的拓展**：将 trajectory count 作为独立的 scaling axis 值得推广至 continued pretraining、domain adaptation 及 long post-training pipelines 等场景，形成统一的并行探索→单模型整合范式。
2. **偏差-方差分解框架**：Theorem 4.2/4.3 的三层分解（bias / intra-var / inter-var）为分析任意 checkpoint 合并策略提供了可复用的理论工具，可用于指导其他合并方法的设计。
3. **方向多样性作为选择准则**：Figure 9 发现的余弦相似度与合并增益的强负相关，提供了一个无需额外训练的廉价分支选择指标，可集成到自动化 trajectory 调度管线中。
4. **简单即有效的消融结论**：uniform averaging + Top-K selection + symmetric allocation 的组合在所有 ablation 中表现最佳，提示后续工作应避免过度复杂的权重设计，聚焦于轨迹多样性的系统性利用。
5. **扩展预算下的单调收益**：Extended 设置下性能随轨迹数增加持续上升，说明只要找到足够多样且兼容的分支，计算投入就有稳定回报，为工程上设计弹性 mid-training pipeline 提供了依据。

## 关键术语表
- **Mid-training**：pretraining 与 post-training（SFT）之间的训练阶段，用于赋予 LLM 特定领域知识与推理能力。
- **Trajectory Soup**：本文提出的双层 checkpoint 合并方法，先 intra-trajectory 筛选平均，再 inter-trajectory 平均融合。
- **Intra-trajectory averaging**：在同一优化轨迹内对多个 checkpoint 做均匀平均，衰减轨迹内的随机波动。
- **Inter-trajectory averaging**：在不同优化轨迹之间对 anchor 做均匀平均，整合互补的优化方向。
- **Effective checkpoint count ($K_{\text{eff}}$)**：衡量轨迹内 checkpoint 冗余程度的标量，考虑了时间相关性和 ranking 诱导的依赖性，取值 $\leq K$。
- **Compatibility screen ($\mathcal{I} \leq \varepsilon$)**：分支准入条件，要求相对训练 loss 差距不超过阈值（本文取 0.01），确保各分支优化质量可比。
- **Local quadratic model**：在局部极小点附近用二阶泰勒展开近似验证 loss 的模型，支撑偏差-方差分解的理论推导。
- **Directional diversity**：不同轨迹在参数空间中主导优化方向的差异程度，用 PCA 第一主成分的余弦相似度量化。

## 可复现要素
- **数据集**：论文未公开 mid-training 专用数据集名称，使用高质量 mid-training corpus（含 41 个 benchmark 配置的评估体系），具体数据来源未详细披露。
- **代码/权重**：基础模型 Ling-3.0-Tiny 已公开于 HuggingFace（inclusionAI, 2026）；Trajectory Soup 代码未明确声明开源状态。
- **关键超参**：
  - 每分支训练 token 数：$t = 600\text{B}$（default），checkpoint 保存间隔 25B tokens
  - 兼容性阈值 $\varepsilon = 0.01$
  - 最优每分支 checkpoint 数 $K \approx 4$（3 分支 Limited）至约 10–16 个（Extended）
  - 模型序列长度：262,144；每个 token 估算 FLOPs：36.74 GFLOPs
  - Optimizer：Muon，Weight decay：0.1，Warmup：1%
