---
title: "X-MOD-PRACTICAL-SCALING-LAWS-FOR-SPARSE-DEPTH-ROUTING-BEYOND"
source: https://arxiv.org/pdf/2609.34212v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:27:35"
field: "大语言模型高效架构与缩放定律"
keywords: ["sparse-depth routing", "mixture-of-depths", "scaling laws", "conditional computation", "transformer architecture", "Mixture-of-Experts", "token sparsity"]
innovations: ["提出X-MoD架构，将token稀疏度K与锚点步长A解耦，突破MoD总/激活参数比<2的限制", "建立基于FLOP匹配稠密基线残差的稀疏深度缩放定律，三项分解解释容量增益、上下文修正与锚点交互", "验证X-MoD在936M规模下较MoE基线实现1.70×训练吞吐和1.42×推理吞吐提升"]
benchmarks: ["FineWeb-Edu", "HellaSwag", "ARC-Easy", "ARC-Challenge", "PIQA", "LAMBADA", "BoolQ"]
---

# 论文速读：X-MOD: PRACTICAL SCALING LAWS FOR SPARSE-DEPTH ROUTING BEYOND MIXTURE-OF-DEPTHS

## 一句话总结
X-MoD 提出了一种可扩展的稀疏深度 Transformer 架构，将 Token 稀疏度 K 与锚点步长 A 解耦，使总参数量可以在激活等效容量基本固定的情况下大幅增长；同时建立了一套基于 FLOP 匹配的缩放定律框架，能够定量预测不同路由配置下的验证损失并指导超参选择。

## 研究问题与动机
- **MoD 的总容量-激活容量耦合瓶颈**：原始 Mixture-of-Depths (MoD) 采用严格的"一稀疏一稠密"交替结构，总参数量与激活等效参数量之比恒小于 2（式 (2)），无法像 MoE 那样实现总容量与每样本计算量的解耦，限制了预训练可扩展性。
- **深层稀疏路由的可训练性问题**：在稠密锚点之间堆叠大量稀疏层会导致优化不稳定和 token 路由失衡（token collapse，即少数 token 被反复选中），需要稳定的训练机制。
- **稀疏深度架构缺乏设计法则**：给定计算预算、上下文长度和激活等效容量，如何选择稀疏度 K 和锚点步长 A 以最小化损失，目前仍是纯启发式探索，缺乏可分析的缩放定律指导。

## 核心贡献（创新点）
- **架构解耦**：提出 X-MoD，将原始 MoD 中紧耦合的 token 稀疏度 K 和锚点步长 A 变为独立可控的自由度，使总/激活等效参数比随 K 近似线性增长，突破了 MoD <2 的结构上限。
- **三层稳定化机制**：设计了方差缩放的逐层门控（替代 MoD 仅门控 FFN 残差）、深度方向 token 平衡偏置（防止连续稀疏层反复选择相同 token 子集）、以及非对称层级（前置稠密层 + 后端稀疏深度）。
- **FLOP 匹配缩放定律**：将稀疏深度路由形式化为条件架构设计问题，构造相对于 FLOP 匹配稠密基线的残差，分解为稀疏容量增益、稀疏上下文修正、锚点步长交互三项，$R^2_\Delta = 0.9853$，支持组外预测。
- **系统性能验证**：在 936M 激活等效规模下，X-MoD 相比总参数相近的 MoE 基线实现了 1.70× 训练吞吐和 1.42× 推理吞吐提升。

## 方法详解
**架构设计（式 3）**：
$$[Dense]_{\times N_0} + ([Sparse(K)]_{\times AK} + [Dense])_{\times N_1}$$
其中 $N_0$ 为初始稠密前缀，$N_1$ 为稀疏-稠密块数，$A$ 为锚点步长。每个块内 $AK$ 个稀疏层、每层激活 $1/K$ 的 token，平均每 token 获得 $A$ 次稀疏更新和 1 次稠密锚点更新。总参数（式 4）与激活等效参数（式 5）之比为 $\frac{N_0 + N_1(1+AK)}{N_0 + N_1(1+A)}$，固定 A 时随 K 近似线性增长。

**稳定化机制**：
1. **方差缩放逐层门控**（式 7）：学习标量 $\varsigma^\ell$ 对完整残差（attention + FFN）进行门控，而非 MoD 仅对 FFN 残差门控（式 6），稳定了残差方差。
2. **深度方向 token 平衡**（式 8）：路由 logit 减去偏置项 $\tau b_i$，$b_i$ 在 token 被选中时递增、在下一个稠密锚点处重置，抑制重复选择但不强制均匀路由。
3. **非对称层级**：保留前 $N_0$ 层为全稠密，稀疏路由仅作用于深层，符合早期层负责稳定词汇/句法表示的直觉。

**缩放定律（式 14）**：
定义残差 $\Delta = \mathcal{L}_{X\text{-}MoD} - \widehat{\mathcal{L}}_{dense}$，拟合得到三项分解：
- **稀疏容量增益** $R_N = N_{act}^{-\eta_N}[1-(N(K,A)/N_{act})^{-\rho_N}]$，体现总参数增加带来的收益；
- **稀疏上下文修正** $P_L = L^{-\eta_L}(K^{\rho_L}-1)$，刻画稀疏层仅看到 $L_s = L/K$ 个 token 时的惩罚；
- **锚点步长交互** $R_{A,K} = (K^{\rho_K}-1)[(2A/(A+1))^{\rho_A}-1]$，捕捉 A 与 K 的协同效应。
最优稀疏度近似为 $\hat{K} \propto L^{\eta_L/(\rho_N+\rho_L)} N_{act}^{-\eta_N/(\rho_N+\rho_L)}$（式 16），预测 K 与上下文长度正相关、与模型规模弱负相关。

**路由对齐**：训练使用非因果 top-k 路由，评估使用因果阈值规则（prob ≥ 0.5），通过 BCE 辅助损失（式 42）对齐两者，系数取 $10^{-4}$。

## 实验与结果
**数据集**：FineWeb-Edu，GPT-2 tokenizer。评估任务：HellaSwag、ARC-Easy、ARC-Challenge、PIQA、LAMBADA、BoolQ（zero-shot）。

**基线**：Dense、MoD（1-sparse-1-dense, $K=8$）、MoE 8:64、MoE 8:128。

**主要结果（Table 1，按 matched $N_{act}$ 和 FLOPs 对齐）**：
- 556M 规模：X-MoD A3K16（总参数 4.79B，$\phi/\phi_D = 0.46$）验证损失 **2.503**，优于 Dense(2.773)、MoD(2.692)、MoE 8:64(2.564) 和 MoE 8:128(2.528)，下游平均 **48.64** 为最高。
- 936M 规模：X-MoD A3K16（8.24B，$\phi/\phi_D=0.50$）验证损失 **2.402**，优于 Dense(2.635)、MoD(2.570)、MoE 8:64(2.445)、MoE 8:128(2.412)，下游平均 **51.84**。
- 1.65B 规模：X-MoD A3K16（14.73B，$\phi/\phi_D=0.56$）验证损失 **2.322**，优于所有对照，下游平均 **53.52**。

**缩放定律验证**：109 个 in-range 观测点的 $R^2_\Delta = 0.9853$；留一组交叉验证（按 L、A、$N_{act}$ 分组）预测误差小；$L=64k$ 和 $N_{act}=1.65B$ 两个 out-of-range 扩展亦保持定性趋势一致。

**消融结论**：移除 dense anchors（+0.103 loss）、移除 token-choice bias（+0.181 loss，最大退化）、移除 gated scaling（+0.016 loss）、移除 dense prefix（+0.011 loss）。Routed-FFN control 证明条件容量增益本身有效，但 Selected-Q/full-KV 表明稀疏层注意力上下文选择比扩展 KV 更经济。

**系统效率（Table 11）**：X-MoD A4K8 较 MoE 8:64 训练吞吐提升 **1.93×**，推理吞吐提升 **1.73×**；X-MoD A3K16 较 MoE 8:128 提升 **1.70× / 1.42×**。

## 相关工作脉络
- **Mixture-of-Experts (MoE)**：Shazeer et al. (2017)、Lepikhin et al. (2021)、Fedus et al. (2022)、Dai et al. (2024)。MoE 在宽度维度解耦总/激活容量；X-MoD 在同理心下在深度维度实现同等解耦，且每 token FLOP 显著更低。
- **Mixture-of-Depths (MoD)**：Raposo et al. (2024)。原始 MoD 严格 1-sparse-1-dense 交替，总/激活参数比 <2；X-MoD 将其推广为可独立调节 K 和 A 的稀疏深度层次结构。
- **MoE 缩放定律**：Clark et al. (2022)、Abnar et al. (2025)、Krajewski et al. (2024)、Tian et al. (2026) 等。本文延续"相对于 FLOP 匹配稠密基线的残差建模"思路，但首次将其应用于稀疏深度路由。
- **稀疏注意力**：Piąkos et al. (2025)、Xiao et al. (2024)、Yuan et al. (2025) 等。这些工作聚焦减少/复用计算，而 X-MoD 研究在固定激活等效预算下如何扩展总稀疏深度容量。
- **多模态 MoD 变体**：Lin et al. (2024)、Zhang et al. (2026)、Luo et al. (2025) 等在多模态场景应用条件深度；本文专注预训练时间稀疏深度可扩展性，不涉及多模态。
- **推理阶段 token 跳过**：Elhoushi et al. (2024)、Fan et al. (2025)、Jiang et al. (2024) 等用于下游预算缩减；X-MoD 面向预训练阶段设计，与这些方法互补。

## 局限性与未来方向
- **单一机制研究**：X-MoD 仅研究稀疏深度一种条件计算形式，未与稀疏宽度（MoE）、循环深度（Loop Transformers）或推理 token 剪枝结合；其组合效应是 additive 还是 nonlinear 仍是开放问题。
- **规模外推未验证**：缩放定律在 165M–1.65B 范围拟合，生产级大模型规模（如数十 B/数百 B）下最优 K 对上下文长度和模型的依赖是否依然成立尚待验证。
- **系统优化未充分展开**：连续稀疏层处理部分不相交 token 子集，dense anchors 提供天然同步点，routing 决策可跨深度复用或流水线化，高效 kernel 和并行方案尚未开发。
- **out-of-range 预测精度有限**：虽然趋势一致，但 $L=64k$ 和 $N_{act}=1.65B$ 的预测未重新拟合，实际误差需进一步评估。

## 研究启发与可借鉴点
- **残差建模范式**：相对于 FLOP 匹配稠密基线的残差设计（$\Delta = \mathcal{L}_{sparse} - \widehat{\mathcal{L}}_{dense}$）是隔离稀疏特定增益/代价的优雅方法，可迁移到其他稀疏架构（如稀疏宽度、混合稀疏）的缩放定律研究。
- **路由对齐技巧**：top-k 训练 + 阈值评估之间的 BCE 辅助损失（式 42）以低开销弥合 train-eval gap，对任何需要部署时因果阈值路由的稀疏架构均有参考价值。
- **深度方向 token 平衡机制**：递增式偏置 $b_i$ + 锚点重置的设计简洁有效，可推广到任意需要在多层中维持 token 分布多样性的稀疏路由系统。
- **与 MoE 的横向对比策略**：在 matched $N_{act}$ 和 FLOPs 双重约束下选取 best-of-group 的 X-MoD 与 MoE 配置进行对比（Fig. 4 pilot sweep），公平且具有说服力，可作为后续工作的实验设计模板。
- **创新机会**：团队可将 X-MoD 的缩放定律思路与 MoE 的缩放定律（Abnar et al. 2025）结合，探索稀疏深度+稀疏宽度的联合缩放行为；或将其应用于多模态场景（当前 MoD 多模态变体尚未引入此架构）。

## 关键术语表
- **Mixture-of-Depths (MoD)**：沿 Transformer 深度维度进行条件计算，仅将选中 token 送入特定层，其余 token 经恒等路径跳过。
- **X-MoD**：本文提出的可扩展稀疏深度架构，独立控制 token 稀疏度 K 和锚点步长 A，打破 MoD 总/激活参数比 <2 的限制。
- **Anchor stride (A)**：两个相邻稠密锚点之间平均每 token 获得的稀疏更新次数，控制稀疏精炼与稠密同步的配比。
- **Token sparsity (K)**：每个稀疏层激活的 token 比例，为 top-$1/K$ 分数最高的 token 通过稀疏层。
- **Active-equivalent parameter count ($N_{act}$)**：每 token 平均 traversed 的参数数量，作为模型实际激活容量的度量。
- **FLOP-matched residual**：稀疏模型验证损失减去 FLOP、上下文长度和 backbone 家族均匹配的稠密基线损失，用于剥离稀疏深度专属效应。
- **Token-choice bias**：深度方向上累积的路由偏置 $b_i$，选中递增、锚点重置，防止 token collapse。
- **Gated residual scaling ($\varsigma^\ell$)**：逐层可学习标量，对完整残差（attention + FFN）进行门控缩放，稳定稀疏层残差方差。

## 可复现要素
- **数据集**：FineWeb-Edu（公开），GPT-2 tokenizer。
- **代码/权重**：论文未明确声明代码或权重开源。
- **关键超参**：sparsity K ∈ {4, 8, 12, 16}，anchor stride A ∈ {0.5, 1, 2, 3, 4, 5}，context length L ∈ {2k, 4k, 8k, 16k, 32k, 64k}，dense prefix $N_0=4$，GQA（16 query heads, 8 KV heads），Muon optimizer（peak LR $3.2 \times 10^{-3}$, $\beta=0.95$, Newton-Schulz steps=5），warmup 1%、min LR 为 peak 的 10%，global batch × L ≈ 2.5M tokens。
