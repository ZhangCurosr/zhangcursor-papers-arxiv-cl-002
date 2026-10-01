---
title: "USA-Update-aware-SAM-for-Cross-Domain-On-Policy-Distillation"
source: https://arxiv.org/pdf/2609.34225v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:23:32"
---

# 论文速读：USA-Update-aware-SAM-for-Cross-Domain-On-Policy-Distillation

## 一句话总结
本文针对多轮 Agent 后训练中单域 On-Policy Distillation (OPD) 早饱和与多域联合训练分布冲突的矛盾，提出 Update-aware SAM（USA）。该方法在每域独立训练的极短预热阶段测量参数更新幅度，将其转化为逐坐标扰动半径，使模型在合并前即适应跨域更新耦合带来的位移，从而在所有六条跨域转移方向上均超越单域基线并逆转负迁移。

## 研究问题与动机
- **单域 OPD 早饱和瓶颈**：Agent 推理模式多样性有限，仅靠扩大同域蒸馏语料无法持续提供新学习信号，性能很快进入窄幅波动平台。
- **多域混合训练的固有缺陷**：直接混合不同领域数据（mix-data）会导致分布冲突与优化信号相互干扰，且任一域的增删均需全量重训，工程灵活性差。
- **模型合并的负迁移现象**：独立训练各域 expert 后再参数空间合并可避开分布冲突，但在部分域对（如 Math ↔ Science）上所有评估的合并算子均低于单域基线。
- **根因诊断：跨域更新耦合**：冲突对的参数层分析显示，大量坐标被多个域以相近幅度更新（耦合分布右移），线性合并时其他域的注入项在该坐标产生的位移幅度可与本域更新相当，导致合并干扰不可控。

## 核心贡献（创新点）
- **揭示跨域更新耦合为负迁移的结构性根因**：指出该耦合在独立训练阶段已悄然形成，与后续合并算子选择无关，打破了“换合并公式即可解决冲突”的直觉。
- **提出 Update-aware SAM（USA）**：将 OPD 预热步的每参数更新幅度映射为逐坐标扰动半径，差异化分配平坦性约束预算，使高更新显著性参数在合并位移下具备更强的容错性。
- **给出合并干扰的曲率-位移分解上界**：证明合并后目标域的损失增量上界正比于 $\lambda_{\max}(S H_t S) \|S^{-1}\delta_t\|_2^2$，且 USA 的训练目标在二阶意义下直接最小化该上界的曲率因子。
- **系统验证跨尺度与跨环境的泛化性**：在数学/科学/代码三域、1.7B/4B 双学生尺度、四种合并算子及检索 Agent 环境中均取得最强结果，且无需针对新环境重调超参。

## 方法详解
- **预热与更新幅度度量**：每个域 $k$ 从共享初始化 $\theta_0$ 出发，用标准 OPD 目标训练 $N$ 步（论文取 $N=10$），记录每参数更新幅度 $m_k(i) = |\theta_N^{(k)}(i) - \theta_0(i)|$。OPD 的有效更新高度稀疏且早期锁定，故短预热即可刻画更新显著的参数轮廓。
- **逐坐标缩放因子构造**：在每个权重矩阵 $g$ 内取 $m_k(i)$ 的 $(1-p)$-分位数（$p=1\%$）作为参考 $D_k(g)$，定义 $s_k(i) = 1 + (\alpha-1)\min(m_k(i)/D_k(g_i), 1)$，其中放大上限 $\alpha=5$。该构造对更新整体缩放不变，参考取矩阵内分位数避免重尾分布被极端值主导，且呈连续映射无需硬阈值。
- **更新感知 SAM 目标**：预热后固定 $S_k=\mathrm{diag}(s_k)$，后续每步求解 $\min_\theta \max_{\|S_k^{-1}\epsilon\|_2 \leq \rho} \mathcal{L}_k(\theta+\epsilon)$，约束为轴对齐椭球，第 $i$ 轴半长 $\rho s_k(i)$。线性化后闭式扰动为 $\hat{\epsilon}_k = \rho S_k^2 \nabla_\theta \mathcal{L}_k / \|S_k \nabla_\theta \mathcal{L}_k\|$，仅在扰动点计算梯度完成更新，不改变参数本身。
- **理论联系**：合并干扰 $\Xi_t = \mathcal{L}_t(\theta_t+\delta_t)-\mathcal{L}_t(\theta_t)$ 的二阶上界为 $\frac{1}{2}\lambda_{\max}(S H_t S)\|S^{-1}\delta_t\|_2^2 + O(\|\delta_t\|^3)$。USA 增大 $s_k(i)$ 会同步压低曲率因子并压缩重加权位移，使得训练目标与合并性能在同一个 $S$-度量下对齐；$\alpha$ 的上界由该乘积权衡决定，防止曲率因子反噬。

## 实验与结果
- **数据集与基准**：训练数据为 DAPO-Math、Skywork-OR1 (Code)、MegaScience 各 4k 条；评测基准包括 AIME 2025、HMMT Feb 2026、LiveCodeBench-v6、NaturalCodeBench、GPQA-Diamond、SciBench-Atkins。学生模型 Qwen3-1.7B/4B，教师为 Qwen3-14B（GRPO 优化）。
- **主要结果**：USA 在 1.7B 与 4B 全部 6 条转移方向上均位列第一，平均超越单域 OPD 基线 4.55（1.7B）/ 4.01（4B）个百分点。在冲突对 Math → Science 上，最优合并算子（Weight Average）低于基线 2.37 分，而 USA 高出基线 5.08 分，实现负迁移逆转。
- **消融结论**：
  - 自适应平坦性（USA）较各向同性 Vanilla SAM 平均再高 1.75 分，证实预算的坐标分配比约束本身更关键。
  - 随机打乱或反转尺度向量均导致性能跌破各向同性约束，证实增益来自幅度与尺度的因果对齐。
  - 基于参数绝对幅度的指标等效于各向同性，说明应关注“更新位移”而非“参数大小”。
  - 与 TIES/DARE/AdaMerging 搭配时 USA 依然全面领先，最弱者（DARE）在 USA 加持下平均提升 5.39 分。
- **开销与扩展**：1.7B 训练时间从 44h 增至 50h（+13.6%），4B 为 +11.8%；扩展至 5 个源域时 USA 增益单调上升而 Weight Average 在 2-3 源后触顶回落；更换为检索 Agent 环境（NQ/HotpotQA/PopQA）后超参无需重调即保持最优排序。

## 相关工作脉络
- **On-Policy Distillation**：Agarwal et al. (2024)、Gu et al. (2024) 及近年的 Agent 后训练工作（Zhong et al. 2026、MOPD）均以
