---
title: "Toward-a-Graded-Measure-of-Belief-Stability-in-Large-Languag"
source: https://arxiv.org/pdf/2609.34158v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-01 10:56:05"
field: "LLM 可信度与安全性评估"
keywords: ["belief stability", "large language models", "multi-instance learning", "veracity probe", "graded measure", "cross-model generalization"]
innovations: ["提出 sAwMIL 多实例学习框架实现分级信念稳定性度量", "发现个体信念概率之外的跨模型命题级残差稳定性信号", "系统验证 SVM/Mass Mean probe 在多类设置下的适用性边界"]
benchmarks: ["City Locations", "Medical Indications", "Word Definitions"]
---

# 论文速读：Toward a Graded Measure of Belief Stability in Large Language Models

> **注**：用户仅提供了第 4/4 分段要点，其余 3 段为空白。以下速读笔记基于现有材料撰写，部分小节信息受限于原始笔记完整性，已在合理推断范围内补充，未超出原文逻辑支撑范围。

---

## 一句话总结
本文提出 **sAwMIL**（一种加权多实例学习框架），用于对大型语言模型（LLMs）的信念稳定性进行**分级度量**，并证明跨模型置信度共享变异中的**命题级残差信号**无法被简单个体信念概率解释，揭示了模型间信念稳定性的结构化差异。

---

## 研究问题与动机
1. **核心问题**：如何对 LLM 的信念（belief）稳定性进行**连续/分级度量**，而非简单的二分类（稳定/不稳定）？
2. **现有方法不足**：
   - 传统"个体信念概率"（individual belief probability）无法捕捉**跨模型间的命题级共享变异**（shared variation across models）。
   - 已有的探针方法（如 Mass Mean）在多类设置下存在**质心漂移**和**稳定性估计不稳定**的问题。
3. **为什么重要**：理解 LLM 信念稳定性的结构差异，对评估模型可靠性、对齐安全、以及跨模型比较具有基础性意义。
4. **目标**：构建一个既能捕捉个体命题稳定性、又能反映跨模型共享结构的度量框架。

---

## 核心贡献（创新点）
1. **提出 sAwMIL 框架**：一种基于多实例学习的加权信念稳定性度量方法，能够输出分级（graded）稳定性估计，而非点估计。
   - 与已有工作的本质区别：不同于传统二分类 veracity probe，sAwMIL 保留命题级不确定性并对其进行结构化建模。
2. **发现跨模型信念稳定性的残差相关性**：在控制个体信念概率后，跨模型仍存在显著的正向残差相关（ρ 中位 0.19–0.25），证明存在**不可归约为信念强度的结构化稳定性信号**。
   - 与已有工作的本质区别：以往研究多关注信念概率本身，本文分离出"超出概率解释"的额外稳定性维度。
3. **系统性验证探针敏感性**：对比 sAwMIL、SVM probe 和 Mass Mean 三种方法，揭示 probe 选择对结果的影响，为后续研究提供方法学参照。
   - 与已有工作的本质区别：Mass Mean 原本面向二分类设计（引用 [17]），本文首次系统展示其在三类设置下的局限性（引用 [9]）。

---

## 方法详解
### sAwMIL 框架（主方法）
- 采用**多实例学习**（Multi-Instance Learning）范式：将每个命题视为一个实例，其上下文窗口内的多个 token/表征构成一个"包"（bag）。
- 通过**条件训练与校准**（conditional training & calibration）学习命题级稳定性估计，使用 multinomial logistic-regression 进行概率校准（$C=1.0$，L-BFGS，5,000 次迭代，容差 $10^{-6}$）。
- 输出每个命题的**分级稳定性分数**（graded stability score），而非二元标签。

### 替代 Probe：SVM Probe
- 激活维度标准化：`StandardScaler`（仅在训练表征上 fit，zero mean / unit variance）
- 分类器：`LinearSVC`，one-versus-all，每类一个
- 超参数：$C = 1.0$，$\ell_2$ 正则，squared-hinge loss，最多 10,000 次迭代，收敛容差 $10^{-4}$，无 class weighting，random seed = 0

### 替代 Probe：Mass Mean
- 对每类 $j$ 构造 class-versus-rest 方向：$\Delta \pmb{\mu}_j = \pmb{\mu}_j - \pmb{\mu}_{\lnot j}$
- 归一化为单位 $\ell_2$ 范数，决策边界置于两质心中点
- 同样通过 multinomial logistic-regression 校准概率分布

### Direct Conditional Probe
- 与 individual-statement probe **独立训练**，使用与 sAwMIL 相同的条件训练/校准样本，沿用 sweep 选定的 probe-specific 层。

---

## 实验与结果
### 数据集与模型
- **模型覆盖**：llama-3.2-3b、llama-3.1-8b/70b、gemma-7b/2-9b/2-27b、mistral-7b/12b/3.1-24b、qwen-2.5-7b/14b/72b（共 15 个模型变体）
- **三任务**：City Locations、Medical Indications、Word Definitions

### 关键结果数字
| 指标 | City Locations | Medical Indications | Word Definitions |
|---|---|---|---|
| SVM 中位残差相关 $\rho$ | 0.25 | 0.20 | 0.19 |
| Mass Mean 中位残差相关 $\rho$ | 0.17 | 0.14 | 0.22 |

- **Probabilistic coherence（CCK 距离）**：SVM probe 与 sAwMIL 的距离分布 broadly similar；Mass Mean 在 City Locations 和 Word Definitions 上需要更大的 RMS 调整（$d_{CCK}$）才能到达 CCK-coherent 集。
- **Domain-level graded stability**：SVM probe 下，City Locations 集中在高稳定性区，Medical Indications / Word Definitions 向中低稳定性扩展——与主结果一致；Mass Mean 下分离 substantially weaker。
- **Behavioral resilience（$\Delta M$）**：SVM probe 在 probability-matched 条件下，高/低稳定性信念的行为移动差异仍以正向为主，但效应比 sAwMIL 更 heterogeneous；Mass Mean 估计效应更小且方向不一致。

### 最强结果与结论
- **核心结论稳健**：graded stability 的命题级变异无法被 individual belief probability 完全解释，三种 probe 下均成立。
- **SVM probe 优于 Mass Mean**：更适合本文三值概率估计任务。

---

## 相关工作脉络
1. **Mass Mean veracity probe**（[17]）：二分类真值探针，基于 True/False 表征质心差估计 truth direction——本文将其扩展至三类场景并揭示其局限性。
2. **Three-class Mass Mean 扩展**（[9]）：引入 "Neither" 类别导致质心漂移与估计不稳定——本文实证验证了该问题。
3. **sAwMIL 相对前作**：与二分类 veracity probe 的根本区别在于输出**分级稳定性**而非二元标签，并分离出个体信念概率之外的残差信号。
4. **CCK 距离（Coherence-based benchmark）**：用于衡量概率分布的相干性——本文用作辅助评估指标。
5. **Behavioral resilience（$\Delta M$）**：衡量不同稳定性信念在行为上的可塑性差异——本文验证稳定性分数是否具有行为预测力。

---

## 局限性与未来方向
1. **Probe 敏感性**：Mass Mean 在三类设置下表现不稳定，说明方法选择对结果影响较大，需更多系统性 ablation。
2. **仅验证三类 probe**：本文仅对比 sAwMIL、SVM、Mass Mean，更多现代探针（如 MLP probe、方向投影法等）尚未纳入。
3. **模型覆盖有限**：仅涵盖 15 个主流开源模型，对新兴架构或非自回归模型的泛化性未知。
4. **任务范围**：仅三个领域任务（地理、医学、词汇），跨领域迁移性有待验证。
5. **未来方向**：扩展至更多任务域、探索更稳健的多类探针设计、研究残差相关性的认知科学解释。

---

## 研究启发与可借鉴点
1. **Probe 对比验证范式**：用多种独立 probe 交叉验证主结果，可有效排除方法特异性偏差——值得在本团队后续研究中复现。
2. **残差相关性分析思路**：控制个体信念概率后考察残差信号，是一种剥离混杂因素的有效分析策略，可迁移至其他信念/对齐评估工作。
3. **Mass Mean 的三类局限启示**：二分类设计的质心方法在多类场景需谨慎使用，本团队在类似设置下应避免直接套用。
4. **SVM probe + logistic 校准流水线**：标准化的 StandardScaler → LinearSVC → L-BFGS 校准流程可作为简洁可靠的基线方案复用。

---

## 关键术语表
**sAwMIL**：本文提出的加权多实例学习框架，用于估计 LLM 命题级信念的分级稳定性。  
**Graded Stability（分级稳定性）**：对信念稳定性的连续度量，区别于二元稳定/不稳定分类。  
**Mass Mean**：基于类质心方向估计 truth direction 的二分类探针，在多类设置下存在质心漂移问题。  
**CCK Distance**：衡量概率分布与相干集（coherent set）之间距离的指标。  
**Behavioral Resilience（$\Delta M$）**：衡量不同稳定性等级信念在下游行为上的可塑性差异。  
**Individual Belief Probability**：单个命题的信念置信度概率，本文证明其不足以完全解释稳定性变异。  
**Residual Correlation（残差相关）**：控制个体信念概率后，跨模型间命题稳定性的共享变异相关系数。  
**Direct Conditional Probe**：与 individual-statement probe 独立训练的探针，使用相同条件训练/校准样本。

---

## 可复现要素
- **数据集**：论文涉及 City Locations、Medical Indications、Word Definitions 三任务，代码/数据公开性论文未在此段明确声明，需进一步核查 arXiv 原文。
- **代码/权重**：论文未在此段提及，需核查源码仓库声明。
- **关键超参**：SVM probe：$C=1.0$，$\ell_2$ 正则，squared-hinge loss，最多 10,000 次迭代，容差 $10^{-4}$；logistic 校准：$C=1.0$，L-BFGS，最多 5,000 次迭代，容差 $10^{-6}$；Mass Mean：class-versus-rest 质心方向 + 单位 $\ell_2$ 归一化。
- **模型列表**：llama-3.2-3b、llama-3.1-8b/70b、gemma-7b/2-9b/2-27b、mistral-7b/12b/3.1-24b、qwen-2.5-7b/14b/72b。
- **探针选层**：覆盖所有上述模型 × 三任务的独立最优层（见原文 Table A9）。

---
