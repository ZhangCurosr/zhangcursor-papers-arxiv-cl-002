---
title: "Toward-a-Graded-Measure-of-Belief-Stability-in-Large-Languag"
source: https://arxiv.org/pdf/2609.34158v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-01 10:56:20"
---

# 论文速读：Toward-a-Graded-Measure-of-Belief-Stability-in-Large-Languag

## 一句话总结
本文提出“分级信念稳定性”（Graded belief stability）指标，从关系性视角量化LLM中某一命题在其完整信念系统内经受条件更新后仍保持可信的程度，弥补了现有准确率、校准等孤立评估维度无法捕捉信念持久性的缺陷。

## 研究问题与动机
1. **现有评估维度的盲区**：当前LLM事实可靠性评估主要依赖逐条准确率、不确定性估计或校准曲线，将命题视为孤立实体，无法刻画信念在信息语境变化下的系统性持久能力。
2. **同一支持强度的异质性**：相同先验概率的两个命题（如“Paris is in France”与“Oslo is in Norway”），若嵌入不同的背景信念网络，其实际稳定性可存在显著差异。
3. **关系性属性的缺失**：信念并非静态概率点，而是嵌入于信念集合 $\mathcal{B}_{\mathcal{M}}$ 与非不信集合 $\mathcal{X}_{\mathcal{M}}$ 中的关系性结构，需通过条件概率的跨命题比较来度量。

## 核心贡献（创新点）
1. **提出分级信念稳定性的形式化定义**：基于Lockean Thesis与Leitgeb's Humean Thesis，将稳定性定义为“条件后仍高于经验阈值的命题比例”，与仅关注单命题置信度的传统校准方法本质不同，强调信念的系谱关系。
2. **构建三值（T/F/N）探针与双路径条件估计框架**：设计Direct Conditional与Joint-to-Conditional两种估计器，显式建模悬置/不确定状态，区别于传统二值探针，更贴合LLM真实输出分布。
3. **规模化验证命题级稳定性的可复现结构**：在12个模型（3B–72B）与3个领域的实验中证明，个体信念概率仅能解释约40%的稳定性变异，剥离后的残差稳定性跨模型呈高度正相关（99%为正，$\rho=0.23\sim0.31$），提示存在命题级系统性脆弱点。
4. **引入CCK概率相干距离作为系统级诊断工具**：通过极值规划计算观测分布与相干信念系统的最小RMS距离，发现所有 tested 模型的信念系统仅需1.5~6.2个百分点的微调即可投影至相干状态，填补了大模型信念系统结构性质量的评估空白。

## 方法详解
- **形式化基础**：设信念集 $\mathcal{B}_{\mathcal{M}}$、非不信集 $\mathcal{X}_{\mathcal{M}}$（信念+悬置信念），经验阈值 $t_{\mathcal{M}} = \min_{s_i \in \mathcal{B}_{\mathcal{M}}} \pi_T(s_i)$。分级稳定性公式为：
  $$\gamma_{\mathcal{M}}(P) = \frac{|\{x \in \mathcal{X}_{\mathcal{M}}\setminus\{P\} : \text{Pr}_{\mathcal{M}}(P|x) > t_{\mathcal{M}}\}|}{|\mathcal{X}_{\mathcal{M}}\setminus\{P\}|} \in [0,1]$$
- **三值探针架构**：
  - **主探针 sAwMIL**：稀疏感知多实例学习，基于token级表示的max-margin分类，输出 $\pi_T/\pi_F/\pi_N$。
  - **次级探针**：SVM（取最终token，C=1.0，L2正则，平方铰钉损失，max_iter=10,000）与 Mass Mean（类质心对比方向 $\Delta\mu_j = \mu_j - \mu_{-j}$ 归一化）。
  - 所有探针输出经 multinomial logistic-regression 校准（C=1.0，L-BFGS，tol=10⁻⁶）。
- **Direct Conditional 估计器（主方法）**：构造条件句“Given x, P.”，直接探针隐藏表示，以 $\pi_T^{\text{direct}}(P,x)$ 作为 $\widehat{\text{Pr
