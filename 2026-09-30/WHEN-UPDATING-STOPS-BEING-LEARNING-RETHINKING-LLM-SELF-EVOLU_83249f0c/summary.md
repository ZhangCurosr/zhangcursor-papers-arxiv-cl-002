---
title: "WHEN-UPDATING-STOPS-BEING-LEARNING-RETHINKING-LLM-SELF-EVOLU"
source: https://arxiv.org/pdf/2609.36535v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:59:47"
field: "大语言模型自我演化与训练稳定性"
keywords: ["self-evolution", "LLM self-improvement", "information gain", "model collapse", "reinforcement learning", "data selection", "early stopping"]
innovations: ["提出可学习信息增益Ct作为系统级诊断指标，证明其分解为KL散度加熵变", "ATRI框架实现样本级重加权与无需验证集的跨轮早停，有效抑制自我演化退化"]
benchmarks: ["GSM8K", "MATH-500", "AMC", "Minerva", "Olympiad", "AIME24", "AIME25", "MMLU-Pro", "SuperGPQA", "BBEH"]
---

# 论文速读：WHEN UPDATING STOPS BEING LEARNING: RETHINKING LLM SELF-EVOLUTION VIA LEARNABLE INFORMATION GAIN

## 一句话总结
本文从系统层面重新审视大语言模型自我演化（self-evolution）中的性能退化问题，提出了基于可学习信息增益（learnable information gain）的全局诊断框架 ATRI，通过小代理模型衡量每轮新增可参数化信息量，实现样本级重加权与跨轮早停，有效延缓乃至消除自我演化退化。

## 研究问题与动机
- **核心问题**：LLM 自我演化（让模型用自身生成数据迭代训练）常遭遇"自我演化退化"——性能先升后平再降，现有工作仅在单一组件（Questioner 或 Solver）层面干预，忽略了这是一个紧密耦合系统。
- **现有方法不足**：
  1. **机制解释类**（奖励黑客、分布坍缩、生成-验证差距关闭）各自只捕捉到退化链条中的一环，无法统一刻画。
  2. **缓解类**（reward-variance 过滤、多样性正则）在下游组件上打补丁，上游已发生退化时无法逆转整体优化轨迹，甚至可能在其他阶段加剧退化。
  3. **信号类**（reward variance、policy entropy、n-gram distinct、self-BLEU）仅反映局部组件，实验证明它们无法可靠追踪性能退化趋势。

## 核心贡献（创新点）
1. **系统级诊断理论**：定义"可学习信息增益" $C_t$，证明在精确拟合下它等于相邻两轮数据分布的 KL 散度加上熵变，为自我演化退化提供统一的信息论刻画。
2. **ATRI 框架**：提出信息增益驱动的自适应训练调控模块，在单轮内对样本按增益正部重加权，跨轮当 $C_t$ 连续两轮低于阈值时自动早停，无需外部验证集。
3. **三方生命周期理论+实验验证**：提出 Phase I/II/III 三段论，并以 Proposition 6 将 Phase III 形式化连接至自蒸馏理论；在 6 种已有方法、2 种模型族、2 种任务上均复现该生命周期。
4. **方向感知外部数据选择**：扩展 $C_t$ 为有向信息增益 $C_t^d$，区分"新颖"与"针对模型当前薄弱点"，实现外部数据的精准引入，Math AVG 进一步提升 +2.12。

## 方法详解
**核心思想**：训练一个小代理语言模型 $M_{t-1}$ 拟合上一轮数据 $\mathcal{D}_{t-1}$，用 NLL（负对数似然）度量当前轮数据相对于上一轮的"信息增益"。

**可学习信息增益定义**：
- 代理模型 $M_{t-1}$ 在 $\mathcal{D}_{t-1}$ 上做单轮自回归训练，得分函数为 per-token NLL：$\ell(x) = -\frac{1}{|x|}\sum_{k}\log M_{t-1}(x_k|x_{<k})$。
- 单个 question 的增益：$c_t^Q(q) = \ell(q) - \bar{\ell}_q(\mathcal{D}_{t-1})$，即当前问题预测难度相对上一轮均值的偏差。
- 整个 round 的增益：$C_t = \bar{\ell}(\mathcal{D}_t) - \bar{\ell}(\mathcal{D}_{t-1})$。

**理论性质（Proposition 2 & Corollary 1）**：精确拟合下，$C_t = \mathrm{KL}(p_t\|p_{t-1}) + [H(p_t) - H(p_{t-1})]$，即 KL 散度 + 熵变，给出严格的信息论解释。

**训练集成**：
- Questioner 端：GRPO 优势项 $\hat{A}_i$ 乘以权重 $w_t^Q(q_i)=\max(c_t^Q(q_i),0)$，得到校准后优势 $\tilde{A}_i$，更新 Questioner 使用 $\mathcal{L}^Q$（含 KL 正则）。
- Solver 端：同理，按答案条件增益 $c_t^S(s_j|q_i)$ 重加权 GRPO 优势。
- 早停规则：$C_t < \tau = 0.1 C_1$ 连续两轮即终止演化。

**外部数据阶段（Appendix J）**：引入有向增益 $C_t^d(e) = \ell(e;P_t^+) - \ell(e;P_t^-)$，其中 $P_t^+/P_t^-$ 分别是正确/错误解的代理分布，双门控（$c_t>0$ 且 $C_t^d>0$）筛选外部样本。

## 实验与结果
**数据集**：7 个数学推理（GSM8K、MATH-500、AMC、Minerva、Olympiad、AIME24、AIME25）+ 3 个通用推理（MMLU-Pro、SuperGPQA、BBEH），遵循 R-Diverse 评测套件。

**基线**：STaR、SPIN、AZR、R-Zero、R-Diverse。

**关键结果（Qwen3-4B-Base）**：
- ATRI Math AVG = **53.55**，超越最佳基线 R-Diverse（52.56）达 **+0.99**；Overall AVG = **47.39**，超越 R-Diverse（46.20）达 **+1.19**。
- MATH 单榜提升最大：81.20 vs R-Diverse 78.65（+2.55）；AIME25：15.20 vs 10.88（+4.32）。

**关键结果（Qwen3-8B-Base）**：
- ATRI Math AVG = **58.94**，超越 R-Diverse（56.49）达 **+2.45**；AMC 提升显著：73.90 vs 66.02（+7.88）。

**生命周期复现**：在 Vanilla 实验中，$C_t$ 在 round 3 归零，准确率在 round 4 达峰，round 9 开始下降，$C_t$ 领先准确率峰值 1 轮、领先衰退 6 轮。

**Phase III 收缩证据**：round 4→8，pass@8 下降 19.8 点，path entropy 下降 36.5%，distinct correct paths 下降 46.6%。

**早停质量**：Passive $C_t$ 早停在 6 种方法上平均准确率 46.32，仅比 oracle best 差 0.83，且无需验证集。

**计算开销**：Pythia-160M 代理额外开销约 +2.2% PFLOPs/GPU-hours，随基座模型增大趋近于 0。

## 相关工作脉络
1. **STaR (Zelikman et al., 2022)**：自蒸馏式自我训练，仅用正确/错误标签反馈，未考虑迭代中数据分布演化，易陷入退化。
2. **SPIN (Chen et al., 2024)**：自博弈 SFT，利用偏好对训练，但未监控多轮信息衰减，长期训练会出现 pass@1 升而多样性降。
3. **R-Zero (Huang et al., 2025b)**：Questioner-Solver GRPO 框架，本文主要对比基线，其不确定性奖励易导致问题坍塌。
4. **R-Diverse (Li et al., 2026)**：引入多样性惩罚缓解同质化，属组件级干预，本文证明其仍会经历 Phase III 退化。
5. **AZR (Zhao et al., 2025)**：绝对零数据强化自博弈，与 R-Zero 类似面临闭环信息饱和问题。
6. **Model Collapse 文献 (Shumailov et al., 2024; Dohmatob et al., 2024)**：揭示递归生成数据导致分布坍缩，本文从信息增益角度给出更精细的三段论刻画与可操作监控信号。

## 局限性与未来方向
- **闭环结构假设**：$C_t$ 依赖明确的 per-round 数据集，无轮边界的流式场景需选窗口大小（Appendix H 显示影响有限）。
- **代理近似误差**：Pythia-160M 是欠拟合代理，极端数据分布（如极端长度偏斜）可能破坏校准，实践中未遇到。
- **外部池依赖**：外部数据阶段需要存在高质量候选池，完全新领域无外部数据时该阶段无法启用。
- **验证器可靠性**：本文用 majority-vote pseudo-label，未涉及 learned reward model，部署中 reward-hacking 风险仍需关注。
- **阈值选择**：$\tau = 0.1 C_1$ 为经验值，$\alpha \in [0.05, 0.30]$ 范围稳定但最优值未理论推导。
- **泛化边界**：目前仅在 Qwen/Llama + MATH/MBPP + Pythia/OPT proxy 上验证，其他模型族和任务需进一步检验。

## 研究启发与可借鉴点
1. **小代理模型作为数据分布诊断器**：用轻量 LM 拟合上一轮数据并计算 NLL 作为新颖性度量，这一范式可迁移至任何迭代训练场景（如 RLHF 多轮、continued pretraining）。
2. **系统级而非组件级视角**：本文证明局部信号（reward variance 等）无法可靠追踪退化，全局信息增益能统一刻画所有方法的共性生命周期，启示后续工作应优先设计系统级监控指标。
3. **KL 散度 + 熵变分解**：Proposition 2 的理论恒等式为理解生成模型迭代训练提供了可解释的分解框架，未来可用于分析 diffusion model 自训练、agent 自博弈等场景。
4. **无需验证集的早停策略**：Passive $C_t$ 早停仅需内部数据，避免对 held-out 集的依赖，适合资源受限或数据敏感场景。
5. **有向信息增益 + 双门控筛选**：$C_t^d$ 同时兼顾"新颖性"和"针对性"，这一双重标准可推广至 curriculum learning、active learning 中的样本选择。

## 关键术语表
- **Self-evolution degeneration**：大模型自我演化中性能先升后降的现象，由闭环训练导致数据分布饱和引发。
- **Learnable information gain ($C_t$)**：第 $t$ 轮数据相对第 $t-1$ 轮新增的可参数化信息量，等于代理模型 NLL 差值。
- **Proxy model ($M_{t-1}$)**：在上一轮数据上单轮微调的小型语言模型（Pythia-160M），用于估计数据分布并提供新颖性打分。
- **Three-phase lifecycle**：Phase I（$C_t>0$，性能上升）→ Phase II（$C_t\to 0$，性能 plateau）→ Phase III（$C_t\approx 0$，性能退化）。
- **Generative Containment**：闭环自我演化的信息论边界，由数据处理不等式保证，表明闭环本身无法产生关于目标能力的新信息。
- **Directional learnable information gain ($C_t^d$)**：区分样本与正确/错误解分布的相似度，用于外部数据的有针对性的选择。
- **GRPO advantage**：Group Relative Policy Optimization 中基于 batch 内归一化的优势函数，用于策略梯度更新。
- **Self-distillation**：模型在自身高概率输出上继续训练导致解空间收缩的过程，Phase III 被形式化为此过程。

## 可复现要素
- **数据集**：GSM8K、MATH-500、AMC、Minerva、Olympiad、AIME24、AIME25、MMLU-Pro、SuperGPQA、BBEH，均公开可用。
- **代码/权重**：论文未提及开源声明（PDF 解析文本中未见 GitHub 链接），需联系作者获取。
- **基座模型**：Qwen3-4B-Base、Qwen3-8B-Base；代理模型 Pythia-160M。
- **关键超参**：学习率 $1\times 10^{-5}$（AdamW），batch size 128，温度 0.7/top-p 0.95，每轮 1 epoch，proxy 单轮微调，停止阈值 $\tau=0.1 C_1$，每问题生成 8 个答案。
- **训练硬件**：NVIDIA H800/A100 80GB，单节点 8 GPU。
