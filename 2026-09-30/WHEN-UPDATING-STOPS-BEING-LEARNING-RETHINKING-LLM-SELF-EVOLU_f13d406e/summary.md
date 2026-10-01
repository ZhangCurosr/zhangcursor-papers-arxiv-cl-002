---
title: "WHEN-UPDATING-STOPS-BEING-LEARNING-RETHINKING-LLM-SELF-EVOLU"
source: https://arxiv.org/pdf/2609.36535v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:59:58"
field: "大语言模型自我演化与训练稳定性"
keywords: ["self-evolution", "LLM自进化", "信息增益", "模型退化", "GRPO", "数据筛选", "早期停止"]
innovations: ["提出可学习信息增益Ct作为系统级诊断信号，证明其在精确拟合下分解为KL散度与熵变", "构建ATRI框架，通过轮次内信息增益重加权与轮次间三阶段停止规则缓解自进化退化", "扩展方向性增益Cd实现闭环饱和后的高效外部样本选择"]
benchmarks: ["GSM8K", "MATH-500", "AMC", "Minerva", "Olympiad", "AIME-2024", "AIME-2025", "MMLU-Pro", "SuperGPQA", "BBEH"]
---

## 论文速读：WHEN-UPDATING-STOPS-BEING-LEARNING-RETHINKING-LLM-SELF-EVOLU

## 一句话总结
本文从信息论视角重新审视大语言模型自进化（self-evolution）过程中的性能退化问题，提出可学习信息增益（learnable information gain）$C_t$ 作为系统级诊断指标，并基于此构建 ATRI 框架——通过轮次内样本重加权与轮次间早期停止来缓解自进化退化。

## 研究问题与动机
- **核心问题**：自进化中 Solver 性能通常经历"早期提升 → 平台期 → 衰退"的退化现象（self-evolution degeneration），现有方法未能从根本上阻止退化。
- **现有方法不足一**：既有方法（如 reward-variance 滤波、多样性惩罚）仅针对 Questioner 或 Solver 单个组件进行干预，忽略了两者作为紧耦合系统共同演化的事实。
- **现有方法不足二**：现有监控信号（reward variance、policy entropy、output diversity）均只能捕捉自进化回路的局部侧面，实验证明它们无法可靠预测性能拐点。
- **根本原因**：自进化是一个信息封闭回路——根据 Generative Containment 定理（Appendix B），闭环内每轮产生的训练数据所含关于目标能力的互信息不会超过上一轮，继续训练最终等价于对自身高置信输出的蒸馏，必然导致分布收缩。

## 核心贡献（创新点）
- **系统级诊断信号**：定义可学习信息增益 $C_t$，证明在精确拟合下 $C_t = \text{KL}(p_t \| p_{t-1}) + [H(p_t) - H(p_{t-1})]$，将退化机制统一为信息论语言，而不仅是经验观察。
- **信息增益驱动的训练范式 ATRI**：在 Questioner 和 Solver 两侧分别以 $c_t^Q$ 和 $c_t^S$ 重加权 GRPO 梯度，使更新集中于"上轮未覆盖的新内容"，这是第一个将信息增益同时作用于生成侧与求解侧的框架。
- **多阶段生命周期揭示与早期停止规则**：提出 Phase I（$C_t>0$，准确率上升）→ Phase II（$C_t\to 0$，平台期）→ Phase III（$C_t\approx 0$，衰退）的三阶段模型，并在六个基线方法上验证 $C_t$ 均可在准确率峰值前 1–4 轮触发告警。
- **方向感知外部样本选择**：扩展 $C_t$ 为方向性增益 $C_t^d$，区分"模型已掌握的新内容"与"模型当前薄弱环节的新内容"，实现闭环饱和后的高效外部数据接入。

## 方法详解
**代理模型拟合**：每轮 $t-1$ 结束后，用小语言模型 $M_{t-1}$（默认 Pythia-160M）在 $\mathcal{D}_{t-1}$ 上拟合一个 epoch，得到代理分布。代理的目标是提供低方差的轮次数据分布估计，而非解决下游任务。

**单样本信息增益估计**：对序列 $x$，其编码代价为 token 归一化负对数似然 $\ell(x) = -\frac{1}{|x|}\sum_k \log M_{t-1}(x_k|x_{<k})$。问题的增益为 $c_t^Q(q) = \ell(q) - \bar{\ell}_q(\mathcal{D}_{t-1})$，答案的增益为 $c_t^S(s|q) = \ell(s|q) - \bar{\ell}_{s|q}(\mathcal{D}_{t-1})$。

**轮次内重加权**：取正部 $w_t^Q(q) = \max(c_t^Q(q), 0)$ 乘以 GRPO 优势值 $\hat{A}_i$ 得到校准优势 $\tilde{A}_i$，代入标准 GRPO loss 训练 Questioner；Solver 侧同理。这使梯度更新聚焦于"比上轮更难以预测"的样本。

**轮次间早期停止**：轮次级增益 $C_t = \bar{\ell}(\mathcal{D}_t) - \bar{\ell}(\mathcal{D}_{t-1})$，当 $C_t < \tau = 0.1 C_1$ 连续两轮时触发停止，对应 Phase II 结束点。

**方向性增益与外部阶段**：将 $\mathcal{D}_t$ 按正确/错误划分为 $\mathcal{D}_t^+$ 和 $\mathcal{D}_t^-$，分别训练代理 $M_t^+$ 和 $M_t^-$，外部样本 $e$ 的方向增益 $C_t^d(e) = \ell(e; M_t^+) - \ell(e; M_t^-)$，正值表示该样本更接近模型当前错误模式，与 $c_t(e)>0$ 组合形成双门控筛选。

## 实验与结果
- **数据集**：R-Diverse 评测套件，含 7 个数学推理基准（GSM8K、MATH-500、AMC、Minerva、Olympiad、AIME-2024、AIME-2025）和 3 个通用推理基准（MMLU-Pro、SuperGPQA、BBEH）。
- **基线**：STaR、SPIN、AZR、R-Zero、R-Diverse，均在相同 base model 和协议下运行。
- **主模型**：Qwen3-4B-Base 与 Qwen3-8B-Base。
- **关键数字**（Qwen3-4B-Base，Math AVG）：Base 42.58 → R-Diverse 52.56 → **ATRI 53.55**（+0.99 vs 最强基线）；Overall AVG：47.39（+1.19）。
- **关键数字**（Qwen3-8B-Base，Math AVG）：**ATRI 58.94**（+2.45 vs R-Diverse 56.49）；Overall AVG：**52.75**（+2.56）。
- **泛化验证**：Llama-3.1-8B/MATH、Qwen3-4B/MBPP 代码生成任务上生命周期与告警行为一致；OPT-125M 替换 Pythia-160M 代理，告警轮次相同。
- **计算开销**：ATRI 每轮额外成本仅 +2.2%（Qwen3-8B-Base），相对开销随主模型规模增大而下降（4B: 4.1% → 14B: 1.3%）。

## 相关工作脉络
- **STaR / SPIN / AZR / R-Zero / R-Diverse**：均为自进化/自博弈框架，解决思路在组件层面（reward filtering、diversity penalty），缺乏系统级诊断信号。
- **Shumailov et al. (2024)**：提出模型坍缩（model collapse）概念，指出递归训练自生成数据会导致分布坍缩，本文从信息论角度给出更精确的定量刻画。
- **Wang et al. (2025) RAGEN**：引入 reward variance 作为退化诊断信号，本文证明该信号仅是局部信号，无法可靠追踪拐点。
- **Cui et al. (2025)**：将强化学习饱和归因于 policy entropy 坍塌，本文证明 entropy 同样只是单侧信号。
- **Mindermann et al. (2022) RHO-LOSS**：用小参考模型优先高损失但可学习的样本，但参考模型固定于 held-out 数据；本文参考模型动态更新（每轮重新拟合），并直接用于停止决策。
- **Mobahi et al. (2020) 自蒸馏理论**：证明反复自蒸馏会逐步限制解空间表达能力，本文 Proposition 6 将该结论映射到 Phase III 的解释。

## 局限性与未来方向
- **闭环结构假设**：$C_t$ 依赖明确的轮次划分，流式/在线场景需用固定 token window 替代，窗口大小需人为选择。
- **代理近似误差**：$C_t$ 通过有限容量的小语言模型估计，极端数据集分布（如极端长度偏斜）可能破坏代理校准，本文未系统性研究此类 case。
- **验证器可靠性**：本文使用 majority-vote pseudo-label，未涉及 learned reward model 场景下的 reward-hacking 复发问题。
- **阈值经验性**：$\tau = 0.1 C_1$ 为经验选择，仅证明在 $\alpha \in [0.05, 0.30]$ 范围内鲁棒，非严格理论最优。
- **外部数据依赖**：外部阶段需要可获取的外部候选池，在完全无外部数据的领域无法进一步提升。
- **泛化范围**：目前证据覆盖 Qwen/Llama 系列与数学推理/代码生成，在更多模型家族、任务类型、代理架构上的泛化仍需验证。

## 研究启发与可借鉴点
- **信息论诊断框架的可迁移性**：$C_t$ 的构造思路（小代理拟合 + NLL 对比）可直接迁移至其他迭代训练范式（如 self-play RL、在线 RLHF），作为模型-agnostic 的健康度监控器。
- **Phase I/II/III 生命周期分析的实验范式**：建议科研团队在自己的自进化实验中定期记录 $C_t$ 轨迹，而非仅报告最终快照精度，以建立更诚实的方法对比基准。
- **方向性增益 $C_t^d$ 的数据筛选策略**：双门控（新颖性 + 针对性）的外部样本选择思想可应用于数据增强、课程学习等领域，优先选取"模型脆弱但可学"的样本。
- **小代理模型的解耦设计**：代理模型与主模型完全解耦（独立 checkpoint、独立训练），使 $C_t$ 不依赖主模型超参，这一设计模式值得在跨方法比较中推广。
- **三阶段停止规则的实证价值**：停止规则仅需关闭回路自身数据即可实现，无需 held-out 验证集，降低了自进化实验的部署门槛。

## 关键术语表
- **Self-evolution degeneration**：自进化退化，指 Solver 性能在自进化早期提升、随后进入平台期并最终衰退的现象。
- **Learnable information gain ($C_t$)**：可学习信息增益，衡量第 $t$ 轮数据相比第 $t-1$ 轮引入的新信息量，理论分解为 KL 散度加熵变。
- **Proxy model**：代理模型，小语言模型 $M_{t-1}$，用于拟合上一轮数据分布并提供 NLL 编码代价估计。
- **Generative Containment**：生成封闭性定理，指出闭环内每轮互信息 $I(P_{t+1}; X) \leq I(P_t; X)$，即闭环无法凭空产生新信息。
- **Three-phase lifecycle**：三阶段生命周期，Phase I（$C_t>0$，性能上升）→ Phase II（$C_t\to 0$，平台期）→ Phase III（$C_t\approx 0$，分布收缩衰退）。
- **Directional learnable information gain ($C_t^d$)**：方向性增益，区分外部样本更接近模型正确输出还是错误输出，用于精准筛选薄弱环节数据。
- **GRPO advantage**：GRPO 优势函数，将 batch 内 reward 归一化后的训练信号，本文在此基础上引入信息增益重加权。
- **Self-distillation (Phase III)**：自蒸馏，Phase III 中模型反复在自己高置信输出上训练，等价于 Mobahi et al. (2020) 描述的自蒸馏收缩过程。

## 可复现要素
- **数据集**：R-Diverse 评测套件（7 数学 + 3 通用推理基准）；外部池使用 NuminaMath、MetaMathQA、OpenMathInstruct。
- **代码/权重**：论文未明确声明开源；附录提供详细超参与实验设置。
- **关键超参**：代理模型 Pythia-160M，$\alpha=0.1$（$\tau=\alpha C_1$），learning rate $1\times10^{-5}$，batch size 128，temperature 0.7，top-p 0.95，5 次重复实验取均值。
- **训练框架**：GRPO loss + KL 正则，Pythia-160M 代理每轮单 epoch 微调。
