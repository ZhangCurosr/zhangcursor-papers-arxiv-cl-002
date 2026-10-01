---
title: "WHEN-WORDS-FALL-SHORT-ITERATIVE-SYNERGY-BETWEEN-VERBALIZED-R"
source: https://arxiv.org/pdf/2609.34454v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:24:02"
field: "LLM可信度与校准"
keywords: ["置信度估计", "大语言模型", "强化学习", "隐藏状态", "校准", "verbalized confidence"]
innovations: ["提出IPoET交替训练框架，将置信度估计器嵌入RL策略循环实现co-adaptation", "揭示估计器过拟合是prior work中verbalization优势结论的混杂因素，重新确立estimator基线强度", "设计estimator-derived reward，以负Brier形式将隐藏状态置信度误差转化为策略训练信号"]
benchmarks: ["HotpotQA", "Big-Math", "MATH-500", "GSM8K", "TriviaQA", "GPQA", "CommonsenseQA", "SimpleQA"]
---

# 论文速读：WHEN-WORDS-FALL-SHORT-ITERATIVE-SYNERGY-BETWEEN-VERBALIZED-R

## 一句话总结
本文提出IPoET（Iterative Policy-Estimator Training），通过交替优化策略与置信度估计器，融合LLM隐藏特征的丰富置信度信号与推理链生成的结构化能力，在多个数据集和骨干模型上实现了优于纯估计器或纯verbalization方法的域内置信度估计，并具备优异的域外泛化性能。

## 研究问题与动机
- LLM在高风险领域（医疗、法律、金融）的应用需要可靠的置信度估计，但现有模型常以过度自信的方式生成错误答案。
- 现有方法分为estimator-based（利用隐藏状态特征）和verbalization-based（通过prompt或训练让模型输出自然语言置信度）两大范式，近期研究普遍认为verbalized confidence优于独立估计器，但该结论可能存在方法学偏差。
- 本文实证研究发现：在控制估计器过拟合后，独立置信度估计器可大幅超越RL训练的verbalized confidence（AUROC提升0.161域内/0.078域外），揭示了先前对比中"估计器过拟合"是导致verbalization优势结论的重要混杂因素。
- 两种范式各有互补优势：隐藏特征包含更丰富的置信度信号但难以直接转化为自然语言；verbalization能利用LLM的推理能力但受prompt和答案依赖影响较大；核心问题是如何协同利用二者优势。

## 核心贡献（创新点）
- **重新评估并纠正了两种范式的性能对比**：本文通过全训练周期的checkpoint追踪发现，过拟合是导致prior work中verbalization优于estimator的关键因素，在合理过拟合控制下estimator可作为强基线。
- **提出IPoET交替训练框架**：首次将置信度估计器嵌入策略RL训练循环，通过交替冻结策略/估计器的方式实现co-adaptation，本质区别于此前"先训练策略再训练估计器"的串行范式或联合训练的相互干扰问题。
- **设计了estimator-derived reward机制**：将隐藏状态层面的置信度估计误差转化为负Brier-style奖励信号，引导策略生成更易被估计器准确评估的推理trace，无需策略输出显式置信度。
- **系统验证了域内外泛化性能**：在HotpotQA和Big-Math训练、Qwen3-8B-Base和Llama-3.1-8B-Instruct两种骨干下，IPoET在所有域内指标上均取得最优或次优，域外指标亦达到第一或第二，显著优于RLCR等verbalization基线。

## 方法详解
- **初始化阶段**：先用标准RLVR对base model进行warm-up得到初始策略$\pi_{\theta_1}$，再用其rollouts训练置信度估计器$s_{\phi_1}$（MSE损失，线性头作用于最后一个非padding token的隐藏状态$h$：$c = \text{clip}(\mathbf{w}^\top h + b, 0, 1)$）。
- **交替迭代（每次round）**：
  - **Step 1（策略更新）**：冻结$s_{\phi_t}$，使用GRPO算法更新策略$\pi_{\theta_t} \to \pi_{\theta_{t+1}}$，奖励函数为$R_{\text{IPoET}} = R_{\text{fmt}} + R_{\text{acc}} + R_{\text{est}}$，其中$R_{\text{est}} = -(s_\phi(x,y) - z)^2$为负Brier风格项，鼓励策略生成使估计置信度接近真实正确性的推理。
  - **Step 2（估计器更新）**：冻结$\pi_{\theta_{t+1}}$，采样新rollouts并标注正确性，以相同MSE损失更新$s_{\phi_t} \to s_{\phi_{t+1}}$，使用固定训练预算防止过拟合。
- **估计器架构**：与策略使用相同backbone架构的完整模型（而非轻量probe head），通过端到端梯度优化使最终隐藏状态凝聚网络各中间层的置信度信号。
- **训练策略**：不采用KL正则化，与reasoning model的常见RL训练惯例一致。

## 实验与结果
- **数据集**：训练用HotpotQA（ factual QA）和Big-Math（math reasoning）；评估涵盖6类benchmark：HotpotQA、TriviaQA、SimpleQA（factual）；CommonsenseQA、GPQA（常识/专家知识）；MATH-500、GSM8K、Big-Math（数学推理）。
- **评估指标**：Accuracy（任务性能）、AUROC（判别力）、Brier score（校准质量）、ECE（分组校准误差）。
- **基线**：RLVR、Answer-Prob、P(True)、RLVR+Estimator、RLCR、RLCR (Step-aligned)、RLCR+Estimator。
- **主要结果（Qwen3-8B-Base）**：HotpotQA训练下，IPoET AUROC=0.800，Brier=0.184，ECE=0.107；相对于RLVR+Estimator（AUROC=0.757，Brier=0.195），提升+0.043/-0.011，配对bootstrap检验显著（p=0.0006/0.0317）。Big-Math训练下AUROC=0.878，Brier=0.100，同样最优。
- **主要结果（Llama-3.1-8B-Instruct）**：HotpotQA训练AUROC=0.781，Brier=0.188；Big-Math训练AUROC=0.824，Brier=0.152，均取得最优或次优。
- **域外泛化**：IPoET在所有OOD设置下AUROC第一或第二，Brier最低或次低，优于RLCR(Step-aligned)（额外训练步数不带来持续提升，甚至在大Math训练下损害OOD性能）。
- **消融关键发现**：交替训练优于joint training（Accuracy +2.8pp，AUROC +0.027，Brier -0.025）；2轮交替（Round 2）为最优配置，第3轮边际收益趋近于零。

## 相关工作脉络
- **Answer-Prob / P(True)**：基于token概率或模型自我评估的隐式置信度方法，仅对固定输出打分，不介入策略训练；IPoET则将内部估计转化为主动引导生成行为的反馈信号。
- **RLVR (Shao et al., 2024)**：使用二元答案奖励训练reasoning policy，仅在eval时输出verbalized confidence；IPoET通过estimator reward在训练中即融入置信度校准目标。
- **RLCR (Damani et al., 2025)**：RL-based verbalized confidence方法，结合答案正确性与负Brier奖励优化显式置信度表达；本文证明其显式自评估可能稀释reasoning trace中的正确性相关信号，RLVR+Estimator已可超越RLCR。
- **Rewarding Doubt (Bani-Harouni et al., 2025)**：另一RL-based verbalization方法，强调discrimination和generalization；本文通过过拟合控制下的重评估表明estimator范式具有更大潜力。
- **内部状态探测工作（Azaria & Mitchell, 2023; Subramani et al., 2025）**：证明LLM隐藏状态编码factuality信息，但多为离线probing；IPoET实现了对隐藏特征的在线利用与策略协同优化。
- **SaySelf / LACIE (Xu et al., 2024; Stengel-Eskin et al., 2024)**：基于微调的verbalized confidence方法；IPoET完全避免了对显式置信度表达的依赖，减少了prompt敏感度。

## 局限性与未来方向
- **计算开销**：估计器使用与策略相同架构的完整模型，参数量翻倍，尽管单次forward pass评估效率高于RLCR的autoregressive生成，但仍增加了显存占用。
- **迭代轮次收敛性**：Round 2后边际收益递减，但最优轮数可能因任务和数据集而异，尚未系统探索自适应停止准则。
- **任务范围限制**：实验聚焦于有明确正确答案的数学推理和QA任务，对于开放生成、创意写作等模糊答案场景的适用性有待验证。
- **交替优化的稳定性**：虽然joint training存在干扰问题，但交替更新本身也可能引入策略-估计器之间的震荡，当前采用固定训练预算缓解，缺乏理论保证。

## 研究启发与可借鉴点
- **"估计器作为reward信号"的设计范式**：将离线预测模型转化为在线训练反馈，这一思路可迁移至其他需要校准模型自我评估的场景（如self-correction、tool selection）。
- **交替优化vs联合优化的权衡分析**：本文通过消融明确区分了"额外训练量"和"交替协同"两个因素，为后续设计multi-component RL训练提供了可复用的实验方法论。
- **过拟合控制的实证重要性**：在比较不同范式时，必须考虑训练动态和checkpoint选择策略，否则可能得出误导性结论——这一教训适用于几乎所有"method A vs method B"的benchmark论文。
- **无KL正则的reasoning RL训练**：与DeepSeek-R1等open-reasoner-zero实践一致，去掉KL约束可释放更强的policy改进空间，值得在置信度校准任务中借鉴。
- **隐藏状态聚合的替代方案探索**：本文使用末端隐藏状态，但 prior work指出关键信号分布在中间层；结合多粒度特征聚合（如expectation of aggregated internal belief）可能是进一步提升方向。

## 关键术语表
- **IPoET**：Iterative Policy-Estimator Training，本文提出的交替优化策略与置信度估计器的训练框架。
- **Verbalized Confidence**：通过自然语言显式输出的模型置信度，通常伴随reasoning trace生成。
- **Estimator-based Confidence**：基于模型隐藏状态特征训练独立预测器（probing head或完整模型）得到的置信度估计。
- **AUROC**：Area Under the Receiver Operating Characteristic Curve，衡量置信度分数对正确/错误样本的排序判别能力。
- **Brier Score**：置信度预测与二值正确性标签之间平方误差的均值，衡量逐样本校准质量，越低越好。
- **ECE**：Expected Calibration Error，将预测分组为M个置信度bin后，各bin内平均正确性与平均置信度的加权偏差。
- **GRPO**：Group Relative Policy Optimization，一种策略梯度RL算法，本文用作policy更新的核心优化器。
- **RLCR**：Reinforcement Learning for Calibrated Reasoning，Damani et al. (2025)提出的结合答案正确性与负Brier奖励的verbalized confidence训练方法。

## 可复现要素
- **数据集**：训练用HotpotQA（modified distractor版）和Big-Math（filtered版），均为公开数据集；评估用HotpotQA、TriviaQA、SimpleQA、CommonsenseQA、GPQA、MATH-500、GSM8K，全部公开。
- **代码**：开源，地址 https://github.com/xyk829/ipoet。
- **权重**：使用公开backbone Qwen3-8B-Base和Llama-3.1-8B-Instruct。
- **框架**：verl（Sheng et al., 2025），GRPO算法。
- **关键超参**：论文未详细列出学习率、batch size、GRPO group size等，详见Appendix A（prompt template及训练细节）。
- **评估**：AUROC/Brier/ECE计算公式见Appendix B，Accuracy采用各数据集标准eval流程（exact match / math verify / LLM-as-a-judge）。
