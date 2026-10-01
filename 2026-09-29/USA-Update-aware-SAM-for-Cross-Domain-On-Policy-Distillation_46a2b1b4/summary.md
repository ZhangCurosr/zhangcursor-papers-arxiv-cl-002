---
title: "USA-Update-aware-SAM-for-Cross-Domain-On-Policy-Distillation"
source: https://arxiv.org/pdf/2609.34225v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:22:34"
field: "大语言模型多域微调与参数空间合并"
keywords: ["On-Policy Distillation", "Model Merging", "Sharpness-Aware Minimization", "Cross-Domain Transfer", "Update-Salient Parameters"]
innovations: ["将 SAM 的等向扰动推广为按更新幅度各向异性的椭球扰动，使合并干扰上界的曲率因子在训练阶段被直接最小化", "通过短热身分位数估算替代 hard threshold 识别更新敏感参数，并据此分配平坦度预算", "从参数层面量化跨域更新耦合现象，揭示冲突对负迁移的机理并给出训练期防御"]
benchmarks: ["AIME 2025", "HMMT February 2026", "LiveCodeBench-v6", "NaturalCodeBench", "GPQA-Diamond", "SciBench-Atkins"]
---

# 论文速读：USA-Update-aware-SAM-for-Cross-Domain-On-Policy-Distillation

## 一句话总结
论文提出 Update-aware SAM（USA），一种面向跨域 On-Policy Distillation 的自适应最优化方法：通过短热身在每个域训练中识别"更新敏感参数"，将其对应的扰动半径放大，从而在合并时抵抗跨域更新耦合引起的负迁移；在数学、科学、代码三个领域六个跨域方向上，USA 均优于所有基线并将冲突对（如 Math↔Science）的负迁移逆转成正向增益。

## 研究问题与动机
- **单域 OPD 训练早饱和**：同一领域内推理模式有限，增加数据量无法突破天花板，需引入其他领域的监督信号。
- **多域混训存在分布冲突**：直接混合不同域的数据会使优化信号相互干扰；且新增/修改任一域都必须整体重训，缺乏灵活性。
- **模型合并虽解耦训练但仍可能负迁移**：独立蒸馏各域后再在参数空间合并可避免混训冲突，但在部分域对上（如 Math↔Science）所有合并算子均低于单域参考。
- **负迁移的根源是跨域更新耦合**：当两个域以相近幅度更新同一参数坐标时，合并产生的位移量与该坐标自身更新相当，线性组合在此类坐标上极易破坏已学行为。

## 核心贡献（创新点）
- **首次从参数层面量化"跨域更新耦合"并解释负迁移机理**：通过 per-parameter 耦合度量发现冲突对的耦合分布整体右移，为后续防御提供可操作的诊断依据。
- **提出 USA：将 SAM 的等向扰动改为各向异性椭球扰动**：在 OPD 初始 N 步热身中记录每参数更新幅度，按矩阵内分位数构造缩放因子，使更新敏感参数的扰动半径放大至 α 倍，其余参数保持标准半径。
- **在 6 个跨域方向、2 种学生规模上全面最优，逆转冲突对负迁移**：USA 在 Code↔Math、Math↔Science、Sci↔Code 全部方向均优于单域参考（1.7B 平均 +4.55 分，4B 平均 +4.01 分），并显著缩小不同合并算子间的性能差距。
- **给出合并干扰上界的理论分析并建立与训练目标的对应关系**：证明 USA 最小化的二阶项恰好是干扰上界中可由训练控制的曲率因子 λ_max(SHS)，使优化目标与合并后性能损失在同一度量下被收紧。

## 方法详解
- **短热身后记录更新幅度**：每个域从共享初始化 θ_0 起用标准 OPD 目标训练 N 步，得到 θ_N^(k)，并记录每参数更新幅度 m_k(i)=|θ_N^(k)(i)-θ_0(i)|；文献观察到 OPD 有效更新高度稀疏且早期即稳定，故短热身足以刻画更新敏感参数集合。
- **按矩阵内分位数构造缩放因子**：对每个权重矩阵 g，取其更新幅度的 (1-p) 分位数 D_k(g) 作为参考，坐标 i 的缩放因子为 s_k(i)=1+(α-1)·min(m_k(i)/D_k(g_i),1)，使 s_k(i)∈[1,α]；该构造具有整体尺度不变性，且在不同矩阵间分别归一化以适配 OPD 更新的 heavy-tailed 分布。
- **椭球约束 SAM 训练目标**：每个域在热身结束后求解 min_θ max_{||S_k^{-1}ε||₂≤ρ} L_k(θ+ε)，其中 S_k=diag(s_k)，ρ 为基本扰动半径；该椭球约束使更新敏感参数在一个更宽的轴向上被要求保持低损失，从而降低其局部曲率。
- **闭合形式扰动推导**：通过变量替换 ε=S_k v 将约束变为各向同性球，由 Cauchy–Schwarz 得最优 v*=ρ S_k ∇L/||S_k ∇L||，进而得到实际扰动 ε̂_k=ρ S_k² ∇L/||S_k ∇L||；每步在扰动点 θ+ε̂_k 处计算梯度并下降，扰动本身不更新参数。
- **理论界与训练目标的对应**：对局部极小点展开合并干扰 Ξ_t=L_t(θ_t+δ_t)-L_t(θ_t)，得到上界 |Ξ_t|≤½λ_max(SHS)·||S^{-1}δ_t||₂²+O(||δ_t||³)；USA 的目标在二阶意义下最小化的恰好是 λ_max(S_k H_k S_k)，即上界中仅由训练决定的曲率因子，而位移项 ||S^{-1}δ_t||² 在域训练完成后即固定。

## 实验与结果
- **数据集**：Math 域使用 DAPO-Math 中随机抽取的 4k 样本；Code 域使用 Skywork-OR1(Code) 的 4k 样本；Science 域使用 MegaScience 的 4k 样本；全部使用可程序验证答案的设定。
- **评估基准**：Math—AIME 2025、HMMT Feb. 2026；Code—LiveCodeBench-v6、NaturalCodeBench(Python dev split 58 题)；Science—GPQA-Diamond、SciBench-Atkins；解码用 temperature=1.0、top_p=0.6，每题 32 次采样，报告 average@32。
- **模型规模**：教师为 Qwen3-14B（用 GRPO 进一步优化），学生分别使用 Qwen3-1.7B 与 Qwen3-4B；所有方法共享同一 VeRL 训练基础设施。
- **主要结果（1.7B 学生）**：USA 在 Code→Math 41.51%、Math→Code 35.64%、Math→Sci 53.50%、Sci→Math 38.76%、Sci→Code 33.63%、Code→Sci 52.80% 均取得最优；单域参考均值约 36.96%，USA 均值约 42.51%，平均提升 4.55 分。
- **主要结果（4B 学生）**：USA 在 Code→Math 52.16%、Math→Code 56.25%、Math→Sci 67.04%、Sci→Math 50.72%、Sci→Code 54.46%、Code→Sci 65.65% 均取得最优；单域参考均值约 55.15%，USA 均值约 57.76%，平均提升 4.01 分。
- **冲突对逆转**：Math↔Science 在所有四种合并基线下均低于单域参考，USA 在该方向上仍以 53.50%（1.7B）显著超越单域参考 48.42%，将负迁移转为约 5 分的正向增益。
- **消融结论**：去掉 SAM 或改用等向 Vanilla SAM 均显著劣于 USA；将缩放因子随机置换或反向同样下降，说明增益来自"大缩放对应高更新"而非各向异性本身；由参数值本身而非更新幅度构造尺度会退化到各向同性结果。
- **超参鲁棒性**：α 在 [1,5] 范围内 USA 始终高于单域参考，但 α=1 时仅比无 SAM 提升 2.24 分；ρ 在 0.005–0.05 十倍范围内波动不到 0.62 分；α=5、ρ=0.03 为全局统一设置。
- **可扩展至更多源域**：将源域从 1 个增至 5 个，Weight Average 在 2–3 个源后开始回落，而 USA 在全部三个目标上随源域增加持续改善；USA 与 Weight Average 的差距从 1 源到 5 源分别扩大 4.3/2.3/1.9 分。
- **兼容不同合并算子**：将下游算子换为 TIES、DARE+WA、AdaMerging，USA 训练出的专家在所有算子和方向上仍全面优于对应 plain OPD 专家；plain OPD 在 DARE 下六方向全低于单域参考，USA 弥补了这一缺陷。
- **跨环境泛化**：在检索型 search-agent 环境中（NQ/HOTPotQA/PopQA）以相同超参复现，USA 仍六方向最强，平均领先第二名 2.50 分，且无需任何搜索环境特化调参。

## 相关工作脉络
- **On-Policy Distillation**：与 Agarwal et al.(2024)、Gu et al.(2024) 一脉相承，但既往工作均以单域单教师为目标设计，未考虑权重后续需与其他域共存。
- **Model Merging（Task Arithmetic）**：以 Ilharco et al.(2022) 为起点，Wortsman 等(2022)、Yadav 等(2023)、Yu 等(2024)、Yang 等(2024) 分别从加权平均、符号选举、稀疏化和系数学习角度减少干扰；这些方法均在训练结束后作用于 task vector，USA 则在训练阶段提前塑造向量形态。
- **Sharpness-Aware Minimization**：Foret 等(2020) 提出等向扰动 SAM；USA 将其推广为各向异性椭球 SAM，并将扰动预算与合并干扰上界的关键因子直接对齐。
- **Multi-Teacher OPD / MOPD**：Ma 等(2026) 将多域教师统一用于学生蒸馏，仍属混合训练范式；USA 坚持完全独立蒸馏 + 参数空间合并，保留每域独立迭代的能力。
- **Distillation Geometry 分析**：Cai 等(2026)、Shen 等(2026)、Yu 等(2026a) 指出 OPD 有效更新高度稀疏且早期锁定，USA 正是利用这一特性以短热身确定 perturbation 预算分布。

## 局限性与未来方向
- 模型仅限 Qwen3 家族，未在其他骨干架构上验证泛化性。
- 教师规模固定为 14B，未研究教师规模与 USA 效果的交互。
- 热身步数 N 取绝对值 10，未探究其与训练步数的比例关系。
- 假设"更新敏感参数即合并干扰主要承担者"在更广泛的域对/任务上仍需更多实证检验。

## 研究启发与可借鉴点
- **短热身 + 分位数参考**：用极短步数采样 per-parameter 更新幅度，并以矩阵内分位数而非全局极值构造参考，可有效适配 heavy-tailed 更新分布；这一模式可迁移至其他稀疏更新场景（如 LoRA 合并、持续学习）。
- **将合并干扰上界转化为训练目标**：通过 Taylor 展开把目标函数的曲率因子与合并后性能损失绑定，使得"训练阶段的平坦度约束"与"合并阶段的干扰"在同一度量下被优化；该方法学可用于任何"先独立训练再合并"的流程。
- **各向异性扰动的闭合形式**：从椭球约束到 ε̂=S²∇L/||S∇L|| 的推导简洁通用，可直接嵌入现有 SAM 实现，代价仅为一次矩阵对角乘法。
- **实验设计上的双基准 + 多方向 + 多规模**：每个域配两个 benchmark、六个有向转移方向、1.7B/4B 两档学生，使结论不受单基准或单规模偶然性影响，值得在系统方法论文中效仿。
- **与下游合并算子的正交性验证**：论文固定 USA 训练规则而逐一更换 Weight Average/TIES/DARE/AdaMerging，清晰分离"训练质量"与"合并规则"的贡献；这一对比范式适用于任何训练端改进工作。

## 关键术语表
- **On-Policy Distillation (OPD)**：学生在自身轨迹上采样，教师逐 token 提供密集监督，最小化学生与教师分布之间的 reverse KL 的训练范式。
- **Task Vector**：微调后模型相对共享初始化的参数差值 Δ_k=θ_k-θ_0，用于在参数空间线性组合多个域的知识。
- **Update-Salient Parameters**：OPD 训练早期即被大幅更新、承载大部分已学知识的少数参数坐标。
- **Cross-Domain Update Coupling**：两个域以相近幅度更新同一参数的程度，文中用 |c_i|=2|Δ_t(i)Δ_s(i)|/(Δ_t(i)²+Δ_s(i)²) 度量。
- **Sharpness-Aware Minimization (SAM)**：同时最小化损失值及其在扰动邻域内的最大值，以提升最优解的平坦性。
- **Ellipsoidal Perturbation**：USA 中由 S_k 对角缩放定义的扰动集合 {ε: ||S_k^{-1}ε||₂≤ρ}，使高更新参数承受更宽扰动。
- **Interference Bound**：合并后目标域损失相对于单域最优的增量上界，由 λ_max(SHS) 与 ||S^{-1}δ_t||² 共同决定。

## 可复现要素
- **数据集**：DAPO-Math、Skywork-OR1(Code)、MegaScience 均为公开数据集；每域 4k 样本。
- **代码/框架**：实验基于开源 VeRL 框架；论文未明确声明 USA 代码仓库。
- **权重**：学生起始于公开 Qwen3 checkpoint θ_0；教师为 Qwen3-14B+GRPO 微调模型。
- **关键超参**：热身步数 N=10；参考分位数 p=1%；放大上限 α=5；基础半径 ρ=0.03；合并系数 λ_t=λ_s=0.5。
- **优化器/学习率**：AdamW，lr=1e-6，batch_size=64，mini-batch=16；最大 prompt 长度 2560 tokens，最大 response 长度 20480 tokens。
- **硬件**：单节点 8× NVIDIA H20 96GB；1.7B 学生 vanilla OPD 约 44 小时，USA 约 50 小时（+13.6%）。
- **随机性**：所有 interpreter 实验重复 5 次，报告 mean±std；search-agent 扩展实验为单次运行。
