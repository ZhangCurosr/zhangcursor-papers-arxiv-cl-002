---
title: "UNLOCKING-THE-CRITIC-REWARD-FREE-POLICY-OPTIMIZATION-FOR-LLM"
source: https://arxiv.org/pdf/2609.37119v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:57:25"
field: "大语言模型强化学习后训练"
keywords: ["reinforcement learning", "LLM post-training", "reward-free optimization", "critic-based RL", "chain-of-thought reasoning", "policy optimization"]
innovations: ["将单一冻结 critic 同时用作 reward、GAE baseline 和未完成 prefix 预测器", "证明 critic 不稳定主要来自优化配方而非网络本身，单步低方差更新可恢复稳定"]
benchmarks: ["AIME 2025", "AIME 2026", "AMC 2023", "GPQA-Diamond"]
---

# 论文速读：UNLOCKING-THE-CRITIC-REWARD-FREE-POLICY-OPTIMIZATION-FOR-LLM

## 一句话总结
论文提出 RFPO（Reward-Free Policy Optimization），将预训练好的 critic 冻结后既作为 rollout 级 reward 信号，也作为 GAE baseline 和未完成轨迹的成功预测器，无需任何外部标注即可驱动策略优化，在数学推理任务上与监督 PPO 持平，同时减少约 19% 的 GPU 开销并支持提前对截断 rollout 打分。

## 研究问题与动机
- 当前 LLM RL 后训练趋势倾向于移除 critic，认为其训练成本高且长 CoT 推理下不稳定。
- 移除 critic 或仅保留其 baseline 作用时，reward 仍需来自外部：人类反馈、过程/结果奖励模型或 verifier，每条 pipeline 都要为同样东西付出不同代价。
- 若 critic 已能被信任为 baseline，它是否也足够好到直接作为 reward？这决定 reward 这一最耗资源的环节能否被大幅压缩。
- 长程推理中生成占主导成本，稀疏终态 reward 要求每条 trajectory 完成后才能计分，因此能否提前给未完成 prefix 打分非常关键。

## 核心贡献（创新点）
- Critic 不稳定性主要源于优化配方而非 value network 本身：将每次 policy update 控制在单步、低方差范围内即可恢复稳定收敛。
- 同一个冻结且校准好的 critic 可同时担任 rollout-level reward、GAE baseline 以及 unfinished prefix 的成功预测器，训练循环中不再需要 verifier。
- 对去偏后的 critic 分数进行二值化，能阻止 policy 对 critic 长度偏差的利用；连续使用同一 critic 反而会出现后期性能回落。
- RFPO 在 0% 标签条件下匹配监督 PPO 性能，并在 4,096-token 截断设置下相对同预算监督 PPO 节省约 19% GPU-hours。

## 方法详解
- Critic 来源：在数学推理上进行 supervised PPO，前 500 步 token cap 为 8,192，后 300 步 cap 为 5,120；在第 800 步提取冻结 critic，第 700 步提取初始策略 π_θ0。
- 关键固定点性质：当 γ = λ = 1 时，critic 回归目标是终态 reward，理想情况下 V_φ(x, y≤t) ≈ Pr(success | x, y≤t)。
- 冻结与校准：对 512 条 on-policy rollout 拟合长度去偏项 b(ℓ)，再按校准集的正样本率确定阈值 τ；部署奖励为 r̂(x, y) = 1[v(x, y) − b(ℓ(y)) > τ]。
- 训练更新：advantage A_t = r̂ − V_φ(x, y≤t)，仅更新 actor，critic 完全冻结；可选监督比例 p 以 Bernoulli 方式混入真 verifier 标签。
- 单步更新规则：每 batch 只做一次梯度更新，mini-batch 等于全 batch，取消内循环，从而降低梯度方差并避免多步累积导致的坍缩或发散。

## 实验与结果
- 数据集与模型：基于 Qwen3-4B-Base，使用 OpenR1-Math-220k 子集做 SFT，再用 DAPO-Math-17k 做 critic 预训练与初始策略。
- 评测基线：AIME 2025、AIME 2026、AMC 2023、GPQA-Diamond；pass@1 / pass@8 / maj@16。
- 5,120-token cap 主结果：300 步时 RFPO 0% 标签 macro pass@1 为 41.1%，监督 PPO 为 41.8%；AIME 2026 峰值 RFPO 达 25.7% 略高于 PPO 的 25.0%。
- 4,096-token cap 结果：RFPO macro 均值 47.6%，同 cap 监督 PPO 为 46.6%；RFPO 训练 300 步使用 264 GPU-hours，监督 PPO 为 327 GPU-hours，节省约 19%。
- 单步耗时：RFPO 421 秒 vs 监督 PPO 582 秒（−28%），峰值显存每卡降低约 9.4 GB。
- 截断滚动支持：5,120-token cap 下约 26.5–28.5% 训练生成被截断，但 critic 仍能给出精度约 0.89 的评分；1,024-token cap 时完成率低、精度坍缩至 0.21，说明该方法有下限。

## 相关工作脉络
- 移除或替换 critic 的当前主流做法：用同 prompt 多次采样构造 baseline，避免维护价值网络；本文定位是重启 critic 并放大其用途。
- 过程奖励模型：提供逐 step _dense signal，但需要 step 级标注；critic 由终态 outcomes 自学习并可预测未完成 prefix。
- 无标签替代方案：多数投票、自奖励、模型置信度、生成 rubric 等；缺乏 grounding 时易 reward hacking，而本文 reward 基于 verifier 校准后冻结。
- DPO、PRIME 等隐式 reward 方法：仍在训练循环中需要标签；RFPO 在 RL 阶段完全不依赖 verifier。
- 延长 rollout 场景下的 credit assignment 困难：已有方案多依赖截断或稀疏 attention，无法对未完成 trajectory 打分；critic 预测填补了这一空白。

## 局限性与未来方向
- 实验仅在 4B 模型与数学推理任务上验证，尚未扩展到更大模型或其他任务类型。
- 单次 300 步 RL 运行需数百 GPU-hours，成本仍较高，限制更快扫参与更大规模验证。
- 连续 critic 奖励曾达到更高峰值但最终回落，作者指出需要更可靠的早停或健康度指标，这仍是开放问题。
- 极低 token cap 下 critic 对未完成任务的区分能力明显下降，说明存在可用下限。

## 研究启发与可借鉴点
- “critic 不稳定”可优先归因于优化配方：单步、低方差更新是一个可迁移的稳定化技巧，值得在其它基于 critic 的 RL pipeline 中验证。
- 将 value head 输出在校准集上重标定量后再二值化，是防止 policy  exploitation 的有效工程手段；可复用该思路处理其它连续信号转 reward 的场景。
- 用同一 forward pass 同时产出 reward 和 baseline，避免额外前向开销；这种“一石三鸟”的复用思路适用于内存紧张场景。
- 用 explained variance 作为 critic 质量的低成本训练期监控指标，可替代昂贵离线评测用于早期 checkpoint 选择。
- 短截断训练与长截断评测的组合设计，能让验证分数更贴近方法真实能力，避免被训练 cap 误导。

## 关键术语表
**critic / value network**：学习给定状态价值或成功概率的网络，用于 baseline 和优势估计。  
**GAE**：广义优势估计，结合多步回报与 value baseline 的低方差优势 estimator。  
**supervised PPO**：使用 verifier 标签并对 actor 与 critic 同时更新的经典 RL 配方。  
**pass@k**：对每题采样 k 次后至少一次正确的估计成功率。  
**calibration**：在校准集上估计去偏偏移与阈值，使二值 reward 的正样本率与 verifier 一致。  
**length bias**：critic 分数随响应长度变化的系统性偏移，易被 policy 利用而偏离正确性。  
**explained variance**：value head 对 outcome 方差的解释比例，作为 critic 训练质量指标。  
**debiasing**：在正确/错误类内分别估计长度偏移，保留类间差距而消除类内长度趋势。

## 可复现要素
- 数据集：OpenR1-Math-220k 子集、DAPO-Math-17k；训练日志、校准 artifacts、benchmark outputs 随 submission 公开。
- 代码与权重：论文声明每个数字均可由附录与归档文件复现，提供脚本可从日志重绘所有图表。
- 关键超参：actor lr=1e-6、critic lr=1e-5、clip=0.2、γ=λ=1、温度 1.0、prompt cap 2,048、train cap 5,120/4,096、val cap 12,288、每步 128×8 rollout、单步单更新；详见附录 Table 6。
