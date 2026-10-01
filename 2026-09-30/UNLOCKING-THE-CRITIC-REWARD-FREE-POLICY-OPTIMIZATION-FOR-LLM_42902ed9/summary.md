---
title: "UNLOCKING-THE-CRITIC-REWARD-FREE-POLICY-OPTIMIZATION-FOR-LLM"
source: https://arxiv.org/pdf/2609.37119v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:57:36"
field: "大语言模型强化学习与后训练"
keywords: ["reward-free RL", "critic-based policy optimization", "LLM post-training", "chain-of-thought reasoning", "value network reuse", "length bias calibration"]
innovations: ["证明 critic-based RL 长链 CoT 不稳定主要源于更新配方而非价值网络本身", "将单一冻结 critic 同时复用为 rollout 奖励、GAE 基线与未完成前缀成功预测器", "通过同 outcome 类内长度去偏 + 二值化阈值阻断策略对长度偏差的利用"]
benchmarks: ["AIME 2025", "AIME 2026", "AMC 2023", "GPQA-Diamond"]
---

# 论文速读：UNLOCKING THE CRITIC: REWARD-FREE POLICY OPTIMIZATION FOR LLM POST-TRAINING

## 一句话总结
论文提出 **RFPO（Reward-Free Policy Optimization）**，将预训练的 critic 网络冻结并校准后复用为 rollout 级奖励、GAE 基线以及未完成前缀的成功预测器，在 RL 训练循环中完全无需外部验证器标签；在数学推理任务上，二值化版本的 RFPO 以约 **19% 更少 GPU 小时** 追平全监督 PPO，且天然适配截断 rollouts 的长推理场景。

## 研究问题与动机
- **现状趋势与浪费**：近期 LLM RL 后训练流行移除 critic（用多采样 baseline 替代），理由是 critic 显存/计算开销大且估计不可靠；即便训练了 critic，训练结束后也被丢弃，但其已学会预测轨迹结果，该能力未被充分利用。
- **critic 不稳定的本质**：学界将 critic-based RL 在长链 CoT 中的不稳定归咎于价值网络本身；本文发现这主要是**优化配方**问题——保持策略更新幅度小、低方差即可恢复稳定收敛，而非 critic 不可用。
- **奖励信号昂贵且不可扩展**：人类反馈、 outcome/process 奖励模型均需额外标注项目；可验证奖励又要求每题有参考答案；随着推理轨迹变长，**生成成本**占比主导，而稀疏终端奖励导致信用分配困难，且现有方法无法对未完成轨迹打分。
- **核心设问**：若一个 critic 已足够可靠到可作为 baseline，它是否也足够可靠到**直接充当奖励**？

## 核心贡献（创新点）
1. **揭示 critic 不稳定性为优化伪影**：通过控制更新步数（每 batch 单次更新）与方差，监督 PPO 与无标签变体均可在数百步内稳定训练，打破“critic 不适合长 CoT"的既有假设。
2. **单一冻结 critic 承担三重角色**：同一 $ \bar{V}_\phi $ 同时作为 rollout 级奖励、GAE 基线与未完成前缀的成功先验预测器，实现训练时无 verifier、无价值网络反向、无需等待轨迹完成。
3. **去偏 + 二值化闭环长度利用**：提出 $ \hat{r}=\mathbf{1}[v-b(\ell)>\tau] $，在同一正确性类内估计长度十分位偏移 $b(\ell)$ 以剥离长度偏差，再以分位数阈值 $\tau$ 二值化；连续版虽能短暂超越监督 PPO（AIME 2026 pass@1 达 29.2%），但策略会沿剩余长度偏差漂移并衰减，二值化则换取长期稳定。
4. **零标签追平全监督且节省算力**：在 5,120-token cap 下，RFPO（0% 标签）宏平均 pass@1 达 41.1%，与监督 PPO 的 41.8% 接近；在 4,096-token cap（约半数 rollout 未完成）下，RFPO 宏平均 47.6% 高于同 cap 监督 PPO 的 46.6%，且 300 步仅需 264 GPU-h vs. 327 GPU-h（**-19%**）。
5. **提前为未完成轨迹定价**：critic 在 4,096 token 处对同题尝试的 AUC 已达 0.84，对耗尽预算的 rollout 也能给出与已完成 rollout 相当 Ranking，使训练不必等待所有轨迹完结即可更新。

## 方法详解
- **Critic 来源**：在 DAPO-Math-17k 上做两段监督 PPO——Phase 1（500 步，8,192-token 预算、64 prompts/步）后接 Phase 2（300 步，5,120-token cap、128 prompts/步）。累计 step 700 取初始策略 $ \pi_{\theta_0} $，step 800 取冻结 critic $ \bar{V}_\phi $；全程仅约 563,200 条 verifier 标注轨迹用于预训练。
- **固定点含义**：当 $ \gamma=\lambda=1 $ 时，critic 回归目标为轨迹终端正确性，其不动点近似为 $ V_\phi(x,y_{\le t})\approx \Pr(\text{success}\mid x,y_{\le t}) $。
- **校准公式（唯一在线依赖）**：
  $$\hat{r}(x,y)=\mathbf{1}\big[v(x,y)-b(\ell(y))>\tau\big]$$
  其中 $ v=\bar{V}_\phi(x,y_{\le T}) $ 为终端价值；$ b(\ell) $ 在正确/错误两类内分别按长度十分位估计偏移后再平均，以剥离"outcome 固定时仍随长度下降”的偏差；$ \tau $ 取校准集上校正后分数的 $ (1-q) $ 分位数（$ q $ 为 verifier 正样本率），部署点 $ \tau=0.5215 $，在 512 条 on-policy 校准 rollout 上与 verifier 一致率达 93.4%。两项均**一次性拟合、不再更新**。
- ** Advantage 简化**：因 $ \gamma=\lambda=1 $ 且奖励来自同一 frozen critic，GAE 退化为 $ A_t=\hat{r}-\bar{V}_\phi(x,y_{\le t}) $（归一化前）；Actor 使用 clipped PPO surrogate 更新，$ \phi $ 永不更新。
- **部分标签插值**：引入监督比例 $ p $，每条 trajectory 以 Bernoulli(p) 决定是否替换为真 verifier 标签 $ r^\star $；$ p=0 $ 为全 reward-free，$ p=1 $ 回落为监督 PPO。
- **稳定训练规则**：每 batch 仅做 **1 次梯度更新**（mini-batch = 全 batch，无内循环）， clip ratio=0.2、$ \epsilon=0.2 $、actor lr=$ 10^{-6} $、critic lr=$ 10^{-5} $（仅监督臂使用）、无 KL/entropy penalty；该规则同时使监督 PPO 与 RFPO 均稳定数百步。

## 实验与结果
- **模型与数据**：Qwen3-4B-Base → 在 OpenR1-Math-220k 子集（45k 长 CoT，三 epoch）SFT → 进入 DAPO-Math-17k 监督 PPO 获取 $ (\pi_{\theta_0},\bar{V}_\phi) $。
- **评估基准**：AIME 2025/2026、AMC 2023、GPQA-Diamond；每 10 步在 12,288-token 预算、8 samples/problem、temp=0.7 下验证并报告宏平均 pass@k。
- **主要结果（5,120-token cap，RL step 300 独立评测，16 samples）**：
  - 监督 PPO（100% 标签）：AIME25 25.2 / AIME26 21.7 / AMC23 73.6 / GPQA 46.6，Avg **41.8%**
  - RFPO（50% 标签）：Avg **41.3%**
  - RFPO（0% 标签，mean of 3 runs）：Avg **41.1%**（AIME25 +0.2、AIME26 持平、GPQA +0.7、AMC23 -3.6）
  - 训练期峰值（Table 3）：RFPO 0% labels 在 AIME 2026 pass@1 达 **26.2%**，高于监督 PPO 的 25.0%。
- **短 cap 鲁棒性（4,096-token cap，~50% rollout 未完成）**：RFPO 宏平均 47.6% vs. 监督 PPO 46.6%，300 步 264 vs. 327 GPU-h。
- **连续奖励 vs. 二值化**：连续去偏奖励可达 AIME 2026 pass@1 = **29.2%**（超过全监督 PPO 25.0%），但后期回落下滑至 16.2% 并伴随长度漂移（+33.6%）与回答重复率飙升；二值化舍弃峰值换取稳定。
- **算力节省**：单步时间 421s vs. 582s（-28%），吞吐 1,122 vs. 828 tok/s（+35%），每卡峰值保留内存降 9.4 GB（critic 优化器状态与反向传播消除）。

## 相关工作脉络
1. **Process Reward Models（Lightman et al., 2024; Wang et al., 2024）**：提供每步密集信号但需步骤级标注且只评判已写步骤；RFPO 的 critic 从结果中学习并可**预测**未完成的 prefix。
2. **标签替代方案（多数投票、self-rewarding、置信度、generated rubrics）**：脱离 grounded supervision 易诱发 reward hacking；RFPO 的奖励来自**一次冻结且经校准的 verifier 分布**，受有界约束。
3. **隐式奖励 DPO/PRIME**：训练仍需标签；RFPO 在 RL 阶段完全不依赖新标签。
4. **VinePPO（Kazemnejad et al., 2025）、EVPO（Pan et al., 2026）**：通过权重解释方差或优化配方保留 critic，但奖励仍由外部提供；本文把 critic 自身**直接当作奖励**。
5. **无价值网络的推理训练（DeepSeek-R1、Kimi、Tulu 3 等）**：以移除 critic 换取稳定；本文证明稳定性的根源在更新规则而非网络本身，且 critic 可在冻结后充当 reward。
6. **长 rollout 稀疏奖励处理（Sparrow、截断/稀疏注意力）**：仅缓解等待成本却无法给未完成轨迹定价；critic 的前缀预测能力填补这一空白。

## 局限性与未来方向
- **规模与领域局限**：实验仅限 Qwen3-4B 与数学推理；未验证更大参数规模及非推理任务（如 agentic、代码、对话）。
- **离线校准漂移**：阈值 $ \tau $ 在策略分布移动后出现 false positive 上升（奖励正率较 verifier solve rate 拉开 4–5 点），当前靠二值化+单标签保护；缺乏动态重校准机制。
- **峰值-稳定权衡**：连续 critic 奖励能触及更高精度天花板，但需可靠的早停/稳定化信号（文章提示“回答复述率”可作为预警指标），尚未工程化。
- **准备成本**：critic 预训练需 800 步监督 PPO（~901 GPU-h），对资源受限场景仍是门槛。
- **未来方向**：扩展至更大模型与多领域；开发稳定跟踪连续奖励峰值的停止规则；研究动态/在线校准与 partial-label 调度策略。

## 研究启发与可借鉴点
1. **“单步更新 + 全 batch mini-batch" 是稳定 critic-based RL 的关键开关**：该规则同时使监督 PPO 与 reward-free 变体稳定，可作为后续工作基线 recipe；对 REINFORCE++ 等单步方法亦有借鉴。
2. **用 "explained variance" 作为训练期 critic 质量的廉价代理**：离线 within-problem AUC 与 EV 相关系数达 0.91，便于在线监控而不必频繁跑 AUC。
3. **去偏 + 二值化的双层防护范式**：先在同 outcome 类内剥离可控维度（长度）的系统偏移，再以指示函数封顶消除剩余利用通道；可迁移至其他易被策略利用的连续标量信号。
4. **用前缀预测实现“未完成 rollout 即时定价”**：在长 horizon 任务中，以 $ \bar{V}_\phi(x,y_{\le t}) $ 对截断轨迹打分，避免等待终端结果，可直接用于 agent rollout、代码生成等延迟敏感场景。
5. **部分标签作为安全旋钮**：$ p \in [0,1] $ 允许在零标签效率与全标签鲁棒性间连续插值；当 critic 出现校准漂移时，少量 verifier 标签即可稳住性能。

## 关键术语表
- **RFPO（Reward-Free Policy Optimization）**：将单一冻结且校准后的 critic 同时用作 rollout 奖励、GAE 基线与未完成前缀成功预测器的策略优化框架。
- **GAE（Generalized Advantage Estimation）**：Schulman 提出的优势函数估计，通过 $ \gamma $ 与 $ \lambda $ 折衷偏差-方差；本文取 $ \gamma=\lambda=1 $ 后退化。
- **Explained Variance（EV）**：价值头对终端奖励方差的解释比例，$ 1-\mathrm{Var}(y-v)/\mathrm{Var}(y) $，本文用作训练期 critic 质量的廉价在线指标。
- **Length bias / 长度偏差**：critic 原始分数与响应长度之间的负相关，既混杂于正确性分布又独立存在，会被策略直接利用以“走捷径”。
- **Calibration（校准）**：在固定 rollout 样本上估计去偏偏移 $ b(\ell) $ 与二值阈值 $ \tau $，使奖励正率与 verifier 正率对齐。
- **Within-problem AUC**：在同一题目内比较正确/错误 rollout 的 ranking 能力，剔除“区分题目难度”的混杂因素，更能反映奖励可用性。
- **Clipped PPO surrogate**：目标函数中对 importance ratio 施加双边 $ \epsilon $ clip，限制策略更新幅度；本文 clip=0.2。
- **Answer restatement**：响应中重复声明答案的次数；在连续奖励衰退实验中与 held-out 准确率呈 -0.87 相关，可作为早期退化预警信号。

## 可复现要素
- **数据集**：OpenR1-Math-220k（公开，Hugging Face）、DAPO-Math-17k（公开）、AIME 2025/2026、AMC 2023、GPQA-Diamond（公开）。
- **代码/权重**：论文 Appendix H 给出每个 arm 对应的训练日志路径；附录打包内含全部日志、校准 artifact、score 汇总、benchmark 输出及图表复现脚本；模型权重指向 Qwen3-4B-Base（官方开源）。**未明确提供单独开源仓库链接**，但声明“每个数字均可由提交材料复现”。
- **关键超参**：prompt cap 2,048；train/val generation cap 5,120/12,288（4,096 实验另行 refit 校准）；batch=128 prompts × 8 responses；temperature 1.0（训练）/0.7（验证）；clip 0.2；$ \gamma=\lambda=1 $；actor lr $ 10^{-6} $、critic lr $ 10^{-5} $；AdamW $ \beta=(0.9,0.999) $、weight decay 0.1、gradient clip 1.0；单更新/无内循环；无 KL/entropy penalty；bf16 + FSDP + sequence parallelism 2 + gradient checkpointing；8×A100-80GB。
- **校准细节**：512 条 on-policy rollout；$ b(\ell) $ 按长度十分位在正确/错误类内分别估计（Table 7 给出每分位偏移）；$ \tau=0.5215 $；一致率 93.4%。
