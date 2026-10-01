---
title: "VSTRESS-CORRELATION-AWARE-AUDITING-ANDADAPTIVE-BUDGET-ALLOCA"
source: https://arxiv.org/pdf/2609.36958v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:59:09"
---

# 论文速读：VSTRESS: CORRELATION-AWARE AUDITING AND ADAPTIVE BUDGET ALLOCATION FOR REPEATED VERIFIERS

## 一句话总结
本文提出 VSTRESS 可审计回放合约与 VSTRESS-CA 相关性感知分配策略，通过密封校准集估计未查询校验通道的条件边际信息量，在固定调用预算下自适应选择通道、动态停止或弃权，并将校验误差相关性从“事后警告”转化为“可审计的在线分配决策”，在多项式相关误差边界与下游 RLVR 任务上显著优于固定多数投票与等预算广度/冗余基线。

## 研究问题与动机
- 重复调用验证器仅在额外视图携带独立条件信息时才有效；若错误同源或高度相关，追加调用仅增加成本而不提供新证据，甚至因相关性反转损害最终决策质量。
- 现有验证器评估与选择性预测工作多关注整体准确率或单通道边际质量，缺乏统一的“后决策 Oracle 接入”审计边界，难以在线区分“边际质量”与“条件边际价值”。
- 部署时通道间的依赖结构可能偏离校准分布（dependence shift），现有方法缺少可观测的漂移报警与保守回退机制，导致校准好的偏好策略在外推时失效。
- 固定预算下的“广度（更多 item 单调用）vs 冗余（更少 item 多调用）”对比缺乏统一计费与失败记账口径，难以公平判断自适应策略的真实前沿。

## 核心贡献（创新点）
1. **可审计二进制反馈合约**：在线聚合器在写入决策与成本账本后冻结账本，再与隔离的干净 Oracle 对接，防止事后打分反向污染在线投票。
2. **条件边际可分性度量**：以密封校准集估计 $D(j|S)=I(Y;V_j|V_S)$ 作为未查询通道的估值量，区分“边际准确率”与“已有证据之外的条件信息增益”。
3. **VSTRESS-CA 自适应分配策略**：构建成本归一化、Bootstrap 不确定性折扣的效用函数 $U(j|S)$，支持阈值停止、预算停止与相关性漂移fallback 到 exact-stop。
4. **机制到学习的端到端验证**：在受控相关误差、重复真实校验器、以及下游 RLVR  learner 固定 seed/optimizer/task split 的匹配预算对比中，证明条件信息驱动的选择可提升 quality–coverage–cost 前沿。

## 方法详解
- **四阶段协议**：① 校准：在密封 split 上估计每通道可靠性、失败率、成本与条件依赖；② 在线获取：VSTRESS-CA 在预算 $B$ 内按 $U(j|S)$ 选通道，每次原始裁决、失败码、延迟、计费写入 ledger；③ 冻结与离线上报：控制器先冻结预测/弃权状态/已获取集合，再 join 干净 Oracle 计算质量–覆盖率–成本向量；④ 下游复用：相同 ledger  schema 同时用于重复校验器分析与 RLVR 训练。
- **腐坏与聚合模型**：对称腐坏以率 $\rho$ 翻转每条视图标签；多数决策 $\hat{y}=\mathbb{1}[z\ge 3]$（$z=\sum_j v_j$），固定弃权规则仅当 $a=\max(z,5-z)/5\ge 0.8$ 时接受；独立视图假设下多数错误概率为 $\sum_{j=3}^5 \binom{5}{j}\rho^j(1-\rho)^{5-j}$，但部分相关可在 $\rho<1/2$ 时破坏该直觉。
- **VSTRESS-CA 核心公式**：对已查集合 $S$，未查通道 $j$ 的条件边际可分性 $D(j|S)=I(Y;V_j|V_S)$ 由校准行估计；引入 Bootstrap 标准误 $\widehat{\sigma}_{j,S}$ 与校准成本 $\widehat{c}_j$，定义效用
  $$U(j|S)=\frac{[\widehat{D}(j|S)-\beta \widehat{\sigma}_{j,S}]_+}{\widehat{c}_j+\epsilon},\quad j_t^*=\arg\max_{j\notin S_t} U(j|S_t).$$
  即仅对“不在已有视图中”的信息付费，模型/提供商身份不作为独立性代理。
- **停止与漂移回退**：每获一手视图后计算后验 $\widehat{p}_t=P(Y=1|V_{S_t})$，若 $\max(\widehat{p}_t,1-\widehat{p}_t)\ge \tau$ 则接受；否则仅在 $|S_t|<B$ 且 $\max_{j\notin S_t}U(j|S_t)>\gamma$ 时继续，否则弃权。部署窗口以 Jensen–Shannon 统计 $S_{\mathrm{shift}}$ 比对裁决频率/分歧/失败率与校准分布，当 $S_{\mathrm{shift}}>\delta$ 时禁用通道偏好并回退到 exact-stop，避免在校准支撑外推。
- **合约与复杂度**：在线路径每条 item 最多 $O(k)$ 次调用与 $O(k)$ ledger 字段；离线上报按 manifest key 排序后 join 为 $O(n)$；失败（超时/格式非法/不可用）同样计费并记录失败码，保持 fail-closed 轨迹。

## 实验与结果
- **设置**：512 条有序 fixture，GSM8K 提供不可变标识；7 个确定性腐坏 seed；对比基线含 majority-5、随机分配、边际准确率贪婪、忽略条件依赖的边际信息策略、等预算广度/冗余与 equal-accepted-update 控制。
- **依赖诊断**：Table 1 显示跨家族通道 error overlap 最低（0.3187）、Phi 0.6924、Kappa 0.7368、MI 0.3975、$\Delta$D-gain 最高（0.0913）；同模型重复 overlap 达 0.7826、gain 仅 0.0126。
- **质量–覆盖–成本前沿**：对称 35% 腐坏下 majority-5 将 BA 从 0.6578 提升至 0.7739（+0.1161），coverage 0.4919，sel. acc. 0.8953，calls=5；对称 65% 时 BA 降至 0.2362，损失 -0.1226。
- **等预算自适应 vs 广度/冗余**（Table 2/45）：breadth（1 次/item）BA 0.6048、RLVR 0.5826；redundancy（5 次/item）BA 0.6375、RLVR 0.6148；VSTRESS-CA（3.4216 次/item）BA 0.6538、sel. acc. 0.6892、RLVR 0.6417；equal-accepted-update 控制 BA 0.6319、RLVR 0.6073。
- **真实校验器 held-out**（Table 4/13）：低相关通道 5 次调用 BA 0.8017、coverage 0.5874、sel. acc. 0.9186、cost 4.96×；共同成因通道 BA 0.7224、coverage 0.9048、sel. acc. 0.7489、cost 4.91×；单调用参考 BA 0.7186。
- **下游 RLVR**（Table 4/6/14/32）：held-out tasks 上 single-call 0.6127、majority-5 0.6429（+0.0302）、sequential-safe 0.6408（calls 4.6934、cost 4.52×）；36k 步延长稳定期 majority-5 达 0.6537（+0.0348）、sequential-safe 达 0.6516；verifier shift 下 0.6287、cost shift 下 0.6418 均保持正收益。
- **停止策略对比**（Table 9/30）：exact-stop 保持 majority 决策同时降 calls（如对称 35% 由 5.0000 降至 4.7999）；prefix-stop 进一步降至 4.0477 但 sel. acc. 下降；adaptive-confidence 以 3.8641 calls 取得 sel. acc. 0.9026。
- **校准样本效率**（Table 46）：calibration $n$ 从 32 增至 512，CMD error 从 0.0867 降至 0.0000，BA 从 0.6269 升至 0.6538，RLVR 从 0.6106 升至 0.6417。
- **相关性漂移测试**（Table 47）：JS  statistic 从 0.0143（none）升至 0.2269（severe）时，no fallback 的 BA 降至 0.5869，启用 fallback 后恢复至 0.6206，证明保守回退有效。
- **跨域/难度迁移**（Table 28）：math/hard  paired gain +0.0716；sequential-safe 保留 majority 质量的 ≥98.4% 并节省 0.4873 calls/item。
- **失败分类与人类裁定**（Table 40/31）：独立分歧占比最高（29/512，0.0566）；高 margin accepted  human–oracle agreement 达 0.9478，abstained  stratum 为 0.8117，支持弃权作为质量控制信号。
- **最强结果**：VSTRESS-CA 在等预算固定调用下取得 BA 0.6538、RLVR 0.6417、calls/item 3.4216；real-verifier 低相关通道 BA 0.8017、sel. acc. 0.9186；36k 步 RLVR 延长实验中 majority-5 达 0.6537。

## 相关工作脉络
- **Reward-model/verifier 评估**（Lambert et al., 2024; Zheng et al., 2024; Skalse et al., 2022）：关注 held-out 质量与系统失败模式，但未给出“后决策 Oracle 接入”的审计边界与在线条件信息信号。
- **RLVR 与可验证奖励**（Lightman et al., 2024; Chen et al., 2025; Patel et al., 2026）：使用可验证奖励训练推理/代码系统；本文聚焦重复调用聚合机制本身，而非提出新 learner。
- **选择性预测与拒绝选项**（Geifman & El-Yaniv, 2019）：形式化 abstention；本文将其与成本归一化、条件 MI、漂移报警结合为固定预算分配问题。
- **相关误差与校验鲁棒性**（Kim et al., 2025）：外部最相近对比点；本文的关键区别在于用校准估计的 $I(Y;V_j|V_S)$ 指导在线 channel 选择，并以 exact-stop fallback 处理部署依赖偏移。
- **成本感知路由与序列决策**：动机来源之一；本文差异是将“实现多样性≠统计独立”显式度量，并在同一总调用预算下比较 breadth vs redundancy vs adaptive。

## 局限性与未来方向
- 条件信息估计依赖校准 split；通道池大、校准样本少时估计噪声显著，论文列出样本效率表但未声称最优样本复杂度。
- 低测量相关性不等于因果独立；共享训练数据、推理模板、基础设施或失败根因可能隐藏在可观测摘要之外。
- 当前评估限定二元决策、有界通道池与匹配候选 cache 的下游 learner；多分类与更大池需要结构化估计器。
- 日志包含原始裁决、实现 ID、延迟与失败码，发布前需访问控制与去敏感；共同提供商偏差在扩展部署中可能集中。
- 未来方向：可扩展至多分类/大规模通道池的结构化依赖估计；引入更丰富的漂移源（prompt/时间/对抗对齐）在线检测；与 process verifier 和工具调用链路联合审计。

## 研究启发与可借鉴点
- **审计合约设计范式**：先冻结在线证据与成本账本、再 join 干净 Oracle 的分离时序，能有效避免事后评分反向污染，适用于任何“重复验证/多次调用聚合”的在线系统。
- **条件边际信息作为采集信号**：用 $I(Y;V_j|V_S)$ 替代单通道边际准确率进行 next-channel 选择，能将“看似高准但错误同源”的通道自动降级，值得迁移到多模型路由、工具调用与多视角集成。
- **等预算 breadth–redundancy 对照**：将“单 item 单次”与“多 item 单次”放在同一总调用预算下比较，避免把质量增益错误归因于更少调用；该对照设计可用于任何 API-cost 敏感的多步验证 pipeline。
- **依赖漂移 alarm + 保守回退**：用 JS 统计监测部署分布相对校准的偏移，超阈值时关闭 Learned 通道偏好、退回 exact-stop，可在不重训的前提下提供可解释的安全兜底。
- **失败分类与弃权信号联动**：将超时/格式非法/共同成因/校准边界等失败显式分类并计入成本，配合 abstention 的 quality–coverage 联合报告，有助于在工程侧建立可追溯的质量 gate。

## 关键术语表
- **VSTRESS**：一种二进制反馈的可审计回放合约，在线聚合器冻结决策与成本账本后才接入干净 Oracle，确保事后评分不反向影响在线投票。
- **VSTRESS-CA**：相关性感知的自适应分配策略，基于校准估计的条件边际可分性、成本归一化与不确定性折扣，在固定预算下选择下一通道、决定停止或弃权。
- **Conditional marginal discriminability**：$D(j|S)=I(Y;V_j|V_S)$，衡量在未查询通道 $j$ 对标签 $Y$ 的贡献中，扣除已有集合 $S$ 已提供信息后的剩余可分性。
- **Balanced accuracy (BA)**：正负类召回的算术平均，用于在类别或拒绝分布不均时更稳定地反映整体判别能力。
- **Selective accuracy**：仅在被接受（未弃权）的子集上计算的准确率，与 coverage 联合报告以刻画质量–覆盖权衡。
- **Coverage**：被接受项数占原始 item 分母的比例；本文 abstention 定义在该分母上，避免以“排除困难项”虚增准确率。
- **RLVR**：Reinforcement Learning with Verifiable Rewards，利用可验证奖励信号训练推理/代码系统的下游学习范式。
- **Dependence-shift fallback**：当部署窗口的 JS 统计超过阈值时，禁用基于校准学习的通道偏好并回退到 exact-stop，避免在校准支撑外推。

## 可复现要素
- **数据集**：512 条有序 fixture（GSM8K 提供不可变标识），7 个确定性腐坏 seed；split 为 calibration/held-out 隔离，源 hash 与 payload digest 在附录 C.1/C.2 给出。论文声明补充材料含完整 fixture、corruption seeds、ledger、checkpoint 与 figure 资产。
- **代码/权重**：论文声明 Supplementary Material 包含代码、配置、replay schemas、表格/图示源与 artifact metadata；附录 C.1/C.13 给出五阶段 hash-checked 回放流程（fixture regeneration → corruption replay → aggregation replay → learner replay → table rendering），每阶段输入/输出行计数与 digest 一致，跨表数值交叉校验闭合。
- **关键超参**：多数投票视图数 $k\in\{1,3,5,7\}$；固定弃权阈值 $\tau=0.8$；VSTRESS-CA 校准样本 $n=512$ 时 CMD error 0.0000、calls/item 3.4216、BA 0.6538、RLVR 0.6417；漂移报警 JS 阈值在 mild/moderate/severe 三档分别为 0.0578/0.1216/0.2269；bootstrap 不确定性折扣系数 $\beta$ 与继续阈值 $\gamma$ 由校准 grid 选取、不在测试集上调参；成本归一化加小常数 $\epsilon$。

<!--META
{"keywords": ["repeated verifier", "correlation-aware allocation", "conditional mutual information", "selective prediction", "RLVR", "auditable replay contract", "cost-normalized stopping"],
