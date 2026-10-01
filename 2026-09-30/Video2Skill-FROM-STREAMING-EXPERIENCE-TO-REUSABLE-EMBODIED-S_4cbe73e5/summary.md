---
title: "Video2Skill-FROM-STREAMING-EXPERIENCE-TO-REUSABLE-EMBODIED-S"
source: https://arxiv.org/pdf/2609.36691v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:59:19"
field: "具身智能技能发现"
keywords: ["Streaming Embodied Skill Discovery", "Video2Skill", "Vision-Language Models", "Skill Library", "Creation-Reuse Decision", "Embodied Skill Discovery"]
innovations: ["形式化流式具身技能发现(SESD)任务并构建Video2Skill基准，联合评测事件定位、命名不变分组和创建-复用决策", "揭示统一/因式分解两种范式的系统性偏差(过度合并vs碎片化)及规模增长的局限", "提出CLARE反事实库状态重建等三种监督策略，发现新颖性识别是剩余核心瓶颈"]
benchmarks: ["ROBOINTER", "HD-EPIC"]
---

# 论文速读：Video2Skill: From Streaming Experience to Reusable Embodied Skills

## 一句话总结
本文提出**流式具身技能发现（Streaming Embodied Skill Discovery, SESD）**任务，并构建 **Video2Skill** 基准评测集，系统性评估 19 个开源视觉语言模型能否从连续视频流中积累并维护一个持久的符号化可重用技能库；研究发现当前 VLM 普遍在"何时复用已有技能、何时创建新技能"这一核心判断上存在严重不足，监督微调可改善技能分组但难以解决新颖性识别瓶颈。

## 研究问题与动机
- **问题**：具身智能体依赖技能进行规划，但技能必须先被"发现"才能被使用。现有的技能复用方法依赖预定义技能库或通过代码执行/环境反馈扩展，无法解决从**被动视觉流中在线累积符号化技能库**的问题。
- **现有方法不足**：① 多数方法依赖预设行为库或环境反馈信号，而非仅从视频观察中学习；② VLM 能描述单个操作事件，但缺乏将连续事件组织为持久技能抽象的能力；③ 统一处理容易过度合并不同变换，因式分解处理容易为重复变换创建重复模式——两种范式各有系统性偏差；④ 模型即使经过监督训练，也几乎无法为训练中未见过的变换分配新技能。

## 核心贡献（创新点）
1. **形式化 SESED 任务并构建 Video2Skill 基准**：首次将"从流式视觉经验中维护持久技能库"定义为可系统评估的任务，覆盖机器人桌面操作与人类厨房活动两个领域，同时评测事件定位、命名不变的技能分组、创建/复用决策三个维度。
2. **揭示统一与因式分解范式的系统性偏差**：统一模型倾向过度合并（高复用召回率、低创建召回率），因式分解模型倾向碎片化（高创建召回率、低复用召回率），且模型规模增大并不带来一致的聚类质量提升。
3. **提出三种监督微调策略并系统比较**：Oracle-history SFT、Counterfactual Library-state Rebalancing (CLARE) 和 On-policy correction，表明监督可改善技能分组，但训练后技能库仍停滞在参考规模的不到一半。
4. **揭示深层瓶颈：新颖性识别仍是核心难题**：即使训练提高了分组质量，模型对训练未见过的变换仍能定位其时间，却几乎不会为其分配新技能名；高频技能改进显著，低频技能改进有限。

## 方法详解
**问题形式化**：将有序视频流 $\mathcal{V} = (v_1, \ldots, v_N)$ 分块处理，在步骤 $t$ 接收视频块 $x_t$ 和状态 $z_{t-1} = (\mathcal{L}_{t-1}, u_{t-1})$（持久技能库 + 可选未完成事件），输出 $(\mathcal{D}_t, \mathcal{E}_t, u_t)$，其中 $\mathcal{D}_t$ 为新创建模式、$\mathcal{E}_t$ 为已完成事件。状态更新为 $\mathcal{L}_t = \mathcal{L}_{t-1} \cup \mathcal{D}_t$。事件等价性基于底层物理变换 $\tau(e)$ 而非模式名称。

**基准数据集**：
- **ROBOINTER**：机器人桌面操作视频（来源 DROID / RH20T），训练集 2992 视频 / 5032 已验证调用，测试集 132 视频 / 534 已验证片段，覆盖 15 个规范变换类。
- **HD-EPIC**：第一人称厨房活动视频，训练集 42 视频 / 5726 调用，测试集 40 片段 / 1137 参考片段，覆盖 46 个规范变换类。

**三种监督策略**（共享目标 $\mathcal{L}_{SFT} = -\mathbb{E}\log p_\theta(y_t^* | o_t, z_{t-1})$）：
- **Oracle-history SFT**：用参考标注重放时序列构造训练样本，库状态来自前序参考输出。
- **CLARE（反事实库状态重新平衡）**：在保持观测不变的情况下编辑库状态（删除必需模式/添加语义无关条目），监督模型对关键变更敏感、对无关变更鲁棒。
- **On-policy correction**：让学生模型在训练序列上自生成轨迹，用教师对实际到达状态的标签进行微调，覆盖学生累积错误。

**两种推理范式**：
- **Unified**：VLM 同时定位事件并更新技能库（视频+库状态联合输入）。
- **Factorized**：先无库访问地生成文本事件描述，再用同一模型仅从文本更新库。

**评估指标**：① **Coverage**：基于 tIoU≥0.3 的最大匹配，衡量参考事件被恢复的比例；② **Pair P/R 与 ARI**：命名不变的技能分组一致性；③ **Create/Reuse Recall**：衡量模型正确创建新技能和复用过有技能的决策质量。

## 实验与结果
**数据集**：ROBOINTER（机器人桌面操作）、HD-EPIC（第一人称厨房活动）。

**基线模型**：19 个开源 VLM（Qwen3.5 系列 0.8B~35B-A3B、Qwen3-VL、Qwen3.8、InternVL3.5 系列 4B~38B、Ovis2.5-9B、Cosmos-Reason2 系列 8B/32B、LLaVA-OneVision-2-8B、GLM-4.1V-9B、MiniCPM-V-4.5、Gemma-4-31B）。

**主要结果**：
- **零样本**：多数模型技能分组接近随机水平（ARI 普遍 ≤27，HD-EPIC 上多模型 ARI < 10）；规模增大不带来一致提升。
- **最强分组结果**：统一范式下 LLaVA-OneVision-2-8B 的 **Pair Recall = 100%**（ROBOINTER），但 Pair Precision 仅 22.5%，ARI = 0.0——说明过度合并严重；统一 CLARE Qwen3.5-4B 在 HD-EPIC 上 ARI 达 31.0，为最高之一。
- **监督最佳**：Oracle-history SFT 在统一范式下实现最高 ARI（LLaVA-ROBOINTER: 60.9，Qwen-HD-EPIC: 31.0）；CLARE 在 Qwen3.5-4B 统一 HD-EPIC 上提升 Creation Recall 至 48.4。
- **核心失败模式**：所有训练模型 **Reuse Recall ≥ 97%** 而 **Create Recall 仅 19.1%~48.4%**，说明模型倾向于复用而非创建。
- **训练未见变换**：11 个 HD-EPIC 测试类（59 片段）未在训练中覆盖，监督后 Coverage 达 57.6%~66.1%，但 Create 仅 0~1/10~11 个类别获得新技能名。

## 相关工作脉络
1. **视频理解与流式模型**（Action recognition, temporal localization, streaming video models）：Video2Skill 与现有视频理解工作不同，后者评估预定义类别下的预测或基于问题的回答，而本文关注从连续观察中**动态构建和维护**持久技能抽象。
2. **无监督动作分割**（Sener & Yao, 2018; Kukleva et al., 2019）：该类工作在完整序列上离线聚类发现动作结构；Video2Skill 要求在线增量处理，并维护跨视频持久的技能库。
3. **技能发现与演进概念库**（XSkill, UniSkill, LOTUS）：XSkill/UniSkill 从人类和机器人视频学习共享技能表示用于下游控制；本文强调构建**符号化、带参数绑定**的可重组技能抽象，而非直接的控制策略。
4. **LLM Agent 技能扩展**（Voyager, SkillWeaver, JARVIS-1）：这些方法通过代码执行和环境反馈验证新技能；Video2Skill 完全基于**被动视觉观察**，不依赖环境交互信号。
5. **连续学习与新类别发现**（MetaGCD, Happy, Grow and Merge）：相关工作关注开放世界场景下的增量学习；Video2Skill 将其具体化为技能库的创建-复用决策问题，强调命名不变性分组。
6. **流式视频理解基准**（StreamScout, StreamingBench, OVO-Bench）：评估模型在观察到达时的增量理解能力；Video2Skill 进一步要求模型将观察**组织为持久结构化抽象**，而非仅响应即时查询。

## 局限性与未来方向
- **领域局限**：仅覆盖机器人桌面操作和第一人称厨房活动，户外等更广泛场景尚未探索。
- **技能抽象性质**：当前技能描述为观察到的物理变换，其**对下游规划和可执行控制的实际效用**尚待验证。
- **新颖性识别瓶颈**：模型能有效定位时间范围内的变换，却极少为未见过的变换创建新技能名，如何突破这一障碍是核心开放问题。
- **未来方向**：探索更有效的创建-复用决策机制、将技能库应用于下游任务规划验证、扩展到更多样的物理交互场景。

## 研究启发与可借鉴点
1. **CLARE 反事实库状态重建方法可迁移**：通过编辑库状态（删除/添加条目）构造对比训练样本，可推广至任何需要模型理解"上下文敏感性"与"无关信息鲁棒性"的场景，如长期记忆管理、文档整理系统。
2. **创建/复用评估分离的设计思路**：将 Coverage、Pair P/R、ARI 和 Create/Reuse Recall 独立报告而非单一指标，避免了"高分掩盖系统性偏差"的问题，值得在类似技能/概念发现任务中借鉴。
3. **统一 vs. 因式分解范式的系统性比较**：两种范式的对比揭示了错误模式的互补性（合并 vs. 碎片化），为设计混合架构或集成策略提供了明确方向。
4. **对频度不均匀训练收益的分析框架**：按训练频率分组报告性能差异（高频改善显著、低频改善有限），揭示了技能学习中数据多样性的关键作用，可作为后续研究的诊断工具。
5. **On-policy correction 的自生成轨迹微调整体思路**：对学生累积错误的针对性监督（而非仅参考轨迹），与 on-policy 蒸馏思路一致，可迁移至需要长程依赖的连续决策任务。

## 关键术语表
- **Streaming Embodied Skill Discovery (SESD)**：流式具身技能发现，指模型从连续视频观察中增量识别操作事件并维护持久可重用技能库的任务。
- **Unified vs. Factorized Paradigm**：统一范式（VLM 同时定位事件与更新库）与因式分解范式（先生成文本描述再更新库）；前者易过度合并，后者易碎片化。
- **Pair Precision / Recall**：配对精确率/召回率，衡量预测分组与参考分组的一致性；低 P 表示过度合并，低 R 表示碎片化。
- **Creation / Reuse Recall**：创建召回率（首次出现参考类时正确创建新技能的概率）与复用召回率（已出现过时正确复用的概率）。
- **CLARE (Counterfactual Library-state Rebalancing)**：反事实库状态重新平衡，通过编辑库状态（删除/添加条目）构造对比训练样本的监督策略。
- **On-policy Correction**：基于学生模型自生成轨迹的状态分布进行微调，修正累积错误带来的训练-推理分布失配。
- **Canonical Transformation Class**：规范变换类，基准中人工定义的 48 个变换类别（如 grasp、transfer、place），作为命名不变的分组参考。
- **tIoU (temporal Intersection over Union)**：时序 IoU，用于计算预测时间段与参考时间段的时间对齐程度，阈值 0.3。

## 可复现要素
- **数据集**：HD-EPIC（公开）和 ROBOINTER（公开，来源 DROID / RH20T）；论文提供了从原始数据重新标注的 verified test split，原始标注未被直接使用。
- **代码/权重**：项目页面 https://andyzworks.github.io/video2skill，论文未明确说明代码是否开源；19 个 VLM 均为开源模型。
- **关键超参**：tIoU 阈值 0.3；统一/因式分解两种范式；三种监督策略（Oracle-history SFT / CLARE / On-policy correction）；两个骨干网络（Qwen3.5-4B 和 LLaVA-OneVision-2-8B）。
- **评估协议**：按视频顺序评估，测试集验证了参考类覆盖、命名不变分组和创建/复用决策三个维度。
