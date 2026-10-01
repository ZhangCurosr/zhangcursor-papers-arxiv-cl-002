---
title: "Video2Skill-FROM-STREAMING-EXPERIENCE-TO-REUSABLE-EMBODIED-S"
source: https://arxiv.org/pdf/2609.36691v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:59:12"
field: "具身智能与技能学习"
keywords: ["Streaming Embodied Skill Discovery", "Video2Skill", "Skill Library", "Vision-Language Model", "Creation-Reuse Decision", "Robot Manipulation", "Egocentric Video"]
innovations: ["形式化SESD任务并提出Video2Skill基准，统一评估时序定位、命名不变分组与创建复用决策", "揭示统一模型过度合并与因子化模型碎片化的对立失败模式", "提出CLARE反事实技能状态重平衡等三种监督策略并诊断训练后模型仍难识别未见变换的瓶颈"]
benchmarks: ["ROBOINTER", "HD-EPIC"]
---

# 论文速读：Video2Skill-FROM-STREAMING-EXPERIENCE-TO-REUSABLE-EMBODIED-S

## 一句话总结
本文提出**Video2Skill**基准与**Streaming Embodied Skill Discovery (SESD)** 任务，系统评估VLM能否从流式视频经验中增量构建并维护可复用技能库；在19个开源VLM上的实验揭示了统一模型易"过度合并"、因子化模型易"碎片化"的对立缺陷，且监督微调虽能改善分组但无法可靠解决"何时创建新技能"的核心挑战。

## 研究问题与动机
1. **规划依赖技能知识，但技能从何而来**：具身智能体需通过规划完成长程任务，而规划的前提是已知可用技能集合；人通过观察积累技能并复用于新场景，但机器人在流式观察中如何在线构建符号化技能库尚无系统性研究。
2. **现有方法要么依赖预定义技能库，要么依赖非被动信号**：已有工作或基于固定复用语料、或通过代码执行/环境反馈/任务成功信号/机器人轨迹来扩展技能库，未能解决"仅从被动视频流中构建持久技能库"的问题。
3. **VLM能描述事件，但缺乏将事件组织成可复用技能的证据**：VLM已能在开放词汇下描述操作事件，但是否能跨视频维持一致性技能抽象并做出"复用vs创建"决策仍未知。
4. **误差具有结构性且会累积**：在流式设定下，早期决策塑造后续观察的上下文，错误合并或重复创建会导致技能库偏差持续放大。

## 核心贡献（创新点）
1. **形式化SESD任务并构建Video2Skill基准**：将技能发现定义为在线时序定位+命名不变分组+创建/复用决策的统一框架，基于ROBOINTER和HD-EPIC构建跨机器人/人类场景的评测集，与仅做预定义类别动作识别或离线聚类的工作形成本质区别。
2. **揭示统一vs因子化范式的对立失败模式**：发现统一模型因联合感知与库更新倾向于过度合并（高复用召回、低创建召回、低Pair Precision），因子化模型因分阶段处理倾向于频繁创建重复schema（高创建召回、低复用召回），这种范式依赖性在已有工作中未被系统刻画。
3. **提出并比较三种监督策略（Oracle-history SFT / CLARE / On-policy Correction）**：引入反事实技能状态重平衡（CLARE）与学生生成历史的on-policy校正，揭示监督虽提升分组质量但仍无法可靠解决创建-复用边界判定，深化了对"技能库维持"问题的理解。
4. **提供多维度诊断分析（视频顺序影响、训练频率异质性、未见类别新颖性识别）**：发现训练一致性提升但库规模停滞于参考一半以下、高频技能获益明显而低频及未见变换几乎无法获得新schema，为后续研究明确了瓶颈所在。

## 方法详解
- **SESD形式化**：模型按序处理视频流 $\mathcal{V}=(v_1,\dots,v_N)$，在步骤 $t$ 接收片段 $x_t$ 与状态 $z_{t-1}=(\mathcal{L}_{t-1},u_{t-1})$（持久技能库+可选未完成任务），输出 $y_t=(\mathcal{D}_t,\mathcal{E}_t,u_t)$，其中 $\mathcal{D}_t$ 为新schema、$\mathcal{E}_t$ 为已定位事件、$u_t$ 为跨片段事件延续；状态按 $\mathcal{L}_t=\mathcal{L}_{t-1}\cup\mathcal{D}_t$ 更新。schema为语义变换抽象（含自由名称、定义、类型化参数槽），非可执行控制器；参考等价性基于底层变换 $\tau(e)$ 而非schema名称。
- **评估三维度**：
  - **Temporal Coverage**：以tIoU≥0.3匹配参考段与预测段，Coverage=匹配参考数/总参考段数（未匹配预测不计罚）。
  - **Skill Grouping**：计算参考对集合 $\mathcal{Q}_{\mathrm{ref}}$ 与预测对集合 $\mathcal{Q}_{\mathrm{pred}}$ 的Precision/Recall（跨视频配对），以及Adjusted Rand Index (ARI)。
  - **Creation-Reuse Decision**：按匹配序列顺序统计首次出现指示 $g_i$ 与 $\hat{g}_i$，报告Create R、Reuse R与决策准确率。
- **两种推理范式**：**Unified**（统一）由单一VLM同时接地事件与管理库；**Factorized**（因子化）先无库访问地生成文本事件描述，再用同模型纯文本元策略更新库。
- **三种监督策略**（统一SFT目标 $\mathcal{L}_{\mathrm{SFT}}=-\mathbb{E}\log p_\theta(y_t^*|o_t,z_{t-1})$）：
  - **Oracle-history SFT**：用参考回放产生的真实历史状态训练。
  - **CLARE**：在观测固定下编辑库状态（删除必要schema→改变reuse为create；添加无关条目→保持正确分配），增强对相关变化的敏感度与对无关变化的鲁棒性。
  - **On-policy Correction**：用学生自身rollout到达的状态求teacher标签，使训练分布覆盖推理时实际遇到的错误累积状态。

## 实验与结果
- **数据集**：**HD-EPIC**（40 test clips, 1137 reference segments, 46 canonical classes）与 **ROBOINTER**（132 test videos, 534 verified segments, 15 canonical classes），二者在source-video级别完全分离。
- **基线覆盖**：19个开源VLM在统一/因子化两种范式下的zero-shot性能（表1）。
- **Zero-shot关键数字**：
  - 统一Qwen3.5系列在ROBOINTER上Coverage最高达63.7%（Qwen3.8-27B），但Pair ARI仅34.2；HD-EPIC上Pair Recall普遍低于33。
  - 因子化LLaVA-OneVision-2-8B在ROBOINTER取得最高Coverage 71.7%，但Pair Recall仅0.8、ARI仅0.1，体现极端碎片化。
  - **Scaling不单调改善**：Qwen3.5 4B→27B在HD-EPIC上ARI 8.7→19.9，但在ROBOINTER上19.4→4.6。
- **监督后最优（Table 2）**：统一Qwen3.5-4B Oracle-history SFT在HD-EPIC上ARI提升至31.0（零-shot 8.7）、Pair Precision 31.1、Pair Recall 42.8；但**Create R仅28.6、Reuse R仍高达97.3**，体现创建-复用严重失衡。
- **训练增益的异质性（Table 3a）**：高频类Pair F1从19.0→34.9–40.0（HD-EPIC）、25.5→40.9–51.7（ROBOINTER），中低频类改善有限且不单调。
- **未见类别新颖性识别（Table 3b）**：11个训练中未见的HD-EPIC类别经监督后Coverage达57.6–66.1%，但10个被覆盖类别中仅1个获得新schema命名。
- **视频顺序影响（Figure 2）**：即使创新性密集顺序使46个参考类在前20视频全部出现，oracle SFT模型最终库仅含21–22个schema（参考46的一半），表明**"何时识别已有技能不足"才是核心瓶颈**。

## 相关工作脉络
1. **Video understanding / streaming models**（Damen et al., 2022; Perrett et al., 2025; Niu et al., 2025）：关注预定义类别的动作识别/时序定位或长视频QA，未涉及从流中增量构建持久技能库并据此做创建/复用决策。
2. **Unsupervised action segmentation**（Sener & Yao, 2018; Kukleva et al., 2019）：基于完整序列离线聚类，缺少流式设定下的命名不变分组与跨视频持久性要求。
3. **Skill discovery in RL / imitation**（Bacon et al., 2017; Pertsch et al., 2021; Wan et al., 2024; Kim et al., 2025）：多依赖环境交互、执行反馈或固定原型库，不研究纯被动视觉流的在线库构建。
4. **LLM agents with online skill libraries**（Wang et al., 2024; Luo et al., 2026）：通过执行与环境验证扩展技能，属于主动探索设定；本文聚焦被动视频流的语义变换抽象。
5. **Generalized / continual category discovery**（Vaze et al., 2022; Zhang et al., 2022; Wu et al., 2023; Ma et al., 2024）：研究开放世界类别发现，但未结合具身操作的时序定位、参数绑定与跨视频复用决策。
6. **In-context learning**（Brown et al., 2020; Pi et al., 2025）：将"创建schema"类比为context中引入新概念、"复用schema"类比为利用已有context概念，本文为此提供了具身操作的量化实证。

## 局限性与未来方向
- **领域局限**：仅覆盖机器人桌面操作与第一人称厨房活动，户外等更广泛场景未探索。
- **技能语义局限性**：当前schema仅描述观察到的物理变换，其与下游规划/可执行控制的可用性尚未验证。
- **新颖性识别瓶颈**：训练后模型库规模停滞于参考规模一半以下、未见变换几乎无法获得新命名，表明"感知新颖性"与"创建决策"之间的解耦仍需突破。
- **未来方向**：开发同时具备"巩固熟悉变换"与"适时扩展库"能力的模型；探索将本基准扩展到更多具身场景及下游控制任务。

## 研究启发与可借鉴点
1. **CLARE反事实状态编辑可直接迁移**：在需要增量知识管理（如agent memory、在线分类器更新）的任务中，构造"删除必要项→改变行为；添加无关项→保持行为"的反事实样本可增强模型对状态变化的敏感度。
2. **视频顺序设计作为诊断工具**：通过"创新性密集"与"复用性密集"两种顺序对照，可快速诊断模型是偏向过度复用还是过度创建，这一实验设计适用于任何需要评估"概念稳定性vs可塑性"的流式学习系统。
3. **Coverage-Grouping-Decision联合解读范式**：本文强调不能孤立看Coverage或ARI（如LLaVA-8B Coverage 71.7%但ARI≈0），这一"三角验证"思路可推广至其他"感知+结构化输出"任务的评测设计。
4. **On-policy correction的训练分布偏移**：用学生rollout状态替代oracle历史进行SFT的思想，与on-policy蒸馏一脉相承，可迁移到任何需要缓解"训练-推理状态分布不一致"的在线学习场景。
5. **训练频率异质性揭示数据分配信号**：高频技能改善显著而低频/未见技能几乎无增益，提示未来工作应在数据合成或课程设计中优先考虑稀有变换的覆盖，以突破"熟悉的愈熟悉、新颖的永未见"的马太效应。

## 关键术语表
**SESD (Streaming Embodied Skill Discovery)**：流式具身技能发现，指模型按序观看视频并持续维护持久技能库、据此对每个操作事件做时序定位与创建/复用决策的任务。
**Unified Paradigm**：统一范式，单一VLM在同一前向过程中同时接地事件和管理技能库。
**Factorized Paradigm**：因子化范式，先无库访问地生成文本事件描述，再由同一模型基于纯文本元策略更新技能库的两阶段处理。
**CLARE (Counterfactual Library-state Rebalancing)**：反事实技能状态重平衡，通过在训练中编辑库状态（增删schema/未完成任务）构造反事实样本以提升模型对相关变化的敏感度和对无关变化的鲁棒性。
**Pair Precision / Recall**：基于配对的一致性度量，衡量参考类对与预测类对的交集比例，分别反映过度合并（低Precision）与碎片化（低Recall）倾向。
**ARI (Adjusted Rand Index)**：调整兰德指数，修正了固定簇大小时随机期望的聚类一致性度量，值可为负。
**Creation-Reuse Decision**：创建-复用决策，判断当前变换首次出现时应创建新schema还是复用已有schema的二元判断。
**Coverage**：时序覆盖度，tIoU≥0.3下匹配的参考段数占总参考段数的比例。

## 可复现要素
- **数据集**：HD-EPIC 与 ROBOINTER（均开源）；Paper Page: https://andyzworks.github.io/video2skill
- **代码/权重**：项目页面已提供（论文未明确给出GitHub链接，代码开源状态以项目页面为准）
- **关键超参**：论文未详细报告训练超参（学习率、epoch数、batch size等均未在正文中给出），需查阅附录或代码仓库
